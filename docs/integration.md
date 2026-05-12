# Integration guide

This document describes how a consumer application plugs `interviewer`
into its stack. The engine is a library: it owns the conversation logic,
not deployment, persistence, or UI. See [SCOPE.md](../SCOPE.md) for the
design contract; this guide is operational.

## Deployment shape

Two processes share one `Engine` factory:

1. **Web app** — handles operator-facing requests:
   `create_conversation`, `provision_session`, `cancel_session`,
   `get_session_status`, `get_transcript`, `get_extract`. Short-lived per
   request.
2. **LiveKit agent worker** — long-lived process running
   `livekit-agents` `AgentServer`. Dispatches one job per active session;
   each job invokes `Engine.entrypoint(ctx)`.

Both construct an `Engine` from the same store, event sink, LLM client,
and `LiveKitConfig`. The web app reads and writes through the store; the
worker reads-writes during the live call.

```
operator → web app → Engine.provision_session → SessionCredentials
                                                       │
                              link delivered ──────────┘
                                                       │
respondent  → join page → LiveKit room  ───────────────┘
                              │
                              ↓
                       AgentServer dispatch
                              │
                              ↓
                  Engine.entrypoint(ctx) (worker process)
                              │
                  ┌───────────┴──────────────┐
                  ↓                          ↓
              ConversationStore          EventSink
              (turns, runtime,           (lifecycle events
               extract)                   for the operator UI)
```

## Implementing `ConversationStore`

The protocol is all `async def` (D3). The engine writes a
`SessionRuntimeState` and any `Turn` before the next agent utterance, and
calls `update_session_state` as soon as a transition is decided.

Use [`src/interviewer/stores/sqlite.py`](../src/interviewer/stores/sqlite.py)
as a reference. Key points:

- One row per session in `sessions`, with a JSON column holding the
  `conversation_snapshot` (D10).
- One row per turn in `turns`, indexed on `(session_id, turn_index)`.
- One row per session in `runtime_states` and `extracts` (no FK in the
  reference impl — the in-memory store accepts these without a Session
  row; see DECISIONS.md "Step 14 — no FK").
- All Pydantic blobs serialise via `model_dump_json` / `model_validate_json`.

For Postgres or another store, follow the same shape. The shared
round-trip suite in `tests/stores/_round_trip.py` is the contract — make
your implementation pass it.

## Implementing `EventSink`

The protocol has one method: `async def emit(event: SessionEvent) -> None`.

Reference impls:
- [`sinks/memory.py`](../src/interviewer/sinks/memory.py) — for tests.
- [`sinks/logging.py`](../src/interviewer/sinks/logging.py) — writes to a
  `logging.Logger`.
- [`sinks/webhook.py`](../src/interviewer/sinks/webhook.py) — POSTs to a
  URL via `httpx.AsyncClient`, retries `httpx.HTTPError` up to N times.

Event payload shapes are described in the
[`types/events.py`](../src/interviewer/types/events.py) docstring; notably,
`turn_recorded.payload` carries per-turn LLM usage (D11) and
`goal_status_changed` is emitted only at completion (D5).

## The `provision_session` → AgentServer dispatch → `entrypoint` flow

1. Web app calls `engine.provision_session(conversation_id)`. The engine
   snapshots the Conversation onto a new Session (D10), mints a LiveKit
   `AccessToken` for room `iv:{session_id}` with identity
   `respondent:{session_id}` (24-h TTL, D9), and emits
   `session_provisioned`. Returns `(Session, SessionCredentials)`.
2. The web app composes a user-facing link embedding `room_url` and
   `token`, on its own domain.
3. The respondent opens the link; the join page (consumer-owned, see
   [`examples/join_page.html`](../examples/join_page.html) for a minimal
   reference) connects to the LiveKit room.
4. LiveKit's AgentServer detects the new room and dispatches a job to
   the worker. The worker's entrypoint calls
   `await engine.entrypoint(ctx)`, which builds an `AgentSession`
   (Deepgram STT + Cartesia TTS + Silero VAD + `InterviewerLLM` wrapping
   the configured `LLMClient`) and runs the loop to completion or
   disconnect.
5. On clean completion the engine writes `Extract`, emits
   `goal_status_changed` for any goal whose loop-time hint differs from
   the canonical pass, then emits `completed`. On disconnect mid-call,
   state goes ABANDONED, runtime state is preserved, and a future
   re-provision of the same session resumes from `pending_follow_up` (the
   first agent utterance after resume is a fixed acknowledgement
   template).

## Example: FastAPI + AgentServer worker

```python
# shared_engine.py
import os
from interviewer import Engine, LiveKitConfig
from interviewer.llm.anthropic import AnthropicLLMClient
from interviewer.sinks.webhook import WebhookEventSink
from interviewer.stores.sqlite import SQLiteConversationStore

async def build_engine() -> Engine:
    store = SQLiteConversationStore(os.environ["DATABASE_PATH"])
    await store.connect()
    return Engine(
        store=store,
        events=WebhookEventSink(url=os.environ["WEBHOOK_URL"]),
        llm=AnthropicLLMClient(api_key=os.environ["ANTHROPIC_API_KEY"]),
        livekit=LiveKitConfig(
            url=os.environ["LIVEKIT_URL"],
            api_key=os.environ["LIVEKIT_API_KEY"],
            api_secret=os.environ["LIVEKIT_API_SECRET"],
            agent_name="interviewer",
        ),
    )
```

```python
# web_app.py — FastAPI
from fastapi import FastAPI
from interviewer import Background, Goal, Persona
from shared_engine import build_engine

app = FastAPI()

@app.post("/conversations")
async def create_conversation(body: dict) -> dict:
    engine = await build_engine()
    conv = await engine.create_conversation(
        persona=Persona(**body["persona"]),
        purpose=body["purpose"],
        background=Background(**body["background"]),
        goals=[Goal(**g) for g in body["goals"]],
        opening=body.get("opening"),
        closing=body.get("closing"),
    )
    return {"conversation_id": conv.id}

@app.post("/conversations/{conv_id}/sessions")
async def new_session(conv_id: str) -> dict:
    engine = await build_engine()
    session, creds = await engine.provision_session(conv_id)
    link = f"https://interviews.example.com/join/{session.id}?token={creds.token}"
    return {"session_id": session.id, "link": link,
            "expires_at": creds.expires_at.isoformat()}

@app.get("/sessions/{session_id}/status")
async def status(session_id: str) -> dict:
    engine = await build_engine()
    return (await engine.get_session_status(session_id)).model_dump()

@app.post("/sessions/{session_id}/cancel")
async def cancel(session_id: str) -> None:
    engine = await build_engine()
    await engine.cancel_session(session_id, reason="operator cancelled")
```

```python
# worker.py — the LiveKit AgentServer process
from livekit import agents
from shared_engine import build_engine

async def entrypoint(ctx: agents.JobContext) -> None:
    engine = await build_engine()
    await engine.entrypoint(ctx)

if __name__ == "__main__":
    agents.cli.run_app(agents.WorkerOptions(
        entrypoint_fnc=entrypoint, agent_name="interviewer",
    ))
```

Run them as separate deployments. The web app handles HTTP; the worker is
a long-running process invoked by LiveKit's dispatch system.

## Operational gaps the consumer must handle

The engine deliberately does not solve these:

- **Orphaned sessions.** A session can be provisioned but never joined
  (the link is shared but the respondent never clicks it). The engine
  has no TTL sweeper. Periodically scan the store for sessions in `READY`
  whose `created_at` exceeds your policy and mark them ABANDONED.
- **Token refresh.** `SessionCredentials.expires_at` is 24h by default
  (D9). The engine does not auto-refresh. If a link is shared and the
  respondent waits longer than the TTL, the consumer must call
  `reprovision_session` and reshare. Cap the token TTL via your own
  expiry layer if needed.
- **AgentServer scaling.** The LiveKit worker process is yours to deploy
  and scale. One worker process can handle multiple concurrent sessions;
  size pool by expected concurrent calls × audio resource budget.
- **Persistent storage of `conversation_snapshot`.** Make sure your
  store's `sessions` table holds the full snapshot (JSON column is fine).
  The engine reads it back from `Session.conversation_snapshot` on resume
  — losing it loses the in-flight conversation's contract.
- **Auth, RBAC, multi-tenancy.** Not modelled here. Wrap engine calls in
  your own authorisation layer.
- **Webhook delivery durability.** `WebhookEventSink` retries
  `httpx.HTTPError` 3× with exponential backoff and otherwise drops the
  event. For at-least-once delivery, supply your own `EventSink` that
  writes to a durable queue.
- **Audio recording / file storage.** Optional. If you want recordings,
  configure LiveKit server-side recording and wire storage yourself; the
  engine treats the transcript as the system of record.

## Testing your integration

Use `Engine.simulate_session(conversation_id, simulator)` to rehearse a
Conversation against a synthetic respondent. The simulator path goes
through the same `LLMClient` calls as the voice path, so you can
exercise the real Anthropic prompts end-to-end without a LiveKit room.

```python
from interviewer.testing.simulators import TerseEvasiveSimulator

extract = await engine.simulate_session(conv.id, TerseEvasiveSimulator())
```

For deterministic tests, swap `AnthropicLLMClient` for
`interviewer.testing.fake_llm.FakeLLMClient` driven by a scripted queue
of `EvalResult` and utterance strings. See `examples/simulated.py` for
a runnable template.
