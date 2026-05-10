# Interviewer — Project Scope

## What this is

A Python package that conducts adaptive voice interviews on behalf of a human
operator. The operator defines a **Conversation** — a persona for the
interviewer, a purpose, and a list of **Goals** (things to find out, each with
a "what good looks like" standard). The package runs the conversation as a
voice agent, adapts mid-call (clarifying when answers are thin, drilling on
interesting tangents, skipping goals that earlier answers already covered),
and produces a structured **Extract** mapping every claim back to who said it
and when.

This package is the **engine**. It owns the conversation logic and the agent
loop. It does not own the database, the web server, the link domain, the
dashboard, or any UI. Those are the consumer's responsibility.

## Where it sits

The lifecycle is two-phase. The operator's app calls `provision_session` to
allocate a LiveKit room and get join credentials — fast, no audio resources
yet. The credentials are embedded in a link and shared. When the respondent
opens the link and joins the room, LiveKit's `AgentServer` (which the
consumer runs as a separate process or worker) automatically dispatches a
job invoking `Engine.entrypoint(ctx)`. The entrypoint constructs a
`livekit-agents` `AgentSession` (Deepgram STT, Anthropic LLM with our
prompt orchestration, Cartesia TTS, Silero VAD) and runs the interview to
completion or disconnect. STT/LLM/TTS resources are only held while the
respondent is actually on the call.

From the user's perspective the agent is a real-time speech-to-speech
participant: they speak, they hear, full duplex with interruption support.
The STT→LLM→TTS pipeline is internal.

```
operator                 respondent              consumer app             engine
  │                          │                       │                       │
  ├─ create_conversation() ──────────────────────────────────────────────────│
  ├─ provision_session(c)  ──────────────────────────────────────────────────│ → SessionCredentials
  │   ← {room_url, token} ──────────────────────────────────────────────────
  │
  ├─ shares link ─────────→ │
  │                         │
  │                         ├─ opens link → joins LiveKit room
  │                         │                       │
  │                         │   LiveKit AgentServer dispatches a job ───────→ entrypoint(ctx)
  │                         │                       │                       │
  │                         │                       │                       │ AgentSession:
  │                         │ ←── voice call (LiveKit room) ────────────────→│   Deepgram STT
  │                         │                       │                       │   InterviewerLLM
  │                         │                       │                       │   Cartesia TTS
  │                         │                       │                       │   Silero VAD
  │                         │                       │                       │   per turn:
  │                         │                       │                       │     evaluate → compose
  │                         │                       │                       │       → stream TTS
  │                         │                       │                       │     persist + emit
  │                         │                       │                       │
  ├─ get_session_status() ─────────→ (reads from store)                     │
  ├─ subscribe(events) ───────────── (consumes from EventSink)              │
  │                         │                       │                       │
  │                         ├─ disconnects ─────────────────────────────────│ COMPLETED or ABANDONED
  │                                                                         │
  ├─ get_transcript() / get_extract() ─────────────────────────────────────→│
```

The package is a library, not a framework. The consumer constructs an
`Engine`, calls its methods, and implements four `Protocol`s for the things
the consumer owns (storage, eventing, voice transport, LLM client).

---

## What the package does

- Defines the **Conversation / Goal / Persona** configuration types.
- Provisions a voice room per **Session** and returns join credentials.
- Runs the agent loop end-to-end on demand: STT → next-question generation →
  TTS → turn capture → answer evaluation → adaptive next-goal selection.
- Adapts mid-conversation:
  - Re-asks (within a per-goal retry budget) when an answer is partial.
  - Drills (within a tangent budget) when the respondent surfaces something
    interesting.
  - Skips goals rendered redundant by earlier answers.
  - Marks goals as `meets`, `partial`, `skipped_redundant`, or `gave_up`.
- Handles unhappy paths: silence, refusal, "I don't know," disconnect,
  LLM errors, turn-cap exhaustion (see Unhappy paths section).
- Persists every turn and runtime state through a consumer-supplied
  **`ConversationStore`** so the consumer can read progress and the final
  transcript from their own database, and so the engine can crash-recover.
- Emits **lifecycle events** through a consumer-supplied **`EventSink`** so
  the operator's UI can show "in progress" vs "finished" without listening to
  audio.
- Produces a structured **`Extract`** at the end: per-goal status with
  evidence turn indices and verbatim quotes, plus any unprompted findings.
- Provides a **`RespondentSimulator`** test harness so Conversations can be
  rehearsed against synthetic respondents (text-only in v1) before being sent
  to real humans.
- Ships sensible defaults (LiveKit voice transport, Anthropic LLM client)
  while letting the consumer swap them.

## What the package does NOT do

These are the consumer's responsibility and the package intentionally avoids
them:

- **HTTP / web server.** The consumer runs whatever stack they like and calls
  into the engine.
- **Link generation and link domain.** The engine returns `SessionCredentials`
  (a room URL and a join token). The consumer composes the user-facing link
  on their own domain, with their own expiry policy.
- **Database.** The engine reads and writes through the `ConversationStore`
  protocol. The consumer picks the store (SQLite, Postgres, files, an ORM)
  and owns schema, migrations, and indexing.
- **Operator dashboard UI.** The engine exposes status and events; the
  consumer renders them.
- **Respondent join page.** The consumer builds the page that takes the link
  token and joins the LiveKit room (LiveKit ships a JS/React client).
- **LiveKit AgentServer process.** The consumer runs the
  `livekit-agents` `AgentServer` worker that dispatches `Engine.entrypoint`
  per session. They own deployment, scaling, and health checks; the engine
  just supplies the entrypoint function.
- **Auth / RBAC / multi-tenant scoping.** Not modeled in the engine.
- **Email / Slack / notification side effects.** The consumer subscribes to
  events and decides what to do.
- **Audio recording / file storage.** Optional. If recordings are needed, the
  voice transport (LiveKit) can record server-side, and the consumer wires
  storage. The engine treats the transcript as the system of record.
- **Live audio streaming to the operator.** Out of scope for v1. Operators
  see state and transcript, not live audio.
- **Multi-respondent aggregation, cross-interview triangulation, mid-flight
  brief amendments.** Phase 3 of the product roadmap. The engine is built so
  these are *possible* later but it does not implement them.

---

## Public API

### Configuration types

```python
class Persona:
    system_prompt: str           # "You are a sim engineer interviewing..."
    style: Literal["warm", "neutral", "terse"]
    voice_id: str                # TTS voice identifier

class Background:
    """Structured context about who's being interviewed and why.
    Replaces a single free-text field to keep the system prompt bounded."""
    interviewee_role: str            # one line, e.g. "20-year warehouse ops lead"
    interviewee_expertise: str       # one line, e.g. "process flow at warehouse X"
    relevant_context: str = ""       # free-form, hard cap 1000 chars (engine truncates)

class Goal:
    id: str
    intent: str                  # what we want to know (not the literal question)
    standard: str                # rubric for "answered well enough"
    max_retries: int = 2         # follow-up budget when answer is partial
    depends_on: list[str] = []   # goal ids that must be addressed first
    redundant_when: str = ""     # rubric for "skip if earlier answers covered
                                 # this"; injected into the evaluate_turn system
                                 # prompt as per-goal guidance the model reads
                                 # when deciding whether to flag a goal as
                                 # redundant. Empty string disables the check.

class Conversation:
    id: str
    persona: Persona
    purpose: str                 # why we're talking to this person
    background: Background       # structured, bounded
    goals: list[Goal]            # operator's default ordering
    opening: str | None = None   # optional scripted intro
    closing: str | None = None   # optional scripted close
    max_tangent_followups: int = 2
    max_total_turns: int = 80
```

### Runtime types

```python
class Session:
    id: str
    conversation_id: str
    conversation_snapshot: Conversation   # frozen at provision_session time
    state: SessionState
    created_at: datetime
    started_at: datetime | None
    completed_at: datetime | None

class SessionCredentials:
    """What the consumer embeds in a shareable link."""
    room_url: str
    token: str
    expires_at: datetime         # default 24h; re-provision to refresh

class SessionRuntimeState:
    """In-flight state for crash recovery. Flushed to store before every
    agent utterance. On worker restart, load this and resume the loop."""
    session_id: str
    active_goal_id: str | None       # goal currently being asked
    retries_used_on_active: int      # follow-ups consumed on active goal
    tangent_followups_used: int      # against Conversation.max_tangent_followups
    total_turns: int                 # against Conversation.max_total_turns
    pending_follow_up: str | None    # next utterance if loop was interrupted
    last_event_index: int            # last turn flushed to event sink
    updated_at: datetime

class Turn:
    index: int
    speaker: Literal["agent", "respondent"]
    text: str
    timestamp: datetime
    addressed_goal_ids: list[str]    # live hint set by evaluate_answer;
                                     # NOT canonical (see Canonical truth below)

class GoalStatus:
    goal_id: str
    status: Literal["pending", "meets", "partial", "skipped_redundant", "gave_up"]
    evidence_turn_indices: list[int] # CANONICAL — derived in final Extract pass
    retries_used: int
    rationale: str                   # why this status was assigned

class Finding:
    """Something the respondent volunteered that wasn't asked.
    Surfaced by derive_extract when the model spots a claim outside any goal."""
    text: str                # the claim, in the respondent's own words (short)
    evidence_turn_index: int # which turn it came from
    category: str | None = None  # optional operator-defined tag

class Extract:
    session_id: str
    conversation_id: str
    goal_statuses: list[GoalStatus]
    unprompted_findings: list[Finding]
    full_transcript: list[Turn]
    completed_at: datetime

class SessionStatus:
    """Cheap dashboard read; aggregated from store, not the live agent.
    May be one turn stale relative to the running loop."""
    session_id: str
    state: SessionState
    active_goal_id: str | None
    total_turns: int
    goals_resolved: int      # count of goals not in {pending}
    goals_total: int
    started_at: datetime | None
    last_turn_at: datetime | None

# LLM I/O types (used by LLMClient protocol)

class TurnContext:
    """Everything the LLM needs for one compose-or-evaluate call."""
    conversation: Conversation        # the snapshot, not the live config
    transcript: list[Turn]            # full history so far (may be empty)
    active_goal: Goal | None          # selected for this turn (None on opening)
    goal_statuses: list[GoalStatus]   # current state of each goal
    retries_used_on_active: int
    tangent_followups_used: int
    total_turns: int
    last_phrasing_failure: str | None # set on regen attempt; None otherwise

class EvalResult:
    active_goal_status: Literal["pending", "meets", "partial", "gave_up"]
    redundant_goal_ids: list[str]     # other goals now skippable
    interesting_tangent: str | None   # short phrase if drilling is warranted
    next_action: Literal["advance", "retry", "drill", "close"]
    rationale: str                    # short explanation, persisted on GoalStatus
```

#### Canonical truth between `Turn.addressed_goal_ids` and `GoalStatus.evidence_turn_indices`

These two fields cross-reference each other and can diverge.

- `Turn.addressed_goal_ids` is a **live hint** populated during the loop from
  what `evaluate_answer` returned. It exists so `select_next_goal` and the
  dashboard's "goal progress" view have something to read mid-session. It is
  not authoritative.
- `GoalStatus.evidence_turn_indices`, set by `derive_extract` at session
  end, is **canonical**. The final extraction pass re-reads the whole
  transcript holistically and may revise which turns count as evidence for
  which goals.

In short: trust `GoalStatus` after completion; trust `Turn.addressed_goal_ids`
during the loop.

### State model

```python
class SessionState(StrEnum):
    CREATED      = "created"        # session record exists, room not yet provisioned
    READY        = "ready"          # room provisioned, token issued, awaiting respondent
    IN_PROGRESS  = "in_progress"    # respondent joined, agent running
    COMPLETED    = "completed"      # finished normally
    ABANDONED    = "abandoned"      # respondent dropped, did not return; or cancelled
    FAILED       = "failed"         # internal error; surface to operator
```

### Lifecycle events

```python
class SessionEvent:
    session_id: str
    conversation_id: str
    timestamp: datetime
    type: Literal[
        "session_provisioned",
        "respondent_joined",
        "turn_recorded",
        "goal_status_changed",
        "completed",
        "abandoned",
        "failed",
    ]
    payload: dict
```

### Consumer-implemented protocols

All protocols are async. The engine itself runs on `asyncio` and there is no
sync surface; consumer code that calls into the engine awaits.

```python
class ConversationStore(Protocol):
    """The consumer owns persistence. SQLite, Postgres, files — engine doesn't care."""
    async def save_conversation(self, c: Conversation) -> None: ...
    async def load_conversation(self, conversation_id: str) -> Conversation: ...
    async def save_session(self, s: Session) -> None: ...
    async def load_session(self, session_id: str) -> Session: ...
    async def update_session_state(self, session_id: str, state: SessionState) -> None: ...
    async def append_turn(self, session_id: str, turn: Turn) -> None: ...
    async def list_turns(self, session_id: str) -> list[Turn]: ...
    async def save_runtime_state(self, rs: SessionRuntimeState) -> None: ...
    async def load_runtime_state(self, session_id: str) -> SessionRuntimeState | None: ...
    async def save_extract(self, extract: Extract) -> None: ...
    async def load_extract(self, session_id: str) -> Extract | None: ...

class EventSink(Protocol):
    """The consumer routes events: webhook, websocket, queue, log."""
    async def emit(self, event: SessionEvent) -> None: ...

class LLMClient(Protocol):
    """The brain. Default impl is Anthropic Claude.

    Each agent turn is two sequential calls:
      1. evaluate_turn — small, structured (Haiku 4.5 in the default impl).
         Reads the prior respondent utterance plus state, decides
         next_action and surfaces redundancy / tangent hints.
      2. compose_utterance — streaming text (Sonnet 4.6 in the default impl).
         Drives the next agent utterance from the eval result. Yields tokens
         as they arrive so the voice transport can begin TTS playback before
         the full utterance is generated.
    derive_extract is a single non-streaming call at session end."""
    async def evaluate_turn(self, ctx: TurnContext) -> EvalResult: ...
    def compose_utterance(
        self, ctx: TurnContext, eval_result: EvalResult,
    ) -> AsyncIterator[str]: ...
    async def derive_extract(
        self, transcript: list[Turn], conv: Conversation,
    ) -> Extract: ...

class RespondentSimulator(Protocol):
    """Test harness: a synthetic respondent that drives the agent loop without audio.
    v1 ships text-mode simulators; voice-mode rehearsal is deferred."""
    async def respond(self, agent_utterance: str, history: list[Turn]) -> str: ...
    def persona_name(self) -> str    # e.g. "terse_evasive", "rambly_knowledgeable"
```

There is no `VoiceTransport` protocol in v1. Voice runs through the
`livekit-agents` `AgentSession` primitive, wired up by `Engine.entrypoint`
(see Entry points below) which the consumer registers with their LiveKit
`AgentServer`. Swapping LiveKit for a different voice runtime in v2 will
require a new protocol; v1 commits to LiveKit on purpose to keep the surface
small.

### Entry points

```python
class Engine:
    def __init__(
        self,
        store: ConversationStore,
        events: EventSink,
        llm: LLMClient,
        livekit: LiveKitConfig | None = None,   # required for voice; None for simulator-only
    ): ...

    async def create_conversation(
        self, *, persona: Persona, purpose: str, background: Background,
        goals: list[Goal], opening: str | None = None, closing: str | None = None,
    ) -> Conversation:
        """Persists the Conversation. Does not generate a link.
        Emits no event (creation is a config action, not a session action)."""

    async def provision_session(
        self, conversation_id: str,
    ) -> tuple[Session, SessionCredentials]:
        """Creates a Session, snapshots the Conversation onto it, and
        provisions a LiveKit room (if `livekit` was supplied at __init__).
        Returns credentials the consumer embeds in a shareable link.
        State: → READY. Emits `session_provisioned`."""

    async def reprovision_session(self, session_id: str) -> SessionCredentials:
        """Issues fresh credentials for the same Session (e.g. after token
        expiry). The room is re-created; previous tokens are invalidated."""

    async def entrypoint(self, ctx: "agents.JobContext") -> None:
        """LiveKit AgentSession entrypoint — the consumer registers this with
        their `agents.AgentServer`. Reads the session_id from the room
        metadata, loads the snapshotted Conversation, constructs an
        AgentSession with Deepgram STT + Cartesia TTS + an InterviewerLLM
        (a livekit-agents `LLM` subclass that delegates to `self.llm`), and
        runs until the participant disconnects.
        State: READY → IN_PROGRESS → COMPLETED/FAILED.
        On crash, runtime state in the store lets a re-dispatched job resume."""

    async def cancel_session(self, session_id: str, reason: str = "") -> None:
        """Operator-initiated termination. Writes ABANDONED to the store,
        emits `abandoned`, then deletes the LiveKit room (which disconnects
        the in-flight AgentSession). For simulator-mode sessions, the loop
        observes the state change at its next iteration boundary."""

    async def get_session_status(self, session_id: str) -> SessionStatus:
        """Cheap read for the operator's dashboard: current state, turn
        count, active goal id, started_at, last_turn_at. Reads from the
        store; may be one turn stale relative to the live agent."""

    async def get_transcript(self, session_id: str) -> list[Turn]:
        """Full transcript. Available progressively while IN_PROGRESS,
        final when COMPLETED."""

    async def get_extract(self, session_id: str) -> Extract | None:
        """Structured output. Available when state == COMPLETED."""

    async def simulate_session(
        self, conversation_id: str, simulator: RespondentSimulator,
    ) -> Extract:
        """Run the agent loop against a synthetic respondent — no voice room,
        no audio, no AgentSession. Useful for rehearsing a Conversation in
        tests. Goes through the same LLMClient calls (so it exercises the
        Anthropic prompt path end-to-end if you pass AnthropicLLMClient)."""
```

`LiveKitConfig` is a tiny dataclass (`url`, `api_key`, `api_secret`,
`agent_name`) used by `provision_session` to mint room creds and by
`entrypoint` to know which AgentServer subject it's serving. Consumers in
simulator-only deployments leave `livekit=None`.

### Store consistency

The engine writes `SessionRuntimeState` and any `Turn` to the store *before*
the agent speaks the next utterance, and `update_session_state` is called as
soon as a state transition is decided. `get_session_status` reads from the
store; it is therefore at most one turn stale relative to the live agent
process. For most dashboard use cases this is invisible; consumers needing
finer resolution should subscribe to the `EventSink` instead.

---

## Example consumer integration

This is what a consumer's app looks like. Nothing here is part of the package
— it shows where the engine fits.

```python
# consumer's app: shared engine factory
def build_engine() -> Engine:
    return Engine(
        store=PostgresConversationStore(dsn=os.environ["DATABASE_URL"]),
        events=WebhookEventSink(url=os.environ["WEBHOOK_URL"]),
        llm=AnthropicLLMClient(api_key=os.environ["ANTHROPIC_API_KEY"]),
        livekit=LiveKitConfig(
            url=os.environ["LIVEKIT_URL"],
            api_key=os.environ["LIVEKIT_API_KEY"],
            api_secret=os.environ["LIVEKIT_API_SECRET"],
            agent_name="interviewer",
        ),
    )

# consumer's web app: operator creates a conversation
@app.post("/conversations")
async def create_conversation(body: CreateConversationRequest):
    engine = build_engine()
    conv = await engine.create_conversation(
        persona=Persona(
            system_prompt="You are a simulation engineer conducting a discovery interview...",
            style="neutral",
            voice_id="cartesia-sonic-female-1",
        ),
        purpose="Understand the end-to-end process flow at a single warehouse.",
        background=Background(
            interviewee_role="20-year warehouse operations lead",
            interviewee_expertise="end-to-end process flow at warehouse X",
            relevant_context="Multi-shift facility; recent ERP migration in Q1.",
        ),
        goals=[
            Goal(id="flow",  intent="Map each major process step with the role that performs it.",
                 standard="At least 4 named steps, each with a role and a typical duration."),
            Goal(id="excep", intent="Find common exception paths.",
                 standard="At least 2 exception types named with a rough frequency."),
        ],
    )
    return {"conversation_id": conv.id}

# consumer's web app: operator requests a shareable link
@app.post("/conversations/{conv_id}/sessions")
async def new_session(conv_id: str):
    engine = build_engine()
    session, creds = await engine.provision_session(conv_id)
    link = f"https://interviews.example.com/join/{session.id}?token={creds.token}"
    return {"session_id": session.id, "link": link, "expires_at": creds.expires_at}

# consumer's worker: a separate process running the LiveKit AgentServer
# This is what actually runs the interview when the respondent joins.
from livekit import agents

async def my_entrypoint(ctx: agents.JobContext):
    engine = build_engine()
    await engine.entrypoint(ctx)

if __name__ == "__main__":
    agents.cli.run_app(
        agents.WorkerOptions(entrypoint_fnc=my_entrypoint, agent_name="interviewer"),
    )

# consumer's web app: dashboard polls status
@app.get("/sessions/{session_id}/status")
async def status(session_id: str):
    engine = build_engine()
    return await engine.get_session_status(session_id)

# consumer's web app: event sink picks up `completed`, fetches extract
@app.post("/internal/webhooks/session-events")
async def on_event(event: SessionEvent):
    if event.type == "completed":
        engine = build_engine()
        extract = await engine.get_extract(event.session_id)
        await deliver_to_operator(extract)

# consumer's test suite: rehearse a Conversation before sending it
async def test_warehouse_brief():
    engine = build_engine_for_tests()  # uses InMemoryStore, FakeLLMClient
    conv = await engine.create_conversation(...)
    extract = await engine.simulate_session(conv.id, simulator=TerseEvasiveSimulator())
    assert any(gs.status == "gave_up" for gs in extract.goal_statuses)  # expected
```

### Worker model

The consumer runs a `livekit-agents` `AgentServer` as a separate worker
process (`python my_worker.py dev` for local, `start` for production).
LiveKit's dispatch system invokes `entrypoint_fnc` per room — one job per
active session — and tears it down on disconnect. The consumer does not
manage a webhook receiver, an asyncio task group, or session-level
worker-pool plumbing; AgentServer handles dispatch.

The web app (handling `provision_session`, status reads, etc.) and the
worker process can run from the same engine factory but are different
deployments. The web app is short-lived per request; the worker is
long-lived and dispatches per-session jobs.

---

## Conversation flow (what `entrypoint` and `simulate_session` do)

Both entry points run the same `run_loop` body — the only difference is
where the agent's voice goes (LiveKit `AgentSession` vs. simulator) and
where the respondent's reply comes from.

```
1. Load Session (with conversation_snapshot), and SessionRuntimeState if
   present.
2. If runtime state exists: resume — restore active_goal_id, retries_used,
   tangent_followups_used, total_turns. The next agent utterance is the
   resume acknowledgement template ("we got cut off — picking up...").
   Else: set state IN_PROGRESS; speak the opening (scripted if provided).
3. Main loop, until terminal condition:
   a. select_next_goal() — pure function over conversation_snapshot.goals
      and the current goal_statuses table:
        - filter out: meets, skipped_redundant, gave_up
        - require: depends_on satisfied
        - prefer: operator's default order
      Returns None when all goals are resolved.
      (Redundancy is decided inside evaluate_turn, not here.)
   b. evaluate_turn() — Haiku 4.5 call with forced tool-use, structured JSON:
        EvalResult {
          active_goal_status: pending | meets | partial | gave_up,
          redundant_goal_ids: list[str],     # other goals now skippable
          interesting_tangent: str | None,
          next_action: advance | retry | drill | close,
        }
      Skipped on the very first turn (nothing to evaluate yet).
   c. Apply EvalResult to the goal-status table:
        - active_goal_status updated
        - redundant_goal_ids → status `skipped_redundant`
        - tangent budget consumed if next_action == drill
        - retry budget consumed if next_action == retry
   d. flush updated SessionRuntimeState to store BEFORE composing.
   e. compose_utterance() — Sonnet 4.6 streaming text. Tokens stream to
      Cartesia TTS via AgentSession (voice mode) or are accumulated
      verbatim (simulator mode). The full utterance is also accumulated
      for transcript persistence and validation.
   f. validate_voice_phrasing(full_utterance) — word count + question count.
      On failure, regenerate compose_utterance once with the failure surfaced
      in the prompt; if still failing, speak verbatim.
   g. record turn (speaker=agent) and append to store.
   h. emit `turn_recorded` event with usage telemetry.
   i. Respondent reply arrives:
        - voice mode: AgentSession's STT yields a final transcript per turn
          via the InterviewerLLM's `chat()` boundary
        - simulator mode: simulator.respond(agent_utterance, history)
      Record turn (speaker=respondent), append to store, emit `turn_recorded`.
4. Terminal conditions:
   - select_next_goal() returns None (all goals resolved) → step 5
   - max_total_turns reached → finish current probe, then step 5
   - cancel_session called externally → state observed ABANDONED, speak
     short scripted closing, exit
   - LiveKit room disconnects (respondent left) → state ABANDONED, exit
5. Speak the closing (scripted if provided, else a brief default).
6. derive_extract(transcript, conversation_snapshot) → Extract;
   persist via store.save_extract.
7. Set state COMPLETED; emit `completed` event with the goal-status diff.
```

---

## Latency strategy

Each agent turn runs two sequential Anthropic calls (see DECISIONS.md D2):

1. **Evaluate** — Haiku 4.5 with forced tool-use, returning structured JSON
   for `next_action` plus redundancy / tangent hints. ~200–400 ms typical.
2. **Compose** — Sonnet 4.6 plain text streaming, no tools, driven by the
   eval result. The first text token streams to Cartesia TTS the moment it
   arrives. ~150–300 ms TTFT typical.

Combined with Deepgram endpoint detection (~300 ms) and Cartesia first-audio
(~150–300 ms), the budget is:

- **Typical:** 1.0–1.5 s from end-of-respondent-speech to start-of-agent-audio.
- **Worst case:** 2.5–3.5 s (cold cache, slow LLM, network jitter).

These are implementation targets, not guarantees.

Two things keep this from feeling broken on voice:

- **Streaming TTS.** Cartesia begins synthesizing audio as soon as the first
  Sonnet token streams in; the agent starts speaking before the full
  utterance is generated.
- **AgentSession turn-taking.** `livekit-agents` 1.x ships VAD-based turn
  detection and ML-based backchannel filtering by default — short
  acknowledgements ("mm-hmm", "uh-huh") from the respondent during the
  agent's compose latency don't trigger interruption, and the perceived
  latency budget for a "natural pause" is generous in practice.

Prompt caching on the Conversation config (system prompt with persona,
background, goals) is enabled by default in `AnthropicLLMClient` (5-minute
ephemeral cache breakpoint). It is load-bearing: it cuts per-turn LLM cost
and latency by avoiding re-sending the stable context.

We do NOT implement a separate filler-audio channel in v1. AgentSession's
backchannel handling and Cartesia's fast first-audio cover the perceptual
budget without us synthesizing extra audio.

---

## Voice phrasing constraints

Voice questions must read differently from written ones — short, single, no
enumeration. The engine enforces this with deterministic post-checks rather
than relying on prompt discipline alone:

- One question per utterance (no `?` followed by another `?`).
- ≤25 words (configurable internally; not exposed on Conversation in v1).
- No enumerative cues ("first", "second", "and also", "along with", commas
  separating multiple distinct questions).

The compose pass produces a streaming candidate; the engine accumulates
tokens, runs the deterministic check (word count + question count) on the
final string before persisting the turn. On failure the engine regenerates
once with the failure surfaced in the prompt. If the second attempt also
fails, the engine speaks the utterance verbatim — v1 does not attempt
programmatic splitting (the brittle clause-splitter has been removed; see
DECISIONS.md D7).

The check is deliberately narrow — word count and question-mark count, not
a keyword-based enumeration heuristic — because the false-positive rate of
keyword detectors is high enough to make regen worse than the original.
Sonnet's adherence to the prompt's "single short question" instruction is
the primary mechanism; the validator is a guard.

---

## Unhappy paths

The loop has explicit handling for the cases below. They are part of v1; the
engine does not punt them to the consumer.

- **Silence after the agent speaks.** Handled by `livekit-agents`
  `AgentSession`'s built-in VAD + user-away detection. The engine does not
  implement a custom silence-nudge in v1. If the respondent goes silent for
  longer than the framework's idle threshold, AgentSession surfaces a
  disconnect, which the engine treats the same as a respondent disconnect
  (see below). v2 may add a configurable custom-nudge utterance.
- **Refusal** ("I'd rather not answer that"). The agent accepts once. If the
  goal allows retries, the agent reformulates the goal *once* from a
  different angle, then if refused again marks the goal `gave_up` with
  rationale and moves on. The agent does not push.
- **"I don't know."** Treated similarly to refusal: a single deflection
  probe ("Is there someone on your team who would?") is allowed, then the
  goal is marked `gave_up`. The deflection probe consumes a retry. This is
  the line between *discovery* and *challenger*: the agent never implies the
  respondent should know.
- **LLM API failure** (network error, rate limit, malformed response). The
  engine retries `evaluate_turn` and `compose_utterance` up to 3 times each
  with exponential backoff. On persistent failure the engine speaks a short
  apology, sets state FAILED, emits `failed`, and persists what it has.
  Resume is possible if the consumer re-dispatches a job for the same
  session.
- **STT/TTS failure inside AgentSession.** AgentSession surfaces these as
  exceptions in the entrypoint. The engine catches once, attempts to keep
  the loop running, and on a second failure sets state FAILED.
- **Respondent disconnects mid-call.** AgentSession's room-disconnect
  handler exits the entrypoint. The engine writes ABANDONED, preserves
  runtime state, emits `abandoned`. If the consumer calls
  `reprovision_session` and the same respondent joins the new room within
  24 h, the next dispatched job's entrypoint detects the runtime state and
  resumes; the agent's first utterance is the resume acknowledgement
  ("we got cut off earlier — picking up where we left off…").
- **`max_total_turns` reached.** The loop finishes the current goal's probe
  naturally (does not interrupt mid-question), then proceeds to closing.
  Goals still in `pending` are written to the Extract with that status.
- **Worker / job crash.** On next dispatch for the same `session_id`, the
  runtime state in the store is loaded and the loop resumes from
  `pending_follow_up`. The agent's first utterance after resume is the
  resume acknowledgement template.
- **`cancel_session` mid-loop.** The engine writes ABANDONED to the store
  and deletes the LiveKit room; AgentSession exits cleanly on disconnect
  (no graceful goodbye in this path — the room is already gone). For
  simulator mode the loop checks state at each iteration boundary and
  speaks a short scripted closing on observing ABANDONED.

---

## Defaults the package ships

- **`Engine.entrypoint(ctx)`** — adapter that wires `livekit-agents`
  `AgentSession` with Deepgram STT, Cartesia TTS, Silero VAD, and a custom
  `InterviewerLLM` (a livekit-agents `LLM` subclass) that delegates to the
  configured `LLMClient`. The consumer plugs this entrypoint into their
  `AgentServer`.
- **`AnthropicLLMClient`** — Claude Sonnet 4.6 for compose and derive,
  Claude Haiku 4.5 for evaluate. Prompt caching enabled on the
  Conversation config (system prompt with persona, background, goals).
  See DECISIONS.md D2 for model choices.
- **`RespondentSimulator` reference impls** — `ScriptedSimulator`,
  `TerseEvasiveSimulator`, `RamblyKnowledgeableSimulator`,
  `ConfusedSimulator`. Text-mode only in v1.
- **`InMemoryConversationStore`, `InMemoryEventSink`** — for tests and
  simulator examples. Not for production.
- **`SQLiteConversationStore`** — async (aiosqlite). A reasonable default
  for single-process deployments.
- **`LoggingEventSink`, `WebhookEventSink`** — reference event sinks.

---

## What v1 deliberately does not handle

- Multi-respondent aggregation across Sessions sharing a Conversation. The
  data model supports many Sessions per Conversation; aggregation logic does
  not exist in v1. Note: a future Phase 3 "respondent-slanted brief" feature
  will require additive types (likely a `RespondentProfile` on `Session`)
  not present today. The schema is not pre-fit for it.
- Operator mid-call intervention beyond `cancel_session` (drop in, redirect,
  amend a goal). The engine does not expose mid-session mutation hooks in v1.
- Languages other than English. Default STT/TTS configs assume English. The
  protocols allow swapping, but v1 ships English-tuned defaults only.
- PSTN telephony. v1 assumes WebRTC (browser-based respondent). PSTN is a
  later transport plugin.
- A "challenger" posture. The agent is a discovery interviewer — it probes,
  clarifies, and drills for elaboration. It does not push back on claims it
  believes are wrong.
- Domain knowledge / RAG. The agent relies on the persona and Conversation
  background. There is no retrieval layer in v1.
- Voice-mode rehearsal. `RespondentSimulator` is text-only in v1; running a
  voice-against-voice rehearsal is deferred.

---

## Open questions — resolved before implementation

All resolutions are recorded in DECISIONS.md (Step 0). Summary:

1. **STT/TTS provider defaults.** Deepgram (STT) + Cartesia (TTS) — both
   are first-party `livekit-agents` plugins, actively maintained, and the
   common production defaults. (Resolved: D1.)
2. **Resumability semantics.** v1 resumes when `Session.state` is
   `IN_PROGRESS` or `ABANDONED` and last-updated within 24 h. The
   Conversation is snapshotted onto the Session at `provision_session`
   so mid-call edits never affect a live session. (Resolved: D9, D10.)
3. **Schema versioning.** Pre-v1 we accept breaking changes; the Pydantic
   models do not carry a `schema_version` field. v1 will introduce one if
   we decide to commit to backwards compatibility post-release.
4. **Token policy.** 24 h default; refresh is consumer-driven via
   `reprovision_session`. The engine does not auto-refresh. (Resolved: D9.)
5. **Filler utterance content.** Removed from v1 entirely — no separate
   filler-audio channel. AgentSession's backchannel handling and Cartesia
   first-audio cover the perceptual budget. (Resolved: D1, latency
   strategy.)
6. **Crash-recovery acknowledgement.** A module-level constant default
   template ("we got cut off — picking up where we left off"). Per-
   Conversation customization is deferred to v2.

---

## Glossary

- **Conversation** — the configuration template: persona, purpose,
  background, and goals. One Conversation can spawn many Sessions.
- **Session** — one run of a Conversation with one respondent. Owns a
  transcript, runtime state, and extract.
- **Goal** — a single thing the operator wants to find out, with a rubric
  for "answered well enough."
- **Persona** — the interviewer's identity and voice: who they're playing,
  domain framing, conversational style, TTS voice.
- **Background** — structured context (interviewee role, expertise,
  bounded free-form notes) injected into the system prompt.
- **Turn** — one utterance from either party, timestamped and tagged with
  which goals it touched (live hint, not canonical).
- **Extract** — the structured output for one Session: goal-by-goal status
  with evidence quotes, unprompted findings, and the full transcript.
- **Engine** — the package's top-level orchestrator; what the consumer
  instantiates and calls into.
- **Consumer** — the app or service that imports this package. Owns
  storage, HTTP, UI, link generation, and the LiveKit webhook receiver.
- **Operator** — the human user who authors Conversations and reads
  Extracts.
- **Respondent** — the human being interviewed by the agent.
- **`RespondentSimulator`** — a synthetic respondent used in tests; runs
  against `simulate_session` without a voice room.
