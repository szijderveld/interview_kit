# Interviewer — Build Plan

This plan takes the project from an empty directory to v1 against
[SCOPE.md](SCOPE.md) and [DECISIONS.md](DECISIONS.md). Each step is
self-contained: it states what to read, what to build, and how to verify
success. A fresh Claude session should be able to execute any single step
cold given this file plus SCOPE.md plus DECISIONS.md.

## How to use this plan

- Run `/next-step` to execute the first uncompleted step.
- The command reads SCOPE.md, DECISIONS.md, and the step's prompt; executes
  the deliverables; runs acceptance criteria; marks the step done; appends
  any non-obvious decisions to DECISIONS.md.
- Do not skip steps. If you must, edit the checkbox manually.
- Each step ends with a commit. Commits use the message `step N: <title>`.
- If a step fails its acceptance criteria, fix and re-run rather than
  moving on.

## Global conventions

These apply to every step. They are decisions, not suggestions. Override
only by amending this section and recording the change in DECISIONS.md.

- **Python:** 3.11+
- **Layout:** `src/` layout under `src/interviewer/`
- **Package manager:** `uv`
- **Async runtime:** `asyncio`. Every store / sink / engine method that
  touches I/O is `async def`.
- **Validation:** Pydantic v2; all data classes are frozen.
- **Tests:** `pytest` + `pytest-asyncio` (asyncio_mode = "auto").
- **Lint / format:** `ruff` (line length 100).
- **Type check:** `mypy --strict`.
- **Docstrings:** terse, only where the WHY is non-obvious.
- **Code style:** type hints required everywhere; no `Any` without
  justification; use `X | None`, not `Optional[X]`.
- **Commit policy:** one commit per completed plan step.

## Module layout (target)

This is the layout v1 ends with. Steps fill it in incrementally.

```
src/interviewer/
    __init__.py             # public re-exports
    py.typed
    types/
        __init__.py
        config.py           # Persona, Background, Goal, Conversation
        runtime.py          # Session, SessionCredentials, SessionRuntimeState,
                            # Turn, GoalStatus, Finding, Extract, SessionStatus,
                            # TurnContext, EvalResult
        events.py           # SessionEvent
        state.py            # SessionState enum
    protocols.py            # ConversationStore, EventSink, LLMClient,
                            # RespondentSimulator (all async)
    engine.py               # Engine class
    livekit_config.py       # LiveKitConfig dataclass
    loop/
        __init__.py
        runner.py           # main loop body
        selection.py        # select_next_goal (pure)
        phrasing.py         # word-count + question-count validator
        extract.py          # derive_extract_with_llm
        heuristics.py       # refusal/IDK keyword sets
        resume.py           # resume-acknowledgement template
    llm/
        __init__.py
        anthropic.py        # AnthropicLLMClient
    voice/
        __init__.py
        livekit_entry.py    # Engine.entrypoint helpers + InterviewerLLM subclass
    stores/
        __init__.py
        memory.py           # InMemoryConversationStore
        sqlite.py           # SQLiteConversationStore (aiosqlite)
    sinks/
        __init__.py
        memory.py           # InMemoryEventSink
        logging.py          # LoggingEventSink
        webhook.py          # WebhookEventSink (httpx)
    testing/
        __init__.py
        fake_llm.py         # FakeLLMClient
        simulators.py       # ScriptedSimulator, TerseEvasive, etc.
examples/
    simulated.py            # FakeLLMClient + ScriptedSimulator (no API keys)
    local_voice.py          # AnthropicLLMClient + LiveKit AgentSession
    join_page.html          # static page for respondent to join the room
tests/
    ...                     # one test module per src module + integration/
```

There is no `examples/local_cli.py` and no `voice/cli.py` — see
DECISIONS.md D8.

---

## Steps

### Step 0: Verify decisions are locked (verification-only)

- [x] complete

**Goal:** Confirm DECISIONS.md and SCOPE.md are in their pre-build
resolved state before any code is written. **This step produces no diff
and no commit** — it is a checklist gate.

**Prerequisites:** none.

**Read first:**
- DECISIONS.md (whole file)
- SCOPE.md → Public API, Defaults, Open Questions

**Verification checklist (all must be true):**

- DECISIONS.md contains entries D1 through at least D15.
- SCOPE.md's "Open questions" section title reads "Open questions —
  resolved before implementation".
- SCOPE.md defines `Finding` and `SessionStatus` types.
- SCOPE.md's protocols are all `async def`.
- SCOPE.md does not reference `VoiceTransport`, `compose_and_evaluate`,
  or `speak_filler` (except in explicit "removed" callouts).
- This PLAN.md's module layout includes `loop/`, `llm/`, `voice/`,
  `stores/`, `sinks/`, `testing/` and no `voice/cli.py`.

**On success:** mark this step's checkbox complete in PLAN.md. **Do not
commit** — the next code-producing step (Step 1) bundles the checkbox
change into its own commit.

**On failure:** stop. The blocker is upstream of the build. Resolve the
mismatch in DECISIONS.md or SCOPE.md before proceeding to Step 1.

---

### Step 1: Bootstrap project

- [x] complete

**Goal:** Empty Python package with build system, test runner, linter, and
type checker configured and green.

**Prerequisites:** Step 0.

**Read first:**
- SCOPE.md (whole document, for context)
- Global conventions above

**Deliverables:**

- `pyproject.toml` — project name `interviewer`, version `0.0.1`, Python
  `>=3.11`, build backend `hatchling`. Runtime deps: `pydantic>=2,<3`. Dev
  deps: `pytest`, `pytest-asyncio`, `ruff`, `mypy`. Configure
  `[tool.ruff]` (line-length 100, target-version py311),
  `[tool.mypy]` (strict = true, python_version = "3.11"),
  `[tool.pytest.ini_options]` (asyncio_mode = "auto", testpaths = ["tests"]).
- `src/interviewer/__init__.py` — empty for now.
- `src/interviewer/py.typed` — empty file.
- `tests/__init__.py` — empty.
- `tests/test_smoke.py` — `assert importlib.import_module("interviewer")`.
- `.gitignore` — Python standard.
- `README.md` — one paragraph, link to SCOPE.md.

**Acceptance criteria:**

- `uv sync --all-extras --dev` succeeds.
- `uv run pytest` passes.
- `uv run ruff check .` clean.
- `uv run mypy src/` clean.

**Constraints:**

- Do not implement any types, protocols, or engine code yet.
- Do not pin `anthropic` or `livekit-agents` — they're added in later steps.

**On success:** mark complete, commit `step 1: bootstrap project`.

---

### Step 2: Configuration types

- [x] complete

**Goal:** `Persona`, `Background`, `Goal`, `Conversation` as immutable
Pydantic models with full validation.

**Prerequisites:** Step 1.

**Read first:**
- SCOPE.md → Public API → Configuration types
- DECISIONS.md → D6

**Deliverables:**

- `src/interviewer/types/__init__.py` — empty namespace.
- `src/interviewer/types/config.py` — `Persona`, `Background`, `Goal`,
  `Conversation`. All `model_config = ConfigDict(frozen=True)`. Field
  definitions match SCOPE.md exactly.
- Validation rules:
  - `Background.relevant_context`: max 1000 chars; **raises
    `ValidationError` on overflow** (no silent truncation, no `warnings.warn`
    — see DECISIONS.md D6).
  - `Conversation`: goal IDs unique across `goals`; every
    `Goal.depends_on` entry references an existing goal id;
    `max_total_turns >= 4`; `max_tangent_followups >= 0`.
  - `Persona.style`: must match the literal type.
- `src/interviewer/__init__.py` — re-export `Persona`, `Background`,
  `Goal`, `Conversation`.
- `tests/types/test_config.py` — happy + failing cases per validation rule;
  frozen behavior; Pydantic round-trip
  (`model_dump_json` → `model_validate_json`).

**Acceptance criteria:**

- `uv run pytest tests/types/test_config.py -v` passes.
- `uv run ruff check src/ tests/` clean.
- `uv run mypy src/` clean.

**Constraints:**

- Frozen models; mutate via `model_copy(update={...})` only.
- Do not import from any other `interviewer` module — this is the
  foundation layer.

**On success:** mark complete, commit `step 2: configuration types`.

---

### Step 3: Runtime types, state enums, events, LLM I/O types

- [x] complete

**Goal:** Every runtime type from SCOPE.md, plus the state enum, event
type, and the LLM I/O types (`TurnContext`, `EvalResult`).

**Prerequisites:** Step 2.

**Read first:**
- SCOPE.md → Public API → Runtime types, State model, Lifecycle events,
  LLM I/O types
- SCOPE.md → Canonical truth subsection
- DECISIONS.md → D5, D10, D11

**Deliverables:**

- `src/interviewer/types/state.py` — `SessionState` (StrEnum, six values).
- `src/interviewer/types/runtime.py` — `Session` (with
  `conversation_snapshot: Conversation` per D10), `SessionCredentials`,
  `SessionRuntimeState`, `Turn`, `GoalStatus`, `Finding`, `Extract`,
  `SessionStatus`, `TurnContext`, `EvalResult`. Match SCOPE exactly.
  `Session.state` defaults to `CREATED`. `EvalResult.next_action` is the
  literal union `"advance" | "retry" | "drill" | "close"`.
- `src/interviewer/types/events.py` — `SessionEvent` with the literal
  event-type union. Document in module docstring that
  `goal_status_changed` is emitted only at completion (D5) and that
  `turn_recorded.payload` includes usage telemetry (D11).
- Update `src/interviewer/__init__.py` to re-export all the above.
- `tests/types/test_runtime.py` — round-trip tests, defaults, enum
  membership, `Session.conversation_snapshot` survives serialization.

**Acceptance criteria:**

- `uv run pytest tests/types/` passes.
- `uv run ruff check .` clean.
- `uv run mypy src/` clean.

**Constraints:**

- No imports from `protocols.py`, `engine.py`, or `loop/` — types are
  leaves in the dep graph.
- Do not invent fields not in SCOPE.

**On success:** mark complete, commit `step 3: runtime types`.

---

### Step 4: Async protocols + in-memory reference implementations

- [ ] complete

**Goal:** All consumer-facing `Protocol`s defined as **async**, plus
minimal in-memory implementations of `ConversationStore` and `EventSink`.

**Prerequisites:** Step 3.

**Read first:**
- SCOPE.md → Public API → Consumer-implemented protocols
- DECISIONS.md → D2, D3

**Deliverables:**

- `src/interviewer/protocols.py` — `ConversationStore`, `EventSink`,
  `LLMClient`, `RespondentSimulator` as `typing.Protocol`s. ALL methods
  that do I/O are `async def` (D3). `LLMClient` has three methods per D2:
  - `async def evaluate_turn(self, ctx: TurnContext) -> EvalResult: ...`
  - `def compose_utterance(self, ctx: TurnContext, eval_result:
    EvalResult) -> AsyncIterator[str]: ...`
  - `async def derive_extract(self, transcript: list[Turn], conv:
    Conversation) -> Extract: ...`
  Use `runtime_checkable` only on `RespondentSimulator`.
- `src/interviewer/stores/__init__.py` — empty namespace.
- `src/interviewer/stores/memory.py` — `InMemoryConversationStore`.
  Thread-safe via `asyncio.Lock`; all methods async. Sets the contract for
  real impls.
- `src/interviewer/sinks/__init__.py` — empty namespace.
- `src/interviewer/sinks/memory.py` — `InMemoryEventSink` with
  `async def emit` appending to a list; expose `.events` for assertions.
- `tests/stores/test_memory_store.py` — full protocol round-trip.
- `tests/sinks/test_memory_sink.py` — emit + ordering preserved.

**Acceptance criteria:**

- `uv run pytest tests/stores/ tests/sinks/` passes.
- `uv run mypy src/` clean.
- `uv run ruff check .` clean.

**Constraints:**

- Protocols only define signatures — no method bodies beyond `...`.
- The in-memory impls are intentionally minimal.

**On success:** mark complete, commit `step 4: async protocols and in-memory impls`.

---

### Step 5: Engine — non-loop async methods

- [ ] complete

**Goal:** `Engine` class with all non-loop async methods working end-to-end
against the in-memory store and sink. `entrypoint` and `simulate_session`
are stubbed (raise `NotImplementedError`). `provision_session` works
without LiveKit by accepting `livekit=None` (returns dummy creds).

**Prerequisites:** Step 4.

**Read first:**
- SCOPE.md → Public API → Entry points
- SCOPE.md → Store consistency
- DECISIONS.md → D3, D4, D9, D10

**Deliverables:**

- `src/interviewer/livekit_config.py` — `LiveKitConfig` frozen dataclass
  (url, api_key, api_secret, agent_name).
- `src/interviewer/engine.py` — `Engine.__init__(store, events, llm,
  livekit=None)`. Implement async: `create_conversation`,
  `provision_session` (creates Session with conversation_snapshot per D10;
  if `livekit is None`, returns `SessionCredentials` with placeholder
  room_url+token+expires_at=now+24h; if set, defers actual LiveKit room
  creation to Step 13 — for now generate a stable room name `iv:{session_id}`
  and a placeholder token), `reprovision_session`, `cancel_session`
  (writes ABANDONED + emits, room-delete logic deferred to Step 13),
  `get_session_status`, `get_transcript`, `get_extract`. Stubs for
  `entrypoint`, `simulate_session`.
- All state transitions emit appropriate `SessionEvent`s. No
  `goal_status_changed` events anywhere yet (D5).
- Update `src/interviewer/__init__.py` to re-export `Engine`,
  `LiveKitConfig`, `SessionStatus`.
- `tests/test_engine_methods.py` — each method, including error cases
  (unknown id, double-cancel, reprovision on completed session).

**Acceptance criteria:**

- `uv run pytest tests/test_engine_methods.py` passes.
- `uv run mypy src/` clean.

**Constraints:**

- Use the in-memory store/sink from Step 4. Do not pull in LiveKit yet.
- The Conversation snapshot is taken inside `provision_session` and stored
  on the `Session` (D10).

**On success:** mark complete, commit `step 5: engine methods`.

---

### Step 6: Goal selection logic

- [ ] complete

**Goal:** `select_next_goal` as a **pure function** with no LLM callable.
Honors default order and dependencies. Redundancy is applied via
`GoalStatus.status == "skipped_redundant"` set externally (by the loop
based on `EvalResult.redundant_goal_ids`).

**Prerequisites:** Step 3 (types only).

**Read first:**
- SCOPE.md → Conversation flow → step 3a
- DECISIONS.md → D2

**Deliverables:**

- `src/interviewer/loop/__init__.py` — empty namespace.
- `src/interviewer/loop/selection.py` — `select_next_goal(conversation:
  Conversation, goal_statuses: list[GoalStatus]) -> Goal | None`. Logic:
  - Filter out goals with status in `{"meets", "skipped_redundant",
    "gave_up"}`.
  - Require all `depends_on` entries to be in `{"meets",
    "skipped_redundant"}` (a redundant dependency still satisfies it).
  - Among eligible goals, return the first by operator-default order.
  - Returns `None` when no goal is eligible (all resolved).
- `tests/loop/test_selection.py` — cases: default order; dependency
  ordering; dependency satisfied by skipped_redundant; all-resolved → None;
  cycle detection (depends_on cycle should raise on `Conversation`
  validation, not here — but assert no infinite loop in selection).

**Acceptance criteria:**

- `uv run pytest tests/loop/test_selection.py` passes.
- `uv run mypy src/` clean.

**Constraints:**

- No callable parameters. No LLM calls. No I/O.
- Do not implement loop runner here.

**On success:** mark complete, commit `step 6: goal selection logic`.

---

### Step 7: Voice-phrasing validator

- [ ] complete

**Goal:** Pure function that checks an utterance for word-count and
question-count, returning either OK or a list of failures. **No keyword
detection, no splitter** (per DECISIONS.md D7).

**Prerequisites:** Step 1.

**Read first:**
- SCOPE.md → Voice phrasing constraints
- DECISIONS.md → D7

**Deliverables:**

- `src/interviewer/loop/phrasing.py`:
  - `PhrasingFailure` enum: `TOO_LONG`, `MULTI_QUESTION`, `EMPTY`.
  - `validate_voice_phrasing(text: str, max_words: int = 25) ->
    list[PhrasingFailure]` — returns empty list on pass.
- `tests/loop/test_phrasing.py` — happy case, exactly 25 words, 26 words,
  multi-question, empty/whitespace.

**Acceptance criteria:**

- `uv run pytest tests/loop/test_phrasing.py` passes.
- `uv run mypy src/` clean.

**Constraints:**

- No regex magic. Word-tokenize via `text.split()` is fine.
- No `split_compound_utterance` function — explicitly out of scope (D7).

**On success:** mark complete, commit `step 7: voice phrasing validator`.

---

### Step 8: Happy-path simulator loop + simulated example

- [ ] complete

**Goal:** `Engine.simulate_session` runs end-to-end against a
`RespondentSimulator` and a scripted `FakeLLMClient`, producing a complete
`Extract`. First demoable behavior.

**Prerequisites:** Steps 4, 5, 6, 7.

**Read first:**
- SCOPE.md → Conversation flow
- DECISIONS.md → D2, D5

**Deliverables:**

- `src/interviewer/testing/__init__.py` — empty namespace.
- `src/interviewer/testing/fake_llm.py` — `FakeLLMClient` implementing
  `LLMClient`:
  - `evaluate_turn`: driven by a queue of scripted `EvalResult` values;
    raises on underflow.
  - `compose_utterance`: yields a queued utterance string in 1–3 chunks
    (simulating streaming).
  - `derive_extract`: reads `Turn.addressed_goal_ids` (set by
    `evaluate_turn` results applied during the loop) and constructs a
    plausible Extract.
- `src/interviewer/testing/simulators.py` — `ScriptedSimulator` (driven
  by a list of pre-written respondent utterances). `TerseEvasiveSimulator`,
  `RamblyKnowledgeableSimulator`, `ConfusedSimulator` as concrete classes
  using simple text rules (no LLM).
- `src/interviewer/loop/runner.py` — the loop body. `async def
  run_loop(engine, session_id, simulator)`. Implements the flow from
  SCOPE → Conversation flow:
  - opening
  - main loop: select_next_goal → evaluate_turn (skipped on first turn)
    → apply EvalResult to GoalStatus table → flush runtime state →
    compose_utterance (accumulate stream) → validate_voice_phrasing
    (regen once on failure) → record agent turn → simulator.respond →
    record respondent turn → emit `turn_recorded`
  - closing
  - derive_extract → save Extract → COMPLETED + emit `completed`
- `src/interviewer/engine.py` — implement `simulate_session` to call
  `run_loop` with a simulator (no LiveKit, no AgentSession).
- `examples/simulated.py` — runnable, no API keys: builds a small
  Conversation, runs `simulate_session` with `FakeLLMClient` and
  `ScriptedSimulator`, prints transcript + extract.
- `tests/loop/test_runner_happy_path.py` — Conversation with 3 goals,
  scripted LLM+simulator, assert all goals reach `meets`, transcript has
  expected turn count, extract is well-formed, no `goal_status_changed`
  events emitted before `completed` (D5).

**Acceptance criteria:**

- `uv run pytest tests/loop/test_runner_happy_path.py` passes.
- `uv run python examples/simulated.py` exits 0 and prints a transcript.
- `uv run mypy src/` clean.

**Constraints:**

- Do not implement unhappy paths in this step — that's Step 9.
- Do not implement runtime-state persistence beyond what's needed for the
  happy path. Step 10 makes resume work.
- Phrasing validator is wired with regen-once behavior now.

**On success:** mark complete, commit `step 8: simulator happy path`.

---

### Step 9: Unhappy paths in the loop

- [ ] complete

**Goal:** Every unhappy path from SCOPE.md handled in `run_loop` (simulator
mode), with tests covering each.

**Prerequisites:** Step 8.

**Read first:**
- SCOPE.md → Unhappy paths (whole section)
- DECISIONS.md → D4

**Deliverables:**

- Loop changes in `src/interviewer/loop/runner.py`:
  - **Refusal / IDK:** detect via heuristic on respondent utterance
    (keyword set: "won't answer", "rather not", "don't know", "no idea",
    "no comment", …). On detection: trigger single deflection probe
    (compose with hint `last_phrasing_failure=None` and a runtime context
    flag); consume retry. On second refusal/IDK on same goal, mark
    `gave_up` and advance.
  - **LLM API failure:** wrap `evaluate_turn` and `compose_utterance` in
    retry-with-backoff (3 attempts, exponential, no `tenacity`). On
    persistent failure: speak short apology utterance, set FAILED, emit,
    return.
  - **Turn cap:** when `total_turns` reaches `max_total_turns`, do not
    start a new probe; finish current utterance, go to closing.
  - **Cancel mid-loop (simulator):** check session state from store at
    each iteration boundary; on ABANDONED, speak short closing, exit
    cleanly. (Voice-mode cancel is via room-delete in Step 13 per D4.)
- `src/interviewer/loop/heuristics.py` — refusal/IDK keyword lists,
  mutable for future tuning.
- `tests/loop/test_runner_unhappy.py` — one test per unhappy path. Use
  scripted simulators yielding empty / refusal / IDK; use `FakeLLMClient`
  configured to raise after N calls.

**Acceptance criteria:**

- `uv run pytest tests/loop/test_runner_unhappy.py -v` — all pass.
- `examples/simulated.py` still works (regression).

**Constraints:**

- Refusal/IDK detection is a pure function over utterance text — no LLM
  call to decide. The LLM does the deflection-probe wording.
- Retry/backoff uses `asyncio.sleep`. No `tenacity`.

**On success:** mark complete, commit `step 9: unhappy paths`.

---

### Step 10: Runtime state persistence and resume

- [ ] complete

**Goal:** `SessionRuntimeState` is flushed before every agent utterance;
`simulate_session` resumes from saved state when present.

**Prerequisites:** Step 9.

**Read first:**
- SCOPE.md → Runtime types → `SessionRuntimeState`
- SCOPE.md → Store consistency
- SCOPE.md → Conversation flow → step 1 (resume case)
- DECISIONS.md → D9

**Deliverables:**

- Loop changes: before each agent utterance, build `SessionRuntimeState`
  reflecting `active_goal_id`, `retries_used_on_active`,
  `tangent_followups_used`, `total_turns`, `pending_follow_up`,
  `last_event_index`; call `store.save_runtime_state`.
- On `run_loop` entry: `await store.load_runtime_state(session_id)`. If
  present, rehydrate counters and use the resume acknowledgement template
  as the first agent utterance.
- `src/interviewer/loop/resume.py` — module with the resume
  acknowledgement template constant: `RESUME_ACK = "we got cut off — let
  me pick up where we left off."` Marked as known-good (skips phrasing
  validation per its stable wording).
- `tests/loop/test_runner_resume.py` — simulate a crash by stopping the
  loop after N turns (raise from the simulator), restart `simulate_session`,
  assert resume completes correctly with no duplicate turns.

**Acceptance criteria:**

- `uv run pytest tests/loop/test_runner_resume.py` passes.
- All prior tests still pass.

**Constraints:**

- Resume is idempotent — calling `simulate_session` on a completed
  session is a no-op returning the existing extract.
- Use only fields in SCOPE's `SessionRuntimeState`.

**On success:** mark complete, commit `step 10: runtime state and resume`.

---

### Step 11: derive_extract — canonical post-hoc mapping

- [ ] complete

**Goal:** A real `derive_extract` that produces canonical
`GoalStatus.evidence_turn_indices` over the full transcript, then emits
`goal_status_changed` events for the diff against the loop-time
best-guesses (D5, D13).

**Prerequisites:** Steps 8–10.

**Read first:**
- SCOPE.md → Canonical truth subsection
- DECISIONS.md → D5, D13

**Loop-time goal-status tracking (the hint table):**

The runner already builds and updates a `dict[str, GoalStatus]` keyed by
goal id during the loop — every `EvalResult` updates `active_goal_status`,
`redundant_goal_ids`, and retry counters on that table. This in-memory
table is the "loop-time hint" used for the diff against the canonical
Extract. It is also what the runner reads to call `select_next_goal`. It
is not persisted as canonical state; only `Turn.addressed_goal_ids` is
persisted per turn as a durable hint.

**Deliverables:**

- `src/interviewer/loop/extract.py` — `derive_extract_with_llm(transcript:
  list[Turn], conversation: Conversation, llm: LLMClient) -> Extract`. Calls
  `llm.derive_extract`; returns the Extract unchanged. The LLM client is
  responsible for the structure.
- `src/interviewer/loop/runner.py`:
  - Make the loop-time `goal_status_table: dict[str, GoalStatus]`
    explicitly named in the runner state (not buried in a local
    variable). After each `EvalResult` is applied, this table is the
    runner's current best-guess of per-goal status.
  - At completion (flow step 6): take a snapshot of `goal_status_table`,
    call `derive_extract_with_llm`, then compute the diff:
    `diff = [goal_id for goal_id in canonical if canonical[goal_id].status
    != snapshot[goal_id].status]`. For each diffed goal, emit a
    `goal_status_changed` event with payload `{goal_id, from_status,
    to_status, rationale}`. Emit them BEFORE the `completed` event.
  - The `completed` event payload includes the final canonical
    `goal_statuses` table plus aggregate eval-call usage (D11).
- `FakeLLMClient.derive_extract` upgraded to read
  `Turn.addressed_goal_ids` and construct a plausible Extract. Add a
  test-only `force_disagreement_for: list[str]` constructor parameter
  that lets a test deliberately flip the status of specified goals so the
  diff path is exercised.
- `tests/loop/test_extract.py`:
  - Synthetic transcripts with hand-built hints; assert canonical
    `evidence_turn_indices` matches.
  - Test the diff path: configure FakeLLMClient with
    `force_disagreement_for=["g2"]`; assert exactly one
    `goal_status_changed` event for `g2` is emitted, and that it is
    ordered before the `completed` event in the sink's event list.

**Acceptance criteria:**

- `uv run pytest tests/loop/test_extract.py` passes.
- End-to-end test from Step 8 still produces a valid Extract.

**Constraints:**

- `derive_extract` is the single source of truth for
  `GoalStatus.evidence_turn_indices`.
- `goal_status_changed` events are only emitted at completion (D5).

**On success:** mark complete, commit `step 11: derive_extract`.

---

### Step 12: AnthropicLLMClient

- [ ] complete

**Goal:** A real LLM client implementing the three methods. Verified via
`examples/simulated.py` (substituting AnthropicLLMClient for FakeLLMClient
behind ANTHROPIC_API_KEY) and unit tests with mocked responses.

**Prerequisites:** Steps 8–11.

**Read first:**
- SCOPE.md → Defaults → `AnthropicLLMClient`
- SCOPE.md → Latency strategy
- DECISIONS.md → D2, D11, D12, D15

#### Module structure

- `src/interviewer/llm/__init__.py` — empty namespace.
- `src/interviewer/llm/anthropic.py` — `AnthropicLLMClient` class.
- `src/interviewer/llm/prompts.py` — prompt builders (see "Prompt design"
  below). Kept separate so they can be unit-tested without instantiating
  the Anthropic client.
- `src/interviewer/llm/schemas.py` — Anthropic tool-input schema derivation
  helper (see "Tool schema derivation" below).

#### Prompt design

The system prompt is identical across all three LLM methods so the
ephemeral prompt cache (D2) is shared between `evaluate_turn` and
`compose_utterance` within the 5-minute window. Build it once per
TurnContext via `prompts.build_system_prompt(conversation)`:

```
You are the interviewer described below. You are conducting a voice
interview. Respond as the interviewer would, in short conversational
sentences, one question at a time.

# Your persona
{conversation.persona.system_prompt}
Style: {conversation.persona.style}.

# Why you are talking to this person
{conversation.purpose}

# Who they are
Role: {conversation.background.interviewee_role}
Expertise: {conversation.background.interviewee_expertise}
Additional context: {conversation.background.relevant_context or "(none)"}

# What you are trying to find out
For each goal you have an INTENT (what you want to know), a STANDARD
("answered well enough" rubric), and optionally a REDUNDANT_WHEN rubric
("skip if earlier answers covered this").

{for each goal in conversation.goals:}
## Goal {goal.id}
INTENT: {goal.intent}
STANDARD: {goal.standard}
REDUNDANT_WHEN: {goal.redundant_when or "(no redundancy rubric)"}

# Voice phrasing rules (must follow)
- One question per utterance. No "first... then..." enumerations.
- 25 words or fewer per utterance.
- Conversational, not written.
```

Mark this whole block as cached:
`system=[{"type":"text","text":<the above>,"cache_control":{"type":"ephemeral"}}]`.
The user message is short and varies per turn; it's NOT cached.

**For `evaluate_turn`** the user message is:

```
The conversation so far:
{format last K turns, default K=12, oldest first; agent prefixed
 "AGENT:", respondent prefixed "RESPONDENT:"}

The active goal is: {ctx.active_goal.id} — {ctx.active_goal.intent}
The respondent's most recent answer is the final RESPONDENT turn above.

Call the `evaluate` tool with your assessment.
```

Forced via `tool_choice={"type":"tool","name":"evaluate"}`.

**For `compose_utterance`** the user message is:

```
The conversation so far:
{same transcript format as evaluate}

The previous evaluation determined:
- active_goal_status: {eval_result.active_goal_status}
- next_action: {eval_result.next_action}
- {if eval_result.next_action == "drill"}: interesting_tangent:
  {eval_result.interesting_tangent}

Write the next single voice utterance, following the voice phrasing rules.
Output only the utterance text, no preamble.
{if ctx.last_phrasing_failure: "Your previous attempt failed: {failure}.
 Fix it."}
```

No tools, plain streaming.

**For `derive_extract`** the user message is:

```
The full transcript of an interview:
{format all turns; include turn index prefix [N]}

For each goal, decide the canonical status and which turn indices contain
evidence. Also extract any unprompted findings — claims the respondent
volunteered that weren't directly asked about.

Call the `extract` tool with the structured Extract.
```

Forced via `tool_choice={"type":"tool","name":"extract"}`.

#### Tool schema derivation (D15)

Anthropic's `input_schema` accepts a subset of JSON Schema and rejects
certain Pydantic-emitted keys. Provide a helper:

```python
# llm/schemas.py
def anthropic_tool_schema(model: type[BaseModel]) -> dict:
    schema = model.model_json_schema()
    # Inline $defs into property definitions
    schema = _inline_defs(schema)
    # Strip Pydantic-only keys Anthropic doesn't accept
    for key in ("title", "$defs", "definitions"):
        schema.pop(key, None)
    for prop in schema.get("properties", {}).values():
        prop.pop("title", None)
    return schema
```

Two tools:
- `evaluate` — `input_schema=anthropic_tool_schema(EvalResult)`.
- `extract` — `input_schema=anthropic_tool_schema(Extract)`.

Tested in `tests/llm/test_schemas.py` against the Anthropic SDK's
`Tool` Pydantic model (which validates schemas on the client side).

#### `AnthropicLLMClient` class

```python
# constants at module top
EVAL_MODEL    = "claude-haiku-4-5"
COMPOSE_MODEL = "claude-sonnet-4-6"
EXTRACT_MODEL = "claude-sonnet-4-6"

class AnthropicLLMClient:
    def __init__(
        self,
        api_key: str,
        *,
        eval_model: str = EVAL_MODEL,
        compose_model: str = COMPOSE_MODEL,
        extract_model: str = EXTRACT_MODEL,
        max_transcript_turns: int = 12,
    ) -> None:
        self._client = anthropic.AsyncAnthropic(api_key=api_key)
        ...

    last_eval_usage: Usage | None  # side-channel for the runner (D11)
    last_compose_usage: Usage | None
    last_extract_usage: Usage | None
```

Where `Usage` is a small dataclass with fields `tokens_in, tokens_out,
cache_read_tokens, cache_write_tokens, llm_latency_ms`. Defined in
`llm/anthropic.py`. Set after each call, before the method returns or
the iterator exhausts.

#### Implementation specifics

- `anthropic.AsyncAnthropic.messages.stream(...)` with `async for` over
  `stream.text_stream` for `compose_utterance`. After the iterator
  exhausts, read `stream.get_final_message()` to capture usage.
- `anthropic.AsyncAnthropic.messages.create(...)` (non-streaming) for
  `evaluate_turn` and `derive_extract`. Inspect
  `response.content[0]` for the `tool_use` block; `json.loads(...)` its
  `input` is the structured object. Validate via
  `EvalResult.model_validate(...)` or `Extract.model_validate(...)` —
  raises `ValidationError` if the model returned malformed JSON, which
  the retry loop in Step 9 catches.
- Verify Anthropic SDK version against current PyPI at implementation
  time. Pin `anthropic>=0.40,<1.0` or current stable major; record the
  pinned version in DECISIONS.md if a different one is chosen.

#### Other deliverables

- `pyproject.toml`: add `anthropic` to runtime deps.
- `src/interviewer/loop/runner.py` — read `llm.last_compose_usage` and
  `llm.last_eval_usage` after each LLM call. Build `turn_recorded`
  payload from `last_compose_usage`. Accumulate `last_eval_usage` and
  emit aggregated totals on the `completed` event (D11).
- `examples/simulated.py` gains a `--use-anthropic` flag that swaps in
  `AnthropicLLMClient`. Still simulator-mode (no voice).
- `tests/llm/test_anthropic.py` — mock `AsyncAnthropic` at the SDK
  level. Cases: `evaluate_turn` returns parsed `EvalResult`;
  `compose_utterance` yields chunks and captures usage; `derive_extract`
  parses the tool's input into `Extract`; system prompt has
  `cache_control` set; tool input_schema lacks `title` and `$defs`.
- `tests/llm/test_prompts.py` — golden-file tests for
  `build_system_prompt(conversation)` output shape (snapshot test, not
  exact wording match — assert section headers and goal IDs appear).

**Acceptance criteria:**

- `uv run pytest tests/llm/` passes.
- `ANTHROPIC_API_KEY=… uv run python examples/simulated.py --use-anthropic`
  runs a full simulated interview against the real API and prints a
  transcript + extract. Acceptance is "completes without crashing and
  extract has the right shape." (Cost: ~10–30 cents per run.)
- `uv run mypy src/` clean.

**Constraints:**

- Do not mock the Anthropic SDK in non-test code.
- Retry/backoff lives in the loop (Step 9), not here. The client raises
  on failure (`anthropic.APIError`, `pydantic.ValidationError`).
- No CLI voice transport (D8).

**On success:** mark complete, commit `step 12: anthropic client`.

---

### Step 13: LiveKit AgentSession integration + voice example

- [ ] complete

**Goal:** A user can speak to the agent and hear the agent speak back via
a LiveKit room, end-to-end. The engine's `entrypoint(ctx)` adapts our
`LLMClient` into a `livekit-agents` `LLM` subclass, wired into an
`AgentSession` with Deepgram STT and Cartesia TTS.

**Prerequisites:** Steps 8–12.

**Read first:**
- SCOPE.md → Defaults → entrypoint
- SCOPE.md → Latency strategy
- SCOPE.md → Conversation flow
- DECISIONS.md → D1, D4, D9, D10, D14
- LiveKit Agents docs: https://docs.livekit.io/agents/. The
  `livekit-agents` 1.x API is the binding reference; if the API has
  shifted, prefer the framework's current API and note divergences in
  DECISIONS.md.

#### How our two-call flow fits into one `chat()` invocation

`AgentSession` calls `InterviewerLLM.chat(chat_ctx, ...)` once per
respondent turn (after STT finalizes a transcript). The method must
return an `LLMStream` whose iteration yields the assistant's text tokens,
which AgentSession streams to TTS.

Inside one `chat()` call:

```
1. The latest user message in chat_ctx is the new respondent utterance.
   Append it to our store as a respondent Turn. Emit `turn_recorded`.
2. Build a TurnContext from the per-session SessionState object that
   InterviewerLLM holds: conversation_snapshot, current goal_status_table,
   retries_used_on_active, tangent_followups_used, total_turns,
   transcript-so-far (loaded from store; the chat_ctx history is a
   secondary source — we trust the store).
3. Call select_next_goal(); update ctx.active_goal.
4. If select_next_goal returned None → emit terminating signal: yield a
   short closing utterance into the stream, then return.
5. Await self.llm_client.evaluate_turn(ctx). Apply EvalResult to the
   goal_status_table (active_goal_status, redundant_goal_ids,
   retries/tangent counters per next_action).
6. If next_action == "close" OR total_turns >= max_total_turns →
   yield closing utterance, return.
7. Build updated SessionRuntimeState and await store.save_runtime_state.
8. Start streaming self.llm_client.compose_utterance(ctx, eval_result).
   Yield each chunk into the LLMStream. Accumulate full text alongside.
9. After the stream exhausts: validate_voice_phrasing(full_text). If it
   fails AND no regen attempt has been made yet, restart from step 8
   with ctx.last_phrasing_failure set. (If second attempt also fails,
   accept verbatim per D7.)
10. Append agent Turn to store with addressed_goal_ids derived from
    EvalResult; emit `turn_recorded` with usage telemetry from
    `llm.last_compose_usage` and `llm.last_eval_usage`.
11. Return from chat().
```

The `LLMStream` subclass is the framework's primitive for yielding
chunked content. Implement the minimal interface: an async iterator of
`ChatChunk` objects (or current equivalent) wrapping text deltas.

Per-session state (`goal_status_table`, counters, run_id) lives on the
`InterviewerLLM` instance itself — one InterviewerLLM is created per
session inside `entrypoint`, not shared across sessions.

#### Opening utterance trigger

`AgentSession` waits for user audio by default. To make the agent speak
first, the entrypoint calls `await session.generate_reply(instructions=
opening_instructions)` AFTER `session.start(...)`. This causes AgentSession
to invoke `chat()` with a special "opening" marker the InterviewerLLM
detects (e.g., `chat_ctx` has no user messages yet) and produces:

- If `SessionRuntimeState` exists → yield `RESUME_ACK` from
  `loop/resume.py`. Skip evaluate; just speak the resume line.
- Else if `conversation.opening` is set → yield it verbatim.
- Else → compose a default opening via `llm.compose_utterance` with a
  synthetic eval_result indicating "this is the first turn, introduce
  yourself and ask about the first goal."

#### Closing and termination

When the loop reaches a terminal condition mid-`chat()` (no more goals,
turn cap, eval says `close`), InterviewerLLM yields the closing utterance
(scripted from `conversation.closing` if set, else a brief default), then
the entrypoint awaits `session.aclose()` to disconnect cleanly.

#### Disconnect handling

Register a callback on `ctx.room.on("disconnected", ...)` (current
livekit-agents pattern) that:
1. Checks whether the loop reached COMPLETED (all goals resolved or
   closing spoken) vs. mid-flight.
2. If COMPLETED: call `derive_extract_with_llm`, save, emit `completed`.
3. If mid-flight: write ABANDONED, preserve runtime state, emit
   `abandoned`.

#### Cancel-from-operator

`Engine.cancel_session(session_id)` is callable from the web app process,
not the worker. It:
1. Writes ABANDONED to the store; emits `abandoned`.
2. Calls `livekit.api.RoomService.delete_room(name=room_name)` over
   HTTPS to the LiveKit server.
3. The room close propagates to the worker; AgentSession exits;
   `disconnected` callback above fires and sees ABANDONED state, skips
   the derive_extract path (or runs it on the partial transcript per
   policy — for v1: skip).

#### Deliverables

- `pyproject.toml`: add `livekit-agents>=1.0` as a runtime dep with
  extras `[deepgram,cartesia,silero]`. Verify the current extras names
  against PyPI at implementation time (the research suggests this is the
  pattern but plugin names may have shifted). Add `livekit-api` for
  `RoomService` and `AccessToken`. Group these under an optional
  `voice` install group so simulator-only consumers can skip the audio
  deps:
  ```toml
  [project.optional-dependencies]
  voice = ["livekit-agents[deepgram,cartesia,silero]>=1.0", "livekit-api>=0.6"]
  ```
- `src/interviewer/voice/__init__.py` — empty namespace.
- `src/interviewer/voice/livekit_entry.py`:
  - `class InterviewerAgent(livekit.agents.Agent)` — minimal `Agent`
    subclass with `instructions=""` (our system prompt lives inside
    `AnthropicLLMClient`). The `Agent` class is required by
    `AgentSession.start(agent=...)`.
  - `class InterviewerLLM(livekit.agents.llm.LLM)` — holds:
    - `engine_state: PerSessionState` (a dataclass with
      goal_status_table, counters, conversation_snapshot, session_id,
      store, events, llm_client)
    - Implements `chat(chat_ctx, ...)` per the semantics above.
  - `def build_agent_session(engine: Engine, session_id: str,
    state: PerSessionState) -> AgentSession`:
    ```python
    return AgentSession(
        stt=deepgram.STT(model="nova-3", language="en"),
        llm=InterviewerLLM(state=state),
        tts=cartesia.TTS(model="sonic-2",
                         voice=state.conversation_snapshot.persona.voice_id),
        vad=silero.VAD.load(),
    )
    ```
    (Pin Deepgram `nova-3` and Cartesia `sonic-2` as defaults; verify
    these are current model IDs at implementation time. If newer
    defaults are documented, prefer them and log in DECISIONS.md.)
- `src/interviewer/engine.py` — implement `Engine.entrypoint(ctx)`:
  ```python
  async def entrypoint(self, ctx: agents.JobContext) -> None:
      session_id = ctx.job.metadata or ctx.room.name.removeprefix("iv:")
      session = await self.store.load_session(session_id)
      runtime = await self.store.load_runtime_state(session_id)
      conv = session.conversation_snapshot

      await self.store.update_session_state(session_id,
                                            SessionState.IN_PROGRESS)
      await self.events.emit(SessionEvent(type="respondent_joined", ...))

      state = PerSessionState(session_id=session_id,
                              conversation_snapshot=conv,
                              goal_status_table=_initial_status_table(conv),
                              ...,
                              store=self.store, events=self.events,
                              llm_client=self.llm)
      if runtime is not None:
          _rehydrate(state, runtime)

      agent_session = build_agent_session(self, session_id, state)
      await agent_session.start(room=ctx.room, agent=InterviewerAgent())

      # Kick off the agent speaking first.
      opening_instructions = (
          "say the resume acknowledgement and continue the interview"
          if runtime is not None else
          "deliver the opening and ask the first goal's question"
      )
      await agent_session.generate_reply(instructions=opening_instructions)

      # Wait for room disconnect; AgentSession runs the loop via
      # InterviewerLLM.chat() in the background.
      await ctx.wait_for_disconnect()

      # Disconnect path: state is either COMPLETED (loop finished cleanly)
      # or in-flight. Derive extract if appropriate.
      final_state = (await self.store.load_session(session_id)).state
      if final_state == SessionState.IN_PROGRESS:
          # Respondent dropped mid-call.
          await self.store.update_session_state(session_id,
                                                SessionState.ABANDONED)
          await self.events.emit(SessionEvent(type="abandoned", ...))
      elif final_state == SessionState.COMPLETED:
          await self._finalize_extract(session_id, state)
  ```
  Helper `_finalize_extract` calls `derive_extract_with_llm`, persists
  the Extract, computes the loop-time-hint diff (Step 11), emits
  `goal_status_changed` then `completed`.
- `Engine.cancel_session` upgraded — when `self.livekit is not None`,
  also call `livekit.api.LiveKitAPI(...).room.delete_room(name=
  f"iv:{session_id}")` after writing ABANDONED.
- `Engine.provision_session` upgraded — when `self.livekit is not None`,
  mint a real token:
  ```python
  token = (
      livekit.api.AccessToken(self.livekit.api_key, self.livekit.api_secret)
      .with_identity(f"respondent:{session_id}")
      .with_name("respondent")
      .with_grants(livekit.api.VideoGrants(
          room_join=True, room=f"iv:{session_id}",
          can_publish=True, can_subscribe=True,
      ))
      .with_ttl(timedelta(hours=24))
      .to_jwt()
  )
  return SessionCredentials(room_url=self.livekit.url, token=token,
                            expires_at=datetime.utcnow() + timedelta(hours=24))
  ```
- `examples/local_voice.py` — runnable. Structure:
  ```python
  """Run interviewer end-to-end against a local LiveKit dev server.

  Required env vars: ANTHROPIC_API_KEY, DEEPGRAM_API_KEY, CARTESIA_API_KEY,
  LIVEKIT_URL (default ws://localhost:7880), LIVEKIT_API_KEY (default
  devkey), LIVEKIT_API_SECRET (default secret).

  Setup:
    1. brew install livekit-server (or download from livekit.io)
    2. livekit-server --dev   # in a separate terminal
    3. ANTHROPIC_API_KEY=... DEEPGRAM_API_KEY=... CARTESIA_API_KEY=... \\
       uv run python examples/local_voice.py
    4. Open the printed URL in a browser — it joins as the respondent.
  """
  import asyncio, os
  from livekit import agents
  from interviewer import (Engine, LiveKitConfig, Persona, Background,
                           Goal, Conversation, AnthropicLLMClient)
  from interviewer.stores.memory import InMemoryConversationStore
  from interviewer.sinks.logging import LoggingEventSink

  def build_engine() -> Engine:
      return Engine(
          store=InMemoryConversationStore(),
          events=LoggingEventSink(),
          llm=AnthropicLLMClient(api_key=os.environ["ANTHROPIC_API_KEY"]),
          livekit=LiveKitConfig(
              url=os.environ.get("LIVEKIT_URL", "ws://localhost:7880"),
              api_key=os.environ.get("LIVEKIT_API_KEY", "devkey"),
              api_secret=os.environ.get("LIVEKIT_API_SECRET", "secret"),
              agent_name="interviewer",
          ),
      )

  async def main_provision():
      engine = build_engine()
      conv = await engine.create_conversation(
          persona=Persona(system_prompt="...", style="neutral",
                          voice_id="..."),  # cartesia voice id
          purpose="...",
          background=Background(...),
          goals=[Goal(id="g1", intent="...", standard="...")],
      )
      session, creds = await engine.provision_session(conv.id)
      join_url = f"file://{os.path.abspath('examples/join_page.html')}?" \\
                 f"url={creds.room_url}&token={creds.token}"
      print(f"\\n→ Open: {join_url}\\n")
      print("Then wait for the agent to start speaking.")

  async def worker_entrypoint(ctx: agents.JobContext):
      engine = build_engine()
      await engine.entrypoint(ctx)

  if __name__ == "__main__":
      import sys
      if "--provision" in sys.argv:
          asyncio.run(main_provision())
      else:
          agents.cli.run_app(agents.WorkerOptions(
              entrypoint_fnc=worker_entrypoint,
              agent_name="interviewer",
          ))
  ```
  Usage: run `--provision` once in one terminal, then run without args in
  another (the LiveKit worker). Open the printed URL to join.
- `examples/join_page.html` — minimal page using LiveKit JS Client SDK
  from CDN:
  ```html
  <!DOCTYPE html>
  <html><head><script src="https://unpkg.com/livekit-client/dist/livekit-client.umd.min.js"></script></head>
  <body><h1>interviewer — respondent</h1>
  <div id="status">connecting...</div>
  <script>
    const params = new URLSearchParams(window.location.search);
    const room = new LivekitClient.Room({ adaptiveStream: true });
    await room.connect(params.get("url"), params.get("token"));
    document.getElementById("status").textContent = "connected — speak";
  </script></body></html>
  ```
  Pin the unpkg path to a major version (`livekit-client@2`) at
  implementation time.
- `tests/voice/test_livekit_entry.py` — unit tests with `livekit-agents`
  mocked at the SDK level: `provision_session` token generation
  (assert identity prefix `respondent:`, room name `iv:{session_id}`,
  TTL ~24h), `cancel_session` calls `room.delete_room`,
  `InterviewerLLM.chat()` happy-path with mocked `AsyncAnthropic`.

**Acceptance criteria:**

- `uv run pytest tests/voice/` passes.
- Manual smoke test: with a local LiveKit dev server + API keys, run
  `uv run python examples/local_voice.py --provision` to mint creds,
  then `uv run python examples/local_voice.py` to start the worker,
  open the printed URL, speak with the agent for ~5 turns. Acceptance
  is "I conducted a voice interview and got a structured extract
  end-to-end."
- `uv run mypy src/` clean.

**Constraints:**

- Don't introduce a hard dependency on LiveKit Cloud; default to local.
- If `livekit-agents` 1.x API has shifted, match the current API and
  log in DECISIONS.md. Likely-stable surface: `AgentSession`,
  `JobContext`, `Agent`, `llm.LLM`, `cli.run_app`,
  `WorkerOptions(entrypoint_fnc=...)`. Plugin import paths
  (`from livekit.plugins import deepgram` vs.
  `livekit_agents.inference.STT(...)`) may have moved.
- Voice and Deepgram model IDs are mutable; pin reasonable defaults in
  code but allow consumer override via kwargs on a future
  `LiveKitConfig` field if needed.

**On success:** mark complete, commit `step 13: livekit voice integration`.

---

### Step 14: SQLite store + reference event sinks

- [ ] complete

**Goal:** A SQLite-backed `ConversationStore` and two reference
`EventSink`s (logging + webhook), suitable as starting points for consumer
adoption.

**Prerequisites:** Steps 4, 11.

**Read first:**
- SCOPE.md → Defaults
- DECISIONS.md → D3

**Deliverables:**

- `pyproject.toml`: add `aiosqlite` and `httpx` to runtime deps.
- `src/interviewer/stores/sqlite.py` — `SQLiteConversationStore`,
  fully async via `aiosqlite`. Schema with tables: `conversations`,
  `sessions` (with a `conversation_snapshot` JSON column per D10),
  `turns`, `runtime_states`, `extracts`. JSON columns for the Pydantic
  blobs; foreign keys; index on `(session_id, turn_index)`. On init,
  creates tables if missing (no migrations system in v1).
- `src/interviewer/sinks/logging.py` — `LoggingEventSink` writing events
  to a `logging.Logger` (configurable name).
- `src/interviewer/sinks/webhook.py` — `WebhookEventSink` posting events
  to a URL via `httpx.AsyncClient`. Retries 3× with exponential backoff.
- `tests/stores/test_sqlite_store.py` — same protocol round-trip suite
  used for `InMemoryConversationStore`, run against SQLite. Refactor
  `tests/stores/test_memory_store.py` into a parametrized shared suite.
- `tests/sinks/test_logging_sink.py`, `tests/sinks/test_webhook_sink.py`
  (the latter using `httpx.MockTransport`).

**Acceptance criteria:**

- `uv run pytest tests/stores/ tests/sinks/` passes.
- `uv run mypy src/` clean.

**Constraints:**

- No ORM. `aiosqlite` + JSON columns.
- The SQLite store must pass exactly the same test suite as
  `InMemoryConversationStore`.

**On success:** mark complete, commit `step 14: sqlite store and event sinks`.

---

### Step 15: Integration test suite

- [ ] complete

**Goal:** End-to-end tests covering realistic combinations of Conversation
configs and simulator personas. Catches regressions across loop, extract,
and store layers together.

**Prerequisites:** Steps 8–14.

**Read first:**
- SCOPE.md → Conversation flow
- SCOPE.md → Unhappy paths
- DECISIONS.md (whole)

**Deliverables:**

- `tests/integration/test_full_loop.py` covering:
  - "Engineer interviewing engineer" Conversation with 5 goals, all met.
  - Same Conversation against `TerseEvasiveSimulator` — expect some
    `gave_up` statuses; extract is still well-formed.
  - Crash mid-loop, resume, complete — assert no duplicate turns.
  - Cancel mid-loop (simulator path) — assert ABANDONED state and partial
    extract.
  - SQLite store backend — same scenarios.
- `tests/integration/conftest.py` — fixtures for each Conversation
  template and each simulator persona.

**Acceptance criteria:**

- `uv run pytest tests/integration/ -v` passes deterministically.
- Run the suite three times in a row; identical results
  (deterministic `FakeLLMClient` driving).

**Constraints:**

- Integration tests use `FakeLLMClient`. Do NOT hit Anthropic — that's
  Step 12.
- Simulator path only — no LiveKit in integration tests.

**On success:** mark complete, commit `step 15: integration test suite`.

---

### Step 16: Documentation

- [ ] complete

**Goal:** README + integration guide + complete docstrings on the public
API surface.

**Prerequisites:** Step 15.

**Read first:**
- SCOPE.md (whole)
- DECISIONS.md (whole)

**Deliverables:**

- `README.md` — expanded:
  - 1-paragraph "what this is"
  - quickstart: install with `uv`, run `examples/simulated.py`
  - run a local voice interview: `examples/local_voice.py`
  - link to SCOPE.md for the full design
- `docs/integration.md` — for consumer applications:
  - implementing `ConversationStore` (point at `stores/sqlite.py`)
  - implementing `EventSink` (point at `sinks/webhook.py`)
  - the `provision_session` → AgentServer-dispatch → `entrypoint` flow
  - example FastAPI + AgentServer worker sketch (matches SCOPE's example)
  - operational gaps the consumer must handle: orphaned sessions
    (provisioned but never joined), token refresh, AgentServer scaling,
    persistent storage of the conversation_snapshot column.
- Docstrings on every public class and method in `engine.py`,
  `protocols.py`, and the `types/` modules. Terse — one line per
  signature unless WHY is non-obvious.

**Acceptance criteria:**

- README quickstart commands work as written.
- A reader lands on README → quickstart → first successful run in under
  5 minutes (no API keys needed for the simulated example).

**Constraints:**

- No emojis. No hype. Terse, technical, accurate.
- Do not duplicate SCOPE.md content; link to it.

**On success:** mark complete, commit `step 16: documentation`.

---

### Step 17: Final pass — types, lint, API stability

- [ ] complete

**Goal:** mypy strict clean across the whole tree; ruff clean; the
exported API surface matches SCOPE.md exactly.

**Prerequisites:** Step 16.

**Read first:**
- SCOPE.md → Public API (whole)
- All prior step deliverables
- DECISIONS.md

**Deliverables:**

- `uv run mypy src/` clean with `strict = true`.
- `uv run ruff check . --fix` clean.
- `src/interviewer/__init__.py` exports exactly the API documented in
  SCOPE.md. Internal-only modules (`loop/heuristics.py`, etc.) are not
  re-exported.
- `pyproject.toml` version bumped to `0.1.0`.
- `CHANGELOG.md` — single entry: "0.1.0 — initial release per SCOPE.md."
- API-surface comparison written into DECISIONS.md as the closing entry:
  every public name vs. SCOPE.md; mark deliberate deviations.

**Acceptance criteria:**

- `uv run pytest && uv run mypy src/ && uv run ruff check .` all clean.
- Public API audit complete; deviations from SCOPE.md are either fixed or
  explicitly justified in DECISIONS.md.

**Constraints:**

- This is a closing pass; do NOT add new features. Missing-vs-SCOPE items
  become follow-ups, not scope expansion.

**On success:** mark complete, commit `step 17: final pass and 0.1.0
release`. Tag `v0.1.0`.
