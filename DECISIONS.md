# Decisions

Running log of non-obvious choices. `/next-step` reads this before starting and
appends entries when a step makes a notable choice.

## Format

```
## Step N — <step title>

- **Decision:** <one sentence>
- **Why:** <one sentence>
- **Affects:** <which future steps or files this constrains>
```

## What to log here vs. not

Log:
- Choices between two reasonable alternatives where the loser is non-obvious.
- Deviations from PLAN.md or SCOPE.md, with justification.
- Version pins or library substitutions forced by API drift.
- Internal abstractions a later step needs to honor.

Don't log:
- Anything obvious from reading the code.
- Routine refactors.
- Bug fixes within the same step.

---

## Step 0 — Pre-build resolutions (locked in before Step 1)

These resolve the open questions in SCOPE and the architectural risks raised in
plan review. They are binding for all subsequent steps.

### D1. LiveKit Agents framework — adopt, do not reimplement

- **Decision:** v1 wraps `livekit-agents` `AgentSession` as the voice runtime
  rather than building a `VoiceTransport` protocol with a hand-rolled audio
  loop.
- **Why:** AgentSession owns VAD (Silero), turn-detection, interruption,
  backchannel filtering, and streaming STT/TTS plumbing — reimplementing them
  is hundreds of lines of fragile code we cannot test as well as the framework
  is already tested.
- **Affects:** removes `VoiceTransport` protocol from the public API; replaces
  Step 13's `LiveKitVoiceTransport` work with an `Engine.entrypoint(ctx)`
  function the consumer registers with their `AgentServer`. The CLI voice
  transport is dropped entirely (see D9).

### D2. Two LLM calls per turn — eval (Haiku) then compose-stream (Sonnet)

- **Decision:** each agent turn is two sequential Anthropic calls:
  1. `evaluate_turn` — Claude Haiku 4.5 with `tool_choice` forced to a single
     tool returning structured JSON: `{active_goal_status,
     redundant_goal_ids, interesting_tangent, next_action}`.
  2. `compose_utterance` — Claude Sonnet 4.6 with plain text streaming
     (no tools), driven by the eval result and engine state.
- **Why:** the original plan's merged "compose+evaluate" call promised
  streaming TTS *and* structured tool-use output from one call. In practice
  forcing a specific tool suppresses preamble text and breaks streaming-to-TTS;
  text-then-tool ordering is fragile across model versions. Splitting into
  two calls costs ~200–300 ms but is robust, testable, and eliminates the
  contradiction.
- **Affects:** `LLMClient` protocol exposes `evaluate_turn` and
  `compose_utterance` (the latter as `AsyncIterator[str]`) plus
  `derive_extract`. The merged `compose_and_evaluate` from earlier drafts is
  removed. Replaces the previous `select_next_goal(redundancy_judge=...)`
  callable: redundancy is decided inside `evaluate_turn` and surfaced in its
  output, so selection becomes a pure function over `GoalStatus`.

### D3. Async-everywhere protocols

- **Decision:** every `ConversationStore` and `EventSink` method is `async
  def`. Engine methods that talk to either are async too
  (`create_conversation`, `provision_session`, `cancel_session`,
  `get_session_status`, `get_transcript`, `get_extract`,
  `reprovision_session`).
- **Why:** the engine is async; the original sync-store-with-asyncio.Lock
  design was internally inconsistent (you can't `await` a lock from a sync
  method). Async-everywhere makes the SQLite, Postgres, and webhook
  implementations natural, and the in-memory impl just uses `asyncio.Lock`.
- **Affects:** SCOPE.md → Public API → Consumer-implemented protocols
  (rewritten); Engine signatures (rewritten); test files use `await`.

### D4. Cancellation = state flag + room teardown

- **Decision:** `cancel_session` (a) writes `ABANDONED` to the store and
  emits `abandoned`, then (b) calls the LiveKit room-delete API to disconnect
  participants. The running `AgentSession` observes the room close and exits
  cleanly. For simulator mode the loop checks state at each iteration
  boundary as before.
- **Why:** the original "check state at iteration boundary" plan had no way
  to interrupt a respondent mid-monologue in voice mode. Letting LiveKit do
  the disconnect makes cancellation effectively immediate without coupling
  the engine to the audio loop's internals.
- **Affects:** Step 5 (cancel_session implementation), Step 13 (entrypoint
  must handle clean shutdown on room close).

### D5. Mid-session goal-status events suppressed

- **Decision:** the engine does NOT emit `goal_status_changed` events
  during the loop. The canonical statuses come from `derive_extract` at
  completion; the `completed` event payload includes the per-goal status
  diff. Live progress visibility for dashboards comes from `Turn` count and
  active-goal-id only.
- **Why:** the original design allowed live "best guess" status changes that
  could be revised by the final extract pass — operators would see a goal
  change from `meets` to `partial` after completion, which is a confusing UX.
  v1 trades real-time goal status for consistency.
- **Affects:** `SessionEvent` type still lists `goal_status_changed` in its
  literal union, but it is only emitted from `derive_extract`. Step 8 must
  not emit it.

### D6. `Background.relevant_context` over-length raises, never silently truncates

- **Decision:** `relevant_context` longer than 1000 chars raises a
  `pydantic.ValidationError`. No `warnings.warn`. No silent truncation.
- **Why:** silent truncation via `warnings` is invisible in production
  logging and produces non-deterministic prompts. Forcing the consumer to
  decide is the correct boundary.
- **Affects:** Step 2 validator implementation and tests.

### D7. Phrasing validator is lightweight: word count + question count, regen-once

- **Decision:** `validate_voice_phrasing` checks `len(words) <= 25` and
  `text.count("?") <= 1`. On failure, the engine regenerates the compose
  call once with the failure surfaced in the prompt. If the second attempt
  also fails, the engine speaks the utterance verbatim (does not split).
- **Why:** the keyword-based enumeration detector misses phrasings like
  "to start with"; the programmatic splitter produces ungrammatical
  fragments. Trusting Sonnet's prompt discipline plus a hard word-count
  guard is sufficient for v1 and removes ~80 lines of brittle code.
- **Affects:** Step 7 deliverables shrink; no `split_compound_utterance`
  function.

### D8. Drop CLI voice transport

- **Decision:** v1 ships two examples — `examples/simulated.py` (text-only,
  no API keys) and `examples/local_voice.py` (LiveKit + Anthropic +
  Deepgram + Cartesia). No `examples/local_cli.py`.
- **Why:** the CLI transport tests neither the real voice path nor anything
  the simulator doesn't already test, and adding it requires a
  `VoiceTransport` protocol we no longer want (see D1). Lightweight first
  version.
- **Affects:** removes Step 12's CLI deliverable; Step 12 becomes solely
  about `AnthropicLLMClient`. Step 13 is the only voice-bringup step.

### D9. Token policy and resume window

- **Decision:** `SessionCredentials.expires_at` defaults to 24 hours.
  Refresh is explicit — the consumer calls `reprovision_session` if a token
  expires. Resume of an interrupted session is permitted as long as the
  underlying `Session` state is `IN_PROGRESS` or `ABANDONED` and not older
  than 24 h since last update.
- **Why:** automatic token refresh requires a long-lived background task
  the engine doesn't own; consumer-driven refresh is simpler and matches
  how operator dashboards already poll status. The 24 h window matches
  the default token lifetime.
- **Affects:** `Engine.provision_session` and `reprovision_session`
  implementations; resume logic in `run_loop`.

### D10. Conversation snapshot on session start

- **Decision:** `start_session` (and `simulate_session`) snapshot the
  Conversation into a `Session.conversation_snapshot` field at first
  invocation. Subsequent edits to the underlying Conversation do not affect
  in-flight or completed Sessions.
- **Why:** prevents mid-call brief drift if the operator edits goals while
  a session is live. Resolves the resumability fine-print from SCOPE Open
  Question 2.
- **Affects:** `Session` gets a `conversation_snapshot: Conversation` field;
  resume/simulate logic reads from the snapshot, not from
  `store.load_conversation` (which reflects the current edited version).

### D11. Per-turn LLM telemetry on `turn_recorded`

- **Decision:** the `turn_recorded` event payload includes
  `{tokens_in, tokens_out, cache_read_tokens, cache_write_tokens,
  llm_latency_ms}` for the agent's compose call. Eval-call usage is
  aggregated separately on the `completed` event.
- **Why:** voice-agent operators routinely need cost/latency telemetry to
  tune prompts and pick models. Plumbing it through events now is cheap
  and avoids a second observability sweep later.
- **Affects:** `SessionEvent.payload` schema for `turn_recorded`; the
  Anthropic client surfaces usage on each call.

### D12. Latency targets revised

- **Decision:** target end-of-respondent-utterance to start-of-agent-audio
  is 1.0–1.5 s typical, 2.5–3.5 s worst-case. Components budgeted as:
  Deepgram endpoint detection ~300 ms, evaluate (Haiku) ~200–400 ms,
  compose first-token (Sonnet) ~150–300 ms, Cartesia first audio ~150–300 ms.
- **Why:** the original 1.5 s typical target assumed a one-call merged
  pipeline that we no longer use. Two-call honesty plus Cartesia/Deepgram
  realistic numbers gives the revised band.
- **Affects:** SCOPE.md latency strategy section; no implementation
  changes, just documentation accuracy.

### D13. Loop-time goal-status hint table — in-memory, runner-owned

- **Decision:** the runner maintains an in-memory `dict[str, GoalStatus]`
  (the "goal_status_table") updated on every `EvalResult`. This is the
  loop-time hint authority used by `select_next_goal` and by Step 11 to
  diff against the canonical Extract. It is NOT persisted as canonical;
  durable hints live only on `Turn.addressed_goal_ids`.
- **Why:** the original plan said "emit goal_status_changed for goals
  that differ from loop-time hints" without specifying where those hints
  live. Without an explicit table, the runner has no clean way to
  compute the diff. Making it an explicit runner field makes the diff a
  three-line computation in Step 11.
- **Affects:** Step 8 runner introduces the field; Step 11 reads it to
  compute the diff; tests assert events fire only for diffed goals.

### D14. Silence handling delegated to AgentSession

- **Decision:** v1 does not implement custom silence-nudge / 15s-timeout
  logic. `livekit-agents` `AgentSession` includes VAD-based turn
  detection plus user-away handling; prolonged silence surfaces as a
  framework-level disconnect, which the engine treats the same as a
  respondent disconnect.
- **Why:** writing custom silence detection inside our `chat()`
  implementation fights the framework's turn-detection, doubles
  bookkeeping, and adds an edge case we can't easily test. AgentSession's
  defaults are acceptable for v1; revisit if user feedback shows the
  defaults are too eager or too patient.
- **Affects:** SCOPE.md Unhappy paths "Silence" entry rewritten;
  Step 9's deliverables don't include a custom silence path.

### D15. Anthropic tool input_schema derivation helper

- **Decision:** Anthropic tool input_schema is generated from Pydantic
  models via a small helper that strips Pydantic-specific keys
  (`title`, `$defs`, nested `title`) and inlines `$defs`. Lives in
  `src/interviewer/llm/schemas.py`.
- **Why:** `BaseModel.model_json_schema()` includes keys Anthropic
  rejects (and references `$defs` instead of inlining). Without a
  dedicated helper, the Anthropic client would either fail at runtime
  with cryptic schema errors or open up to "any" tool_choice and lose
  structure guarantees. One helper, tested once, used by both
  `evaluate` and `extract` tools.
- **Affects:** Step 12 adds `llm/schemas.py` and a test for it.

---

## Step 1 — bootstrap project

- **Decision:** `[tool.pytest.ini_options]` includes `pythonpath = ["src"]`.
- **Why:** the project lives under `~/Documents/`, where macOS auto-sets
  the `hidden` file flag on filenames starting with `_`. CPython 3.12's
  `site.py` skips hidden `.pth` files, so the editable-install pth that
  `uv sync` writes (`_editable_impl_interviewer.pth`) gets ignored and
  `import interviewer` fails. Setting pytest's `pythonpath` makes the
  test runner add `src/` directly, bypassing the editable-install path
  for the suite.
- **Affects:** all subsequent test runs work regardless of the macOS
  hidden-flag state; if a future tool (other than pytest / mypy)
  needs the package importable from the venv, run
  `chflags -R nohidden .venv` once after `uv sync`.

---

## Step 5 — engine methods

- **Decision:** `cancel_session` and `reprovision_session` raise
  `ValueError` when the session is in any terminal state (COMPLETED,
  ABANDONED, FAILED); they are NOT idempotent.
- **Why:** PLAN Step 5 lists double-cancel and "reprovision on
  completed session" as error cases. Surfacing them as `ValueError`
  forces consumer apps to model session lifecycle explicitly instead
  of silently no-ooping over stale UI state.
- **Affects:** Step 13's voice-mode cancel path (room-teardown) must
  preserve the same guard; consumer integration in Step 16 docs must
  call out that cancel/reprovision require the session to be
  non-terminal.

## Step 8 — Extract.session_id and completed_at are runner-owned

- **Decision:** the runner overwrites ``Extract.session_id`` and
  ``Extract.completed_at`` via ``model_copy`` after ``LLMClient.derive_extract``
  returns; LLM implementations fill these with a non-empty placeholder
  (FakeLLMClient uses ``conv.id``) so model validation passes.
- **Why:** the ``derive_extract(transcript, conv)`` signature in SCOPE
  has no session handle, but ``Extract.session_id`` is required and
  has ``min_length=1``. Forcing the LLM to learn the session id would
  bloat the prompt and couple model state to session lifecycle;
  letting the runner own the two metadata fields keeps the LLM
  contract narrow.
- **Affects:** Step 12 (AnthropicLLMClient) follows the same pattern.
  Step 11's diff path reads ``extract.goal_statuses`` only — these
  two fields are not part of the canonical-vs-hint comparison.

## Step 8 — Loop ordering: eval-then-select, not select-then-eval

- **Decision:** the runner evaluates the prior respondent turn against
  ``last_active`` first, applies the result to the goal_status_table,
  and only then calls ``select_next_goal`` for the new iteration's
  active goal. The very first main-loop iteration skips eval.
- **Why:** SCOPE's flow lists ``select_next_goal`` before
  ``evaluate_turn`` (3a → 3b), but eval needs to judge the goal that
  was active when the prior respondent answer was given — not the
  newly-selected one. Re-ordering preserves SCOPE's semantic intent
  ("apply eval, then pick next goal from the updated table") without
  the ambiguity of evaluating against a freshly-selected goal that
  was not yet probed.
- **Affects:** Step 9's unhappy paths (retry / drill / refusal) all
  read ``last_active`` for eval context and rely on this ordering.
  Step 13's voice-mode entrypoint mirrors the same pattern inside
  ``InterviewerLLM.chat()``.

## Step 9 — run_loop signals cancel/failure via exceptions

- **Decision:** ``run_loop`` raises ``LoopCancelled`` on operator cancel
  (state observed ABANDONED mid-iteration) and ``LoopFailure`` after
  LLM retry exhaustion (state has been set to FAILED, apology turn
  appended, ``failed`` event emitted). ``simulate_session`` lets both
  propagate.
- **Why:** SCOPE pins ``simulate_session(...) -> Extract``. The cancel
  and LLM-failure paths persist state to ABANDONED / FAILED and do
  not produce an Extract. Returning ``Extract | None`` would expand
  the surface; ignoring the failure mode would lose the signal.
  Raising preserves the type contract for the happy path and gives
  callers a clean ``pytest.raises`` boundary.
- **Affects:** Step 13's voice ``entrypoint`` mirrors the same two
  exception types (or framework-level equivalents). Step 15's
  integration tests assert ABANDONED / FAILED state and the
  appropriate exception per scenario.

## Step 9 — refusal_count is a separate counter from retries_used_on_active

- **Decision:** ``_RunnerState`` adds ``refusal_count_on_active``
  alongside ``retries_used_on_active``. The two diverge: a refusal
  bumps both (via the synthesized retry-action eval and the explicit
  refusal counter), but a non-refusal partial answer that the LLM
  judges ``retry`` only bumps ``retries_used_on_active``. The refusal
  counter is reset by (a) a goal change or (b) a non-refusal
  respondent turn.
- **Why:** SCOPE's give-up rule is "two consecutive refusals/IDK on
  the same goal," not "two retries used." A single counter conflates
  the two and would either give up too early (on legit retries) or
  too late (mixing in refusals across non-consecutive turns).
- **Affects:** the ``goal_status_table`` entries record
  ``retries_used`` (which include the deflection probe);
  ``refusal_count_on_active`` is runner-internal and never persisted.

## Step 10 — Resume rehydrates the hint table from `Turn.addressed_goal_ids`

- **Decision:** on resume, the runner reconstructs the goal_status_table
  by marking every goal_id appearing in any persisted
  `Turn.addressed_goal_ids` as `status="meets"`. The runtime state
  itself only persists counters (active goal id, retries, tangent
  budget, total turns), not the table.
- **Why:** `SessionRuntimeState` is small by design and the
  goal_status_table is explicitly runner-owned and not persisted
  (D13). Without a hint, `select_next_goal` would re-probe goals
  already covered before the crash. Using `addressed_goal_ids` is
  intentionally lossy — a goal that was being retried (not yet
  meeting its standard) is treated as meets on resume — but the
  canonical statuses come from `derive_extract` at completion
  (D5), so the lossiness is invisible in the final Extract.
- **Affects:** Step 11's diff computation: a resumed session may
  show more `goal_status_changed` events because the hint table
  was reconstructed without nuance. Acceptable since it surfaces
  the right canonical truth. Step 13's voice entrypoint mirrors
  the same rehydration.

## Step 11 — goal_status_changed emission ordered by `Conversation.goals`

- **Decision:** the runner iterates `conv.goals` (not the canonical
  Extract's list, not the runner's hint dict) when computing the
  diff and emitting `goal_status_changed` events.
- **Why:** the canonical `Extract.goal_statuses` is a list whose
  internal ordering is decided by the LLM (or the FakeLLMClient
  loop), and `dict` insertion order would mirror that — making
  event order LLM-dependent. Using `conv.goals` order makes diff
  emission deterministic, which matters for test assertions and
  for consumers wiring sequencing logic onto the event stream.
- **Affects:** Step 12's AnthropicLLMClient: order of returned
  `goal_statuses` is not load-bearing for the runner. Tests in
  Step 15 can rely on event-order assertions against this scheme.

## Step 11 — `eval_usage_totals` reserved in `completed` payload now, populated in Step 12

- **Decision:** the `completed` event payload includes an
  `eval_usage_totals` dict whose keys match the `turn_recorded`
  usage shape (D11). In Step 11 (FakeLLMClient only), every value
  is zero. Step 12 (AnthropicLLMClient) plumbs real per-call usage
  through the runner and writes accumulated totals here.
- **Why:** consumers subscribing to `completed` get a stable
  payload shape from day one; Step 12's only change is populating
  values, not adding a new field. The contract is fixed at the
  earliest step that ships the event.
- **Affects:** Step 12 adds a `_RunnerState.eval_usage_totals`
  accumulator and writes to it after every successful
  `evaluate_turn`; the `completed` emit site reads it instead of
  the zero constant.

## Step 10 — `_record_agent_turn` owns runtime-state flush

- **Decision:** the runtime-state flush for D9 lives inside
  `_record_agent_turn`, parameterised by an `active_goal_id` keyword.
  Every agent utterance (opening, probe, retry, deflection,
  RESUME_ACK, closing, cancel-closing, apology) flushes uniformly.
  The explicit pre-compose flush in the main loop body is removed.
- **Why:** PLAN Step 10 says "before each agent utterance ...
  save_runtime_state". Centralising the flush at the single
  recording site is fewer save points to keep in sync than a
  bespoke save before each compose call, and it guarantees the
  spec literally regardless of which utterance path the runner is
  on.
- **Affects:** Step 13's voice path uses the same recording helper
  (or mirrors the flush rule inside `InterviewerLLM.chat()`).
  Step 14's SQLite store gets the same flush cadence — one save per
  agent turn rather than two.

## Step 5 — goals_resolved best-effort

- **Decision:** mid-session `SessionStatus.goals_resolved` counts
  distinct goal ids appearing in any `Turn.addressed_goal_ids` until
  an Extract exists; once `Extract` is saved, the count comes from
  non-pending `GoalStatus` entries.
- **Why:** SCOPE defines `goals_resolved` as "count of goals not in
  {pending}" but no canonical per-goal status exists mid-session
  (D5, D13). The live-hint count from turns is the best available
  proxy and is the same data the loop will populate.
- **Affects:** Step 11 (which finalizes per-goal statuses) does not
  need to retrofit dashboard math — the same accessor flips to the
  canonical Extract count automatically once `save_extract` is called.
