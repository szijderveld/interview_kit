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

## Step 12 — Pinned `anthropic>=0.40,<1.0`; installed 0.100.0

- **Decision:** dependency pin is `anthropic>=0.40,<1.0`. The SDK
  currently published on PyPI is 0.100.0 and is what gets installed.
- **Why:** the SDK is still pre-1.0 but stable enough; the pin avoids
  a future 1.x with breaking changes auto-installing. PLAN suggested
  this exact pin; 0.100.0 is verified compatible with the calls used
  here (`messages.create`, `messages.stream`, tool-use blocks, the
  `cache_control` system block).
- **Affects:** Step 13's `livekit-agents` integration shares this
  Anthropic SDK pin and will need to be re-verified if either SDK
  pins lifecycle. Step 17's API stability sweep should re-verify the
  installed `anthropic` version still matches the runtime calls.

## Step 12 — `derive_extract` tool uses a `_ExtractToolInput` partial, not full `Extract`

- **Decision:** the Anthropic `extract` tool's `input_schema` is
  derived from a private `_ExtractToolInput` model containing only
  `goal_statuses` + `unprompted_findings`. The wrapper reconstructs
  the full `Extract` (filling `session_id` / `conversation_id` /
  `full_transcript` / `completed_at` itself).
- **Why:** PLAN suggested `anthropic_tool_schema(Extract)`, but the
  full Extract requires `session_id` (runner-owned, D8 Step 8),
  `completed_at` (runner-owned), and `full_transcript` (we just sent
  it — round-tripping wastes tokens and risks transcription
  drift). The narrower schema is cheaper and removes a class of LLM
  errors.
- **Affects:** Step 15's integration tests can assume
  `Extract.full_transcript` always matches the transcript that was
  passed in. Step 13's voice entrypoint can reuse the same client
  unchanged.

## Step 12 — `AnthropicLLMClient` accepts an injected `client` kwarg

- **Decision:** the constructor accepts either `api_key` (constructs
  `anthropic.AsyncAnthropic` internally) or `client` (an existing
  SDK client / test fake). At least one is required.
- **Why:** PLAN's signature is `api_key: str` (required). Adding the
  `client` kwarg makes the unit tests (which fake the SDK at the
  client surface) clean — no monkeypatching of module-level imports,
  no `_client` attribute reach-through.
- **Affects:** Step 13 may use the same pattern when wiring
  `InterviewerLLM` (a livekit-agents subclass) — it can pass an
  existing AsyncAnthropic instance so the SDK is constructed once.
  Step 17's API audit should treat both kwargs as part of the
  public surface.

## Step 12 — runner uses `getattr` to read usage off `LLMClient`

- **Decision:** the runner reads `last_compose_usage` / `last_eval_usage`
  via `getattr(llm, attr, None)`; the `LLMClient` protocol does NOT
  declare these attributes. Clients that don't expose usage
  (FakeLLMClient) yield zero dicts.
- **Why:** `Usage` is an Anthropic-specific concept (D11). Adding
  it to the protocol would force every implementer (including the
  in-memory FakeLLMClient and any future client) to surface it,
  which is scope creep. Duck-typing keeps the protocol minimal.
- **Affects:** Step 14's SQLite store is unaffected. Step 13's
  voice entrypoint uses the same usage-read helpers when emitting
  `turn_recorded`. If a second LLM client is ever added, it should
  expose `Usage`-shaped attributes (any object with int fields
  `tokens_in`, `tokens_out`, `cache_read_tokens`,
  `cache_write_tokens`, `llm_latency_ms`) to participate in
  telemetry.

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

## Step 13 — voice path accumulates compose output before yielding

- **Decision:** in voice mode the chat() body fully accumulates
  `compose_utterance` before pushing a `ChatChunk`. PLAN's pseudocode
  streamed each chunk as it arrived (PLAN step 13 pseudocode lines
  8-9), with phrasing regen as a second pass. We diverge: validate
  first, regen-once if needed (D7), and only then push the final text
  as one or more chunks.
- **Why:** streaming-then-regen means the respondent hears the failed
  candidate before the regen plays, doubling the audio for a one-in-N
  edge case. Accepting the TTFT cost (~150-300 ms loss vs. true
  streaming) preserves the validator's purpose; the simulator runner
  still streams because no audio is involved.
- **Affects:** Step 17 should consider whether to re-enable streaming
  + an audio-truncation primitive once the validator's false-positive
  rate is measured in production.

## Step 13 — `livekit-agents` is an optional `voice` extra

- **Decision:** `pyproject.toml` defines `[project.optional-dependencies] voice`
  pulling `livekit-agents[deepgram,cartesia,silero]>=1,<2` and
  `livekit-api>=1,<2`. The same deps are also in the dev dependency-group
  so test/mypy/ruff runs see them. Engine guards LiveKit-only code paths
  with local imports so simulator-only consumers never hit a missing-module
  error.
- **Why:** the voice install pulls ~30 transitive deps (PyTorch optional
  plugins, av, opentelemetry, etc.). PLAN explicitly groups them under a
  `voice` extra. Local imports in engine.py keep the
  simulator-only call paths cold-fast.
- **Affects:** consumers installing simulator-only never need the audio
  stack. Step 14's SQLite store and Step 15's integration tests remain
  voice-free. Step 17 should keep the runtime `dependencies` list slim.

## Step 13 — pinned `livekit-agents>=1,<2`, `livekit-api>=1,<2`; installed 1.5.8 / 1.1.0

- **Decision:** dependency pins are open-ended within the 1.x major.
  PLAN suggested `>=1.0` for livekit-agents and `>=0.6` for livekit-api;
  the latter has already shipped a 1.0, so 1.x is the right cap on both.
- **Why:** both packages are first-party LiveKit and follow loose semver;
  pinning to the current 1.x major is enough to avoid surprise 2.x
  breakage. Plugin extras (`deepgram`, `cartesia`, `silero`) resolved
  cleanly against 1.5.8 with their corresponding `livekit-plugins-*`
  packages.
- **Affects:** Step 17's API stability sweep should re-verify the
  framework surface (LLM/LLMStream/AgentSession/JobContext, plugin
  imports `cartesia.TTS(model="sonic-2", ...)` / `deepgram.STT(model="nova-3", ...)`)
  against whatever's on PyPI at release time.

## Step 14 — class-based shared suite for ConversationStore round-trip tests

- **Decision:** the round-trip protocol tests live in a non-`Test*`-prefixed
  base class `StoreRoundTripSuite` in `tests/stores/_round_trip.py`. Each
  per-store test module subclasses it and provides a `store`
  `pytest_asyncio.fixture` returning the concrete implementation. Both
  `tests/stores/test_memory_store.py` and `tests/stores/test_sqlite_store.py`
  exist; each gets the full suite via inheritance.
- **Why:** PLAN.md explicitly names both filenames and says "same protocol
  round-trip suite ... run against SQLite." A single parametrized
  fixture in `conftest.py` would have flattened both stores into one
  file and obscured store-specific tests (persistence across reconnect,
  `connect()` idempotence). The class-based pattern keeps the suite
  shared while leaving room for store-specific cases at module level.
- **Affects:** future stores (Postgres, etc.) subclass the same suite
  by providing only a `store` fixture. Step 15's integration tests
  rely on the SQLite store passing the protocol contract verbatim.

## Step 14 — no FK on `runtime_states` / `extracts`, only on `turns`

- **Decision:** the SQLite schema declares
  `FOREIGN KEY (session_id) REFERENCES sessions(id)` on `turns` only.
  `runtime_states` and `extracts` use a bare `session_id TEXT PRIMARY KEY`.
- **Why:** `InMemoryConversationStore` permits saving a runtime state
  or an extract before any matching `Session` row exists (the existing
  protocol tests rely on this — `test_save_and_load_runtime_state` and
  `test_save_and_load_extract` skip the session-create step). Enforcing
  FKs there would break the shared round-trip suite and diverge from
  the protocol's de-facto permissiveness. `turns` already validates
  session existence at the Python layer (raising `KeyError`); declaring
  the FK there is belt-and-suspenders.
- **Affects:** Step 15's integration tests can persist runtime state
  before inserting a Session row. If a future store enforces stricter
  referential integrity, the round-trip suite must be amended uniformly
  rather than per-store.

## Step 14 — `WebhookEventSink` retries every `httpx.HTTPError` uniformly

- **Decision:** the retry loop catches `httpx.HTTPError` (which
  includes `HTTPStatusError` raised by `raise_for_status()` for 4xx
  and 5xx) and retries up to `max_attempts` for every failure class.
  4xx, 5xx, and transport errors are treated identically.
- **Why:** PLAN said "Retries 3× with exponential backoff" without
  carving out client errors. Splitting 4xx (no retry) from 5xx
  (retry) requires a richer policy than the plan asks for and only
  saves a handful of redundant requests; the simpler loop is one
  fewer branch to keep tested.
- **Affects:** consumers needing smarter policy supply their own
  `httpx.AsyncClient` (with a retry transport, custom timeout, etc.)
  via the `client=` kwarg. The sink stays minimal.

## Step 15 — gave_up scenario uses scripted IDK answers, not `TerseEvasiveSimulator`

- **Decision:** the "some gave_up statuses" integration test scripts
  two consecutive IDK replies on one goal via `ScriptedSimulator`; the
  separate `TerseEvasiveSimulator` test only asserts well-formed
  completion. PLAN.md Step 15 listed a single scenario combining both
  ("Same Conversation against `TerseEvasiveSimulator` — expect some
  `gave_up` statuses").
- **Why:** the production `TerseEvasiveSimulator` cycles through five
  responses with the two IDK phrases (`"Couldn't say."`, `"Not sure."`)
  non-adjacent. The runner's refusal heuristic (Step 9) requires *two
  consecutive* IDK replies on the same goal to mark `gave_up`; with the
  cycle's spacing, two consecutive IDKs never naturally occur, so
  vanilla `TerseEvasiveSimulator` cannot produce the path. Splitting the
  scenarios is more honest than mutating production simulator state or
  side-channeling a `force_gave_up` flag into `FakeLLMClient` just for
  one test.
- **Affects:** Step 16's documentation should treat `TerseEvasiveSimulator`
  as "rough but well-formed answers", not "gave_up generator". If a
  future step needs a built-in `RefusalSimulator` it should be added
  explicitly rather than retrofitted onto `TerseEvasiveSimulator`.

## Step 15 — refusal-path canonical extract reports `meets`, not `gave_up`

- **Decision:** the "refusal → gave_up" integration test asserts the
  Step 11 diff path (a `goal_status_changed` event with
  `from_status="gave_up"`) instead of the canonical
  `Extract.goal_statuses` showing `gave_up`.
- **Why:** `FakeLLMClient.derive_extract` reads canonical status from
  `Turn.addressed_goal_ids` only (no hint-table awareness; Step 11). A
  goal that hit the refusal heuristic still has its probe + deflection
  turns tagged with its id, so FakeLLM reports it as `meets`. The
  runner's loop-time hint table holds `gave_up` for that goal, which
  surfaces in the diff event Step 11 emits before `completed`. The
  integration test asserts at the event layer where the gave_up state
  is observable.
- **Affects:** integration tests against the real `AnthropicLLMClient`
  (out of scope per Step 15 constraints) would see `gave_up` in the
  canonical Extract directly; the assertion shape would change.

## Step 13 — `_finalize_extract` lives in `voice/livekit_entry.py`, not on `Engine`

- **Decision:** the canonical-extract pass for the voice path is a
  module-level helper inside ``voice/livekit_entry.py``, mirroring the
  end-of-loop work in ``runner.py``. PLAN's pseudocode put it on Engine
  as ``self._finalize_extract``.
- **Why:** the runner's terminate path already handles its own
  finalization without going through Engine. Keeping the voice
  finalization next to ``InterviewerLLM`` keeps the two paths
  symmetrical and avoids leaking voice-specific helpers onto the
  Engine surface. Engine.entrypoint just calls into the module.
- **Affects:** Step 16's integration docs should point at
  ``voice/livekit_entry.py`` for the voice flow. Step 17's API audit
  should not consider ``_finalize_extract`` part of Engine's public
  surface (it's not).

## Step 17 — Public API audit vs. SCOPE.md

- **Decision:** ``src/interviewer/__init__.py`` re-exports every name
  in SCOPE.md's Public API section: configuration types (``Persona``,
  ``Background``, ``Goal``, ``Conversation``), runtime types
  (``Session``, ``SessionCredentials``, ``SessionRuntimeState``,
  ``Turn``, ``GoalStatus``, ``Finding``, ``Extract``, ``SessionStatus``,
  ``TurnContext``, ``EvalResult``), ``SessionState``, ``SessionEvent``,
  the four consumer protocols (``ConversationStore``, ``EventSink``,
  ``LLMClient``, ``RespondentSimulator``), ``Engine``, and
  ``LiveKitConfig``. SCOPE's "Defaults the package ships" entries
  (``AnthropicLLMClient``, ``InMemoryConversationStore``,
  ``InMemoryEventSink``, ``SQLiteConversationStore``,
  ``LoggingEventSink``, ``WebhookEventSink``, the four reference
  simulators, ``FakeLLMClient``) are intentionally NOT re-exported at
  the top level — consumers import them from their owning submodule
  (``interviewer.llm.anthropic``, ``interviewer.stores.sqlite``, etc.),
  matching the pattern already established in ``examples/`` and
  ``docs/integration.md``.
- **Why:** the SCOPE Public API section enumerates the types and
  protocols the package guarantees stability on; the Defaults section
  describes pluggable reference implementations whose import paths
  already document themselves. Top-level re-exports of the defaults
  would put a longer surface under the package's stability contract
  than SCOPE actually pins; submodule imports keep the contract narrow
  while still giving consumers one obvious place to find each impl.
- **Affects:** v0.1.0 stability promise covers exactly the names in
  ``interviewer.__all__``. Future moves of default impls between
  submodules are NOT breaking changes per this audit; renaming or
  removing anything in ``__all__`` IS breaking.

## Step 17 — Internal-only modules deliberately excluded from re-export

- **Decision:** the following modules are present in the source tree
  but NOT re-exported by ``interviewer/__init__.py``:
  ``interviewer.loop.runner``, ``interviewer.loop.selection``,
  ``interviewer.loop.phrasing``, ``interviewer.loop.extract``,
  ``interviewer.loop.heuristics``, ``interviewer.loop.resume``,
  ``interviewer.llm.prompts``, ``interviewer.llm.schemas``,
  ``interviewer.voice.livekit_entry``. They remain importable for
  advanced consumers and tests but are NOT part of the package's
  stability contract.
- **Why:** PLAN Step 17 says internal modules (``loop/heuristics.py``
  etc.) are not re-exported. These all sit behind ``Engine`` /
  ``simulate_session`` / ``entrypoint``; consumers reach them only
  when extending or testing. Freezing their signatures would make
  routine loop refactors breaking changes.
- **Affects:** post-v0.1.0 changes inside ``loop/``, ``llm/prompts``,
  ``llm/schemas``, and ``voice/livekit_entry`` are NOT breaking.
  Anything a consumer pins onto from these modules is at their own
  risk; integration tests in this repo are allowed to do so.

## Step 18 — distribution renamed to `interview-kit`; import path unchanged

- **Decision:** PyPI distribution name is `interview-kit`, but the
  Python package directory stays `src/interviewer/` and consumers still
  `import interviewer`.
- **Why:** `interviewer` on PyPI is already taken; renaming the import
  path would break every existing consumer for no benefit.
- **Affects:** Step 19+ docs and CLI must use `pip install interview-kit`
  on the install line and `import interviewer` in code. The wheel
  filename pattern is `interview_kit-<version>-*.whl` (underscored).

## Step 18 — added `license-files = ["LICENSE"]` to `[project]`

- **Decision:** Declared the LICENSE file explicitly via
  `project.license-files` rather than relying on hatch auto-detection.
- **Why:** PEP 639 / modern build backends prefer explicit license-file
  declarations; without it some tooling emits metadata warnings and the
  file may not be bundled in the sdist.
- **Affects:** future bumps to the LICENSE filename must update this
  field; do not remove it when reorganizing project metadata.

## Step 19 — `create_conversation` accepts `max_tangent_followups` and `max_total_turns`

- **Decision:** Added optional `max_tangent_followups` and `max_total_turns`
  kwargs (with the same defaults as `Conversation`) to
  `Engine.create_conversation`.
- **Why:** The documented Step-19 quickstart unpacks
  `Conversation.from_yaml(...).model_dump(exclude={"id"})` into
  `create_conversation`. Without these kwargs, any YAML setting non-default
  turn budgets would raise `TypeError`. Pure addition; existing callers
  continue to work unchanged.
- **Affects:** Step 20's CLI `simulate` subcommand can rely on YAML-supplied
  turn budgets flowing through `create_conversation` unchanged.

## Step 20 — `interviewer demo` delegates to `interviewer.examples.simulated`

- **Decision:** The shipped CLI's `demo` subcommand forwards to
  `interviewer.examples.simulated.cli(argv)` rather than reimplementing
  the demo wiring inside `cli.py`. The repo-root `examples/simulated.py`
  is reduced to a thin shim doing the same.
- **Why:** The plan caps `cli.py` at ~150 lines and forbids business
  logic. Keeping the demo construction in one importable module means
  the CLI and the contributor-facing `uv run python examples/...` story
  share exactly the same code path.
- **Affects:** Future demo variants should grow under
  `src/interviewer/examples/`, not under `cli.py`. Tests target
  `cli.main(["demo"])`, which transitively exercises the example.

## Step 20 — `interviewer simulate` synthesizes FakeLLMClient inputs from goals

- **Decision:** Without `--use-anthropic`, the `simulate` subcommand
  builds a `FakeLLMClient` with one auto-generated "meets/advance"
  eval and one templated utterance per goal in the loaded
  `Conversation`, and seeds `ScriptedSimulator` with the contents of
  `--responses` (or a baked-in default list).
- **Why:** A YAML-driven simulation must run with zero API keys for
  the acceptance test. The runner consumes exactly one eval and one
  compose call per goal in the advance path, so the count is
  deterministic from `len(conv.goals)` and avoids exposing internal
  loop arithmetic to CLI users.
- **Affects:** If the loop's per-goal call count ever changes (e.g., an
  extra implicit closing eval), `_autoeval_results` /
  `_autoutter_utterances` in `cli.py` must be revisited. Custom eval
  paths (`partial`, `retry`, `drill`) are not reachable from the CLI
  by design — operators wanting that should write a Python harness.

## Step 19 — `pyyaml` added to core dependencies, not an extra

- **Decision:** `pyyaml>=6,<7` is in `[project].dependencies`, not behind a
  `[yaml]` optional-dependency group. `types-PyYAML` lives in dev only.
- **Why:** `Conversation.from_yaml` is an advertised public helper; gating
  it behind an extra creates a "pip install interview-kit[yaml]" footgun
  for the headline quickstart. pyyaml is small, ubiquitous, and pure
  Python — the cost of adding it to the base install is negligible.
- **Affects:** Step 20's CLI and Step 21's README can assume YAML loading
  is available without an extras hint. Any future YAML schema changes
  remain breaking under the package's stability contract.
