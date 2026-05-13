"""Agent loop body — happy + unhappy paths, with crash recovery.

Crash recovery rests on three pieces:

- **Runtime-state flush**: a ``SessionRuntimeState`` is written to the
  store BEFORE every agent utterance — opening, probe, retry,
  deflection, RESUME_ACK, closing, cancel-closing, apology. Implemented
  inside :func:`_record_agent_turn` so every code path picks it up
  uniformly.
- **Resume on re-entry**: if ``run_loop`` finds a stored
  ``SessionRuntimeState`` on entry, it rehydrates counters, marks any
  goal addressed in the existing transcript as ``meets`` (loop-time
  hint; ``derive_extract`` gives the canonical truth), skips the
  opening, and speaks :data:`RESUME_ACK` as the first agent utterance.
  ``RESUME_ACK`` bypasses :func:`validate_voice_phrasing` — the wording
  is fixed.
- **Idempotency**: calling ``run_loop`` on a session whose
  ``Session.state`` is ``COMPLETED`` is a no-op that returns the stored
  Extract. No new turns, no new events.

At completion the runner snapshots the loop-time ``goal_status_table``,
routes the transcript through :func:`derive_extract_with_llm`, diffs
the canonical statuses against the snapshot, and emits one
``goal_status_changed`` event per differing goal — strictly before the
``completed`` event. The ``completed`` payload carries the final
canonical goal_statuses table plus an ``eval_usage_totals`` dict.

Loop ordering: this runner evaluates the prior respondent turn against
the *previously-active* goal first, then re-selects from the updated
table. Re-selecting before evaluation would discard the eval result
that just arrived.
"""

from __future__ import annotations

import asyncio
from collections.abc import AsyncIterator
from datetime import UTC, datetime
from typing import TYPE_CHECKING, Any

from interview_kit.loop.extract import derive_extract_with_llm
from interview_kit.loop.heuristics import detect_idk, detect_refusal
from interview_kit.loop.openings import DEFAULT_OPENING
from interview_kit.loop.phrasing import validate_voice_phrasing
from interview_kit.loop.resume import RESUME_ACK
from interview_kit.loop.selection import select_next_goal
from interview_kit.protocols import LLMClient, RespondentSimulator
from interview_kit.types.config import Conversation, Goal
from interview_kit.types.events import SessionEvent, SessionEventType
from interview_kit.types.runtime import (
    EvalResult,
    Extract,
    GoalStatus,
    SessionRuntimeState,
    Turn,
    TurnContext,
)
from interview_kit.types.state import SessionState

if TYPE_CHECKING:
    from interview_kit.engine import Engine


DEFAULT_CLOSING = "Thanks for your time — that's everything I needed."
CANCEL_CLOSING = "Looks like we need to wrap. Thanks for the time you gave."
APOLOGY = "I'm sorry — something on my end isn't working. Let's pause for now."

_LLM_MAX_ATTEMPTS = 3
_LLM_BACKOFF_BASE_SECONDS = 0.05

# Telemetry shape. Same keys used on ``turn_recorded`` (per-compose) and
# on ``completed`` (aggregated across all eval calls). Concrete LLM
# clients expose ``last_eval_usage`` / ``last_compose_usage`` side
# channels with these fields; clients that don't (FakeLLMClient) yield
# zeros.
_USAGE_KEYS: tuple[str, ...] = (
    "tokens_in",
    "tokens_out",
    "cache_read_tokens",
    "cache_write_tokens",
    "llm_latency_ms",
)
_ZERO_USAGE: dict[str, int] = dict.fromkeys(_USAGE_KEYS, 0)


def _read_usage(llm: LLMClient, attr: str) -> dict[str, int]:
    """Read ``llm.<attr>`` if exposed; else a zeroed Usage dict.

    The :class:`LLMClient` protocol does not declare usage attributes
    (they're Anthropic-specific). We duck-type the integer fields off
    whatever object is present.
    """
    usage = getattr(llm, attr, None)
    if usage is None:
        return dict(_ZERO_USAGE)
    return {
        "tokens_in": int(getattr(usage, "tokens_in", 0) or 0),
        "tokens_out": int(getattr(usage, "tokens_out", 0) or 0),
        "cache_read_tokens": int(getattr(usage, "cache_read_tokens", 0) or 0),
        "cache_write_tokens": int(getattr(usage, "cache_write_tokens", 0) or 0),
        "llm_latency_ms": int(getattr(usage, "llm_latency_ms", 0) or 0),
    }


def _accumulate_eval_usage(state: _RunnerState, llm: LLMClient) -> None:
    """Add ``llm.last_eval_usage`` (if any) into ``state.eval_usage_totals``."""
    delta = _read_usage(llm, "last_eval_usage")
    for key in _USAGE_KEYS:
        state.eval_usage_totals[key] += delta[key]


class LoopCancelled(RuntimeError):
    """Raised when ``Session.state`` flips to ABANDONED mid-flight."""


class LoopFailure(RuntimeError):
    """Raised after LLM retry exhaustion. Session state is set to FAILED."""


async def run_loop(
    engine: Engine, session_id: str, simulator: RespondentSimulator
) -> Extract:
    """Drive the loop against ``simulator`` until terminal condition; return Extract.

    On entry, an existing ``Session.state == COMPLETED`` short-circuits
    to the cached Extract (idempotent). A stored
    ``SessionRuntimeState`` triggers the resume path: counters are
    rehydrated from the runtime state and the transcript, the opening
    is skipped, and the first agent utterance is :data:`RESUME_ACK`.

    Raises :class:`LoopCancelled` on operator-initiated cancel
    (``cancel_session``) and :class:`LoopFailure` after LLM retry
    exhaustion.
    """
    session = await engine.store.load_session(session_id)
    conv = session.conversation_snapshot

    # Idempotency: a completed session returns its cached Extract with
    # no further work and no additional events emitted.
    if session.state == SessionState.COMPLETED:
        cached = await engine.store.load_extract(session_id)
        if cached is None:
            raise RuntimeError(
                f"session {session_id!r} is COMPLETED but has no extract"
            )
        return cached

    runtime = await engine.store.load_runtime_state(session_id)
    state = _RunnerState(conv)
    last_active: Goal | None = None
    last_eval: EvalResult | None = None

    if runtime is not None:
        await _resume_bootstrap(engine, session_id, conv, state, runtime, simulator)
    else:
        await _fresh_bootstrap(engine, session_id, conv, state, simulator)

    while state.total_turns < conv.max_total_turns:
        current = await engine.store.load_session(session_id)
        if current.state == SessionState.ABANDONED:
            await _record_agent_turn(
                engine, session_id, conv, state, CANCEL_CLOSING, addressed=[]
            )
            raise LoopCancelled("session abandoned mid-loop")

        deflection = False

        if last_active is not None:
            transcript = await engine.store.list_turns(session_id)
            last_resp = next(
                (t for t in reversed(transcript) if t.speaker == "respondent"),
                None,
            )
            resp_text = last_resp.text if last_resp is not None else ""

            if last_resp is not None and detect_refusal(resp_text):
                # Consent-decline: mark skipped_refused and advance. No
                # deflection probe — pressing on a refused goal would be
                # ignoring a stated boundary.
                state.goal_status_table[last_active.id] = (
                    state.goal_status_table[last_active.id].model_copy(
                        update={
                            "status": "skipped_refused",
                            "rationale": "respondent declined to answer",
                            "retries_used": state.retries_used_on_active,
                        }
                    )
                )
                last_eval = None
                last_active = None
                state.idk_count_on_active = 0
            elif last_resp is not None and detect_idk(resp_text):
                state.idk_count_on_active += 1
                if state.idk_count_on_active >= 2:
                    last_eval = EvalResult(
                        active_goal_status="gave_up",
                        next_action="advance",
                        rationale="two consecutive IDKs on this goal",
                    )
                    _apply_eval(state, last_active, last_eval)
                    last_active = None
                else:
                    last_eval = EvalResult(
                        active_goal_status="partial",
                        next_action="retry",
                        rationale="IDK — sending deflection probe",
                    )
                    _apply_eval(state, last_active, last_eval)
                    deflection = True
            else:
                state.idk_count_on_active = 0
                eval_ctx = _build_ctx(conv, transcript, state, last_active)
                try:
                    last_eval = await _evaluate_with_retry(engine.llm, eval_ctx)
                except _LLMRetriesExhausted as exc:
                    await _persist_llm_failure(engine, session_id, conv, state)
                    raise LoopFailure("evaluate_turn persistent failure") from exc
                _accumulate_eval_usage(state, engine.llm)
                _apply_eval(state, last_active, last_eval)
                last_eval = _maybe_force_clarify(state, last_active, last_eval)
                if last_eval.next_action == "close":
                    break

        active: Goal
        if deflection and last_active is not None:
            active = last_active
        else:
            candidate = select_next_goal(
                conv, list(state.goal_status_table.values())
            )
            if candidate is None:
                break
            active = candidate
            if last_active is None or active.id != last_active.id:
                state.retries_used_on_active = 0
                state.idk_count_on_active = 0
                state.clarify_used_on_active = 0

        transcript = await engine.store.list_turns(session_id)
        compose_ctx = _build_ctx(conv, transcript, state, active)
        compose_eval = last_eval or _placeholder_eval()
        try:
            text = await _compose_with_regen_and_retry(
                engine.llm, compose_ctx, compose_eval
            )
        except _LLMRetriesExhausted as exc:
            await _persist_llm_failure(engine, session_id, conv, state)
            raise LoopFailure("compose_utterance persistent failure") from exc

        await _record_agent_turn(
            engine,
            session_id,
            conv,
            state,
            text,
            addressed=[active.id],
            active_goal_id=active.id,
            with_compose_usage=True,
        )
        await _record_respondent_turn(
            engine, session_id, conv, state, simulator, addressed=[active.id]
        )

        last_active = active

    if conv.closing is not None:
        closing_text = conv.closing
    else:
        closing_transcript = await engine.store.list_turns(session_id)
        closing_ctx = TurnContext(
            conversation=conv,
            transcript=closing_transcript,
            active_goal=None,
            goal_statuses=list(state.goal_status_table.values()),
            retries_used_on_active=state.retries_used_on_active,
            tangent_followups_used=state.tangent_followups_used,
            total_turns=state.total_turns,
            last_phrasing_failure=None,
        )
        closing_text = await _compose_closing_recap_with_fallback(
            engine.llm, closing_ctx
        )
    await _record_agent_turn(
        engine, session_id, conv, state, closing_text, addressed=[]
    )

    # Snapshot the loop-time hint table BEFORE the canonical LLM pass.
    # The diff between snapshot and canonical is what produces the
    # ``goal_status_changed`` events below.
    hint_snapshot: dict[str, GoalStatus] = dict(state.goal_status_table)

    transcript = await engine.store.list_turns(session_id)
    raw_extract = await derive_extract_with_llm(transcript, conv, engine.llm)
    now = _utcnow()
    extract = raw_extract.model_copy(
        update={"session_id": session_id, "completed_at": now}
    )
    await engine.store.save_extract(extract)
    await engine.store.update_session_state(session_id, SessionState.COMPLETED)

    # Emit goal_status_changed exactly once per diffed goal, before
    # the completed event. Iterate in conv.goals order so emission is
    # deterministic across runs regardless of dict insertion order.
    canonical_by_id = {gs.goal_id: gs for gs in extract.goal_statuses}
    for goal in conv.goals:
        canonical_gs = canonical_by_id.get(goal.id)
        if canonical_gs is None:
            continue
        snapshot_gs = hint_snapshot.get(goal.id)
        snapshot_status = snapshot_gs.status if snapshot_gs is not None else None
        if snapshot_status == canonical_gs.status:
            continue
        await _emit(
            engine,
            session_id,
            conv.id,
            "goal_status_changed",
            {
                "goal_id": goal.id,
                "from_status": snapshot_status,
                "to_status": canonical_gs.status,
                "rationale": canonical_gs.rationale,
            },
        )

    await _emit(
        engine,
        session_id,
        conv.id,
        "completed",
        {
            "goal_statuses": [gs.model_dump() for gs in extract.goal_statuses],
            "total_turns": state.total_turns,
            "eval_usage_totals": dict(state.eval_usage_totals),
        },
    )
    return extract


class _RunnerState:
    """Mutable loop-time hint state. Per D13: in-memory only; runner-owned."""

    def __init__(self, conv: Conversation) -> None:
        self.goal_status_table: dict[str, GoalStatus] = {
            g.id: GoalStatus(goal_id=g.id, status="pending") for g in conv.goals
        }
        self.total_turns: int = 0
        self.retries_used_on_active: int = 0
        self.tangent_followups_used: int = 0
        # Tracks consecutive IDKs on the current active goal. Reset when
        # (a) the active goal changes, or (b) a non-IDK respondent turn
        # breaks the streak. Refusals are a one-shot path (no counter).
        self.idk_count_on_active: int = 0
        # Per-goal cap on the hedge-driven clarify override (Step 25). Reset
        # on goal change so each goal gets at most one forced clarify probe.
        self.clarify_used_on_active: int = 0
        # D11: aggregated across every successful ``evaluate_turn`` call.
        # Emitted on the ``completed`` event payload.
        self.eval_usage_totals: dict[str, int] = dict(_ZERO_USAGE)


async def _fresh_bootstrap(
    engine: Engine,
    session_id: str,
    conv: Conversation,
    state: _RunnerState,
    simulator: RespondentSimulator,
) -> None:
    """Initial-run bootstrap: IN_PROGRESS, respondent_joined, optional opening."""
    await engine.store.update_session_state(session_id, SessionState.IN_PROGRESS)
    await _emit(
        engine,
        session_id,
        conv.id,
        "respondent_joined",
        {"simulator": simulator.persona_name()},
    )
    opening_text = conv.opening if conv.opening is not None else DEFAULT_OPENING
    await _record_agent_turn(
        engine, session_id, conv, state, opening_text, addressed=[]
    )
    await _record_respondent_turn(
        engine, session_id, conv, state, simulator, addressed=[]
    )


async def _resume_bootstrap(
    engine: Engine,
    session_id: str,
    conv: Conversation,
    state: _RunnerState,
    runtime: SessionRuntimeState,
    simulator: RespondentSimulator,
) -> None:
    """Crash-recovery bootstrap: rehydrate counters, speak RESUME_ACK, continue.

    Goals already touched in the persisted transcript are marked
    ``meets`` on the goal_status_table as a loop-time hint — this is
    intentionally lossy (a goal that was being retried looks like a
    completed one), but the canonical statuses come from
    :func:`derive_extract` at the end. The runner just needs a hint
    table that lets :func:`select_next_goal` advance past covered
    goals.
    """
    existing_turns = await engine.store.list_turns(session_id)
    state.total_turns = len(existing_turns)
    state.retries_used_on_active = runtime.retries_used_on_active
    state.tangent_followups_used = runtime.tangent_followups_used

    addressed_so_far: set[str] = set()
    for turn in existing_turns:
        addressed_so_far.update(turn.addressed_goal_ids)
    for goal_id in addressed_so_far:
        if goal_id in state.goal_status_table:
            current = state.goal_status_table[goal_id]
            state.goal_status_table[goal_id] = current.model_copy(
                update={
                    "status": "meets",
                    "rationale": "resumed: prior coverage in transcript",
                }
            )

    # Resume from ABANDONED is permitted; flip the flag so the
    # cancel-observation check in the main loop doesn't immediately fire
    # on a session that was previously abandoned and is now being
    # intentionally resumed.
    await engine.store.update_session_state(session_id, SessionState.IN_PROGRESS)

    # RESUME_ACK is a fixed known-good utterance; skip phrasing
    # validation deliberately.
    await _record_agent_turn(
        engine, session_id, conv, state, RESUME_ACK, addressed=[]
    )
    await _record_respondent_turn(
        engine, session_id, conv, state, simulator, addressed=[]
    )


def _build_ctx(
    conv: Conversation,
    transcript: list[Turn],
    state: _RunnerState,
    active: Goal,
) -> TurnContext:
    return TurnContext(
        conversation=conv,
        transcript=transcript,
        active_goal=active,
        goal_statuses=list(state.goal_status_table.values()),
        retries_used_on_active=state.retries_used_on_active,
        tangent_followups_used=state.tangent_followups_used,
        total_turns=state.total_turns,
        last_phrasing_failure=None,
    )


def _apply_eval(
    state: _RunnerState, active: Goal, eval_result: EvalResult
) -> None:
    current = state.goal_status_table[active.id]
    state.goal_status_table[active.id] = current.model_copy(
        update={
            "status": eval_result.active_goal_status,
            "rationale": eval_result.rationale,
            "retries_used": state.retries_used_on_active,
        }
    )
    for gid in eval_result.redundant_goal_ids:
        if gid in state.goal_status_table:
            gs = state.goal_status_table[gid]
            state.goal_status_table[gid] = gs.model_copy(
                update={
                    "status": "skipped_redundant",
                    "rationale": "redundant per evaluate_turn",
                }
            )
    if eval_result.next_action == "retry":
        state.retries_used_on_active += 1
    elif eval_result.next_action == "probe":
        state.tangent_followups_used += 1
    # advance / close: no in-place counter change. retries reset on goal change.


def _maybe_force_clarify(
    state: _RunnerState, active: Goal, eval_result: EvalResult
) -> EvalResult:
    """Override ``next_action`` to a clarify probe when the answer was hedged.

    Fires at most once per active goal (cap via
    ``state.clarify_used_on_active``). Skipped when the eval already
    resolved the goal (``meets`` / ``gave_up``) or already routed to
    close. Bumps ``tangent_followups_used`` to mirror the natural
    ``next_action == "probe"`` path in :func:`_apply_eval`.
    """
    if eval_result.clarity not in ("hedged", "vague"):
        return eval_result
    if state.clarify_used_on_active >= 1:
        return eval_result
    current_status = state.goal_status_table[active.id].status
    if current_status in ("meets", "gave_up"):
        return eval_result
    if eval_result.next_action == "close":
        return eval_result
    state.clarify_used_on_active += 1
    # Mirror _apply_eval's probe accounting only when the original
    # action wasn't already a probe (which already bumped the counter).
    if eval_result.next_action != "probe":
        state.tangent_followups_used += 1
    return eval_result.model_copy(
        update={"next_action": "probe", "probe_kind": "clarify"}
    )


class _LLMRetriesExhausted(Exception):
    """Internal: signals that the retry budget for an LLM call is spent."""


async def _evaluate_with_retry(llm: LLMClient, ctx: TurnContext) -> EvalResult:
    """Retry ``evaluate_turn`` up to ``_LLM_MAX_ATTEMPTS`` with exponential backoff."""
    last_exc: BaseException | None = None
    for attempt in range(_LLM_MAX_ATTEMPTS):
        try:
            return await llm.evaluate_turn(ctx)
        # LLM client implementations may raise arbitrary exception types
        # (anthropic.APIError, pydantic.ValidationError, …). Catching
        # Exception is the right scope: BaseException-only types
        # (CancelledError, KeyboardInterrupt) still propagate.
        except Exception as exc:
            last_exc = exc
            if attempt < _LLM_MAX_ATTEMPTS - 1:
                await asyncio.sleep(_LLM_BACKOFF_BASE_SECONDS * (2**attempt))
                continue
    raise _LLMRetriesExhausted("evaluate_turn") from last_exc


async def _compose_with_retry(
    llm: LLMClient, ctx: TurnContext, eval_result: EvalResult
) -> str:
    """Retry ``compose_utterance`` up to ``_LLM_MAX_ATTEMPTS`` with backoff."""
    last_exc: BaseException | None = None
    for attempt in range(_LLM_MAX_ATTEMPTS):
        try:
            return await _accumulate(llm.compose_utterance(ctx, eval_result))
        except Exception as exc:
            last_exc = exc
            if attempt < _LLM_MAX_ATTEMPTS - 1:
                await asyncio.sleep(_LLM_BACKOFF_BASE_SECONDS * (2**attempt))
                continue
    raise _LLMRetriesExhausted("compose_utterance") from last_exc


async def _compose_with_regen_and_retry(
    llm: LLMClient, ctx: TurnContext, eval_result: EvalResult
) -> str:
    """Compose with retry; regen once on phrasing failure."""
    text = await _compose_with_retry(llm, ctx, eval_result)
    failures = validate_voice_phrasing(text)
    if failures:
        regen_ctx = ctx.model_copy(
            update={
                "last_phrasing_failure": ",".join(f.value for f in failures),
            }
        )
        text = await _compose_with_retry(llm, regen_ctx, eval_result)
        # Speak verbatim if the second attempt also fails — no third.
    return text


async def _recap_with_retry(llm: LLMClient, ctx: TurnContext) -> str:
    """Retry ``compose_closing_recap`` up to ``_LLM_MAX_ATTEMPTS`` with backoff."""
    last_exc: BaseException | None = None
    for attempt in range(_LLM_MAX_ATTEMPTS):
        try:
            return await llm.compose_closing_recap(ctx)
        except Exception as exc:
            last_exc = exc
            if attempt < _LLM_MAX_ATTEMPTS - 1:
                await asyncio.sleep(_LLM_BACKOFF_BASE_SECONDS * (2**attempt))
                continue
    raise _LLMRetriesExhausted("compose_closing_recap") from last_exc


async def _compose_closing_recap_with_fallback(
    llm: LLMClient, ctx: TurnContext
) -> str:
    """Compose closing recap with retry; fall back to DEFAULT_CLOSING on failure.

    Tries the recap up to twice — once, and once more if the first
    result fails phrasing. No regen prompting (no `last_phrasing_failure`
    is threaded back). If both attempts fail phrasing, or the LLM
    persistently errors, return :data:`DEFAULT_CLOSING` so the session
    still ends cleanly.
    """
    for _attempt in range(2):
        try:
            text = await _recap_with_retry(llm, ctx)
        except _LLMRetriesExhausted:
            return DEFAULT_CLOSING
        if not validate_voice_phrasing(text):
            return text
    return DEFAULT_CLOSING


async def _accumulate(stream: AsyncIterator[str]) -> str:
    chunks: list[str] = []
    async for chunk in stream:
        chunks.append(chunk)
    return "".join(chunks)


def _placeholder_eval() -> EvalResult:
    return EvalResult(
        active_goal_status="pending",
        redundant_goal_ids=[],
        interesting_tangent=None,
        next_action="advance",
        rationale="",
    )


async def _persist_llm_failure(
    engine: Engine, session_id: str, conv: Conversation, state: _RunnerState
) -> None:
    """Record apology turn, set state FAILED, emit ``failed`` event."""
    await _record_agent_turn(
        engine, session_id, conv, state, APOLOGY, addressed=[]
    )
    await engine.store.update_session_state(session_id, SessionState.FAILED)
    await _emit(
        engine,
        session_id,
        conv.id,
        "failed",
        {"reason": "llm_persistent_failure"},
    )


async def _record_agent_turn(
    engine: Engine,
    session_id: str,
    conv: Conversation,
    state: _RunnerState,
    text: str,
    *,
    addressed: list[str],
    active_goal_id: str | None = None,
    with_compose_usage: bool = False,
) -> None:
    # Flush SessionRuntimeState BEFORE every agent utterance so a crash
    # between save and speak leaves the store in a state that resumes
    # cleanly. ``active_goal_id`` is the goal this utterance is
    # probing (None for opening / closing / RESUME_ACK / APOLOGY /
    # CANCEL_CLOSING).
    await engine.store.save_runtime_state(
        SessionRuntimeState(
            session_id=session_id,
            active_goal_id=active_goal_id,
            retries_used_on_active=state.retries_used_on_active,
            tangent_followups_used=state.tangent_followups_used,
            total_turns=state.total_turns,
            pending_follow_up=None,
            last_event_index=max(state.total_turns - 1, 0),
            updated_at=_utcnow(),
        )
    )
    turn = Turn(
        index=state.total_turns,
        speaker="agent",
        text=text,
        timestamp=_utcnow(),
        addressed_goal_ids=list(addressed),
    )
    await engine.store.append_turn(session_id, turn)
    state.total_turns += 1
    # D11: only probe utterances were composed by the LLM; opening,
    # closing, RESUME_ACK, APOLOGY, CANCEL_CLOSING are scripted strings
    # and emit zeros. ``with_compose_usage`` is set by the probe call
    # site after a successful ``compose_utterance``.
    usage = (
        _read_usage(engine.llm, "last_compose_usage")
        if with_compose_usage
        else dict(_ZERO_USAGE)
    )
    await _emit(
        engine,
        session_id,
        conv.id,
        "turn_recorded",
        {
            "index": turn.index,
            "speaker": "agent",
            "text": text,
            **usage,
        },
    )


async def _record_respondent_turn(
    engine: Engine,
    session_id: str,
    conv: Conversation,
    state: _RunnerState,
    simulator: RespondentSimulator,
    *,
    addressed: list[str],
) -> None:
    history = await engine.store.list_turns(session_id)
    last_agent = next(t for t in reversed(history) if t.speaker == "agent")
    text = await simulator.respond(last_agent.text, history)
    turn = Turn(
        index=state.total_turns,
        speaker="respondent",
        text=text,
        timestamp=_utcnow(),
        addressed_goal_ids=list(addressed),
    )
    await engine.store.append_turn(session_id, turn)
    state.total_turns += 1
    await _emit(
        engine,
        session_id,
        conv.id,
        "turn_recorded",
        {"index": turn.index, "speaker": "respondent", "text": text},
    )


async def _emit(
    engine: Engine,
    session_id: str,
    conversation_id: str,
    type_: SessionEventType,
    payload: dict[str, Any],
) -> None:
    await engine.events.emit(
        SessionEvent(
            session_id=session_id,
            conversation_id=conversation_id,
            timestamp=_utcnow(),
            type=type_,
            payload=payload,
        )
    )


def _utcnow() -> datetime:
    return datetime.now(UTC)
