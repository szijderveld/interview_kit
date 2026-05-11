"""Agent loop body — happy + unhappy paths.

Step 8 implemented the happy path; Step 9 adds:

- **Refusal / IDK** (D2): a keyword pre-check on the respondent utterance
  triggers a single deflection probe on first occurrence (consumes a
  retry) and ``gave_up`` + advance on second consecutive occurrence.
- **LLM retries with backoff**: each ``evaluate_turn`` and
  ``compose_utterance`` call retries up to 3× with exponential backoff
  (``_LLM_BACKOFF_BASE_SECONDS × 2**attempt``, no ``tenacity``). On
  exhaustion the runner appends an apology utterance, sets state
  FAILED, emits ``failed``, and raises :class:`LoopFailure`.
- **Turn cap**: the loop exits naturally to closing once
  ``state.total_turns`` reaches ``conversation.max_total_turns``; a
  probe in flight finishes before exit because the cap is checked at
  iteration top.
- **Operator cancel** (D4): at the top of every iteration the runner
  re-reads ``Session.state``. On ``ABANDONED`` it speaks a short
  closing and raises :class:`LoopCancelled`; no ``completed`` event is
  emitted.

Step 10 will persist + resume from ``SessionRuntimeState``; Step 11
implements the canonical-extract diff against ``goal_status_table``.

Loop ordering. SCOPE lists ``select_next_goal`` before ``evaluate_turn``;
this runner evaluates the prior respondent turn against the
*previously-active* goal first, then re-selects from the updated
table. See DECISIONS Step 8.
"""

from __future__ import annotations

import asyncio
from collections.abc import AsyncIterator
from datetime import UTC, datetime
from typing import TYPE_CHECKING, Any

from interviewer.loop.heuristics import detect_refusal_or_idk
from interviewer.loop.phrasing import validate_voice_phrasing
from interviewer.loop.selection import select_next_goal
from interviewer.protocols import LLMClient, RespondentSimulator
from interviewer.types.config import Conversation, Goal
from interviewer.types.events import SessionEvent, SessionEventType
from interviewer.types.runtime import (
    EvalResult,
    Extract,
    GoalStatus,
    SessionRuntimeState,
    Turn,
    TurnContext,
)
from interviewer.types.state import SessionState

if TYPE_CHECKING:
    from interviewer.engine import Engine


DEFAULT_CLOSING = "Thanks for your time — that's everything I needed."
CANCEL_CLOSING = "Looks like we need to wrap. Thanks for the time you gave."
APOLOGY = "I'm sorry — something on my end isn't working. Let's pause for now."

_LLM_MAX_ATTEMPTS = 3
_LLM_BACKOFF_BASE_SECONDS = 0.05


class LoopCancelled(RuntimeError):
    """Raised when ``Session.state`` flips to ABANDONED mid-flight."""


class LoopFailure(RuntimeError):
    """Raised after LLM retry exhaustion. Session state is set to FAILED."""


async def run_loop(
    engine: Engine, session_id: str, simulator: RespondentSimulator
) -> Extract:
    """Drive the loop against ``simulator`` until terminal condition; return Extract.

    Raises :class:`LoopCancelled` on operator-initiated cancel
    (``cancel_session``) and :class:`LoopFailure` after LLM retry
    exhaustion.
    """
    session = await engine.store.load_session(session_id)
    conv = session.conversation_snapshot

    await engine.store.update_session_state(session_id, SessionState.IN_PROGRESS)
    await _emit(
        engine,
        session_id,
        conv.id,
        "respondent_joined",
        {"simulator": simulator.persona_name()},
    )

    state = _RunnerState(conv)

    if conv.opening:
        await _record_agent_turn(
            engine, session_id, conv, state, conv.opening, addressed=[]
        )
        await _record_respondent_turn(
            engine, session_id, conv, state, simulator, addressed=[]
        )

    last_active: Goal | None = None
    last_eval: EvalResult | None = None

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
            refusal = last_resp is not None and detect_refusal_or_idk(last_resp.text)

            if refusal:
                state.refusal_count_on_active += 1
                if state.refusal_count_on_active >= 2:
                    last_eval = EvalResult(
                        active_goal_status="gave_up",
                        next_action="advance",
                        rationale="two consecutive refusals/IDK on this goal",
                    )
                    _apply_eval(state, last_active, last_eval)
                    last_active = None
                else:
                    last_eval = EvalResult(
                        active_goal_status="partial",
                        next_action="retry",
                        rationale="refusal/IDK — sending deflection probe",
                    )
                    _apply_eval(state, last_active, last_eval)
                    deflection = True
            else:
                state.refusal_count_on_active = 0
                eval_ctx = _build_ctx(conv, transcript, state, last_active)
                try:
                    last_eval = await _evaluate_with_retry(engine.llm, eval_ctx)
                except _LLMRetriesExhausted as exc:
                    await _persist_llm_failure(engine, session_id, conv, state)
                    raise LoopFailure("evaluate_turn persistent failure") from exc
                _apply_eval(state, last_active, last_eval)
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
                state.refusal_count_on_active = 0

        await engine.store.save_runtime_state(
            SessionRuntimeState(
                session_id=session_id,
                active_goal_id=active.id,
                retries_used_on_active=state.retries_used_on_active,
                tangent_followups_used=state.tangent_followups_used,
                total_turns=state.total_turns,
                pending_follow_up=None,
                last_event_index=0,
                updated_at=_utcnow(),
            )
        )

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
            engine, session_id, conv, state, text, addressed=[active.id]
        )
        await _record_respondent_turn(
            engine, session_id, conv, state, simulator, addressed=[active.id]
        )

        last_active = active

    closing_text = conv.closing or DEFAULT_CLOSING
    await _record_agent_turn(
        engine, session_id, conv, state, closing_text, addressed=[]
    )

    transcript = await engine.store.list_turns(session_id)
    raw_extract = await engine.llm.derive_extract(transcript, conv)
    now = _utcnow()
    extract = raw_extract.model_copy(
        update={"session_id": session_id, "completed_at": now}
    )
    await engine.store.save_extract(extract)

    await engine.store.update_session_state(session_id, SessionState.COMPLETED)
    await _emit(
        engine,
        session_id,
        conv.id,
        "completed",
        {
            "goal_statuses": [gs.model_dump() for gs in extract.goal_statuses],
            "total_turns": state.total_turns,
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
        # Tracks consecutive refusals/IDK on the current active goal. Reset
        # when (a) the active goal changes, or (b) a non-refusal respondent
        # turn breaks the streak.
        self.refusal_count_on_active: int = 0


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
    elif eval_result.next_action == "drill":
        state.tangent_followups_used += 1
    # advance / close: no in-place counter change. retries reset on goal change.


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
    """Compose with retry; regen once on phrasing failure (D7)."""
    text = await _compose_with_retry(llm, ctx, eval_result)
    failures = validate_voice_phrasing(text)
    if failures:
        regen_ctx = ctx.model_copy(
            update={
                "last_phrasing_failure": ",".join(f.value for f in failures),
            }
        )
        text = await _compose_with_retry(llm, regen_ctx, eval_result)
        # D7: speak verbatim if the second attempt also fails.
    return text


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
) -> None:
    turn = Turn(
        index=state.total_turns,
        speaker="agent",
        text=text,
        timestamp=_utcnow(),
        addressed_goal_ids=list(addressed),
    )
    await engine.store.append_turn(session_id, turn)
    state.total_turns += 1
    # D11 telemetry placeholders. FakeLLMClient does not meter; Step 12
    # plumbs real values from AnthropicLLMClient.
    await _emit(
        engine,
        session_id,
        conv.id,
        "turn_recorded",
        {
            "index": turn.index,
            "speaker": "agent",
            "text": text,
            "tokens_in": 0,
            "tokens_out": 0,
            "cache_read_tokens": 0,
            "cache_write_tokens": 0,
            "llm_latency_ms": 0,
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
