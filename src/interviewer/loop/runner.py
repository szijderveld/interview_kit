"""Agent loop body — happy path only (Step 8).

Unhappy paths (refusal, IDK, LLM API failure, turn cap, cancel) land in
Step 9; runtime-state persistence + resume in Step 10; canonical extract
diff in Step 11.

Loop ordering note. SCOPE's flow lists ``select_next_goal`` before
``evaluate_turn``. In practice the two need to interleave because
``evaluate_turn`` judges the prior respondent turn against the goal that
was active when that turn happened — not against whatever new goal
``select_next_goal`` would have picked. So this runner evaluates the
``last_active`` goal at the top of each iteration, applies the result to
the in-memory ``goal_status_table`` (D13), then calls ``select_next_goal``
against the updated table to pick the goal we'll actually probe.

The ``goal_status_table`` is loop-time hint state per D13: it drives
selection and seeds the Step 11 diff, but is not the canonical
authority. ``derive_extract`` at completion produces the canonical
``GoalStatus.evidence_turn_indices``.
"""

from __future__ import annotations

from collections.abc import AsyncIterator
from datetime import UTC, datetime
from typing import TYPE_CHECKING, Any

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


async def run_loop(
    engine: Engine, session_id: str, simulator: RespondentSimulator
) -> Extract:
    """Drive the loop against ``simulator`` until terminal condition; return Extract."""
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
        # Evaluate the prior respondent turn against last_active (skipped on
        # the first main-loop iteration — nothing to evaluate yet).
        if last_active is not None:
            transcript = await engine.store.list_turns(session_id)
            eval_ctx = _build_ctx(conv, transcript, state, last_active)
            last_eval = await engine.llm.evaluate_turn(eval_ctx)
            _apply_eval(state, last_active, last_eval)
            if last_eval.next_action == "close":
                break

        active = select_next_goal(conv, list(state.goal_status_table.values()))
        if active is None:
            break

        if last_active is None or active.id != last_active.id:
            state.retries_used_on_active = 0

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
        text = await _compose_with_regen(engine.llm, compose_ctx, compose_eval)

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


async def _compose_with_regen(
    llm: LLMClient, ctx: TurnContext, eval_result: EvalResult
) -> str:
    """Compose, validate phrasing, regen once on failure (D7)."""
    text = await _accumulate(llm.compose_utterance(ctx, eval_result))
    failures = validate_voice_phrasing(text)
    if failures:
        regen_ctx = ctx.model_copy(
            update={
                "last_phrasing_failure": ",".join(f.value for f in failures),
            }
        )
        text = await _accumulate(llm.compose_utterance(regen_ctx, eval_result))
        # D7: speak verbatim if the second attempt also fails — no further regen.
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
