from __future__ import annotations

from datetime import UTC, datetime

import pytest
from pydantic import ValidationError

from interviewer import (
    Background,
    Conversation,
    EvalResult,
    Extract,
    Finding,
    Goal,
    GoalStatus,
    Persona,
    Session,
    SessionCredentials,
    SessionEvent,
    SessionRuntimeState,
    SessionState,
    SessionStatus,
    Turn,
    TurnContext,
)


def _conv() -> Conversation:
    return Conversation(
        id="conv-1",
        persona=Persona(
            system_prompt="You are an interviewer.",
            style="neutral",
            voice_id="cartesia-1",
        ),
        purpose="discovery",
        background=Background(
            interviewee_role="role", interviewee_expertise="expertise"
        ),
        goals=[Goal(id="g1", intent="i", standard="s")],
    )


def _ts() -> datetime:
    return datetime(2026, 5, 10, 12, 0, 0, tzinfo=UTC)


# ---------- SessionState ----------------------------------------------------


def test_session_state_membership() -> None:
    expected = {"created", "ready", "in_progress", "completed", "abandoned", "failed"}
    assert {s.value for s in SessionState} == expected


def test_session_state_is_str() -> None:
    assert SessionState.READY == "ready"


# ---------- Session ---------------------------------------------------------


def test_session_state_default_is_created() -> None:
    s = Session(
        id="s1",
        conversation_id="conv-1",
        conversation_snapshot=_conv(),
        created_at=_ts(),
    )
    assert s.state is SessionState.CREATED
    assert s.started_at is None and s.completed_at is None


def test_session_is_frozen() -> None:
    s = Session(
        id="s1",
        conversation_id="conv-1",
        conversation_snapshot=_conv(),
        created_at=_ts(),
    )
    with pytest.raises(ValidationError):
        s.state = SessionState.READY  # type: ignore[misc]


def test_session_conversation_snapshot_round_trip() -> None:
    s = Session(
        id="s1",
        conversation_id="conv-1",
        conversation_snapshot=_conv(),
        created_at=_ts(),
    )
    data = s.model_dump_json()
    restored = Session.model_validate_json(data)
    assert restored == s
    # D10: the snapshot survives serialization with goals intact.
    assert [g.id for g in restored.conversation_snapshot.goals] == ["g1"]


# ---------- SessionCredentials ---------------------------------------------


def test_session_credentials_round_trip() -> None:
    c = SessionCredentials(room_url="wss://lk/x", token="t", expires_at=_ts())
    assert SessionCredentials.model_validate_json(c.model_dump_json()) == c


# ---------- SessionRuntimeState ---------------------------------------------


def test_session_runtime_state_defaults_and_round_trip() -> None:
    rs = SessionRuntimeState(
        session_id="s1",
        active_goal_id="g1",
        retries_used_on_active=0,
        tangent_followups_used=0,
        total_turns=0,
        last_event_index=0,
        updated_at=_ts(),
    )
    assert rs.pending_follow_up is None
    assert SessionRuntimeState.model_validate_json(rs.model_dump_json()) == rs


def test_session_runtime_state_rejects_negative_counters() -> None:
    with pytest.raises(ValidationError):
        SessionRuntimeState(
            session_id="s1",
            active_goal_id=None,
            retries_used_on_active=-1,
            tangent_followups_used=0,
            total_turns=0,
            last_event_index=0,
            updated_at=_ts(),
        )


# ---------- Turn ------------------------------------------------------------


def test_turn_round_trip_and_speaker_literal() -> None:
    t = Turn(
        index=0,
        speaker="agent",
        text="hello",
        timestamp=_ts(),
        addressed_goal_ids=["g1"],
    )
    assert Turn.model_validate_json(t.model_dump_json()) == t


def test_turn_rejects_unknown_speaker() -> None:
    with pytest.raises(ValidationError):
        Turn(
            index=0,
            speaker="other",  # type: ignore[arg-type]
            text="x",
            timestamp=_ts(),
        )


# ---------- GoalStatus ------------------------------------------------------


def test_goal_status_round_trip_and_status_literal() -> None:
    gs = GoalStatus(
        goal_id="g1",
        status="meets",
        evidence_turn_indices=[0, 2],
        retries_used=1,
        rationale="answered fully",
    )
    assert GoalStatus.model_validate_json(gs.model_dump_json()) == gs


def test_goal_status_rejects_unknown_status() -> None:
    with pytest.raises(ValidationError):
        GoalStatus(goal_id="g1", status="bogus")  # type: ignore[arg-type]


# ---------- Finding ---------------------------------------------------------


def test_finding_defaults_category_none() -> None:
    f = Finding(text="they mentioned ERP migration in Q1", evidence_turn_index=3)
    assert f.category is None
    assert Finding.model_validate_json(f.model_dump_json()) == f


# ---------- Extract ---------------------------------------------------------


def test_extract_round_trip() -> None:
    ex = Extract(
        session_id="s1",
        conversation_id="conv-1",
        goal_statuses=[GoalStatus(goal_id="g1", status="meets")],
        unprompted_findings=[
            Finding(text="ERP migration in Q1", evidence_turn_index=3)
        ],
        full_transcript=[
            Turn(index=0, speaker="agent", text="hi", timestamp=_ts()),
        ],
        completed_at=_ts(),
    )
    assert Extract.model_validate_json(ex.model_dump_json()) == ex


# ---------- SessionStatus ---------------------------------------------------


def test_session_status_round_trip() -> None:
    st = SessionStatus(
        session_id="s1",
        state=SessionState.IN_PROGRESS,
        active_goal_id="g1",
        total_turns=4,
        goals_resolved=1,
        goals_total=3,
        started_at=_ts(),
        last_turn_at=_ts(),
    )
    assert SessionStatus.model_validate_json(st.model_dump_json()) == st


# ---------- TurnContext -----------------------------------------------------


def test_turn_context_round_trip() -> None:
    conv = _conv()
    ctx = TurnContext(
        conversation=conv,
        transcript=[],
        active_goal=conv.goals[0],
        goal_statuses=[GoalStatus(goal_id="g1", status="pending")],
        retries_used_on_active=0,
        tangent_followups_used=0,
        total_turns=0,
        last_phrasing_failure=None,
    )
    assert TurnContext.model_validate_json(ctx.model_dump_json()) == ctx


# ---------- EvalResult ------------------------------------------------------


def test_eval_result_round_trip_and_literals() -> None:
    er = EvalResult(
        active_goal_status="partial",
        redundant_goal_ids=["g2"],
        interesting_tangent="erp migration",
        next_action="retry",
        rationale="answer thin on durations",
    )
    assert EvalResult.model_validate_json(er.model_dump_json()) == er


def test_eval_result_rejects_skipped_redundant_status() -> None:
    # active_goal_status does not include skipped_redundant — that's
    # a state assigned to *other* goals via redundant_goal_ids.
    with pytest.raises(ValidationError):
        EvalResult(
            active_goal_status="skipped_redundant",  # type: ignore[arg-type]
            next_action="advance",
        )


def test_eval_result_rejects_unknown_action() -> None:
    with pytest.raises(ValidationError):
        EvalResult(
            active_goal_status="pending",
            next_action="ponder",  # type: ignore[arg-type]
        )


# ---------- SessionEvent ----------------------------------------------------


def test_session_event_round_trip() -> None:
    e = SessionEvent(
        session_id="s1",
        conversation_id="conv-1",
        timestamp=_ts(),
        type="turn_recorded",
        payload={
            "turn_index": 3,
            "tokens_in": 120,
            "tokens_out": 14,
            "cache_read_tokens": 800,
            "cache_write_tokens": 0,
            "llm_latency_ms": 240,
        },
    )
    assert SessionEvent.model_validate_json(e.model_dump_json()) == e


def test_session_event_rejects_unknown_type() -> None:
    with pytest.raises(ValidationError):
        SessionEvent(
            session_id="s1",
            conversation_id="conv-1",
            timestamp=_ts(),
            type="something_else",  # type: ignore[arg-type]
        )
