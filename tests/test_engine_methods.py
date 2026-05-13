from __future__ import annotations

from collections.abc import AsyncIterator
from datetime import UTC, datetime

import pytest

from interview_kit import (
    Background,
    Conversation,
    Engine,
    EvalResult,
    Extract,
    Goal,
    GoalStatus,
    LiveKitConfig,
    Persona,
    SessionState,
    Turn,
    TurnContext,
)
from interview_kit.sinks.memory import InMemoryEventSink
from interview_kit.stores.memory import InMemoryConversationStore


class _StubLLM:
    """Minimal LLMClient — none of these methods are exercised in Step 5."""

    async def evaluate_turn(self, ctx: TurnContext) -> EvalResult:
        raise AssertionError("LLM not exercised in Step 5")

    def compose_utterance(
        self, ctx: TurnContext, eval_result: EvalResult
    ) -> AsyncIterator[str]:
        raise AssertionError("LLM not exercised in Step 5")

    async def derive_extract(
        self, transcript: list[Turn], conv: Conversation
    ) -> Extract:
        raise AssertionError("LLM not exercised in Step 5")


def _engine(*, livekit: LiveKitConfig | None = None) -> Engine:
    return Engine(
        store=InMemoryConversationStore(),
        events=InMemoryEventSink(),
        llm=_StubLLM(),
        livekit=livekit,
    )


def _persona() -> Persona:
    return Persona(
        system_prompt="You are an interview_kit.",
        style="neutral",
        voice_id="cartesia-1",
    )


def _bg() -> Background:
    return Background(interviewee_role="r", interviewee_expertise="e")


def _goals() -> list[Goal]:
    return [
        Goal(id="g1", intent="i1", standard="s1"),
        Goal(id="g2", intent="i2", standard="s2"),
    ]


async def _make_conv(engine: Engine) -> Conversation:
    return await engine.create_conversation(
        persona=_persona(),
        purpose="discovery",
        background=_bg(),
        goals=_goals(),
    )


# ---------- create_conversation --------------------------------------------


async def test_create_conversation_persists_and_returns() -> None:
    engine = _engine()
    conv = await _make_conv(engine)
    assert conv.id.startswith("conv-")
    loaded = await engine.store.load_conversation(conv.id)
    assert loaded == conv


async def test_create_conversation_emits_no_event() -> None:
    engine = _engine()
    await _make_conv(engine)
    sink = engine.events
    assert isinstance(sink, InMemoryEventSink)
    assert sink.events == []


# ---------- provision_session ----------------------------------------------


async def test_provision_session_returns_ready_session_and_creds() -> None:
    engine = _engine()
    conv = await _make_conv(engine)
    session, creds = await engine.provision_session(conv.id)

    assert session.state is SessionState.READY
    assert session.conversation_id == conv.id
    assert session.conversation_snapshot == conv  # D10 snapshot
    assert creds.token.startswith("stub-token-")
    assert creds.room_url == f"stub://room/iv:{session.id}"
    assert (creds.expires_at - session.created_at).total_seconds() == pytest.approx(
        86400, abs=2
    )


async def test_provision_session_emits_session_provisioned() -> None:
    engine = _engine()
    conv = await _make_conv(engine)
    session, _ = await engine.provision_session(conv.id)

    sink = engine.events
    assert isinstance(sink, InMemoryEventSink)
    assert [e.type for e in sink.events] == ["session_provisioned"]
    e = sink.events[0]
    assert e.session_id == session.id
    assert e.conversation_id == conv.id
    assert "room_url" in e.payload and "expires_at" in e.payload


async def test_provision_session_uses_livekit_url_when_configured() -> None:
    cfg = LiveKitConfig(
        url="wss://livekit.example.com",
        api_key="k-secret-at-least-thirty-two-bytes-long-for-jwt",
        api_secret="s-secret-at-least-thirty-two-bytes-long-for-jwt",
        agent_name="interviewer",
    )
    engine = _engine(livekit=cfg)
    conv = await _make_conv(engine)
    # With livekit configured, room_url is the LiveKit WebSocket URL
    # verbatim; the room name lives inside the JWT (Step 13).
    _, creds = await engine.provision_session(conv.id)
    assert creds.room_url == "wss://livekit.example.com"


async def test_provision_session_unknown_conversation_raises() -> None:
    engine = _engine()
    with pytest.raises(KeyError):
        await engine.provision_session("missing")


async def test_provision_session_uses_snapshot_not_live_conversation() -> None:
    """D10: subsequent edits to the underlying Conversation don't bleed through."""
    engine = _engine()
    conv = await _make_conv(engine)
    session, _ = await engine.provision_session(conv.id)

    # Operator "edits" the conversation in storage.
    edited = conv.model_copy(update={"purpose": "different purpose"})
    await engine.store.save_conversation(edited)

    # Session's snapshot is unchanged.
    loaded = await engine.store.load_session(session.id)
    assert loaded.conversation_snapshot.purpose == "discovery"


# ---------- reprovision_session --------------------------------------------


async def test_reprovision_issues_new_credentials() -> None:
    engine = _engine()
    conv = await _make_conv(engine)
    session, creds1 = await engine.provision_session(conv.id)
    creds2 = await engine.reprovision_session(session.id)
    assert creds1.token != creds2.token


async def test_reprovision_emits_session_provisioned_with_flag() -> None:
    engine = _engine()
    conv = await _make_conv(engine)
    session, _ = await engine.provision_session(conv.id)
    await engine.reprovision_session(session.id)
    sink = engine.events
    assert isinstance(sink, InMemoryEventSink)
    types = [e.type for e in sink.events]
    assert types == ["session_provisioned", "session_provisioned"]
    assert sink.events[1].payload.get("reprovisioned") is True


async def test_reprovision_on_completed_raises() -> None:
    engine = _engine()
    conv = await _make_conv(engine)
    session, _ = await engine.provision_session(conv.id)
    await engine.store.update_session_state(session.id, SessionState.COMPLETED)
    with pytest.raises(ValueError, match="terminal"):
        await engine.reprovision_session(session.id)


async def test_reprovision_on_abandoned_raises() -> None:
    engine = _engine()
    conv = await _make_conv(engine)
    session, _ = await engine.provision_session(conv.id)
    await engine.cancel_session(session.id)
    with pytest.raises(ValueError, match="terminal"):
        await engine.reprovision_session(session.id)


async def test_reprovision_unknown_session_raises() -> None:
    engine = _engine()
    with pytest.raises(KeyError):
        await engine.reprovision_session("missing")


# ---------- cancel_session -------------------------------------------------


async def test_cancel_session_writes_abandoned_and_emits() -> None:
    engine = _engine()
    conv = await _make_conv(engine)
    session, _ = await engine.provision_session(conv.id)
    await engine.cancel_session(session.id, reason="operator killed")

    loaded = await engine.store.load_session(session.id)
    assert loaded.state is SessionState.ABANDONED

    sink = engine.events
    assert isinstance(sink, InMemoryEventSink)
    abandoned_events = [e for e in sink.events if e.type == "abandoned"]
    assert len(abandoned_events) == 1
    assert abandoned_events[0].payload["reason"] == "operator killed"


async def test_double_cancel_raises() -> None:
    engine = _engine()
    conv = await _make_conv(engine)
    session, _ = await engine.provision_session(conv.id)
    await engine.cancel_session(session.id)
    with pytest.raises(ValueError, match="terminal"):
        await engine.cancel_session(session.id)


async def test_cancel_completed_raises() -> None:
    engine = _engine()
    conv = await _make_conv(engine)
    session, _ = await engine.provision_session(conv.id)
    await engine.store.update_session_state(session.id, SessionState.COMPLETED)
    with pytest.raises(ValueError, match="terminal"):
        await engine.cancel_session(session.id)


async def test_cancel_unknown_session_raises() -> None:
    engine = _engine()
    with pytest.raises(KeyError):
        await engine.cancel_session("missing")


async def test_cancel_does_not_emit_goal_status_changed() -> None:
    # D5: no goal_status_changed events anywhere in Step 5.
    engine = _engine()
    conv = await _make_conv(engine)
    session, _ = await engine.provision_session(conv.id)
    await engine.cancel_session(session.id)
    sink = engine.events
    assert isinstance(sink, InMemoryEventSink)
    assert all(e.type != "goal_status_changed" for e in sink.events)


# ---------- get_session_status ---------------------------------------------


async def test_get_status_freshly_provisioned() -> None:
    engine = _engine()
    conv = await _make_conv(engine)
    session, _ = await engine.provision_session(conv.id)

    status = await engine.get_session_status(session.id)
    assert status.session_id == session.id
    assert status.state is SessionState.READY
    assert status.active_goal_id is None
    assert status.total_turns == 0
    assert status.goals_resolved == 0
    assert status.goals_total == 2
    assert status.started_at is None
    assert status.last_turn_at is None


async def test_get_status_reflects_appended_turns() -> None:
    engine = _engine()
    conv = await _make_conv(engine)
    session, _ = await engine.provision_session(conv.id)
    now = datetime.now(UTC)
    await engine.store.append_turn(
        session.id,
        Turn(
            index=0,
            speaker="agent",
            text="hello",
            timestamp=now,
            addressed_goal_ids=["g1"],
        ),
    )
    status = await engine.get_session_status(session.id)
    assert status.total_turns == 1
    assert status.last_turn_at == now
    assert status.goals_resolved == 1  # g1 touched by a turn


async def test_get_status_uses_extract_when_present() -> None:
    engine = _engine()
    conv = await _make_conv(engine)
    session, _ = await engine.provision_session(conv.id)
    extract = Extract(
        session_id=session.id,
        conversation_id=conv.id,
        goal_statuses=[
            GoalStatus(goal_id="g1", status="meets"),
            GoalStatus(goal_id="g2", status="pending"),
        ],
        full_transcript=[],
        completed_at=datetime.now(UTC),
    )
    await engine.store.save_extract(extract)
    status = await engine.get_session_status(session.id)
    assert status.goals_resolved == 1
    assert status.goals_total == 2


async def test_get_status_unknown_session_raises() -> None:
    engine = _engine()
    with pytest.raises(KeyError):
        await engine.get_session_status("missing")


# ---------- get_transcript / get_extract -----------------------------------


async def test_get_transcript_returns_turns() -> None:
    engine = _engine()
    conv = await _make_conv(engine)
    session, _ = await engine.provision_session(conv.id)
    t = Turn(index=0, speaker="agent", text="hi", timestamp=datetime.now(UTC))
    await engine.store.append_turn(session.id, t)
    assert await engine.get_transcript(session.id) == [t]


async def test_get_extract_returns_none_before_completion() -> None:
    engine = _engine()
    conv = await _make_conv(engine)
    session, _ = await engine.provision_session(conv.id)
    assert await engine.get_extract(session.id) is None


# Voice entrypoint is exercised in tests/voice/test_livekit_entry.py
# (Step 13). simulate_session is implemented in Step 8 — end-to-end
# coverage lives in tests/loop/test_runner_happy_path.py.
