"""Shared round-trip test suite for ``ConversationStore`` implementations.

Subclasses provide a ``store`` async fixture yielding a fresh
``ConversationStore`` for each test. Subclass names start with ``Test``
so pytest collects them; this base does not, so it is not collected.
"""

from __future__ import annotations

from collections.abc import AsyncIterator
from datetime import UTC, datetime

import pytest

from interviewer import (
    Background,
    Conversation,
    Extract,
    Goal,
    GoalStatus,
    Persona,
    Session,
    SessionRuntimeState,
    SessionState,
    Turn,
)
from interviewer.protocols import ConversationStore


def _ts() -> datetime:
    return datetime(2026, 5, 10, 12, 0, 0, tzinfo=UTC)


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


def _session(conv: Conversation) -> Session:
    return Session(
        id="s1",
        conversation_id=conv.id,
        conversation_snapshot=conv,
        created_at=_ts(),
    )


class StoreRoundTripSuite:
    """Round-trip suite covering every ``ConversationStore`` protocol method."""

    # Subclasses override this fixture.
    async def store(self) -> AsyncIterator[ConversationStore]:  # pragma: no cover
        raise NotImplementedError

    # ---------- Conversation -----------------------------------------------

    async def test_save_and_load_conversation_round_trip(
        self, store: ConversationStore
    ) -> None:
        conv = _conv()
        await store.save_conversation(conv)
        assert await store.load_conversation(conv.id) == conv

    async def test_load_unknown_conversation_raises(
        self, store: ConversationStore
    ) -> None:
        with pytest.raises(KeyError):
            await store.load_conversation("missing")

    # ---------- Session ----------------------------------------------------

    async def test_save_and_load_session_round_trip(
        self, store: ConversationStore
    ) -> None:
        conv = _conv()
        await store.save_conversation(conv)
        s = _session(conv)
        await store.save_session(s)
        assert await store.load_session(s.id) == s

    async def test_load_unknown_session_raises(
        self, store: ConversationStore
    ) -> None:
        with pytest.raises(KeyError):
            await store.load_session("missing")

    async def test_update_session_state_mutates_via_copy(
        self, store: ConversationStore
    ) -> None:
        conv = _conv()
        await store.save_conversation(conv)
        s = _session(conv)
        await store.save_session(s)

        await store.update_session_state(s.id, SessionState.READY)
        loaded = await store.load_session(s.id)
        assert loaded.state is SessionState.READY
        # The original frozen instance is unchanged.
        assert s.state is SessionState.CREATED

    async def test_update_session_state_unknown_raises(
        self, store: ConversationStore
    ) -> None:
        with pytest.raises(KeyError):
            await store.update_session_state("missing", SessionState.READY)

    # ---------- Turns ------------------------------------------------------

    async def test_append_and_list_turns_preserves_order(
        self, store: ConversationStore
    ) -> None:
        conv = _conv()
        await store.save_conversation(conv)
        await store.save_session(_session(conv))

        turns = [
            Turn(index=0, speaker="agent", text="hi", timestamp=_ts()),
            Turn(index=1, speaker="respondent", text="hello", timestamp=_ts()),
            Turn(index=2, speaker="agent", text="next q", timestamp=_ts()),
        ]
        for t in turns:
            await store.append_turn("s1", t)

        assert await store.list_turns("s1") == turns

    async def test_list_turns_for_session_with_no_turns_is_empty(
        self, store: ConversationStore
    ) -> None:
        conv = _conv()
        await store.save_conversation(conv)
        await store.save_session(_session(conv))
        assert await store.list_turns("s1") == []

    async def test_list_turns_returns_copy_not_internal_list(
        self, store: ConversationStore
    ) -> None:
        conv = _conv()
        await store.save_conversation(conv)
        await store.save_session(_session(conv))
        await store.append_turn(
            "s1", Turn(index=0, speaker="agent", text="x", timestamp=_ts())
        )

        snapshot = await store.list_turns("s1")
        snapshot.clear()
        # Mutating the returned list must not affect the store.
        assert len(await store.list_turns("s1")) == 1

    async def test_append_turn_for_unknown_session_raises(
        self, store: ConversationStore
    ) -> None:
        with pytest.raises(KeyError):
            await store.append_turn(
                "missing",
                Turn(index=0, speaker="agent", text="x", timestamp=_ts()),
            )

    # ---------- Runtime state ----------------------------------------------

    async def test_save_and_load_runtime_state(
        self, store: ConversationStore
    ) -> None:
        rs = SessionRuntimeState(
            session_id="s1",
            active_goal_id="g1",
            retries_used_on_active=1,
            tangent_followups_used=0,
            total_turns=2,
            pending_follow_up=None,
            last_event_index=2,
            updated_at=_ts(),
        )
        await store.save_runtime_state(rs)
        assert await store.load_runtime_state("s1") == rs

    async def test_load_runtime_state_missing_returns_none(
        self, store: ConversationStore
    ) -> None:
        assert await store.load_runtime_state("missing") is None

    async def test_save_runtime_state_overwrites_previous(
        self, store: ConversationStore
    ) -> None:
        rs1 = SessionRuntimeState(
            session_id="s1",
            active_goal_id="g1",
            retries_used_on_active=0,
            tangent_followups_used=0,
            total_turns=1,
            last_event_index=0,
            updated_at=_ts(),
        )
        rs2 = rs1.model_copy(update={"total_turns": 5})
        await store.save_runtime_state(rs1)
        await store.save_runtime_state(rs2)
        assert await store.load_runtime_state("s1") == rs2

    # ---------- Extract ----------------------------------------------------

    async def test_save_and_load_extract(self, store: ConversationStore) -> None:
        ex = Extract(
            session_id="s1",
            conversation_id="conv-1",
            goal_statuses=[GoalStatus(goal_id="g1", status="meets")],
            full_transcript=[
                Turn(index=0, speaker="agent", text="hi", timestamp=_ts()),
            ],
            completed_at=_ts(),
        )
        await store.save_extract(ex)
        assert await store.load_extract("s1") == ex

    async def test_load_extract_missing_returns_none(
        self, store: ConversationStore
    ) -> None:
        assert await store.load_extract("missing") is None
