"""Round-trip tests for ``SQLiteConversationStore``.

Uses the shared ``StoreRoundTripSuite`` against a fresh on-disk SQLite
database per test (via ``tmp_path``). Sqlite-specific behaviors
(persistence across reconnect, ``connect()`` idempotence) live in
module-level tests.
"""

from __future__ import annotations

from collections.abc import AsyncIterator
from datetime import UTC, datetime
from pathlib import Path

import pytest
import pytest_asyncio

from interviewer import (
    Background,
    Conversation,
    Goal,
    Persona,
    Session,
)
from interviewer.protocols import ConversationStore
from interviewer.stores.sqlite import SQLiteConversationStore
from tests.stores._round_trip import StoreRoundTripSuite


def _store_as_protocol(store: ConversationStore) -> ConversationStore:
    """Static check that SQLiteConversationStore satisfies ConversationStore."""
    return store


async def test_protocol_conformance_static() -> None:
    store = SQLiteConversationStore()
    await store.connect()
    try:
        _store_as_protocol(store)
    finally:
        await store.close()


class TestSQLiteConversationStore(StoreRoundTripSuite):
    @pytest_asyncio.fixture
    async def store(self, tmp_path: Path) -> AsyncIterator[ConversationStore]:
        s = SQLiteConversationStore(tmp_path / "iv.sqlite")
        await s.connect()
        try:
            yield s
        finally:
            await s.close()


# ---------- SQLite-specific behavior --------------------------------------


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


async def test_data_persists_across_reconnect(tmp_path: Path) -> None:
    path = tmp_path / "iv.sqlite"
    s1 = SQLiteConversationStore(path)
    await s1.connect()
    conv = _conv()
    await s1.save_conversation(conv)
    await s1.save_session(
        Session(
            id="s1",
            conversation_id=conv.id,
            conversation_snapshot=conv,
            created_at=_ts(),
        )
    )
    await s1.close()

    s2 = SQLiteConversationStore(path)
    await s2.connect()
    try:
        assert await s2.load_conversation(conv.id) == conv
        loaded = await s2.load_session("s1")
        assert loaded.id == "s1"
        # The snapshot survives the JSON round-trip with all nested fields intact.
        assert loaded.conversation_snapshot == conv
    finally:
        await s2.close()


async def test_connect_is_idempotent(tmp_path: Path) -> None:
    s = SQLiteConversationStore(tmp_path / "iv.sqlite")
    await s.connect()
    await s.connect()  # second call is a no-op, must not raise
    await s.save_conversation(_conv())
    await s.close()


async def test_usage_before_connect_raises(tmp_path: Path) -> None:
    s = SQLiteConversationStore(tmp_path / "iv.sqlite")
    with pytest.raises(RuntimeError, match="connect"):
        await s.save_conversation(_conv())


async def test_async_context_manager(tmp_path: Path) -> None:
    async with SQLiteConversationStore(tmp_path / "iv.sqlite") as s:
        await s.save_conversation(_conv())
        assert (await s.load_conversation("conv-1")).id == "conv-1"
