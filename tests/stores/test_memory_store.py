"""Round-trip tests for ``InMemoryConversationStore``.

The bulk of the suite lives in ``_round_trip.StoreRoundTripSuite``; this
module supplies the store fixture for that suite and adds a static
protocol-conformance check.
"""

from __future__ import annotations

from collections.abc import AsyncIterator

import pytest_asyncio

from interview_kit.protocols import ConversationStore
from interview_kit.stores.memory import InMemoryConversationStore
from tests.stores._round_trip import StoreRoundTripSuite


def _store_as_protocol(store: ConversationStore) -> ConversationStore:
    """Static check that InMemoryConversationStore satisfies ConversationStore."""
    return store


def test_protocol_conformance_static() -> None:
    _store_as_protocol(InMemoryConversationStore())


class TestInMemoryConversationStore(StoreRoundTripSuite):
    @pytest_asyncio.fixture
    async def store(self) -> AsyncIterator[ConversationStore]:
        yield InMemoryConversationStore()
