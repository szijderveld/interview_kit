"""Shared fixtures for the Step 15 integration suite.

Each integration test exercises the full loop + extract + store stack.
The ``store`` fixture is parametrized over the two reference stores
(``InMemoryConversationStore`` and ``SQLiteConversationStore``) so every
scenario runs against both backends and the suite stays honest about the
SQLite persistence path.
"""

from __future__ import annotations

from collections.abc import AsyncIterator
from pathlib import Path

import pytest
import pytest_asyncio

from interview_kit import Background, Goal, Persona
from interview_kit.protocols import ConversationStore
from interview_kit.stores.memory import InMemoryConversationStore
from interview_kit.stores.sqlite import SQLiteConversationStore

# ---------- Conversation template: engineer interviewing engineer ----------


@pytest.fixture
def engineer_persona() -> Persona:
    return Persona(
        system_prompt=(
            "You are a senior engineer doing a discovery interview with "
            "another engineer to understand how their team works."
        ),
        style="neutral",
        voice_id="cartesia-1",
    )


@pytest.fixture
def engineer_background() -> Background:
    return Background(
        interviewee_role="backend platform engineer",
        interviewee_expertise="distributed systems at a mid-stage SaaS startup",
        relevant_context="Team recently migrated off a monolith.",
    )


@pytest.fixture
def engineer_goals() -> list[Goal]:
    """Five goals covering role, stack, failure modes, process, and direction."""
    return [
        Goal(
            id="role",
            intent="Their team and primary responsibility.",
            standard="At least one named team and one named duty.",
        ),
        Goal(
            id="stack",
            intent="Languages and core infrastructure.",
            standard="At least two named technologies.",
        ),
        Goal(
            id="bugs",
            intent="Where bugs most often surface.",
            standard="One concrete failure pattern.",
        ),
        Goal(
            id="review",
            intent="How code review is run.",
            standard="One named convention or tool.",
        ),
        Goal(
            id="future",
            intent="What the next 6 months focus on.",
            standard="One specific initiative.",
        ),
    ]


# ---------- store fixture (parametrized over backends) ----------


@pytest_asyncio.fixture(params=["memory", "sqlite"])
async def store(
    request: pytest.FixtureRequest, tmp_path: Path
) -> AsyncIterator[ConversationStore]:
    backend = request.param
    if backend == "memory":
        yield InMemoryConversationStore()
    else:
        s = SQLiteConversationStore(tmp_path / "iv.sqlite")
        await s.connect()
        try:
            yield s
        finally:
            await s.close()
