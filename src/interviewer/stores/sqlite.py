"""SQLite-backed ConversationStore via aiosqlite.

Reasonable default for single-process deployments. The schema is one
JSON column per Pydantic blob; queries are mostly key lookups. No ORM,
no migrations system in v1 — ``connect()`` creates missing tables.

Foreign keys are declared and enforced on ``turns`` (the only protocol
method that validates session existence). ``runtime_states`` and
``extracts`` deliberately omit FKs to mirror ``InMemoryConversationStore``
(which lets the runner write either before the Session row exists).
"""

from __future__ import annotations

from pathlib import Path
from types import TracebackType
from typing import Self

import aiosqlite

from interviewer.types.config import Conversation
from interviewer.types.runtime import (
    Extract,
    Session,
    SessionRuntimeState,
    Turn,
)
from interviewer.types.state import SessionState

_SCHEMA = """
CREATE TABLE IF NOT EXISTS conversations (
    id   TEXT PRIMARY KEY,
    data TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS sessions (
    id                     TEXT PRIMARY KEY,
    conversation_id        TEXT NOT NULL,
    conversation_snapshot  TEXT NOT NULL,
    state                  TEXT NOT NULL,
    data                   TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS turns (
    session_id TEXT    NOT NULL REFERENCES sessions(id),
    turn_index INTEGER NOT NULL,
    data       TEXT    NOT NULL,
    PRIMARY KEY (session_id, turn_index)
);

CREATE INDEX IF NOT EXISTS idx_turns_session_index
    ON turns (session_id, turn_index);

CREATE TABLE IF NOT EXISTS runtime_states (
    session_id TEXT PRIMARY KEY,
    data       TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS extracts (
    session_id TEXT PRIMARY KEY,
    data       TEXT NOT NULL
);
"""


class SQLiteConversationStore:
    """aiosqlite-backed ConversationStore. Call ``connect()`` before use."""

    def __init__(self, path: str | Path = ":memory:") -> None:
        self._path = str(path)
        self._conn: aiosqlite.Connection | None = None

    async def connect(self) -> None:
        if self._conn is not None:
            return
        self._conn = await aiosqlite.connect(self._path)
        await self._conn.execute("PRAGMA foreign_keys = ON")
        await self._conn.executescript(_SCHEMA)
        await self._conn.commit()

    async def close(self) -> None:
        if self._conn is not None:
            await self._conn.close()
            self._conn = None

    async def __aenter__(self) -> Self:
        await self.connect()
        return self

    async def __aexit__(
        self,
        exc_type: type[BaseException] | None,
        exc: BaseException | None,
        tb: TracebackType | None,
    ) -> None:
        await self.close()

    @property
    def _db(self) -> aiosqlite.Connection:
        if self._conn is None:
            raise RuntimeError("SQLiteConversationStore: call .connect() before use")
        return self._conn

    async def save_conversation(self, c: Conversation) -> None:
        await self._db.execute(
            "INSERT OR REPLACE INTO conversations (id, data) VALUES (?, ?)",
            (c.id, c.model_dump_json()),
        )
        await self._db.commit()

    async def load_conversation(self, conversation_id: str) -> Conversation:
        async with self._db.execute(
            "SELECT data FROM conversations WHERE id = ?", (conversation_id,)
        ) as cur:
            row = await cur.fetchone()
        if row is None:
            raise KeyError(f"conversation {conversation_id!r} not found")
        return Conversation.model_validate_json(row[0])

    async def save_session(self, s: Session) -> None:
        await self._db.execute(
            "INSERT OR REPLACE INTO sessions "
            "(id, conversation_id, conversation_snapshot, state, data) "
            "VALUES (?, ?, ?, ?, ?)",
            (
                s.id,
                s.conversation_id,
                s.conversation_snapshot.model_dump_json(),
                s.state.value,
                s.model_dump_json(),
            ),
        )
        await self._db.commit()

    async def load_session(self, session_id: str) -> Session:
        async with self._db.execute(
            "SELECT data FROM sessions WHERE id = ?", (session_id,)
        ) as cur:
            row = await cur.fetchone()
        if row is None:
            raise KeyError(f"session {session_id!r} not found")
        return Session.model_validate_json(row[0])

    async def update_session_state(self, session_id: str, state: SessionState) -> None:
        current = await self.load_session(session_id)
        updated = current.model_copy(update={"state": state})
        await self._db.execute(
            "UPDATE sessions SET state = ?, data = ? WHERE id = ?",
            (state.value, updated.model_dump_json(), session_id),
        )
        await self._db.commit()

    async def append_turn(self, session_id: str, turn: Turn) -> None:
        async with self._db.execute(
            "SELECT 1 FROM sessions WHERE id = ?", (session_id,)
        ) as cur:
            if await cur.fetchone() is None:
                raise KeyError(f"session {session_id!r} not found")
        await self._db.execute(
            "INSERT INTO turns (session_id, turn_index, data) VALUES (?, ?, ?)",
            (session_id, turn.index, turn.model_dump_json()),
        )
        await self._db.commit()

    async def list_turns(self, session_id: str) -> list[Turn]:
        async with self._db.execute(
            "SELECT 1 FROM sessions WHERE id = ?", (session_id,)
        ) as cur:
            if await cur.fetchone() is None:
                raise KeyError(f"session {session_id!r} not found")
        async with self._db.execute(
            "SELECT data FROM turns WHERE session_id = ? ORDER BY turn_index ASC",
            (session_id,),
        ) as cur:
            rows = await cur.fetchall()
        return [Turn.model_validate_json(row[0]) for row in rows]

    async def save_runtime_state(self, rs: SessionRuntimeState) -> None:
        await self._db.execute(
            "INSERT OR REPLACE INTO runtime_states (session_id, data) VALUES (?, ?)",
            (rs.session_id, rs.model_dump_json()),
        )
        await self._db.commit()

    async def load_runtime_state(
        self, session_id: str
    ) -> SessionRuntimeState | None:
        async with self._db.execute(
            "SELECT data FROM runtime_states WHERE session_id = ?", (session_id,)
        ) as cur:
            row = await cur.fetchone()
        if row is None:
            return None
        return SessionRuntimeState.model_validate_json(row[0])

    async def save_extract(self, extract: Extract) -> None:
        await self._db.execute(
            "INSERT OR REPLACE INTO extracts (session_id, data) VALUES (?, ?)",
            (extract.session_id, extract.model_dump_json()),
        )
        await self._db.commit()

    async def load_extract(self, session_id: str) -> Extract | None:
        async with self._db.execute(
            "SELECT data FROM extracts WHERE session_id = ?", (session_id,)
        ) as cur:
            row = await cur.fetchone()
        if row is None:
            return None
        return Extract.model_validate_json(row[0])
