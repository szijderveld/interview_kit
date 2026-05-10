"""In-memory ConversationStore — reference implementation for tests and examples.

Not for production. The dict-backed state is thread-safe under cooperative
asyncio scheduling via a single ``asyncio.Lock``; persistence does not
survive process restart.
"""

from __future__ import annotations

import asyncio

from interviewer.types.config import Conversation
from interviewer.types.runtime import (
    Extract,
    Session,
    SessionRuntimeState,
    Turn,
)
from interviewer.types.state import SessionState


class InMemoryConversationStore:
    """Dict-backed ConversationStore. Reference impl only."""

    def __init__(self) -> None:
        self._lock = asyncio.Lock()
        self._conversations: dict[str, Conversation] = {}
        self._sessions: dict[str, Session] = {}
        self._turns: dict[str, list[Turn]] = {}
        self._runtime: dict[str, SessionRuntimeState] = {}
        self._extracts: dict[str, Extract] = {}

    async def save_conversation(self, c: Conversation) -> None:
        async with self._lock:
            self._conversations[c.id] = c

    async def load_conversation(self, conversation_id: str) -> Conversation:
        async with self._lock:
            try:
                return self._conversations[conversation_id]
            except KeyError as exc:
                raise KeyError(
                    f"conversation {conversation_id!r} not found"
                ) from exc

    async def save_session(self, s: Session) -> None:
        async with self._lock:
            self._sessions[s.id] = s
            self._turns.setdefault(s.id, [])

    async def load_session(self, session_id: str) -> Session:
        async with self._lock:
            try:
                return self._sessions[session_id]
            except KeyError as exc:
                raise KeyError(f"session {session_id!r} not found") from exc

    async def update_session_state(
        self, session_id: str, state: SessionState
    ) -> None:
        async with self._lock:
            try:
                current = self._sessions[session_id]
            except KeyError as exc:
                raise KeyError(f"session {session_id!r} not found") from exc
            self._sessions[session_id] = current.model_copy(update={"state": state})

    async def append_turn(self, session_id: str, turn: Turn) -> None:
        async with self._lock:
            if session_id not in self._sessions:
                raise KeyError(f"session {session_id!r} not found")
            self._turns.setdefault(session_id, []).append(turn)

    async def list_turns(self, session_id: str) -> list[Turn]:
        async with self._lock:
            if session_id not in self._sessions:
                raise KeyError(f"session {session_id!r} not found")
            return list(self._turns.get(session_id, []))

    async def save_runtime_state(self, rs: SessionRuntimeState) -> None:
        async with self._lock:
            self._runtime[rs.session_id] = rs

    async def load_runtime_state(
        self, session_id: str
    ) -> SessionRuntimeState | None:
        async with self._lock:
            return self._runtime.get(session_id)

    async def save_extract(self, extract: Extract) -> None:
        async with self._lock:
            self._extracts[extract.session_id] = extract

    async def load_extract(self, session_id: str) -> Extract | None:
        async with self._lock:
            return self._extracts.get(session_id)
