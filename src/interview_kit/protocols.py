"""Consumer-implemented protocols.

All I/O-bearing methods are ``async def``; engine code awaits them.

- ``ConversationStore`` — persistence. Consumers swap in SQLite, Postgres,
  files, an ORM; the engine doesn't care.
- ``EventSink`` — routes ``SessionEvent`` to wherever the consumer wants
  (webhook, queue, log, websocket).
- ``LLMClient`` — the brain. Each agent turn is two sequential calls:
  ``evaluate_turn`` (structured JSON, small/fast model) followed by
  ``compose_utterance`` (streaming text, larger model). ``derive_extract``
  is a single non-streaming call at session end.
- ``RespondentSimulator`` — test harness for ``simulate_session``. Only
  this protocol is ``runtime_checkable`` because tests sometimes need an
  ``isinstance`` guard when assembling simulator fixtures.

Implementations don't inherit from these protocols — structural typing
applies.
"""

from __future__ import annotations

from collections.abc import AsyncIterator
from typing import Protocol, runtime_checkable

from interview_kit.types.config import Conversation
from interview_kit.types.events import SessionEvent
from interview_kit.types.runtime import (
    EvalResult,
    Extract,
    Session,
    SessionRuntimeState,
    Turn,
    TurnContext,
)
from interview_kit.types.state import SessionState


class ConversationStore(Protocol):
    """Persistence for Conversations, Sessions, Turns, runtime state, Extracts."""

    async def save_conversation(self, c: Conversation) -> None: ...
    async def load_conversation(self, conversation_id: str) -> Conversation: ...
    async def save_session(self, s: Session) -> None: ...
    async def load_session(self, session_id: str) -> Session: ...
    async def update_session_state(
        self, session_id: str, state: SessionState
    ) -> None: ...
    async def append_turn(self, session_id: str, turn: Turn) -> None: ...
    async def list_turns(self, session_id: str) -> list[Turn]: ...
    async def save_runtime_state(self, rs: SessionRuntimeState) -> None: ...
    async def load_runtime_state(
        self, session_id: str
    ) -> SessionRuntimeState | None: ...
    async def save_extract(self, extract: Extract) -> None: ...
    async def load_extract(self, session_id: str) -> Extract | None: ...


class EventSink(Protocol):
    """Routes one SessionEvent to the consumer's chosen destination."""

    async def emit(self, event: SessionEvent) -> None: ...


class LLMClient(Protocol):
    """The brain. Two sequential calls per agent turn."""

    async def evaluate_turn(self, ctx: TurnContext) -> EvalResult: ...
    def compose_utterance(
        self, ctx: TurnContext, eval_result: EvalResult
    ) -> AsyncIterator[str]: ...
    async def compose_closing_recap(self, ctx: TurnContext) -> str: ...
    async def derive_extract(
        self, transcript: list[Turn], conv: Conversation
    ) -> Extract: ...


@runtime_checkable
class RespondentSimulator(Protocol):
    """Synthetic respondent for ``simulate_session``. Text-mode only."""

    async def respond(self, agent_utterance: str, history: list[Turn]) -> str: ...
    def persona_name(self) -> str: ...
