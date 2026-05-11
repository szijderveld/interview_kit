"""Top-level engine — what consumers instantiate and call into.

Step 5 wired up the non-loop methods against the in-memory store; Step 8
filled in ``simulate_session``; Step 13 makes the LiveKit voice path
live. When ``livekit`` is configured, ``provision_session`` mints a real
LiveKit AccessToken for the room ``iv:{session_id}``, ``cancel_session``
deletes that room over the LiveKit server API after writing ABANDONED,
and ``entrypoint`` delegates to :mod:`interviewer.voice.livekit_entry`.

State transitions emit ``SessionEvent``s through the configured sink.
Per D5, ``goal_status_changed`` events are only emitted at completion
(inside the canonical extract pass).
"""

from __future__ import annotations

import uuid
from datetime import UTC, datetime, timedelta

from interviewer.livekit_config import LiveKitConfig
from interviewer.protocols import (
    ConversationStore,
    EventSink,
    LLMClient,
    RespondentSimulator,
)
from interviewer.types.config import Background, Conversation, Goal, Persona
from interviewer.types.events import SessionEvent
from interviewer.types.runtime import (
    Extract,
    Session,
    SessionCredentials,
    SessionStatus,
    Turn,
)
from interviewer.types.state import SessionState

_TOKEN_TTL = timedelta(hours=24)  # D9
_TERMINAL_STATES: frozenset[SessionState] = frozenset(
    {SessionState.COMPLETED, SessionState.ABANDONED, SessionState.FAILED}
)


class Engine:
    """Orchestrates Conversations, Sessions, and the agent loop."""

    def __init__(
        self,
        store: ConversationStore,
        events: EventSink,
        llm: LLMClient,
        livekit: LiveKitConfig | None = None,
    ) -> None:
        self.store = store
        self.events = events
        self.llm = llm
        self.livekit = livekit

    # ---- conversation lifecycle ------------------------------------------

    async def create_conversation(
        self,
        *,
        persona: Persona,
        purpose: str,
        background: Background,
        goals: list[Goal],
        opening: str | None = None,
        closing: str | None = None,
    ) -> Conversation:
        """Persist a Conversation. Emits no event — config, not session, action."""
        conv = Conversation(
            id=_new_id("conv"),
            persona=persona,
            purpose=purpose,
            background=background,
            goals=goals,
            opening=opening,
            closing=closing,
        )
        await self.store.save_conversation(conv)
        return conv

    # ---- session lifecycle -----------------------------------------------

    async def provision_session(
        self, conversation_id: str
    ) -> tuple[Session, SessionCredentials]:
        """Snapshot the Conversation onto a new Session (D10) and issue creds.

        State → READY. Emits ``session_provisioned``.
        """
        conv = await self.store.load_conversation(conversation_id)
        session_id = _new_id("sess")
        now = _utcnow()
        session = Session(
            id=session_id,
            conversation_id=conv.id,
            conversation_snapshot=conv,
            state=SessionState.READY,
            created_at=now,
        )
        await self.store.save_session(session)
        creds = self._mint_credentials(session_id, now)
        await self.events.emit(
            SessionEvent(
                session_id=session_id,
                conversation_id=conv.id,
                timestamp=now,
                type="session_provisioned",
                payload={
                    "room_url": creds.room_url,
                    "expires_at": creds.expires_at.isoformat(),
                },
            )
        )
        return session, creds

    async def reprovision_session(self, session_id: str) -> SessionCredentials:
        """Re-issue credentials. Allowed only while non-terminal."""
        session = await self.store.load_session(session_id)
        if session.state in _TERMINAL_STATES:
            raise ValueError(
                f"cannot reprovision session in terminal state {session.state}"
            )
        now = _utcnow()
        creds = self._mint_credentials(session_id, now)
        await self.events.emit(
            SessionEvent(
                session_id=session_id,
                conversation_id=session.conversation_id,
                timestamp=now,
                type="session_provisioned",
                payload={
                    "room_url": creds.room_url,
                    "expires_at": creds.expires_at.isoformat(),
                    "reprovisioned": True,
                },
            )
        )
        return creds

    async def cancel_session(self, session_id: str, reason: str = "") -> None:
        """Operator-initiated termination. Writes ABANDONED, emits ``abandoned``.

        When ``livekit`` is configured, also deletes the room — which
        disconnects any in-flight ``AgentSession`` and lets the entrypoint
        unwind via its room-disconnect handler (D4).
        """
        session = await self.store.load_session(session_id)
        if session.state in _TERMINAL_STATES:
            raise ValueError(
                f"cannot cancel session in terminal state {session.state}"
            )
        await self.store.update_session_state(session_id, SessionState.ABANDONED)
        await self.events.emit(
            SessionEvent(
                session_id=session_id,
                conversation_id=session.conversation_id,
                timestamp=_utcnow(),
                type="abandoned",
                payload={"reason": reason},
            )
        )
        if self.livekit is not None:
            from interviewer.voice.livekit_entry import delete_room

            await delete_room(self.livekit, session_id)

    # ---- reads -----------------------------------------------------------

    async def get_session_status(self, session_id: str) -> SessionStatus:
        """Cheap dashboard read. May be one turn stale vs the live agent."""
        session = await self.store.load_session(session_id)
        turns = await self.store.list_turns(session_id)
        runtime = await self.store.load_runtime_state(session_id)
        extract = await self.store.load_extract(session_id)

        goals_total = len(session.conversation_snapshot.goals)
        if extract is not None:
            goals_resolved = sum(
                1 for gs in extract.goal_statuses if gs.status != "pending"
            )
        else:
            # Pre-completion best-effort: count goals touched by any Turn.
            touched: set[str] = set()
            for turn in turns:
                touched.update(turn.addressed_goal_ids)
            goals_resolved = len(touched)

        active_goal_id = runtime.active_goal_id if runtime is not None else None
        last_turn_at = turns[-1].timestamp if turns else None

        return SessionStatus(
            session_id=session_id,
            state=session.state,
            active_goal_id=active_goal_id,
            total_turns=len(turns),
            goals_resolved=goals_resolved,
            goals_total=goals_total,
            started_at=session.started_at,
            last_turn_at=last_turn_at,
        )

    async def get_transcript(self, session_id: str) -> list[Turn]:
        """Full transcript. Progressive while IN_PROGRESS, final when COMPLETED."""
        return await self.store.list_turns(session_id)

    async def get_extract(self, session_id: str) -> Extract | None:
        """Structured output. Returns None until state is COMPLETED."""
        return await self.store.load_extract(session_id)

    # ---- stubs (filled in later steps) -----------------------------------

    async def entrypoint(self, ctx: object) -> None:
        """LiveKit AgentSession entrypoint — consumer registers this with their AgentServer.

        Resolves ``session_id`` from the job metadata (or room name),
        loads the snapshotted Conversation + any persisted runtime
        state, builds an ``AgentSession`` around ``InterviewerLLM``, and
        runs until the room disconnects.
        """
        # Local import keeps the optional LiveKit deps out of the
        # simulator-only install path.
        from interviewer.voice.livekit_entry import run_entrypoint

        await run_entrypoint(self, ctx)

    async def simulate_session(
        self, conversation_id: str, simulator: RespondentSimulator
    ) -> Extract:
        """Run the loop against a synthetic respondent — no voice room.

        Provisions a fresh Session (emits ``session_provisioned``), then
        drives ``run_loop`` to completion. Useful for rehearsing a
        Conversation against ``ScriptedSimulator``/``TerseEvasiveSimulator``
        et al. before sending the link to a real human.
        """
        # Local import to keep engine ↔ runner module-load acyclic.
        from interviewer.loop.runner import run_loop

        session, _creds = await self.provision_session(conversation_id)
        return await run_loop(self, session.id, simulator)

    # ---- internals -------------------------------------------------------

    def _mint_credentials(
        self, session_id: str, now: datetime
    ) -> SessionCredentials:
        """Generate join credentials.

        When ``livekit`` is set, mint a real LiveKit JWT for the room
        ``iv:{session_id}`` with identity ``respondent:{session_id}``
        (24-h TTL, D9). Simulator-only deployments get stub credentials
        with a sentinel URL so consumer code can still flow.
        """
        room_name = f"iv:{session_id}"
        if self.livekit is not None:
            from interviewer.voice.livekit_entry import mint_join_token

            token, expires_at = mint_join_token(
                self.livekit, session_id, ttl=_TOKEN_TTL
            )
            return SessionCredentials(
                room_url=self.livekit.url,
                token=token,
                expires_at=expires_at,
            )
        token = f"stub-token-{session_id}-{uuid.uuid4().hex[:8]}"
        return SessionCredentials(
            room_url=f"stub://room/{room_name}",
            token=token,
            expires_at=now + _TOKEN_TTL,
        )


def _utcnow() -> datetime:
    return datetime.now(UTC)


def _new_id(prefix: str) -> str:
    return f"{prefix}-{uuid.uuid4().hex}"
