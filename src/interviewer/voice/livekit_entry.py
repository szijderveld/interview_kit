"""LiveKit AgentSession integration — Step 13.

Wires our :class:`LLMClient` into a ``livekit-agents`` :class:`AgentSession`
via the :class:`InterviewerLLM` subclass. ``AgentSession`` invokes
``InterviewerLLM.chat()`` once per respondent turn after STT finalizes a
transcript; the chat() body owns the full per-turn flow:

1. read the latest user message off ``chat_ctx`` (the respondent's
   transcribed utterance), append a respondent :class:`Turn` to the
   store, emit ``turn_recorded``;
2. build a :class:`TurnContext` from the per-session state;
3. detect refusal/IDK with the same heuristic the simulator runner uses,
   short-circuiting eval on the second consecutive refusal;
4. otherwise call ``LLMClient.evaluate_turn``;
5. apply the :class:`EvalResult` to the goal_status_table (D13);
6. ``select_next_goal`` against the updated table — terminal None / a
   ``close`` eval / the turn cap yields the closing utterance and marks
   the session done;
7. accumulate ``compose_utterance`` into a single text, validate, regen
   once on phrasing failure (D7), then push one or more
   :class:`ChatChunk`\\ s into the LLMStream so AgentSession can stream
   to TTS;
8. flush :class:`SessionRuntimeState` (D9) and append the agent Turn.

Per-session state lives on :class:`PerSessionState`, attached to each
:class:`InterviewerLLM` instance. ``entrypoint`` constructs one instance
per session — never reuse across sessions.

Opening trigger. ``entrypoint`` calls ``session.generate_reply(...)``
after ``session.start(...)``; this triggers a chat() call with no
respondent input. The first chat() call detects ``opening_done is
False`` and yields :data:`RESUME_ACK` (if runtime state was rehydrated),
the configured ``conversation.opening``, or a composed default. The
respondent-turn flow runs on every subsequent chat() invocation.

Phrasing regen in voice mode (D7 divergence). The simulator runner
streams compose chunks, validates after the stream exhausts, and
regens once on failure. In voice mode we accumulate first, validate,
regen once if needed, then push one ``ChatChunk`` containing the final
text — otherwise the respondent would hear both attempts spoken aloud.
See DECISIONS Step 13.

Cancel and disconnect (D4). ``cancel_session`` writes ABANDONED to the
store and deletes the LiveKit room; AgentSession exits on disconnect
and ``entrypoint`` finalises depending on the observed terminal state.
The chat() body also short-circuits if it observes ABANDONED at the
top of an iteration.
"""

from __future__ import annotations

import asyncio
import uuid
from collections.abc import AsyncIterator
from dataclasses import dataclass, field
from datetime import UTC, datetime, timedelta
from typing import TYPE_CHECKING, Any, cast

from livekit import api as lkapi
from livekit.agents import Agent, AgentSession
from livekit.agents.llm import LLM, ChatChunk, ChatContext, ChoiceDelta, LLMStream
from livekit.agents.types import (
    DEFAULT_API_CONNECT_OPTIONS,
    NOT_GIVEN,
    APIConnectOptions,
    NotGivenOr,
)

from interviewer.livekit_config import LiveKitConfig
from interviewer.loop.extract import derive_extract_with_llm
from interviewer.loop.heuristics import detect_refusal_or_idk
from interviewer.loop.phrasing import validate_voice_phrasing
from interviewer.loop.resume import RESUME_ACK
from interviewer.loop.selection import select_next_goal
from interviewer.protocols import ConversationStore, EventSink, LLMClient
from interviewer.types.config import Conversation, Goal
from interviewer.types.events import SessionEvent, SessionEventType
from interviewer.types.runtime import (
    EvalResult,
    GoalStatus,
    SessionRuntimeState,
    Turn,
    TurnContext,
)
from interviewer.types.state import SessionState

if TYPE_CHECKING:
    from interviewer.engine import Engine


DEFAULT_OPENING = "Hi — thanks for taking the time. Mind if I jump in?"
DEFAULT_CLOSING = "Thanks for your time — that's everything I needed."
CANCEL_CLOSING = "Looks like we need to wrap. Thanks for the time you gave."
APOLOGY = "I'm sorry — something on my end isn't working. Let's pause for now."

# Mirror the runner's telemetry shape (D11). Voice path emits the same
# fields on ``turn_recorded``.
_USAGE_KEYS: tuple[str, ...] = (
    "tokens_in",
    "tokens_out",
    "cache_read_tokens",
    "cache_write_tokens",
    "llm_latency_ms",
)
_ZERO_USAGE: dict[str, int] = dict.fromkeys(_USAGE_KEYS, 0)


def _read_usage(llm: LLMClient, attr: str) -> dict[str, int]:
    """Read ``llm.<attr>`` if surfaced; else zeroed Usage (D11 / Step 12)."""
    usage = getattr(llm, attr, None)
    if usage is None:
        return dict(_ZERO_USAGE)
    return {
        "tokens_in": int(getattr(usage, "tokens_in", 0) or 0),
        "tokens_out": int(getattr(usage, "tokens_out", 0) or 0),
        "cache_read_tokens": int(getattr(usage, "cache_read_tokens", 0) or 0),
        "cache_write_tokens": int(getattr(usage, "cache_write_tokens", 0) or 0),
        "llm_latency_ms": int(getattr(usage, "llm_latency_ms", 0) or 0),
    }


@dataclass
class PerSessionState:
    """Per-session mutable state held by :class:`InterviewerLLM`.

    One instance per session — never shared. Mirrors the runner's
    ``_RunnerState`` shape so the voice and simulator paths derive
    identical loop-time hints.
    """

    session_id: str
    conversation_snapshot: Conversation
    store: ConversationStore
    events: EventSink
    llm_client: LLMClient
    goal_status_table: dict[str, GoalStatus]
    total_turns: int = 0
    retries_used_on_active: int = 0
    tangent_followups_used: int = 0
    refusal_count_on_active: int = 0
    last_active: Goal | None = None
    last_eval: EvalResult | None = None
    eval_usage_totals: dict[str, int] = field(default_factory=lambda: dict(_ZERO_USAGE))
    opening_done: bool = False
    # Set when the chat() body decides this session is terminal — entrypoint
    # awaits this to know when to tear down.
    done: bool = False
    done_event: asyncio.Event = field(default_factory=asyncio.Event)
    # ``True`` if we already played RESUME_ACK on entry (vs. a fresh opening).
    resumed: bool = False


def initial_status_table(conv: Conversation) -> dict[str, GoalStatus]:
    """Build the initial loop-time goal_status_table — all goals ``pending``."""
    return {g.id: GoalStatus(goal_id=g.id, status="pending") for g in conv.goals}


def rehydrate_state(
    state: PerSessionState,
    runtime: SessionRuntimeState,
    existing_turns: list[Turn],
) -> None:
    """Apply persisted runtime state + transcript hints to ``state`` (D9, Step 10).

    Mirrors :func:`interviewer.loop.runner._resume_bootstrap`: counters
    come from the runtime row; goal_status_table is reconstructed from
    every ``Turn.addressed_goal_ids`` as a ``meets`` hint. ``opening_done``
    is set so the first ``chat()`` call speaks :data:`RESUME_ACK`
    instead of an opening.
    """
    state.total_turns = len(existing_turns)
    state.retries_used_on_active = runtime.retries_used_on_active
    state.tangent_followups_used = runtime.tangent_followups_used
    addressed_so_far: set[str] = set()
    for turn in existing_turns:
        addressed_so_far.update(turn.addressed_goal_ids)
    for goal_id in addressed_so_far:
        if goal_id in state.goal_status_table:
            current = state.goal_status_table[goal_id]
            state.goal_status_table[goal_id] = current.model_copy(
                update={
                    "status": "meets",
                    "rationale": "resumed: prior coverage in transcript",
                }
            )
    state.resumed = True


class InterviewerAgent(Agent):
    """Minimal ``livekit-agents`` :class:`Agent` shim.

    All interview logic runs inside :class:`InterviewerLLM.chat`; the
    Agent's ``instructions`` are intentionally empty.
    """

    def __init__(self) -> None:
        super().__init__(instructions="")


class InterviewerLLM(LLM):
    """``livekit-agents`` LLM subclass — delegates to our :class:`LLMClient`.

    AgentSession calls :meth:`chat` once per respondent turn (after the
    STT finalizes). The returned :class:`LLMStream` yields :class:`ChatChunk`
    objects whose ``delta.content`` is the agent's next utterance — the
    framework forwards them to TTS.
    """

    def __init__(self, state: PerSessionState) -> None:
        super().__init__()
        self.state = state

    @property
    def model(self) -> str:
        return "interviewer-engine"

    @property
    def provider(self) -> str:
        return "interviewer"

    def chat(
        self,
        *,
        chat_ctx: ChatContext,
        tools: list[Any] | None = None,
        conn_options: APIConnectOptions = DEFAULT_API_CONNECT_OPTIONS,
        parallel_tool_calls: NotGivenOr[bool] = NOT_GIVEN,
        tool_choice: NotGivenOr[Any] = NOT_GIVEN,
        extra_kwargs: NotGivenOr[dict[str, Any]] = NOT_GIVEN,
    ) -> LLMStream:
        return _InterviewerStream(
            self, chat_ctx=chat_ctx, tools=tools or [], conn_options=conn_options
        )


class _InterviewerStream(LLMStream):
    """Single-turn stream — runs the engine flow inside ``_run``.

    The base class spawns a background task that invokes ``_run()``;
    chunks pushed onto ``self._event_ch`` propagate to the AgentSession's
    consumer (TTS). The channel closes automatically when ``_run``
    returns.
    """

    def __init__(
        self,
        llm: InterviewerLLM,
        *,
        chat_ctx: ChatContext,
        tools: list[Any],
        conn_options: APIConnectOptions,
    ) -> None:
        super().__init__(
            llm, chat_ctx=chat_ctx, tools=tools, conn_options=conn_options
        )
        # Convenience handle so _run() doesn't repeatedly cast self._llm.
        self._interviewer = llm

    async def _run(self) -> None:
        state = self._interviewer.state

        # Cancel-mid-flight observation (D4). If the operator cancelled
        # between turns the room may not yet have torn down — bail before
        # we speak anything else.
        session = await state.store.load_session(state.session_id)
        if session.state == SessionState.ABANDONED:
            await _push_text(self, CANCEL_CLOSING)
            await _record_agent_turn(state, CANCEL_CLOSING, addressed=[])
            state.done = True
            state.done_event.set()
            return

        if not state.opening_done:
            await self._handle_opening()
            state.opening_done = True
            return

        respondent_text = _extract_latest_user_text(self._chat_ctx)
        if respondent_text is None:
            # No user utterance present — likely a synthetic re-prompt
            # before the respondent spoke. Treat as a no-op so we don't
            # double-speak.
            return
        await _record_respondent_turn(state, respondent_text, addressed=[])

        # Terminal: turn cap reached → speak closing and finish.
        if state.total_turns >= state.conversation_snapshot.max_total_turns:
            await self._terminate(reason="max_total_turns")
            return

        # Refusal/IDK detection — mirrors runner's branch in
        # ``run_loop``. Two consecutive refusals on the same goal mark
        # the active goal ``gave_up`` and advance; one refusal triggers
        # a deflection probe.
        deflection = False
        if state.last_active is not None:
            if detect_refusal_or_idk(respondent_text):
                state.refusal_count_on_active += 1
                if state.refusal_count_on_active >= 2:
                    state.last_eval = EvalResult(
                        active_goal_status="gave_up",
                        next_action="advance",
                        rationale="two consecutive refusals/IDK on this goal",
                    )
                    _apply_eval(state, state.last_active, state.last_eval)
                    state.last_active = None
                else:
                    state.last_eval = EvalResult(
                        active_goal_status="partial",
                        next_action="retry",
                        rationale="refusal/IDK — sending deflection probe",
                    )
                    _apply_eval(state, state.last_active, state.last_eval)
                    deflection = True
            else:
                state.refusal_count_on_active = 0
                transcript = await state.store.list_turns(state.session_id)
                eval_ctx = _build_ctx(state, transcript, state.last_active)
                try:
                    state.last_eval = await state.llm_client.evaluate_turn(eval_ctx)
                except Exception:
                    await self._terminate(reason="llm_evaluate_failure")
                    return
                _accumulate_eval_usage(state)
                _apply_eval(state, state.last_active, state.last_eval)
                if state.last_eval.next_action == "close":
                    await self._terminate(reason="eval_close")
                    return

        # Goal selection (deflection re-uses the prior active goal).
        if deflection and state.last_active is not None:
            active = state.last_active
        else:
            candidate = select_next_goal(
                state.conversation_snapshot,
                list(state.goal_status_table.values()),
            )
            if candidate is None:
                await self._terminate(reason="all_goals_resolved")
                return
            active = candidate
            if state.last_active is None or active.id != state.last_active.id:
                state.retries_used_on_active = 0
                state.refusal_count_on_active = 0

        # Compose — accumulate, validate, regen once on phrasing failure.
        # Voice-mode divergence from the runner (D7): we yield the final
        # text in one chunk rather than streaming both attempts, to
        # avoid speaking the failed candidate out loud.
        transcript = await state.store.list_turns(state.session_id)
        compose_ctx = _build_ctx(state, transcript, active)
        compose_eval = state.last_eval or _placeholder_eval()
        try:
            text = await _compose_full_text(
                state.llm_client, compose_ctx, compose_eval
            )
        except Exception:
            await self._terminate(reason="llm_compose_failure")
            return

        await _push_text(self, text)
        await _record_agent_turn(
            state,
            text,
            addressed=[active.id],
            active_goal_id=active.id,
            with_compose_usage=True,
        )
        state.last_active = active

    async def _handle_opening(self) -> None:
        state = self._interviewer.state
        # Update session state to IN_PROGRESS the first time we speak.
        await state.store.update_session_state(
            state.session_id, SessionState.IN_PROGRESS
        )
        await _emit(
            state,
            "respondent_joined",
            {"simulator": None},
        )
        if state.resumed:
            text = RESUME_ACK
        elif state.conversation_snapshot.opening:
            text = state.conversation_snapshot.opening
        else:
            text = DEFAULT_OPENING
        await _push_text(self, text)
        await _record_agent_turn(state, text, addressed=[])

    async def _terminate(self, *, reason: str) -> None:
        """Speak closing, set state.done, signal the entrypoint."""
        state = self._interviewer.state
        if reason == "llm_evaluate_failure" or reason == "llm_compose_failure":
            closing = APOLOGY
            await _push_text(self, closing)
            await _record_agent_turn(state, closing, addressed=[])
            await state.store.update_session_state(
                state.session_id, SessionState.FAILED
            )
            await _emit(state, "failed", {"reason": reason})
        else:
            closing = state.conversation_snapshot.closing or DEFAULT_CLOSING
            await _push_text(self, closing)
            await _record_agent_turn(state, closing, addressed=[])
            await state.store.update_session_state(
                state.session_id, SessionState.COMPLETED
            )
        state.done = True
        state.done_event.set()


def _placeholder_eval() -> EvalResult:
    return EvalResult(
        active_goal_status="pending",
        redundant_goal_ids=[],
        interesting_tangent=None,
        next_action="advance",
        rationale="",
    )


def _build_ctx(
    state: PerSessionState,
    transcript: list[Turn],
    active: Goal,
) -> TurnContext:
    return TurnContext(
        conversation=state.conversation_snapshot,
        transcript=transcript,
        active_goal=active,
        goal_statuses=list(state.goal_status_table.values()),
        retries_used_on_active=state.retries_used_on_active,
        tangent_followups_used=state.tangent_followups_used,
        total_turns=state.total_turns,
        last_phrasing_failure=None,
    )


def _apply_eval(
    state: PerSessionState, active: Goal, eval_result: EvalResult
) -> None:
    current = state.goal_status_table[active.id]
    state.goal_status_table[active.id] = current.model_copy(
        update={
            "status": eval_result.active_goal_status,
            "rationale": eval_result.rationale,
            "retries_used": state.retries_used_on_active,
        }
    )
    for gid in eval_result.redundant_goal_ids:
        if gid in state.goal_status_table:
            gs = state.goal_status_table[gid]
            state.goal_status_table[gid] = gs.model_copy(
                update={
                    "status": "skipped_redundant",
                    "rationale": "redundant per evaluate_turn",
                }
            )
    if eval_result.next_action == "retry":
        state.retries_used_on_active += 1
    elif eval_result.next_action == "drill":
        state.tangent_followups_used += 1


def _accumulate_eval_usage(state: PerSessionState) -> None:
    delta = _read_usage(state.llm_client, "last_eval_usage")
    for key in _USAGE_KEYS:
        state.eval_usage_totals[key] += delta[key]


async def _compose_full_text(
    llm: LLMClient, ctx: TurnContext, eval_result: EvalResult
) -> str:
    """Accumulate compose stream into a single string; regen once on phrasing failure.

    Voice-mode divergence (D7): the simulator runner streams chunks to
    its caller and validates after the stream exhausts, accepting that a
    failed first attempt is spoken verbatim because it has nowhere else
    to stream the corrected text. In voice mode we accumulate first so
    a regen does not double-speak.
    """
    text = await _accumulate(llm.compose_utterance(ctx, eval_result))
    failures = validate_voice_phrasing(text)
    if failures:
        regen_ctx = ctx.model_copy(
            update={
                "last_phrasing_failure": ",".join(f.value for f in failures),
            }
        )
        text = await _accumulate(llm.compose_utterance(regen_ctx, eval_result))
        # D7: speak verbatim if the second attempt also fails — no third.
    return text


async def _accumulate(stream: AsyncIterator[str]) -> str:
    chunks: list[str] = []
    async for chunk in stream:
        chunks.append(chunk)
    return "".join(chunks)


def _extract_latest_user_text(chat_ctx: ChatContext) -> str | None:
    """Return the most recent user-role text from ``chat_ctx``, or None.

    AgentSession appends the STT-finalised transcript as a user-role
    :class:`ChatMessage` before invoking ``chat()``. ``content`` is a
    list of ``ChatContent`` items (str, ImageContent, AudioContent,
    Instructions) — we only consume string parts.
    """
    for item in reversed(list(chat_ctx.items)):
        role = getattr(item, "role", None)
        if role != "user":
            continue
        content = getattr(item, "content", None)
        if not content:
            continue
        parts: list[str] = []
        for part in content:
            if isinstance(part, str):
                parts.append(part)
        text = " ".join(parts).strip()
        if text:
            return text
        return None
    return None


async def _push_text(stream: LLMStream, text: str) -> None:
    """Push ``text`` as a single :class:`ChatChunk` onto the stream channel."""
    chunk = ChatChunk(
        id=uuid.uuid4().hex,
        delta=ChoiceDelta(role="assistant", content=text),
    )
    # ``_event_ch`` is created in the LLMStream base ``__init__``.
    stream._event_ch.send_nowait(chunk)


async def _record_agent_turn(
    state: PerSessionState,
    text: str,
    *,
    addressed: list[str],
    active_goal_id: str | None = None,
    with_compose_usage: bool = False,
) -> None:
    """Flush runtime state, append the agent Turn, emit ``turn_recorded``."""
    await state.store.save_runtime_state(
        SessionRuntimeState(
            session_id=state.session_id,
            active_goal_id=active_goal_id,
            retries_used_on_active=state.retries_used_on_active,
            tangent_followups_used=state.tangent_followups_used,
            total_turns=state.total_turns,
            pending_follow_up=None,
            last_event_index=max(state.total_turns - 1, 0),
            updated_at=_utcnow(),
        )
    )
    turn = Turn(
        index=state.total_turns,
        speaker="agent",
        text=text,
        timestamp=_utcnow(),
        addressed_goal_ids=list(addressed),
    )
    await state.store.append_turn(state.session_id, turn)
    state.total_turns += 1
    usage = (
        _read_usage(state.llm_client, "last_compose_usage")
        if with_compose_usage
        else dict(_ZERO_USAGE)
    )
    await _emit(
        state,
        "turn_recorded",
        {"index": turn.index, "speaker": "agent", "text": text, **usage},
    )


async def _record_respondent_turn(
    state: PerSessionState, text: str, *, addressed: list[str]
) -> None:
    turn = Turn(
        index=state.total_turns,
        speaker="respondent",
        text=text,
        timestamp=_utcnow(),
        addressed_goal_ids=list(addressed),
    )
    await state.store.append_turn(state.session_id, turn)
    state.total_turns += 1
    await _emit(
        state,
        "turn_recorded",
        {"index": turn.index, "speaker": "respondent", "text": text},
    )


async def _emit(
    state: PerSessionState,
    type_: SessionEventType,
    payload: dict[str, Any],
) -> None:
    await state.events.emit(
        SessionEvent(
            session_id=state.session_id,
            conversation_id=state.conversation_snapshot.id,
            timestamp=_utcnow(),
            type=type_,
            payload=payload,
        )
    )


def _utcnow() -> datetime:
    return datetime.now(UTC)


# ---- AgentSession + entrypoint plumbing ----------------------------------


def build_agent_session(state: PerSessionState) -> AgentSession:
    """Wire Deepgram STT + Cartesia TTS + Silero VAD + InterviewerLLM.

    Plugin imports are lazy so ``import interviewer.voice.livekit_entry``
    succeeds in environments where the optional ``[deepgram,cartesia,silero]``
    plugins were not installed; only callers that actually need a live
    AgentSession pay the import cost.
    """
    from livekit.plugins import cartesia, deepgram, silero

    stt = deepgram.STT(model="nova-3", language="en-US")
    tts = cartesia.TTS(
        model="sonic-2",
        voice=state.conversation_snapshot.persona.voice_id,
    )
    vad = silero.VAD.load()
    return AgentSession(stt=stt, llm=InterviewerLLM(state=state), tts=tts, vad=vad)


def mint_join_token(
    livekit: LiveKitConfig, session_id: str, *, ttl: timedelta = timedelta(hours=24)
) -> tuple[str, datetime]:
    """Mint a respondent join token for ``iv:{session_id}``.

    Identity prefix ``respondent:`` is part of the contract — consumer
    dashboards may filter participants by it. Token TTL is 24 h by
    default (D9).
    """
    now = _utcnow()
    token = (
        lkapi.AccessToken(livekit.api_key, livekit.api_secret)
        .with_identity(f"respondent:{session_id}")
        .with_name("respondent")
        .with_grants(
            lkapi.VideoGrants(
                room_join=True,
                room=f"iv:{session_id}",
                can_publish=True,
                can_subscribe=True,
            )
        )
        .with_ttl(ttl)
        .to_jwt()
    )
    return cast(str, token), now + ttl


async def delete_room(livekit: LiveKitConfig, session_id: str) -> None:
    """Delete the room for ``session_id`` via the LiveKit server API."""
    api = lkapi.LiveKitAPI(
        url=livekit.url,
        api_key=livekit.api_key,
        api_secret=livekit.api_secret,
    )
    try:
        await api.room.delete_room(lkapi.DeleteRoomRequest(room=f"iv:{session_id}"))
    finally:
        await api.aclose()


async def run_entrypoint(engine: Engine, ctx: Any) -> None:
    """``Engine.entrypoint(ctx)`` body — invoked by the LiveKit AgentServer.

    ``ctx`` is a ``livekit.agents.JobContext``; kept loosely typed so this
    module remains importable in test contexts where a real JobContext is
    not available.
    """
    # Resolve session id from job metadata; fall back to the room name.
    job_metadata = getattr(getattr(ctx, "job", None), "metadata", None) or ""
    if job_metadata:
        session_id = job_metadata
    else:
        room_name = getattr(getattr(ctx, "room", None), "name", "") or ""
        session_id = room_name.removeprefix("iv:")
    if not session_id:
        raise RuntimeError("entrypoint: could not resolve session_id from JobContext")

    await ctx.connect()
    session = await engine.store.load_session(session_id)
    runtime = await engine.store.load_runtime_state(session_id)
    conv = session.conversation_snapshot

    state = PerSessionState(
        session_id=session_id,
        conversation_snapshot=conv,
        store=engine.store,
        events=engine.events,
        llm_client=engine.llm,
        goal_status_table=initial_status_table(conv),
    )
    if runtime is not None:
        existing = await engine.store.list_turns(session_id)
        rehydrate_state(state, runtime, existing)

    agent_session = build_agent_session(state)
    await agent_session.start(room=ctx.room, agent=InterviewerAgent())

    instructions = (
        "resume — speak the resume acknowledgement"
        if state.resumed
        else "deliver the opening and ask the first goal's question"
    )
    agent_session.generate_reply(instructions=instructions)

    # Wait for either the chat() body to terminate the session or the
    # respondent to drop. ``ctx.room.on("disconnected", ...)`` is the
    # current livekit-agents pattern for room teardown notification.
    disconnect_event = asyncio.Event()

    def _on_disconnect(*_args: object) -> None:
        disconnect_event.set()

    ctx.room.on("disconnected", _on_disconnect)

    done_wait = asyncio.create_task(state.done_event.wait())
    disc_wait = asyncio.create_task(disconnect_event.wait())
    await asyncio.wait(
        {done_wait, disc_wait}, return_when=asyncio.FIRST_COMPLETED
    )
    for task in (done_wait, disc_wait):
        if not task.done():
            task.cancel()

    final = await engine.store.load_session(session_id)
    if final.state == SessionState.COMPLETED:
        await _finalize_extract(engine, state)
    elif final.state in (SessionState.IN_PROGRESS, SessionState.READY):
        # Respondent disconnected before terminal — mark ABANDONED.
        await engine.store.update_session_state(
            session_id, SessionState.ABANDONED
        )
        await _emit(state, "abandoned", {"reason": "respondent_disconnect"})

    await agent_session.aclose()


async def _finalize_extract(engine: Engine, state: PerSessionState) -> None:
    """Run the canonical derive_extract pass and emit ``completed`` (Step 11)."""
    transcript = await state.store.list_turns(state.session_id)
    hint_snapshot: dict[str, GoalStatus] = dict(state.goal_status_table)
    raw = await derive_extract_with_llm(transcript, state.conversation_snapshot, engine.llm)
    now = _utcnow()
    extract = raw.model_copy(
        update={"session_id": state.session_id, "completed_at": now}
    )
    await state.store.save_extract(extract)

    canonical_by_id = {gs.goal_id: gs for gs in extract.goal_statuses}
    for goal in state.conversation_snapshot.goals:
        canonical = canonical_by_id.get(goal.id)
        if canonical is None:
            continue
        prior = hint_snapshot.get(goal.id)
        prior_status = prior.status if prior is not None else None
        if prior_status == canonical.status:
            continue
        await _emit(
            state,
            "goal_status_changed",
            {
                "goal_id": goal.id,
                "from_status": prior_status,
                "to_status": canonical.status,
                "rationale": canonical.rationale,
            },
        )
    await _emit(
        state,
        "completed",
        {
            "goal_statuses": [gs.model_dump() for gs in extract.goal_statuses],
            "total_turns": state.total_turns,
            "eval_usage_totals": dict(state.eval_usage_totals),
        },
    )
