"""Voice integration tests — Step 13.

Three focal points (per PLAN):

- ``provision_session`` mints a real LiveKit JWT for ``iv:{session_id}``
  with the right identity and TTL when ``LiveKitConfig`` is supplied.
- ``cancel_session`` writes ABANDONED and deletes the room over the
  LiveKit server API.
- ``InterviewerLLM.chat()`` walks the opening + respondent-turn flow
  cleanly against a ``FakeLLMClient`` — the same code path the live
  AgentSession invokes per turn.
"""

from __future__ import annotations

from collections.abc import Iterator
from datetime import timedelta

import jwt
import pytest
from livekit.agents.llm import ChatContext

from interviewer import (
    Background,
    Conversation,
    Engine,
    Goal,
    LiveKitConfig,
    Persona,
    SessionState,
)
from interviewer.sinks.memory import InMemoryEventSink
from interviewer.stores.memory import InMemoryConversationStore
from interviewer.testing.fake_llm import FakeLLMClient
from interviewer.types.runtime import EvalResult
from interviewer.voice.livekit_entry import (
    InterviewerLLM,
    PerSessionState,
    initial_status_table,
)

API_KEY = "test-key-at-least-thirty-two-bytes-long"
API_SECRET = "test-secret-at-least-thirty-two-bytes-long-for-jwt"


def _persona() -> Persona:
    return Persona(
        system_prompt="discovery interviewer",
        style="neutral",
        voice_id="voice-x",
    )


def _conv() -> Conversation:
    return Conversation(
        id="conv-1",
        persona=_persona(),
        purpose="map a typical week",
        background=Background(
            interviewee_role="staff engineer",
            interviewee_expertise="pipeline ownership",
        ),
        goals=[
            Goal(id="g1", intent="rituals", standard="two named"),
            Goal(id="g2", intent="exceptions", standard="one named"),
        ],
        opening="thanks for jumping on",
        closing="that's all I need.",
    )


def _livekit_config() -> LiveKitConfig:
    return LiveKitConfig(
        url="wss://livekit.example.com",
        api_key=API_KEY,
        api_secret=API_SECRET,
        agent_name="interviewer",
    )


# ---------- provision_session token minting -------------------------------


async def test_provision_session_mints_real_jwt() -> None:
    engine = Engine(
        store=InMemoryConversationStore(),
        events=InMemoryEventSink(),
        llm=FakeLLMClient(),
        livekit=_livekit_config(),
    )
    await engine.store.save_conversation(_conv())
    session, creds = await engine.provision_session("conv-1")

    assert creds.room_url == "wss://livekit.example.com"
    # Decode without signature verification to inspect the claims —
    # we already trust the SDK to produce a valid signature.
    claims = jwt.decode(creds.token, options={"verify_signature": False})
    assert claims["sub"] == f"respondent:{session.id}"
    assert claims["name"] == "respondent"
    assert claims["iss"] == API_KEY
    assert claims["video"]["room"] == f"iv:{session.id}"
    assert claims["video"]["roomJoin"] is True
    # TTL is 24 h from now (D9); allow a generous window for clock skew.
    ttl_seconds = claims["exp"] - claims["nbf"]
    assert abs(ttl_seconds - int(timedelta(hours=24).total_seconds())) < 60


# ---------- cancel_session room deletion ----------------------------------


class _FakeRoomService:
    def __init__(self) -> None:
        self.deleted: list[str] = []

    async def delete_room(self, req: object) -> None:
        # ``req`` is a ``DeleteRoomRequest`` proto; the ``room`` field is
        # the room name.
        self.deleted.append(req.room)


class _FakeLiveKitAPI:
    last_init_kwargs: dict[str, object] | None = None
    last_instance: _FakeLiveKitAPI | None = None

    def __init__(self, **kwargs: object) -> None:
        type(self).last_init_kwargs = kwargs
        self.room = _FakeRoomService()
        type(self).last_instance = self
        self.closed = False

    async def aclose(self) -> None:
        self.closed = True


@pytest.fixture
def fake_livekit_api(monkeypatch: pytest.MonkeyPatch) -> Iterator[type[_FakeLiveKitAPI]]:
    """Patch ``livekit.api.LiveKitAPI`` for the cancel-path test."""
    import livekit.api as lkapi

    _FakeLiveKitAPI.last_init_kwargs = None
    _FakeLiveKitAPI.last_instance = None
    monkeypatch.setattr(lkapi, "LiveKitAPI", _FakeLiveKitAPI)
    yield _FakeLiveKitAPI


async def test_cancel_session_deletes_livekit_room(
    fake_livekit_api: type[_FakeLiveKitAPI],
) -> None:
    engine = Engine(
        store=InMemoryConversationStore(),
        events=InMemoryEventSink(),
        llm=FakeLLMClient(),
        livekit=_livekit_config(),
    )
    await engine.store.save_conversation(_conv())
    session, _creds = await engine.provision_session("conv-1")

    await engine.cancel_session(session.id, reason="operator")

    persisted = await engine.store.load_session(session.id)
    assert persisted.state is SessionState.ABANDONED
    instance = fake_livekit_api.last_instance
    assert instance is not None
    assert instance.room.deleted == [f"iv:{session.id}"]
    assert instance.closed is True


async def test_cancel_session_skips_room_delete_when_livekit_none() -> None:
    engine = Engine(
        store=InMemoryConversationStore(),
        events=InMemoryEventSink(),
        llm=FakeLLMClient(),
    )
    await engine.store.save_conversation(_conv())
    session, _ = await engine.provision_session("conv-1")
    # No livekit config → cancel_session must still complete cleanly,
    # writing ABANDONED without touching the LiveKit server API.
    await engine.cancel_session(session.id)
    persisted = await engine.store.load_session(session.id)
    assert persisted.state is SessionState.ABANDONED


# ---------- InterviewerLLM.chat() happy path ------------------------------


async def _bootstrap_state() -> PerSessionState:
    """Provision a session and return a PerSessionState ready for chat()."""
    store = InMemoryConversationStore()
    events = InMemoryEventSink()
    conv = _conv()
    await store.save_conversation(conv)
    # Mirror Engine.provision_session: save a Session row so chat()'s
    # ABANDONED-check load_session() succeeds.
    from datetime import UTC, datetime

    from interviewer.types.runtime import Session as SessionRow

    session = SessionRow(
        id="sess-1",
        conversation_id=conv.id,
        conversation_snapshot=conv,
        state=SessionState.READY,
        created_at=datetime.now(UTC),
    )
    await store.save_session(session)

    llm = FakeLLMClient(
        eval_results=[
            EvalResult(
                active_goal_status="meets",
                next_action="advance",
                rationale="ritual named",
            ),
        ],
        utterances=["what does a typical morning look like?"],
    )
    return PerSessionState(
        session_id=session.id,
        conversation_snapshot=conv,
        store=store,
        events=events,
        llm_client=llm,
        goal_status_table=initial_status_table(conv),
    )


async def test_interviewer_llm_chat_opening_yields_scripted_opening() -> None:
    state = await _bootstrap_state()
    llm = InterviewerLLM(state=state)

    # First chat() call: opening trigger — no user message in chat_ctx.
    stream = llm.chat(chat_ctx=ChatContext.empty())
    chunks: list[str] = []
    async for chunk in stream:
        if chunk.delta and chunk.delta.content:
            chunks.append(chunk.delta.content)
    await stream.aclose()
    text = "".join(chunks)
    assert text == "thanks for jumping on"
    # Opening is recorded as an agent Turn and the session flipped to IN_PROGRESS.
    turns = await state.store.list_turns(state.session_id)
    assert len(turns) == 1
    assert turns[0].speaker == "agent"
    assert turns[0].text == "thanks for jumping on"
    session = await state.store.load_session(state.session_id)
    assert session.state is SessionState.IN_PROGRESS
    assert state.opening_done is True


async def test_interviewer_llm_chat_respondent_turn_does_eval_compose_record() -> None:
    state = await _bootstrap_state()
    llm = InterviewerLLM(state=state)

    # Opening to advance opening_done.
    stream = llm.chat(chat_ctx=ChatContext.empty())
    async for _ in stream:
        pass
    await stream.aclose()

    # Respondent turn: chat_ctx has the new user message.
    ctx = ChatContext.empty()
    ctx.add_message(role="user", content="i'm a staff engineer")
    stream = llm.chat(chat_ctx=ctx)
    chunks: list[str] = []
    async for chunk in stream:
        if chunk.delta and chunk.delta.content:
            chunks.append(chunk.delta.content)
    await stream.aclose()
    text = "".join(chunks)
    assert text == "what does a typical morning look like?"

    turns = await state.store.list_turns(state.session_id)
    # opening (agent) + respondent + probe (agent) = 3 turns
    assert [(t.speaker, t.text) for t in turns] == [
        ("agent", "thanks for jumping on"),
        ("respondent", "i'm a staff engineer"),
        ("agent", "what does a typical morning look like?"),
    ]
    # The probe turn was tagged with the active goal it addressed.
    assert turns[2].addressed_goal_ids == ["g1"]
    # Eval-then-select ordering (DECISIONS Step 8): the first respondent
    # turn has no prior active goal, so this chat() call selects g1 but
    # does NOT evaluate yet. The status stays ``pending`` until the next
    # respondent reply on g1 is judged.
    assert state.goal_status_table["g1"].status == "pending"
    # last_active reflects what was just probed.
    assert state.last_active is not None and state.last_active.id == "g1"


async def test_interviewer_llm_chat_terminates_when_all_goals_resolved() -> None:
    state = await _bootstrap_state()
    # Pre-mark both goals as ``meets`` so select_next_goal returns None.
    for goal_id in ("g1", "g2"):
        state.goal_status_table[goal_id] = state.goal_status_table[goal_id].model_copy(
            update={"status": "meets"}
        )
    state.opening_done = True
    # Last-active set so the eval branch runs (and pulls from the queue).
    state.last_active = state.conversation_snapshot.goals[0]
    state.llm_client = FakeLLMClient(
        eval_results=[
            EvalResult(
                active_goal_status="meets",
                next_action="advance",
                rationale="done",
            ),
        ],
        utterances=[],
    )

    llm = InterviewerLLM(state=state)
    ctx = ChatContext.empty()
    ctx.add_message(role="user", content="that's all i can think of")
    stream = llm.chat(chat_ctx=ctx)
    chunks: list[str] = []
    async for chunk in stream:
        if chunk.delta and chunk.delta.content:
            chunks.append(chunk.delta.content)
    await stream.aclose()
    closing = "".join(chunks)
    assert closing == "that's all I need."
    # Terminal: state COMPLETED, done flag set, done_event signalled.
    session = await state.store.load_session(state.session_id)
    assert session.state is SessionState.COMPLETED
    assert state.done is True
    assert state.done_event.is_set()
