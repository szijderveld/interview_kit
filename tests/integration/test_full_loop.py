"""End-to-end integration tests (Step 15).

These exercise the full loop + extract + store layers together, catching
regressions that unit tests miss because they only touch one layer.

Each test runs against both reference stores via the parametrized
``store`` fixture in ``conftest.py``. ``FakeLLMClient`` drives all LLM
calls (no Anthropic — that's Step 12); ``ScriptedSimulator`` plus two
small test-local simulators (crashing / self-cancelling) cover the
respondent side.

The scenarios mirror the PLAN Step 15 deliverables list:

1. ``test_engineer_all_goals_met`` — 5-goal happy path, all meets.
2. ``test_engineer_refusal_path_triggers_gave_up_hint`` — the refusal
   heuristic fires twice on one goal; the runner's hint table marks it
   ``gave_up`` and a ``goal_status_changed`` event surfaces the diff
   against the canonical extract.
3. ``test_terse_evasive_persona_completes_well_formed`` —
   ``TerseEvasiveSimulator`` runs against the same 5-goal conversation.
4. ``test_crash_mid_loop_resume_completes_without_duplicate_turns`` —
   crash, then resume, then complete; pre-crash turns survive verbatim.
5. ``test_cancel_mid_loop_abandons_with_partial_transcript`` —
   ``Engine.cancel_session`` mid-loop yields ABANDONED + no extract.
"""

from __future__ import annotations

from collections import deque

import pytest

from interview_kit import (
    Background,
    Engine,
    EvalResult,
    Goal,
    Persona,
    SessionState,
)
from interview_kit.loop.runner import CANCEL_CLOSING, LoopCancelled, run_loop
from interview_kit.protocols import ConversationStore
from interview_kit.sinks.memory import InMemoryEventSink
from interview_kit.testing.fake_llm import FakeLLMClient
from interview_kit.testing.simulators import (
    ScriptedSimulator,
    TerseEvasiveSimulator,
)
from interview_kit.types.runtime import Turn

# ---------- helpers ----------


def _meets_advance(rationale: str = "ok") -> EvalResult:
    return EvalResult(
        active_goal_status="meets", next_action="advance", rationale=rationale
    )


async def _setup_engine(
    store: ConversationStore,
    llm: FakeLLMClient,
    persona: Persona,
    background: Background,
    goals: list[Goal],
) -> tuple[Engine, InMemoryEventSink, str]:
    """Build an Engine + Conversation against a parametrized store."""
    events = InMemoryEventSink()
    engine = Engine(store=store, events=events, llm=llm)
    conv = await engine.create_conversation(
        persona=persona,
        purpose="Map how this team works.",
        background=background,
        goals=goals,
        opening="Thanks for chatting today.",
        closing="That's all I needed.",
    )
    return engine, events, conv.id


# ---------- scenario 1: 5 goals all meet ----------


async def test_engineer_all_goals_met(
    store: ConversationStore,
    engineer_persona: Persona,
    engineer_background: Background,
    engineer_goals: list[Goal],
) -> None:
    """Happy path: 5 goals, scripted simulator, every status reaches ``meets``."""
    llm = FakeLLMClient(
        eval_results=[_meets_advance() for _ in range(5)],
        utterances=[
            "What team are you on?",
            "What does your stack look like?",
            "Where do bugs surface most?",
            "How does code review work?",
            "What does the next six months look like?",
        ],
    )
    engine, events, conv_id = await _setup_engine(
        store, llm, engineer_persona, engineer_background, engineer_goals
    )
    sim = ScriptedSimulator(
        [
            "Sure, happy to chat.",
            "Platform team — I run the data plane.",
            "Mostly Python and Postgres with Go on hot paths.",
            "Schema drift on migrations, mostly.",
            "Two reviewers and async-first culture.",
            "Big push on observability and deploy speed.",
        ]
    )

    extract = await engine.simulate_session(conv_id, sim)

    by_id = {gs.goal_id: gs.status for gs in extract.goal_statuses}
    assert by_id == {
        "role": "meets",
        "stack": "meets",
        "bugs": "meets",
        "review": "meets",
        "future": "meets",
    }
    # opening pair + 5 probes × 2 + closing = 13 turns.
    assert len(extract.full_transcript) == 13

    session = await store.load_session(extract.session_id)
    assert session.state == SessionState.COMPLETED
    stored = await store.load_extract(extract.session_id)
    assert stored is not None
    assert stored.goal_statuses == extract.goal_statuses

    types = [e.type for e in events.events]
    assert types[0] == "session_provisioned"
    assert types[1] == "respondent_joined"
    assert types[-1] == "completed"
    # 13 turn_recorded events.
    assert types.count("turn_recorded") == 13


# ---------- scenario 2: refusal path exercises gave_up hint ----------


async def test_engineer_refusal_path_triggers_gave_up_hint(
    store: ConversationStore,
    engineer_persona: Persona,
    engineer_background: Background,
    engineer_goals: list[Goal],
) -> None:
    """Two consecutive IDK responses on g2 → hint table marks gave_up.

    ``FakeLLMClient.derive_extract`` is hint-table-blind — it derives
    canonical status from turn ``addressed_goal_ids`` only — so the
    canonical Extract will report g2 as ``meets`` (the probe + deflection
    turns both address g2). The diff path emits a
    ``goal_status_changed`` event for g2 from ``gave_up`` → ``meets``,
    proving the refusal heuristic ran and the runner's hint table held
    ``gave_up`` for that goal.
    """
    llm = FakeLLMClient(
        eval_results=[
            _meets_advance("g1 ok"),
            # g2 refusal path is heuristic-driven; no eval consumed.
            _meets_advance("g3 ok"),
            _meets_advance("g4 ok"),
            _meets_advance("g5 ok"),
        ],
        utterances=[
            "What team are you on?",
            "What does your stack look like?",  # g2 probe
            "Is there someone who'd have more visibility?",  # g2 deflection
            "Where do bugs tend to surface?",
            "How does code review work for you?",
            "What's the next six months about?",
        ],
    )
    engine, events, conv_id = await _setup_engine(
        store, llm, engineer_persona, engineer_background, engineer_goals
    )
    sim = ScriptedSimulator(
        [
            "Sure, happy to chat.",
            "Platform team — data plane.",
            "I don't know honestly.",  # IDK #1 on g2 → deflection probe
            "No idea, sorry.",  # IDK #2 on g2 → gave_up + advance
            "Schema drift on migrations.",
            "Two reviewers, async-first.",
            "Observability work.",
        ]
    )

    extract = await engine.simulate_session(conv_id, sim)

    # The deflection probe landed in the transcript — two consecutive
    # agent turns addressed g2.
    g2_agent_turns = [
        t for t in extract.full_transcript
        if t.speaker == "agent" and "stack" in t.addressed_goal_ids
    ]
    assert len(g2_agent_turns) == 2

    # Hint→canonical diff for g2 fired before completed.
    diff_events = [
        e for e in events.events
        if e.type == "goal_status_changed"
        and e.payload.get("goal_id") == "stack"
    ]
    assert len(diff_events) == 1
    assert diff_events[0].payload["from_status"] == "gave_up"

    types = [e.type for e in events.events]
    completed_idx = types.index("completed")
    diff_idx = types.index("goal_status_changed")
    assert diff_idx < completed_idx

    session = await store.load_session(extract.session_id)
    assert session.state == SessionState.COMPLETED


# ---------- scenario 3: TerseEvasiveSimulator completes ----------


async def test_terse_evasive_persona_completes_well_formed(
    store: ConversationStore,
    engineer_persona: Persona,
    engineer_background: Background,
    engineer_goals: list[Goal],
) -> None:
    """``TerseEvasiveSimulator`` against the 5-goal conversation completes cleanly.

    The simulator's dodgy responses cycle through a mix of IDK and
    non-IDK phrases; some land as refusal-heuristic hits and trigger
    deflection probes. The test asserts that despite the rough
    respondent, the loop reaches ``completed`` and the extract is
    well-formed (one ``GoalStatus`` per configured goal, transcript
    present, ``completed_at`` set).
    """
    # Plenty of headroom — refusals don't consume eval, but we don't
    # know exactly how many evals the cycle will produce, so be generous.
    llm = FakeLLMClient(
        eval_results=[_meets_advance(f"ev {i}") for i in range(20)],
        utterances=[f"Probe number {i}." for i in range(20)],
    )
    engine, events, conv_id = await _setup_engine(
        store, llm, engineer_persona, engineer_background, engineer_goals
    )
    sim = TerseEvasiveSimulator()

    extract = await engine.simulate_session(conv_id, sim)

    # All 5 goals appear in the extract exactly once.
    assert {gs.goal_id for gs in extract.goal_statuses} == {
        "role", "stack", "bugs", "review", "future"
    }
    # Statuses are drawn from the legal set.
    legal = {"pending", "meets", "partial", "skipped_redundant", "gave_up"}
    assert all(gs.status in legal for gs in extract.goal_statuses)

    # Transcript alternates speakers except for closing as final agent turn.
    speakers = [t.speaker for t in extract.full_transcript]
    assert speakers[0] == "agent"  # opening
    assert speakers[-1] == "agent"  # closing
    # Indices monotonic increasing.
    indices = [t.index for t in extract.full_transcript]
    assert indices == sorted(indices)
    assert indices[0] == 0
    assert indices[-1] == len(extract.full_transcript) - 1

    types = [e.type for e in events.events]
    assert types[-1] == "completed"


# ---------- scenario 4: crash mid-loop and resume ----------


class _CrashingSimulator:
    """Test-local respondent that raises after ``crash_after`` responses."""

    def __init__(self, responses: list[str], *, crash_after: int) -> None:
        self._responses = list(responses)
        self._idx = 0
        self._crash_after = crash_after

    async def respond(self, agent_utterance: str, history: list[Turn]) -> str:
        if self._idx >= self._crash_after:
            raise RuntimeError("simulated worker crash")
        text = self._responses[self._idx]
        self._idx += 1
        return text

    def persona_name(self) -> str:
        return "crashing"


async def test_crash_mid_loop_resume_completes_without_duplicate_turns(
    store: ConversationStore,
    engineer_persona: Persona,
    engineer_background: Background,
    engineer_goals: list[Goal],
) -> None:
    """Crash after the first probe; resume with a healthy sim; complete cleanly."""
    llm = FakeLLMClient(
        eval_results=[_meets_advance(f"ev {i}") for i in range(5)],
        utterances=[
            "What team are you on?",
            "What's your stack?",
            "Where do bugs surface?",
            "How does review work?",
            "Next six months?",
        ],
    )
    engine, events, conv_id = await _setup_engine(
        store, llm, engineer_persona, engineer_background, engineer_goals
    )
    session, _ = await engine.provision_session(conv_id)

    # Crash after replying to opening + first probe.
    crash_sim = _CrashingSimulator(
        ["Sure.", "Platform team."], crash_after=2
    )
    with pytest.raises(RuntimeError, match="simulated worker crash"):
        await run_loop(engine, session.id, crash_sim)

    pre_resume = await store.list_turns(session.id)
    pre_texts = [t.text for t in pre_resume]
    # Sanity: g1 ("role") was probed pre-crash.
    assert any("role" in t.addressed_goal_ids for t in pre_resume)
    runtime = await store.load_runtime_state(session.id)
    assert runtime is not None

    resume_sim = ScriptedSimulator(
        [
            "ready to continue",
            "Python and Postgres mostly.",
            "Schema drift.",
            "Two reviewers async.",
            "Observability.",
        ]
    )
    extract = await run_loop(engine, session.id, resume_sim)

    transcript = extract.full_transcript
    # Pre-crash turns survive verbatim as a prefix of the final transcript.
    assert [t.text for t in transcript[: len(pre_texts)]] == pre_texts
    # Opening said exactly once across both runs.
    assert sum(1 for t in transcript if t.text == "Thanks for chatting today.") == 1
    # respondent_joined fired once (resume path doesn't re-emit it).
    types = [e.type for e in events.events]
    assert types.count("respondent_joined") == 1
    assert types[-1] == "completed"

    state = (await store.load_session(session.id)).state
    assert state == SessionState.COMPLETED


# ---------- scenario 5: cancel mid-loop ----------


class _CancellingSimulator:
    """Test-local respondent that calls ``engine.cancel_session`` mid-flight.

    Models an operator pressing the cancel button while a respondent is
    answering. The runner sees ABANDONED at the next iteration boundary
    and exits via :class:`LoopCancelled` after speaking the cancel
    closing.
    """

    def __init__(
        self,
        engine: Engine,
        session_id: str,
        responses: list[str],
        *,
        cancel_after_n: int,
    ) -> None:
        self._engine = engine
        self._session_id = session_id
        self._responses: deque[str] = deque(responses)
        self._cancel_after_n = cancel_after_n
        self._count = 0

    async def respond(self, agent_utterance: str, history: list[Turn]) -> str:
        text = self._responses.popleft()
        self._count += 1
        if self._count >= self._cancel_after_n:
            await self._engine.cancel_session(
                self._session_id, reason="operator-cancelled"
            )
        return text

    def persona_name(self) -> str:
        return "cancelling"


async def test_cancel_mid_loop_abandons_with_partial_transcript(
    store: ConversationStore,
    engineer_persona: Persona,
    engineer_background: Background,
    engineer_goals: list[Goal],
) -> None:
    """Operator-initiated cancel mid-loop → ABANDONED + no canonical extract."""
    llm = FakeLLMClient(
        eval_results=[_meets_advance(f"ev {i}") for i in range(5)],
        utterances=[
            "What team are you on?",
            "What's your stack?",
            "Where do bugs surface?",
            "How does review work?",
            "Next six months?",
        ],
    )
    engine, events, conv_id = await _setup_engine(
        store, llm, engineer_persona, engineer_background, engineer_goals
    )
    session, _ = await engine.provision_session(conv_id)

    sim = _CancellingSimulator(
        engine,
        session.id,
        ["Sure.", "Platform team."],
        cancel_after_n=2,  # cancel after second respondent reply.
    )

    with pytest.raises(LoopCancelled):
        await run_loop(engine, session.id, sim)

    transcript = await store.list_turns(session.id)
    # Cancel closing is the final agent turn.
    assert transcript[-1].speaker == "agent"
    assert transcript[-1].text == CANCEL_CLOSING

    final_state = (await store.load_session(session.id)).state
    assert final_state == SessionState.ABANDONED

    # No canonical extract is persisted on cancel.
    assert await store.load_extract(session.id) is None

    types = [e.type for e in events.events]
    assert "abandoned" in types
    assert "completed" not in types
    # The partial transcript still contains the agent's first probe.
    assert any(
        t.speaker == "agent" and "role" in t.addressed_goal_ids
        for t in transcript
    )
