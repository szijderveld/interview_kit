"""Unhappy-path coverage for ``run_loop`` (Step 9)."""

from __future__ import annotations

from collections import deque

import pytest

import interview_kit.loop.runner as _runner_mod
from interview_kit import (
    Background,
    Engine,
    EvalResult,
    Goal,
    Persona,
    SessionState,
)
from interview_kit.loop.runner import (
    APOLOGY,
    CANCEL_CLOSING,
    LoopCancelled,
    LoopFailure,
    run_loop,
)
from interview_kit.sinks.memory import InMemoryEventSink
from interview_kit.stores.memory import InMemoryConversationStore
from interview_kit.testing.fake_llm import FakeLLMClient
from interview_kit.testing.simulators import ScriptedSimulator
from interview_kit.types.runtime import Turn


@pytest.fixture(autouse=True)
def _zero_backoff(monkeypatch: pytest.MonkeyPatch) -> None:
    """Skip retry sleeps so failure tests run instantly."""
    monkeypatch.setattr(_runner_mod, "_LLM_BACKOFF_BASE_SECONDS", 0.0)


def _persona() -> Persona:
    return Persona(system_prompt="You are an interview_kit.", style="neutral", voice_id="v")


def _background() -> Background:
    return Background(interviewee_role="r", interviewee_expertise="e")


async def _bootstrap(
    llm: FakeLLMClient, goals: list[Goal], *, max_total_turns: int = 80
) -> tuple[Engine, InMemoryEventSink, str]:
    store = InMemoryConversationStore()
    events = InMemoryEventSink()
    engine = Engine(store=store, events=events, llm=llm)
    conv = await engine.create_conversation(
        persona=_persona(),
        purpose="p",
        background=_background(),
        goals=goals,
        opening="Hi.",
        closing="Bye.",
    )
    if max_total_turns != 80:
        # Rebuild Conversation with a tighter cap (frozen → model_copy).
        conv2 = conv.model_copy(update={"max_total_turns": max_total_turns})
        await engine.store.save_conversation(conv2)
        conv = conv2
    session, _ = await engine.provision_session(conv.id)
    return engine, events, session.id


# ---------- refusal / IDK ----------


async def test_single_refusal_triggers_one_deflection_probe() -> None:
    """First IDK on g1 → one deflection probe consuming a retry; loop continues."""
    llm = FakeLLMClient(
        eval_results=[
            # After deflection succeeds, eval marks g1 meets, advance.
            EvalResult(active_goal_status="meets", next_action="advance"),
            # Then eval g2 meets, advance.
            EvalResult(active_goal_status="meets", next_action="advance"),
        ],
        utterances=[
            "What does your morning look like?",  # iter 1 probe (g1)
            "Is there someone who'd know more?",  # iter 2 deflection (g1, retry)
            "How do you handle exceptions?",  # iter 3 probe (g2)
        ],
    )
    engine, events, session_id = await _bootstrap(
        llm,
        [
            Goal(id="g1", intent="i", standard="s"),
            Goal(id="g2", intent="i", standard="s"),
        ],
    )
    simulator = ScriptedSimulator(
        [
            "Sure.",
            "I don't know, honestly.",  # iter 1 → triggers deflection on g1
            "Maybe ask my colleague Alex.",  # iter 2 deflection answer
            "We page the floor lead.",  # iter 3 g2 answer
        ]
    )

    extract = await run_loop(engine, session_id, simulator)

    transcript = extract.full_transcript
    # 2 opening + 2 probe1 (g1) + 2 deflection (g1) + 2 probe2 (g2) + 1 closing = 9
    assert len(transcript) == 9
    # Closing was the normal one (not the cancel closing)
    assert transcript[-1].text == "Bye."

    state = (await engine.store.load_session(session_id)).state
    assert state == SessionState.COMPLETED


async def test_double_refusal_marks_gave_up_and_advances() -> None:
    """Two consecutive refusals on g1 → g1 gave_up; loop advances to g2."""
    llm = FakeLLMClient(
        eval_results=[
            # After advancing to g2 and getting a real answer.
            EvalResult(active_goal_status="meets", next_action="advance"),
        ],
        utterances=[
            "What does your morning look like?",  # iter 1 probe g1
            "Is there someone who would know?",  # iter 2 deflection on g1
            "How do you handle exceptions?",  # iter 3 probe g2 (after give-up)
        ],
    )
    engine, events, session_id = await _bootstrap(
        llm,
        [
            Goal(id="g1", intent="i", standard="s"),
            Goal(id="g2", intent="i", standard="s"),
        ],
    )
    simulator = ScriptedSimulator(
        [
            "Sure.",
            "I don't know.",  # refusal #1 on g1 → deflection
            "No idea, really.",  # refusal #2 on g1 → gave_up + advance
            "We page the lead.",  # g2 answer
        ]
    )

    extract = await run_loop(engine, session_id, simulator)

    # 2 opening + 2 probe1+refusal + 2 deflection+refusal + 2 g2 probe+answer + 1 closing = 9.
    assert len(extract.full_transcript) == 9
    assert extract.full_transcript[-1].text == "Bye."

    # The 'completed' event fired (no LoopCancelled, no LoopFailure).
    types = [e.type for e in events.events]
    assert "completed" in types
    assert "failed" not in types


async def test_idk_keyword_triggers_same_path_as_refusal() -> None:
    """Coverage: ``no idea`` is detected the same as ``rather not``."""
    llm = FakeLLMClient(
        eval_results=[EvalResult(active_goal_status="meets", next_action="advance")],
        utterances=[
            "What does your day look like?",
            "Anyone on your team have visibility?",
        ],
    )
    engine, _events, session_id = await _bootstrap(
        llm, [Goal(id="g1", intent="i", standard="s")]
    )
    simulator = ScriptedSimulator(
        [
            "ok.",
            "no idea.",  # IDK → deflection
            "Probably Sam knows that one.",  # normal answer
        ]
    )

    extract = await run_loop(engine, session_id, simulator)
    # Opening (2) + probe (2) + deflection (2) + closing (1) = 7.
    assert len(extract.full_transcript) == 7


# ---------- LLM persistent / transient failure ----------


async def test_eval_persistent_failure_sets_failed_state_and_apology() -> None:
    """3 consecutive ``evaluate_turn`` failures → FAILED + apology + ``failed`` event."""
    llm = FakeLLMClient(
        eval_results=[],
        utterances=["What does your morning look like?"],
        eval_failures=100,
    )
    engine, events, session_id = await _bootstrap(
        llm, [Goal(id="g1", intent="i", standard="s")]
    )
    simulator = ScriptedSimulator(["ok.", "Standup at nine."])

    with pytest.raises(LoopFailure):
        await run_loop(engine, session_id, simulator)

    session_state = (await engine.store.load_session(session_id)).state
    assert session_state == SessionState.FAILED

    transcript = await engine.store.list_turns(session_id)
    assert transcript[-1].speaker == "agent"
    assert transcript[-1].text == APOLOGY

    types = [e.type for e in events.events]
    assert "failed" in types
    assert "completed" not in types


async def test_eval_transient_failure_recovers_after_two_attempts() -> None:
    """``evaluate_turn`` fails twice then succeeds — loop completes normally."""
    llm = FakeLLMClient(
        eval_results=[EvalResult(active_goal_status="meets", next_action="advance")],
        utterances=["What does your morning look like?"],
        eval_failures=2,  # attempts 1 and 2 fail; attempt 3 returns the eval result
    )
    engine, _events, session_id = await _bootstrap(
        llm, [Goal(id="g1", intent="i", standard="s")]
    )
    simulator = ScriptedSimulator(["ok.", "Standup at nine."])

    extract = await run_loop(engine, session_id, simulator)

    session_state = (await engine.store.load_session(session_id)).state
    assert session_state == SessionState.COMPLETED
    by_id = {gs.goal_id: gs.status for gs in extract.goal_statuses}
    assert by_id["g1"] == "meets"


async def test_compose_persistent_failure_sets_failed_state() -> None:
    """3 consecutive ``compose_utterance`` failures → FAILED."""
    llm = FakeLLMClient(
        eval_results=[],
        utterances=[],
        compose_failures=100,
    )
    engine, events, session_id = await _bootstrap(
        llm, [Goal(id="g1", intent="i", standard="s")]
    )
    simulator = ScriptedSimulator(["ok."])  # only the opening reply is needed

    with pytest.raises(LoopFailure):
        await run_loop(engine, session_id, simulator)

    session_state = (await engine.store.load_session(session_id)).state
    assert session_state == SessionState.FAILED

    types = [e.type for e in events.events]
    assert "failed" in types
    assert "completed" not in types


# ---------- turn cap ----------


async def test_turn_cap_stops_loop_before_starting_a_new_probe() -> None:
    """``max_total_turns`` reached → no new probe; loop falls through to closing."""
    llm = FakeLLMClient(
        eval_results=[
            EvalResult(active_goal_status="partial", next_action="retry"),
        ],
        utterances=[
            "What does your morning look like?",  # iter 1 probe
            "Could you give an example?",  # iter 2 probe (won't be reached)
        ],
    )
    # max_total_turns=4: opening (2) + iter 1 (2) = 4 turns → loop exits at top.
    engine, _events, session_id = await _bootstrap(
        llm,
        [Goal(id="g1", intent="i", standard="s", max_retries=5)],
        max_total_turns=4,
    )
    simulator = ScriptedSimulator(["ok.", "Standup at nine."])

    extract = await run_loop(engine, session_id, simulator)

    # 2 opening + 2 probe1 + 1 closing = 5 turns total.
    assert len(extract.full_transcript) == 5
    state = (await engine.store.load_session(session_id)).state
    assert state == SessionState.COMPLETED


# ---------- cancel mid-loop ----------


class _SelfCancellingSimulator:
    """Test-local simulator: writes ABANDONED to the store after N responses."""

    def __init__(
        self,
        store: InMemoryConversationStore,
        session_id: str,
        responses: list[str],
        *,
        cancel_after_n: int,
    ) -> None:
        self._store = store
        self._session_id = session_id
        self._responses: deque[str] = deque(responses)
        self._cancel_after_n = cancel_after_n
        self._count = 0

    async def respond(self, agent_utterance: str, history: list[Turn]) -> str:
        text = self._responses.popleft()
        self._count += 1
        if self._count >= self._cancel_after_n:
            await self._store.update_session_state(
                self._session_id, SessionState.ABANDONED
            )
        return text

    def persona_name(self) -> str:
        return "self_cancelling"


async def test_cancel_mid_loop_speaks_closing_and_raises() -> None:
    """ABANDONED state mid-loop → closing turn, LoopCancelled raised, no completed."""
    llm = FakeLLMClient(
        eval_results=[
            EvalResult(active_goal_status="meets", next_action="advance"),
        ],
        utterances=["What does your morning look like?"],
    )
    engine, events, session_id = await _bootstrap(
        llm, [Goal(id="g1", intent="i", standard="s")]
    )
    simulator = _SelfCancellingSimulator(
        engine.store,
        session_id,
        ["ok.", "Standup at nine."],
        cancel_after_n=2,  # cancel after the 2nd respond call (iter 1's reply)
    )

    with pytest.raises(LoopCancelled):
        await run_loop(engine, session_id, simulator)

    transcript = await engine.store.list_turns(session_id)
    assert transcript[-1].text == CANCEL_CLOSING
    assert transcript[-1].speaker == "agent"

    final_state = (await engine.store.load_session(session_id)).state
    assert final_state == SessionState.ABANDONED

    types = [e.type for e in events.events]
    assert "completed" not in types
