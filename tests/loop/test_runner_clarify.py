"""Clarity-override coverage for ``run_loop`` (Step 25).

When ``evaluate_turn`` returns ``clarity="hedged"`` (or ``"vague"``) on
an active goal that isn't already resolved, the runner overrides
``next_action`` to a ``probe_clarify`` so the next agent utterance asks
the respondent to firm up the hedged answer. The override fires at most
once per goal — a second hedged answer is allowed to follow the
original action.
"""

from __future__ import annotations

from collections.abc import AsyncIterator

from interview_kit import (
    Background,
    Engine,
    EvalResult,
    Goal,
    Persona,
    SessionState,
)
from interview_kit.loop.runner import run_loop
from interview_kit.sinks.memory import InMemoryEventSink
from interview_kit.stores.memory import InMemoryConversationStore
from interview_kit.testing.fake_llm import FakeLLMClient
from interview_kit.testing.simulators import ScriptedSimulator
from interview_kit.types.runtime import TurnContext


class _RecordingLLM(FakeLLMClient):
    """FakeLLMClient that records the ``EvalResult`` passed to each compose."""

    def __init__(self, **kwargs: object) -> None:
        super().__init__(**kwargs)  # type: ignore[arg-type]
        self.compose_calls: list[EvalResult] = []

    def compose_utterance(
        self, ctx: TurnContext, eval_result: EvalResult
    ) -> AsyncIterator[str]:
        self.compose_calls.append(eval_result)
        return super().compose_utterance(ctx, eval_result)


def _persona() -> Persona:
    return Persona(
        system_prompt="You are an interviewer.", style="neutral", voice_id="v"
    )


def _background() -> Background:
    return Background(interviewee_role="r", interviewee_expertise="e")


async def _bootstrap(
    llm: FakeLLMClient, goals: list[Goal]
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
    session, _ = await engine.provision_session(conv.id)
    return engine, events, session.id


async def test_hedged_answer_forces_clarify_probe_then_advances() -> None:
    """First hedged answer → forced probe_clarify; second answer → advance."""
    llm = _RecordingLLM(
        eval_results=[
            # After "I guess maybe?": model wanted to advance with partial
            # status. clarity=hedged should force a clarify probe.
            EvalResult(
                active_goal_status="partial",
                next_action="advance",
                clarity="hedged",
                rationale="hedged answer",
            ),
            # After "Actually around 9am every day.": clear and meets.
            EvalResult(
                active_goal_status="meets",
                next_action="advance",
                clarity="clear",
                rationale="answered fully",
            ),
        ],
        utterances=[
            "What does your morning look like?",  # iter 1 probe g1
            "When you say 'maybe', what time exactly?",  # iter 2 clarify probe
        ],
    )
    engine, _events, session_id = await _bootstrap(
        llm, [Goal(id="g1", intent="i", standard="s")]
    )
    simulator = ScriptedSimulator(
        [
            "Sure.",
            "I guess maybe?",  # iter 1 hedged
            "Actually around 9am every day.",  # iter 2 clear answer
        ]
    )

    extract = await run_loop(engine, session_id, simulator)

    # 2 opening + 2 first-probe + 2 clarify-probe + 1 closing = 7 turns.
    assert len(extract.full_transcript) == 7
    assert extract.full_transcript[-1].text == "Bye."

    # The second compose call (the agent's clarify probe) saw the
    # overridden EvalResult with next_action='probe' and
    # probe_kind='clarify' even though the model originally said advance.
    assert len(llm.compose_calls) == 2
    second_call = llm.compose_calls[1]
    assert second_call.next_action == "probe"
    assert second_call.probe_kind == "clarify"
    # The original clarity reading is preserved on the overridden copy.
    assert second_call.clarity == "hedged"

    state = (await engine.store.load_session(session_id)).state
    assert state == SessionState.COMPLETED


async def test_clarify_override_caps_at_one_per_goal() -> None:
    """Second consecutive hedged answer on the same goal does NOT re-override."""
    llm = _RecordingLLM(
        eval_results=[
            # 1st hedged → override forces clarify probe.
            EvalResult(
                active_goal_status="partial",
                next_action="advance",
                clarity="hedged",
            ),
            # 2nd hedged on the same goal → override is exhausted; the
            # original advance stands.
            EvalResult(
                active_goal_status="meets",
                next_action="advance",
                clarity="hedged",
            ),
            # g2 answered fully on first try.
            EvalResult(
                active_goal_status="meets",
                next_action="advance",
                clarity="clear",
            ),
        ],
        utterances=[
            "Tell me about mornings.",  # iter 1 probe g1
            "Can you pin that down?",  # iter 2 forced clarify probe g1
            "How do you handle exceptions?",  # iter 3 probe g2
        ],
    )
    engine, _events, session_id = await _bootstrap(
        llm,
        [
            Goal(id="g1", intent="i", standard="s"),
            Goal(id="g2", intent="i", standard="s"),
        ],
    )
    simulator = ScriptedSimulator(
        [
            "Sure.",
            "I guess maybe?",  # 1st hedged on g1 → forces clarify
            "Sort of, kinda.",  # 2nd hedged on g1 → override exhausted
            "We page the floor lead.",  # g2 clear answer
        ]
    )

    extract = await run_loop(engine, session_id, simulator)

    assert extract.full_transcript[-1].text == "Bye."
    assert len(llm.compose_calls) == 3
    # 1st compose: original probe for g1 (no eval yet, placeholder advance).
    # 2nd compose: forced clarify probe on g1.
    assert llm.compose_calls[1].next_action == "probe"
    assert llm.compose_calls[1].probe_kind == "clarify"
    # 3rd compose: g2 probe — override on g1 already used, the 2nd
    # hedged g1 answer was allowed to advance, so g2 is now active.
    assert llm.compose_calls[2].next_action == "advance"
    assert llm.compose_calls[2].probe_kind is None


async def test_clear_clarity_does_not_force_clarify() -> None:
    """clarity='clear' leaves next_action untouched (regression guard)."""
    llm = _RecordingLLM(
        eval_results=[
            EvalResult(
                active_goal_status="meets",
                next_action="advance",
                clarity="clear",
            ),
        ],
        utterances=["What does your morning look like?"],
    )
    engine, _events, session_id = await _bootstrap(
        llm, [Goal(id="g1", intent="i", standard="s")]
    )
    simulator = ScriptedSimulator(["Sure.", "Standup at nine."])

    extract = await run_loop(engine, session_id, simulator)

    assert len(extract.full_transcript) == 5  # 2 opening + 2 probe + 1 closing
    assert len(llm.compose_calls) == 1
    # No override fired; the only compose used the placeholder advance.
    assert llm.compose_calls[0].next_action == "advance"
