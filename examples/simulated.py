"""Run an interviewer session against a synthetic respondent — no API keys.

Builds a small Conversation, runs ``Engine.simulate_session`` against
``FakeLLMClient`` + ``ScriptedSimulator``, prints transcript + extract.

Usage:
    uv run python examples/simulated.py
"""

from __future__ import annotations

import asyncio

from interviewer import Background, Engine, EvalResult, Goal, Persona
from interviewer.sinks.memory import InMemoryEventSink
from interviewer.stores.memory import InMemoryConversationStore
from interviewer.testing.fake_llm import FakeLLMClient
from interviewer.testing.simulators import ScriptedSimulator


def _eval_results() -> list[EvalResult]:
    # Three goals, each judged "meets/advance" once it gets answered.
    return [
        EvalResult(
            active_goal_status="meets",
            next_action="advance",
            rationale="rituals named with timing.",
        ),
        EvalResult(
            active_goal_status="meets",
            next_action="advance",
            rationale="exception path described.",
        ),
        EvalResult(
            active_goal_status="meets",
            next_action="advance",
            rationale="metric for a good day given.",
        ),
    ]


def _agent_utterances() -> list[str]:
    return [
        "What does a typical morning look like for you?",
        "How do you handle exceptions when something breaks?",
        "And what does a good day look like at the end?",
    ]


def _respondent_responses() -> list[str]:
    return [
        "Happy to walk through it.",
        "Standup at nine, then code review for an hour.",
        "We page the floor lead and run a quick triage.",
        "Throughput up about thirty percent versus last quarter.",
    ]


async def main() -> None:
    store = InMemoryConversationStore()
    events = InMemoryEventSink()
    llm = FakeLLMClient(eval_results=_eval_results(), utterances=_agent_utterances())
    engine = Engine(store=store, events=events, llm=llm)

    conv = await engine.create_conversation(
        persona=Persona(
            system_prompt="You are a process engineer doing a discovery interview.",
            style="neutral",
            voice_id="demo-voice",
        ),
        purpose="Understand the end-to-end flow at this team.",
        background=Background(
            interviewee_role="staff engineer",
            interviewee_expertise="end-to-end pipeline ownership",
        ),
        goals=[
            Goal(id="g1", intent="Map the day's main rituals.",
                 standard="At least two rituals with timing."),
            Goal(id="g2", intent="Find common exception paths.",
                 standard="At least one exception flow named."),
            Goal(id="g3", intent="What does a good day look like.",
                 standard="At least one metric named."),
        ],
        opening="Thanks for joining — I want to learn how you spend a typical week.",
        closing="That's everything I needed. Appreciate your time.",
    )

    simulator = ScriptedSimulator(_respondent_responses())
    extract = await engine.simulate_session(conv.id, simulator)

    print("--- TRANSCRIPT ---")
    for turn in extract.full_transcript:
        print(f"[{turn.index}] {turn.speaker.upper():<11} {turn.text}")
    print()
    print("--- EXTRACT ---")
    for gs in extract.goal_statuses:
        print(
            f"{gs.goal_id:>4}  {gs.status:<16}  "
            f"evidence={gs.evidence_turn_indices}"
        )


if __name__ == "__main__":
    asyncio.run(main())
