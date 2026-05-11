"""Run an interviewer session against a synthetic respondent — no voice.

Default mode uses ``FakeLLMClient`` + ``ScriptedSimulator`` — no API
keys, deterministic transcript.

``--use-anthropic`` swaps in ``AnthropicLLMClient`` and a cycling
``RamblyKnowledgeableSimulator``. Requires ``ANTHROPIC_API_KEY``. Cost is
~10-30 cents per run; acceptance is "completes without crashing and the
extract has the right shape" — the transcript may be noisier than the
deterministic fake.

Usage:
    uv run python examples/simulated.py
    ANTHROPIC_API_KEY=sk-... uv run python examples/simulated.py --use-anthropic
"""

from __future__ import annotations

import argparse
import asyncio
import os

from interviewer import Background, Engine, EvalResult, Goal, Persona
from interviewer.protocols import LLMClient, RespondentSimulator
from interviewer.sinks.memory import InMemoryEventSink
from interviewer.stores.memory import InMemoryConversationStore
from interviewer.testing.fake_llm import FakeLLMClient
from interviewer.testing.simulators import (
    RamblyKnowledgeableSimulator,
    ScriptedSimulator,
)


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


def _build_llm_and_simulator(use_anthropic: bool) -> tuple[LLMClient, RespondentSimulator]:
    if use_anthropic:
        api_key = os.environ.get("ANTHROPIC_API_KEY")
        if not api_key:
            raise SystemExit(
                "ANTHROPIC_API_KEY is required when --use-anthropic is set."
            )
        # Local import: AnthropicLLMClient pulls in the SDK only when
        # this flag is actually requested.
        from interviewer.llm.anthropic import AnthropicLLMClient

        llm: LLMClient = AnthropicLLMClient(api_key=api_key)
        # A cycling simulator can't exhaust its response queue if the
        # real Anthropic model decides to drill or retry.
        sim: RespondentSimulator = RamblyKnowledgeableSimulator()
    else:
        llm = FakeLLMClient(
            eval_results=_eval_results(), utterances=_agent_utterances()
        )
        sim = ScriptedSimulator(_respondent_responses())
    return llm, sim


async def main(use_anthropic: bool) -> None:
    store = InMemoryConversationStore()
    events = InMemoryEventSink()
    llm, simulator = _build_llm_and_simulator(use_anthropic)
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
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0] if __doc__ else None)
    parser.add_argument(
        "--use-anthropic",
        action="store_true",
        help="Swap FakeLLMClient for AnthropicLLMClient (requires ANTHROPIC_API_KEY).",
    )
    args = parser.parse_args()
    asyncio.run(main(args.use_anthropic))
