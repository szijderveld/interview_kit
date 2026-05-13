"""Refusal vs IDK split coverage for ``run_loop`` (Step 26).

The runner draws a line between a consent-decline ("I'd rather not") and
a knowledge gap ("I don't know"). Refusal marks the active goal
``skipped_refused`` on the first hit and advances without a deflection
probe; IDK keeps the two-strike deflection-then-gave_up behavior.
"""

from __future__ import annotations

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


def _persona() -> Persona:
    return Persona(system_prompt="You are an interviewer.", style="neutral", voice_id="v")


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


async def test_single_refusal_marks_skipped_refused_and_advances() -> None:
    """One "rather not" on g1 → g1 ``skipped_refused``, no deflection, loop advances."""
    llm = FakeLLMClient(
        eval_results=[
            EvalResult(active_goal_status="meets", next_action="advance"),
        ],
        utterances=[
            "What does your morning look like?",  # iter 1 probe (g1)
            "How do you handle exceptions?",  # iter 2 probe (g2) — straight to next goal
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
            "I'd rather not answer that.",  # refusal on g1
            "We page the floor lead.",  # g2 answer
        ]
    )

    extract = await run_loop(engine, session_id, simulator)

    transcript = extract.full_transcript
    # 2 opening + 2 g1 probe+refusal + 2 g2 probe+answer + 1 closing = 7.
    assert len(transcript) == 7
    # Exactly one agent turn addressed g1 — i.e. no deflection probe followed
    # the refusal.
    g1_agent_turns = [
        t for t in transcript
        if t.speaker == "agent" and "g1" in t.addressed_goal_ids
    ]
    assert len(g1_agent_turns) == 1

    # Loop-time hint table held ``skipped_refused`` for g1 before the
    # canonical LLM pass. FakeLLMClient.derive_extract maps by
    # addressed_goal_ids, so the canonical status will be ``meets`` (one
    # probe touched g1). The diff event surfaces the hint→canonical
    # change.
    diff_events = [
        e for e in events.events
        if e.type == "goal_status_changed"
        and e.payload.get("goal_id") == "g1"
    ]
    assert len(diff_events) == 1
    assert diff_events[0].payload["from_status"] == "skipped_refused"

    session_state = (await engine.store.load_session(session_id)).state
    assert session_state == SessionState.COMPLETED


async def test_refused_goal_is_not_reselected_after_advance() -> None:
    """``skipped_refused`` counts as resolved — selection never returns to g1."""
    llm = FakeLLMClient(
        eval_results=[
            EvalResult(active_goal_status="meets", next_action="advance"),
        ],
        utterances=[
            "What does your morning look like?",
            "How do you handle exceptions?",
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
            "Prefer not to share that.",  # refusal on g1
            "We page the floor lead.",  # g2 answer
        ]
    )

    extract = await run_loop(engine, session_id, simulator)

    agent_targets = [
        sorted(t.addressed_goal_ids)
        for t in extract.full_transcript
        if t.speaker == "agent" and t.addressed_goal_ids
    ]
    # Exactly one probe per goal, no return to g1.
    assert agent_targets == [["g1"], ["g2"]]


async def test_idk_path_still_uses_two_strike_gave_up() -> None:
    """Two consecutive IDKs on g1 → ``gave_up`` (the existing path is preserved)."""
    llm = FakeLLMClient(
        eval_results=[
            EvalResult(active_goal_status="meets", next_action="advance"),
        ],
        utterances=[
            "What does your morning look like?",  # iter 1 probe (g1)
            "Is there someone who'd know?",  # iter 2 deflection (g1)
            "How do you handle exceptions?",  # iter 3 probe (g2) after give-up
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
            "I don't know.",  # IDK #1 → deflection
            "No idea.",  # IDK #2 → gave_up
            "We page the lead.",
        ]
    )

    extract = await run_loop(engine, session_id, simulator)

    # Two agent turns addressed g1 — the original probe plus the deflection.
    g1_agent_turns = [
        t for t in extract.full_transcript
        if t.speaker == "agent" and "g1" in t.addressed_goal_ids
    ]
    assert len(g1_agent_turns) == 2

    types = [e.type for e in events.events]
    assert "completed" in types
    assert "failed" not in types
