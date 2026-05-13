"""End-to-end happy-path test for ``Engine.simulate_session`` (Step 8)."""

from __future__ import annotations

from interview_kit import (
    Background,
    Engine,
    EvalResult,
    Goal,
    Persona,
    SessionState,
)
from interview_kit.sinks.memory import InMemoryEventSink
from interview_kit.stores.memory import InMemoryConversationStore
from interview_kit.testing.fake_llm import FakeLLMClient
from interview_kit.testing.simulators import ScriptedSimulator


def _eval_meets_advance(rationale: str = "ok") -> EvalResult:
    return EvalResult(
        active_goal_status="meets", next_action="advance", rationale=rationale
    )


async def _build_engine_and_conv() -> tuple[Engine, InMemoryEventSink, str]:
    store = InMemoryConversationStore()
    events = InMemoryEventSink()
    llm = FakeLLMClient(
        eval_results=[
            _eval_meets_advance("g1 covered"),
            _eval_meets_advance("g2 covered"),
            _eval_meets_advance("g3 covered"),
        ],
        utterances=[
            "What does a typical morning look like?",
            "How do you handle exceptions in the flow?",
            "What does a good day look like at the end?",
        ],
    )
    engine = Engine(store=store, events=events, llm=llm)

    conv = await engine.create_conversation(
        persona=Persona(
            system_prompt="You are a discovery interview_kit.",
            style="neutral",
            voice_id="v1",
        ),
        purpose="Understand the end-to-end process.",
        background=Background(
            interviewee_role="operator", interviewee_expertise="floor ops"
        ),
        goals=[
            Goal(id="g1", intent="Day's rituals.", standard="Two named with timing."),
            Goal(id="g2", intent="Exception paths.", standard="One flow described."),
            Goal(id="g3", intent="Good-day signal.", standard="One metric."),
        ],
        opening="Thanks for joining today.",
        closing="That's all I needed.",
    )
    return engine, events, conv.id


async def test_happy_path_three_goals_all_meet() -> None:
    engine, events, conv_id = await _build_engine_and_conv()
    simulator = ScriptedSimulator(
        [
            "Sure, happy to talk.",
            "Standup at nine, then code review.",
            "We page the floor lead.",
            "Throughput up thirty percent.",
        ]
    )

    extract = await engine.simulate_session(conv_id, simulator)

    status_by_id = {gs.goal_id: gs.status for gs in extract.goal_statuses}
    assert status_by_id == {"g1": "meets", "g2": "meets", "g3": "meets"}

    # 2 (opening pair) + 3 probes × 2 + 1 closing = 9.
    assert len(extract.full_transcript) == 9

    # Speakers alternate agent/respondent through the opening + 3 probes,
    # then the closing is a final agent turn.
    expected_speakers = ["agent", "respondent"] * 4 + ["agent"]
    assert [t.speaker for t in extract.full_transcript] == expected_speakers


async def test_extract_session_id_and_completed_at_are_runner_owned() -> None:
    engine, _events, conv_id = await _build_engine_and_conv()
    simulator = ScriptedSimulator(
        ["a.", "b.", "c.", "d."],
    )

    extract = await engine.simulate_session(conv_id, simulator)

    # Session id came from the runner (matches the engine's session record);
    # placeholder used by FakeLLMClient (conv.id) was overwritten.
    assert extract.session_id != conv_id
    assert extract.session_id.startswith("sess-")
    assert extract.conversation_id == conv_id
    assert extract.completed_at is not None


async def test_no_goal_status_changed_events_before_completed() -> None:
    """D5: goal_status_changed is emitted only at completion, never mid-loop."""
    engine, events, conv_id = await _build_engine_and_conv()
    simulator = ScriptedSimulator(["a.", "b.", "c.", "d."])

    await engine.simulate_session(conv_id, simulator)

    types = [e.type for e in events.events]
    completed_idx = types.index("completed")
    assert "goal_status_changed" not in types[:completed_idx]
    # Step 8 doesn't emit them at all — Step 11 adds the canonical diff.
    assert "goal_status_changed" not in types


async def test_session_state_completes_and_extract_persisted() -> None:
    engine, _events, conv_id = await _build_engine_and_conv()
    simulator = ScriptedSimulator(["a.", "b.", "c.", "d."])

    extract = await engine.simulate_session(conv_id, simulator)

    session = await engine.store.load_session(extract.session_id)
    assert session.state == SessionState.COMPLETED
    stored = await engine.store.load_extract(extract.session_id)
    assert stored is not None
    assert stored.session_id == extract.session_id


async def test_event_sequence_contains_provision_join_turns_completed() -> None:
    engine, events, conv_id = await _build_engine_and_conv()
    simulator = ScriptedSimulator(["a.", "b.", "c.", "d."])

    await engine.simulate_session(conv_id, simulator)

    types = [e.type for e in events.events]
    assert types[0] == "session_provisioned"
    assert types[1] == "respondent_joined"
    assert types[-1] == "completed"
    # 9 turns → 9 turn_recorded events between join and completed.
    assert types.count("turn_recorded") == 9


async def test_redundancy_in_eval_marks_other_goals_skipped() -> None:
    """When evaluate_turn flags redundant_goal_ids, the table reflects it."""
    store = InMemoryConversationStore()
    events = InMemoryEventSink()
    # g1 covers g3 incidentally; eval flags g3 as redundant.
    llm = FakeLLMClient(
        eval_results=[
            EvalResult(
                active_goal_status="meets",
                redundant_goal_ids=["g3"],
                next_action="advance",
                rationale="g1 answer also covered g3.",
            ),
            _eval_meets_advance("g2 covered"),
        ],
        utterances=[
            "What does a typical morning look like?",
            "How do you handle exceptions?",
        ],
    )
    engine = Engine(store=store, events=events, llm=llm)
    conv = await engine.create_conversation(
        persona=Persona(
            system_prompt="You are a discovery interview_kit.",
            style="neutral",
            voice_id="v1",
        ),
        purpose="Understand the process.",
        background=Background(
            interviewee_role="r", interviewee_expertise="e"
        ),
        goals=[
            Goal(id="g1", intent="A", standard="x"),
            Goal(id="g2", intent="B", standard="x"),
            Goal(id="g3", intent="C", standard="x"),
        ],
        opening="Hi.",
        closing="Bye.",
    )
    simulator = ScriptedSimulator(["ok.", "a.", "b."])

    extract = await engine.simulate_session(conv.id, simulator)

    # FakeLLM.derive_extract is hint-table-blind; it derives status from
    # addressed_goal_ids in the transcript. g3 was skipped (never probed),
    # so it has no evidence turns and derive_extract reports "pending".
    # The fact that the runner's loop-time table marked g3 skipped_redundant
    # will surface as a goal_status_changed in Step 11.
    by_id = {gs.goal_id: gs.status for gs in extract.goal_statuses}
    assert by_id["g1"] == "meets"
    assert by_id["g2"] == "meets"
    assert by_id["g3"] == "pending"
    # Only 2 probes ran (g3 skipped) → 2 opening + 4 probes + 1 closing = 7.
    assert len(extract.full_transcript) == 7


async def test_default_opening_used_when_conversation_opening_is_none() -> None:
    """conv.opening = None → first agent utterance is DEFAULT_OPENING."""
    from interview_kit.loop.openings import DEFAULT_OPENING

    store = InMemoryConversationStore()
    events = InMemoryEventSink()
    llm = FakeLLMClient(
        eval_results=[_eval_meets_advance("g1 covered")],
        utterances=["What does a typical morning look like?"],
    )
    engine = Engine(store=store, events=events, llm=llm)
    conv = await engine.create_conversation(
        persona=Persona(
            system_prompt="You are a discovery interview_kit.",
            style="neutral",
            voice_id="v1",
        ),
        purpose="Understand the process.",
        background=Background(
            interviewee_role="op", interviewee_expertise="ops"
        ),
        goals=[
            Goal(id="g1", intent="Day's rituals.", standard="x"),
        ],
        opening=None,
        closing="That's all I needed.",
    )
    simulator = ScriptedSimulator(["sure.", "standup at nine."])

    extract = await engine.simulate_session(conv.id, simulator)

    assert extract.full_transcript[0].speaker == "agent"
    assert extract.full_transcript[0].text == DEFAULT_OPENING


async def test_closing_recap_used_when_conversation_closing_is_none() -> None:
    """conv.closing = None → final agent utterance comes from compose_closing_recap."""
    store = InMemoryConversationStore()
    events = InMemoryEventSink()
    recap = "Thanks — your standup-at-nine detail was helpful."
    llm = FakeLLMClient(
        eval_results=[_eval_meets_advance("g1 covered")],
        utterances=["What does a typical morning look like?"],
        closings=[recap],
    )
    engine = Engine(store=store, events=events, llm=llm)
    conv = await engine.create_conversation(
        persona=Persona(
            system_prompt="You are a discovery interview_kit.",
            style="neutral",
            voice_id="v1",
        ),
        purpose="Understand the process.",
        background=Background(
            interviewee_role="op", interviewee_expertise="ops"
        ),
        goals=[
            Goal(id="g1", intent="Day's rituals.", standard="x"),
        ],
        opening="Hi.",
        closing=None,
    )
    simulator = ScriptedSimulator(["sure.", "standup at nine."])

    extract = await engine.simulate_session(conv.id, simulator)

    assert extract.full_transcript[-1].speaker == "agent"
    assert extract.full_transcript[-1].text == recap


async def test_closing_recap_falls_back_to_default_when_phrasing_fails() -> None:
    """Recap that fails phrasing twice → DEFAULT_CLOSING."""
    from interview_kit.loop.runner import DEFAULT_CLOSING

    store = InMemoryConversationStore()
    events = InMemoryEventSink()
    # Two over-25-word closings → phrasing fails twice → fallback.
    bad_recap = " ".join(["word"] * 40)
    llm = FakeLLMClient(
        eval_results=[_eval_meets_advance("g1 covered")],
        utterances=["What does a typical morning look like?"],
        closings=[bad_recap, bad_recap],
    )
    engine = Engine(store=store, events=events, llm=llm)
    conv = await engine.create_conversation(
        persona=Persona(
            system_prompt="You are a discovery interview_kit.",
            style="neutral",
            voice_id="v1",
        ),
        purpose="Understand the process.",
        background=Background(
            interviewee_role="op", interviewee_expertise="ops"
        ),
        goals=[
            Goal(id="g1", intent="Day's rituals.", standard="x"),
        ],
        opening="Hi.",
        closing=None,
    )
    simulator = ScriptedSimulator(["sure.", "standup at nine."])

    extract = await engine.simulate_session(conv.id, simulator)

    assert extract.full_transcript[-1].text == DEFAULT_CLOSING
