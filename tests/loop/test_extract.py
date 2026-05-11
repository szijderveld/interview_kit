"""Canonical Extract pass and goal_status_changed diff tests (Step 11)."""

from __future__ import annotations

from datetime import UTC, datetime

from interviewer import (
    Background,
    Engine,
    EvalResult,
    Goal,
    Persona,
)
from interviewer.loop.extract import derive_extract_with_llm
from interviewer.loop.runner import run_loop
from interviewer.sinks.memory import InMemoryEventSink
from interviewer.stores.memory import InMemoryConversationStore
from interviewer.testing.fake_llm import FakeLLMClient
from interviewer.testing.simulators import ScriptedSimulator
from interviewer.types.config import Conversation
from interviewer.types.runtime import Turn


def _persona() -> Persona:
    return Persona(
        system_prompt="You are an interviewer.", style="neutral", voice_id="v"
    )


def _background() -> Background:
    return Background(interviewee_role="r", interviewee_expertise="e")


def _turn(index: int, speaker: str, text: str, addressed: list[str]) -> Turn:
    return Turn(
        index=index,
        speaker=speaker,  # type: ignore[arg-type]
        text=text,
        timestamp=datetime.now(UTC),
        addressed_goal_ids=addressed,
    )


# ---------- canonical evidence_turn_indices ----------


async def test_evidence_turn_indices_match_addressed_goal_ids() -> None:
    """FakeLLMClient maps every Turn.addressed_goal_ids to evidence indices."""
    conv = Conversation(
        id="conv-1",
        persona=_persona(),
        purpose="p",
        background=_background(),
        goals=[
            Goal(id="g1", intent="i1", standard="s"),
            Goal(id="g2", intent="i2", standard="s"),
            Goal(id="g3", intent="i3", standard="s"),
        ],
    )
    transcript = [
        _turn(0, "agent", "open", []),
        _turn(1, "respondent", "ok", []),
        _turn(2, "agent", "probe g1", ["g1"]),
        _turn(3, "respondent", "answer g1", ["g1"]),
        _turn(4, "agent", "probe g2", ["g2"]),
        _turn(5, "respondent", "answer g2", ["g2"]),
        # g3 never probed → no evidence → pending
    ]
    llm = FakeLLMClient()

    extract = await derive_extract_with_llm(transcript, conv, llm)

    by_id = {gs.goal_id: gs for gs in extract.goal_statuses}
    assert by_id["g1"].status == "meets"
    assert by_id["g1"].evidence_turn_indices == [2, 3]
    assert by_id["g2"].status == "meets"
    assert by_id["g2"].evidence_turn_indices == [4, 5]
    assert by_id["g3"].status == "pending"
    assert by_id["g3"].evidence_turn_indices == []


async def test_derive_extract_with_llm_returns_unchanged() -> None:
    """The helper is a passthrough; it does not mutate session_id/completed_at."""
    conv = Conversation(
        id="conv-x",
        persona=_persona(),
        purpose="p",
        background=_background(),
        goals=[Goal(id="g1", intent="i", standard="s")],
    )
    transcript = [_turn(0, "agent", "probe", ["g1"])]
    llm = FakeLLMClient()

    extract = await derive_extract_with_llm(transcript, conv, llm)

    # FakeLLMClient uses conv.id as placeholder session_id. The helper
    # returns it unchanged — the runner overwrites these fields after.
    assert extract.session_id == conv.id
    assert extract.conversation_id == conv.id


async def test_force_disagreement_flips_status_to_partial() -> None:
    """``force_disagreement_for`` overrides the natural addressed-id derivation."""
    conv = Conversation(
        id="conv-2",
        persona=_persona(),
        purpose="p",
        background=_background(),
        goals=[
            Goal(id="g1", intent="i", standard="s"),
            Goal(id="g2", intent="i", standard="s"),
        ],
    )
    transcript = [
        _turn(0, "agent", "probe g1", ["g1"]),
        _turn(1, "respondent", "a", ["g1"]),
        _turn(2, "agent", "probe g2", ["g2"]),
        _turn(3, "respondent", "b", ["g2"]),
    ]
    llm = FakeLLMClient(force_disagreement_for=["g2"])

    extract = await derive_extract_with_llm(transcript, conv, llm)

    by_id = {gs.goal_id: gs for gs in extract.goal_statuses}
    assert by_id["g1"].status == "meets"
    # g2 had evidence so would naturally be "meets"; force flips to "partial".
    assert by_id["g2"].status == "partial"
    assert "forced disagreement" in by_id["g2"].rationale


# ---------- diff path emits goal_status_changed before completed ----------


async def _bootstrap_engine(
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


async def test_diff_emits_one_goal_status_changed_before_completed() -> None:
    """Force g2 into disagreement → exactly one g2 event, ordered pre-completed."""
    llm = FakeLLMClient(
        eval_results=[
            EvalResult(active_goal_status="meets", next_action="advance"),
            EvalResult(active_goal_status="meets", next_action="advance"),
        ],
        utterances=["probe g1?", "probe g2?"],
        force_disagreement_for=["g2"],
    )
    engine, events, session_id = await _bootstrap_engine(
        llm,
        [
            Goal(id="g1", intent="i", standard="s"),
            Goal(id="g2", intent="i", standard="s"),
        ],
    )
    sim = ScriptedSimulator(["ok.", "a.", "b."])

    await run_loop(engine, session_id, sim)

    types = [e.type for e in events.events]
    completed_idx = types.index("completed")
    diff_events = [
        i for i, e in enumerate(events.events) if e.type == "goal_status_changed"
    ]
    assert len(diff_events) == 1
    diff_idx = diff_events[0]
    assert diff_idx < completed_idx

    diff_event = events.events[diff_idx]
    assert diff_event.payload["goal_id"] == "g2"
    assert diff_event.payload["from_status"] == "meets"
    assert diff_event.payload["to_status"] == "partial"
    assert "forced disagreement" in diff_event.payload["rationale"]


async def test_no_diff_means_no_goal_status_changed_events() -> None:
    """When canonical agrees with the hint table, no diff events fire."""
    llm = FakeLLMClient(
        eval_results=[
            EvalResult(active_goal_status="meets", next_action="advance"),
            EvalResult(active_goal_status="meets", next_action="advance"),
        ],
        utterances=["probe g1?", "probe g2?"],
    )
    engine, events, session_id = await _bootstrap_engine(
        llm,
        [
            Goal(id="g1", intent="i", standard="s"),
            Goal(id="g2", intent="i", standard="s"),
        ],
    )
    sim = ScriptedSimulator(["ok.", "a.", "b."])

    await run_loop(engine, session_id, sim)

    types = [e.type for e in events.events]
    assert "goal_status_changed" not in types


async def test_redundancy_hint_surfaces_as_goal_status_changed() -> None:
    """Loop-time skipped_redundant vs canonical pending → diff event fires."""
    llm = FakeLLMClient(
        eval_results=[
            EvalResult(
                active_goal_status="meets",
                redundant_goal_ids=["g3"],
                next_action="advance",
                rationale="g1 answer covered g3",
            ),
            EvalResult(active_goal_status="meets", next_action="advance"),
        ],
        utterances=["probe g1?", "probe g2?"],
    )
    engine, events, session_id = await _bootstrap_engine(
        llm,
        [
            Goal(id="g1", intent="i", standard="s"),
            Goal(id="g2", intent="i", standard="s"),
            Goal(id="g3", intent="i", standard="s"),
        ],
    )
    sim = ScriptedSimulator(["ok.", "a.", "b."])

    await run_loop(engine, session_id, sim)

    diff_events = [e for e in events.events if e.type == "goal_status_changed"]
    # g3 was marked skipped_redundant in the hint table but the
    # canonical pass sees no evidence and reports pending.
    assert len(diff_events) == 1
    assert diff_events[0].payload["goal_id"] == "g3"
    assert diff_events[0].payload["from_status"] == "skipped_redundant"
    assert diff_events[0].payload["to_status"] == "pending"


# ---------- completed event payload shape ----------


async def test_completed_payload_includes_eval_usage_totals() -> None:
    """D11: completed event payload reserves an eval_usage_totals slot."""
    llm = FakeLLMClient(
        eval_results=[
            EvalResult(active_goal_status="meets", next_action="advance"),
        ],
        utterances=["probe g1?"],
    )
    engine, events, session_id = await _bootstrap_engine(
        llm, [Goal(id="g1", intent="i", standard="s")]
    )
    sim = ScriptedSimulator(["ok.", "a."])

    await run_loop(engine, session_id, sim)

    completed = next(e for e in events.events if e.type == "completed")
    assert "eval_usage_totals" in completed.payload
    usage = completed.payload["eval_usage_totals"]
    assert set(usage) == {
        "tokens_in",
        "tokens_out",
        "cache_read_tokens",
        "cache_write_tokens",
        "llm_latency_ms",
    }
    # Step 11 with FakeLLMClient: usage is structurally present but zero;
    # Step 12 plumbs real Anthropic counts.
    assert all(v == 0 for v in usage.values())
    assert "goal_statuses" in completed.payload
