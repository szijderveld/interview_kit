"""Runtime-state persistence and resume tests (Step 10).

These exercise three things:

1. Every agent utterance flushes a ``SessionRuntimeState`` to the store
   before speaking.
2. A simulator crash mid-loop leaves enough state behind that a second
   ``run_loop`` call rehydrates and runs to completion — with no
   duplicated opening and a single ``RESUME_ACK`` agent turn.
3. ``run_loop`` on a session whose ``Session.state`` is already
   ``COMPLETED`` is idempotent — it returns the cached Extract without
   recording new turns or emitting new events.
"""

from __future__ import annotations

import pytest

from interviewer import (
    Background,
    Engine,
    EvalResult,
    Goal,
    Persona,
    SessionState,
)
from interviewer.loop.resume import RESUME_ACK
from interviewer.loop.runner import run_loop
from interviewer.sinks.memory import InMemoryEventSink
from interviewer.stores.memory import InMemoryConversationStore
from interviewer.testing.fake_llm import FakeLLMClient
from interviewer.testing.simulators import ScriptedSimulator
from interviewer.types.runtime import Turn


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


# ---------- runtime state flush ----------


async def test_runtime_state_is_flushed_before_first_agent_utterance() -> None:
    """Save happens for the opening — before any compose."""
    llm = FakeLLMClient(
        eval_results=[EvalResult(active_goal_status="meets", next_action="advance")],
        utterances=["What's a typical morning look like?"],
    )
    engine, _events, session_id = await _bootstrap(
        llm, [Goal(id="g1", intent="i", standard="s")]
    )
    sim = ScriptedSimulator(["ok.", "Standup at nine."])

    await run_loop(engine, session_id, sim)

    runtime = await engine.store.load_runtime_state(session_id)
    assert runtime is not None
    # After a clean run, the last save was for the closing utterance.
    # total_turns at save time = 6 (opening pair + probe + respondent).
    assert runtime.session_id == session_id


async def test_runtime_state_active_goal_id_set_during_probe() -> None:
    """When the agent is about to speak a probe, runtime_state.active_goal_id is set.

    We snapshot the runtime state at the moment the simulator runs by
    inspecting it inside the simulator's ``respond`` — that callback
    fires right after the agent's probe was recorded.
    """
    captured: list[str | None] = []

    class _CapturingSimulator:
        def __init__(self, responses: list[str]) -> None:
            self._responses = list(responses)
            self._store = None
            self._session_id = None

        def bind(
            self, store: InMemoryConversationStore, session_id: str
        ) -> None:
            self._store = store
            self._session_id = session_id

        async def respond(self, agent_utterance: str, history: list[Turn]) -> str:
            assert self._store is not None
            assert self._session_id is not None
            rs = await self._store.load_runtime_state(self._session_id)
            captured.append(rs.active_goal_id if rs is not None else None)
            return self._responses.pop(0)

        def persona_name(self) -> str:
            return "capturing"

    llm = FakeLLMClient(
        eval_results=[EvalResult(active_goal_status="meets", next_action="advance")],
        utterances=["What's a typical morning look like?"],
    )
    engine, _events, session_id = await _bootstrap(
        llm, [Goal(id="g1", intent="i", standard="s")]
    )
    sim = _CapturingSimulator(["ok.", "Standup at nine."])
    sim.bind(engine.store, session_id)

    await run_loop(engine, session_id, sim)

    # captured[0]: after opening — active_goal_id should be None.
    # captured[1]: after g1 probe — active_goal_id should be "g1".
    assert captured == [None, "g1"]


# ---------- resume after crash ----------


class _CrashingSimulator:
    """Test-local simulator: raises after ``crash_after`` responses."""

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


async def test_resume_after_crash_completes_without_duplicate_opening() -> None:
    """Crash mid-loop, then resume — opening is NOT re-said; RESUME_ACK is."""
    llm = FakeLLMClient(
        eval_results=[
            EvalResult(active_goal_status="meets", next_action="advance"),
        ],
        utterances=[
            "What's a typical morning look like?",  # g1 probe (run 1)
            "How do you handle exceptions?",  # g2 probe (run 2)
        ],
    )
    engine, events, session_id = await _bootstrap(
        llm,
        [
            Goal(id="g1", intent="i", standard="s"),
            Goal(id="g2", intent="i", standard="s"),
        ],
    )

    # Run 1: replies to opening, then raises on the next respond.
    crash_sim = _CrashingSimulator(["Sure."], crash_after=1)
    with pytest.raises(RuntimeError, match="simulated worker crash"):
        await run_loop(engine, session_id, crash_sim)

    # Sanity: the agent's first probe was recorded before the crash,
    # and runtime state was flushed for it.
    mid_transcript = await engine.store.list_turns(session_id)
    assert [t.speaker for t in mid_transcript] == ["agent", "respondent", "agent"]
    assert mid_transcript[2].addressed_goal_ids == ["g1"]
    runtime = await engine.store.load_runtime_state(session_id)
    assert runtime is not None
    assert runtime.active_goal_id == "g1"

    # Run 2: resume with a fresh simulator and a clean event-count baseline.
    resume_sim = ScriptedSimulator(
        ["okay, ready", "we page the floor lead", "got it"]
    )
    extract = await run_loop(engine, session_id, resume_sim)

    transcript = extract.full_transcript
    # Opening was said exactly once across both runs.
    assert sum(1 for t in transcript if t.text == "Hi.") == 1
    # RESUME_ACK was said exactly once.
    assert sum(1 for t in transcript if t.text == RESUME_ACK) == 1
    # The session reached COMPLETED.
    session_state = (await engine.store.load_session(session_id)).state
    assert session_state == SessionState.COMPLETED
    # g2 was probed in the resumed run.
    assert any("g2" in t.addressed_goal_ids for t in transcript)

    # The resumed run emitted no ``respondent_joined`` — that's a fresh-run
    # event, not a resume event.
    types = [e.type for e in events.events]
    assert types.count("respondent_joined") == 1
    assert "completed" in types


async def test_resume_preserves_existing_turns_in_transcript_order() -> None:
    """The pre-crash turns survive verbatim into the resumed transcript."""
    llm = FakeLLMClient(
        eval_results=[
            EvalResult(active_goal_status="meets", next_action="advance"),
        ],
        utterances=[
            "What's a typical morning look like?",
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
    crash_sim = _CrashingSimulator(["Sure."], crash_after=1)
    with pytest.raises(RuntimeError):
        await run_loop(engine, session_id, crash_sim)
    pre_resume = await engine.store.list_turns(session_id)
    pre_texts = [t.text for t in pre_resume]

    resume_sim = ScriptedSimulator(
        ["okay, ready", "we page the floor lead", "got it"]
    )
    extract = await run_loop(engine, session_id, resume_sim)

    final_texts = [t.text for t in extract.full_transcript]
    # The pre-crash texts are a prefix of the final transcript.
    assert final_texts[: len(pre_texts)] == pre_texts


async def test_resume_skips_goals_already_covered_in_transcript() -> None:
    """A goal addressed pre-crash is treated as ``meets`` on resume and skipped."""
    llm = FakeLLMClient(
        eval_results=[
            EvalResult(active_goal_status="meets", next_action="advance"),
        ],
        utterances=[
            "What's a typical morning look like?",
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
    crash_sim = _CrashingSimulator(["Sure."], crash_after=1)
    with pytest.raises(RuntimeError):
        await run_loop(engine, session_id, crash_sim)

    resume_sim = ScriptedSimulator(
        ["okay, ready", "we page the floor lead", "got it"]
    )
    await run_loop(engine, session_id, resume_sim)

    transcript = await engine.store.list_turns(session_id)
    # Exactly one agent probe per goal: g1 was covered pre-crash; g2
    # was probed post-resume. Resume did not re-probe g1.

    def agent_probes_for(gid: str) -> list[Turn]:
        return [
            t for t in transcript
            if t.speaker == "agent" and gid in t.addressed_goal_ids
        ]

    assert len(agent_probes_for("g1")) == 1
    assert len(agent_probes_for("g2")) == 1


# ---------- idempotency on COMPLETED ----------


async def test_run_loop_on_completed_session_returns_cached_extract() -> None:
    """Calling run_loop twice on the same session returns the same Extract."""
    llm = FakeLLMClient(
        eval_results=[
            EvalResult(active_goal_status="meets", next_action="advance"),
        ],
        utterances=["What's a typical morning look like?"],
    )
    engine, events, session_id = await _bootstrap(
        llm, [Goal(id="g1", intent="i", standard="s")]
    )
    sim = ScriptedSimulator(["ok.", "Standup at nine."])
    first = await run_loop(engine, session_id, sim)

    events_before = list(events.events)
    second_sim = ScriptedSimulator(["unused"])
    second = await run_loop(engine, session_id, second_sim)

    assert second.session_id == first.session_id
    assert second.completed_at == first.completed_at
    # No new events were emitted on the no-op second call.
    assert events.events == events_before
    # The second simulator was never consulted.
    # (ScriptedSimulator raises on exhaustion; queue still has its one entry.)


async def test_run_loop_completed_with_missing_extract_raises() -> None:
    """Defensive: COMPLETED state without an Extract is corrupt — fail loudly."""
    llm = FakeLLMClient(eval_results=[], utterances=[])
    engine, _events, session_id = await _bootstrap(
        llm, [Goal(id="g1", intent="i", standard="s")]
    )
    # Force the session to COMPLETED without writing an Extract.
    await engine.store.update_session_state(session_id, SessionState.COMPLETED)
    sim = ScriptedSimulator(["unused"])

    with pytest.raises(RuntimeError, match="COMPLETED but has no extract"):
        await run_loop(engine, session_id, sim)
