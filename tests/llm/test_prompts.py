"""Snapshot-shape tests for the prompt builders (Step 12).

The tests assert structural properties (section headers, goal IDs,
voice rules) rather than exact wording — the wording is allowed to
drift; the shape is the contract.
"""

from __future__ import annotations

from datetime import UTC, datetime

import pytest

from interview_kit.llm.prompts import (
    build_closing_recap_user_message,
    build_compose_user_message,
    build_evaluate_user_message,
    build_extract_user_message,
    build_system_prompt,
    format_full_transcript,
    format_transcript_window,
)
from interview_kit.types.config import Background, Conversation, Goal, Persona
from interview_kit.types.runtime import EvalResult, Turn, TurnContext


def _conv() -> Conversation:
    return Conversation(
        id="conv-1",
        persona=Persona(
            system_prompt="You are a sim engineer interviewing a peer.",
            style="warm",
            voice_id="v1",
        ),
        purpose="Map the day-to-day at a warehouse.",
        background=Background(
            interviewee_role="20-year operator",
            interviewee_expertise="end-to-end process",
            relevant_context="multi-shift facility",
        ),
        goals=[
            Goal(id="g1", intent="Find rituals.", standard="two examples",
                 redundant_when="if covered already"),
            Goal(id="g2", intent="Find exceptions.", standard="one flow"),
        ],
    )


def _turn(index: int, speaker: str, text: str) -> Turn:
    return Turn(
        index=index,
        speaker=speaker,  # type: ignore[arg-type]
        text=text,
        timestamp=datetime.now(UTC),
    )


# ---------- build_system_prompt ----------


def test_system_prompt_contains_all_section_headers() -> None:
    prompt = build_system_prompt(_conv())
    for header in (
        "# Your persona",
        "# Why you are talking to this person",
        "# Who they are",
        "# What you are trying to find out",
        "# Interviewing rules",
        "# Voice phrasing rules",
    ):
        assert header in prompt


def test_system_prompt_interviewing_rules_appear_between_goals_and_phrasing() -> None:
    prompt = build_system_prompt(_conv())
    goals_idx = prompt.index("# What you are trying to find out")
    rules_idx = prompt.index("# Interviewing rules")
    phrasing_idx = prompt.index("# Voice phrasing rules")
    assert goals_idx < rules_idx < phrasing_idx


def test_system_prompt_forbids_leading_frames() -> None:
    import re

    prompt = build_system_prompt(_conv())
    # The block must explicitly call out leading frames as forbidden.
    assert re.search(r"don't you", prompt) is not None
    assert re.search(r"would you agree", prompt) is not None


def test_system_prompt_includes_funnel_and_vocabulary_and_pivot_rules() -> None:
    prompt = build_system_prompt(_conv())
    assert "Funnel" in prompt
    assert "vocabulary" in prompt
    assert "acknowledgement" in prompt


def test_system_prompt_lists_every_goal_id_and_intent() -> None:
    prompt = build_system_prompt(_conv())
    for goal_id, intent in (("g1", "Find rituals."), ("g2", "Find exceptions.")):
        assert f"## Goal {goal_id}" in prompt
        assert intent in prompt


def test_system_prompt_includes_redundancy_rubric_when_present() -> None:
    prompt = build_system_prompt(_conv())
    assert "REDUNDANT_WHEN: if covered already" in prompt
    # g2 has no redundancy rubric → placeholder appears.
    assert "REDUNDANT_WHEN: (no redundancy rubric)" in prompt


def test_system_prompt_includes_background_role_and_expertise() -> None:
    prompt = build_system_prompt(_conv())
    assert "Role: 20-year operator" in prompt
    assert "Expertise: end-to-end process" in prompt
    assert "Additional context: multi-shift facility" in prompt


def test_system_prompt_uses_placeholder_for_empty_relevant_context() -> None:
    conv = _conv()
    conv = conv.model_copy(
        update={
            "background": conv.background.model_copy(update={"relevant_context": ""}),
        }
    )
    prompt = build_system_prompt(conv)
    assert "Additional context: (none)" in prompt


def test_system_prompt_lists_voice_phrasing_rules() -> None:
    prompt = build_system_prompt(_conv())
    assert "One question per utterance" in prompt
    assert "25 words" in prompt


# ---------- transcript formatters ----------


def test_format_transcript_window_oldest_first_with_speaker_prefix() -> None:
    turns = [
        _turn(0, "agent", "first"),
        _turn(1, "respondent", "second"),
        _turn(2, "agent", "third"),
        _turn(3, "respondent", "fourth"),
    ]
    formatted = format_transcript_window(turns, max_turns=3)
    lines = formatted.split("\n")
    # window of 3, oldest first
    assert lines == [
        "RESPONDENT: second",
        "AGENT: third",
        "RESPONDENT: fourth",
    ]


def test_format_transcript_window_empty_returns_placeholder() -> None:
    assert format_transcript_window([], max_turns=12) == "(no turns yet)"


def test_format_full_transcript_includes_index_prefix() -> None:
    turns = [_turn(0, "agent", "hello"), _turn(1, "respondent", "hi back")]
    formatted = format_full_transcript(turns)
    assert formatted == "[0] AGENT: hello\n[1] RESPONDENT: hi back"


# ---------- user messages ----------


def test_evaluate_user_message_calls_out_active_goal_and_tool() -> None:
    conv = _conv()
    ctx = TurnContext(
        conversation=conv,
        transcript=[_turn(0, "agent", "q"), _turn(1, "respondent", "a")],
        active_goal=conv.goals[0],
        goal_statuses=[],
        retries_used_on_active=0,
        tangent_followups_used=0,
        total_turns=2,
    )
    msg = build_evaluate_user_message(ctx, max_transcript_turns=12)
    assert "g1 — Find rituals." in msg
    assert "Call the `evaluate` tool" in msg


def test_evaluate_user_message_requires_active_goal() -> None:
    conv = _conv()
    ctx = TurnContext(
        conversation=conv,
        transcript=[],
        active_goal=None,
        goal_statuses=[],
        retries_used_on_active=0,
        tangent_followups_used=0,
        total_turns=0,
    )
    with pytest.raises(ValueError, match="active goal"):
        build_evaluate_user_message(ctx, max_transcript_turns=12)


def test_compose_user_message_includes_eval_signals() -> None:
    conv = _conv()
    ctx = TurnContext(
        conversation=conv,
        transcript=[_turn(0, "respondent", "a")],
        active_goal=conv.goals[0],
        goal_statuses=[],
        retries_used_on_active=0,
        tangent_followups_used=0,
        total_turns=1,
    )
    eval_result = EvalResult(
        active_goal_status="meets",
        next_action="drill",
        interesting_tangent="kanban board snag",
        rationale="ok",
    )
    msg = build_compose_user_message(ctx, eval_result, max_transcript_turns=12)
    assert "active_goal_status: meets" in msg
    assert "next_action: drill" in msg
    assert "interesting_tangent: kanban board snag" in msg
    assert "Output only the utterance text" in msg


def test_compose_user_message_includes_ack_instruction_on_advance() -> None:
    conv = _conv()
    ctx = TurnContext(
        conversation=conv,
        transcript=[_turn(0, "respondent", "a")],
        active_goal=conv.goals[0],
        goal_statuses=[],
        retries_used_on_active=0,
        tangent_followups_used=0,
        total_turns=1,
    )
    eval_result = EvalResult(active_goal_status="meets", next_action="advance")
    msg = build_compose_user_message(ctx, eval_result, max_transcript_turns=12)
    assert "Lead with a brief acknowledgement" in msg


def test_compose_user_message_includes_ack_instruction_on_drill() -> None:
    conv = _conv()
    ctx = TurnContext(
        conversation=conv,
        transcript=[_turn(0, "respondent", "a")],
        active_goal=conv.goals[0],
        goal_statuses=[],
        retries_used_on_active=0,
        tangent_followups_used=0,
        total_turns=1,
    )
    eval_result = EvalResult(active_goal_status="partial", next_action="drill")
    msg = build_compose_user_message(ctx, eval_result, max_transcript_turns=12)
    assert "Lead with a brief acknowledgement" in msg


def test_compose_user_message_omits_ack_on_retry() -> None:
    conv = _conv()
    ctx = TurnContext(
        conversation=conv,
        transcript=[_turn(0, "respondent", "a")],
        active_goal=conv.goals[0],
        goal_statuses=[],
        retries_used_on_active=0,
        tangent_followups_used=0,
        total_turns=1,
    )
    eval_result = EvalResult(active_goal_status="partial", next_action="retry")
    msg = build_compose_user_message(ctx, eval_result, max_transcript_turns=12)
    assert "acknowledgement" not in msg


def test_compose_user_message_omits_ack_on_close() -> None:
    conv = _conv()
    ctx = TurnContext(
        conversation=conv,
        transcript=[_turn(0, "respondent", "a")],
        active_goal=conv.goals[0],
        goal_statuses=[],
        retries_used_on_active=0,
        tangent_followups_used=0,
        total_turns=1,
    )
    eval_result = EvalResult(active_goal_status="meets", next_action="close")
    msg = build_compose_user_message(ctx, eval_result, max_transcript_turns=12)
    assert "acknowledgement" not in msg


def test_compose_user_message_surfaces_phrasing_failure_on_regen() -> None:
    conv = _conv()
    ctx = TurnContext(
        conversation=conv,
        transcript=[_turn(0, "respondent", "a")],
        active_goal=conv.goals[0],
        goal_statuses=[],
        retries_used_on_active=0,
        tangent_followups_used=0,
        total_turns=1,
        last_phrasing_failure="too_long",
    )
    eval_result = EvalResult(active_goal_status="meets", next_action="advance")
    msg = build_compose_user_message(ctx, eval_result, max_transcript_turns=12)
    assert "Your previous attempt failed: too_long" in msg


def test_closing_recap_user_message_includes_transcript_and_recap_instruction() -> None:
    conv = _conv()
    ctx = TurnContext(
        conversation=conv,
        transcript=[
            _turn(0, "agent", "What does a morning look like?"),
            _turn(1, "respondent", "Standup at nine, then code review."),
        ],
        active_goal=None,
        goal_statuses=[],
        retries_used_on_active=0,
        tangent_followups_used=0,
        total_turns=2,
    )
    msg = build_closing_recap_user_message(ctx, max_transcript_turns=12)
    assert "Standup at nine" in msg
    assert "thanks" in msg.lower()
    assert "25 words" in msg
    assert "Output only the utterance text" in msg


def test_extract_user_message_uses_full_transcript_with_indices() -> None:
    turns = [_turn(0, "agent", "open"), _turn(1, "respondent", "ok")]
    msg = build_extract_user_message(turns)
    assert "[0] AGENT: open" in msg
    assert "[1] RESPONDENT: ok" in msg
    assert "Call the `extract` tool" in msg
