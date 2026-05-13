"""Prompt builders for :class:`AnthropicLLMClient`.

Pure functions that turn :class:`Conversation` and :class:`TurnContext`
into the strings shipped to Anthropic. Kept in their own module so they
can be unit-tested without instantiating the SDK client.

The system prompt is intentionally identical across the three LLM
methods so the ephemeral prompt cache is shared between
``evaluate_turn`` and ``compose_utterance`` within the 5-minute cache
window.
"""

from __future__ import annotations

from interview_kit.types.config import Conversation
from interview_kit.types.runtime import EvalResult, Turn, TurnContext


def build_system_prompt(conv: Conversation) -> str:
    """One stable system block per Conversation — cached by Anthropic."""
    bg = conv.background
    relevant = bg.relevant_context or "(none)"
    parts: list[str] = []
    parts.append(
        "You are the interviewer described below. You are conducting a voice "
        "interview. Respond as the interviewer would, in short conversational "
        "sentences, one question at a time."
    )
    parts.append("")
    parts.append("# Your persona")
    parts.append(conv.persona.system_prompt)
    parts.append(f"Style: {conv.persona.style}.")
    parts.append("")
    parts.append("# Why you are talking to this person")
    parts.append(conv.purpose)
    parts.append("")
    parts.append("# Who they are")
    parts.append(f"Role: {bg.interviewee_role}")
    parts.append(f"Expertise: {bg.interviewee_expertise}")
    parts.append(f"Additional context: {relevant}")
    parts.append("")
    parts.append("# What you are trying to find out")
    parts.append(
        "For each goal you have an INTENT (what you want to know), a STANDARD "
        '("answered well enough" rubric), and optionally a REDUNDANT_WHEN '
        'rubric ("skip if earlier answers covered this").'
    )
    parts.append("")
    for goal in conv.goals:
        redundant = goal.redundant_when or "(no redundancy rubric)"
        parts.append(f"## Goal {goal.id}")
        parts.append(f"INTENT: {goal.intent}")
        parts.append(f"STANDARD: {goal.standard}")
        parts.append(f"REDUNDANT_WHEN: {redundant}")
        parts.append("")
    parts.append("# Interviewing rules (must follow)")
    parts.append(
        '- One topic per question. No double-barreled "X and Y?" phrasing.'
    )
    parts.append(
        "- Neutral framing. No leading frames such as \"don't you\", "
        "\"wouldn't you\", \"isn't it true\", \"would you agree\", or "
        '"you must". No value judgments on the respondent\'s answers.'
    )
    parts.append(
        "- Funnel inside a goal: the first probe is broad "
        '("walk me through…", "tell me about…"); narrow only after the '
        "respondent sets the frame."
    )
    parts.append(
        "- Use the respondent's own vocabulary back. Do not substitute "
        "jargon they did not use."
    )
    parts.append(
        "- When pivoting to a new goal, open the utterance with a "
        "≤5-word acknowledgement of the previous answer before the "
        "question."
    )
    parts.append("")
    parts.append("# Voice phrasing rules (must follow)")
    parts.append('- One question per utterance. No "first... then..." enumerations.')
    parts.append("- 25 words or fewer per utterance.")
    parts.append("- Conversational, not written.")
    return "\n".join(parts)


def format_transcript_window(turns: list[Turn], max_turns: int) -> str:
    """Format the last ``max_turns`` turns, oldest first, for eval/compose."""
    if not turns:
        return "(no turns yet)"
    window = turns[-max_turns:]
    lines: list[str] = []
    for turn in window:
        prefix = "AGENT" if turn.speaker == "agent" else "RESPONDENT"
        lines.append(f"{prefix}: {turn.text}")
    return "\n".join(lines)


def format_full_transcript(turns: list[Turn]) -> str:
    """Format every turn with a ``[N]`` index prefix for derive_extract."""
    if not turns:
        return "(no turns)"
    lines: list[str] = []
    for turn in turns:
        prefix = "AGENT" if turn.speaker == "agent" else "RESPONDENT"
        lines.append(f"[{turn.index}] {prefix}: {turn.text}")
    return "\n".join(lines)


_PROBE_KIND_GUIDE = (
    "When next_action='probe', pick a probe_kind:\n"
    "- clarify: the answer was vague or hedged; ask them to restate or define what they meant.\n"
    "- example: ask for one concrete example to make an abstract claim concrete.\n"
    "- importance: ask why that matters to them.\n"
    "- contrast: ask how it compares to another case they could speak to.\n"
    "- elaborate: ask them to keep going / what happened next."
)


_CLARITY_GUIDE = (
    "Also assess the respondent's clarity by reading hedge language "
    '("I guess", "kind of", "maybe", "I think", "probably", "sort of") '
    "and answer length / specificity. Set `clarity`:\n"
    "- clear: concrete, no hedges.\n"
    "- hedged: some hedges, partial specificity.\n"
    "- vague: mostly hedges or platitudes; little concrete content."
)


_PROBE_KIND_SHAPES: dict[str, str] = {
    "clarify": (
        "Ask the respondent to restate or define what they meant — pick the "
        "one word or phrase that was vague."
    ),
    "example": "Ask for one concrete example.",
    "importance": "Ask why that matters to them.",
    "contrast": "Ask how it compares to another case.",
    "elaborate": "Ask them to keep going — what happened next.",
}


def build_evaluate_user_message(ctx: TurnContext, *, max_transcript_turns: int) -> str:
    """User message for evaluate_turn — narrow context plus active goal callout."""
    if ctx.active_goal is None:
        raise ValueError("evaluate_turn requires an active goal in TurnContext")
    transcript = format_transcript_window(ctx.transcript, max_transcript_turns)
    return (
        "The conversation so far:\n"
        f"{transcript}\n\n"
        f"The active goal is: {ctx.active_goal.id} — {ctx.active_goal.intent}\n"
        "The respondent's most recent answer is the final RESPONDENT turn above.\n\n"
        f"{_PROBE_KIND_GUIDE}\n\n"
        f"{_CLARITY_GUIDE}\n\n"
        "Call the `evaluate` tool with your assessment."
    )


def build_compose_user_message(
    ctx: TurnContext, eval_result: EvalResult, *, max_transcript_turns: int
) -> str:
    """User message for compose_utterance — eval result drives the prompt."""
    transcript = format_transcript_window(ctx.transcript, max_transcript_turns)
    lines: list[str] = [
        "The conversation so far:",
        transcript,
        "",
        "The previous evaluation determined:",
        f"- active_goal_status: {eval_result.active_goal_status}",
        f"- next_action: {eval_result.next_action}",
    ]
    if eval_result.next_action == "probe":
        if eval_result.probe_kind is not None:
            lines.append(f"- probe_kind: {eval_result.probe_kind}")
        if eval_result.interesting_tangent:
            lines.append(f"- interesting_tangent: {eval_result.interesting_tangent}")
    lines.append("")
    lines.append(
        "Write the next single voice utterance, following the voice phrasing "
        "rules. Output only the utterance text, no preamble."
    )
    if eval_result.next_action == "probe" and eval_result.probe_kind is not None:
        shape = _PROBE_KIND_SHAPES[eval_result.probe_kind]
        lines.append(f"Probe shape: {shape}")
    if eval_result.next_action in ("advance", "probe"):
        lines.append(
            "Lead with a brief acknowledgement of the previous answer "
            '(≤5 words, e.g. "Got it.", "Makes sense.", "Hmm.") and then '
            "ask the next question. Do not echo the answer back. The "
            "acknowledgement counts against the 25-word budget."
        )
    if ctx.last_phrasing_failure:
        lines.append(
            f"Your previous attempt failed: {ctx.last_phrasing_failure}. Fix it."
        )
    return "\n".join(lines)


def build_closing_recap_user_message(
    ctx: TurnContext, *, max_transcript_turns: int
) -> str:
    """User message for compose_closing_recap — one ≤25-word recap utterance.

    Asks the model to thank the respondent and name 1–2 concrete things
    they said. Reuses the same cached system prompt as evaluate /
    compose so no second cache key is introduced.
    """
    transcript = format_transcript_window(ctx.transcript, max_transcript_turns)
    return (
        "The conversation so far:\n"
        f"{transcript}\n\n"
        "The interview is now wrapping. Write a single closing utterance "
        "that (a) thanks the respondent and (b) names one or two "
        "concrete things they said. Follow the voice phrasing rules: one "
        "utterance, 25 words or fewer, no question. Output only the "
        "utterance text, no preamble."
    )


def build_extract_user_message(transcript: list[Turn]) -> str:
    """User message for derive_extract — full transcript + structured tool call."""
    formatted = format_full_transcript(transcript)
    return (
        "The full transcript of an interview:\n"
        f"{formatted}\n\n"
        "For each goal, decide the canonical status and which turn indices "
        "contain evidence. Also extract any unprompted findings — claims the "
        "respondent volunteered that weren't directly asked about.\n\n"
        "Call the `extract` tool with the structured Extract."
    )
