"""Voice phrasing validator tests — D7."""

from __future__ import annotations

import pytest

from interview_kit.loop.phrasing import PhrasingFailure, validate_voice_phrasing


def test_happy_case_short_single_question() -> None:
    assert validate_voice_phrasing("What does a typical morning look like?") == []


def test_short_statement_no_question_mark_is_fine() -> None:
    # Not every agent utterance is a question — opening / closing lines pass.
    assert validate_voice_phrasing("Thanks for taking the time today.") == []


def test_exactly_max_words_passes() -> None:
    text = " ".join(["word"] * 25)
    assert validate_voice_phrasing(text) == []


def test_one_over_max_words_flags_too_long() -> None:
    text = " ".join(["word"] * 26)
    assert validate_voice_phrasing(text) == [PhrasingFailure.TOO_LONG]


def test_multi_question_flagged() -> None:
    text = "What's first? And what's next?"
    failures = validate_voice_phrasing(text)
    assert PhrasingFailure.MULTI_QUESTION in failures
    assert PhrasingFailure.TOO_LONG not in failures


def test_too_long_and_multi_question_both_flagged() -> None:
    base = " ".join(["word"] * 26)
    text = f"{base}? And really?"
    failures = validate_voice_phrasing(text)
    assert PhrasingFailure.TOO_LONG in failures
    assert PhrasingFailure.MULTI_QUESTION in failures


def test_empty_string_flagged_empty() -> None:
    assert validate_voice_phrasing("") == [PhrasingFailure.EMPTY]


def test_whitespace_only_flagged_empty() -> None:
    assert validate_voice_phrasing("   \n\t  ") == [PhrasingFailure.EMPTY]


def test_single_question_mark_is_not_multi_question() -> None:
    assert validate_voice_phrasing("Can you walk me through it?") == []


def test_custom_max_words_override() -> None:
    text = " ".join(["word"] * 10)
    assert validate_voice_phrasing(text, max_words=5) == [PhrasingFailure.TOO_LONG]
    assert validate_voice_phrasing(text, max_words=10) == []


@pytest.mark.parametrize(
    "text",
    [
        "What?",
        "Could you elaborate?",
        " ".join(["w"] * 25) + "?",
    ],
)
def test_parametrized_passing_utterances(text: str) -> None:
    assert validate_voice_phrasing(text) == []
