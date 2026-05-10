"""Voice phrasing validator — word count + question-mark count.

Deterministic post-check on a composed agent utterance (D7). Returns the
list of failures rather than raising so the runner can choose between
regen-once and speak-verbatim. There is no keyword detector and no
clause splitter — Sonnet's prompt discipline is the primary mechanism;
this validator is only a guard against the egregious cases.
"""

from __future__ import annotations

from enum import StrEnum


class PhrasingFailure(StrEnum):
    EMPTY = "empty"
    TOO_LONG = "too_long"
    MULTI_QUESTION = "multi_question"


def validate_voice_phrasing(text: str, max_words: int = 25) -> list[PhrasingFailure]:
    """Return failures for the given utterance; empty list means pass.

    Checks (per D7):
      - EMPTY: ``text.split()`` yields no tokens.
      - TOO_LONG: more than ``max_words`` whitespace-separated tokens.
      - MULTI_QUESTION: more than one ``?`` character.
    """

    failures: list[PhrasingFailure] = []

    words = text.split()
    if not words:
        failures.append(PhrasingFailure.EMPTY)
    elif len(words) > max_words:
        failures.append(PhrasingFailure.TOO_LONG)

    if text.count("?") > 1:
        failures.append(PhrasingFailure.MULTI_QUESTION)

    return failures
