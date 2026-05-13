"""Refusal / IDK detection — case-insensitive keyword scan over respondent text.

The LLM does not decide refusal/IDK — these pre-checks do, so the loop
can react before paying for an eval call. The two predicates trigger
different paths in the runner:

- :func:`detect_refusal` matches a *consent* boundary ("I'd rather not").
  One hit is enough: the active goal is marked ``skipped_refused`` and
  the loop advances without a deflection probe.
- :func:`detect_idk` matches a *knowledge* gap ("I don't know"). The
  runner gives this one deflection probe; two consecutive hits mark the
  goal ``gave_up``.

False positives are tolerated because the IDK recovery path (one
deflection probe) is cheap; false negatives let the loop continue
normally, which the next ``evaluate_turn`` will catch.
"""

from __future__ import annotations

REFUSAL_KEYWORDS: list[str] = [
    "won't answer",
    "wont answer",
    "rather not",
    "prefer not",
    "no comment",
    "not going to answer",
    "decline to answer",
]

IDK_KEYWORDS: list[str] = [
    "don't know",
    "dont know",
    "no idea",
    "no clue",
    "not sure",
    "can't say",
    "cant say",
    "couldn't say",
    "couldnt say",
]


def detect_refusal(text: str) -> bool:
    """Return True if ``text`` matches any refusal (consent-decline) keyword."""
    lower = text.lower()
    return any(k in lower for k in REFUSAL_KEYWORDS)


def detect_idk(text: str) -> bool:
    """Return True if ``text`` matches any IDK (knowledge-gap) keyword."""
    lower = text.lower()
    return any(k in lower for k in IDK_KEYWORDS)
