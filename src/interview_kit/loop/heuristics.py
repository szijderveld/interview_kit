"""Refusal / IDK detection — case-insensitive keyword scan over respondent text.

The LLM does not decide refusal — this pre-check does, so the loop can
react before paying for an eval call. False positives are tolerated
because the recovery path (one deflection probe) is cheap; false
negatives let the loop continue normally, which the next
``evaluate_turn`` will catch.
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


def detect_refusal_or_idk(text: str) -> bool:
    """Return True if ``text`` matches any refusal or IDK keyword."""
    lower = text.lower()
    return any(k in lower for k in REFUSAL_KEYWORDS) or any(
        k in lower for k in IDK_KEYWORDS
    )
