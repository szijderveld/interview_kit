"""Scripted opening for sessions whose operator did not author one.

A receiver who joins a session with ``Conversation.opening = None``
still needs a coherent welcome: who is talking, that the session is
recorded, and a ready-check so they don't talk over the first probe.

:data:`DEFAULT_OPENING` is one short utterance that covers all three
beats inside the voice-phrasing budget (≤25 words). It's a fixed
known-good string — phrasing-validator-clean — so the runner can speak
it without running it through :func:`validate_voice_phrasing` again.

Operator-authored :attr:`Conversation.opening` keeps precedence; this
constant is only used when the operator left it ``None``.
"""

from __future__ import annotations

DEFAULT_OPENING = (
    "Hi — thanks for joining. I'll ask a few quick questions. "
    "We'll record this so I can review. Ready when you are?"
)
