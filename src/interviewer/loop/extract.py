"""Canonical Extract derivation — Step 11.

The runner reaches this module at the tail of :func:`run_loop`. The
in-memory ``goal_status_table`` (D13) is a loop-time hint authored by
``evaluate_turn`` decisions; this helper hands the full persisted
transcript to :meth:`LLMClient.derive_extract` and returns whatever the
client produced. Per SCOPE → Canonical truth, the LLM — not the runner
— is the source of truth for ``GoalStatus.evidence_turn_indices`` once
the session has finished. The runner overwrites ``Extract.session_id``
and ``Extract.completed_at`` after this call returns (DECISIONS Step 8);
this helper does not.
"""

from __future__ import annotations

from interviewer.protocols import LLMClient
from interviewer.types.config import Conversation
from interviewer.types.runtime import Extract, Turn


async def derive_extract_with_llm(
    transcript: list[Turn],
    conversation: Conversation,
    llm: LLMClient,
) -> Extract:
    """Pass-through to ``LLMClient.derive_extract`` for the canonical Extract."""
    return await llm.derive_extract(transcript, conversation)
