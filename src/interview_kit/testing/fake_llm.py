"""Scripted LLM client for tests and the no-API-keys example.

``FakeLLMClient`` is deterministic: ``evaluate_turn`` pops from an
``EvalResult`` queue, ``compose_utterance`` pops from an utterance queue
and yields its text in 1–3 chunks (simulating streaming), and
``derive_extract`` maps every ``Turn.addressed_goal_ids`` reference to
``GoalStatus.evidence_turn_indices`` with ``status="meets"`` where there
is any evidence and ``"pending"`` otherwise.

``force_disagreement_for`` lets a test deliberately diverge the canonical
Extract from the runner's loop-time hint table: any goal id in that set
gets ``status="partial"`` in the returned Extract regardless of evidence,
so the diff path that emits ``goal_status_changed`` events can be
exercised end-to-end.

The runner overrides ``Extract.session_id`` and ``Extract.completed_at``
after this client returns — this client only has to produce a
structurally valid Extract using ``conv.id`` as a placeholder.
"""

from __future__ import annotations

from collections import deque
from collections.abc import AsyncIterator
from datetime import UTC, datetime

from interview_kit.types.config import Conversation
from interview_kit.types.runtime import (
    EvalResult,
    Extract,
    Finding,
    GoalStatus,
    GoalStatusValue,
    Turn,
    TurnContext,
)


class FakeLLMClient:
    """Deterministic ``LLMClient`` implementation for tests."""

    def __init__(
        self,
        *,
        eval_results: list[EvalResult] | None = None,
        utterances: list[str] | None = None,
        closings: list[str] | None = None,
        findings: list[Finding] | None = None,
        eval_failures: int = 0,
        compose_failures: int = 0,
        force_disagreement_for: list[str] | None = None,
    ) -> None:
        self._eval_results: deque[EvalResult] = deque(eval_results or [])
        self._utterances: deque[str] = deque(utterances or [])
        self._closings: deque[str] = deque(closings or [])
        self._findings: list[Finding] = list(findings or [])
        # Failure counters: each call decrements; while > 0, the method raises
        # so the runner's retry path is exercised. Set high (e.g., 100) to
        # simulate persistent failure, set to N for a transient outage that
        # recovers on attempt N+1.
        self._eval_failures_remaining = eval_failures
        self._compose_failures_remaining = compose_failures
        # Test hook: goal ids whose canonical status the fake client
        # should force to "partial" so the loop-time-hint diff path
        # emits ``goal_status_changed``. Never naturally produced.
        self._force_disagreement_for: frozenset[str] = frozenset(
            force_disagreement_for or []
        )

    async def evaluate_turn(self, ctx: TurnContext) -> EvalResult:
        if self._eval_failures_remaining > 0:
            self._eval_failures_remaining -= 1
            raise RuntimeError("FakeLLMClient: simulated evaluate_turn failure")
        if not self._eval_results:
            raise RuntimeError("FakeLLMClient: evaluate_turn queue exhausted")
        return self._eval_results.popleft()

    def compose_utterance(
        self, ctx: TurnContext, eval_result: EvalResult
    ) -> AsyncIterator[str]:
        if self._compose_failures_remaining > 0:
            self._compose_failures_remaining -= 1
            raise RuntimeError("FakeLLMClient: simulated compose_utterance failure")
        if not self._utterances:
            raise RuntimeError("FakeLLMClient: compose_utterance queue exhausted")
        text = self._utterances.popleft()
        return _stream_chunks(text)

    async def compose_closing_recap(self, ctx: TurnContext) -> str:
        if self._closings:
            return self._closings.popleft()
        return "Thanks — appreciated your time and the detail you shared."

    async def derive_extract(
        self, transcript: list[Turn], conv: Conversation
    ) -> Extract:
        goal_statuses: list[GoalStatus] = []
        for goal in conv.goals:
            evidence = [t.index for t in transcript if goal.id in t.addressed_goal_ids]
            status: GoalStatusValue
            rationale: str
            if goal.id in self._force_disagreement_for:
                status = "partial"
                rationale = "fake-llm: forced disagreement for test"
            else:
                status = "meets" if evidence else "pending"
                rationale = "fake-llm: addressed_goal_ids mapping"
            goal_statuses.append(
                GoalStatus(
                    goal_id=goal.id,
                    status=status,
                    evidence_turn_indices=evidence,
                    retries_used=0,
                    rationale=rationale,
                )
            )
        # session_id and completed_at are runner-owned; conv.id is a valid
        # non-empty placeholder so model validation passes.
        return Extract(
            session_id=conv.id,
            conversation_id=conv.id,
            goal_statuses=goal_statuses,
            unprompted_findings=list(self._findings),
            full_transcript=transcript,
            completed_at=datetime.now(UTC),
        )


async def _stream_chunks(text: str) -> AsyncIterator[str]:
    """Yield ``text`` in 1–3 chunks to simulate streaming."""
    if len(text) <= 20:
        yield text
        return
    third = len(text) // 3
    yield text[:third]
    yield text[third : third * 2]
    yield text[third * 2 :]
