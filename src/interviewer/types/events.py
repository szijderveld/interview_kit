"""Lifecycle events emitted through the EventSink.

The literal union in ``SessionEvent.type`` lists every event type v1 emits.
Notable conventions:

- ``goal_status_changed`` is emitted **only at completion** (D5). The
  loop's mid-session goal-status updates are an internal hint table; the
  canonical statuses come from ``derive_extract`` at session end, and the
  diff (if any) is surfaced as one ``goal_status_changed`` per differing
  goal, ordered before the ``completed`` event.
- ``turn_recorded.payload`` includes LLM usage telemetry for the agent's
  compose call (D11): ``tokens_in``, ``tokens_out``, ``cache_read_tokens``,
  ``cache_write_tokens``, ``llm_latency_ms``. Eval-call usage is
  aggregated separately on the ``completed`` event.

``payload`` is intentionally schemaless at the type level: each event type
has its own payload shape documented here and at the emit site. A static
TypedDict union per event type is feasible later but adds churn now.
"""

from __future__ import annotations

from datetime import datetime
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field

SessionEventType = Literal[
    "session_provisioned",
    "respondent_joined",
    "turn_recorded",
    "goal_status_changed",
    "completed",
    "abandoned",
    "failed",
]


class SessionEvent(BaseModel):
    """One lifecycle event from a Session."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    session_id: str = Field(min_length=1)
    conversation_id: str = Field(min_length=1)
    timestamp: datetime
    type: SessionEventType
    # Any: payload is heterogeneous and event-type-specific; per-type shape
    # is documented in this module's docstring and at the emit sites.
    payload: dict[str, Any] = Field(default_factory=dict)
