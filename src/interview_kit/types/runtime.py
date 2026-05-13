"""Runtime types for a live or completed Session.

Frozen Pydantic models; mutate via ``model_copy(update={...})``. The
Conversation a Session ran against is snapshotted onto
``Session.conversation_snapshot`` so mid-call edits to the underlying
Conversation never affect an in-flight session.
"""

from __future__ import annotations

from datetime import datetime
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, model_validator

from interview_kit.types.config import Conversation, Goal
from interview_kit.types.state import SessionState

GoalStatusValue = Literal["pending", "meets", "partial", "skipped_redundant", "gave_up"]
EvalGoalStatusValue = Literal["pending", "meets", "partial", "gave_up"]
NextAction = Literal["advance", "retry", "probe", "close"]
ProbeKind = Literal["clarify", "example", "importance", "contrast", "elaborate"]
Speaker = Literal["agent", "respondent"]


class Session(BaseModel):
    """One run of a Conversation with one respondent."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    id: str = Field(min_length=1)
    conversation_id: str = Field(min_length=1)
    conversation_snapshot: Conversation
    state: SessionState = SessionState.CREATED
    created_at: datetime
    started_at: datetime | None = None
    completed_at: datetime | None = None


class SessionCredentials(BaseModel):
    """What the consumer embeds in a shareable link."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    room_url: str = Field(min_length=1)
    token: str = Field(min_length=1)
    expires_at: datetime


class SessionRuntimeState(BaseModel):
    """In-flight loop state, flushed before every agent utterance."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    session_id: str = Field(min_length=1)
    active_goal_id: str | None = None
    retries_used_on_active: int = Field(ge=0)
    tangent_followups_used: int = Field(ge=0)
    total_turns: int = Field(ge=0)
    pending_follow_up: str | None = None
    last_event_index: int = Field(ge=0)
    updated_at: datetime


class Turn(BaseModel):
    """One utterance by one party."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    index: int = Field(ge=0)
    speaker: Speaker
    text: str
    timestamp: datetime
    # Live hint set by evaluate_turn; NOT canonical — the canonical
    # mapping comes from derive_extract at session end.
    addressed_goal_ids: list[str] = Field(default_factory=list)


class GoalStatus(BaseModel):
    """Per-goal outcome. ``evidence_turn_indices`` is canonical only after derive_extract."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    goal_id: str = Field(min_length=1)
    status: GoalStatusValue
    evidence_turn_indices: list[int] = Field(default_factory=list)
    retries_used: int = Field(default=0, ge=0)
    rationale: str = ""


class Finding(BaseModel):
    """An unprompted claim the respondent volunteered."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    text: str = Field(min_length=1)
    evidence_turn_index: int = Field(ge=0)
    category: str | None = None


class Extract(BaseModel):
    """Structured output for one Session, written at completion."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    session_id: str = Field(min_length=1)
    conversation_id: str = Field(min_length=1)
    goal_statuses: list[GoalStatus]
    unprompted_findings: list[Finding] = Field(default_factory=list)
    full_transcript: list[Turn]
    completed_at: datetime


class SessionStatus(BaseModel):
    """Cheap dashboard read; aggregated from store, may be one turn stale."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    session_id: str = Field(min_length=1)
    state: SessionState
    active_goal_id: str | None = None
    total_turns: int = Field(ge=0)
    goals_resolved: int = Field(ge=0)
    goals_total: int = Field(ge=0)
    started_at: datetime | None = None
    last_turn_at: datetime | None = None


class TurnContext(BaseModel):
    """Everything the LLM needs for one compose-or-evaluate call."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    conversation: Conversation
    transcript: list[Turn]
    active_goal: Goal | None = None
    goal_statuses: list[GoalStatus]
    retries_used_on_active: int = Field(ge=0)
    tangent_followups_used: int = Field(ge=0)
    total_turns: int = Field(ge=0)
    last_phrasing_failure: str | None = None


class EvalResult(BaseModel):
    """Output of ``LLMClient.evaluate_turn``: structured judgment for the active goal."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    active_goal_status: EvalGoalStatusValue
    redundant_goal_ids: list[str] = Field(default_factory=list)
    interesting_tangent: str | None = None
    next_action: NextAction
    probe_kind: ProbeKind | None = None
    rationale: str = ""

    @model_validator(mode="after")
    def _probe_kind_matches_action(self) -> EvalResult:
        if self.next_action == "probe" and self.probe_kind is None:
            raise ValueError("probe_kind is required when next_action='probe'")
        if self.next_action != "probe" and self.probe_kind is not None:
            raise ValueError("probe_kind must be None unless next_action='probe'")
        return self
