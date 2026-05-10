"""Operator-authored configuration types — Persona, Background, Goal, Conversation.

Frozen Pydantic models. Mutate only via ``model_copy(update={...})``.
"""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, model_validator

PersonaStyle = Literal["warm", "neutral", "terse"]

_RELEVANT_CONTEXT_MAX = 1000


class Persona(BaseModel):
    """How the interviewer presents itself in voice."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    system_prompt: str = Field(min_length=1)
    style: PersonaStyle
    voice_id: str = Field(min_length=1)


class Background(BaseModel):
    """Structured context about who's being interviewed and why."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    interviewee_role: str = Field(min_length=1)
    interviewee_expertise: str = Field(min_length=1)
    # D6: over-length raises, never silently truncates.
    relevant_context: str = Field(default="", max_length=_RELEVANT_CONTEXT_MAX)


class Goal(BaseModel):
    """One thing the operator wants to find out."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    id: str = Field(min_length=1)
    intent: str = Field(min_length=1)
    standard: str = Field(min_length=1)
    max_retries: int = Field(default=2, ge=0)
    depends_on: list[str] = Field(default_factory=list)
    redundant_when: str = ""


class Conversation(BaseModel):
    """Configuration template — persona, purpose, background, goals."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    id: str = Field(min_length=1)
    persona: Persona
    purpose: str = Field(min_length=1)
    background: Background
    goals: list[Goal]
    opening: str | None = None
    closing: str | None = None
    max_tangent_followups: int = Field(default=2, ge=0)
    max_total_turns: int = Field(default=80, ge=4)

    @model_validator(mode="after")
    def _validate_goal_graph(self) -> Conversation:
        if not self.goals:
            raise ValueError("Conversation must have at least one goal")

        ids: list[str] = [g.id for g in self.goals]
        if len(set(ids)) != len(ids):
            raise ValueError("Goal ids must be unique within a Conversation")

        id_set = set(ids)
        for goal in self.goals:
            for dep in goal.depends_on:
                if dep not in id_set:
                    raise ValueError(
                        f"Goal {goal.id!r} depends on unknown goal id {dep!r}"
                    )
                if dep == goal.id:
                    raise ValueError(f"Goal {goal.id!r} cannot depend on itself")

        _check_no_cycles(self.goals)
        return self


def _check_no_cycles(goals: list[Goal]) -> None:
    graph: dict[str, list[str]] = {g.id: g.depends_on for g in goals}
    WHITE, GRAY, BLACK = 0, 1, 2
    color: dict[str, int] = dict.fromkeys(graph, WHITE)

    def visit(node: str) -> None:
        color[node] = GRAY
        for nxt in graph[node]:
            if color[nxt] == GRAY:
                raise ValueError(f"Goal dependency cycle detected at {nxt!r}")
            if color[nxt] == WHITE:
                visit(nxt)
        color[node] = BLACK

    for node in graph:
        if color[node] == WHITE:
            visit(node)
