"""Selection logic tests — pure function, no I/O, no LLM."""

from __future__ import annotations

import pytest
from pydantic import ValidationError

from interview_kit.loop.selection import select_next_goal
from interview_kit.types.config import Background, Conversation, Goal, Persona
from interview_kit.types.runtime import GoalStatus, GoalStatusValue


def _persona() -> Persona:
    return Persona(
        system_prompt="You are an interview_kit.",
        style="neutral",
        voice_id="voice-1",
    )


def _background() -> Background:
    return Background(interviewee_role="role", interviewee_expertise="expertise")


def _conversation(goals: list[Goal]) -> Conversation:
    return Conversation(
        id="c1",
        persona=_persona(),
        purpose="purpose",
        background=_background(),
        goals=goals,
    )


def _gs(goal_id: str, status: GoalStatusValue) -> GoalStatus:
    return GoalStatus(goal_id=goal_id, status=status)


def test_returns_first_pending_by_default_order() -> None:
    conv = _conversation(
        [
            Goal(id="a", intent="i", standard="s"),
            Goal(id="b", intent="i", standard="s"),
            Goal(id="c", intent="i", standard="s"),
        ]
    )
    assert select_next_goal(conv, []) == conv.goals[0]


def test_skips_resolved_goals_meets_skipped_gave_up() -> None:
    conv = _conversation(
        [
            Goal(id="a", intent="i", standard="s"),
            Goal(id="b", intent="i", standard="s"),
            Goal(id="c", intent="i", standard="s"),
            Goal(id="d", intent="i", standard="s"),
        ]
    )
    statuses = [
        _gs("a", "meets"),
        _gs("b", "skipped_redundant"),
        _gs("c", "gave_up"),
    ]
    chosen = select_next_goal(conv, statuses)
    assert chosen is not None and chosen.id == "d"


def test_partial_status_remains_eligible() -> None:
    conv = _conversation(
        [
            Goal(id="a", intent="i", standard="s"),
            Goal(id="b", intent="i", standard="s"),
        ]
    )
    chosen = select_next_goal(conv, [_gs("a", "partial")])
    assert chosen is not None and chosen.id == "a"


def test_dependency_ordering_blocks_dependent_until_dep_meets() -> None:
    conv = _conversation(
        [
            Goal(id="dep", intent="i", standard="s"),
            Goal(id="leaf", intent="i", standard="s", depends_on=["dep"]),
        ]
    )
    # leaf is blocked by unmet dep → first eligible is dep
    assert select_next_goal(conv, []) == conv.goals[0]

    # once dep meets, leaf becomes eligible
    chosen = select_next_goal(conv, [_gs("dep", "meets")])
    assert chosen is not None and chosen.id == "leaf"


def test_dependency_satisfied_by_skipped_redundant() -> None:
    conv = _conversation(
        [
            Goal(id="dep", intent="i", standard="s"),
            Goal(id="leaf", intent="i", standard="s", depends_on=["dep"]),
        ]
    )
    chosen = select_next_goal(conv, [_gs("dep", "skipped_redundant")])
    assert chosen is not None and chosen.id == "leaf"


def test_dependency_not_satisfied_by_gave_up_or_partial() -> None:
    conv = _conversation(
        [
            Goal(id="dep", intent="i", standard="s"),
            Goal(id="leaf", intent="i", standard="s", depends_on=["dep"]),
        ]
    )
    # gave_up blocks the dependent; dep itself is resolved, so leaf has no
    # eligible siblings — result is None.
    assert select_next_goal(conv, [_gs("dep", "gave_up")]) is None

    # partial blocks dependent → only dep is eligible
    chosen = select_next_goal(conv, [_gs("dep", "partial")])
    assert chosen is not None and chosen.id == "dep"


def test_skips_dependent_until_all_deps_satisfied() -> None:
    conv = _conversation(
        [
            Goal(id="a", intent="i", standard="s"),
            Goal(id="b", intent="i", standard="s"),
            Goal(id="leaf", intent="i", standard="s", depends_on=["a", "b"]),
        ]
    )
    # one dep met, the other still pending → leaf still blocked, b chosen
    chosen = select_next_goal(conv, [_gs("a", "meets")])
    assert chosen is not None and chosen.id == "b"

    chosen = select_next_goal(conv, [_gs("a", "meets"), _gs("b", "meets")])
    assert chosen is not None and chosen.id == "leaf"


def test_returns_none_when_everything_resolved() -> None:
    conv = _conversation(
        [
            Goal(id="a", intent="i", standard="s"),
            Goal(id="b", intent="i", standard="s"),
        ]
    )
    statuses = [_gs("a", "meets"), _gs("b", "gave_up")]
    assert select_next_goal(conv, statuses) is None


def test_unknown_goal_ids_in_statuses_are_ignored() -> None:
    conv = _conversation([Goal(id="a", intent="i", standard="s")])
    # stale status for a no-longer-listed goal should not crash
    chosen = select_next_goal(conv, [_gs("ghost", "meets")])
    assert chosen is not None and chosen.id == "a"


def test_no_infinite_loop_on_long_dependency_chain() -> None:
    # Conversation.validate already rejects cycles; selection itself does
    # no graph traversal — a deep linear chain should still resolve in O(n).
    goals: list[Goal] = []
    for i in range(50):
        deps = [f"g{i - 1}"] if i > 0 else []
        goals.append(Goal(id=f"g{i}", intent="i", standard="s", depends_on=deps))
    conv = _conversation(goals)
    chosen = select_next_goal(conv, [])
    assert chosen is not None and chosen.id == "g0"


def test_dependency_cycle_is_rejected_at_conversation_validation() -> None:
    with pytest.raises(ValidationError):
        _conversation(
            [
                Goal(id="a", intent="i", standard="s", depends_on=["b"]),
                Goal(id="b", intent="i", standard="s", depends_on=["a"]),
            ]
        )
