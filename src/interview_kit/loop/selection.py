"""Goal selection — pure function over Conversation + GoalStatus table.

Redundancy is decided inside ``evaluate_turn`` and surfaced by the
runner as ``GoalStatus.status == "skipped_redundant"`` on the relevant
entries. Selection is downstream of that — it does not call any LLM.
"""

from __future__ import annotations

from interview_kit.types.config import Conversation, Goal
from interview_kit.types.runtime import GoalStatus, GoalStatusValue

_RESOLVED: frozenset[GoalStatusValue] = frozenset(
    {"meets", "skipped_redundant", "skipped_refused", "gave_up"}
)
_DEP_SATISFIED: frozenset[GoalStatusValue] = frozenset({"meets", "skipped_redundant"})


def select_next_goal(
    conversation: Conversation, goal_statuses: list[GoalStatus]
) -> Goal | None:
    """Return the next goal to probe, or None when all goals are resolved.

    Eligible goals are those whose status is NOT in {meets, skipped_redundant,
    gave_up} and whose declared ``depends_on`` are all in {meets,
    skipped_redundant} (a redundant dependency counts as satisfied — the
    earlier answers already covered it).

    Among eligible goals the first one by operator-declared order in
    ``conversation.goals`` wins.
    """

    status_by_id: dict[str, GoalStatusValue] = {
        gs.goal_id: gs.status for gs in goal_statuses
    }

    for goal in conversation.goals:
        current = status_by_id.get(goal.id, "pending")
        if current in _RESOLVED:
            continue
        if not all(status_by_id.get(dep, "pending") in _DEP_SATISFIED for dep in goal.depends_on):
            continue
        return goal

    return None
