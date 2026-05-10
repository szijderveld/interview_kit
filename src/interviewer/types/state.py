"""Session state enum."""

from __future__ import annotations

from enum import StrEnum


class SessionState(StrEnum):
    """Lifecycle states for a Session."""

    CREATED = "created"
    READY = "ready"
    IN_PROGRESS = "in_progress"
    COMPLETED = "completed"
    ABANDONED = "abandoned"
    FAILED = "failed"
