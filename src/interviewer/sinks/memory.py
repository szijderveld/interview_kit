"""In-memory EventSink — reference implementation for tests and examples."""

from __future__ import annotations

from interviewer.types.events import SessionEvent


class InMemoryEventSink:
    """Appends each emitted event to ``self.events`` in receive order."""

    def __init__(self) -> None:
        self.events: list[SessionEvent] = []

    async def emit(self, event: SessionEvent) -> None:
        self.events.append(event)
