from __future__ import annotations

from datetime import UTC, datetime

from interview_kit import SessionEvent
from interview_kit.protocols import EventSink
from interview_kit.sinks.memory import InMemoryEventSink


def _ts(seconds: int) -> datetime:
    return datetime(2026, 5, 10, 12, 0, seconds, tzinfo=UTC)


def _event(kind: str, seconds: int) -> SessionEvent:
    return SessionEvent(
        session_id="s1",
        conversation_id="conv-1",
        timestamp=_ts(seconds),
        type=kind,  # type: ignore[arg-type]
    )


def _sink_as_protocol(sink: EventSink) -> EventSink:
    """Static check that InMemoryEventSink satisfies EventSink."""
    return sink


def test_protocol_conformance_static() -> None:
    _sink_as_protocol(InMemoryEventSink())


def test_initial_events_list_is_empty() -> None:
    assert InMemoryEventSink().events == []


async def test_emit_appends_in_order() -> None:
    sink = InMemoryEventSink()
    a = _event("session_provisioned", 0)
    b = _event("respondent_joined", 1)
    c = _event("turn_recorded", 2)
    await sink.emit(a)
    await sink.emit(b)
    await sink.emit(c)
    assert sink.events == [a, b, c]
