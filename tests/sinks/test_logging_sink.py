from __future__ import annotations

import logging
from datetime import UTC, datetime

import pytest

from interviewer import SessionEvent
from interviewer.protocols import EventSink
from interviewer.sinks.logging import LoggingEventSink


def _ts(seconds: int) -> datetime:
    return datetime(2026, 5, 10, 12, 0, seconds, tzinfo=UTC)


def _event(kind: str, seconds: int = 0) -> SessionEvent:
    return SessionEvent(
        session_id="s1",
        conversation_id="conv-1",
        timestamp=_ts(seconds),
        type=kind,  # type: ignore[arg-type]
        payload={"k": "v"},
    )


def _sink_as_protocol(sink: EventSink) -> EventSink:
    return sink


def test_protocol_conformance_static() -> None:
    _sink_as_protocol(LoggingEventSink())


async def test_emit_writes_log_record(caplog: pytest.LogCaptureFixture) -> None:
    sink = LoggingEventSink(logger_name="interviewer.events.test_emit")
    caplog.set_level(logging.INFO, logger="interviewer.events.test_emit")
    await sink.emit(_event("turn_recorded"))
    assert any(
        rec.levelno == logging.INFO
        and "type=turn_recorded" in rec.getMessage()
        and "session_id=s1" in rec.getMessage()
        for rec in caplog.records
    )


async def test_emit_uses_configured_level(caplog: pytest.LogCaptureFixture) -> None:
    sink = LoggingEventSink(
        logger_name="interviewer.events.test_level", level=logging.DEBUG
    )
    caplog.set_level(logging.DEBUG, logger="interviewer.events.test_level")
    await sink.emit(_event("completed"))
    assert any(rec.levelno == logging.DEBUG for rec in caplog.records)


async def test_emit_uses_injected_logger() -> None:
    captured: list[tuple[int, str]] = []

    class _CapturingHandler(logging.Handler):
        def emit(self, record: logging.LogRecord) -> None:
            captured.append((record.levelno, record.getMessage()))

    logger = logging.getLogger("interviewer.events.test_injected")
    logger.setLevel(logging.INFO)
    handler = _CapturingHandler()
    logger.addHandler(handler)
    try:
        sink = LoggingEventSink(logger=logger)
        await sink.emit(_event("session_provisioned"))
        await sink.emit(_event("respondent_joined", seconds=1))
    finally:
        logger.removeHandler(handler)

    assert len(captured) == 2
    assert "session_provisioned" in captured[0][1]
    assert "respondent_joined" in captured[1][1]
