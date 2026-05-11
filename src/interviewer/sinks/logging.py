"""EventSink that writes each event to a ``logging.Logger``."""

from __future__ import annotations

import logging

from interviewer.types.events import SessionEvent

_DEFAULT_LOGGER_NAME = "interviewer.events"


class LoggingEventSink:
    """Emits one log record per event on a configurable logger."""

    def __init__(
        self,
        *,
        logger: logging.Logger | None = None,
        logger_name: str = _DEFAULT_LOGGER_NAME,
        level: int = logging.INFO,
    ) -> None:
        self._logger = logger if logger is not None else logging.getLogger(logger_name)
        self._level = level

    async def emit(self, event: SessionEvent) -> None:
        self._logger.log(
            self._level,
            "session_event session_id=%s type=%s payload=%s",
            event.session_id,
            event.type,
            event.payload,
        )
