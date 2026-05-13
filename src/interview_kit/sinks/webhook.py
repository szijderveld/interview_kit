"""EventSink that POSTs each event to an HTTP webhook via ``httpx``.

Retries up to ``max_attempts`` times with exponential backoff on transport
errors and non-2xx responses. The final failure is re-raised; callers
choose whether to swallow it or propagate.
"""

from __future__ import annotations

import asyncio
import logging
from types import TracebackType
from typing import Self

import httpx

from interview_kit.types.events import SessionEvent

_DEFAULT_TIMEOUT = 5.0
_DEFAULT_MAX_ATTEMPTS = 3
_DEFAULT_INITIAL_BACKOFF = 0.1

_log = logging.getLogger(__name__)


class WebhookEventSink:
    """POSTs each ``SessionEvent`` as JSON to ``url`` via ``httpx.AsyncClient``."""

    def __init__(
        self,
        url: str,
        *,
        client: httpx.AsyncClient | None = None,
        timeout: float = _DEFAULT_TIMEOUT,
        max_attempts: int = _DEFAULT_MAX_ATTEMPTS,
        initial_backoff: float = _DEFAULT_INITIAL_BACKOFF,
    ) -> None:
        if max_attempts < 1:
            raise ValueError("max_attempts must be >= 1")
        self._url = url
        self._max_attempts = max_attempts
        self._initial_backoff = initial_backoff
        if client is None:
            self._client = httpx.AsyncClient(timeout=timeout)
            self._owns_client = True
        else:
            self._client = client
            self._owns_client = False

    async def emit(self, event: SessionEvent) -> None:
        body = event.model_dump(mode="json")
        last_exc: Exception | None = None
        for attempt in range(self._max_attempts):
            try:
                response = await self._client.post(self._url, json=body)
                response.raise_for_status()
                return
            except httpx.HTTPError as exc:
                last_exc = exc
                if attempt + 1 < self._max_attempts:
                    delay = self._initial_backoff * (2**attempt)
                    _log.warning(
                        "webhook emit failed (attempt %d/%d): %s; retrying in %.2fs",
                        attempt + 1,
                        self._max_attempts,
                        exc,
                        delay,
                    )
                    await asyncio.sleep(delay)
        assert last_exc is not None
        raise last_exc

    async def aclose(self) -> None:
        if self._owns_client:
            await self._client.aclose()

    async def __aenter__(self) -> Self:
        return self

    async def __aexit__(
        self,
        exc_type: type[BaseException] | None,
        exc: BaseException | None,
        tb: TracebackType | None,
    ) -> None:
        await self.aclose()
