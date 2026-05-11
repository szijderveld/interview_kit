from __future__ import annotations

import json
from datetime import UTC, datetime
from typing import Any

import httpx
import pytest

from interviewer import SessionEvent
from interviewer.protocols import EventSink
from interviewer.sinks.webhook import WebhookEventSink


def _ts() -> datetime:
    return datetime(2026, 5, 10, 12, 0, 0, tzinfo=UTC)


def _event(kind: str = "turn_recorded") -> SessionEvent:
    return SessionEvent(
        session_id="s1",
        conversation_id="conv-1",
        timestamp=_ts(),
        type=kind,  # type: ignore[arg-type]
        payload={"index": 3},
    )


def _sink_as_protocol(sink: EventSink) -> EventSink:
    return sink


def test_protocol_conformance_static() -> None:
    _sink_as_protocol(WebhookEventSink("https://example.invalid/hook"))


def _build_sink(
    handler: httpx.MockTransport,
    *,
    max_attempts: int = 3,
    initial_backoff: float = 0.0,
) -> WebhookEventSink:
    client = httpx.AsyncClient(transport=handler)
    return WebhookEventSink(
        "https://example.invalid/hook",
        client=client,
        max_attempts=max_attempts,
        initial_backoff=initial_backoff,
    )


async def test_emit_posts_event_as_json() -> None:
    captured: list[tuple[str, str, dict[str, Any]]] = []

    def handler(request: httpx.Request) -> httpx.Response:
        captured.append(
            (str(request.url), request.method, json.loads(request.read()))
        )
        return httpx.Response(200, json={"ok": True})

    transport = httpx.MockTransport(handler)
    async with _build_sink(transport) as sink:
        await sink.emit(_event("turn_recorded"))

    assert len(captured) == 1
    url, method, body = captured[0]
    assert url == "https://example.invalid/hook"
    assert method == "POST"
    assert body["type"] == "turn_recorded"
    assert body["session_id"] == "s1"
    assert body["conversation_id"] == "conv-1"
    assert body["payload"] == {"index": 3}


async def test_emit_retries_on_transient_failure_then_succeeds() -> None:
    calls = {"n": 0}

    def handler(request: httpx.Request) -> httpx.Response:
        calls["n"] += 1
        if calls["n"] < 3:
            return httpx.Response(503)
        return httpx.Response(200)

    transport = httpx.MockTransport(handler)
    async with _build_sink(transport, max_attempts=3) as sink:
        await sink.emit(_event())

    assert calls["n"] == 3


async def test_emit_raises_after_max_attempts() -> None:
    calls = {"n": 0}

    def handler(request: httpx.Request) -> httpx.Response:
        calls["n"] += 1
        return httpx.Response(500)

    transport = httpx.MockTransport(handler)
    async with _build_sink(transport, max_attempts=3) as sink:
        with pytest.raises(httpx.HTTPStatusError):
            await sink.emit(_event())

    assert calls["n"] == 3


async def test_emit_retries_on_transport_error_then_raises() -> None:
    calls = {"n": 0}

    def handler(request: httpx.Request) -> httpx.Response:
        calls["n"] += 1
        raise httpx.ConnectError("simulated transport error")

    transport = httpx.MockTransport(handler)
    async with _build_sink(transport, max_attempts=2) as sink:
        with pytest.raises(httpx.ConnectError):
            await sink.emit(_event())

    assert calls["n"] == 2


async def test_owned_client_is_closed_on_aclose() -> None:
    sink = WebhookEventSink("https://example.invalid/hook")
    # Sanity: a fresh owned client is not yet closed.
    assert sink._client.is_closed is False  # type: ignore[attr-defined]
    await sink.aclose()
    assert sink._client.is_closed is True  # type: ignore[attr-defined]


async def test_injected_client_not_closed_on_aclose() -> None:
    def handler(_: httpx.Request) -> httpx.Response:
        return httpx.Response(200)

    transport = httpx.MockTransport(handler)
    client = httpx.AsyncClient(transport=transport)
    sink = WebhookEventSink("https://example.invalid/hook", client=client)
    await sink.aclose()
    assert client.is_closed is False
    await client.aclose()
