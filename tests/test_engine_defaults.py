"""Tests for Engine.with_defaults (Step 19)."""

from __future__ import annotations

import pytest

from interview_kit import Engine
from interview_kit.llm.anthropic import AnthropicLLMClient
from interview_kit.sinks.memory import InMemoryEventSink
from interview_kit.stores.memory import InMemoryConversationStore


def test_with_defaults_wires_in_memory_components_and_anthropic_llm() -> None:
    engine = Engine.with_defaults(anthropic_api_key="sk-test-fake")
    assert isinstance(engine.store, InMemoryConversationStore)
    assert isinstance(engine.events, InMemoryEventSink)
    assert isinstance(engine.llm, AnthropicLLMClient)
    assert engine.livekit is None


def test_with_defaults_reads_env_var(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("ANTHROPIC_API_KEY", "sk-env-fake")
    engine = Engine.with_defaults()
    assert isinstance(engine.llm, AnthropicLLMClient)


def test_with_defaults_raises_when_no_key(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
    with pytest.raises(ValueError, match="ANTHROPIC_API_KEY"):
        Engine.with_defaults()


def test_with_defaults_explicit_key_overrides_env(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("ANTHROPIC_API_KEY", "sk-env")
    engine = Engine.with_defaults(anthropic_api_key="sk-explicit")
    assert isinstance(engine.llm, AnthropicLLMClient)
