"""Unit tests for :class:`AnthropicLLMClient` with a stubbed SDK (Step 12).

The Anthropic SDK is mocked at the client level via a fake passed into
``AnthropicLLMClient(client=...)``. The fakes mirror the duck-typed
attributes the client reads — ``response.content`` (a list of blocks
with ``type``/``name``/``input``), ``response.usage`` (with
``input_tokens`` etc.), and ``messages.stream`` (an async context
manager exposing ``text_stream`` + ``get_final_message``).
"""

from __future__ import annotations

from dataclasses import FrozenInstanceError, dataclass, field
from datetime import UTC, datetime
from typing import Any

import pytest

from interview_kit.llm.anthropic import AnthropicLLMClient, Usage
from interview_kit.types.config import Background, Conversation, Goal, Persona
from interview_kit.types.runtime import EvalResult, Turn, TurnContext

# ---------- SDK fakes ----------


@dataclass
class _FakeUsage:
    input_tokens: int = 100
    output_tokens: int = 50
    cache_read_input_tokens: int | None = 0
    cache_creation_input_tokens: int | None = 0


@dataclass
class _FakeToolUseBlock:
    name: str
    input: dict[str, Any]
    type: str = "tool_use"


@dataclass
class _FakeTextBlock:
    text: str
    type: str = "text"


@dataclass
class _FakeResponse:
    content: list[Any]
    usage: _FakeUsage = field(default_factory=_FakeUsage)


class _FakeStream:
    def __init__(self, chunks: list[str], final: _FakeResponse) -> None:
        self._chunks = list(chunks)
        self._final = final

    async def __aenter__(self) -> _FakeStream:
        return self

    async def __aexit__(self, *_args: object) -> None:
        return None

    @property
    def text_stream(self) -> Any:
        chunks = list(self._chunks)

        async def gen() -> Any:
            for c in chunks:
                yield c

        return gen()

    async def get_final_message(self) -> _FakeResponse:
        return self._final


class _FakeMessages:
    def __init__(
        self,
        *,
        create_response: _FakeResponse | None = None,
        stream_chunks: list[str] | None = None,
        stream_final: _FakeResponse | None = None,
    ) -> None:
        self.create_response = create_response
        self.stream_chunks = stream_chunks or []
        self.stream_final = stream_final
        self.create_calls: list[dict[str, Any]] = []
        self.stream_calls: list[dict[str, Any]] = []

    async def create(self, **kwargs: Any) -> _FakeResponse:
        self.create_calls.append(kwargs)
        assert self.create_response is not None
        return self.create_response

    def stream(self, **kwargs: Any) -> _FakeStream:
        self.stream_calls.append(kwargs)
        assert self.stream_final is not None
        return _FakeStream(self.stream_chunks, self.stream_final)


class _FakeAsyncAnthropic:
    def __init__(self, messages: _FakeMessages) -> None:
        self.messages = messages


# ---------- fixtures ----------


def _conv() -> Conversation:
    return Conversation(
        id="conv-1",
        persona=Persona(system_prompt="you are an interviewer", style="neutral", voice_id="v"),
        purpose="learn the day",
        background=Background(interviewee_role="r", interviewee_expertise="e"),
        goals=[
            Goal(id="g1", intent="i1", standard="s1"),
            Goal(id="g2", intent="i2", standard="s2"),
        ],
    )


def _ctx(conv: Conversation, transcript: list[Turn], active_id: str = "g1") -> TurnContext:
    return TurnContext(
        conversation=conv,
        transcript=transcript,
        active_goal=next(g for g in conv.goals if g.id == active_id),
        goal_statuses=[],
        retries_used_on_active=0,
        tangent_followups_used=0,
        total_turns=len(transcript),
    )


def _turn(index: int, speaker: str, text: str) -> Turn:
    return Turn(
        index=index,
        speaker=speaker,  # type: ignore[arg-type]
        text=text,
        timestamp=datetime.now(UTC),
    )


# ---------- evaluate_turn ----------


async def test_evaluate_turn_parses_tool_use_into_eval_result() -> None:
    conv = _conv()
    transcript = [_turn(0, "agent", "q"), _turn(1, "respondent", "a")]
    response = _FakeResponse(
        content=[
            _FakeToolUseBlock(
                name="evaluate",
                input={
                    "active_goal_status": "meets",
                    "redundant_goal_ids": ["g2"],
                    "interesting_tangent": None,
                    "next_action": "advance",
                    "rationale": "rituals named",
                },
            )
        ],
        usage=_FakeUsage(input_tokens=120, output_tokens=40),
    )
    messages = _FakeMessages(create_response=response)
    client = AnthropicLLMClient(client=_FakeAsyncAnthropic(messages))  # type: ignore[arg-type]

    result = await client.evaluate_turn(_ctx(conv, transcript))

    assert isinstance(result, EvalResult)
    assert result.active_goal_status == "meets"
    assert result.redundant_goal_ids == ["g2"]
    assert result.next_action == "advance"
    assert result.rationale == "rituals named"
    # Usage captured.
    assert client.last_eval_usage is not None
    assert client.last_eval_usage.tokens_in == 120
    assert client.last_eval_usage.tokens_out == 40


async def test_evaluate_turn_forces_evaluate_tool_choice_and_caches_system() -> None:
    conv = _conv()
    response = _FakeResponse(
        content=[
            _FakeToolUseBlock(
                name="evaluate",
                input={
                    "active_goal_status": "partial",
                    "next_action": "retry",
                    "rationale": "thin",
                },
            )
        ]
    )
    messages = _FakeMessages(create_response=response)
    client = AnthropicLLMClient(client=_FakeAsyncAnthropic(messages))  # type: ignore[arg-type]

    await client.evaluate_turn(_ctx(conv, [_turn(0, "respondent", "a")]))

    call = messages.create_calls[0]
    # Forced tool_choice.
    assert call["tool_choice"] == {"type": "tool", "name": "evaluate"}
    # System block has ephemeral cache_control (D2).
    system = call["system"]
    assert isinstance(system, list)
    assert system[0]["type"] == "text"
    assert system[0]["cache_control"] == {"type": "ephemeral"}
    # Tool input_schema has neither title nor $defs (D15 cleanup).
    schema = call["tools"][0]["input_schema"]
    assert "title" not in schema
    assert "$defs" not in schema


async def test_evaluate_turn_raises_when_active_goal_missing() -> None:
    conv = _conv()
    ctx = TurnContext(
        conversation=conv,
        transcript=[],
        active_goal=None,
        goal_statuses=[],
        retries_used_on_active=0,
        tangent_followups_used=0,
        total_turns=0,
    )
    messages = _FakeMessages(create_response=_FakeResponse(content=[]))
    client = AnthropicLLMClient(client=_FakeAsyncAnthropic(messages))  # type: ignore[arg-type]

    with pytest.raises(ValueError, match="active goal"):
        await client.evaluate_turn(ctx)


async def test_evaluate_turn_raises_when_no_tool_use_block_present() -> None:
    conv = _conv()
    response = _FakeResponse(content=[_FakeTextBlock(text="just text, no tool")])
    messages = _FakeMessages(create_response=response)
    client = AnthropicLLMClient(client=_FakeAsyncAnthropic(messages))  # type: ignore[arg-type]

    with pytest.raises(RuntimeError, match="no tool_use block"):
        await client.evaluate_turn(_ctx(conv, [_turn(0, "respondent", "a")]))


# ---------- compose_utterance ----------


async def test_compose_utterance_yields_chunks_and_captures_usage() -> None:
    conv = _conv()
    eval_result = EvalResult(active_goal_status="meets", next_action="advance")
    final = _FakeResponse(
        content=[_FakeTextBlock(text="What's your morning like?")],
        usage=_FakeUsage(input_tokens=80, output_tokens=20),
    )
    messages = _FakeMessages(
        stream_chunks=["What's ", "your ", "morning ", "like?"],
        stream_final=final,
    )
    client = AnthropicLLMClient(client=_FakeAsyncAnthropic(messages))  # type: ignore[arg-type]

    ctx = _ctx(conv, [_turn(0, "respondent", "a")])
    chunks: list[str] = []
    async for chunk in client.compose_utterance(ctx, eval_result):
        chunks.append(chunk)

    assert "".join(chunks) == "What's your morning like?"
    # Usage captured after iterator exhausted.
    assert client.last_compose_usage is not None
    assert client.last_compose_usage.tokens_in == 80
    assert client.last_compose_usage.tokens_out == 20


async def test_compose_utterance_system_block_is_cached() -> None:
    conv = _conv()
    eval_result = EvalResult(active_goal_status="meets", next_action="advance")
    messages = _FakeMessages(
        stream_chunks=["hi"],
        stream_final=_FakeResponse(content=[_FakeTextBlock(text="hi")]),
    )
    client = AnthropicLLMClient(client=_FakeAsyncAnthropic(messages))  # type: ignore[arg-type]

    async for _ in client.compose_utterance(_ctx(conv, []), eval_result):
        pass

    call = messages.stream_calls[0]
    system = call["system"]
    assert system[0]["cache_control"] == {"type": "ephemeral"}


# ---------- derive_extract ----------


async def test_derive_extract_parses_tool_input_into_full_extract() -> None:
    conv = _conv()
    transcript = [
        _turn(0, "agent", "q"),
        _turn(1, "respondent", "a"),
        _turn(2, "agent", "q2"),
        _turn(3, "respondent", "a2"),
    ]
    response = _FakeResponse(
        content=[
            _FakeToolUseBlock(
                name="extract",
                input={
                    "goal_statuses": [
                        {
                            "goal_id": "g1",
                            "status": "meets",
                            "evidence_turn_indices": [0, 1],
                            "retries_used": 0,
                            "rationale": "rituals covered",
                        },
                        {
                            "goal_id": "g2",
                            "status": "gave_up",
                            "evidence_turn_indices": [],
                            "retries_used": 1,
                            "rationale": "respondent unwilling",
                        },
                    ],
                    "unprompted_findings": [
                        {"text": "they use a new ERP", "evidence_turn_index": 3}
                    ],
                },
            )
        ],
        usage=_FakeUsage(input_tokens=800, output_tokens=200),
    )
    messages = _FakeMessages(create_response=response)
    client = AnthropicLLMClient(client=_FakeAsyncAnthropic(messages))  # type: ignore[arg-type]

    extract = await client.derive_extract(transcript, conv)

    assert extract.conversation_id == conv.id
    by_id = {gs.goal_id: gs for gs in extract.goal_statuses}
    assert by_id["g1"].status == "meets"
    assert by_id["g2"].status == "gave_up"
    assert by_id["g2"].retries_used == 1
    assert extract.unprompted_findings[0].text == "they use a new ERP"
    # Wrapper supplies full_transcript from the input, not the LLM.
    assert extract.full_transcript == transcript
    assert client.last_extract_usage is not None
    assert client.last_extract_usage.tokens_in == 800


async def test_derive_extract_forces_extract_tool_choice() -> None:
    conv = _conv()
    response = _FakeResponse(
        content=[
            _FakeToolUseBlock(
                name="extract",
                input={"goal_statuses": [], "unprompted_findings": []},
            )
        ]
    )
    messages = _FakeMessages(create_response=response)
    client = AnthropicLLMClient(client=_FakeAsyncAnthropic(messages))  # type: ignore[arg-type]

    await client.derive_extract([], conv)

    call = messages.create_calls[0]
    assert call["tool_choice"] == {"type": "tool", "name": "extract"}
    # input_schema cleaned of title and $defs.
    schema = call["tools"][0]["input_schema"]
    assert "title" not in schema
    assert "$defs" not in schema


# ---------- construction ----------


def test_constructor_requires_api_key_or_client() -> None:
    with pytest.raises(ValueError, match="api_key or client"):
        AnthropicLLMClient()


def test_usage_dataclass_is_frozen() -> None:
    usage = Usage(
        tokens_in=1,
        tokens_out=2,
        cache_read_tokens=3,
        cache_write_tokens=4,
        llm_latency_ms=5,
    )
    with pytest.raises(FrozenInstanceError):
        usage.tokens_in = 99  # type: ignore[misc]
