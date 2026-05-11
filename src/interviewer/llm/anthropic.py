"""Anthropic-backed :class:`LLMClient` — Step 12.

Two sequential Anthropic calls per agent turn (D2):

1. :meth:`evaluate_turn` — Claude Haiku 4.5 with forced tool-use,
   returning a structured :class:`EvalResult`.
2. :meth:`compose_utterance` — Claude Sonnet 4.6 with plain-text
   streaming. Yields tokens as they arrive so the voice transport
   (Step 13) can begin TTS playback before the full utterance is
   generated.

:meth:`derive_extract` is a single non-streaming Sonnet 4.6 call at
session end. Its tool schema is a *partial* Extract — the runner owns
``session_id``, ``completed_at`` (DECISIONS Step 8) and the wrapper
reconstructs ``full_transcript`` from the input, so the LLM is only
asked to fill in ``goal_statuses`` + ``unprompted_findings``.

The system prompt is identical across all three methods so the
ephemeral prompt cache (5-minute window) is shared between Haiku and
Sonnet calls inside the same session. The per-turn user message
varies and is intentionally NOT cached.

Per-call usage is exposed via ``last_eval_usage`` / ``last_compose_usage``
/ ``last_extract_usage`` as a side channel (D11) — the runner reads
these after every call and threads the numbers into ``turn_recorded``
events and the ``completed`` event's ``eval_usage_totals``. The
:class:`LLMClient` protocol does not declare these attributes; only
this concrete implementation surfaces them.

Retry / backoff lives in the loop (Step 9), not here. This client
raises on the SDK's failure types (``anthropic.APIError``,
``pydantic.ValidationError``) and lets the runner's retry policy
handle them.
"""

from __future__ import annotations

import time
from collections.abc import AsyncIterator
from dataclasses import dataclass
from datetime import UTC, datetime
from typing import TYPE_CHECKING, Any, cast

import anthropic
from pydantic import BaseModel, ConfigDict, Field

from interviewer.llm.prompts import (
    build_compose_user_message,
    build_evaluate_user_message,
    build_extract_user_message,
    build_system_prompt,
)
from interviewer.llm.schemas import anthropic_tool_schema
from interviewer.types.config import Conversation
from interviewer.types.runtime import (
    EvalResult,
    Extract,
    Finding,
    GoalStatus,
    Turn,
    TurnContext,
)

if TYPE_CHECKING:
    from anthropic import AsyncAnthropic


# Default model IDs. Override via constructor kwargs if Anthropic ships
# newer point releases between releases of this package. Verified
# against the latest SDK at implementation time.
EVAL_MODEL = "claude-haiku-4-5"
COMPOSE_MODEL = "claude-sonnet-4-6"
EXTRACT_MODEL = "claude-sonnet-4-6"


@dataclass(frozen=True)
class Usage:
    """Per-call usage telemetry — surfaced to the runner via ``last_*_usage``."""

    tokens_in: int
    tokens_out: int
    cache_read_tokens: int
    cache_write_tokens: int
    llm_latency_ms: int


class _ExtractToolInput(BaseModel):
    """Partial Extract the LLM is asked to fill in via the ``extract`` tool.

    Decoupled from the full ``Extract`` because the runner owns
    ``session_id`` and ``completed_at`` (D8 Step 8), and asking the LLM
    to echo back ``full_transcript`` (which we just sent it) wastes
    tokens. The wrapper reconstructs the full Extract from this plus
    the inputs.
    """

    model_config = ConfigDict(frozen=True, extra="forbid")

    goal_statuses: list[GoalStatus]
    unprompted_findings: list[Finding] = Field(default_factory=list)


class AnthropicLLMClient:
    """Anthropic Claude implementation of :class:`LLMClient`."""

    def __init__(
        self,
        api_key: str | None = None,
        *,
        client: AsyncAnthropic | None = None,
        eval_model: str = EVAL_MODEL,
        compose_model: str = COMPOSE_MODEL,
        extract_model: str = EXTRACT_MODEL,
        max_transcript_turns: int = 12,
    ) -> None:
        if client is not None:
            self._client = client
        elif api_key is not None:
            self._client = anthropic.AsyncAnthropic(api_key=api_key)
        else:
            raise ValueError(
                "AnthropicLLMClient requires either api_key or client"
            )
        self._eval_model = eval_model
        self._compose_model = compose_model
        self._extract_model = extract_model
        self._max_transcript_turns = max_transcript_turns
        self.last_eval_usage: Usage | None = None
        self.last_compose_usage: Usage | None = None
        self.last_extract_usage: Usage | None = None

    async def evaluate_turn(self, ctx: TurnContext) -> EvalResult:
        if ctx.active_goal is None:
            raise ValueError("evaluate_turn requires an active goal")
        system = build_system_prompt(ctx.conversation)
        user = build_evaluate_user_message(
            ctx, max_transcript_turns=self._max_transcript_turns
        )
        start = time.monotonic()
        # SDK params are TypedDicts that don't accept plain dict[str, Any]
        # without contortion; we pass through ``cast(Any, ...)`` rather
        # than wiring our prompt builders to per-SDK-version param types.
        response = await self._client.messages.create(
            model=self._eval_model,
            max_tokens=1024,
            system=cast(Any, _cached_system_blocks(system)),
            messages=cast(Any, [{"role": "user", "content": user}]),
            tools=cast(
                Any,
                [
                    {
                        "name": "evaluate",
                        "description": (
                            "Record your evaluation of the respondent's most "
                            "recent answer for the active goal."
                        ),
                        "input_schema": anthropic_tool_schema(EvalResult),
                    }
                ],
            ),
            tool_choice=cast(Any, {"type": "tool", "name": "evaluate"}),
        )
        elapsed_ms = int((time.monotonic() - start) * 1000)
        block = _first_tool_use(response.content, "evaluate")
        result = EvalResult.model_validate(block.input)
        self.last_eval_usage = _usage_from_response(response, elapsed_ms)
        return result

    def compose_utterance(
        self, ctx: TurnContext, eval_result: EvalResult
    ) -> AsyncIterator[str]:
        return self._compose_stream(ctx, eval_result)

    async def _compose_stream(
        self, ctx: TurnContext, eval_result: EvalResult
    ) -> AsyncIterator[str]:
        system = build_system_prompt(ctx.conversation)
        user = build_compose_user_message(
            ctx, eval_result, max_transcript_turns=self._max_transcript_turns
        )
        start = time.monotonic()
        # The SDK's stream() returns an async context manager; the final
        # message (with usage) is only available after the stream
        # exhausts and is fetched explicitly.
        async with self._client.messages.stream(
            model=self._compose_model,
            max_tokens=512,
            system=cast(Any, _cached_system_blocks(system)),
            messages=cast(Any, [{"role": "user", "content": user}]),
        ) as stream:
            async for text in stream.text_stream:
                yield text
            final = await stream.get_final_message()
        elapsed_ms = int((time.monotonic() - start) * 1000)
        self.last_compose_usage = _usage_from_response(final, elapsed_ms)

    async def derive_extract(
        self, transcript: list[Turn], conv: Conversation
    ) -> Extract:
        system = build_system_prompt(conv)
        user = build_extract_user_message(transcript)
        start = time.monotonic()
        response = await self._client.messages.create(
            model=self._extract_model,
            max_tokens=4096,
            system=cast(Any, _cached_system_blocks(system)),
            messages=cast(Any, [{"role": "user", "content": user}]),
            tools=cast(
                Any,
                [
                    {
                        "name": "extract",
                        "description": (
                            "Record the canonical Extract for this interview: "
                            "per-goal status with evidence turn indices and any "
                            "unprompted findings."
                        ),
                        "input_schema": anthropic_tool_schema(_ExtractToolInput),
                    }
                ],
            ),
            tool_choice=cast(Any, {"type": "tool", "name": "extract"}),
        )
        elapsed_ms = int((time.monotonic() - start) * 1000)
        block = _first_tool_use(response.content, "extract")
        parsed = _ExtractToolInput.model_validate(block.input)
        self.last_extract_usage = _usage_from_response(response, elapsed_ms)
        # session_id and completed_at are runner-owned placeholders here
        # (DECISIONS Step 8); the runner overwrites them after this returns.
        return Extract(
            session_id=conv.id,
            conversation_id=conv.id,
            goal_statuses=list(parsed.goal_statuses),
            unprompted_findings=list(parsed.unprompted_findings),
            full_transcript=list(transcript),
            completed_at=datetime.now(UTC),
        )


def _cached_system_blocks(text: str) -> list[dict[str, Any]]:
    """One ephemeral-cached text block — see D2."""
    return [
        {
            "type": "text",
            "text": text,
            "cache_control": {"type": "ephemeral"},
        }
    ]


def _first_tool_use(content: list[Any], name: str) -> Any:
    """Return the first ``tool_use`` content block named ``name``.

    Any: ``content`` items are the SDK's union of TextBlock / ToolUseBlock
    types; we duck-type ``.type``, ``.name``, ``.input`` rather than
    importing version-specific block classes.
    """
    for block in content:
        if (
            getattr(block, "type", None) == "tool_use"
            and getattr(block, "name", None) == name
        ):
            return block
    raise RuntimeError(f"no tool_use block named {name!r} in Anthropic response")


def _usage_from_response(response: Any, elapsed_ms: int) -> Usage:
    """Map ``response.usage`` to :class:`Usage`.

    Any: the SDK's Usage type is version-specific; we duck-type the
    integer attributes. ``cache_*_input_tokens`` are optional in older
    SDK versions and may be ``None``.
    """
    u = response.usage
    return Usage(
        tokens_in=int(getattr(u, "input_tokens", 0) or 0),
        tokens_out=int(getattr(u, "output_tokens", 0) or 0),
        cache_read_tokens=int(getattr(u, "cache_read_input_tokens", 0) or 0),
        cache_write_tokens=int(getattr(u, "cache_creation_input_tokens", 0) or 0),
        llm_latency_ms=elapsed_ms,
    )
