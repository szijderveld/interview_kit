"""Anthropic tool ``input_schema`` derivation from Pydantic models (D15).

Pydantic v2's :meth:`BaseModel.model_json_schema` emits keys Anthropic's
tool schema validator rejects (``title``, ``$defs``, ``definitions``),
and references shared types via ``$ref: "#/$defs/Name"`` instead of
inlining them. This module inlines the refs and strips the noise so the
resulting dict can be fed straight to ``tools=[{"input_schema": ...}]``.

Used by :class:`AnthropicLLMClient` for both the ``evaluate`` and
``extract`` tools. Tested against Pydantic's own emitted schemas in
``tests/llm/test_schemas.py``.
"""

from __future__ import annotations

from typing import Any

from pydantic import BaseModel

# Any: JSON Schema is intrinsically heterogeneous (mix of strings,
# lists, nested dicts of unknown depth). Typing it strictly would
# require a recursive TypedDict union we'd then have to fight; the
# helper is small enough that runtime correctness is verified by tests.


def anthropic_tool_schema(model: type[BaseModel]) -> dict[str, Any]:
    """Return an Anthropic-compatible JSON Schema for ``model``."""
    schema: dict[str, Any] = model.model_json_schema()
    schema = _inline_defs(schema)
    return _strip_pydantic_noise(schema)


def _inline_defs(schema: dict[str, Any]) -> dict[str, Any]:
    """Replace ``$ref`` occurrences with their inlined ``$defs`` bodies."""
    defs: dict[str, Any] = schema.get("$defs") or schema.get("definitions") or {}
    if not defs:
        return schema

    def resolve(obj: Any) -> Any:
        if isinstance(obj, dict):
            if "$ref" in obj and isinstance(obj["$ref"], str):
                name = obj["$ref"].rsplit("/", 1)[-1]
                target = defs.get(name)
                if target is not None:
                    # Recursively resolve in case the target also refs.
                    return resolve(target)
            return {k: resolve(v) for k, v in obj.items()}
        if isinstance(obj, list):
            return [resolve(item) for item in obj]
        return obj

    resolved = resolve(schema)
    assert isinstance(resolved, dict)
    return resolved


def _strip_pydantic_noise(schema: dict[str, Any]) -> dict[str, Any]:
    """Remove keys Anthropic rejects from a Pydantic-emitted schema."""

    def strip(obj: Any) -> Any:
        if isinstance(obj, dict):
            cleaned = {k: strip(v) for k, v in obj.items() if k not in _NOISE_KEYS}
            return cleaned
        if isinstance(obj, list):
            return [strip(item) for item in obj]
        return obj

    result = strip(schema)
    assert isinstance(result, dict)
    return result


_NOISE_KEYS: frozenset[str] = frozenset({"title", "$defs", "definitions"})
