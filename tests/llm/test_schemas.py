"""Tests for ``anthropic_tool_schema`` (Step 12 / D15).

Verifies the helper produces JSON Schemas Anthropic accepts: Pydantic
noise keys stripped (``title``, ``$defs``, ``definitions``); ``$ref``
cross-references inlined into the property dictionaries; required
fields preserved.
"""

from __future__ import annotations

import json
from typing import Any

from pydantic import BaseModel, Field

from interviewer.llm.schemas import anthropic_tool_schema
from interviewer.types.runtime import EvalResult, Extract


class _Nested(BaseModel):
    value: str


class _WithNested(BaseModel):
    name: str = Field(description="display name")
    nested: _Nested
    nested_list: list[_Nested] = Field(default_factory=list)


def _all_keys(obj: Any) -> set[str]:
    """Walk a dict/list tree and return every dict key encountered."""
    seen: set[str] = set()

    def visit(o: Any) -> None:
        if isinstance(o, dict):
            seen.update(o.keys())
            for v in o.values():
                visit(v)
        elif isinstance(o, list):
            for v in o:
                visit(v)

    visit(obj)
    return seen


def test_top_level_title_dollar_defs_definitions_stripped() -> None:
    schema = anthropic_tool_schema(_WithNested)
    assert "$defs" not in schema
    assert "definitions" not in schema
    assert "title" not in schema


def test_property_title_keys_stripped_recursively() -> None:
    schema = anthropic_tool_schema(_WithNested)
    keys = _all_keys(schema)
    assert "title" not in keys
    assert "$defs" not in keys
    assert "$ref" not in keys


def test_nested_model_ref_is_inlined_into_property() -> None:
    schema = anthropic_tool_schema(_WithNested)
    nested = schema["properties"]["nested"]
    assert nested.get("type") == "object"
    assert "value" in nested["properties"]
    assert nested["properties"]["value"]["type"] == "string"


def test_inlining_handles_arrays_of_nested_models() -> None:
    schema = anthropic_tool_schema(_WithNested)
    items_schema = schema["properties"]["nested_list"]["items"]
    assert items_schema.get("type") == "object"
    assert "value" in items_schema["properties"]


def test_required_fields_preserved() -> None:
    schema = anthropic_tool_schema(_WithNested)
    # ``name`` and ``nested`` are required (no defaults); ``nested_list`` is not.
    required = set(schema.get("required", []))
    assert "name" in required
    assert "nested" in required
    assert "nested_list" not in required


def test_eval_result_schema_is_serializable_and_clean() -> None:
    """EvalResult has Literal types but no nested BaseModels — sanity check."""
    schema = anthropic_tool_schema(EvalResult)
    # Round-trips through JSON.
    serialized = json.dumps(schema)
    redumped = json.loads(serialized)
    assert redumped == schema
    # Top-level keys: properties + required + type.
    assert schema["type"] == "object"
    assert {"active_goal_status", "next_action"}.issubset(set(schema["required"]))


def test_extract_schema_inlines_goal_statuses_and_findings() -> None:
    """Even though Extract is more complex, $refs are gone after inlining."""
    schema = anthropic_tool_schema(Extract)
    assert "$ref" not in _all_keys(schema)
    # goal_statuses is an array of inlined GoalStatus objects.
    gs_items = schema["properties"]["goal_statuses"]["items"]
    assert gs_items["type"] == "object"
    assert "goal_id" in gs_items["properties"]
    assert "status" in gs_items["properties"]
