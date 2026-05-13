"""Tests for Conversation.from_dict / Conversation.from_yaml (Step 19)."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pytest
import yaml

from interview_kit import Conversation


def _conversation_payload() -> dict[str, Any]:
    return {
        "id": "conv-yaml-1",
        "persona": {
            "system_prompt": "You are a process engineer doing discovery.",
            "style": "neutral",
            "voice_id": "cartesia-neutral-1",
        },
        "purpose": "Understand the warehouse process flow.",
        "background": {
            "interviewee_role": "warehouse ops lead",
            "interviewee_expertise": "process flow at warehouse X",
            "relevant_context": "Multi-shift facility.",
        },
        "goals": [
            {
                "id": "flow",
                "intent": "Map process steps.",
                "standard": "≥4 steps named.",
            },
            {
                "id": "excep",
                "intent": "Find exception paths.",
                "standard": "≥2 exception types.",
                "depends_on": ["flow"],
            },
        ],
        "opening": "Thanks for joining.",
        "closing": "Appreciate your time.",
        "max_tangent_followups": 2,
        "max_total_turns": 40,
    }


def test_from_dict_builds_conversation() -> None:
    conv = Conversation.from_dict(_conversation_payload())
    assert conv.id == "conv-yaml-1"
    assert conv.persona.style == "neutral"
    assert [g.id for g in conv.goals] == ["flow", "excep"]
    assert conv.goals[1].depends_on == ["flow"]


def test_from_yaml_round_trips(tmp_path: Path) -> None:
    payload = _conversation_payload()
    yaml_path = tmp_path / "conversation.yaml"
    yaml_path.write_text(yaml.safe_dump(payload), encoding="utf-8")

    conv = Conversation.from_yaml(yaml_path)
    dumped = conv.model_dump()

    assert dumped["id"] == payload["id"]
    assert dumped["persona"] == payload["persona"]
    assert dumped["background"] == payload["background"]
    assert dumped["goals"][1]["depends_on"] == ["flow"]
    # Round-trip the dumped form back through from_dict.
    again = Conversation.from_dict(dumped)
    assert again == conv


def test_from_yaml_accepts_str_path(tmp_path: Path) -> None:
    yaml_path = tmp_path / "conv.yaml"
    yaml_path.write_text(yaml.safe_dump(_conversation_payload()), encoding="utf-8")
    conv = Conversation.from_yaml(str(yaml_path))
    assert conv.id == "conv-yaml-1"


def test_from_yaml_rejects_non_mapping(tmp_path: Path) -> None:
    bad = tmp_path / "list.yaml"
    bad.write_text("- a\n- b\n", encoding="utf-8")
    with pytest.raises(ValueError, match="mapping at the top level"):
        Conversation.from_yaml(bad)


def test_from_dict_propagates_validation_errors() -> None:
    payload = _conversation_payload()
    payload["goals"] = []  # Conversation requires at least one goal
    with pytest.raises(ValueError, match="at least one goal"):
        Conversation.from_dict(payload)
