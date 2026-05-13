from __future__ import annotations

import pytest
from pydantic import ValidationError

from interview_kit import Background, Conversation, Goal, Persona


def _persona() -> Persona:
    return Persona(
        system_prompt="You are an interview_kit.",
        style="neutral",
        voice_id="cartesia-neutral-1",
    )


def _background() -> Background:
    return Background(
        interviewee_role="warehouse ops lead",
        interviewee_expertise="process flow at warehouse X",
    )


def _conversation(**overrides: object) -> Conversation:
    defaults: dict[str, object] = {
        "id": "conv-1",
        "persona": _persona(),
        "purpose": "Understand the warehouse process flow.",
        "background": _background(),
        "goals": [
            Goal(id="flow", intent="map process steps", standard="≥4 steps"),
            Goal(
                id="excep",
                intent="exception paths",
                standard="≥2 exception types",
                depends_on=["flow"],
            ),
        ],
    }
    defaults.update(overrides)
    return Conversation.model_validate(defaults)


# ---------- Persona ---------------------------------------------------------


def test_persona_happy_path() -> None:
    p = _persona()
    assert p.style == "neutral"
    assert p.voice_id == "cartesia-neutral-1"


def test_persona_rejects_unknown_style() -> None:
    with pytest.raises(ValidationError):
        Persona(system_prompt="x", style="aggressive", voice_id="v")  # type: ignore[arg-type]


def test_persona_requires_nonempty_strings() -> None:
    with pytest.raises(ValidationError):
        Persona(system_prompt="", style="neutral", voice_id="v")


def test_persona_is_frozen() -> None:
    p = _persona()
    with pytest.raises(ValidationError):
        p.style = "terse"  # type: ignore[misc]


# ---------- Background ------------------------------------------------------


def test_background_relevant_context_at_cap_ok() -> None:
    Background(
        interviewee_role="r",
        interviewee_expertise="e",
        relevant_context="x" * 1000,
    )


def test_background_relevant_context_overflow_raises() -> None:
    # D6: must raise ValidationError. No silent truncation, no warnings.
    with pytest.raises(ValidationError):
        Background(
            interviewee_role="r",
            interviewee_expertise="e",
            relevant_context="x" * 1001,
        )


def test_background_default_relevant_context_is_empty() -> None:
    b = Background(interviewee_role="r", interviewee_expertise="e")
    assert b.relevant_context == ""


def test_background_is_frozen() -> None:
    b = _background()
    with pytest.raises(ValidationError):
        b.interviewee_role = "other"  # type: ignore[misc]


# ---------- Goal ------------------------------------------------------------


def test_goal_defaults() -> None:
    g = Goal(id="g1", intent="i", standard="s")
    assert g.max_retries == 2
    assert g.depends_on == []
    assert g.redundant_when == ""


def test_goal_rejects_negative_retries() -> None:
    with pytest.raises(ValidationError):
        Goal(id="g", intent="i", standard="s", max_retries=-1)


def test_goal_is_frozen() -> None:
    g = Goal(id="g", intent="i", standard="s")
    with pytest.raises(ValidationError):
        g.intent = "other"  # type: ignore[misc]


# ---------- Conversation ----------------------------------------------------


def test_conversation_happy_path() -> None:
    c = _conversation()
    assert c.max_total_turns == 80
    assert c.max_tangent_followups == 2
    assert c.opening is None and c.closing is None


def test_conversation_rejects_duplicate_goal_ids() -> None:
    with pytest.raises(ValidationError):
        _conversation(
            goals=[
                Goal(id="dup", intent="a", standard="a"),
                Goal(id="dup", intent="b", standard="b"),
            ]
        )


def test_conversation_rejects_unknown_dependency() -> None:
    with pytest.raises(ValidationError):
        _conversation(
            goals=[
                Goal(
                    id="g1",
                    intent="i",
                    standard="s",
                    depends_on=["missing"],
                )
            ]
        )


def test_conversation_rejects_self_dependency() -> None:
    with pytest.raises(ValidationError):
        _conversation(
            goals=[Goal(id="g1", intent="i", standard="s", depends_on=["g1"])]
        )


def test_conversation_rejects_dependency_cycle() -> None:
    with pytest.raises(ValidationError):
        _conversation(
            goals=[
                Goal(id="a", intent="i", standard="s", depends_on=["b"]),
                Goal(id="b", intent="i", standard="s", depends_on=["a"]),
            ]
        )


def test_conversation_rejects_max_total_turns_below_4() -> None:
    with pytest.raises(ValidationError):
        _conversation(max_total_turns=3)


def test_conversation_accepts_max_total_turns_equal_4() -> None:
    c = _conversation(max_total_turns=4)
    assert c.max_total_turns == 4


def test_conversation_rejects_negative_tangent_followups() -> None:
    with pytest.raises(ValidationError):
        _conversation(max_tangent_followups=-1)


def test_conversation_accepts_zero_tangent_followups() -> None:
    c = _conversation(max_tangent_followups=0)
    assert c.max_tangent_followups == 0


def test_conversation_rejects_empty_goals() -> None:
    with pytest.raises(ValidationError):
        _conversation(goals=[])


def test_conversation_is_frozen() -> None:
    c = _conversation()
    with pytest.raises(ValidationError):
        c.purpose = "different"  # type: ignore[misc]


def test_conversation_model_copy_update_returns_new_instance() -> None:
    c = _conversation()
    other = c.model_copy(update={"purpose": "different"})
    assert other.purpose == "different"
    assert c.purpose != "different"


# ---------- Round trip ------------------------------------------------------


def test_conversation_json_round_trip() -> None:
    c = _conversation(
        opening="hi",
        closing="thanks",
        goals=[
            Goal(
                id="g1",
                intent="i",
                standard="s",
                max_retries=3,
                redundant_when="if g2 already answered",
            ),
            Goal(id="g2", intent="i", standard="s", depends_on=["g1"]),
        ],
    )
    data = c.model_dump_json()
    restored = Conversation.model_validate_json(data)
    assert restored == c
