"""RespondentSimulator reference impls — text-rule respondents (no LLM)."""

from __future__ import annotations

from collections import deque

from interview_kit.types.runtime import Turn


class ScriptedSimulator:
    """Respondent that replies from a pre-scripted queue."""

    def __init__(self, responses: list[str], *, name: str = "scripted") -> None:
        self._responses: deque[str] = deque(responses)
        self._name = name

    async def respond(self, agent_utterance: str, history: list[Turn]) -> str:
        if not self._responses:
            raise RuntimeError("ScriptedSimulator: response queue exhausted")
        return self._responses.popleft()

    def persona_name(self) -> str:
        return self._name


class _CyclingSimulator:
    """Internal helper: cycle through a fixed response list."""

    _RESPONSES: tuple[str, ...] = ()
    _NAME: str = ""

    def __init__(self) -> None:
        self._i = 0

    async def respond(self, agent_utterance: str, history: list[Turn]) -> str:
        resp = self._RESPONSES[self._i % len(self._RESPONSES)]
        self._i += 1
        return resp

    def persona_name(self) -> str:
        return self._NAME


class TerseEvasiveSimulator(_CyclingSimulator):
    """Short, dodgy answers — useful for surfacing ``gave_up`` paths in tests."""

    _RESPONSES = (
        "Hmm, not really.",
        "Depends.",
        "Couldn't say.",
        "Maybe.",
        "Not sure.",
    )
    _NAME = "terse_evasive"


class RamblyKnowledgeableSimulator(_CyclingSimulator):
    """Long, detail-rich answers, occasionally tangential."""

    _RESPONSES = (
        "Oh, lots of moving pieces. Picking pulls from totes, then a manual QC pass.",
        "Yeah and the exception path used to involve a lot of paper but we cleaned it up.",
        "Going back to your earlier question — we did a 2x speedup on cycle time.",
        "Depends on the shift, but typically two operators handle the morning bay.",
    )
    _NAME = "rambly_knowledgeable"


class ConfusedSimulator(_CyclingSimulator):
    """Asks for clarification, mixes up topics."""

    _RESPONSES = (
        "Sorry, can you rephrase that?",
        "I think you mean the other thing?",
        "Hmm, that's like what I said before.",
        "Not sure I follow.",
    )
    _NAME = "confused"
