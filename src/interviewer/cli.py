"""``interviewer`` console script — thin shim over the public API.

Subcommands:

- ``interviewer demo`` — run the shipped simulator example. No API key
  required by default; ``--use-anthropic`` swaps in the real LLM.
- ``interviewer simulate <conversation.yaml>`` — load a Conversation
  from YAML and run :func:`Engine.simulate_session`. ``--responses``
  feeds a ``ScriptedSimulator``; ``--use-anthropic`` swaps in
  ``AnthropicLLMClient`` + ``RamblyKnowledgeableSimulator``.
- ``interviewer --version`` — print the installed package version.

The CLI is intentionally a thin shim. Anything resembling business
logic belongs in the library.
"""

from __future__ import annotations

import argparse
import asyncio
from importlib import metadata
from pathlib import Path

from interviewer import Conversation, Engine, EvalResult, Extract
from interviewer.examples import simulated as simulated_demo
from interviewer.protocols import LLMClient, RespondentSimulator
from interviewer.sinks.memory import InMemoryEventSink
from interviewer.stores.memory import InMemoryConversationStore
from interviewer.testing.fake_llm import FakeLLMClient
from interviewer.testing.simulators import (
    RamblyKnowledgeableSimulator,
    ScriptedSimulator,
)

_DEFAULT_FAKE_RESPONSES: tuple[str, ...] = (
    "Sure, happy to walk through it.",
    "We hit it daily, takes about an hour.",
    "Exceptions go to the floor lead — we triage on the spot.",
    "A good day is no escalations and throughput on target.",
    "Anything else you want to dig into?",
)


def _package_version() -> str:
    try:
        return metadata.version("interview-kit")
    except metadata.PackageNotFoundError:
        return "unknown"


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="interviewer",
        description="Adaptive voice-interview engine CLI.",
    )
    parser.add_argument(
        "--version", action="version", version=f"interview-kit {_package_version()}"
    )
    sub = parser.add_subparsers(dest="command", required=True, metavar="COMMAND")

    demo = sub.add_parser("demo", help="Run the shipped simulator demo.")
    demo.add_argument(
        "--use-anthropic",
        action="store_true",
        help="Use AnthropicLLMClient (requires ANTHROPIC_API_KEY).",
    )

    sim = sub.add_parser(
        "simulate", help="Run a YAML-defined Conversation against a simulator."
    )
    sim.add_argument(
        "yaml_path", type=Path, help="Path to a Conversation YAML file."
    )
    sim.add_argument(
        "--responses",
        type=Path,
        default=None,
        help=(
            "Text file with one respondent utterance per line. "
            "Ignored when --use-anthropic is set."
        ),
    )
    sim.add_argument(
        "--use-anthropic",
        action="store_true",
        help=(
            "Use AnthropicLLMClient + RamblyKnowledgeableSimulator "
            "(requires ANTHROPIC_API_KEY)."
        ),
    )
    return parser


def _load_responses(path: Path | None) -> list[str]:
    if path is None:
        return list(_DEFAULT_FAKE_RESPONSES)
    lines = [
        line.strip()
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    if not lines:
        raise SystemExit(f"--responses file {path!s} contained no usable lines.")
    return lines


def _autoeval_results(num_goals: int) -> list[EvalResult]:
    return [
        EvalResult(
            active_goal_status="meets",
            next_action="advance",
            rationale=f"goal {i + 1} addressed",
        )
        for i in range(num_goals)
    ]


def _autoutter_utterances(conv: Conversation) -> list[str]:
    return [f"Tell me about {goal.intent.lower()}" for goal in conv.goals]


def _build_simulate_llm_and_simulator(
    conv: Conversation, responses_path: Path | None, use_anthropic: bool
) -> tuple[LLMClient, RespondentSimulator]:
    if use_anthropic:
        import os

        api_key = os.environ.get("ANTHROPIC_API_KEY")
        if not api_key:
            raise SystemExit(
                "ANTHROPIC_API_KEY is required when --use-anthropic is set."
            )
        from interviewer.llm.anthropic import AnthropicLLMClient

        return AnthropicLLMClient(api_key=api_key), RamblyKnowledgeableSimulator()

    llm = FakeLLMClient(
        eval_results=_autoeval_results(len(conv.goals)),
        utterances=_autoutter_utterances(conv),
    )
    simulator = ScriptedSimulator(_load_responses(responses_path))
    return llm, simulator


async def _run_simulate(
    yaml_path: Path, responses_path: Path | None, use_anthropic: bool
) -> Extract:
    template = Conversation.from_yaml(yaml_path)
    llm, simulator = _build_simulate_llm_and_simulator(
        template, responses_path, use_anthropic
    )
    engine = Engine(
        store=InMemoryConversationStore(),
        events=InMemoryEventSink(),
        llm=llm,
    )
    conv = await engine.create_conversation(
        **template.model_dump(exclude={"id"})
    )
    return await engine.simulate_session(conv.id, simulator)


def main(argv: list[str] | None = None) -> None:
    parser = _build_parser()
    args = parser.parse_args(argv)

    if args.command == "demo":
        simulated_demo.cli(["--use-anthropic"] if args.use_anthropic else [])
        return

    if args.command == "simulate":
        extract = asyncio.run(
            _run_simulate(args.yaml_path, args.responses, args.use_anthropic)
        )
        simulated_demo.print_extract(extract)
        return

    parser.error(f"unknown command {args.command!r}")


if __name__ == "__main__":
    main()
