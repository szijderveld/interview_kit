"""Run interview_kit end-to-end against a local LiveKit dev server.

Required env vars: ``ANTHROPIC_API_KEY``, ``DEEPGRAM_API_KEY``,
``CARTESIA_API_KEY``, ``LIVEKIT_URL`` (default ``ws://localhost:7880``),
``LIVEKIT_API_KEY`` (default ``devkey``), ``LIVEKIT_API_SECRET``
(default ``secret``).

Setup:
    1. brew install livekit-server  # or download from livekit.io
    2. livekit-server --dev          # in a separate terminal
    3. ANTHROPIC_API_KEY=... DEEPGRAM_API_KEY=... CARTESIA_API_KEY=... \\
       uv run python examples/local_voice.py --provision
       # prints a join URL — open it in a browser
    4. In another terminal, start the worker:
       uv run python examples/local_voice.py
"""

from __future__ import annotations

import asyncio
import os
import sys
from pathlib import Path
from typing import TYPE_CHECKING

from interview_kit import (
    Background,
    Conversation,
    Engine,
    Goal,
    LiveKitConfig,
    Persona,
)
from interview_kit.llm.anthropic import AnthropicLLMClient
from interview_kit.sinks.memory import InMemoryEventSink
from interview_kit.stores.sqlite import SQLiteConversationStore

_STORE_PATH = os.environ.get("INTERVIEW_KIT_DB", "/tmp/interview_kit.db")

if TYPE_CHECKING:
    from livekit.agents import JobContext


async def build_engine() -> Engine:
    store = SQLiteConversationStore(_STORE_PATH)
    await store.connect()
    return Engine(
        store=store,
        events=InMemoryEventSink(),
        llm=AnthropicLLMClient(api_key=os.environ["ANTHROPIC_API_KEY"]),
        livekit=LiveKitConfig(
            url=os.environ.get("LIVEKIT_URL", "ws://localhost:7880"),
            api_key=os.environ.get("LIVEKIT_API_KEY", "devkey"),
            api_secret=os.environ.get("LIVEKIT_API_SECRET", "secret"),
            agent_name="interviewer",
        ),
    )


async def _create_conversation(engine: Engine) -> Conversation:
    return await engine.create_conversation(
        persona=Persona(
            system_prompt=(
                "You are a discovery interviewer learning how the respondent "
                "spends a typical workweek. Listen, then probe one thing at a time."
            ),
            style="neutral",
            voice_id="694f9389-aac1-45b6-b726-9d9369183238",  # Cartesia "Sarah"
        ),
        purpose="Understand the end-to-end flow at this team.",
        background=Background(
            interviewee_role="staff engineer",
            interviewee_expertise="end-to-end pipeline ownership",
        ),
        goals=[
            Goal(id="g1", intent="Map the day's main rituals.",
                 standard="At least two rituals with timing."),
            Goal(id="g2", intent="Find common exception paths.",
                 standard="At least one exception flow named."),
            Goal(id="g3", intent="What does a good day look like.",
                 standard="At least one metric named."),
        ],
        opening="Thanks for joining — mind walking me through how your week tends to go?",
        closing="That's everything I needed. Appreciate your time.",
    )


async def main_provision() -> None:
    engine = await build_engine()
    conv = await _create_conversation(engine)
    session, creds = await engine.provision_session(conv.id)
    join_html = Path(__file__).parent / "join_page.html"
    join_url = f"file://{join_html.resolve()}?url={creds.room_url}&token={creds.token}"
    print()
    print(f"→ session id: {session.id}")
    print("→ open this URL in a browser to join as respondent:")
    print(f"  {join_url}")
    print()
    print("Then in another terminal, run `uv run python examples/local_voice.py`")
    print("to start the worker. The agent will speak first once you connect.")


async def worker_entrypoint(ctx: JobContext) -> None:
    engine = await build_engine()
    await engine.entrypoint(ctx)


def _run_worker() -> None:
    # ``livekit.agents.cli.run_app`` is the standard worker bootstrap;
    # it parses CLI args (``dev``, ``start``, …) and dispatches jobs.
    from livekit import agents

    agents.cli.run_app(
        agents.WorkerOptions(
            entrypoint_fnc=worker_entrypoint,
        )
    )


if __name__ == "__main__":
    if "--provision" in sys.argv:
        asyncio.run(main_provision())
    else:
        _run_worker()
