# interviewer

A Python package that conducts adaptive voice interviews on behalf of a
human operator. The operator defines a Conversation — a persona for the
interviewer, a purpose, and a list of Goals (things to find out, each
with a "what good looks like" standard). The engine runs the conversation
as a voice agent (LiveKit + Deepgram STT + Anthropic LLM + Cartesia TTS),
adapts mid-call (clarifying, drilling, skipping redundant goals), and
produces a structured Extract that maps every claim back to who said it
and when.

This package is the engine. It owns the conversation logic and the agent
loop. It does not own the database, the web server, the link domain, or
any UI — those are the consumer's responsibility. See [SCOPE.md](SCOPE.md)
for the full design contract.

## Install

Requires Python 3.11+.

```sh
pip install interview-kit
```

The voice extra pulls in LiveKit and the audio plugins:

```sh
pip install "interview-kit[voice]"
```

The distribution name on PyPI is `interview-kit`, but the import name is
`interviewer`:

```python
import interviewer
```

### Develop from source

Requires [uv](https://docs.astral.sh/uv/).

```sh
uv sync --all-extras --dev
```

## Quickstart — simulated interview (no API keys)

Runs the full agent loop against a synthetic respondent and a deterministic
fake LLM. Prints the transcript and the structured Extract.

```sh
uv run python examples/simulated.py
```

Pass `--use-anthropic` (and set `ANTHROPIC_API_KEY`) to swap in the real
`AnthropicLLMClient`:

```sh
ANTHROPIC_API_KEY=sk-... uv run python examples/simulated.py --use-anthropic
```

## Quickstart — local voice interview

End-to-end voice agent against a local LiveKit dev server. Requires
`ANTHROPIC_API_KEY`, `DEEPGRAM_API_KEY`, `CARTESIA_API_KEY` and a running
`livekit-server --dev` instance.

```sh
# terminal 1
livekit-server --dev

# terminal 2 — mint join credentials
ANTHROPIC_API_KEY=... DEEPGRAM_API_KEY=... CARTESIA_API_KEY=... \
  uv run python examples/local_voice.py --provision
# prints a join URL; open it in a browser

# terminal 3 — run the LiveKit agent worker
ANTHROPIC_API_KEY=... DEEPGRAM_API_KEY=... CARTESIA_API_KEY=... \
  uv run python examples/local_voice.py
```

See [examples/local_voice.py](examples/local_voice.py) for the wiring.

## Consumer integration

See [docs/integration.md](docs/integration.md) for how to plug the engine
into a web app + LiveKit AgentServer deployment, including `ConversationStore`
and `EventSink` implementation guidance and the operational gaps the
consumer must close.

## Development

```sh
uv run pytest
uv run mypy src/
uv run ruff check .
```
