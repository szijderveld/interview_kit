# Contributing

Development workflow for the `interview_kit` repository. Requires
[uv](https://docs.astral.sh/uv/).

## Source checkout

```sh
git clone https://github.com/szijderveld/interview_kit.git
cd interview_kit
uv sync --all-extras --dev
```

## Checks

```sh
uv run pytest
uv run mypy src/
uv run ruff check .
```

All three must pass before a commit.

## Repo-root simulator entry

The repo ships an `examples/simulated.py` shim equivalent to
`interview_kit demo`:

```sh
uv run python examples/simulated.py
```

Pass `--use-anthropic` (and set `ANTHROPIC_API_KEY`) to swap in the real
`AnthropicLLMClient`:

```sh
ANTHROPIC_API_KEY=sk-... uv run python examples/simulated.py --use-anthropic
```

## Local voice interview

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
