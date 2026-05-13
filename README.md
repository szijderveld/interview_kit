# interview-kit

Adaptive voice-interview engine. The operator defines a Conversation — a
persona for the interviewer, a purpose, and a list of Goals (each with a
"what good looks like" standard). The engine runs the conversation as a
voice agent (LiveKit + Deepgram STT + Anthropic LLM + Cartesia TTS),
adapts mid-call (clarifying, drilling, skipping redundant goals), and
produces a structured Extract mapping every claim back to who said it
and when. This package is the engine — storage, web layer, link domain,
and UI are the consumer's responsibility. See
[SCOPE.md](SCOPE.md) for the design contract.

## Install

Requires Python 3.11+.

```sh
pip install interview-kit
```

The voice extra pulls in LiveKit and the audio plugins:

```sh
pip install "interview-kit[voice]"
```

The distribution name on PyPI is `interview-kit`; the import name is
`interviewer`.

## Smoke test (no API key)

```sh
interviewer demo
```

Runs the full agent loop against a synthetic respondent and a
deterministic fake LLM. Prints the transcript and the structured Extract.

## Quickstart

Save the following as `interview.yaml`:

```yaml
persona:
  system_prompt: You are running a discovery interview about morning routines.
  style: neutral
  voice_id: demo-voice
purpose: Understand the interviewee's morning routine.
background:
  interviewee_role: staff engineer
  interviewee_expertise: end-to-end pipeline ownership
goals:
  - id: routine
    intent: Map the morning routine
    standard: At least two rituals named with timing.
  - id: exceptions
    intent: Find common exception paths
    standard: At least one exception flow named.
```

Then, with `ANTHROPIC_API_KEY` set:

```python
import asyncio
from interviewer import Conversation, Engine
from interviewer.testing.simulators import RamblyKnowledgeableSimulator

async def main() -> None:
    engine = Engine.with_defaults()
    template = Conversation.from_yaml("interview.yaml")
    conv = await engine.create_conversation(**template.model_dump(exclude={"id"}))
    extract = await engine.simulate_session(conv.id, RamblyKnowledgeableSimulator())
    print(extract.model_dump_json(indent=2))

asyncio.run(main())
```

## Production / voice integration

See [docs/integration.md](docs/integration.md) for the FastAPI + LiveKit
AgentServer wiring, `ConversationStore` and `EventSink` implementation
guidance, and the operational gaps the consumer must close.

## Development

See [CONTRIBUTING.md](CONTRIBUTING.md) for the source checkout, test, and
local-voice workflows.
