"""LiveKitConfig — credentials and routing for a LiveKit deployment.

Used by ``Engine.provision_session`` to mint room creds and by
``Engine.entrypoint`` to identify which AgentServer subject the engine is
serving. Consumers in simulator-only deployments leave it None.
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True, slots=True)
class LiveKitConfig:
    """Frozen config bundle for a LiveKit deployment."""

    url: str
    api_key: str
    api_secret: str
    agent_name: str
