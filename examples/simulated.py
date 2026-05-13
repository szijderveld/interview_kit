"""Repo-root entry to the simulator demo.

Thin shim that defers to ``interviewer.examples.simulated``. Kept so the
README's ``uv run python examples/simulated.py`` invocation continues to
work for contributors working out of the checkout.

For pip users, the same demo is reachable as ``interviewer demo``.
"""

from __future__ import annotations

from interviewer.examples.simulated import cli

if __name__ == "__main__":
    cli()
