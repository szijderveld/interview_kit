"""Tests for the ``interview_kit`` console script (Step 20)."""

from __future__ import annotations

from pathlib import Path

import pytest

from interview_kit import cli

_FIXTURE_YAML = Path(__file__).parent / "fixtures" / "cli_simulate.yaml"


def test_demo_subcommand_prints_transcript_and_extract(
    capsys: pytest.CaptureFixture[str],
) -> None:
    cli.main(["demo"])
    captured = capsys.readouterr()
    assert "TRANSCRIPT" in captured.out
    assert "EXTRACT" in captured.out


def test_simulate_subcommand_runs_yaml_fixture(
    capsys: pytest.CaptureFixture[str],
) -> None:
    cli.main(["simulate", str(_FIXTURE_YAML)])
    captured = capsys.readouterr()
    assert "TRANSCRIPT" in captured.out
    assert "EXTRACT" in captured.out
    # The fixture's two goals should both appear in the rendered extract table.
    assert "routine" in captured.out
    assert "exceptions" in captured.out


def test_version_flag_prints_package_version(
    capsys: pytest.CaptureFixture[str],
) -> None:
    with pytest.raises(SystemExit) as exc_info:
        cli.main(["--version"])
    assert exc_info.value.code == 0
    captured = capsys.readouterr()
    assert "interview_kit" in captured.out


def test_missing_command_exits_nonzero(
    capsys: pytest.CaptureFixture[str],
) -> None:
    with pytest.raises(SystemExit) as exc_info:
        cli.main([])
    assert exc_info.value.code != 0
