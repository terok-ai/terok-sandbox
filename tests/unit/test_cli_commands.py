# SPDX-FileCopyrightText: 2026 Jiri Vyskocil
# SPDX-License-Identifier: Apache-2.0

"""Tests for the terok-sandbox CLI."""

from __future__ import annotations

from io import StringIO
from unittest.mock import patch

from terok_sandbox.cli import main


def _run_cli(*args: str) -> tuple[str, str, int]:
    """Run CLI in-process, capturing stdout/stderr and exit code."""
    stdout, stderr = StringIO(), StringIO()
    code = 0
    with (
        patch("sys.argv", ["terok-sandbox", *args]),
        patch("sys.stdout", stdout),
        patch("sys.stderr", stderr),
    ):
        try:
            main()
        except SystemExit as e:
            code = e.code if isinstance(e.code, int) else 1
    return stdout.getvalue(), stderr.getvalue(), code


class TestNoCommand:
    """Verify behaviour with no subcommand."""

    def test_shows_help(self) -> None:
        out, err, rc = _run_cli()
        assert rc == 1
        combined = (out + err).lower()
        assert "usage:" in combined

    def test_version(self) -> None:
        out, _, rc = _run_cli("--version")
        assert rc == 0
        assert "terok-sandbox" in out


class TestShieldSubcommand:
    """Verify shield subcommand structure."""

    def test_shield_no_subcommand_shows_help(self) -> None:
        out, _, _ = _run_cli("shield")
        combined = out.lower()
        assert "setup" in combined or "status" in combined

    def test_shield_status_runs(self) -> None:
        from terok_shield import EnvironmentCheck

        mock_env = EnvironmentCheck(ok=True, hooks="per-container", health="ok")
        mock_cfg = {"mode": "hook", "profiles": ["dev-standard"], "audit": True}
        with (
            patch("terok_sandbox.shield.check_environment", return_value=mock_env),
            patch("terok_sandbox.shield.status", return_value=mock_cfg),
        ):
            out, _, rc = _run_cli("shield", "status")
        assert rc == 0
        assert "hook" in out
        assert "dev-standard" in out


class TestGateSubcommand:
    """Verify gate subcommand structure."""

    def test_gate_no_subcommand_shows_help(self) -> None:
        out, _, _ = _run_cli("gate")
        combined = out.lower()
        assert "start" in combined or "stop" in combined

    def test_gate_status_runs(self) -> None:
        from terok_sandbox.gate_server import GateServerStatus

        mock_status = GateServerStatus(mode="none", running=False, port=9418)
        with (
            patch("terok_sandbox.gate_server.get_server_status", return_value=mock_status),
            patch("terok_sandbox.gate_server.get_gate_base_path", return_value="/tmp/gate"),
            patch("terok_sandbox.gate_server.check_units_outdated", return_value=None),
        ):
            out, _, rc = _run_cli("gate", "status")
        assert rc == 0
        assert "none" in out
        assert "9418" in out
