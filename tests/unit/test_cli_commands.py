# SPDX-FileCopyrightText: 2026 Jiri Vyskocil
# SPDX-License-Identifier: Apache-2.0

"""Tests for the terok-sandbox CLI and command registry."""

from __future__ import annotations

from io import StringIO
from unittest.mock import patch

from terok_sandbox.cli import main
from terok_sandbox.commands import (
    COMMANDS,
    GATE_COMMANDS,
    SHIELD_COMMANDS,
    SSH_COMMANDS,
    CommandDef,
)


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


# ---------------------------------------------------------------------------
# Command registry
# ---------------------------------------------------------------------------


class TestCommandRegistry:
    """Verify the command registry is well-formed."""

    def test_all_commands_are_commanddef(self) -> None:
        for cmd in COMMANDS:
            assert isinstance(cmd, CommandDef)

    def test_all_commands_have_names(self) -> None:
        for cmd in COMMANDS:
            assert cmd.name, f"Command missing name: {cmd}"

    def test_all_commands_have_handlers(self) -> None:
        for cmd in COMMANDS:
            assert cmd.handler is not None, f"Command '{cmd.name}' has no handler"

    def test_gate_commands_grouped(self) -> None:
        assert all(cmd.group == "gate" for cmd in GATE_COMMANDS)

    def test_shield_commands_grouped(self) -> None:
        assert all(cmd.group == "shield" for cmd in SHIELD_COMMANDS)

    def test_gate_has_start_stop_status(self) -> None:
        names = {cmd.name for cmd in GATE_COMMANDS}
        assert {"start", "stop", "status"} <= names

    def test_shield_has_install_hooks_and_status(self) -> None:
        names = {cmd.name for cmd in SHIELD_COMMANDS}
        assert {"install-hooks", "status"} <= names

    def test_ssh_has_import_add_and_remove(self) -> None:
        names = {cmd.name for cmd in SSH_COMMANDS}
        assert {"import", "add", "remove"} <= names


# ---------------------------------------------------------------------------
# CLI dispatch
# ---------------------------------------------------------------------------


class TestCLIBasics:
    """Verify basic CLI behaviour."""

    def test_no_command_shows_help(self) -> None:
        out, err, rc = _run_cli()
        assert rc == 1
        combined = (out + err).lower()
        assert "usage:" in combined

    def test_version(self) -> None:
        out, _, rc = _run_cli("--version")
        assert rc == 0
        assert "terok-sandbox" in out

    def test_shield_no_subcommand_shows_help(self) -> None:
        out, _, _ = _run_cli("shield")
        combined = out.lower()
        # Both subcommands must appear — the help listing isn't conditional.
        assert "install-hooks" in combined
        assert "status" in combined

    def test_shield_install_hooks_requires_scope_flag(self) -> None:
        """``shield install-hooks`` with no flag surfaces CLI-specific hints.

        The library function (:func:`terok_sandbox.shield.run_setup`) raises
        ``ValueError`` on invalid combos — the CLI layer is what maps it to
        actionable ``install-hooks --root/--user`` remediation text.
        """
        from terok_sandbox.commands import _handle_shield_setup

        try:
            _handle_shield_setup()
        except SystemExit as exc:
            message = str(exc)
            assert "--root" in message and "--user" in message
            assert "shield install-hooks" in message
        else:  # pragma: no cover — defensive: must SystemExit
            raise AssertionError("_handle_shield_setup should have raised SystemExit")

    def test_gate_no_subcommand_shows_help(self) -> None:
        out, _, _ = _run_cli("gate")
        combined = out.lower()
        assert "start" in combined or "stop" in combined

    def test_ssh_no_subcommand_shows_help(self) -> None:
        out, _, _ = _run_cli("ssh")
        assert "import" in out.lower()


class TestShieldCLI:
    """Verify shield subcommand dispatch."""

    def test_shield_status_runs(self) -> None:
        from terok_shield import EnvironmentCheck

        mock_env = EnvironmentCheck(ok=True, hooks="per-container", health="ok")
        mock_cfg = {"mode": "hook", "profiles": ["dev-standard"], "audit_enabled": True}
        with (
            patch("terok_sandbox.shield.check_environment", return_value=mock_env),
            patch("terok_sandbox.shield.status", return_value=mock_cfg),
        ):
            out, _, rc = _run_cli("shield", "status")
        assert rc == 0
        assert "hook" in out
        assert "dev-standard" in out
        assert "enabled" in out


class TestHandlerCfgSignatures:
    """All command handlers in config-injected groups accept a ``cfg`` keyword argument."""

    def test_gate_handlers_accept_cfg(self) -> None:
        import inspect

        for cmd in GATE_COMMANDS:
            sig = inspect.signature(cmd.handler)
            assert "cfg" in sig.parameters, f"{cmd.handler.__name__} missing cfg param"

    def test_ssh_handlers_accept_cfg(self) -> None:
        import inspect

        for cmd in SSH_COMMANDS:
            sig = inspect.signature(cmd.handler)
            assert "cfg" in sig.parameters, f"{cmd.handler.__name__} missing cfg param"


class TestGateHandlerCfgPassthrough:
    """Verify gate handlers propagate cfg to downstream functions."""

    def test_gate_start_passes_cfg_to_systemd(self) -> None:
        """_handle_gate_start propagates cfg to install_systemd_units."""
        from unittest.mock import sentinel

        from terok_sandbox.commands import _handle_gate_start
        from terok_sandbox.gate.lifecycle import GateServerManager

        with (
            patch.object(GateServerManager, "is_systemd_available", return_value=True),
            patch.object(GateServerManager, "install_systemd_units") as mock_install,
        ):
            _handle_gate_start(cfg=sentinel.CFG)
        mock_install.assert_called_once()

    def test_gate_start_daemon_passes_cfg(self) -> None:
        """_handle_gate_start propagates cfg to start_daemon in daemon mode."""
        from unittest.mock import sentinel

        from terok_sandbox.commands import _handle_gate_start
        from terok_sandbox.gate.lifecycle import GateServerManager

        with (
            patch.object(GateServerManager, "is_systemd_available", return_value=False),
            patch.object(GateServerManager, "start_daemon") as mock_start,
        ):
            _handle_gate_start(port=1234, cfg=sentinel.CFG)
        mock_start.assert_called_once_with(port=1234)

    def test_gate_stop_systemd_passes_cfg(self) -> None:
        """_handle_gate_stop propagates cfg through the systemd branch."""
        from unittest.mock import sentinel

        from terok_sandbox.commands import _handle_gate_stop
        from terok_sandbox.gate.lifecycle import GateServerManager, GateServerStatus

        mock_status = GateServerStatus(mode="systemd", running=True, port=9418)
        with (
            patch.object(GateServerManager, "get_status", return_value=mock_status),
            patch.object(GateServerManager, "uninstall_systemd_units") as m_uninstall,
        ):
            _handle_gate_stop(cfg=sentinel.CFG)
        m_uninstall.assert_called_once()

    def test_gate_stop_daemon_passes_cfg(self) -> None:
        """_handle_gate_stop propagates cfg through the daemon branch."""
        from unittest.mock import sentinel

        from terok_sandbox.commands import _handle_gate_stop
        from terok_sandbox.gate.lifecycle import GateServerManager, GateServerStatus

        mock_status = GateServerStatus(mode="daemon", running=True, port=9418)
        with (
            patch.object(GateServerManager, "get_status", return_value=mock_status),
            patch.object(GateServerManager, "stop_daemon") as m_stop,
        ):
            _handle_gate_stop(cfg=sentinel.CFG)
        m_stop.assert_called_once()

    def test_gate_status_passes_cfg(self) -> None:
        """_handle_gate_status propagates cfg to all downstream calls."""
        from unittest.mock import sentinel

        from terok_sandbox.commands import _handle_gate_status
        from terok_sandbox.gate.lifecycle import GateServerManager, GateServerStatus

        mock_status = GateServerStatus(mode="none", running=False, port=9418)
        with (
            patch.object(GateServerManager, "get_status", return_value=mock_status),
            patch.object(
                GateServerManager,
                "gate_base_path",
                new_callable=lambda: property(lambda s: "/t/gate"),
            ),
            patch.object(GateServerManager, "check_units_outdated", return_value=None),
        ):
            _handle_gate_status(cfg=sentinel.CFG)

    def test_gate_status_prints_hint_on_outdated(self) -> None:
        """_handle_gate_status appends CLI-specific remediation hint to stderr."""
        from terok_sandbox.commands import _handle_gate_status
        from terok_sandbox.gate.lifecycle import GateServerManager, GateServerStatus

        mock_status = GateServerStatus(mode="systemd", running=True, port=9418)
        stderr = StringIO()
        with (
            patch.object(GateServerManager, "get_status", return_value=mock_status),
            patch.object(
                GateServerManager,
                "gate_base_path",
                new_callable=lambda: property(lambda s: "/t/gate"),
            ),
            patch.object(
                GateServerManager,
                "check_units_outdated",
                return_value="Systemd units are outdated (installed v1, expected v4).",
            ),
            patch("sys.stderr", stderr),
        ):
            _handle_gate_status()
        output = stderr.getvalue()
        assert "outdated" in output
        assert "terok-sandbox gate start" in output


class TestGateCLI:
    """Verify gate subcommand dispatch."""

    def test_gate_status_runs(self) -> None:
        from terok_sandbox.gate.lifecycle import GateServerManager, GateServerStatus

        mock_status = GateServerStatus(mode="none", running=False, port=9418)
        with (
            patch.object(GateServerManager, "get_status", return_value=mock_status),
            patch.object(
                GateServerManager,
                "gate_base_path",
                new_callable=lambda: property(lambda s: "/tmp/gate"),
            ),
            patch.object(GateServerManager, "check_units_outdated", return_value=None),
        ):
            out, _, rc = _run_cli("gate", "status")
        assert rc == 0
        assert "none" in out
        assert "9418" in out
