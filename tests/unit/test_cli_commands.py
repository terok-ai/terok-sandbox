# SPDX-FileCopyrightText: 2026 Jiri Vyskocil
# SPDX-License-Identifier: Apache-2.0

"""Tests for the terok-sandbox CLI and command registry."""

from __future__ import annotations

from io import StringIO
from unittest.mock import patch

import pytest

from terok_sandbox.cli import main
from terok_sandbox.commands import (
    COMMANDS,
    GATE_COMMANDS,
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
        for _path, cmd in COMMANDS.walk():
            assert cmd.name, f"Command missing name: {cmd}"

    def test_every_leaf_has_a_handler(self) -> None:
        """Group nodes have ``children`` and no handler; leaves have a handler."""
        for path, cmd in COMMANDS.walk():
            if cmd.children:
                assert cmd.handler is None, f"group {'.'.join(path)} has a handler"
            else:
                assert cmd.handler is not None, f"leaf {'.'.join(path)} has no handler"

    def test_gate_subverbs_present(self) -> None:
        gate = COMMANDS.find_at(("gate",))
        names = {c.name for c in gate.children}
        assert {"path"} <= names

    def test_shield_subverbs_present(self) -> None:
        shield = COMMANDS.find_at(("shield",))
        names = {c.name for c in shield.children}
        assert {"install-hooks", "status"} <= names

    def test_ssh_subverbs_present(self) -> None:
        ssh = COMMANDS.find_at(("ssh",))
        names = {c.name for c in ssh.children}
        assert {"import", "add", "remove"} <= names

    def test_vault_passphrase_nested_subgroup(self) -> None:
        """The ``vault passphrase`` subgroup is reachable as a nested CommandDef."""
        passphrase = COMMANDS.find_at(("vault", "passphrase"))
        names = {c.name for c in passphrase.children}
        assert {"seal", "to-keyring", "reveal", "acknowledge", "destroy"} == names


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

    def test_shield_install_hooks_delegates(self) -> None:
        """``shield install-hooks`` calls ``ShieldHooks.install()`` — no scope flags."""
        from terok_sandbox.commands import _handle_shield_setup

        with patch("terok_sandbox.integrations.shield.ShieldHooks.install") as install:
            _handle_shield_setup()
        install.assert_called_once_with()

    def test_gate_no_subcommand_shows_help(self) -> None:
        out, _, _ = _run_cli("gate")
        assert "path" in out.lower()

    def test_ssh_no_subcommand_shows_help(self) -> None:
        out, _, _ = _run_cli("ssh")
        assert "import" in out.lower()


class TestShieldCLI:
    """Verify shield subcommand dispatch."""

    def test_install_hooks_delegates_to_shield_hooks_install(self) -> None:
        """``shield install-hooks`` reaches ``ShieldHooks.install()``."""
        from terok_sandbox.commands import _handle_shield_setup

        with patch("terok_sandbox.integrations.shield.ShieldHooks.install") as install:
            _handle_shield_setup()
        install.assert_called_once_with()

    def test_shield_status_runs(self) -> None:
        """``shield status`` resolves through shield's own registry handler.

        Sandbox no longer hand-rolls a separate status function — the
        verb is consumed from terok-shield's COMMANDS via the
        CommandTree.  Mock at the Shield instance level so the
        sandbox-wrapped path exercises the same code shield's
        standalone CLI runs.
        """
        from terok_shield import EnvironmentCheck

        mock_env = EnvironmentCheck(ok=True, hooks="per-container", health="ok")
        mock_cfg = {"mode": "hook", "profiles": ["dev-standard"], "audit_enabled": True}
        with (
            patch("terok_shield.Shield.status", return_value=mock_cfg),
            patch("terok_shield.Shield.check_environment", return_value=mock_env),
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

        gate = GATE_COMMANDS[0]
        for cmd in gate.children:
            sig = inspect.signature(cmd.handler)
            assert "cfg" in sig.parameters, f"{cmd.handler.__name__} missing cfg param"

    def test_ssh_handlers_accept_cfg(self) -> None:
        import inspect

        ssh = SSH_COMMANDS[0]
        for cmd in ssh.children:
            sig = inspect.signature(cmd.handler)
            assert "cfg" in sig.parameters, f"{cmd.handler.__name__} missing cfg param"


class TestGatePathVerb:
    """Verify the read-only ``gate path`` verb prints the mirror's file:// URL."""

    def test_prints_file_url_under_gate_base(self, capsys, tmp_path) -> None:
        from terok_sandbox import SandboxConfig
        from terok_sandbox.commands import _handle_gate_path

        cfg = SandboxConfig(state_dir=tmp_path / "state")
        _handle_gate_path(project="myproj", cfg=cfg)
        out = capsys.readouterr().out.strip()
        expected = (cfg.gate_base_path / "myproj.git").as_uri()
        assert out == expected
        assert out.startswith("file://")

    def test_gate_path_via_cli(self) -> None:
        """``gate path <project>`` resolves through the CLI and prints the URL."""
        out, _, rc = _run_cli("gate", "path", "myproj")
        assert rc == 0
        assert out.strip().startswith("file://")
        assert "myproj.git" in out

    @pytest.mark.parametrize(
        "project",
        ["..", ".", "../other/repo", "a/b", "/abs/path", "back\\slash"],
    )
    def test_rejects_path_traversal(self, project: str, tmp_path) -> None:
        """A *project* with path separators or parent tokens is rejected."""
        from terok_sandbox import SandboxConfig
        from terok_sandbox.commands import _handle_gate_path

        cfg = SandboxConfig(state_dir=tmp_path / "state")
        with pytest.raises(SystemExit):
            _handle_gate_path(project=project, cfg=cfg)
