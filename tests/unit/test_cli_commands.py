# SPDX-FileCopyrightText: 2026 Jiri Vyskocil
# SPDX-License-Identifier: Apache-2.0

"""Tests for the terok-sandbox CLI and command registry."""

from __future__ import annotations

import argparse
from io import StringIO
from unittest.mock import patch

import pytest
from terok_shield.profiles import UnknownProfileError
from terok_util import SetupRequiredError

from terok_sandbox.cli import main
from terok_sandbox.commands import (
    COMMANDS,
    GATE_COMMANDS,
    SSH_COMMANDS,
    CommandDef,
    CommandTree,
)

#: Fully-resolved view of the lazy forest for structural assertions.  The
#: live ``COMMANDS`` defers each verb to its module via a ``source``
#: string (so building it imports nothing); ``resolve()`` materialises
#: every root's real subtree for the well-formedness checks below.
_RESOLVED = CommandTree([root.resolve() for root in COMMANDS])


def _resolve_handler(handler: object) -> object:
    """Resolve a possibly-lazy handler to the underlying function for introspection.

    Handlers are registered as [`LazyHandler`][terok_util.cli_types.LazyHandler]
    ``"module:qualname"`` targets, so signature checks import and unwrap
    the real callable first.
    """
    import importlib

    target = getattr(handler, "target", None)
    if target is None:
        return handler
    module_name, _, qualname = target.partition(":")
    obj: object = importlib.import_module(module_name)
    for part in qualname.split("."):
        obj = getattr(obj, part)
    return obj


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

    def test_roots_are_lazy(self) -> None:
        """Every top-level verb is a lazy ``source`` reference (imports on dispatch)."""
        for cmd in COMMANDS:
            assert cmd.is_lazy, f"root {cmd.name!r} should defer to a source module"

    def test_every_leaf_has_a_handler(self) -> None:
        """Group nodes have ``children`` and no handler; leaves have a handler."""
        for path, cmd in _RESOLVED.walk():
            if cmd.children:
                assert cmd.handler is None, f"group {'.'.join(path)} has a handler"
            else:
                assert cmd.handler is not None, f"leaf {'.'.join(path)} has no handler"

    def test_gate_subverbs_present(self) -> None:
        gate = _RESOLVED.find_at(("gate",))
        names = {c.name for c in gate.children}
        assert {"path"} <= names

    def test_shield_subverbs_present(self) -> None:
        shield = _RESOLVED.find_at(("shield",))
        names = {c.name for c in shield.children}
        assert {"install-hooks", "status"} <= names

    def test_ssh_subverbs_present(self) -> None:
        ssh = _RESOLVED.find_at(("ssh",))
        names = {c.name for c in ssh.children}
        assert {"import", "add", "default", "pub", "remove"} <= names

    def test_vault_passphrase_nested_subgroup(self) -> None:
        """The ``vault passphrase`` subgroup is reachable as a nested CommandDef."""
        passphrase = _RESOLVED.find_at(("vault", "passphrase"))
        names = {c.name for c in passphrase.children}
        # ``destroy`` folded into ``vault lock`` (lock now clears every tier).
        assert {"seal", "to-desktop-keyring", "reveal", "acknowledge", "change"} == names


# ---------------------------------------------------------------------------
# CLI dispatch
# ---------------------------------------------------------------------------


class TestCLIBasics:
    """Verify basic CLI behaviour."""

    def test_ambiguous_desktop_transfer_command_is_rejected(self) -> None:
        """Only the explicit desktop-keyring command is accepted, without an old alias."""
        _out, err, code = _run_cli("vault", "passphrase", "to-keyring")
        assert code == 2
        assert "invalid choice" in err
        assert "to-desktop-keyring" in err

    def test_no_command_shows_help(self) -> None:
        out, err, rc = _run_cli()
        assert rc == 1
        combined = (out + err).lower()
        assert "usage:" in combined

    def test_version(self) -> None:
        out, _, rc = _run_cli("--version")
        assert rc == 0
        assert "terok-sandbox" in out

    def test_setup_error_is_an_actionable_exit(self) -> None:
        """Typed setup failures reach CLI users without an internal traceback."""
        with patch.object(
            CommandTree, "dispatch", side_effect=SetupRequiredError("Run setup again")
        ):
            with pytest.raises(SystemExit, match="Run setup again") as error:
                main(["setup"])
        assert error.value.__suppress_context__

    def test_unknown_profile_is_an_actionable_exit(self) -> None:
        """A ``--profiles`` name that names no profile lists the ones that exist."""
        with (
            patch(
                "terok_sandbox.launch.compose",
                side_effect=UnknownProfileError("Unknown profile 'typo'; available profiles: base"),
            ),
            pytest.raises(SystemExit, match="available profiles: base"),
        ):
            main(["prepare", "ctr"])

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

    def test_shield_status_runs(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """The real registry handler renders status without requiring host tools."""
        from terok_shield import EnvironmentCheck

        monkeypatch.setenv("PATH", "")
        mock_env = EnvironmentCheck(ok=True, hooks="per-container", health="ok")
        mock_cfg = {"mode": "hook", "profiles": ["dev-standard"], "audit_enabled": True}
        with (
            patch("terok_sandbox.commands.shield._shield_build_config") as build_config,
            patch("terok_sandbox.commands.shield.Shield", autospec=True) as shield_factory,
        ):
            shield = shield_factory.return_value
            shield.status.return_value = mock_cfg
            shield.check_environment.return_value = mock_env
            out, _, rc = _run_cli("shield", "status")
        build_config.assert_called_once_with(None)
        shield_factory.assert_called_once_with(build_config.return_value)
        shield.status.assert_called_once_with()
        shield.check_environment.assert_called_once_with()
        assert rc == 0
        assert "hook" in out
        assert "dev-standard" in out
        assert "enabled" in out


class TestSSHArguments:
    """SSH shortcuts and selectors keep the standalone and embedded CLIs aligned."""

    @pytest.mark.parametrize("flag", ["-c", "--comment"])
    @pytest.mark.parametrize(
        "command",
        [
            ("add", "proj"),
            ("import", "proj", "--private-key", "key"),
            ("remove", "--scope", "proj"),
        ],
    )
    def test_comment_aliases(self, flag: str, command: tuple[str, ...]) -> None:
        """Each existing comment option accepts its long and short spelling."""
        parser = argparse.ArgumentParser()
        CommandTree(SSH_COMMANDS).wire(parser)
        args = parser.parse_args(["ssh", *command, flag, "deploy"])
        assert args.comment == "deploy"

    def test_default_accepts_scope_and_integer_key_id(self) -> None:
        """The default verb selects an existing key by its displayed database ID."""
        parser = argparse.ArgumentParser()
        CommandTree(SSH_COMMANDS).wire(parser)
        args = parser.parse_args(["ssh", "default", "proj", "42"])
        assert args.scope == "proj"
        assert args.key_id == 42
        assert args._cmd.name == "default"

    def test_pub_rejects_obsolete_all_flag(self) -> None:
        """Printing all public keys needs no opt-in flag."""
        parser = argparse.ArgumentParser()
        CommandTree(SSH_COMMANDS).wire(parser)
        with pytest.raises(SystemExit) as exc:
            parser.parse_args(["ssh", "pub", "proj", "--all"])
        assert exc.value.code == 2


class TestHandlerCfgSignatures:
    """All command handlers in config-injected groups accept a ``cfg`` keyword argument."""

    def test_gate_handlers_accept_cfg(self) -> None:
        import inspect

        gate = GATE_COMMANDS[0]
        for cmd in gate.children:
            fn = _resolve_handler(cmd.handler)
            assert "cfg" in inspect.signature(fn).parameters, f"{fn.__name__} missing cfg param"

    def test_ssh_handlers_accept_cfg(self) -> None:
        import inspect

        ssh = SSH_COMMANDS[0]
        for cmd in ssh.children:
            fn = _resolve_handler(cmd.handler)
            assert "cfg" in inspect.signature(fn).parameters, f"{fn.__name__} missing cfg param"


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
