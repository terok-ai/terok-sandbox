# SPDX-FileCopyrightText: 2026 Jiri Vyskocil
# SPDX-License-Identifier: Apache-2.0

"""Tests for vault CLI command handlers."""

from __future__ import annotations

from pathlib import Path
from unittest.mock import patch

import pytest

from terok_sandbox.commands import _handle_vault_start, _handle_vault_status, _handle_vault_stop
from terok_sandbox.vault.daemon.lifecycle import VaultManager, VaultStatus


class TestVaultStart:
    """Verify the vault start command handler."""

    def test_already_running(self, capsys: pytest.CaptureFixture[str]) -> None:
        """Prints message and returns when vault is already running."""
        status = VaultStatus(
            mode="daemon",
            running=True,
            healthy=True,
            socket_path=Path("/s"),
            db_path=Path("/d"),
            routes_path=Path("/r"),
            routes_configured=0,
            credentials_stored=(),
        )
        with patch.object(VaultManager, "get_status", return_value=status):
            _handle_vault_start()

        assert "already running" in capsys.readouterr().out

    def test_starts_daemon(self, capsys: pytest.CaptureFixture[str]) -> None:
        """Calls start_daemon and prints confirmation."""
        status = VaultStatus(
            mode="none",
            running=False,
            healthy=False,
            socket_path=Path("/s"),
            db_path=Path("/d"),
            routes_path=Path("/r"),
            routes_configured=0,
            credentials_stored=(),
        )
        with (
            patch.object(VaultManager, "get_status", return_value=status),
            patch.object(VaultManager, "start_daemon") as mock_start,
        ):
            _handle_vault_start()

        mock_start.assert_called_once()
        assert "started" in capsys.readouterr().out


class TestVaultStop:
    """Verify the vault stop command handler."""

    def test_not_running(self, capsys: pytest.CaptureFixture[str]) -> None:
        """Prints message when vault is not running."""
        with patch.object(VaultManager, "is_daemon_running", return_value=False):
            _handle_vault_stop()

        assert "not running" in capsys.readouterr().out

    def test_stops_daemon(self, capsys: pytest.CaptureFixture[str]) -> None:
        """Calls stop_daemon and prints confirmation."""
        with (
            patch.object(VaultManager, "is_daemon_running", return_value=True),
            patch.object(VaultManager, "stop_daemon") as mock_stop,
        ):
            _handle_vault_stop()

        mock_stop.assert_called_once()
        assert "stopped" in capsys.readouterr().out


class TestVaultStatus:
    """Verify the vault status command handler."""

    def test_shows_running(self, capsys: pytest.CaptureFixture[str]) -> None:
        """Displays running status with socket and DB paths."""
        status = VaultStatus(
            mode="daemon",
            running=True,
            healthy=True,
            socket_path=Path("/run/s.sock"),
            db_path=Path("/d/c.db"),
            routes_path=Path("/r"),
            routes_configured=0,
            credentials_stored=(),
        )
        with patch.object(VaultManager, "get_status", return_value=status):
            _handle_vault_status()

        out = capsys.readouterr().out
        assert "running" in out
        assert "/run/s.sock" in out
        assert "/d/c.db" in out

    def test_shows_stopped(self, capsys: pytest.CaptureFixture[str]) -> None:
        """Displays stopped status."""
        status = VaultStatus(
            mode="none",
            running=False,
            healthy=False,
            socket_path=Path("/s"),
            db_path=Path("/d"),
            routes_path=Path("/r"),
            routes_configured=0,
            credentials_stored=(),
        )
        with patch.object(VaultManager, "get_status", return_value=status):
            _handle_vault_status()

        assert "stopped" in capsys.readouterr().out

    def test_shows_plaintext_passphrase_warning(self, capsys: pytest.CaptureFixture[str]) -> None:
        """``plaintext_passphrase_path`` triggers a stderr WARNING naming the file."""
        config_path = Path("/etc/terok/config.yml")
        status = VaultStatus(
            mode="systemd",
            running=True,
            healthy=True,
            socket_path=Path("/s"),
            db_path=Path("/d"),
            routes_path=Path("/r"),
            routes_configured=0,
            credentials_stored=(),
            plaintext_passphrase_path=config_path,
        )
        with patch.object(VaultManager, "get_status", return_value=status):
            _handle_vault_status()

        captured = capsys.readouterr()
        # Warning lives on stderr so structured stdout fields stay greppable.
        assert "WARNING" in captured.err
        assert "plaintext" in captured.err
        assert str(config_path) in captured.err
        # Stdout still carries the structured fields without warning noise.
        assert "WARNING" not in captured.out

    def test_no_warning_when_plaintext_path_is_none(
        self, capsys: pytest.CaptureFixture[str]
    ) -> None:
        """Default-None case is silent — no plaintext-passphrase line at all."""
        status = VaultStatus(
            mode="systemd",
            running=True,
            healthy=True,
            socket_path=Path("/s"),
            db_path=Path("/d"),
            routes_path=Path("/r"),
            routes_configured=0,
            credentials_stored=(),
        )
        with patch.object(VaultManager, "get_status", return_value=status):
            _handle_vault_status()

        captured = capsys.readouterr()
        assert "WARNING" not in captured.err
        assert "plaintext" not in captured.err


class TestVaultStatusCredentialsListing:
    """Branch where credentials_stored is non-empty — prints the comma-separated list."""

    def test_shows_credentials_when_present(self, capsys: pytest.CaptureFixture[str]) -> None:
        """Stored provider names render as a sanitized comma-separated list."""
        status = VaultStatus(
            mode="daemon",
            running=True,
            healthy=True,
            socket_path=Path("/s"),
            db_path=Path("/d"),
            routes_path=Path("/r"),
            routes_configured=2,
            credentials_stored=("claude", "github"),
        )
        with patch.object(VaultManager, "get_status", return_value=status):
            _handle_vault_status()
        out = capsys.readouterr().out
        assert "Credentials: claude, github" in out
        assert "none stored" not in out


class TestDefaultCfgConstruction:
    """Handlers without an explicit ``cfg=`` argument build a default ``SandboxConfig``."""

    def test_unlock_defaults_cfg(self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
        """``vault unlock`` without ``cfg=`` exercises the SandboxConfig() branch."""
        from terok_sandbox.commands import _handle_vault_unlock, vault as vault_cmds

        captured_cfg = {}

        def _fake_sandbox_config():
            from terok_sandbox.config import SandboxConfig

            cfg = SandboxConfig(
                state_dir=tmp_path / "state",
                runtime_dir=tmp_path / "rt",
                config_dir=tmp_path / "cfg",
                vault_dir=tmp_path / "vault",
                services_mode="socket",
            )
            captured_cfg["cfg"] = cfg
            return cfg

        monkeypatch.setattr(vault_cmds, "SandboxConfig", _fake_sandbox_config)
        monkeypatch.setattr(
            "terok_sandbox.vault.store.encryption.prompt_passphrase", lambda **_: "pw"
        )
        with patch.object(VaultManager, "is_daemon_running", return_value=False):
            _handle_vault_unlock()  # cfg= omitted → default factory fires
        assert "cfg" in captured_cfg

    def test_lock_defaults_cfg(self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
        """``vault lock`` without ``cfg=``."""
        from terok_sandbox.commands import _handle_vault_lock, vault as vault_cmds

        called = {}

        def _fake_sandbox_config():
            from terok_sandbox.config import SandboxConfig

            cfg = SandboxConfig(
                state_dir=tmp_path / "state",
                runtime_dir=tmp_path / "rt",
                config_dir=tmp_path / "cfg",
                vault_dir=tmp_path / "vault",
                services_mode="socket",
            )
            called["cfg"] = cfg
            return cfg

        monkeypatch.setattr(vault_cmds, "SandboxConfig", _fake_sandbox_config)
        with patch.object(VaultManager, "is_daemon_running", return_value=False):
            _handle_vault_lock()
        assert "cfg" in called

    def test_seal_defaults_cfg(self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
        """``vault seal`` without ``cfg=`` (systemd-creds unavailable → early SystemExit)."""
        from terok_sandbox.commands import handle_vault_seal, vault as vault_cmds

        called = {}

        def _fake_sandbox_config():
            from terok_sandbox.config import SandboxConfig

            cfg = SandboxConfig(
                state_dir=tmp_path / "state",
                runtime_dir=tmp_path / "rt",
                config_dir=tmp_path / "cfg",
                vault_dir=tmp_path / "vault",
                services_mode="socket",
            )
            called["cfg"] = cfg
            return cfg

        monkeypatch.setattr(vault_cmds, "SandboxConfig", _fake_sandbox_config)
        monkeypatch.setattr("terok_sandbox.vault.store.systemd_creds.is_available", lambda: False)
        with pytest.raises(SystemExit, match="systemd-creds unavailable"):
            handle_vault_seal()
        assert "cfg" in called
