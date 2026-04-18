# SPDX-FileCopyrightText: 2026 Jiri Vyskocil
# SPDX-License-Identifier: Apache-2.0

"""Tests for vault CLI command handlers."""

from __future__ import annotations

from pathlib import Path
from unittest.mock import patch

import pytest

from terok_sandbox.commands import _handle_vault_start, _handle_vault_status, _handle_vault_stop
from terok_sandbox.vault.lifecycle import VaultManager, VaultStatus


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
