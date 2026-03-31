# SPDX-FileCopyrightText: 2026 Jiri Vyskocil
# SPDX-License-Identifier: Apache-2.0

"""Tests for credential proxy CLI command handlers."""

from __future__ import annotations

from pathlib import Path
from unittest.mock import patch

import pytest

from terok_sandbox.commands import _handle_proxy_start, _handle_proxy_status, _handle_proxy_stop
from terok_sandbox.credential_proxy_lifecycle import CredentialProxyStatus

_LIFECYCLE = "terok_sandbox.credential_proxy_lifecycle"


class TestProxyStart:
    """Verify _handle_proxy_start."""

    def test_already_running(self, capsys: pytest.CaptureFixture[str]) -> None:
        """Prints message and returns when proxy is already running."""
        status = CredentialProxyStatus(
            mode="daemon",
            running=True,
            healthy=True,
            socket_path=Path("/s"),
            db_path=Path("/d"),
            routes_path=Path("/r"),
            routes_configured=0,
            credentials_stored=(),
        )
        with patch(f"{_LIFECYCLE}.get_proxy_status", return_value=status):
            _handle_proxy_start()

        assert "already running" in capsys.readouterr().out

    def test_starts_daemon(self, capsys: pytest.CaptureFixture[str]) -> None:
        """Calls start_daemon and prints confirmation."""
        status = CredentialProxyStatus(
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
            patch(f"{_LIFECYCLE}.get_proxy_status", return_value=status),
            patch(f"{_LIFECYCLE}.start_daemon") as mock_start,
        ):
            _handle_proxy_start()

        mock_start.assert_called_once()
        assert "started" in capsys.readouterr().out


class TestProxyStop:
    """Verify _handle_proxy_stop."""

    def test_not_running(self, capsys: pytest.CaptureFixture[str]) -> None:
        """Prints message when proxy is not running."""
        with patch(f"{_LIFECYCLE}.is_daemon_running", return_value=False):
            _handle_proxy_stop()

        assert "not running" in capsys.readouterr().out

    def test_stops_daemon(self, capsys: pytest.CaptureFixture[str]) -> None:
        """Calls stop_daemon and prints confirmation."""
        with (
            patch(f"{_LIFECYCLE}.is_daemon_running", return_value=True),
            patch(f"{_LIFECYCLE}.stop_daemon") as mock_stop,
        ):
            _handle_proxy_stop()

        mock_stop.assert_called_once()
        assert "stopped" in capsys.readouterr().out


class TestProxyStatus:
    """Verify _handle_proxy_status."""

    def test_shows_running(self, capsys: pytest.CaptureFixture[str]) -> None:
        """Displays running status with socket and DB paths."""
        status = CredentialProxyStatus(
            mode="daemon",
            running=True,
            healthy=True,
            socket_path=Path("/run/s.sock"),
            db_path=Path("/d/c.db"),
            routes_path=Path("/r"),
            routes_configured=0,
            credentials_stored=(),
        )
        with patch(f"{_LIFECYCLE}.get_proxy_status", return_value=status):
            _handle_proxy_status()

        out = capsys.readouterr().out
        assert "running" in out
        assert "/run/s.sock" in out
        assert "/d/c.db" in out

    def test_shows_stopped(self, capsys: pytest.CaptureFixture[str]) -> None:
        """Displays stopped status."""
        status = CredentialProxyStatus(
            mode="none",
            running=False,
            healthy=False,
            socket_path=Path("/s"),
            db_path=Path("/d"),
            routes_path=Path("/r"),
            routes_configured=0,
            credentials_stored=(),
        )
        with patch(f"{_LIFECYCLE}.get_proxy_status", return_value=status):
            _handle_proxy_status()

        assert "stopped" in capsys.readouterr().out
