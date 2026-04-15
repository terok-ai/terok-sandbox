# SPDX-FileCopyrightText: 2026 Jiri Vyskocil
# SPDX-License-Identifier: Apache-2.0

"""Tests for credential proxy lifecycle management."""

from __future__ import annotations

import os
import subprocess
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from terok_sandbox.config import SandboxConfig
from terok_sandbox.credentials.lifecycle import (
    CredentialProxyManager,
    CredentialProxyStatus,
    ProxyUnreachableError,
)


def _make_cfg(tmp_path: Path) -> SandboxConfig:
    """Create a SandboxConfig rooted in tmp_path."""
    return SandboxConfig(
        state_dir=tmp_path / "state",
        runtime_dir=tmp_path / "run",
        credentials_dir=tmp_path / "credentials",
    )


def _make_mgr(tmp_path: Path) -> CredentialProxyManager:
    """Create a CredentialProxyManager with a test config."""
    return CredentialProxyManager(_make_cfg(tmp_path))


class TestPidFile:
    """Verify PID file path resolution."""

    def test_pid_file_uses_config(self, tmp_path: Path) -> None:
        """PID file path comes from config's proxy_pid_file_path."""
        cfg = _make_cfg(tmp_path)
        mgr = CredentialProxyManager(cfg)
        assert mgr._cfg.proxy_pid_file_path == cfg.proxy_pid_file_path

    def test_pid_file_default_config(self) -> None:
        """PID file resolves without explicit config."""
        mgr = CredentialProxyManager()
        assert "credential-proxy.pid" in str(mgr._cfg.proxy_pid_file_path)


class TestIsManagedProxy:
    """Verify cmdline-based PID validation."""

    def test_no_proc_entry(self, tmp_path: Path) -> None:
        """Non-existent PID returns False."""
        assert _make_mgr(tmp_path)._is_managed_proxy(999999999) is False

    def test_current_process_is_not_proxy(self, tmp_path: Path) -> None:
        """Current process (pytest) is not the proxy."""
        assert _make_mgr(tmp_path)._is_managed_proxy(os.getpid()) is False

    def test_matches_when_flag_present(self, tmp_path: Path) -> None:
        """Returns True when /proc/PID/cmdline contains --pid-file=<expected>."""
        mgr = _make_mgr(tmp_path)
        expected_flag = f"--pid-file={mgr._cfg.proxy_pid_file_path}"
        fake_cmdline = f"terok-credential-proxy\x00{expected_flag}\x00".encode()

        with (
            patch("pathlib.Path.is_file", return_value=True),
            patch("pathlib.Path.read_bytes", return_value=fake_cmdline),
        ):
            assert mgr._is_managed_proxy(12345) is True

    def test_rejects_different_pid_file(self, tmp_path: Path) -> None:
        """Returns False when --pid-file points to a different path."""
        mgr = _make_mgr(tmp_path)
        fake_cmdline = b"terok-credential-proxy\x00--pid-file=/other/path.pid\x00"

        with (
            patch("pathlib.Path.is_file", return_value=True),
            patch("pathlib.Path.read_bytes", return_value=fake_cmdline),
        ):
            assert mgr._is_managed_proxy(12345) is False


class TestStartDaemon:
    """Verify start_daemon behaviour."""

    def test_missing_routes_file_creates_empty_with_restricted_perms(self, tmp_path: Path) -> None:
        """start_daemon creates an empty routes file with 0o600 permissions."""
        cfg = _make_cfg(tmp_path)
        mgr = CredentialProxyManager(cfg)

        mock_proc = MagicMock()
        mock_proc.poll.return_value = None

        with (
            patch("subprocess.Popen", return_value=mock_proc),
            patch.object(CredentialProxyManager, "_wait_for_ready", return_value=True),
        ):
            mgr.start_daemon()

        assert cfg.proxy_routes_path.is_file()
        assert cfg.proxy_routes_path.read_text() == "{}\n"
        assert oct(cfg.proxy_routes_path.stat().st_mode & 0o777) == oct(0o600)

    def test_start_launches_subprocess(self, tmp_path: Path) -> None:
        """start_daemon calls Popen with the correct command."""
        cfg = _make_cfg(tmp_path)
        mgr = CredentialProxyManager(cfg)
        routes = cfg.proxy_routes_path
        routes.parent.mkdir(parents=True, exist_ok=True)
        routes.write_text("{}")

        mock_proc = MagicMock()
        mock_proc.poll.return_value = None

        with (
            patch("subprocess.Popen", return_value=mock_proc) as mock_popen,
            patch.object(CredentialProxyManager, "_wait_for_ready", return_value=True),
        ):
            mgr.start_daemon()

        import sys

        cmd = mock_popen.call_args[0][0]
        assert cmd[0] == sys.executable
        assert cmd[1:3] == ["-m", "terok_sandbox.credentials.proxy"]
        assert any("--socket-path=" in a for a in cmd)
        assert any("--pid-file=" in a for a in cmd)
        assert "--log-level=INFO" in cmd

    def test_immediate_exit_raises(self, tmp_path: Path) -> None:
        """start_daemon raises SystemExit if the process dies during readiness wait."""
        cfg = _make_cfg(tmp_path)
        mgr = CredentialProxyManager(cfg)
        routes = cfg.proxy_routes_path
        routes.parent.mkdir(parents=True, exist_ok=True)
        routes.write_text("{}")

        mock_proc = MagicMock()
        mock_proc.poll.return_value = 1
        mock_proc.stderr.read.return_value = b"error: bad config"

        with (
            patch("subprocess.Popen", return_value=mock_proc),
            patch.object(CredentialProxyManager, "_wait_for_ready", return_value=False),
            pytest.raises(SystemExit, match="failed to start"),
        ):
            mgr.start_daemon()

    def test_debug_log_level_via_env(self, tmp_path: Path) -> None:
        """TEROK_PROXY_LOG_LEVEL env var overrides the default INFO log level."""
        cfg = _make_cfg(tmp_path)
        mgr = CredentialProxyManager(cfg)
        routes = cfg.proxy_routes_path
        routes.parent.mkdir(parents=True, exist_ok=True)
        routes.write_text("{}")

        mock_proc = MagicMock()
        mock_proc.poll.return_value = None

        with (
            patch("subprocess.Popen", return_value=mock_proc) as mock_popen,
            patch.object(CredentialProxyManager, "_wait_for_ready", return_value=True),
            patch.dict(os.environ, {"TEROK_PROXY_LOG_LEVEL": "DEBUG"}),
        ):
            mgr.start_daemon()

        cmd = mock_popen.call_args[0][0]
        assert "--log-level=DEBUG" in cmd

    def test_timeout_without_crash_raises(self, tmp_path: Path) -> None:
        """start_daemon raises SystemExit when proxy stays alive but never becomes ready."""
        cfg = _make_cfg(tmp_path)
        mgr = CredentialProxyManager(cfg)
        routes = cfg.proxy_routes_path
        routes.parent.mkdir(parents=True, exist_ok=True)
        routes.write_text("{}")

        mock_proc = MagicMock()
        mock_proc.poll.return_value = None  # still running, just not ready

        with (
            patch("subprocess.Popen", return_value=mock_proc),
            patch.object(CredentialProxyManager, "_wait_for_ready", return_value=False),
            pytest.raises(SystemExit, match="did not become ready"),
        ):
            mgr.start_daemon()


class TestStopDaemon:
    """Verify stop_daemon behaviour."""

    def test_no_pidfile_is_noop(self, tmp_path: Path) -> None:
        """stop_daemon does nothing when PID file doesn't exist."""
        _make_mgr(tmp_path).stop_daemon()  # should not raise

    def test_sends_sigterm_to_managed_process(self, tmp_path: Path) -> None:
        """stop_daemon sends SIGTERM when the PID is a managed proxy."""
        mgr = _make_mgr(tmp_path)
        pidfile = mgr._cfg.proxy_pid_file_path
        pidfile.parent.mkdir(parents=True, exist_ok=True)
        pidfile.write_text("42")

        with (
            patch.object(CredentialProxyManager, "_is_managed_proxy", return_value=True),
            patch("os.kill") as mock_kill,
        ):
            mgr.stop_daemon()

        mock_kill.assert_called_once_with(42, 15)  # SIGTERM = 15
        assert not pidfile.exists()

    def test_stale_pid_cleans_up(self, tmp_path: Path) -> None:
        """stop_daemon cleans PID file even when process is gone."""
        mgr = _make_mgr(tmp_path)
        pidfile = mgr._cfg.proxy_pid_file_path
        pidfile.parent.mkdir(parents=True, exist_ok=True)
        pidfile.write_text("99999999")

        with (
            patch.object(CredentialProxyManager, "_is_managed_proxy", return_value=True),
            patch("os.kill", side_effect=ProcessLookupError),
        ):
            mgr.stop_daemon()

        assert not pidfile.exists()


class TestIsDaemonRunning:
    """Verify is_daemon_running behaviour."""

    def test_no_pidfile(self, tmp_path: Path) -> None:
        """Returns False when PID file doesn't exist."""
        assert _make_mgr(tmp_path).is_daemon_running() is False

    def test_valid_managed_pid(self, tmp_path: Path) -> None:
        """Returns True for a valid managed PID."""
        mgr = _make_mgr(tmp_path)
        pidfile = mgr._cfg.proxy_pid_file_path
        pidfile.parent.mkdir(parents=True, exist_ok=True)
        pidfile.write_text("42")

        with (
            patch.object(CredentialProxyManager, "_is_managed_proxy", return_value=True),
            patch("os.kill"),
        ):  # signal 0 succeeds
            assert mgr.is_daemon_running() is True

    def test_not_our_daemon(self, tmp_path: Path) -> None:
        """Returns False when PID doesn't match our cmdline."""
        mgr = _make_mgr(tmp_path)
        pidfile = mgr._cfg.proxy_pid_file_path
        pidfile.parent.mkdir(parents=True, exist_ok=True)
        pidfile.write_text("42")

        with patch.object(CredentialProxyManager, "_is_managed_proxy", return_value=False):
            assert mgr.is_daemon_running() is False

    def test_stale_pid(self, tmp_path: Path) -> None:
        """Returns False when PID is gone."""
        mgr = _make_mgr(tmp_path)
        pidfile = mgr._cfg.proxy_pid_file_path
        pidfile.parent.mkdir(parents=True, exist_ok=True)
        pidfile.write_text("99999999")

        with (
            patch.object(CredentialProxyManager, "_is_managed_proxy", return_value=True),
            patch("os.kill", side_effect=ProcessLookupError),
        ):
            assert mgr.is_daemon_running() is False


_LIFECYCLE = "terok_sandbox.credentials.lifecycle"


def _no_systemd():
    """Context manager that patches out systemd detection."""
    return patch.object(CredentialProxyManager, "is_socket_installed", return_value=False)


class TestGetProxyStatus:
    """Verify get_proxy_status."""

    def test_returns_status_dataclass(self, tmp_path: Path) -> None:
        """Returns a CredentialProxyStatus with correct fields."""
        cfg = _make_cfg(tmp_path)
        mgr = CredentialProxyManager(cfg)
        with (
            _no_systemd(),
            patch.object(CredentialProxyManager, "is_daemon_running", return_value=False),
        ):
            status = mgr.get_status()

        assert isinstance(status, CredentialProxyStatus)
        assert status.mode == "none"
        assert status.running is False
        assert status.healthy is False
        assert status.socket_path == cfg.proxy_socket_path
        assert status.db_path == cfg.proxy_db_path
        assert status.routes_path == cfg.proxy_routes_path
        assert status.routes_configured == 0
        assert status.credentials_stored == ()

    def test_counts_routes_from_json(self, tmp_path: Path) -> None:
        """Routes count reflects the number of entries in routes.json."""
        cfg = _make_cfg(tmp_path)
        routes = cfg.proxy_routes_path
        routes.parent.mkdir(parents=True, exist_ok=True)
        routes.write_text('{"github": {}, "gitlab": {}}')

        mgr = CredentialProxyManager(cfg)
        with (
            _no_systemd(),
            patch.object(CredentialProxyManager, "is_daemon_running", return_value=False),
        ):
            status = mgr.get_status()

        assert status.routes_configured == 2

    def test_invalid_routes_json_yields_zero(self, tmp_path: Path) -> None:
        """Invalid JSON in routes.json yields routes_configured=0."""
        cfg = _make_cfg(tmp_path)
        routes = cfg.proxy_routes_path
        routes.parent.mkdir(parents=True, exist_ok=True)
        routes.write_text("not valid json{{{")

        mgr = CredentialProxyManager(cfg)
        with (
            _no_systemd(),
            patch.object(CredentialProxyManager, "is_daemon_running", return_value=False),
        ):
            status = mgr.get_status()

        assert status.routes_configured == 0

    def test_lists_stored_credentials(self, tmp_path: Path) -> None:
        """credentials_stored lists providers from the credential DB."""
        from terok_sandbox.credentials.db import CredentialDB

        cfg = _make_cfg(tmp_path)
        db = CredentialDB(cfg.proxy_db_path)
        db.store_credential("default", "github", {"token": "abc"})
        db.store_credential("default", "anthropic", {"key": "xyz"})
        db.close()

        mgr = CredentialProxyManager(cfg)
        with (
            _no_systemd(),
            patch.object(CredentialProxyManager, "is_daemon_running", return_value=False),
        ):
            status = mgr.get_status()

        assert set(status.credentials_stored) == {"github", "anthropic"}

    def test_no_db_yields_empty_credentials(self, tmp_path: Path) -> None:
        """credentials_stored is empty when the DB file doesn't exist."""
        cfg = _make_cfg(tmp_path)
        assert not cfg.proxy_db_path.is_file()

        mgr = CredentialProxyManager(cfg)
        with (
            _no_systemd(),
            patch.object(CredentialProxyManager, "is_daemon_running", return_value=False),
        ):
            status = mgr.get_status()

        assert status.credentials_stored == ()


class TestEnsureProxyReachable:
    """Verify ensure_proxy_reachable."""

    def test_passes_when_daemon_healthy(self, tmp_path: Path) -> None:
        """No exception when daemon is running and health probe succeeds."""
        mgr = _make_mgr(tmp_path)
        with (
            patch.object(CredentialProxyManager, "is_socket_active", return_value=False),
            patch.object(CredentialProxyManager, "is_daemon_running", return_value=True),
            patch.object(CredentialProxyManager, "_wait_for_ready", return_value=True),
            patch.object(CredentialProxyManager, "_wait_for_tcp_port", return_value=True),
        ):
            mgr.ensure_reachable()  # should not raise

    def test_raises_when_daemon_running_but_unhealthy(self, tmp_path: Path) -> None:
        """Raises SystemExit when daemon is alive but health probe fails."""
        mgr = _make_mgr(tmp_path)
        with (
            patch.object(CredentialProxyManager, "is_socket_active", return_value=False),
            patch.object(CredentialProxyManager, "is_daemon_running", return_value=True),
            patch.object(CredentialProxyManager, "_wait_for_ready", return_value=False),
            pytest.raises(SystemExit, match="not reachable"),
        ):
            mgr.ensure_reachable()

    def test_raises_when_stopped(self, tmp_path: Path) -> None:
        """Raises ProxyUnreachableError when daemon is down."""
        mgr = _make_mgr(tmp_path)
        with (
            patch.object(CredentialProxyManager, "is_socket_active", return_value=False),
            patch.object(CredentialProxyManager, "is_daemon_running", return_value=False),
            pytest.raises(ProxyUnreachableError, match="not reachable"),
        ):
            mgr.ensure_reachable()

    def test_passes_when_socket_active(self, tmp_path: Path) -> None:
        """Socket active → starts service, waits for health + SSH agent port."""
        mgr = _make_mgr(tmp_path)
        with (
            patch.object(CredentialProxyManager, "is_socket_active", return_value=True),
            patch(f"{_LIFECYCLE}.subprocess") as mock_sub,
            patch.object(CredentialProxyManager, "_wait_for_ready", return_value=True),
            patch.object(CredentialProxyManager, "_wait_for_tcp_port", return_value=True),
        ):
            mgr.ensure_reachable()  # should not raise
            mock_sub.run.assert_called_once()  # systemctl --user start

    def test_raises_when_health_unreachable(self, tmp_path: Path) -> None:
        """Service started but health endpoint never responds → SystemExit."""
        mgr = _make_mgr(tmp_path)
        with (
            patch.object(CredentialProxyManager, "is_socket_active", return_value=True),
            patch(f"{_LIFECYCLE}.subprocess"),
            patch.object(CredentialProxyManager, "_wait_for_ready", return_value=False),
            pytest.raises(SystemExit, match="not reachable"),
        ):
            mgr.ensure_reachable()

    def test_raises_when_neither_socket_nor_tcp_reachable(self, tmp_path: Path) -> None:
        """Service started but neither Unix socket nor TCP responds → SystemExit."""
        mgr = _make_mgr(tmp_path)
        with (
            patch.object(CredentialProxyManager, "is_socket_active", return_value=True),
            patch(f"{_LIFECYCLE}.subprocess"),
            patch.object(CredentialProxyManager, "_wait_for_unix_socket", return_value=False),
            patch.object(CredentialProxyManager, "_wait_for_ready", return_value=False),
            pytest.raises(SystemExit, match="not reachable"),
        ):
            mgr.ensure_reachable()


class TestSystemdHelpers:
    """Verify systemd detection and socket status helpers."""

    def test_systemd_unit_dir_default(self) -> None:
        """Unit dir falls back to ~/.config/systemd/user when XDG_CONFIG_HOME is unset."""
        env = {k: v for k, v in os.environ.items() if k != "XDG_CONFIG_HOME"}
        with (
            patch.dict(os.environ, env, clear=True),
            patch("os.geteuid", return_value=1000),
        ):
            d = CredentialProxyManager._systemd_unit_dir()
        assert d == Path.home() / ".config" / "systemd" / "user"

    def test_systemd_unit_dir_xdg(self, tmp_path: Path) -> None:
        """Unit dir respects XDG_CONFIG_HOME when under home."""
        xdg = tmp_path / "custom-config"
        xdg.mkdir()
        with (
            patch.dict(os.environ, {"XDG_CONFIG_HOME": str(xdg)}),
            patch("os.geteuid", return_value=1000),
            patch("pathlib.Path.home", return_value=tmp_path),
        ):
            d = CredentialProxyManager._systemd_unit_dir()
        assert d == xdg / "systemd" / "user"

    def test_systemd_unit_dir_refuses_root(self) -> None:
        """Unit dir refuses to run as root."""
        with (
            patch("os.geteuid", return_value=0),
            pytest.raises(SystemExit, match="root"),
        ):
            CredentialProxyManager._systemd_unit_dir()

    def test_systemd_unit_dir_rejects_outside_home(self, tmp_path: Path) -> None:
        """Unit dir rejects XDG_CONFIG_HOME that resolves outside $HOME."""
        with (
            patch.dict(os.environ, {"XDG_CONFIG_HOME": "/etc/evil"}),
            patch("os.geteuid", return_value=1000),
            pytest.raises(SystemExit, match="outside the home directory"),
        ):
            CredentialProxyManager._systemd_unit_dir()

    def test_is_systemd_available_true(self) -> None:
        """Returns True when systemctl is-system-running exits 0."""
        result = MagicMock(returncode=0)
        with patch("subprocess.run", return_value=result):
            assert CredentialProxyManager().is_systemd_available() is True

    def test_is_systemd_available_degraded(self) -> None:
        """Returns True for degraded state (exit code 1)."""
        result = MagicMock(returncode=1)
        with patch("subprocess.run", return_value=result):
            assert CredentialProxyManager().is_systemd_available() is True

    def test_is_systemd_available_missing(self) -> None:
        """Returns False when systemctl is not found."""
        with patch("subprocess.run", side_effect=FileNotFoundError):
            assert CredentialProxyManager().is_systemd_available() is False

    def test_is_socket_installed_true(self, tmp_path: Path) -> None:
        """Returns True when socket unit file exists."""
        with patch.object(CredentialProxyManager, "_systemd_unit_dir", return_value=tmp_path):
            (tmp_path / "terok-credential-proxy.socket").write_text("[Socket]")
            assert CredentialProxyManager().is_socket_installed() is True

    def test_is_socket_installed_false(self, tmp_path: Path) -> None:
        """Returns False when socket unit file is absent."""
        with patch.object(CredentialProxyManager, "_systemd_unit_dir", return_value=tmp_path):
            assert CredentialProxyManager().is_socket_installed() is False

    def test_is_socket_active_true(self) -> None:
        """Returns True when systemctl reports active."""
        result = MagicMock(stdout="active\n")
        with patch("subprocess.run", return_value=result):
            assert CredentialProxyManager().is_socket_active() is True

    def test_is_socket_active_false(self) -> None:
        """Returns False when systemctl reports inactive."""
        result = MagicMock(stdout="inactive\n")
        with patch("subprocess.run", return_value=result):
            assert CredentialProxyManager().is_socket_active() is False

    def test_is_socket_active_no_systemctl(self) -> None:
        """Returns False when systemctl is not found."""
        with patch("subprocess.run", side_effect=FileNotFoundError):
            assert CredentialProxyManager().is_socket_active() is False

    def test_is_service_active_true(self) -> None:
        """Returns True when the service unit is active."""
        result = MagicMock(stdout="active\n")
        with patch("subprocess.run", return_value=result):
            assert CredentialProxyManager().is_service_active() is True

    def test_is_service_active_false(self) -> None:
        """Returns False when the service unit is inactive."""
        result = MagicMock(stdout="inactive\n")
        with patch("subprocess.run", return_value=result):
            assert CredentialProxyManager().is_service_active() is False

    def test_is_service_active_no_systemctl(self) -> None:
        """Returns False when systemctl is not found."""
        with patch("subprocess.run", side_effect=FileNotFoundError):
            assert CredentialProxyManager().is_service_active() is False

    def test_is_service_active_timeout(self) -> None:
        """Returns False on systemctl timeout."""
        with patch("subprocess.run", side_effect=subprocess.TimeoutExpired("cmd", 5)):
            assert CredentialProxyManager().is_service_active() is False


class TestGetProxyStatusModes:
    """Verify mode detection and health probing in get_proxy_status."""

    def test_systemd_mode_service_active(self, tmp_path: Path) -> None:
        """Reports running=True only when the service unit is active."""
        mgr = _make_mgr(tmp_path)
        with (
            patch.object(CredentialProxyManager, "is_socket_installed", return_value=True),
            patch.object(CredentialProxyManager, "is_service_active", return_value=True),
            patch.object(CredentialProxyManager, "_probe", return_value=True),
        ):
            status = mgr.get_status()
        assert status.mode == "systemd"
        assert status.running is True
        assert status.healthy is True

    def test_systemd_mode_service_idle(self, tmp_path: Path) -> None:
        """Reports running=False when socket is installed but service is idle."""
        mgr = _make_mgr(tmp_path)
        with (
            patch.object(CredentialProxyManager, "is_socket_installed", return_value=True),
            patch.object(CredentialProxyManager, "is_service_active", return_value=False),
        ):
            status = mgr.get_status()
        assert status.mode == "systemd"
        assert status.running is False
        assert status.healthy is False

    def test_daemon_mode_healthy(self, tmp_path: Path) -> None:
        """Reports mode='daemon' and healthy=True when health probe succeeds."""
        mgr = _make_mgr(tmp_path)
        with (
            patch.object(CredentialProxyManager, "is_socket_installed", return_value=False),
            patch.object(CredentialProxyManager, "is_daemon_running", return_value=True),
            patch.object(CredentialProxyManager, "_probe", return_value=True),
        ):
            status = mgr.get_status()
        assert status.mode == "daemon"
        assert status.running is True
        assert status.healthy is True

    def test_daemon_mode_unhealthy(self, tmp_path: Path) -> None:
        """Reports healthy=False when daemon is alive but probe fails."""
        mgr = _make_mgr(tmp_path)
        with (
            patch.object(CredentialProxyManager, "is_socket_installed", return_value=False),
            patch.object(CredentialProxyManager, "is_daemon_running", return_value=True),
            patch.object(CredentialProxyManager, "_probe", return_value=False),
        ):
            status = mgr.get_status()
        assert status.mode == "daemon"
        assert status.running is True
        assert status.healthy is False

    def test_none_mode_when_nothing_running(self, tmp_path: Path) -> None:
        """Reports mode='none' and healthy=False when nothing is active."""
        mgr = _make_mgr(tmp_path)
        with (
            _no_systemd(),
            patch.object(CredentialProxyManager, "is_daemon_running", return_value=False),
        ):
            status = mgr.get_status()
        assert status.mode == "none"
        assert status.running is False
        assert status.healthy is False

    def test_systemd_mode_service_active_but_unhealthy(self, tmp_path: Path) -> None:
        """Reports running=True but healthy=False when service is up but probe fails."""
        mgr = _make_mgr(tmp_path)
        with (
            patch.object(CredentialProxyManager, "is_socket_installed", return_value=True),
            patch.object(CredentialProxyManager, "is_service_active", return_value=True),
            patch.object(CredentialProxyManager, "_probe", return_value=False),
        ):
            status = mgr.get_status()
        assert status.mode == "systemd"
        assert status.running is True
        assert status.healthy is False

    def test_systemd_mode_falls_back_to_daemon(self, tmp_path: Path) -> None:
        """When socket installed but service idle, daemon running is not consulted."""
        mgr = _make_mgr(tmp_path)
        with (
            patch.object(CredentialProxyManager, "is_socket_installed", return_value=True),
            patch.object(CredentialProxyManager, "is_service_active", return_value=False),
            patch.object(CredentialProxyManager, "is_daemon_running", return_value=True),
        ):
            status = mgr.get_status()
        # Systemd takes precedence — daemon state is irrelevant
        assert status.mode == "systemd"
        assert status.running is False


class TestProbeProxy:
    """Verify CredentialProxyManager._probe single-shot health check."""

    def _mock_conn(self, *, status: int = 200) -> MagicMock:
        """Return a mock HTTPConnection whose getresponse() returns *status*."""
        mock_resp = MagicMock()
        mock_resp.status = status
        mock_resp.read.return_value = b""
        conn = MagicMock()
        conn.getresponse.return_value = mock_resp
        return conn

    def test_returns_true_on_200(self) -> None:
        """Returns True when health endpoint responds 200."""
        conn = self._mock_conn(status=200)
        with patch("http.client.HTTPConnection", return_value=conn):
            assert CredentialProxyManager._probe(18731) is True

    def test_returns_false_on_connection_refused(self) -> None:
        """Returns False when the proxy is not listening."""
        conn = MagicMock()
        conn.request.side_effect = ConnectionRefusedError
        with patch("http.client.HTTPConnection", return_value=conn):
            assert CredentialProxyManager._probe(18731) is False

    def test_returns_false_on_timeout(self) -> None:
        """Returns False when the request times out."""
        conn = MagicMock()
        conn.request.side_effect = OSError("timed out")
        with patch("http.client.HTTPConnection", return_value=conn):
            assert CredentialProxyManager._probe(18731) is False


class TestWaitForReady:
    """Verify _wait_for_ready polling loop."""

    def test_returns_true_on_immediate_success(self) -> None:
        """Returns True when the first probe succeeds."""
        with patch.object(CredentialProxyManager, "_probe", return_value=True):
            assert CredentialProxyManager._wait_for_ready(18731, timeout=1.0) is True

    def test_returns_false_on_timeout(self) -> None:
        """Returns False when all probes fail within the timeout."""
        with (
            patch.object(CredentialProxyManager, "_probe", return_value=False),
            patch("time.sleep"),
            patch("time.monotonic", side_effect=[0.0, 0.0, 0.2, 0.4, 6.0]),
        ):
            assert CredentialProxyManager._wait_for_ready(18731, timeout=5.0) is False

    def test_retries_then_succeeds(self) -> None:
        """Returns True after a few failed probes followed by success."""
        with (
            patch.object(CredentialProxyManager, "_probe", side_effect=[False, False, True]),
            patch("time.sleep"),
            patch("time.monotonic", side_effect=[0.0, 0.0, 0.2, 0.4, 0.6]),
        ):
            assert CredentialProxyManager._wait_for_ready(18731, timeout=5.0) is True


class TestInstallSystemdUnits:
    """Verify install_systemd_units."""

    def test_install_creates_units_and_enables_socket(self, tmp_path: Path) -> None:
        """install_systemd_units renders templates and runs systemctl enable."""
        cfg = _make_cfg(tmp_path)
        mgr = CredentialProxyManager(cfg)
        unit_dir = tmp_path / "systemd-units"
        with (
            patch.object(CredentialProxyManager, "_systemd_unit_dir", return_value=unit_dir),
            patch("subprocess.run") as mock_run,
        ):
            mgr.install_systemd_units()
        # Verify unit files were created
        assert (unit_dir / "terok-credential-proxy.socket").is_file()
        assert (unit_dir / "terok-credential-proxy.service").is_file()
        # Service template should reference python -m, not a standalone binary
        svc = (unit_dir / "terok-credential-proxy.service").read_text()
        assert "-m terok_sandbox.credentials.proxy" in svc
        # Verify systemctl was called
        calls = [c.args[0] for c in mock_run.call_args_list]
        assert ["systemctl", "--user", "daemon-reload"] in calls
        assert any("enable" in c and "--now" in c for c in calls)
        assert any("restart" in c for c in calls)

    def test_socket_unit_has_both_listen_streams(self, tmp_path: Path) -> None:
        """Socket unit declares both Unix socket and TCP port ListenStream entries."""
        cfg = _make_cfg(tmp_path)
        mgr = CredentialProxyManager(cfg)
        unit_dir = tmp_path / "systemd-units"
        with (
            patch.object(CredentialProxyManager, "_systemd_unit_dir", return_value=unit_dir),
            patch("subprocess.run"),
        ):
            mgr.install_systemd_units()
        socket_unit = (unit_dir / "terok-credential-proxy.socket").read_text()
        listen_lines = [
            line.strip() for line in socket_unit.splitlines() if line.startswith("ListenStream")
        ]
        assert len(listen_lines) == 2
        assert any(str(cfg.proxy_socket_path) in entry for entry in listen_lines)
        assert any(f"127.0.0.1:{cfg.proxy_port}" in entry for entry in listen_lines)


class TestUninstallSystemdUnits:
    """Verify uninstall_systemd_units."""

    def test_uninstall_removes_units(self, tmp_path: Path) -> None:
        """uninstall_systemd_units removes unit files and reloads."""
        unit_dir = tmp_path / "systemd-units"
        unit_dir.mkdir()
        (unit_dir / "terok-credential-proxy.socket").write_text("[Socket]")
        (unit_dir / "terok-credential-proxy.service").write_text("[Service]")
        with (
            patch.object(CredentialProxyManager, "_systemd_unit_dir", return_value=unit_dir),
            patch("subprocess.run") as mock_run,
        ):
            CredentialProxyManager().uninstall_systemd_units()
        assert not (unit_dir / "terok-credential-proxy.socket").exists()
        assert not (unit_dir / "terok-credential-proxy.service").exists()
        # Verify daemon-reload was called
        calls = [c.args[0] for c in mock_run.call_args_list]
        assert ["systemctl", "--user", "daemon-reload"] in calls
