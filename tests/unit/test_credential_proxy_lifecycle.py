# SPDX-FileCopyrightText: 2026 Jiri Vyskocil
# SPDX-License-Identifier: Apache-2.0

"""Tests for credential proxy lifecycle management."""

from __future__ import annotations

import os
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from terok_sandbox.config import SandboxConfig
from terok_sandbox.credential_proxy_lifecycle import (
    CredentialProxyStatus,
    _is_managed_proxy,
    _pid_file,
    _systemd_unit_dir,
    ensure_proxy_reachable,
    get_proxy_status,
    install_systemd_units,
    is_daemon_running,
    is_socket_active,
    is_socket_installed,
    is_systemd_available,
    start_daemon,
    stop_daemon,
    uninstall_systemd_units,
)


def _make_cfg(tmp_path: Path) -> SandboxConfig:
    """Create a SandboxConfig rooted in tmp_path."""
    return SandboxConfig(
        state_dir=tmp_path / "state",
        runtime_dir=tmp_path / "run",
    )


class TestPidFile:
    """Verify PID file path resolution."""

    def test_pid_file_uses_config(self, tmp_path: Path) -> None:
        """PID file path comes from config's proxy_pid_file_path."""
        cfg = _make_cfg(tmp_path)
        assert _pid_file(cfg) == cfg.proxy_pid_file_path

    def test_pid_file_default_config(self) -> None:
        """PID file resolves without explicit config."""
        path = _pid_file()
        assert "credential-proxy.pid" in str(path)


class TestIsManagedProxy:
    """Verify cmdline-based PID validation."""

    def test_no_proc_entry(self, tmp_path: Path) -> None:
        """Non-existent PID returns False."""
        assert _is_managed_proxy(999999999, _make_cfg(tmp_path)) is False

    def test_current_process_is_not_proxy(self, tmp_path: Path) -> None:
        """Current process (pytest) is not the proxy."""
        assert _is_managed_proxy(os.getpid(), _make_cfg(tmp_path)) is False

    def test_matches_when_flag_present(self, tmp_path: Path) -> None:
        """Returns True when /proc/PID/cmdline contains --pid-file=<expected>."""
        cfg = _make_cfg(tmp_path)
        expected_flag = f"--pid-file={_pid_file(cfg)}"
        fake_cmdline = f"terok-credential-proxy\x00{expected_flag}\x00".encode()

        with (
            patch("pathlib.Path.is_file", return_value=True),
            patch("pathlib.Path.read_bytes", return_value=fake_cmdline),
        ):
            assert _is_managed_proxy(12345, cfg) is True

    def test_rejects_different_pid_file(self, tmp_path: Path) -> None:
        """Returns False when --pid-file points to a different path."""
        cfg = _make_cfg(tmp_path)
        fake_cmdline = b"terok-credential-proxy\x00--pid-file=/other/path.pid\x00"

        with (
            patch("pathlib.Path.is_file", return_value=True),
            patch("pathlib.Path.read_bytes", return_value=fake_cmdline),
        ):
            assert _is_managed_proxy(12345, cfg) is False


class TestStartDaemon:
    """Verify start_daemon behaviour."""

    def test_missing_routes_file_creates_empty(self, tmp_path: Path) -> None:
        """start_daemon creates an empty routes file when missing."""
        cfg = _make_cfg(tmp_path)

        mock_proc = MagicMock()
        mock_proc.poll.return_value = None  # still running

        with patch("subprocess.Popen", return_value=mock_proc), patch("time.sleep"):
            start_daemon(cfg)

        assert cfg.proxy_routes_path.is_file()
        assert cfg.proxy_routes_path.read_text() == "{}\n"

    def test_start_launches_subprocess(self, tmp_path: Path) -> None:
        """start_daemon calls Popen with the correct command."""
        cfg = _make_cfg(tmp_path)
        # Create routes file
        routes = cfg.proxy_routes_path
        routes.parent.mkdir(parents=True, exist_ok=True)
        routes.write_text("{}")

        mock_proc = MagicMock()
        mock_proc.poll.return_value = None  # still running

        with patch("subprocess.Popen", return_value=mock_proc) as mock_popen, patch("time.sleep"):
            start_daemon(cfg)

        import sys

        cmd = mock_popen.call_args[0][0]
        assert cmd[0] == sys.executable
        assert cmd[1:3] == ["-m", "terok_sandbox.credential_proxy"]
        assert any("--socket-path=" in a for a in cmd)
        assert any("--pid-file=" in a for a in cmd)

    def test_immediate_exit_raises(self, tmp_path: Path) -> None:
        """start_daemon raises SystemExit if the process dies immediately."""
        cfg = _make_cfg(tmp_path)
        routes = cfg.proxy_routes_path
        routes.parent.mkdir(parents=True, exist_ok=True)
        routes.write_text("{}")

        mock_proc = MagicMock()
        mock_proc.poll.return_value = 1  # exited immediately
        mock_proc.stderr.read.return_value = b"error: bad config"

        with (
            patch("subprocess.Popen", return_value=mock_proc),
            patch("time.sleep"),
            pytest.raises(SystemExit, match="failed to start"),
        ):
            start_daemon(cfg)


class TestStopDaemon:
    """Verify stop_daemon behaviour."""

    def test_no_pidfile_is_noop(self, tmp_path: Path) -> None:
        """stop_daemon does nothing when PID file doesn't exist."""
        cfg = _make_cfg(tmp_path)
        stop_daemon(cfg)  # should not raise

    def test_sends_sigterm_to_managed_process(self, tmp_path: Path) -> None:
        """stop_daemon sends SIGTERM when the PID is a managed proxy."""
        cfg = _make_cfg(tmp_path)
        pidfile = _pid_file(cfg)
        pidfile.parent.mkdir(parents=True, exist_ok=True)
        pidfile.write_text("42")

        with (
            patch("terok_sandbox.credential_proxy_lifecycle._is_managed_proxy", return_value=True),
            patch("os.kill") as mock_kill,
        ):
            stop_daemon(cfg)

        mock_kill.assert_called_once_with(42, 15)  # SIGTERM = 15
        assert not pidfile.exists()

    def test_stale_pid_cleans_up(self, tmp_path: Path) -> None:
        """stop_daemon cleans PID file even when process is gone."""
        cfg = _make_cfg(tmp_path)
        pidfile = _pid_file(cfg)
        pidfile.parent.mkdir(parents=True, exist_ok=True)
        pidfile.write_text("99999999")

        with (
            patch("terok_sandbox.credential_proxy_lifecycle._is_managed_proxy", return_value=True),
            patch("os.kill", side_effect=ProcessLookupError),
        ):
            stop_daemon(cfg)

        assert not pidfile.exists()


class TestIsDaemonRunning:
    """Verify is_daemon_running behaviour."""

    def test_no_pidfile(self, tmp_path: Path) -> None:
        """Returns False when PID file doesn't exist."""
        assert is_daemon_running(_make_cfg(tmp_path)) is False

    def test_valid_managed_pid(self, tmp_path: Path) -> None:
        """Returns True for a valid managed PID."""
        cfg = _make_cfg(tmp_path)
        pidfile = _pid_file(cfg)
        pidfile.parent.mkdir(parents=True, exist_ok=True)
        pidfile.write_text("42")

        with (
            patch("terok_sandbox.credential_proxy_lifecycle._is_managed_proxy", return_value=True),
            patch("os.kill"),
        ):  # signal 0 succeeds
            assert is_daemon_running(cfg) is True

    def test_not_our_daemon(self, tmp_path: Path) -> None:
        """Returns False when PID doesn't match our cmdline."""
        cfg = _make_cfg(tmp_path)
        pidfile = _pid_file(cfg)
        pidfile.parent.mkdir(parents=True, exist_ok=True)
        pidfile.write_text("42")

        with patch(
            "terok_sandbox.credential_proxy_lifecycle._is_managed_proxy", return_value=False
        ):
            assert is_daemon_running(cfg) is False

    def test_stale_pid(self, tmp_path: Path) -> None:
        """Returns False when PID is gone."""
        cfg = _make_cfg(tmp_path)
        pidfile = _pid_file(cfg)
        pidfile.parent.mkdir(parents=True, exist_ok=True)
        pidfile.write_text("99999999")

        with (
            patch("terok_sandbox.credential_proxy_lifecycle._is_managed_proxy", return_value=True),
            patch("os.kill", side_effect=ProcessLookupError),
        ):
            assert is_daemon_running(cfg) is False


_LIFECYCLE = "terok_sandbox.credential_proxy_lifecycle"


def _no_systemd():
    """Context manager that patches out systemd detection."""
    return patch(f"{_LIFECYCLE}.is_socket_installed", return_value=False)


class TestGetProxyStatus:
    """Verify get_proxy_status."""

    def test_returns_status_dataclass(self, tmp_path: Path) -> None:
        """Returns a CredentialProxyStatus with correct fields."""
        cfg = _make_cfg(tmp_path)
        with (
            _no_systemd(),
            patch(f"{_LIFECYCLE}.is_daemon_running", return_value=False),
        ):
            status = get_proxy_status(cfg)

        assert isinstance(status, CredentialProxyStatus)
        assert status.mode == "none"
        assert status.running is False
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

        with (
            _no_systemd(),
            patch(f"{_LIFECYCLE}.is_daemon_running", return_value=False),
        ):
            status = get_proxy_status(cfg)

        assert status.routes_configured == 2

    def test_invalid_routes_json_yields_zero(self, tmp_path: Path) -> None:
        """Invalid JSON in routes.json yields routes_configured=0."""
        cfg = _make_cfg(tmp_path)
        routes = cfg.proxy_routes_path
        routes.parent.mkdir(parents=True, exist_ok=True)
        routes.write_text("not valid json{{{")

        with (
            _no_systemd(),
            patch(f"{_LIFECYCLE}.is_daemon_running", return_value=False),
        ):
            status = get_proxy_status(cfg)

        assert status.routes_configured == 0

    def test_lists_stored_credentials(self, tmp_path: Path) -> None:
        """credentials_stored lists providers from the credential DB."""
        from terok_sandbox.credential_db import CredentialDB

        cfg = _make_cfg(tmp_path)
        db = CredentialDB(cfg.proxy_db_path)
        db.store_credential("default", "github", {"token": "abc"})
        db.store_credential("default", "anthropic", {"key": "xyz"})
        db.close()

        with (
            _no_systemd(),
            patch(f"{_LIFECYCLE}.is_daemon_running", return_value=False),
        ):
            status = get_proxy_status(cfg)

        assert set(status.credentials_stored) == {"github", "anthropic"}

    def test_no_db_yields_empty_credentials(self, tmp_path: Path) -> None:
        """credentials_stored is empty when the DB file doesn't exist."""
        cfg = _make_cfg(tmp_path)
        assert not cfg.proxy_db_path.is_file()

        with (
            _no_systemd(),
            patch(f"{_LIFECYCLE}.is_daemon_running", return_value=False),
        ):
            status = get_proxy_status(cfg)

        assert status.credentials_stored == ()


class TestEnsureProxyReachable:
    """Verify ensure_proxy_reachable."""

    def test_passes_when_daemon_running(self, tmp_path: Path) -> None:
        """No exception when daemon is running and TCP port is up."""
        cfg = _make_cfg(tmp_path)
        with (
            patch(f"{_LIFECYCLE}.is_socket_active", return_value=False),
            patch(f"{_LIFECYCLE}.is_daemon_running", return_value=True),
            patch(f"{_LIFECYCLE}._wait_for_tcp_port", return_value=True),
        ):
            ensure_proxy_reachable(cfg)  # should not raise

    def test_raises_when_stopped(self, tmp_path: Path) -> None:
        """Raises SystemExit with actionable message when daemon is down."""
        cfg = _make_cfg(tmp_path)
        with (
            patch(f"{_LIFECYCLE}.is_socket_active", return_value=False),
            patch(f"{_LIFECYCLE}.is_daemon_running", return_value=False),
            pytest.raises(SystemExit, match="not running"),
        ):
            ensure_proxy_reachable(cfg)

    def test_passes_when_socket_active(self, tmp_path: Path) -> None:
        """Socket active → starts service, waits for TCP port."""
        cfg = _make_cfg(tmp_path)
        with (
            patch(f"{_LIFECYCLE}.is_socket_active", return_value=True),
            patch(f"{_LIFECYCLE}.subprocess") as mock_sub,
            patch(f"{_LIFECYCLE}._wait_for_tcp_port", return_value=True),
        ):
            ensure_proxy_reachable(cfg)  # should not raise
            mock_sub.run.assert_called_once()  # systemctl --user start

    def test_raises_when_tcp_port_unreachable(self, tmp_path: Path) -> None:
        """Service started but TCP port never comes up → SystemExit."""
        cfg = _make_cfg(tmp_path)
        with (
            patch(f"{_LIFECYCLE}.is_socket_active", return_value=True),
            patch(f"{_LIFECYCLE}.subprocess"),
            patch(f"{_LIFECYCLE}._wait_for_tcp_port", return_value=False),
            pytest.raises(SystemExit, match="not reachable"),
        ):
            ensure_proxy_reachable(cfg)


class TestSystemdHelpers:
    """Verify systemd detection and socket status helpers."""

    def test_systemd_unit_dir_default(self) -> None:
        """Unit dir falls back to ~/.config/systemd/user when XDG_CONFIG_HOME is unset."""
        env = {k: v for k, v in os.environ.items() if k != "XDG_CONFIG_HOME"}
        with patch.dict(os.environ, env, clear=True):
            d = _systemd_unit_dir()
        assert d == Path.home() / ".config" / "systemd" / "user"

    def test_systemd_unit_dir_xdg(self, tmp_path: Path) -> None:
        """Unit dir respects XDG_CONFIG_HOME."""
        with patch.dict(os.environ, {"XDG_CONFIG_HOME": str(tmp_path)}):
            d = _systemd_unit_dir()
        assert d == tmp_path / "systemd" / "user"

    def test_is_systemd_available_true(self) -> None:
        """Returns True when systemctl is-system-running exits 0."""
        result = MagicMock(returncode=0)
        with patch("subprocess.run", return_value=result):
            assert is_systemd_available() is True

    def test_is_systemd_available_degraded(self) -> None:
        """Returns True for degraded state (exit code 1)."""
        result = MagicMock(returncode=1)
        with patch("subprocess.run", return_value=result):
            assert is_systemd_available() is True

    def test_is_systemd_available_missing(self) -> None:
        """Returns False when systemctl is not found."""
        with patch("subprocess.run", side_effect=FileNotFoundError):
            assert is_systemd_available() is False

    def test_is_socket_installed_true(self, tmp_path: Path) -> None:
        """Returns True when socket unit file exists."""
        with patch(f"{_LIFECYCLE}._systemd_unit_dir", return_value=tmp_path):
            (tmp_path / "terok-credential-proxy.socket").write_text("[Socket]")
            assert is_socket_installed() is True

    def test_is_socket_installed_false(self, tmp_path: Path) -> None:
        """Returns False when socket unit file is absent."""
        with patch(f"{_LIFECYCLE}._systemd_unit_dir", return_value=tmp_path):
            assert is_socket_installed() is False

    def test_is_socket_active_true(self) -> None:
        """Returns True when systemctl reports active."""
        result = MagicMock(stdout="active\n")
        with patch("subprocess.run", return_value=result):
            assert is_socket_active() is True

    def test_is_socket_active_false(self) -> None:
        """Returns False when systemctl reports inactive."""
        result = MagicMock(stdout="inactive\n")
        with patch("subprocess.run", return_value=result):
            assert is_socket_active() is False

    def test_is_socket_active_no_systemctl(self) -> None:
        """Returns False when systemctl is not found."""
        with patch("subprocess.run", side_effect=FileNotFoundError):
            assert is_socket_active() is False


class TestGetProxyStatusModes:
    """Verify mode detection in get_proxy_status."""

    def test_systemd_mode_when_socket_installed(self, tmp_path: Path) -> None:
        """Reports mode='systemd' when socket unit is installed."""
        cfg = _make_cfg(tmp_path)
        with (
            patch(f"{_LIFECYCLE}.is_socket_installed", return_value=True),
            patch(f"{_LIFECYCLE}.is_socket_active", return_value=True),
        ):
            status = get_proxy_status(cfg)
        assert status.mode == "systemd"
        assert status.running is True

    def test_daemon_mode_when_pid_running(self, tmp_path: Path) -> None:
        """Reports mode='daemon' when PID file daemon is alive."""
        cfg = _make_cfg(tmp_path)
        with (
            patch(f"{_LIFECYCLE}.is_socket_installed", return_value=False),
            patch(f"{_LIFECYCLE}.is_daemon_running", return_value=True),
        ):
            status = get_proxy_status(cfg)
        assert status.mode == "daemon"
        assert status.running is True

    def test_none_mode_when_nothing_running(self, tmp_path: Path) -> None:
        """Reports mode='none' when neither systemd nor daemon is active."""
        cfg = _make_cfg(tmp_path)
        with (
            _no_systemd(),
            patch(f"{_LIFECYCLE}.is_daemon_running", return_value=False),
        ):
            status = get_proxy_status(cfg)
        assert status.mode == "none"
        assert status.running is False

    def test_systemd_mode_inactive_socket(self, tmp_path: Path) -> None:
        """Reports mode='systemd' and running=False when socket is installed but inactive."""
        cfg = _make_cfg(tmp_path)
        with (
            patch(f"{_LIFECYCLE}.is_socket_installed", return_value=True),
            patch(f"{_LIFECYCLE}.is_socket_active", return_value=False),
        ):
            status = get_proxy_status(cfg)
        assert status.mode == "systemd"
        assert status.running is False


class TestInstallSystemdUnits:
    """Verify install_systemd_units."""

    def test_install_creates_units_and_enables_socket(self, tmp_path: Path) -> None:
        """install_systemd_units renders templates and runs systemctl enable."""
        cfg = _make_cfg(tmp_path)
        unit_dir = tmp_path / "systemd-units"
        with (
            patch(f"{_LIFECYCLE}._systemd_unit_dir", return_value=unit_dir),
            patch("subprocess.run") as mock_run,
        ):
            install_systemd_units(cfg)
        # Verify unit files were created
        assert (unit_dir / "terok-credential-proxy.socket").is_file()
        assert (unit_dir / "terok-credential-proxy.service").is_file()
        # Service template should reference python -m, not a standalone binary
        svc = (unit_dir / "terok-credential-proxy.service").read_text()
        assert "-m terok_sandbox.credential_proxy" in svc
        # Verify systemctl was called
        calls = [c.args[0] for c in mock_run.call_args_list]
        assert ["systemctl", "--user", "daemon-reload"] in calls
        assert any("enable" in c and "--now" in c for c in calls)
        assert any("restart" in c for c in calls)


class TestUninstallSystemdUnits:
    """Verify uninstall_systemd_units."""

    def test_uninstall_removes_units(self, tmp_path: Path) -> None:
        """uninstall_systemd_units removes unit files and reloads."""
        unit_dir = tmp_path / "systemd-units"
        unit_dir.mkdir()
        (unit_dir / "terok-credential-proxy.socket").write_text("[Socket]")
        (unit_dir / "terok-credential-proxy.service").write_text("[Service]")
        with (
            patch(f"{_LIFECYCLE}._systemd_unit_dir", return_value=unit_dir),
            patch("subprocess.run") as mock_run,
        ):
            uninstall_systemd_units()
        assert not (unit_dir / "terok-credential-proxy.socket").exists()
        assert not (unit_dir / "terok-credential-proxy.service").exists()
        # Verify daemon-reload was called
        calls = [c.args[0] for c in mock_run.call_args_list]
        assert ["systemctl", "--user", "daemon-reload"] in calls
