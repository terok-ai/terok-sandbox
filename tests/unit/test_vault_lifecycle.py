# SPDX-FileCopyrightText: 2026 Jiri Vyskocil
# SPDX-License-Identifier: Apache-2.0

"""Tests for vault lifecycle management."""

from __future__ import annotations

import os
import subprocess
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from terok_sandbox.config import SandboxConfig
from terok_sandbox.vault.lifecycle import (
    VaultManager,
    VaultStatus,
    VaultUnreachableError,
)


def _make_cfg(tmp_path: Path) -> SandboxConfig:
    """Create a SandboxConfig rooted in tmp_path."""
    return SandboxConfig(
        state_dir=tmp_path / "state",
        runtime_dir=tmp_path / "run",
        vault_dir=tmp_path / "vault",
    )


def _make_mgr(tmp_path: Path) -> VaultManager:
    """Create a VaultManager with a test config."""
    return VaultManager(_make_cfg(tmp_path))


class TestPidFile:
    """Verify PID file path resolution."""

    def test_pid_file_uses_config(self, tmp_path: Path) -> None:
        """PID file path comes from config vault_pid_path."""
        cfg = _make_cfg(tmp_path)
        mgr = VaultManager(cfg)
        assert mgr._cfg.vault_pid_path == cfg.vault_pid_path

    def test_pid_file_default_config(self) -> None:
        """PID file resolves without explicit config."""
        mgr = VaultManager()
        assert "vault.pid" in str(mgr._cfg.vault_pid_path)


class TestIsManagedVault:
    """Verify cmdline-based vault PID validation."""

    def test_no_proc_entry(self, tmp_path: Path) -> None:
        """Non-existent PID returns False."""
        assert _make_mgr(tmp_path)._is_managed_vault(999999999) is False

    def test_current_process_is_not_vault(self, tmp_path: Path) -> None:
        """Current process (pytest) is not the vault."""
        assert _make_mgr(tmp_path)._is_managed_vault(os.getpid()) is False

    def test_matches_when_flag_present(self, tmp_path: Path) -> None:
        """Returns True when /proc/PID/cmdline contains the expected --pid-file flag."""
        mgr = _make_mgr(tmp_path)
        expected_flag = f"--pid-file={mgr._cfg.vault_pid_path}"
        fake_cmdline = f"terok-vault\x00{expected_flag}\x00".encode()

        with (
            patch("pathlib.Path.is_file", return_value=True),
            patch("pathlib.Path.read_bytes", return_value=fake_cmdline),
        ):
            assert mgr._is_managed_vault(12345) is True

    def test_rejects_different_pid_file(self, tmp_path: Path) -> None:
        """Returns False when --pid-file points to a different path."""
        mgr = _make_mgr(tmp_path)
        fake_cmdline = b"terok-vault\x00--pid-file=/other/path.pid\x00"

        with (
            patch("pathlib.Path.is_file", return_value=True),
            patch("pathlib.Path.read_bytes", return_value=fake_cmdline),
        ):
            assert mgr._is_managed_vault(12345) is False


class TestStartDaemon:
    """Verify start_daemon behaviour."""

    def test_missing_routes_file_creates_empty_with_restricted_perms(self, tmp_path: Path) -> None:
        """start_daemon creates an empty routes file with 0o600 permissions."""
        cfg = _make_cfg(tmp_path)
        mgr = VaultManager(cfg)

        mock_proc = MagicMock()
        mock_proc.poll.return_value = None

        with (
            patch("subprocess.Popen", return_value=mock_proc),
            patch.object(VaultManager, "_wait_for_ready", return_value=True),
            patch.object(VaultManager, "_wait_for_tcp_port", return_value=True),
        ):
            mgr.start_daemon()

        assert cfg.routes_path.is_file()
        assert cfg.routes_path.read_text() == "{}\n"
        assert oct(cfg.routes_path.stat().st_mode & 0o777) == oct(0o600)

    def test_start_launches_subprocess(self, tmp_path: Path) -> None:
        """start_daemon calls Popen with the correct command."""
        cfg = _make_cfg(tmp_path)
        mgr = VaultManager(cfg)
        routes = cfg.routes_path
        routes.parent.mkdir(parents=True, exist_ok=True)
        routes.write_text("{}")

        mock_proc = MagicMock()
        mock_proc.poll.return_value = None

        with (
            patch("subprocess.Popen", return_value=mock_proc) as mock_popen,
            patch.object(VaultManager, "_wait_for_ready", return_value=True),
            patch.object(VaultManager, "_wait_for_tcp_port", return_value=True),
        ):
            mgr.start_daemon()

        import sys

        cmd = mock_popen.call_args[0][0]
        assert cmd[0] == sys.executable
        assert cmd[1:3] == ["-m", "terok_sandbox.vault"]
        assert any("--socket-path=" in a for a in cmd)
        assert any("--pid-file=" in a for a in cmd)
        assert "--log-level=INFO" in cmd

    def test_immediate_exit_raises(self, tmp_path: Path) -> None:
        """start_daemon raises SystemExit if the vault process dies during readiness wait."""
        cfg = _make_cfg(tmp_path)
        mgr = VaultManager(cfg)
        routes = cfg.routes_path
        routes.parent.mkdir(parents=True, exist_ok=True)
        routes.write_text("{}")

        mock_proc = MagicMock()
        mock_proc.poll.return_value = 1
        mock_proc.stderr.read.return_value = b"error: bad config"

        with (
            patch("subprocess.Popen", return_value=mock_proc),
            patch.object(VaultManager, "_wait_for_ready", return_value=False),
            pytest.raises(SystemExit, match="failed to start"),
        ):
            mgr.start_daemon()

    def test_debug_log_level_via_env(self, tmp_path: Path) -> None:
        """TEROK_VAULT_LOG_LEVEL env var overrides the default INFO log level."""
        cfg = _make_cfg(tmp_path)
        mgr = VaultManager(cfg)
        routes = cfg.routes_path
        routes.parent.mkdir(parents=True, exist_ok=True)
        routes.write_text("{}")

        mock_proc = MagicMock()
        mock_proc.poll.return_value = None

        with (
            patch("subprocess.Popen", return_value=mock_proc) as mock_popen,
            patch.object(VaultManager, "_wait_for_ready", return_value=True),
            patch.object(VaultManager, "_wait_for_tcp_port", return_value=True),
            patch.dict(os.environ, {"TEROK_VAULT_LOG_LEVEL": "DEBUG"}),
        ):
            mgr.start_daemon()

        cmd = mock_popen.call_args[0][0]
        assert "--log-level=DEBUG" in cmd

    def test_timeout_without_crash_raises(self, tmp_path: Path) -> None:
        """start_daemon raises SystemExit when vault stays alive but never becomes ready."""
        cfg = _make_cfg(tmp_path)
        mgr = VaultManager(cfg)
        routes = cfg.routes_path
        routes.parent.mkdir(parents=True, exist_ok=True)
        routes.write_text("{}")

        mock_proc = MagicMock()
        mock_proc.poll.return_value = None  # still running, just not ready

        with (
            patch("subprocess.Popen", return_value=mock_proc),
            patch.object(VaultManager, "_wait_for_ready", return_value=False),
            pytest.raises(SystemExit, match="did not become ready"),
        ):
            mgr.start_daemon()

    def test_socket_mode_omits_tcp_ports_and_adds_ssh_signer_socket(self, tmp_path: Path) -> None:
        """In socket mode, ``--port``/``--ssh-signer-port`` are dropped and ``--ssh-signer-socket-path`` is added."""
        with patch("terok_sandbox.config.services_mode", return_value="socket"):
            cfg = _make_cfg(tmp_path)
        mgr = VaultManager(cfg)
        assert cfg.token_broker_port is None and cfg.ssh_signer_port is None

        mock_proc = MagicMock()
        mock_proc.poll.return_value = None

        with (
            patch("subprocess.Popen", return_value=mock_proc) as mock_popen,
            patch.object(VaultManager, "_wait_for_unix_socket", return_value=True),
        ):
            mgr.start_daemon()

        cmd = mock_popen.call_args[0][0]
        assert not any(a.startswith("--port=") for a in cmd), cmd
        assert not any(a.startswith("--ssh-signer-port=") for a in cmd), cmd
        assert f"--ssh-signer-socket-path={cfg.ssh_signer_socket_path}" in cmd


class TestStopDaemon:
    """Verify stop_daemon behaviour."""

    def test_no_pidfile_is_noop(self, tmp_path: Path) -> None:
        """stop_daemon does nothing when PID file doesn't exist."""
        _make_mgr(tmp_path).stop_daemon()  # should not raise

    def test_sends_sigterm_to_managed_process(self, tmp_path: Path) -> None:
        """stop_daemon sends SIGTERM when the PID is a managed vault."""
        mgr = _make_mgr(tmp_path)
        pidfile = mgr._cfg.vault_pid_path
        pidfile.parent.mkdir(parents=True, exist_ok=True)
        pidfile.write_text("42")

        with (
            patch.object(VaultManager, "_is_managed_vault", return_value=True),
            patch("os.kill") as mock_kill,
        ):
            mgr.stop_daemon()

        mock_kill.assert_called_once_with(42, 15)  # SIGTERM = 15
        assert not pidfile.exists()

    def test_stale_pid_cleans_up(self, tmp_path: Path) -> None:
        """stop_daemon cleans PID file even when process is gone."""
        mgr = _make_mgr(tmp_path)
        pidfile = mgr._cfg.vault_pid_path
        pidfile.parent.mkdir(parents=True, exist_ok=True)
        pidfile.write_text("99999999")

        with (
            patch.object(VaultManager, "_is_managed_vault", return_value=True),
            patch("os.kill", side_effect=ProcessLookupError),
        ):
            mgr.stop_daemon()

        assert not pidfile.exists()

    def test_socket_mode_stops_systemd_unit(self, tmp_path: Path) -> None:
        """stop_daemon stops the socket-mode service when systemd-activated (no PID file)."""
        mgr = _make_mgr(tmp_path)
        assert not mgr._cfg.vault_pid_path.exists()

        def _active(_self: VaultManager, unit: str) -> bool:
            return unit == "terok-vault-socket.service"

        with (
            patch.object(VaultManager, "_is_unit_active", _active),
            patch("subprocess.run") as mock_run,
        ):
            mgr.stop_daemon()

        calls = [c for c in mock_run.call_args_list if "stop" in c.args[0]]
        assert len(calls) == 1
        assert calls[0].args[0][:3] == ["systemctl", "--user", "stop"]
        assert "terok-vault-socket.service" in calls[0].args[0]

    def test_wedged_systemctl_does_not_block_pidfile_cleanup(self, tmp_path: Path) -> None:
        """A hung ``systemctl stop`` must not skip the PID-file SIGTERM path — panic's whole point is belt-and-suspenders."""
        mgr = _make_mgr(tmp_path)
        pidfile = mgr._cfg.vault_pid_path
        pidfile.parent.mkdir(parents=True, exist_ok=True)
        pidfile.write_text("42")

        with (
            patch.object(VaultManager, "_is_unit_active", return_value=True),
            patch.object(VaultManager, "_is_managed_vault", return_value=True),
            patch(
                "subprocess.run",
                side_effect=subprocess.TimeoutExpired(cmd="systemctl", timeout=10),
            ),
            patch("os.kill") as mock_kill,
        ):
            mgr.stop_daemon()

        mock_kill.assert_called_once_with(42, 15)
        assert not pidfile.exists()


class TestIsDaemonRunning:
    """Verify is_daemon_running behaviour."""

    def test_no_pidfile(self, tmp_path: Path) -> None:
        """Returns False when PID file doesn't exist."""
        assert _make_mgr(tmp_path).is_daemon_running() is False

    def test_valid_managed_pid(self, tmp_path: Path) -> None:
        """Returns True for a valid managed PID."""
        mgr = _make_mgr(tmp_path)
        pidfile = mgr._cfg.vault_pid_path
        pidfile.parent.mkdir(parents=True, exist_ok=True)
        pidfile.write_text("42")

        with (
            patch.object(VaultManager, "_is_managed_vault", return_value=True),
            patch("os.kill"),
        ):  # signal 0 succeeds
            assert mgr.is_daemon_running() is True

    def test_not_our_daemon(self, tmp_path: Path) -> None:
        """Returns False when PID doesn't match our cmdline."""
        mgr = _make_mgr(tmp_path)
        pidfile = mgr._cfg.vault_pid_path
        pidfile.parent.mkdir(parents=True, exist_ok=True)
        pidfile.write_text("42")

        with patch.object(VaultManager, "_is_managed_vault", return_value=False):
            assert mgr.is_daemon_running() is False

    def test_stale_pid(self, tmp_path: Path) -> None:
        """Returns False when PID is gone."""
        mgr = _make_mgr(tmp_path)
        pidfile = mgr._cfg.vault_pid_path
        pidfile.parent.mkdir(parents=True, exist_ok=True)
        pidfile.write_text("99999999")

        with (
            patch.object(VaultManager, "_is_managed_vault", return_value=True),
            patch("os.kill", side_effect=ProcessLookupError),
        ):
            assert mgr.is_daemon_running() is False


_LIFECYCLE = "terok_sandbox.vault.lifecycle"


def _no_systemd():
    """Return a context manager that patches out systemd detection."""
    return patch.object(VaultManager, "is_socket_installed", return_value=False)


class TestGetVaultStatus:
    """Verify get_vault_status."""

    def test_returns_status_dataclass(self, tmp_path: Path) -> None:
        """Returns a VaultStatus dataclass with correct fields."""
        cfg = _make_cfg(tmp_path)
        mgr = VaultManager(cfg)
        with (
            _no_systemd(),
            patch.object(VaultManager, "is_daemon_running", return_value=False),
        ):
            status = mgr.get_status()

        assert isinstance(status, VaultStatus)
        assert status.mode == "none"
        assert status.running is False
        assert status.healthy is False
        assert status.socket_path == cfg.vault_socket_path
        assert status.db_path == cfg.db_path
        assert status.routes_path == cfg.routes_path
        assert status.routes_configured == 0
        assert status.credentials_stored == ()

    def test_counts_routes_from_json(self, tmp_path: Path) -> None:
        """Routes count reflects the number of entries in routes.json."""
        cfg = _make_cfg(tmp_path)
        routes = cfg.routes_path
        routes.parent.mkdir(parents=True, exist_ok=True)
        routes.write_text('{"github": {}, "gitlab": {}}')

        mgr = VaultManager(cfg)
        with (
            _no_systemd(),
            patch.object(VaultManager, "is_daemon_running", return_value=False),
        ):
            status = mgr.get_status()

        assert status.routes_configured == 2

    def test_invalid_routes_json_yields_zero(self, tmp_path: Path) -> None:
        """Invalid JSON in routes.json yields routes_configured=0."""
        cfg = _make_cfg(tmp_path)
        routes = cfg.routes_path
        routes.parent.mkdir(parents=True, exist_ok=True)
        routes.write_text("not valid json{{{")

        mgr = VaultManager(cfg)
        with (
            _no_systemd(),
            patch.object(VaultManager, "is_daemon_running", return_value=False),
        ):
            status = mgr.get_status()

        assert status.routes_configured == 0

    def test_lists_stored_credentials(self, tmp_path: Path) -> None:
        """credentials_stored lists providers from the credential DB."""
        from terok_sandbox.credentials.db import CredentialDB

        cfg = _make_cfg(tmp_path)
        db = CredentialDB(cfg.db_path)
        db.store_credential("default", "github", {"token": "abc"})
        db.store_credential("default", "anthropic", {"key": "xyz"})
        db.close()

        mgr = VaultManager(cfg)
        with (
            _no_systemd(),
            patch.object(VaultManager, "is_daemon_running", return_value=False),
        ):
            status = mgr.get_status()

        assert set(status.credentials_stored) == {"github", "anthropic"}

    def test_no_db_yields_empty_credentials(self, tmp_path: Path) -> None:
        """credentials_stored is empty when the DB file doesn't exist."""
        cfg = _make_cfg(tmp_path)
        assert not cfg.db_path.is_file()

        mgr = VaultManager(cfg)
        with (
            _no_systemd(),
            patch.object(VaultManager, "is_daemon_running", return_value=False),
        ):
            status = mgr.get_status()

        assert status.credentials_stored == ()


class TestEnsureVaultReachable:
    """Verify ensure_vault_reachable."""

    def test_passes_when_daemon_healthy(self, tmp_path: Path) -> None:
        """No exception when vault daemon is running and health probe succeeds."""
        mgr = _make_mgr(tmp_path)
        with (
            patch.object(VaultManager, "is_socket_active", return_value=False),
            patch.object(VaultManager, "is_daemon_running", return_value=True),
            patch.object(VaultManager, "_wait_for_ready", return_value=True),
            patch.object(VaultManager, "_wait_for_tcp_port", return_value=True),
        ):
            mgr.ensure_reachable()  # should not raise

    def test_raises_when_daemon_running_but_unhealthy(self, tmp_path: Path) -> None:
        """Raises SystemExit when vault daemon is alive but health probe fails."""
        mgr = _make_mgr(tmp_path)
        with (
            patch.object(VaultManager, "is_socket_active", return_value=False),
            patch.object(VaultManager, "is_daemon_running", return_value=True),
            patch.object(VaultManager, "_wait_for_ready", return_value=False),
            pytest.raises(SystemExit, match="not reachable"),
        ):
            mgr.ensure_reachable()

    def test_raises_when_stopped(self, tmp_path: Path) -> None:
        """Raises VaultUnreachableError when vault daemon is down."""
        mgr = _make_mgr(tmp_path)
        with (
            patch.object(VaultManager, "is_socket_active", return_value=False),
            patch.object(VaultManager, "is_daemon_running", return_value=False),
            pytest.raises(VaultUnreachableError, match="not reachable"),
        ):
            mgr.ensure_reachable()

    def test_passes_when_socket_active(self, tmp_path: Path) -> None:
        """Socket active -- starts service, waits for health + SSH signer port."""
        mgr = _make_mgr(tmp_path)
        with (
            patch.object(VaultManager, "is_socket_active", return_value=True),
            patch(f"{_LIFECYCLE}.subprocess") as mock_sub,
            patch.object(VaultManager, "_wait_for_ready", return_value=True),
            patch.object(VaultManager, "_wait_for_tcp_port", return_value=True),
        ):
            mgr.ensure_reachable()  # should not raise
            mock_sub.run.assert_called_once()  # systemctl --user start

    def test_raises_when_health_unreachable(self, tmp_path: Path) -> None:
        """Service started but health endpoint never responds -- raises SystemExit."""
        mgr = _make_mgr(tmp_path)
        with (
            patch.object(VaultManager, "is_socket_active", return_value=True),
            patch(f"{_LIFECYCLE}.subprocess"),
            patch.object(VaultManager, "_wait_for_ready", return_value=False),
            pytest.raises(SystemExit, match="not reachable"),
        ):
            mgr.ensure_reachable()

    def test_raises_when_neither_socket_nor_tcp_reachable(self, tmp_path: Path) -> None:
        """Service started but neither Unix socket nor TCP responds → SystemExit."""
        mgr = _make_mgr(tmp_path)
        with (
            patch.object(VaultManager, "is_socket_active", return_value=True),
            patch(f"{_LIFECYCLE}.subprocess"),
            patch.object(VaultManager, "_wait_for_unix_socket", return_value=False),
            patch.object(VaultManager, "_wait_for_ready", return_value=False),
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
            d = VaultManager._systemd_unit_dir()
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
            d = VaultManager._systemd_unit_dir()
        assert d == xdg / "systemd" / "user"

    def test_systemd_unit_dir_refuses_root(self) -> None:
        """Unit dir refuses to run as root."""
        with (
            patch("os.geteuid", return_value=0),
            pytest.raises(SystemExit, match="root"),
        ):
            VaultManager._systemd_unit_dir()

    def test_systemd_unit_dir_rejects_outside_home(self, tmp_path: Path) -> None:
        """Unit dir rejects XDG_CONFIG_HOME that resolves outside $HOME."""
        with (
            patch.dict(os.environ, {"XDG_CONFIG_HOME": "/etc/evil"}),
            patch("os.geteuid", return_value=1000),
            pytest.raises(SystemExit, match="outside the home directory"),
        ):
            VaultManager._systemd_unit_dir()

    def test_is_systemd_available_true(self) -> None:
        """Returns True when systemctl is-system-running exits 0."""
        result = MagicMock(returncode=0)
        with patch("subprocess.run", return_value=result):
            assert VaultManager().is_systemd_available() is True

    def test_is_systemd_available_degraded(self) -> None:
        """Returns True for degraded state (exit code 1)."""
        result = MagicMock(returncode=1)
        with patch("subprocess.run", return_value=result):
            assert VaultManager().is_systemd_available() is True

    def test_is_systemd_available_missing(self) -> None:
        """Returns False when systemctl is not found."""
        with patch("subprocess.run", side_effect=FileNotFoundError):
            assert VaultManager().is_systemd_available() is False

    def test_is_socket_installed_true(self, tmp_path: Path) -> None:
        """Returns True when socket unit file exists."""
        with patch.object(VaultManager, "_systemd_unit_dir", return_value=tmp_path):
            (tmp_path / "terok-vault.socket").write_text("[Socket]")
            assert VaultManager().is_socket_installed() is True

    def test_is_socket_installed_false(self, tmp_path: Path) -> None:
        """Returns False when socket unit file is absent."""
        with patch.object(VaultManager, "_systemd_unit_dir", return_value=tmp_path):
            assert VaultManager().is_socket_installed() is False

    def test_is_socket_active_true(self) -> None:
        """Returns True when systemctl reports active."""
        result = MagicMock(stdout="active\n")
        with patch("subprocess.run", return_value=result):
            assert VaultManager().is_socket_active() is True

    def test_is_socket_active_false(self) -> None:
        """Returns False when systemctl reports inactive."""
        result = MagicMock(stdout="inactive\n")
        with patch("subprocess.run", return_value=result):
            assert VaultManager().is_socket_active() is False

    def test_is_socket_active_no_systemctl(self) -> None:
        """Returns False when systemctl is not found."""
        with patch("subprocess.run", side_effect=FileNotFoundError):
            assert VaultManager().is_socket_active() is False

    def test_is_service_active_true(self) -> None:
        """Returns True when the service unit is active."""
        result = MagicMock(stdout="active\n")
        with patch("subprocess.run", return_value=result):
            assert VaultManager().is_service_active() is True

    def test_is_service_active_false(self) -> None:
        """Returns False when the service unit is inactive."""
        result = MagicMock(stdout="inactive\n")
        with patch("subprocess.run", return_value=result):
            assert VaultManager().is_service_active() is False

    def test_is_service_active_no_systemctl(self) -> None:
        """Returns False when systemctl is not found."""
        with patch("subprocess.run", side_effect=FileNotFoundError):
            assert VaultManager().is_service_active() is False

    def test_is_service_active_timeout(self) -> None:
        """Returns False on systemctl timeout."""
        with patch("subprocess.run", side_effect=subprocess.TimeoutExpired("cmd", 5)):
            assert VaultManager().is_service_active() is False


class TestGetVaultStatusModes:
    """Verify mode detection and health probing in get_vault_status."""

    def test_systemd_mode_service_active(self, tmp_path: Path) -> None:
        """Reports running=True only when the service unit is active."""
        mgr = _make_mgr(tmp_path)
        with (
            patch.object(VaultManager, "is_socket_installed", return_value=True),
            patch.object(VaultManager, "is_service_active", return_value=True),
            patch.object(VaultManager, "_probe", return_value=True),
        ):
            status = mgr.get_status()
        assert status.mode == "systemd"
        assert status.running is True
        assert status.healthy is True

    def test_systemd_mode_service_idle(self, tmp_path: Path) -> None:
        """Reports running=False when socket is installed but service is idle."""
        mgr = _make_mgr(tmp_path)
        with (
            patch.object(VaultManager, "is_socket_installed", return_value=True),
            patch.object(VaultManager, "is_socket_active", return_value=False),
            patch.object(VaultManager, "is_service_active", return_value=False),
        ):
            status = mgr.get_status()
        assert status.mode == "systemd"
        assert status.running is False
        assert status.healthy is False

    def test_daemon_mode_healthy(self, tmp_path: Path) -> None:
        """Reports mode='daemon' and healthy=True when health probe succeeds."""
        mgr = _make_mgr(tmp_path)
        with (
            patch.object(VaultManager, "is_socket_installed", return_value=False),
            patch.object(VaultManager, "is_daemon_running", return_value=True),
            patch.object(VaultManager, "_probe", return_value=True),
        ):
            status = mgr.get_status()
        assert status.mode == "daemon"
        assert status.running is True
        assert status.healthy is True

    def test_daemon_mode_unhealthy(self, tmp_path: Path) -> None:
        """Reports healthy=False when daemon is alive but probe fails."""
        mgr = _make_mgr(tmp_path)
        with (
            patch.object(VaultManager, "is_socket_installed", return_value=False),
            patch.object(VaultManager, "is_daemon_running", return_value=True),
            patch.object(VaultManager, "_probe", return_value=False),
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
            patch.object(VaultManager, "is_daemon_running", return_value=False),
        ):
            status = mgr.get_status()
        assert status.mode == "none"
        assert status.running is False
        assert status.healthy is False

    def test_systemd_mode_service_active_but_unhealthy(self, tmp_path: Path) -> None:
        """Reports running=True but healthy=False when service is up but probe fails."""
        mgr = _make_mgr(tmp_path)
        with (
            patch.object(VaultManager, "is_socket_installed", return_value=True),
            patch.object(VaultManager, "is_service_active", return_value=True),
            patch.object(VaultManager, "_probe", return_value=False),
        ):
            status = mgr.get_status()
        assert status.mode == "systemd"
        assert status.running is True
        assert status.healthy is False

    def test_systemd_mode_falls_back_to_daemon(self, tmp_path: Path) -> None:
        """When socket installed but service idle, daemon running is not consulted."""
        mgr = _make_mgr(tmp_path)
        with (
            patch.object(VaultManager, "is_socket_installed", return_value=True),
            patch.object(VaultManager, "is_socket_active", return_value=False),
            patch.object(VaultManager, "is_service_active", return_value=False),
            patch.object(VaultManager, "is_daemon_running", return_value=True),
        ):
            status = mgr.get_status()
        # Systemd takes precedence — daemon state is irrelevant
        assert status.mode == "systemd"
        assert status.running is False


class TestProbeVault:
    """Verify VaultManager._probe single-shot vault health check."""

    def _mock_conn(self, *, status: int = 200) -> MagicMock:
        """Return a mock HTTPConnection whose getresponse() returns *status*."""
        mock_resp = MagicMock()
        mock_resp.status = status
        mock_resp.read.return_value = b""
        conn = MagicMock()
        conn.getresponse.return_value = mock_resp
        return conn

    def test_returns_true_on_200(self) -> None:
        """Returns True when vault health endpoint responds 200."""
        conn = self._mock_conn(status=200)
        with patch("http.client.HTTPConnection", return_value=conn):
            assert VaultManager._probe(18731) is True

    def test_returns_false_on_connection_refused(self) -> None:
        """Returns False when the vault is not listening."""
        conn = MagicMock()
        conn.request.side_effect = ConnectionRefusedError
        with patch("http.client.HTTPConnection", return_value=conn):
            assert VaultManager._probe(18731) is False

    def test_returns_false_on_timeout(self) -> None:
        """Returns False when the request times out."""
        conn = MagicMock()
        conn.request.side_effect = OSError("timed out")
        with patch("http.client.HTTPConnection", return_value=conn):
            assert VaultManager._probe(18731) is False


class TestWaitForReady:
    """Verify _wait_for_ready polling loop."""

    def test_returns_true_on_immediate_success(self) -> None:
        """Returns True when the first probe succeeds."""
        with patch.object(VaultManager, "_probe", return_value=True):
            assert VaultManager._wait_for_ready(18731, timeout=1.0) is True

    def test_returns_false_on_timeout(self) -> None:
        """Returns False when all probes fail within the timeout."""
        with (
            patch.object(VaultManager, "_probe", return_value=False),
            patch("time.sleep"),
            patch("time.monotonic", side_effect=[0.0, 0.0, 0.2, 0.4, 6.0]),
        ):
            assert VaultManager._wait_for_ready(18731, timeout=5.0) is False

    def test_retries_then_succeeds(self) -> None:
        """Returns True after a few failed probes followed by success."""
        with (
            patch.object(VaultManager, "_probe", side_effect=[False, False, True]),
            patch("time.sleep"),
            patch("time.monotonic", side_effect=[0.0, 0.0, 0.2, 0.4, 0.6]),
        ):
            assert VaultManager._wait_for_ready(18731, timeout=5.0) is True


class TestInstallSystemdUnits:
    """Verify install_systemd_units."""

    def test_install_creates_units_and_enables_socket(self, tmp_path: Path) -> None:
        """install_systemd_units renders templates and runs systemctl enable."""
        cfg = _make_cfg(tmp_path)
        mgr = VaultManager(cfg)
        unit_dir = tmp_path / "systemd-units"
        with (
            patch.object(VaultManager, "_systemd_unit_dir", return_value=unit_dir),
            patch("subprocess.run") as mock_run,
        ):
            mgr.install_systemd_units()
        # Verify unit files were created
        assert (unit_dir / "terok-vault.socket").is_file()
        assert (unit_dir / "terok-vault.service").is_file()
        # Service template should reference python -m, not a standalone binary
        svc = (unit_dir / "terok-vault.service").read_text()
        assert "-m terok_sandbox.vault" in svc
        # Verify systemctl was called
        calls = [c.args[0] for c in mock_run.call_args_list]
        assert ["systemctl", "--user", "daemon-reload"] in calls
        assert any("enable" in c and "--now" in c for c in calls)
        assert any("restart" in c for c in calls)

    def test_socket_unit_has_both_listen_streams(self, tmp_path: Path) -> None:
        """Socket unit declares both Unix socket and TCP port ListenStream entries."""
        cfg = _make_cfg(tmp_path)
        mgr = VaultManager(cfg)
        unit_dir = tmp_path / "systemd-units"
        with (
            patch.object(VaultManager, "_systemd_unit_dir", return_value=unit_dir),
            patch("subprocess.run"),
        ):
            mgr.install_systemd_units()
        socket_unit = (unit_dir / "terok-vault.socket").read_text()
        listen_lines = [
            line.strip() for line in socket_unit.splitlines() if line.startswith("ListenStream")
        ]
        assert len(listen_lines) == 2
        assert any(str(cfg.vault_socket_path) in entry for entry in listen_lines)
        assert any(f"127.0.0.1:{cfg.token_broker_port}" in entry for entry in listen_lines)


class TestUninstallSystemdUnits:
    """Verify uninstall_systemd_units."""

    def test_uninstall_removes_units(self, tmp_path: Path) -> None:
        """uninstall_systemd_units removes unit files and reloads."""
        unit_dir = tmp_path / "systemd-units"
        unit_dir.mkdir()
        (unit_dir / "terok-vault.socket").write_text("[Socket]")
        (unit_dir / "terok-vault.service").write_text("[Service]")
        with (
            patch.object(VaultManager, "_systemd_unit_dir", return_value=unit_dir),
            patch("subprocess.run") as mock_run,
        ):
            VaultManager().uninstall_systemd_units()
        assert not (unit_dir / "terok-vault.socket").exists()
        assert not (unit_dir / "terok-vault.service").exists()
        # Verify daemon-reload was called
        calls = [c.args[0] for c in mock_run.call_args_list]
        assert ["systemctl", "--user", "daemon-reload"] in calls


class TestOrphanUnitSweep:
    """Verify _sweep_orphan_units removes legacy unit files carrying our marker."""

    def test_legacy_marked_file_removed(self, tmp_path: Path) -> None:
        """A terok-vault-* file with our marker but not in the current set is swept."""
        unit_dir = tmp_path / "systemd-units"
        unit_dir.mkdir()
        legacy = unit_dir / "terok-vault-legacy.service"
        legacy.write_text("# terok-vault-version: 3\n[Service]\n")
        with (
            patch.object(VaultManager, "_systemd_unit_dir", return_value=unit_dir),
            patch("subprocess.run"),
        ):
            VaultManager()._sweep_orphan_units()
        assert not legacy.exists()

    def test_current_name_skipped(self, tmp_path: Path) -> None:
        """Current-version filenames are left for the main removal pass."""
        unit_dir = tmp_path / "systemd-units"
        unit_dir.mkdir()
        current = unit_dir / "terok-vault.socket"
        current.write_text("# terok-vault-version: 5\n[Socket]\n")
        with (
            patch.object(VaultManager, "_systemd_unit_dir", return_value=unit_dir),
            patch("subprocess.run"),
        ):
            VaultManager()._sweep_orphan_units()
        assert current.exists()

    def test_foreign_file_preserved(self, tmp_path: Path) -> None:
        """A user-authored file with matching glob but no marker is not touched."""
        unit_dir = tmp_path / "systemd-units"
        unit_dir.mkdir()
        foreign = unit_dir / "terok-vault-foreign.service"
        foreign.write_text("[Service]\nExecStart=/bin/true\n")
        with (
            patch.object(VaultManager, "_systemd_unit_dir", return_value=unit_dir),
            patch("subprocess.run"),
        ):
            VaultManager()._sweep_orphan_units()
        assert foreign.exists()

    def test_non_matching_glob_preserved(self, tmp_path: Path) -> None:
        """Files outside our glob are never read, regardless of content."""
        unit_dir = tmp_path / "systemd-units"
        unit_dir.mkdir()
        other = unit_dir / "my-service.service"
        other.write_text("# terok-vault-version: 99\n")
        with (
            patch.object(VaultManager, "_systemd_unit_dir", return_value=unit_dir),
            patch("subprocess.run"),
        ):
            VaultManager()._sweep_orphan_units()
        assert other.exists()

    def test_disable_invoked_before_unlink(self, tmp_path: Path) -> None:
        """Each removed unit is systemctl-disabled first (best-effort)."""
        unit_dir = tmp_path / "systemd-units"
        unit_dir.mkdir()
        legacy = unit_dir / "terok-vault-legacy.service"
        legacy.write_text("# terok-vault-version: 3\n[Service]\n")
        with (
            patch.object(VaultManager, "_systemd_unit_dir", return_value=unit_dir),
            patch("subprocess.run") as mock_run,
        ):
            VaultManager()._sweep_orphan_units()
        disable_calls = [c.args[0] for c in mock_run.call_args_list if "disable" in c.args[0]]
        assert any("terok-vault-legacy.service" in cmd for cmd in disable_calls)

    def test_missing_unit_dir_is_a_noop(self, tmp_path: Path) -> None:
        """Running on a host with no user systemd dir must not error."""
        unit_dir = tmp_path / "does-not-exist"
        with (
            patch.object(VaultManager, "_systemd_unit_dir", return_value=unit_dir),
            patch("subprocess.run"),
        ):
            VaultManager()._sweep_orphan_units()  # must not raise
