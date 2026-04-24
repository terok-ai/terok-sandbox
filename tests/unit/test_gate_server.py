# SPDX-FileCopyrightText: 2026 Jiri Vyskocil
# SPDX-License-Identifier: Apache-2.0

"""Tests for gate_server module."""

from __future__ import annotations

import contextlib
import os
import subprocess
import tempfile
import unittest.mock
from collections.abc import Iterator
from contextlib import contextmanager
from pathlib import Path

import pytest

from terok_sandbox.gate.lifecycle import (
    _UNIT_VERSION,
    GateServerManager,
    GateServerStatus,
)
from tests.constants import (
    FAKE_GATE_DIR,
    FAKE_STATE_DIR,
    GATE_PORT,
    LOCALHOST,
    NONEXISTENT_DIR,
)

GATE_BASE_PATH = FAKE_GATE_DIR
STATE_ROOT_PATH = FAKE_STATE_DIR
MISSING_PATH = NONEXISTENT_DIR
SYSTEMD_SOCKET = "terok-gate.socket"
SYSTEMD_SERVICE = "terok-gate@.service"
VERSION_STAMP = f"# terok-gate-version: {_UNIT_VERSION}"


def make_status(
    mode: str = "none",
    *,
    running: bool = False,
    port: int = GATE_PORT,
    transport: str | None = None,
) -> GateServerStatus:
    """Create a gate-server status object for tests."""
    return GateServerStatus(mode=mode, running=running, port=port, transport=transport)


def make_run_result(*, returncode: int, stdout: str = "") -> unittest.mock.Mock:
    """Create a mock ``subprocess.run`` result."""
    return unittest.mock.Mock(returncode=returncode, stdout=stdout)


@contextmanager
def patched_unit_dir(files: dict[str, str] | None = None) -> Iterator[Path]:
    """Create a temporary systemd unit dir and patch gate-server to use it."""
    with tempfile.TemporaryDirectory() as td:
        unit_dir = Path(td)
        for name, content in (files or {}).items():
            (unit_dir / name).write_text(content)
        with unittest.mock.patch.object(
            GateServerManager,
            "_systemd_unit_dir",
            return_value=unit_dir,
        ):
            yield unit_dir


@contextmanager
def patched_install_env(unit_dir: Path) -> Iterator[None]:
    """Patch the standard paths used by install/uninstall/start tests."""
    from terok_sandbox.config import SandboxConfig

    mock_cfg = unittest.mock.MagicMock(spec=SandboxConfig)
    mock_cfg.services_mode = "tcp"
    mock_cfg.gate_port = GATE_PORT
    mock_cfg.gate_base_path = GATE_BASE_PATH
    mock_cfg.token_file_path = STATE_ROOT_PATH / "gate" / "tokens.json"
    mock_cfg.pid_file_path = STATE_ROOT_PATH / "gate-server.pid"
    with (
        unittest.mock.patch.object(
            GateServerManager,
            "_systemd_unit_dir",
            return_value=unit_dir,
        ),
        unittest.mock.patch.object(
            GateServerManager,
            "__init__",
            lambda self, cfg=None: setattr(self, "_cfg", mock_cfg),
        ),
    ):
        yield


@contextmanager
def patched_daemon_paths(base: Path) -> Iterator[Path]:
    """Patch daemon-related runtime paths under a temp directory."""
    from terok_sandbox.config import SandboxConfig

    pid_file = base / "gate-server.pid"
    mock_cfg = unittest.mock.MagicMock(spec=SandboxConfig)
    mock_cfg.gate_base_path = base / "gate"
    mock_cfg.gate_port = GATE_PORT
    mock_cfg.pid_file_path = pid_file
    mock_cfg.token_file_path = base / "gate" / "tokens.json"
    with unittest.mock.patch.object(
        GateServerManager,
        "__init__",
        lambda self, cfg=None: setattr(self, "_cfg", mock_cfg),
    ):
        yield pid_file


def write_pid_file(base: Path, pid: int | str = 99999) -> Path:
    """Write a PID file in ``base`` and return its path."""
    pid_file = base / "gate-server.pid"
    pid_file.write_text(f"{pid}\n")
    return pid_file


def unit_file_contents(
    version: int | None = _UNIT_VERSION, base_path: Path = GATE_BASE_PATH
) -> dict[str, str]:
    """Build socket/service contents with an optional version stamp."""
    prefix = "" if version is None else f"# terok-gate-version: {version}\n"
    return {
        SYSTEMD_SOCKET: f"{prefix}[Socket]\nListenStream={LOCALHOST}:{GATE_PORT}\n",
        SYSTEMD_SERVICE: (
            f"{prefix}[Service]\n"
            f"ExecStart=/usr/local/bin/terok-gate --inetd --base-path={base_path} --token-file=/tmp/tokens.json\n"
        ),
    }


def assert_contains_all(text: str, expected: tuple[str, ...]) -> None:
    """Assert that all fragments in ``expected`` appear in ``text``."""
    for fragment in expected:
        assert fragment in text


class TestUnitVersion:
    """Tests for _UNIT_VERSION."""

    def test_unit_version_is_current(self) -> None:
        assert _UNIT_VERSION == 7


class TestInstalledBasePath:
    """Tests for _installed_base_path — parsing --base-path from installed service unit."""

    def test_parses_base_path_from_service(self) -> None:
        files = unit_file_contents(base_path=Path("/custom/gate"))
        with patched_unit_dir(files):
            assert GateServerManager()._installed_base_path() == Path("/custom/gate")

    def test_returns_none_when_no_service(self) -> None:
        with patched_unit_dir({}):
            assert GateServerManager()._installed_base_path() is None

    def test_returns_none_when_no_base_path_flag(self) -> None:
        files = {
            SYSTEMD_SERVICE: "[Service]\nExecStart=/usr/local/bin/terok-gate --inetd\n",
        }
        with patched_unit_dir(files):
            assert GateServerManager()._installed_base_path() is None


class TestBasePathDiverged:
    """Tests for _base_path_diverged — comparing installed vs expected base path."""

    def test_returns_none_when_paths_match(self) -> None:
        from terok_sandbox.config import SandboxConfig

        files = unit_file_contents(base_path=GATE_BASE_PATH)
        mock_cfg = unittest.mock.MagicMock(spec=SandboxConfig)
        mock_cfg.gate_base_path = GATE_BASE_PATH
        with patched_unit_dir(files):
            assert GateServerManager(mock_cfg)._base_path_diverged() is None

    def test_returns_warning_when_paths_diverge(self) -> None:
        from terok_sandbox.config import SandboxConfig

        files = unit_file_contents(base_path=Path("/old/gate"))
        mock_cfg = unittest.mock.MagicMock(spec=SandboxConfig)
        mock_cfg.gate_base_path = Path("/new/gate")
        with patched_unit_dir(files):
            warning = GateServerManager(mock_cfg)._base_path_diverged()
            assert warning is not None
            assert "/old/gate" in warning
            assert "/new/gate" in warning

    def test_returns_none_when_no_units(self) -> None:
        with patched_unit_dir({}):
            assert GateServerManager()._base_path_diverged() is None


class TestSystemdDetection:
    """Tests for systemd availability detection."""

    @pytest.mark.parametrize(
        ("returncode", "expected"),
        [(0, True), (1, True), (2, False)],
        ids=["ok", "acceptable-nonzero", "unavailable"],
    )
    @unittest.mock.patch("subprocess.run")
    def test_systemd_availability_from_return_code(
        self,
        mock_run: unittest.mock.Mock,
        returncode: int,
        expected: bool,
    ) -> None:
        mock_run.return_value = make_run_result(returncode=returncode)
        assert GateServerManager().is_systemd_available() is expected

    @unittest.mock.patch("subprocess.run", side_effect=FileNotFoundError)
    def test_systemd_not_available_when_missing(self, _mock: unittest.mock.Mock) -> None:
        assert not GateServerManager().is_systemd_available()


class TestSocketInstalled:
    """Tests for socket unit file detection."""

    def test_socket_not_installed(self) -> None:
        with unittest.mock.patch.object(
            GateServerManager,
            "_systemd_unit_dir",
            return_value=MISSING_PATH,
        ):
            assert not GateServerManager().is_socket_installed()

    def test_socket_installed(self) -> None:
        with patched_unit_dir({SYSTEMD_SOCKET: "[Socket]\n"}):
            assert GateServerManager().is_socket_installed()


class TestSocketActive:
    """Tests for socket active check."""

    @pytest.mark.parametrize(
        ("stdout", "returncode", "expected"),
        [("active\n", 0, True), ("inactive\n", 3, False)],
        ids=["active", "inactive"],
    )
    @unittest.mock.patch("subprocess.run")
    def test_socket_active_from_systemctl(
        self,
        mock_run: unittest.mock.Mock,
        stdout: str,
        returncode: int,
        expected: bool,
    ) -> None:
        mock_run.return_value = make_run_result(returncode=returncode, stdout=stdout)
        assert GateServerManager().is_socket_active() is expected

    @unittest.mock.patch("subprocess.run", side_effect=FileNotFoundError)
    def test_socket_inactive_without_systemctl(self, _mock: unittest.mock.Mock) -> None:
        assert not GateServerManager().is_socket_active()


class TestInstallUninstall:
    """Tests for systemd unit install/uninstall."""

    @unittest.mock.patch("subprocess.run")
    @unittest.mock.patch("shutil.which", return_value="/usr/local/bin/terok-gate")
    def test_install_writes_files(
        self,
        _mock_which: unittest.mock.Mock,
        mock_run: unittest.mock.Mock,
    ) -> None:
        mock_run.return_value = make_run_result(returncode=0)
        with tempfile.TemporaryDirectory() as td:
            unit_dir = Path(td) / "systemd" / "user"
            with patched_install_env(unit_dir):
                GateServerManager().install_systemd_units()

            socket_content = (unit_dir / SYSTEMD_SOCKET).read_text()
            service_content = (unit_dir / SYSTEMD_SERVICE).read_text()
            assert (unit_dir / SYSTEMD_SOCKET).is_file()
            assert (unit_dir / SYSTEMD_SERVICE).is_file()
            assert_contains_all(socket_content, (f"{LOCALHOST}:{GATE_PORT}", VERSION_STAMP))
            assert_contains_all(
                service_content,
                (
                    "ExecStart=/usr/local/bin/terok-gate",
                    str(GATE_BASE_PATH),
                    "--token-file=",
                    VERSION_STAMP,
                ),
            )

    @unittest.mock.patch("subprocess.run")
    @unittest.mock.patch("shutil.which", return_value=None)
    def test_install_fails_without_binary(
        self,
        _mock_which: unittest.mock.Mock,
        _mock_run: unittest.mock.Mock,
    ) -> None:
        with pytest.raises(SystemExit, match="terok-gate"):
            GateServerManager().install_systemd_units()

    @unittest.mock.patch("subprocess.run")
    def test_uninstall_removes_files(self, mock_run: unittest.mock.Mock) -> None:
        mock_run.return_value = make_run_result(returncode=0)
        with patched_unit_dir(unit_file_contents()):
            GateServerManager().uninstall_systemd_units()
            assert not GateServerManager().is_socket_installed()


class TestDaemon:
    """Tests for daemon start/stop."""

    @unittest.mock.patch("subprocess.run")
    def test_start_daemon(self, mock_run: unittest.mock.Mock) -> None:
        mock_run.return_value = make_run_result(returncode=0)
        with tempfile.TemporaryDirectory() as td:
            with patched_daemon_paths(Path(td)):
                GateServerManager().start_daemon(port=9999)

        mock_run.assert_called_once()
        cmd = mock_run.call_args[0][0]
        assert cmd[0] == "terok-gate"
        assert "--port=9999" in cmd
        assert "--detach" in cmd
        assert any("--base-path=" in arg for arg in cmd)
        assert any("--token-file=" in arg for arg in cmd)

    @unittest.mock.patch(
        "subprocess.run", side_effect=subprocess.CalledProcessError(1, "terok-gate")
    )
    def test_start_daemon_failure(self, _mock: unittest.mock.Mock) -> None:
        with tempfile.TemporaryDirectory() as td:
            with patched_daemon_paths(Path(td)):
                with pytest.raises(subprocess.CalledProcessError):
                    GateServerManager().start_daemon(port=9999)

    def test_stop_daemon_no_pidfile(self) -> None:
        from terok_sandbox.config import SandboxConfig

        mock_cfg = unittest.mock.MagicMock(spec=SandboxConfig)
        mock_cfg.pid_file_path = MISSING_PATH / "pid"
        GateServerManager(mock_cfg).stop_daemon()

    @pytest.mark.parametrize(
        ("managed", "should_kill"),
        [(True, True), (False, False)],
        ids=["managed", "stale"],
    )
    def test_stop_daemon_with_pidfile(self, managed: bool, should_kill: bool) -> None:
        """Stop removes PID files and only kills managed daemons."""
        with tempfile.TemporaryDirectory() as td:
            pid_file = write_pid_file(Path(td))
            from terok_sandbox.config import SandboxConfig

            mock_cfg = unittest.mock.MagicMock(spec=SandboxConfig)
            mock_cfg.pid_file_path = pid_file
            with (
                unittest.mock.patch.object(
                    GateServerManager,
                    "_is_managed_server",
                    return_value=managed,
                ),
                unittest.mock.patch("os.kill") as mock_kill,
            ):
                GateServerManager(mock_cfg).stop_daemon()
            assert mock_kill.called is should_kill
            if should_kill:
                mock_kill.assert_called_once_with(99999, unittest.mock.ANY)
            assert not pid_file.exists()

    def test_stop_daemon_socket_mode_stops_systemd_unit(self) -> None:
        """In socket mode there is no PID file; ``stop_daemon`` still stops the active unit."""
        from terok_sandbox.config import SandboxConfig

        mock_cfg = unittest.mock.MagicMock(spec=SandboxConfig)
        mock_cfg.pid_file_path = MISSING_PATH / "pid"

        def _active(_self: GateServerManager, unit: str) -> bool:
            return unit == "terok-gate-socket.service"

        with (
            unittest.mock.patch.object(GateServerManager, "_is_unit_active", _active),
            unittest.mock.patch("subprocess.run") as mock_run,
        ):
            GateServerManager(mock_cfg).stop_daemon()

        calls = [c for c in mock_run.call_args_list if "stop" in c.args[0]]
        assert len(calls) == 1
        assert calls[0].args[0][:3] == ["systemctl", "--user", "stop"]
        assert "terok-gate-socket.service" in calls[0].args[0]

    def test_stop_daemon_wedged_systemctl_does_not_block_pidfile_cleanup(self) -> None:
        """A hung ``systemctl stop`` must not skip the PID-file SIGTERM path."""
        with tempfile.TemporaryDirectory() as td:
            pid_file = write_pid_file(Path(td))
            from terok_sandbox.config import SandboxConfig

            mock_cfg = unittest.mock.MagicMock(spec=SandboxConfig)
            mock_cfg.pid_file_path = pid_file

            with (
                unittest.mock.patch.object(GateServerManager, "_is_unit_active", return_value=True),
                unittest.mock.patch.object(
                    GateServerManager, "_is_managed_server", return_value=True
                ),
                unittest.mock.patch(
                    "subprocess.run",
                    side_effect=subprocess.TimeoutExpired(cmd="systemctl", timeout=10),
                ),
                unittest.mock.patch("os.kill") as mock_kill,
            ):
                GateServerManager(mock_cfg).stop_daemon()

            mock_kill.assert_called_once_with(99999, unittest.mock.ANY)
            assert not pid_file.exists()


class TestIsDaemonRunning:
    """Tests for is_daemon_running."""

    def test_no_pidfile(self) -> None:
        from terok_sandbox.config import SandboxConfig

        mock_cfg = unittest.mock.MagicMock(spec=SandboxConfig)
        mock_cfg.pid_file_path = MISSING_PATH / "pid"
        assert not GateServerManager(mock_cfg).is_daemon_running()

    @pytest.mark.parametrize(
        ("pid", "managed", "kill_side_effect", "expected"),
        [
            (99999, True, ProcessLookupError, False),
            (os.getpid(), True, None, True),
            (os.getpid(), False, None, False),
        ],
        ids=["stale-pid", "valid-managed-pid", "not-our-daemon"],
    )
    def test_pidfile_states(
        self,
        pid: int,
        managed: bool,
        kill_side_effect: type[BaseException] | None,
        expected: bool,
    ) -> None:
        with tempfile.TemporaryDirectory() as td:
            pid_file = write_pid_file(Path(td), pid)
            from terok_sandbox.config import SandboxConfig

            mock_cfg = unittest.mock.MagicMock(spec=SandboxConfig)
            mock_cfg.pid_file_path = pid_file
            patches = [
                unittest.mock.patch.object(
                    GateServerManager,
                    "_is_managed_server",
                    return_value=managed,
                ),
                unittest.mock.patch("os.kill", side_effect=kill_side_effect),
            ]

            with contextlib.ExitStack() as stack:
                for patcher in patches:
                    stack.enter_context(patcher)
                assert GateServerManager(mock_cfg).is_daemon_running() is expected


class TestIsManagedServer:
    """Tests for _is_managed_server."""

    def _make_mgr(self, pid_file: Path | None = None) -> GateServerManager:
        """Create a GateServerManager with mock config pointing to *pid_file*."""
        from terok_sandbox.config import SandboxConfig

        mock_cfg = unittest.mock.MagicMock(spec=SandboxConfig)
        mock_cfg.pid_file_path = pid_file or MISSING_PATH / "pid"
        return GateServerManager(mock_cfg)

    def test_no_proc_entry(self) -> None:
        assert not self._make_mgr().is_daemon_running()
        assert not self._make_mgr()._is_managed_server(999999999)

    def test_current_process_is_not_gate_server(self) -> None:
        assert not self._make_mgr()._is_managed_server(os.getpid())

    def _check_cmdline(self, cmdline: bytes, pid_file: Path | None = None) -> bool:
        """Write *cmdline* to a temp file and call ``_is_managed_server``."""
        with tempfile.TemporaryDirectory() as td:
            fake_cmdline = Path(td) / "cmdline"
            fake_cmdline.write_bytes(cmdline)
            mgr = self._make_mgr(pid_file)
            with unittest.mock.patch(
                "terok_sandbox.gate.lifecycle.Path",
                return_value=fake_cmdline,
            ):
                return mgr._is_managed_server(12345)

    def test_matches_managed_server(self) -> None:
        pid_file = Path("/run/user/1000/terok/gate-server.pid")
        cmdline = (
            b"terok-gate\x00--base-path="
            + str(GATE_BASE_PATH).encode()
            + b"\x00--pid-file="
            + str(pid_file).encode()
        )
        assert self._check_cmdline(cmdline, pid_file)

    def test_rejects_different_pid_file(self) -> None:
        cmdline = (
            b"terok-gate\x00--base-path="
            + str(GATE_BASE_PATH).encode()
            + b"\x00--pid-file=/other/pid"
        )
        assert not self._check_cmdline(cmdline, Path("/run/user/1000/terok/gate-server.pid"))

    def test_rejects_unrelated_process(self) -> None:
        assert not self._check_cmdline(
            b"python3\x00-m\x00pytest",
            Path("/run/user/1000/terok/gate-server.pid"),
        )


class TestGetServerStatus:
    """Tests for get_server_status."""

    @pytest.mark.parametrize(
        ("socket_installed", "socket_active", "daemon_running", "transport", "expected"),
        [
            (False, False, False, None, make_status("none", running=False)),
            (True, True, False, "tcp", make_status("systemd", running=True, transport="tcp")),
            (True, False, False, None, make_status("systemd", running=False)),
            (True, False, True, "tcp", make_status("daemon", running=True, transport="tcp")),
            (False, False, True, "tcp", make_status("daemon", running=True, transport="tcp")),
        ],
        ids=[
            "no-server",
            "systemd-active",
            "systemd-inactive",
            "daemon-fallback",
            "daemon-only",
        ],
    )
    def test_status_modes(
        self,
        socket_installed: bool,
        socket_active: bool,
        daemon_running: bool,
        transport: str | None,
        expected: GateServerStatus,
    ) -> None:
        from terok_sandbox.config import SandboxConfig

        mock_cfg = unittest.mock.MagicMock(spec=SandboxConfig)
        mock_cfg.gate_port = GATE_PORT
        with (
            unittest.mock.patch.object(
                GateServerManager,
                "is_socket_installed",
                return_value=socket_installed,
            ),
            unittest.mock.patch.object(
                GateServerManager,
                "is_socket_active",
                return_value=socket_active,
            ),
            unittest.mock.patch.object(
                GateServerManager,
                "is_daemon_running",
                return_value=daemon_running,
            ),
            unittest.mock.patch.object(
                GateServerManager,
                "_detect_transport",
                return_value=transport,
            ),
        ):
            assert GateServerManager(mock_cfg).get_status() == expected


class TestEnsureServerReachable:
    """Tests for ensure_server_reachable."""

    @pytest.mark.parametrize(
        ("status", "systemd_available", "unit_version", "error_match"),
        [
            (make_status("daemon", running=True), True, _UNIT_VERSION, None),
            (make_status("none", running=False), True, _UNIT_VERSION, "systemd socket"),
            (make_status("none", running=False), False, _UNIT_VERSION, "gate daemon"),
            (make_status("systemd", running=True), True, 0, "outdated"),
            (make_status("systemd", running=True), True, None, "unversioned"),
            (make_status("systemd", running=True), True, _UNIT_VERSION, None),
        ],
        ids=[
            "daemon-running",
            "stopped-with-systemd",
            "stopped-without-systemd",
            "outdated-units",
            "unversioned-units",
            "current-units",
        ],
    )
    def test_reachability(
        self,
        status: GateServerStatus,
        systemd_available: bool,
        unit_version: int | None,
        error_match: str | None,
    ) -> None:
        with (
            unittest.mock.patch.object(
                GateServerManager,
                "get_status",
                return_value=status,
            ),
            unittest.mock.patch.object(
                GateServerManager,
                "is_systemd_available",
                return_value=systemd_available,
            ),
            unittest.mock.patch.object(
                GateServerManager,
                "_installed_unit_version",
                return_value=unit_version,
            ),
        ):
            if error_match is None:
                GateServerManager().ensure_reachable()
            else:
                with pytest.raises(SystemExit, match=error_match):
                    GateServerManager().ensure_reachable()


class TestInstalledUnitVersion:
    """Tests for _installed_unit_version."""

    def test_no_file(self) -> None:
        with unittest.mock.patch.object(
            GateServerManager,
            "_systemd_unit_dir",
            return_value=MISSING_PATH,
        ):
            assert GateServerManager()._installed_unit_version() is None

    @pytest.mark.parametrize(
        ("files", "expected"),
        [
            ({SYSTEMD_SOCKET: "# terok-gate-version: 42\n[Socket]\n"}, 42),
            ({SYSTEMD_SOCKET: f"[Socket]\nListenStream={LOCALHOST}:{GATE_PORT}\n"}, None),
        ],
        ids=["stamped", "missing-stamp"],
    )
    def test_reads_version_from_socket(self, files: dict[str, str], expected: int | None) -> None:
        with patched_unit_dir(files):
            assert GateServerManager()._installed_unit_version() is expected


class TestCheckUnitsOutdated:
    """Tests for check_units_outdated."""

    @pytest.mark.parametrize(
        ("socket_installed", "version", "expected"),
        [
            (False, _UNIT_VERSION, None),
            (True, _UNIT_VERSION, None),
            (True, 1, "outdated"),
            (True, None, "unversioned"),
        ],
        ids=["no-socket", "current", "outdated", "unversioned"],
    )
    def test_outdated_message(
        self,
        socket_installed: bool,
        version: int | None,
        expected: str | None,
    ) -> None:
        with (
            unittest.mock.patch.object(
                GateServerManager,
                "is_socket_installed",
                return_value=socket_installed,
            ),
            unittest.mock.patch.object(
                GateServerManager,
                "_installed_unit_version",
                return_value=version,
            ),
        ):
            result = GateServerManager().check_units_outdated()
        if expected is None:
            assert result is None
        else:
            assert result is not None
            assert expected in result

    def test_base_path_divergence_warning(self) -> None:
        """Current units + divergent base path → warning string."""
        with (
            unittest.mock.patch.object(
                GateServerManager,
                "is_socket_installed",
                return_value=True,
            ),
            unittest.mock.patch.object(
                GateServerManager,
                "_installed_unit_version",
                return_value=_UNIT_VERSION,
            ),
            unittest.mock.patch.object(
                GateServerManager,
                "_base_path_diverged",
                return_value="paths diverge",
            ),
        ):
            assert GateServerManager().check_units_outdated() == "paths diverge"


class TestDaemonEnvVars:
    """Tests for start_daemon env var forwarding."""

    @unittest.mock.patch("subprocess.run")
    def test_admin_token_passed_via_env_not_argv(self, mock_run: unittest.mock.Mock) -> None:
        mock_run.return_value = make_run_result(returncode=0)
        with tempfile.TemporaryDirectory() as td:
            with patched_daemon_paths(Path(td)):
                with unittest.mock.patch.dict(os.environ, {"TEROK_GATE_ADMIN_TOKEN": "secret42"}):
                    GateServerManager().start_daemon(port=GATE_PORT)

        cmd = mock_run.call_args[0][0]
        assert not any("admin-token" in arg for arg in cmd), "token must not leak into argv"
        env = mock_run.call_args[1].get("env", {})
        assert env.get("TEROK_GATE_ADMIN_TOKEN") == "secret42"

    @unittest.mock.patch("subprocess.run")
    def test_bind_addr_forwarded(self, mock_run: unittest.mock.Mock) -> None:
        mock_run.return_value = make_run_result(returncode=0)
        with tempfile.TemporaryDirectory() as td:
            with patched_daemon_paths(Path(td)):
                with unittest.mock.patch.dict(os.environ, {"TEROK_GATE_BIND": "0.0.0.0"}):
                    GateServerManager().start_daemon(port=GATE_PORT)

        cmd = mock_run.call_args[0][0]
        assert "--bind=0.0.0.0" in cmd


class TestEnsureReachableBasePath:
    """Tests for base path divergence blocking in ensure_server_reachable."""

    def test_base_path_divergence_blocks(self) -> None:
        with (
            unittest.mock.patch.object(
                GateServerManager,
                "get_status",
                return_value=make_status("systemd", running=True),
            ),
            unittest.mock.patch.object(
                GateServerManager,
                "_installed_unit_version",
                return_value=_UNIT_VERSION,
            ),
            unittest.mock.patch.object(
                GateServerManager,
                "_base_path_diverged",
                return_value="Installed: /old\n  Expected: /new",
            ),
        ):
            with pytest.raises(SystemExit, match="Installed"):
                GateServerManager().ensure_reachable()


class TestPublicApi:
    """Tests for the public API properties."""

    def test_gate_base_path_returns_cfg_value(self) -> None:
        from terok_sandbox.config import SandboxConfig

        mock_cfg = unittest.mock.MagicMock(spec=SandboxConfig)
        mock_cfg.gate_base_path = GATE_BASE_PATH
        assert GateServerManager(mock_cfg).gate_base_path == GATE_BASE_PATH

    def test_server_port_returns_cfg_value(self) -> None:
        from terok_sandbox.config import SandboxConfig

        mock_cfg = unittest.mock.MagicMock(spec=SandboxConfig)
        mock_cfg.gate_port = GATE_PORT
        assert GateServerManager(mock_cfg).server_port == GATE_PORT
