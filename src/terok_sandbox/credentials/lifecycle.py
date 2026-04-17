# SPDX-FileCopyrightText: 2026 Jiri Vyskocil
# SPDX-License-Identifier: Apache-2.0

"""Credential proxy lifecycle management.

Manages the ``terok-credential-proxy`` daemon: start, stop, status, and
pre-task health checks.  Supports systemd socket activation (preferred)
and a manual daemon fallback.

The systemd socket unit listens on both the Unix socket and the TCP
port used by containers.  A connection to either triggers the service.
:meth:`CredentialProxyManager.ensure_reachable` also performs an explicit
start as a belt-and-suspenders measure before task creation.
"""

from __future__ import annotations

import os
import shlex
import signal
import subprocess
import time
from dataclasses import dataclass
from pathlib import Path

from .._util._logging import log_warning
from ..config import SandboxConfig
from .proxy.constants import HEALTH_PATH as _HEALTH_PATH

# ---------- Vocabulary ----------


@dataclass(frozen=True)
class CredentialProxyStatus:
    """Current state of the credential proxy."""

    mode: str
    """``"systemd"``, ``"daemon"``, or ``"none"``."""

    running: bool
    """Whether the proxy is active (systemd socket listening or daemon alive)."""

    healthy: bool
    """Whether the proxy is healthy for its current activation mode.

    HTTP-probe based when the systemd service is active; socket-liveness
    based when the service is idle but the socket is listening.
    """

    socket_path: Path
    """Configured Unix socket path."""

    db_path: Path
    """Configured credential database path."""

    routes_path: Path
    """Configured proxy routes JSON path."""

    routes_configured: int
    """Number of routes in routes.json (0 if missing or invalid)."""

    credentials_stored: tuple[str, ...]
    """Provider names with stored credentials."""

    transport: str | None = None
    """Detected transport: ``"tcp"``, ``"socket"``, or ``None`` if not running."""


class ProxyUnreachableError(RuntimeError):
    """Raised when the credential proxy is not reachable.

    Carries diagnostic paths so CLI layers can append their own
    remediation hints (specific command names vary by package).
    """

    def __init__(self, *, socket_path: Path, db_path: Path) -> None:
        self.socket_path = socket_path
        self.db_path = db_path
        super().__init__(
            "Credential proxy is not reachable.\n"
            "\n"
            "The credential proxy injects real API credentials into container\n"
            "requests without exposing secrets to the container filesystem.\n"
            "\n"
            "Start the credential proxy (socket activation or manual daemon)\n"
            "before creating tasks.\n"
            "\n"
            f"Socket: {socket_path}\n"
            f"DB:     {db_path}"
        )


# ---------- Constants ----------

_UNIT_VERSION = 5
"""Bump when the systemd unit templates change."""

_SOCKET_UNIT = "terok-credential-proxy.socket"
"""Name of the systemd socket unit file (TCP mode)."""

_SERVICE_UNIT = "terok-credential-proxy.service"
"""Name of the systemd service unit file (TCP mode)."""

_SOCKET_MODE_SERVICE = "terok-credential-proxy-socket.service"
"""Name of the systemd service unit for Unix socket mode."""

_ALL_UNIT_NAMES = (_SOCKET_UNIT, _SERVICE_UNIT, _SOCKET_MODE_SERVICE)
"""All unit file names across both transport modes (for cleanup)."""


# ---------- Manager ----------


class CredentialProxyManager:
    """Lifecycle manager for the terok credential proxy.

    Encapsulates configuration, systemd unit management, daemon process
    control, and health probing behind a single object.  Construct with
    an optional :class:`SandboxConfig`; all methods use the bound
    configuration.
    """

    def __init__(self, cfg: SandboxConfig | None = None) -> None:
        self._cfg = cfg or SandboxConfig()

    # -- Public API ----------------------------------------------------------

    def ensure_reachable(self) -> None:
        """Verify the credential proxy is running and reachable.

        Probes the Unix socket first — if the proxy socket accepts connections,
        the service is alive.  Falls back to TCP health probing for setups
        that only expose TCP ports.

        For **systemd** socket activation the service may not have started yet.
        This function triggers a start via ``systemctl --user start`` and waits.

        Raises :class:`ProxyUnreachableError` if the proxy is unreachable.
        Called before task creation when credential proxy is enabled.
        """
        if not self.is_socket_active() and not self.is_daemon_running():
            raise ProxyUnreachableError(
                socket_path=self._cfg.proxy_socket_path,
                db_path=self._cfg.proxy_db_path,
            )

        # Systemd socket activation: the socket unit is active but the service
        # may be idle.  Explicitly start the service so listeners come up.
        if self.is_socket_active():
            unit = (
                _SOCKET_MODE_SERVICE if self._installed_transport() == "socket" else _SERVICE_UNIT
            )
            subprocess.run(
                ["systemctl", "--user", "start", unit],
                check=False,
                timeout=10,
            )

        # Prefer Unix socket probe (works for both socket and TCP modes).
        if self._wait_for_unix_socket(self._cfg.proxy_socket_path):
            return

        # Fallback: TCP health probe (legacy / TCP-only setups).
        if not self._wait_for_ready(self._cfg.proxy_port):
            raise SystemExit(
                "Credential proxy service started but is not reachable.\n"
                f"Socket: {self._cfg.proxy_socket_path}\n"
                f"TCP:    127.0.0.1:{self._cfg.proxy_port}\n"
                "Check: journalctl --user -u terok-credential-proxy"
            )

    def get_status(self) -> CredentialProxyStatus:
        """Return the current credential proxy status.

        Populates route count from the routes JSON (0 if missing/invalid) and
        credential provider names from the database (empty if DB doesn't exist).
        """
        routes_count = 0
        if self._cfg.proxy_routes_path.is_file():
            try:
                import json

                routes_count = len(json.loads(self._cfg.proxy_routes_path.read_text()))
            except (json.JSONDecodeError, OSError):
                pass

        creds: tuple[str, ...] = ()
        if self._cfg.proxy_db_path.is_file():
            try:
                from .db import CredentialDB

                db = CredentialDB(self._cfg.proxy_db_path)
                try:
                    creds = tuple(db.list_credentials("default"))
                finally:
                    db.close()
            except Exception as exc:  # noqa: BLE001
                log_warning(f"Failed to read credential DB for status: {exc}")

        # Systemd takes precedence: when units are installed, report mode="systemd"
        # even if the socket is inactive — the daemon's running state is ignored so
        # operators see the correct activation path and don't get mixed signals.
        if self.is_socket_installed():
            mode = "systemd"
            socket_up = self.is_socket_active()
            service_up = self.is_service_active()
            running = socket_up or service_up
            healthy = self._probe(self._cfg.proxy_port) if service_up else socket_up
        elif self.is_daemon_running():
            mode = "daemon"
            running = True
            healthy = self._probe(self._cfg.proxy_port)
        else:
            mode = "none"
            running = False
            healthy = False

        # Derive transport from installed unit type (not reachability probe,
        # since TCP mode also binds a Unix socket).
        transport = self._installed_transport() if mode == "systemd" else None

        return CredentialProxyStatus(
            mode=mode,
            running=running,
            healthy=healthy,
            socket_path=self._cfg.proxy_socket_path,
            db_path=self._cfg.proxy_db_path,
            routes_path=self._cfg.proxy_routes_path,
            routes_configured=routes_count,
            credentials_stored=creds,
            transport=transport,
        )

    @property
    def proxy_port(self) -> int:
        """Return the configured credential proxy TCP port."""
        return self._cfg.proxy_port

    @property
    def ssh_agent_port(self) -> int:
        """Return the configured SSH agent proxy TCP port."""
        return self._cfg.ssh_agent_port

    # -- Systemd lifecycle ---------------------------------------------------

    def is_systemd_available(self) -> bool:
        """Check whether the systemd user session is reachable."""
        try:
            result = subprocess.run(
                ["systemctl", "--user", "is-system-running"],
                capture_output=True,
                text=True,
                timeout=5,
            )
            return result.returncode in (0, 1)
        except (FileNotFoundError, subprocess.TimeoutExpired):
            return False

    def is_socket_installed(self) -> bool:
        """Check whether any proxy systemd unit file exists (TCP or socket mode)."""
        unit_dir = self._systemd_unit_dir()
        return (unit_dir / _SOCKET_UNIT).is_file() or (unit_dir / _SOCKET_MODE_SERVICE).is_file()

    def _is_unit_active(self, unit: str) -> bool:
        """Check whether a systemd unit is active."""
        try:
            result = subprocess.run(
                ["systemctl", "--user", "is-active", unit],
                capture_output=True,
                text=True,
                timeout=5,
            )
            return result.stdout.strip() == "active"
        except (FileNotFoundError, subprocess.TimeoutExpired):
            return False

    def is_socket_active(self) -> bool:
        """Check whether the TCP socket unit or socket-mode service is active."""
        return self._is_unit_active(_SOCKET_UNIT) or self._is_unit_active(_SOCKET_MODE_SERVICE)

    def is_service_active(self) -> bool:
        """Check whether the proxy daemon itself is running.

        Checks both TCP-mode service and socket-mode service units.
        """
        return self._is_unit_active(_SERVICE_UNIT) or self._is_unit_active(_SOCKET_MODE_SERVICE)

    def _installed_transport(self) -> str | None:
        """Detect installed transport from unit files on disk."""
        unit_dir = self._systemd_unit_dir()
        if (unit_dir / _SOCKET_MODE_SERVICE).is_file():
            return "socket"
        if (unit_dir / _SOCKET_UNIT).is_file():
            return "tcp"
        return None

    def install_systemd_units(self, *, transport: str = "tcp") -> None:
        """Render and install systemd units, then enable+start.

        When *transport* is ``"tcp"`` (default), installs the socket-activated
        pair (socket + service).  When ``"socket"``, installs a single
        long-running service that binds Unix sockets only.
        """
        # A TCP install with no resolved port would render ``ListenStream=
        # 127.0.0.1:None`` into the ``.socket`` template; systemd rejects
        # that.  Fail early, naming the knobs that resolve it, instead of
        # emitting a broken unit file.
        if transport == "tcp" and (
            self._cfg.proxy_port is None or self._cfg.ssh_agent_port is None
        ):
            raise SystemExit(
                "Cannot install tcp-mode credential-proxy units: no port is set.\n"
                "Either configure ``services.mode: tcp`` (auto-allocates ports)\n"
                "or pin ``credential_proxy.port`` / ``credential_proxy.ssh_agent_port`` explicitly."
            )

        import terok_sandbox.credentials.proxy

        from .._util import render_template

        unit_dir = self._systemd_unit_dir()
        unit_dir.mkdir(parents=True, exist_ok=True)

        resource_dir = (
            Path(terok_sandbox.credentials.proxy.__file__).resolve().parent
            / "resources"
            / "systemd"
        )
        variables = {
            "SOCKET_PATH": str(self._cfg.proxy_socket_path),
            "SSH_AGENT_SOCKET_PATH": str(self._cfg.ssh_agent_socket_path),
            "DB_PATH": str(self._cfg.proxy_db_path),
            "ROUTES_PATH": str(self._cfg.proxy_routes_path),
            "PORT": str(self._cfg.proxy_port),
            "SSH_AGENT_PORT": str(self._cfg.ssh_agent_port),
            "SSH_KEYS_FILE": str(self._cfg.ssh_keys_json_path),
            "BIN": shlex.join(self._proxy_exec_prefix()),
            "UNIT_VERSION": str(_UNIT_VERSION),
        }

        # Remove units from the *other* transport mode before installing.
        self._remove_unit_files()

        if transport == "socket":
            templates = [_SOCKET_MODE_SERVICE]
            enable_unit = _SOCKET_MODE_SERVICE
        else:
            templates = [_SOCKET_UNIT, _SERVICE_UNIT]
            enable_unit = _SOCKET_UNIT

        for template_name in templates:
            template_path = resource_dir / template_name
            if not template_path.is_file():
                raise SystemExit(f"Missing systemd template: {template_path}")
            content = render_template(template_path, variables)
            (unit_dir / template_name).write_text(content, encoding="utf-8")

        self._cfg.proxy_socket_path.parent.mkdir(parents=True, exist_ok=True)
        subprocess.run(["systemctl", "--user", "daemon-reload"], check=True, timeout=10)
        subprocess.run(
            ["systemctl", "--user", "enable", "--now", enable_unit],
            check=True,
            timeout=10,
        )
        # Restart to apply updated unit configuration if socket was already active.
        subprocess.run(
            ["systemctl", "--user", "restart", enable_unit],
            check=True,
            timeout=10,
        )

    def _stop_all_units(self) -> None:
        """Stop and disable all proxy units across both transport modes."""
        unit_dir = self._systemd_unit_dir()
        for unit in (_SOCKET_UNIT, _SERVICE_UNIT, _SOCKET_MODE_SERVICE):
            if (unit_dir / unit).is_file():
                subprocess.run(
                    ["systemctl", "--user", "disable", "--now", unit],
                    check=False,
                    capture_output=True,
                    timeout=10,
                )

    def _remove_unit_files(self) -> None:
        """Stop active units and remove all proxy unit files."""
        self._stop_all_units()
        unit_dir = self._systemd_unit_dir()
        for name in _ALL_UNIT_NAMES:
            unit_file = unit_dir / name
            if unit_file.is_file():
                unit_file.unlink()

    def uninstall_systemd_units(self) -> None:
        """Disable+stop all proxy units and remove unit files."""
        self._remove_unit_files()
        subprocess.run(["systemctl", "--user", "daemon-reload"], check=False, timeout=10)

    # -- Daemon lifecycle ----------------------------------------------------

    def start_daemon(self) -> None:
        """Start the credential proxy as a background daemon.

        The proxy listens on a Unix socket and reads credentials from a
        sqlite3 database.  A routes JSON file must exist at the configured
        path (generated by terok-executor from the YAML registry).

        Writes a PID file to ``runtime_root() / "credential-proxy.pid"``.
        """
        sock_path = self._cfg.proxy_socket_path
        db_path = self._cfg.proxy_db_path
        routes_path = self._cfg.proxy_routes_path
        pidfile = self._cfg.proxy_pid_file_path

        sock_path.parent.mkdir(parents=True, exist_ok=True)
        pidfile.parent.mkdir(parents=True, exist_ok=True)

        from .._util import write_sensitive_file

        if write_sensitive_file(routes_path, "{}\n"):
            import logging

            logging.getLogger(__name__).info(
                "Created empty routes file: %s — populate with: terok auth <provider> <project>",
                routes_path,
            )

        ssh_keys_path = self._cfg.ssh_keys_json_path
        write_sensitive_file(ssh_keys_path, "{}\n")

        log_file = self._cfg.state_dir / "proxy" / "credential-proxy.log"
        log_file.parent.mkdir(parents=True, exist_ok=True)
        os.chmod(log_file.parent, 0o700)

        log_level = os.environ.get("TEROK_PROXY_LOG_LEVEL", "INFO")
        cmd = [
            *self._proxy_exec_prefix(),
            f"--socket-path={sock_path}",
            f"--db-path={db_path}",
            f"--routes-file={routes_path}",
            f"--pid-file={pidfile}",
            f"--port={self._cfg.proxy_port}",
            f"--ssh-agent-port={self._cfg.ssh_agent_port}",
            f"--ssh-keys-file={ssh_keys_path}",
            f"--log-file={log_file}",
            f"--log-level={log_level}",
        ]

        # Fork into background so the proxy survives shell exit.
        # stderr=PIPE only for the startup-failure detection window.
        proc = subprocess.Popen(
            cmd,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.PIPE,
            start_new_session=True,
        )

        # Poll the /-/health endpoint until the server is actually ready.
        if self._wait_for_ready(self._cfg.proxy_port):
            proc.stderr.close()
            return

        # Timed out — check whether the process crashed or is just slow.
        ret = proc.poll()
        if ret is not None:
            stderr = (proc.stderr.read() or b"").decode(errors="replace").strip()
            msg = f"Credential proxy failed to start (exit {ret})"
            if stderr:
                msg += f":\n{stderr}"
            raise SystemExit(msg)
        proc.stderr.close()
        raise SystemExit(
            "Credential proxy process started but did not become ready within 5 s.\n"
            f"Check logs or try: curl http://127.0.0.1:{self._cfg.proxy_port}{_HEALTH_PATH}"
        )

    def stop_daemon(self) -> None:
        """Stop the managed proxy daemon by sending SIGTERM."""
        pidfile = self._cfg.proxy_pid_file_path
        if not pidfile.is_file():
            return
        try:
            pid = int(pidfile.read_text().strip())
            if self._is_managed_proxy(pid):
                os.kill(pid, signal.SIGTERM)
        except (ValueError, ProcessLookupError, PermissionError):
            pass
        finally:
            if pidfile.is_file():
                pidfile.unlink()

    def is_daemon_running(self) -> bool:
        """Check whether the managed proxy daemon is alive via its PID file."""
        pidfile = self._cfg.proxy_pid_file_path
        if not pidfile.is_file():
            return False
        try:
            pid = int(pidfile.read_text().strip())
            if not self._is_managed_proxy(pid):
                return False
            os.kill(pid, 0)  # signal 0 = existence check
            return True
        except (ValueError, ProcessLookupError, PermissionError):
            return False

    # -- Private helpers -----------------------------------------------------

    def _is_managed_proxy(self, pid: int) -> bool:
        """Return whether *pid* was started with the expected PID file argument."""
        cmdline_path = Path(f"/proc/{pid}/cmdline")
        if not cmdline_path.is_file():
            return False
        try:
            raw = cmdline_path.read_bytes()
        except OSError:
            return False
        args = raw.rstrip(b"\x00").split(b"\x00")
        args_str = [a.decode("utf-8", errors="ignore") for a in args]
        expected = f"--pid-file={self._cfg.proxy_pid_file_path}"
        return expected in args_str

    @staticmethod
    def _systemd_unit_dir() -> Path:
        """Return the validated systemd user unit directory."""
        from .._util import systemd_user_unit_dir

        return systemd_user_unit_dir()

    @staticmethod
    def _proxy_exec_prefix() -> list[str]:
        """Return the command prefix for launching the credential proxy server.

        Uses ``sys.executable -m terok_sandbox.credentials.proxy`` so the
        server runs under the same Python that owns the installed package.
        """
        import sys as _sys

        return [_sys.executable, "-m", "terok_sandbox.credentials.proxy"]

    @staticmethod
    def _probe(port: int, *, timeout: float = 2.0) -> bool:
        """Return ``True`` if the proxy's health endpoint responds 200.

        Uses :mod:`http.client` (stdlib only) to hit the TCP port.
        """
        import http.client

        conn: http.client.HTTPConnection | None = None
        try:
            conn = http.client.HTTPConnection("127.0.0.1", port, timeout=timeout)
            conn.request("GET", _HEALTH_PATH)
            resp = conn.getresponse()
            resp.read()
            return resp.status == 200
        except (OSError, http.client.HTTPException, ValueError):
            return False
        finally:
            if conn is not None:
                conn.close()

    @staticmethod
    def _wait_for_ready(port: int, *, timeout: float = 5.0, interval: float = 0.2) -> bool:
        """Poll the health endpoint until it responds 200 or *timeout* expires."""
        probe_timeout = min(1.0, interval)
        deadline = time.monotonic() + timeout
        while time.monotonic() < deadline:
            if CredentialProxyManager._probe(port, timeout=probe_timeout):
                return True
            time.sleep(interval)
        return False

    @staticmethod
    def _wait_for_tcp_port(port: int, timeout: float = 5.0) -> bool:
        """Wait up to *timeout* seconds for a TCP port on localhost to accept connections."""
        import socket

        deadline = time.monotonic() + timeout
        while time.monotonic() < deadline:
            sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            try:
                if sock.connect_ex(("127.0.0.1", port)) == 0:
                    return True
            finally:
                sock.close()
            time.sleep(0.2)
        return False

    @staticmethod
    def _wait_for_unix_socket(path: Path, *, timeout: float = 5.0, interval: float = 0.2) -> bool:
        """Wait up to *timeout* seconds for a Unix socket to accept connections."""
        from .._util._net import probe_unix_socket

        probe_timeout = min(1.0, interval)
        deadline = time.monotonic() + timeout
        while time.monotonic() < deadline:
            if probe_unix_socket(path, timeout=probe_timeout):
                return True
            time.sleep(interval)
        return False
