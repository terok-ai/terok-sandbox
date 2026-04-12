# SPDX-FileCopyrightText: 2026 Jiri Vyskocil
# SPDX-License-Identifier: Apache-2.0

"""Credential proxy lifecycle management.

Manages the ``terok-credential-proxy`` daemon: start, stop, status, and
pre-task health checks.  Supports systemd socket activation (preferred)
and a manual daemon fallback.

The systemd socket unit listens on both the Unix socket and the TCP
port used by containers.  A connection to either triggers the service.
:func:`ensure_proxy_reachable` also performs an explicit start as a
belt-and-suspenders measure before task creation.
"""

from __future__ import annotations

import os
import shlex
import signal
import subprocess
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


# ---------- Public API ----------


def ensure_proxy_reachable(cfg: SandboxConfig | None = None) -> None:
    """Verify the credential proxy is running and its TCP ports are up.

    For **systemd** socket activation the service may not have started yet
    (e.g. after a fresh boot).  This function triggers a start via
    ``systemctl --user start`` and waits for the HTTP and SSH agent TCP
    ports to become reachable via ``/-/health`` and raw TCP probes.

    For **daemon** mode the ``/-/health`` endpoint is probed on the TCP port.

    Raises :class:`ProxyUnreachableError` if the proxy is unreachable.
    Called before task creation when credential proxy is enabled.
    """
    c = _cfg(cfg)

    if not is_socket_active() and not is_daemon_running(cfg):
        raise ProxyUnreachableError(
            socket_path=c.proxy_socket_path,
            db_path=c.proxy_db_path,
        )

    # Systemd socket activation: the socket unit is active but the service
    # may be idle.  Explicitly start the service so the TCP ports come up.
    if is_socket_active():
        subprocess.run(
            ["systemctl", "--user", "start", _SERVICE_UNIT],
            check=False,
            timeout=10,
        )

    if not _wait_for_ready(c.proxy_port):
        raise SystemExit(
            f"Credential proxy service started but TCP port {c.proxy_port} "
            "is not reachable. Check: journalctl --user -u terok-credential-proxy"
        )

    if not _wait_for_tcp_port(c.ssh_agent_port):
        raise SystemExit(
            f"Credential proxy service started but SSH agent port {c.ssh_agent_port} "
            "is not reachable. Check: journalctl --user -u terok-credential-proxy"
        )


def get_proxy_status(cfg: SandboxConfig | None = None) -> CredentialProxyStatus:
    """Return the current credential proxy status.

    Populates route count from the routes JSON (0 if missing/invalid) and
    credential provider names from the database (empty if DB doesn't exist).
    """
    c = _cfg(cfg)

    routes_count = 0
    if c.proxy_routes_path.is_file():
        try:
            import json

            routes_count = len(json.loads(c.proxy_routes_path.read_text()))
        except (json.JSONDecodeError, OSError):
            pass

    creds: tuple[str, ...] = ()
    if c.proxy_db_path.is_file():
        try:
            from .db import CredentialDB

            db = CredentialDB(c.proxy_db_path)
            try:
                creds = tuple(db.list_credentials("default"))
            finally:
                db.close()
        except Exception as exc:  # noqa: BLE001
            log_warning(f"Failed to read credential DB for status: {exc}")

    # Systemd takes precedence: when units are installed, report mode="systemd"
    # even if the socket is inactive — the daemon's running state is ignored so
    # operators see the correct activation path and don't get mixed signals.
    if is_socket_installed():
        mode = "systemd"
        socket_up = is_socket_active()
        service_up = is_service_active()
        running = socket_up or service_up
        healthy = _probe_proxy(c.proxy_port) if service_up else socket_up
    elif is_daemon_running(cfg):
        mode = "daemon"
        running = True
        healthy = _probe_proxy(c.proxy_port)
    else:
        mode = "none"
        running = False
        healthy = False

    return CredentialProxyStatus(
        mode=mode,
        running=running,
        healthy=healthy,
        socket_path=c.proxy_socket_path,
        db_path=c.proxy_db_path,
        routes_path=c.proxy_routes_path,
        routes_configured=routes_count,
        credentials_stored=creds,
    )


def get_proxy_port(cfg: SandboxConfig | None = None) -> int:
    """Return the configured credential proxy TCP port."""
    return _cfg(cfg).proxy_port


def get_ssh_agent_port(cfg: SandboxConfig | None = None) -> int:
    """Return the configured SSH agent proxy TCP port."""
    return _cfg(cfg).ssh_agent_port


# ---------- Systemd lifecycle ----------


def is_systemd_available() -> bool:
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


def is_socket_installed() -> bool:
    """Check whether the ``terok-credential-proxy.socket`` unit file exists."""
    return (_systemd_unit_dir() / _SOCKET_UNIT).is_file()


def is_socket_active() -> bool:
    """Check whether the ``terok-credential-proxy.socket`` unit is active."""
    try:
        result = subprocess.run(
            ["systemctl", "--user", "is-active", _SOCKET_UNIT],
            capture_output=True,
            text=True,
            timeout=5,
        )
        return result.stdout.strip() == "active"
    except (FileNotFoundError, subprocess.TimeoutExpired):
        return False


def is_service_active() -> bool:
    """Check whether the ``terok-credential-proxy.service`` unit is active.

    Unlike :func:`is_socket_active`, this tells whether the proxy daemon
    itself is running (TCP ports bound), not just whether the socket is
    listening.  Does not trigger socket activation.
    """
    try:
        result = subprocess.run(
            ["systemctl", "--user", "is-active", _SERVICE_UNIT],
            capture_output=True,
            text=True,
            timeout=5,
        )
        return result.stdout.strip() == "active"
    except (FileNotFoundError, subprocess.TimeoutExpired):
        return False


def install_systemd_units(cfg: SandboxConfig | None = None) -> None:
    """Render and install systemd socket+service units, then enable+start the socket."""
    import terok_sandbox.credentials.proxy

    from .._util import render_template

    c = _cfg(cfg)
    unit_dir = _systemd_unit_dir()
    unit_dir.mkdir(parents=True, exist_ok=True)

    resource_dir = (
        Path(terok_sandbox.credentials.proxy.__file__).resolve().parent / "resources" / "systemd"
    )
    variables = {
        "SOCKET_PATH": str(c.proxy_socket_path),
        "DB_PATH": str(c.proxy_db_path),
        "ROUTES_PATH": str(c.proxy_routes_path),
        "PORT": str(c.proxy_port),
        "SSH_AGENT_PORT": str(c.ssh_agent_port),
        "SSH_KEYS_FILE": str(c.ssh_keys_json_path),
        "BIN": shlex.join(_proxy_exec_prefix()),
        "UNIT_VERSION": str(_UNIT_VERSION),
    }

    for template_name in (_SOCKET_UNIT, _SERVICE_UNIT):
        template_path = resource_dir / template_name
        if not template_path.is_file():
            raise SystemExit(f"Missing systemd template: {template_path}")
        content = render_template(template_path, variables)
        (unit_dir / template_name).write_text(content, encoding="utf-8")

    c.proxy_socket_path.parent.mkdir(parents=True, exist_ok=True)
    subprocess.run(["systemctl", "--user", "daemon-reload"], check=True, timeout=10)
    subprocess.run(
        ["systemctl", "--user", "enable", "--now", _SOCKET_UNIT],
        check=True,
        timeout=10,
    )
    # Restart to apply updated unit configuration if socket was already active.
    subprocess.run(
        ["systemctl", "--user", "restart", _SOCKET_UNIT],
        check=True,
        timeout=10,
    )


def uninstall_systemd_units(cfg: SandboxConfig | None = None) -> None:  # noqa: ARG001
    """Disable+stop the socket and remove unit files."""
    unit_dir = _systemd_unit_dir()

    subprocess.run(
        ["systemctl", "--user", "disable", "--now", _SOCKET_UNIT],
        check=False,
        timeout=10,
    )

    for name in (_SOCKET_UNIT, _SERVICE_UNIT):
        unit_file = unit_dir / name
        if unit_file.is_file():
            unit_file.unlink()

    subprocess.run(["systemctl", "--user", "daemon-reload"], check=False, timeout=10)


# ---------- Daemon lifecycle ----------


def start_daemon(cfg: SandboxConfig | None = None) -> None:
    """Start the credential proxy as a background daemon.

    The proxy listens on a Unix socket and reads credentials from a
    sqlite3 database.  A routes JSON file must exist at the configured
    path (generated by terok-executor from the YAML registry).

    Writes a PID file to ``runtime_root() / "credential-proxy.pid"``.
    """
    c = _cfg(cfg)
    sock_path = c.proxy_socket_path
    db_path = c.proxy_db_path
    routes_path = c.proxy_routes_path
    pidfile = _pid_file(cfg)

    sock_path.parent.mkdir(parents=True, exist_ok=True)
    pidfile.parent.mkdir(parents=True, exist_ok=True)

    routes_path.parent.mkdir(parents=True, exist_ok=True)
    try:
        with routes_path.open("x", encoding="utf-8") as f:
            f.write("{}\n")
        import logging

        logging.getLogger(__name__).info(
            "Created empty routes file: %s — populate with: terok auth <provider> <project>",
            routes_path,
        )
    except FileExistsError:
        pass

    ssh_keys_path = c.ssh_keys_json_path
    ssh_keys_path.parent.mkdir(parents=True, exist_ok=True)
    try:
        with ssh_keys_path.open("x", encoding="utf-8") as f:
            f.write("{}\n")
    except FileExistsError:
        pass

    log_file = c.state_dir / "proxy" / "credential-proxy.log"
    log_file.parent.mkdir(parents=True, exist_ok=True)

    cmd = [
        *_proxy_exec_prefix(),
        f"--socket-path={sock_path}",
        f"--db-path={db_path}",
        f"--routes-file={routes_path}",
        f"--pid-file={pidfile}",
        f"--port={c.proxy_port}",
        f"--ssh-agent-port={c.ssh_agent_port}",
        f"--ssh-keys-file={ssh_keys_path}",
        f"--log-file={log_file}",
        "--log-level=DEBUG",
    ]

    # Fork into background so the proxy survives shell exit.
    # The server writes its own PID file via --pid-file.
    # stderr=PIPE only for the startup-failure detection window; the pipe is
    # closed immediately after so the daemon's stderr does not block on a full buffer.
    proc = subprocess.Popen(
        cmd,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.PIPE,
        start_new_session=True,
    )

    # Poll the /-/health endpoint until the server is actually ready.
    if _wait_for_ready(c.proxy_port):
        # Close our end of the pipe — the daemon logs to the log file, not stderr.
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
        f"Check logs or try: curl http://127.0.0.1:{c.proxy_port}{_HEALTH_PATH}"
    )


def stop_daemon(cfg: SandboxConfig | None = None) -> None:
    """Stop the managed proxy daemon by sending SIGTERM."""
    pidfile = _pid_file(cfg)
    if not pidfile.is_file():
        return
    try:
        pid = int(pidfile.read_text().strip())
        if _is_managed_proxy(pid, cfg):
            os.kill(pid, signal.SIGTERM)
    except (ValueError, ProcessLookupError, PermissionError):
        pass
    finally:
        if pidfile.is_file():
            pidfile.unlink()


def is_daemon_running(cfg: SandboxConfig | None = None) -> bool:
    """Check whether the managed proxy daemon is alive via its PID file."""
    pidfile = _pid_file(cfg)
    if not pidfile.is_file():
        return False
    try:
        pid = int(pidfile.read_text().strip())
        if not _is_managed_proxy(pid, cfg):
            return False
        os.kill(pid, 0)  # signal 0 = existence check
        return True
    except (ValueError, ProcessLookupError, PermissionError):
        return False


# ---------- Private helpers ----------


def _cfg(cfg: SandboxConfig | None = None) -> SandboxConfig:
    """Return *cfg* or a default :class:`SandboxConfig`."""
    return cfg or SandboxConfig()


_UNIT_VERSION = 4
"""Bump when the systemd unit templates change."""

_SOCKET_UNIT = "terok-credential-proxy.socket"
"""Name of the systemd socket unit file."""

_SERVICE_UNIT = "terok-credential-proxy.service"
"""Name of the systemd service unit file."""


def _pid_file(cfg: SandboxConfig | None = None) -> Path:
    """Return the PID file path for the managed proxy daemon."""
    return _cfg(cfg).proxy_pid_file_path


def _is_managed_proxy(pid: int, cfg: SandboxConfig | None = None) -> bool:
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
    expected = f"--pid-file={_pid_file(cfg)}"
    return expected in args_str


def _systemd_unit_dir() -> Path:
    """Return the systemd user unit directory."""
    xdg = os.environ.get("XDG_CONFIG_HOME")
    return (Path(xdg) if xdg else Path.home() / ".config") / "systemd" / "user"


def _proxy_exec_prefix() -> list[str]:
    """Return the command prefix for launching the credential proxy server.

    Uses ``sys.executable -m terok_sandbox.credentials.proxy`` so the
    server runs under the same Python that owns the installed package —
    works in pipx, venvs, and bare installs without requiring the
    ``terok-credential-proxy`` console script on ``$PATH``.  That entry
    point (defined in pyproject.toml) remains available for direct CLI
    use by standalone sandbox users.
    """
    import sys as _sys

    return [_sys.executable, "-m", "terok_sandbox.credentials.proxy"]


def _probe_proxy(port: int, *, timeout: float = 2.0) -> bool:
    """Return ``True`` if the proxy's health endpoint responds 200.

    Uses :mod:`http.client` (stdlib only) to hit the TCP port.
    ``http.client.HTTPConnection`` is restricted to HTTP — no ``file://``
    or custom-scheme risk, so no Bandit B310 suppression needed.
    """
    import http.client

    conn: http.client.HTTPConnection | None = None
    try:
        conn = http.client.HTTPConnection("127.0.0.1", port, timeout=timeout)
        conn.request("GET", _HEALTH_PATH)
        resp = conn.getresponse()
        resp.read()  # drain body before connection reuse/close
        return resp.status == 200
    except (OSError, http.client.HTTPException, ValueError):
        return False
    finally:
        if conn is not None:
            conn.close()


def _wait_for_ready(port: int, *, timeout: float = 5.0, interval: float = 0.2) -> bool:
    """Poll the health endpoint until it responds 200 or *timeout* expires.

    Returns ``True`` when the proxy is ready, ``False`` on timeout.
    """
    import time

    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        if _probe_proxy(port, timeout=min(1.0, interval)):
            return True
        time.sleep(interval)
    return False


def _wait_for_tcp_port(port: int, timeout: float = 5.0) -> bool:
    """Wait up to *timeout* seconds for a TCP port on localhost to accept connections."""
    import socket
    import time

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
