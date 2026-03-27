# SPDX-FileCopyrightText: 2026 Jiri Vyskocil
# SPDX-License-Identifier: Apache-2.0

"""Credential proxy lifecycle management.

Manages the ``terok-credential-proxy`` daemon: start, stop, status, and
pre-task health checks.  Supports systemd socket activation (preferred)
and a manual daemon fallback.

**No auto-start.**  Task creation checks reachability via
:func:`ensure_proxy_reachable` and fails with an actionable error
rather than silently starting a daemon.
"""

from __future__ import annotations

import os
import shlex
import signal
import subprocess
from dataclasses import dataclass
from pathlib import Path

from .config import SandboxConfig


def _cfg(cfg: SandboxConfig | None = None) -> SandboxConfig:
    """Return *cfg* or a default :class:`SandboxConfig`."""
    return cfg or SandboxConfig()


# ---------- Data classes ----------


@dataclass(frozen=True)
class CredentialProxyStatus:
    """Current state of the credential proxy."""

    mode: str
    """``"systemd"``, ``"daemon"``, or ``"none"``."""

    running: bool
    """Whether the proxy is active (systemd socket listening or daemon alive)."""

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


# ---------- PID file helpers ----------


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


# ---------- Systemd helpers ----------

_UNIT_VERSION = 1
"""Bump when the systemd unit templates change."""

_SOCKET_UNIT = "terok-credential-proxy.socket"
"""Name of the systemd socket unit file."""

_SERVICE_UNIT = "terok-credential-proxy.service"
"""Name of the systemd service unit file."""


def _systemd_unit_dir() -> Path:
    """Return the systemd user unit directory."""
    xdg = os.environ.get("XDG_CONFIG_HOME")
    return (Path(xdg) if xdg else Path.home() / ".config") / "systemd" / "user"


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


def _proxy_exec_prefix() -> list[str]:
    """Return the command prefix for launching the credential proxy server.

    Uses ``sys.executable -m terok_sandbox.credential_proxy`` so the
    server runs under the same Python that owns the installed package —
    works in pipx, venvs, and bare installs without requiring the
    ``terok-credential-proxy`` console script on ``$PATH``.  That entry
    point (defined in pyproject.toml) remains available for direct CLI
    use by standalone sandbox users.
    """
    import sys as _sys

    return [_sys.executable, "-m", "terok_sandbox.credential_proxy"]


def install_systemd_units(cfg: SandboxConfig | None = None) -> None:
    """Render and install systemd socket+service units, then enable+start the socket."""
    import terok_sandbox.credential_proxy

    from ._util import render_template

    c = _cfg(cfg)
    unit_dir = _systemd_unit_dir()
    unit_dir.mkdir(parents=True, exist_ok=True)

    resource_dir = (
        Path(terok_sandbox.credential_proxy.__file__).resolve().parent / "resources" / "systemd"
    )
    variables = {
        "SOCKET_PATH": str(c.proxy_socket_path),
        "DB_PATH": str(c.proxy_db_path),
        "ROUTES_PATH": str(c.proxy_routes_path),
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


def uninstall_systemd_units() -> None:
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


# ---------- Public API ----------


def start_daemon(cfg: SandboxConfig | None = None) -> None:
    """Start the credential proxy as a background daemon.

    The proxy listens on a Unix socket and reads credentials from a
    sqlite3 database.  A routes JSON file must exist at the configured
    path (generated by terok-agent from the YAML registry).

    Writes a PID file to ``runtime_root() / "credential-proxy.pid"``.
    """
    c = _cfg(cfg)
    sock_path = c.proxy_socket_path
    db_path = c.proxy_db_path
    routes_path = c.proxy_routes_path
    pidfile = _pid_file(cfg)

    sock_path.parent.mkdir(parents=True, exist_ok=True)
    pidfile.parent.mkdir(parents=True, exist_ok=True)

    if not routes_path.is_file():
        import logging

        routes_path.parent.mkdir(parents=True, exist_ok=True)
        routes_path.write_text("{}\n")
        logging.getLogger(__name__).info(
            "Created empty routes file: %s — add routes via 'terokctl auth <provider>'",
            routes_path,
        )

    cmd = [
        *_proxy_exec_prefix(),
        f"--socket-path={sock_path}",
        f"--db-path={db_path}",
        f"--routes-file={routes_path}",
        f"--pid-file={pidfile}",
    ]

    # Fork into background so the proxy survives shell exit.
    # The server writes its own PID file via --pid-file.
    import time

    proc = subprocess.Popen(
        cmd,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.PIPE,
        start_new_session=True,
    )

    # Brief wait to catch immediate startup failures (bad args, missing deps)
    time.sleep(0.3)
    ret = proc.poll()
    if ret is not None:
        stderr = (proc.stderr.read() or b"").decode(errors="replace").strip()
        msg = f"Credential proxy failed to start (exit {ret})"
        if stderr:
            msg += f":\n{stderr}"
        raise SystemExit(msg)


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
            from .credential_db import CredentialDB

            db = CredentialDB(c.proxy_db_path)
            try:
                creds = tuple(db.list_credentials("default"))
            finally:
                db.close()
        except Exception:  # noqa: BLE001
            pass

    # Systemd takes precedence: when units are installed, report mode="systemd"
    # even if the socket is inactive — the daemon's running state is ignored so
    # operators see the correct activation path and don't get mixed signals.
    if is_socket_installed():
        mode = "systemd"
        running = is_socket_active()
    elif is_daemon_running(cfg):
        mode = "daemon"
        running = True
    else:
        mode = "none"
        running = False

    return CredentialProxyStatus(
        mode=mode,
        running=running,
        socket_path=c.proxy_socket_path,
        db_path=c.proxy_db_path,
        routes_path=c.proxy_routes_path,
        routes_configured=routes_count,
        credentials_stored=creds,
    )


def ensure_proxy_reachable(cfg: SandboxConfig | None = None) -> None:
    """Verify the credential proxy is running.

    Raises ``SystemExit`` with an actionable message if the proxy is down.
    Called before task creation when credential proxy is enabled.
    """
    if is_socket_active() or is_daemon_running(cfg):
        return

    c = _cfg(cfg)
    hint = (
        "  terokctl credentials install   (systemd socket activation)\n"
        "  terokctl credentials start      (manual daemon)"
    )
    msg = (
        "Credential proxy is not running.\n"
        "\n"
        "The credential proxy injects real API credentials into container\n"
        "requests without exposing secrets to the container filesystem.\n"
        "\n"
        f"Start it with:\n{hint}\n"
        f"\n"
        f"Socket: {c.proxy_socket_path}\n"
        f"DB:     {c.proxy_db_path}\n"
    )
    raise SystemExit(msg)
