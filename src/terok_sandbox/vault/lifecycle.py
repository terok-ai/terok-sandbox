# SPDX-FileCopyrightText: 2026 Jiri Vyskocil
# SPDX-License-Identifier: Apache-2.0

"""Vault lifecycle management.

Manages the ``terok-vault`` daemon: start, stop, status, and
pre-task health checks.  Supports systemd socket activation (preferred)
and a manual daemon fallback.

The systemd socket unit listens on both the Unix socket and the TCP
port used by containers.  A connection to either triggers the service.
:meth:`VaultManager.ensure_reachable` also performs an explicit
start as a belt-and-suspenders measure before task creation.
"""

from __future__ import annotations

import os
import shlex
import signal
import subprocess
import sys
import time
from dataclasses import dataclass
from pathlib import Path

from .._util import _systemctl
from .._util._logging import log_warning
from ..config import SandboxConfig
from .constants import HEALTH_PATH as _HEALTH_PATH

# ---------- Vocabulary ----------


@dataclass(frozen=True)
class VaultStatus:
    """Current state of the vault."""

    mode: str
    """``"systemd"``, ``"daemon"``, or ``"none"``."""

    running: bool
    """Whether the vault is active (systemd socket listening or daemon alive)."""

    healthy: bool
    """Whether the vault is healthy for its current activation mode.

    HTTP-probe based when the systemd service is active; socket-liveness
    based when the service is idle but the socket is listening.
    """

    socket_path: Path
    """Configured Unix socket path."""

    db_path: Path
    """Configured credential database path."""

    routes_path: Path
    """Configured routes JSON path."""

    routes_configured: int
    """Number of routes in routes.json (0 if missing or invalid)."""

    credentials_stored: tuple[str, ...]
    """Provider names with stored credentials."""

    transport: str | None = None
    """Detected transport: ``"tcp"``, ``"socket"``, or ``None`` if not running."""


class VaultUnreachableError(RuntimeError):
    """Raised when the vault is not reachable.

    Carries diagnostic paths so CLI layers can append their own
    remediation hints (specific command names vary by package).
    """

    def __init__(self, *, socket_path: Path, db_path: Path) -> None:
        self.socket_path = socket_path
        self.db_path = db_path
        super().__init__(
            "Vault is not reachable.\n"
            "\n"
            "The vault injects real API credentials into container\n"
            "requests without exposing secrets to the container filesystem.\n"
            "\n"
            "Start the vault (socket activation or manual daemon)\n"
            "before creating tasks.\n"
            "\n"
            f"Socket: {socket_path}\n"
            f"DB:     {db_path}"
        )


# ---------- Constants ----------

_UNIT_VERSION = 5
"""Bump when the systemd unit templates change."""

_SOCKET_UNIT = "terok-vault.socket"
"""Name of the systemd socket unit file (TCP mode)."""

_SERVICE_UNIT = "terok-vault.service"
"""Name of the systemd service unit file (TCP mode)."""

_SOCKET_MODE_SERVICE = "terok-vault-socket.service"
"""Name of the systemd service unit for Unix socket mode."""

_ALL_UNIT_NAMES = (_SOCKET_UNIT, _SERVICE_UNIT, _SOCKET_MODE_SERVICE)
"""All unit file names across both transport modes (for cleanup)."""

_OWNED_UNIT_GLOB = "terok-vault*"
"""Glob pattern matching every name this package has ever installed.

Intentionally broader than ``_ALL_UNIT_NAMES`` so the orphan sweep can
find units from prior versions with different filenames.  Ownership is
determined by the ``# terok-vault-version:`` marker inside
the file, not by the glob match — a user-authored
``terok-vault-extra.service`` without the marker survives
untouched.
"""

_OWNED_MARKER_PREFIX = "# terok-vault-version:"
"""First-line marker every shipped vault unit template carries.

The orphan sweep uses this as the ownership check: only files whose
first line begins with this string were written by this package and
are safe to remove when their names no longer match the current set.
"""


# ---------- Manager ----------


class VaultManager:
    """Lifecycle manager for the terok vault.

    Encapsulates configuration, systemd unit management, daemon process
    control, and health probing behind a single object.  Construct with
    an optional :class:`SandboxConfig`; all methods use the bound
    configuration.
    """

    def __init__(self, cfg: SandboxConfig | None = None) -> None:
        self._cfg = cfg or SandboxConfig()

    # -- Public API ----------------------------------------------------------

    def ensure_reachable(self) -> None:
        """Verify the vault is running and its TCP ports are up.

        For **systemd** socket activation the service may not have started yet
        (e.g. after a fresh boot).  This function triggers a start via
        ``systemctl --user start`` and waits for the HTTP and SSH signer TCP
        ports to become reachable via ``/-/health`` and raw TCP probes.

        Raises :class:`VaultUnreachableError` if the vault is unreachable.
        Called before task creation when vault is enabled.
        """
        if not self.is_socket_active() and not self.is_daemon_running():
            raise VaultUnreachableError(
                socket_path=self._cfg.vault_socket_path,
                db_path=self._cfg.db_path,
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

        # Prefer Unix socket probe (works in both transport modes — the
        # vault binds vault.sock regardless of whether TCP ports are bound).
        if self._wait_for_unix_socket(self._cfg.vault_socket_path):
            return

        # Fallback: TCP health probe + SSH signer port (TCP mode only).
        if not self._wait_for_ready(self._cfg.token_broker_port):
            raise SystemExit(
                f"Vault service started but token-broker TCP port "
                f"{self._cfg.token_broker_port} is not reachable."
            )

        if not self._wait_for_tcp_port(self._cfg.ssh_signer_port):
            raise SystemExit(
                f"Vault service started but SSH signer TCP port "
                f"{self._cfg.ssh_signer_port} is not reachable."
            )

    def get_status(self) -> VaultStatus:
        """Return the current vault status.

        Populates route count from the routes JSON (0 if missing/invalid) and
        credential provider names from the database (empty if DB doesn't exist).
        """
        routes_count = 0
        if self._cfg.routes_path.is_file():
            try:
                import json

                routes_count = len(json.loads(self._cfg.routes_path.read_text()))
            except (json.JSONDecodeError, OSError):
                pass

        creds: tuple[str, ...] = ()
        if self._cfg.db_path.is_file():
            try:
                from ..credentials.db import CredentialDB

                db = CredentialDB(self._cfg.db_path)
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
            healthy = self._probe(self._cfg.token_broker_port) if service_up else socket_up
        elif self.is_daemon_running():
            mode = "daemon"
            running = True
            healthy = self._probe(self._cfg.token_broker_port)
        else:
            mode = "none"
            running = False
            healthy = False

        # Derive transport from installed unit type (not reachability probe,
        # since TCP mode also binds a Unix socket).
        transport = self._installed_transport() if mode == "systemd" else None

        return VaultStatus(
            mode=mode,
            running=running,
            healthy=healthy,
            socket_path=self._cfg.vault_socket_path,
            db_path=self._cfg.db_path,
            routes_path=self._cfg.routes_path,
            routes_configured=routes_count,
            credentials_stored=creds,
            transport=transport,
        )

    @property
    def token_broker_port(self) -> int:
        """Return the configured vault token broker TCP port."""
        return self._cfg.token_broker_port

    @property
    def ssh_signer_port(self) -> int:
        """Return the configured vault SSH signer TCP port."""
        return self._cfg.ssh_signer_port

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
        """Check whether any vault systemd unit file exists (TCP or socket mode)."""
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
        """Check whether the vault daemon itself is running.

        Checks both TCP-mode service and socket-mode service units.
        Unlike :meth:`is_socket_active`, this tells whether the vault
        daemon itself is bound (TCP ports bound), not just whether the
        socket is listening.  Does not trigger socket activation.
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
            self._cfg.token_broker_port is None or self._cfg.ssh_signer_port is None
        ):
            raise SystemExit(
                "Cannot install tcp-mode vault units: no port is set.\n"
                "Either configure ``services.mode: tcp`` (auto-allocates ports)\n"
                "or pin ``vault.token_broker_port`` / ``vault.ssh_signer_port`` explicitly."
            )

        import terok_sandbox.vault

        from .._util import render_template

        unit_dir = self._systemd_unit_dir()
        unit_dir.mkdir(parents=True, exist_ok=True)

        resource_dir = Path(terok_sandbox.vault.__file__).resolve().parent / "resources" / "systemd"
        variables = {
            "SOCKET_PATH": str(self._cfg.vault_socket_path),
            "SSH_SIGNER_SOCKET_PATH": str(self._cfg.ssh_signer_socket_path),
            "DB_PATH": str(self._cfg.db_path),
            "ROUTES_PATH": str(self._cfg.routes_path),
            "PORT": str(self._cfg.token_broker_port),
            "SSH_SIGNER_PORT": str(self._cfg.ssh_signer_port),
            "BIN": shlex.join(self._vault_exec_prefix()),
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

        self._cfg.vault_socket_path.parent.mkdir(parents=True, exist_ok=True)
        # Capture the "Created symlink ..." notice systemd prints to stderr —
        # it interleaves into the caller's progress output otherwise.  Any
        # failure is surfaced with stderr attached via _systemctl.run.
        _systemctl.run("daemon-reload")
        _systemctl.run("enable", "--now", enable_unit)
        # Restart to apply updated unit configuration if socket was already active.
        _systemctl.run("restart", enable_unit)

    def _stop_all_units(self) -> None:
        """Stop and disable all proxy units across both transport modes."""
        unit_dir = self._systemd_unit_dir()
        for unit in (_SOCKET_UNIT, _SERVICE_UNIT, _SOCKET_MODE_SERVICE):
            if (unit_dir / unit).is_file():
                _systemctl.run_best_effort("disable", "--now", unit)

    def _remove_unit_files(self) -> None:
        """Stop active units, sweep orphans from prior versions, remove current ones."""
        self._stop_all_units()
        self._sweep_orphan_units()
        unit_dir = self._systemd_unit_dir()
        for name in _ALL_UNIT_NAMES:
            unit_file = unit_dir / name
            if unit_file.is_file():
                unit_file.unlink()

    def _sweep_orphan_units(self) -> None:
        """Disable + remove proxy unit files from prior versions.

        A unit file is considered ours (and eligible for removal) when
        its first line begins with :data:`_OWNED_MARKER_PREFIX` — every
        shipped template carries that marker.  Files matching the
        :data:`_OWNED_UNIT_GLOB` but lacking the marker are left alone
        (user-authored units that happen to share the naming prefix).
        Current-version files are skipped here and handled by the
        subsequent pass in :meth:`_remove_unit_files`.

        This catches legacy filenames from previous releases — e.g. if
        a future rename changes a unit's filename, the sweep cleans up
        the old one on the next ``terok setup`` without a manual
        uninstall step.
        """
        unit_dir = self._systemd_unit_dir()
        if not unit_dir.is_dir():
            return
        for candidate in unit_dir.glob(_OWNED_UNIT_GLOB):
            if candidate.name in _ALL_UNIT_NAMES or not candidate.is_file():
                continue
            try:
                first_line = candidate.read_text(encoding="utf-8").splitlines()[0]
            except (OSError, IndexError, UnicodeDecodeError):
                continue
            if not first_line.startswith(_OWNED_MARKER_PREFIX):
                continue
            _systemctl.run_best_effort("disable", "--now", candidate.name)
            candidate.unlink(missing_ok=True)

    def uninstall_systemd_units(self) -> None:
        """Disable+stop all proxy units and remove unit files."""
        self._remove_unit_files()
        _systemctl.run_best_effort("daemon-reload")

    # -- Daemon lifecycle ----------------------------------------------------

    def start_daemon(self) -> None:
        """Start the vault as a background daemon.

        The vault listens on a Unix socket and reads credentials from a
        sqlite3 database.  A routes JSON file must exist at the configured
        path (generated by terok-executor from the YAML registry).

        Writes a PID file to ``runtime_root() / "vault.pid"``.
        """
        sock_path = self._cfg.vault_socket_path
        db_path = self._cfg.db_path
        routes_path = self._cfg.routes_path
        pidfile = self._cfg.vault_pid_path

        sock_path.parent.mkdir(parents=True, exist_ok=True)
        pidfile.parent.mkdir(parents=True, exist_ok=True)

        from .._util import write_sensitive_file

        if write_sensitive_file(routes_path, "{}\n"):
            import logging

            logging.getLogger(__name__).info(
                "Created empty routes file: %s — populate with: terok auth <provider> <project>",
                routes_path,
            )

        log_file = self._cfg.state_dir / "vault" / "vault.log"
        log_file.parent.mkdir(parents=True, exist_ok=True)
        os.chmod(log_file.parent, 0o700)

        log_level = os.environ.get("TEROK_VAULT_LOG_LEVEL", "INFO")
        cmd = [
            *self._vault_exec_prefix(),
            f"--socket-path={sock_path}",
            f"--db-path={db_path}",
            f"--routes-file={routes_path}",
            f"--pid-file={pidfile}",
            f"--log-file={log_file}",
            f"--log-level={log_level}",
        ]
        # Transport-specific wiring: in socket mode the token-broker has no
        # TCP port and the SSH signer listens on a Unix socket instead.
        if self._cfg.token_broker_port is not None:
            cmd.append(f"--port={self._cfg.token_broker_port}")
        if self._cfg.ssh_signer_port is not None:
            cmd.append(f"--ssh-signer-port={self._cfg.ssh_signer_port}")
        else:
            cmd.append(f"--ssh-signer-socket-path={self._cfg.ssh_signer_socket_path}")

        # Under Nix (and other setups where ``sys.executable`` is a wrapper
        # that normally rewrites the env on startup) spawning ``python -m
        # terok_sandbox.vault`` from a running terok_sandbox process bypasses
        # that wrapper, and the vault daemon can't find its own package on
        # the import path.  Passing the parent's ``sys.path`` through as
        # ``PYTHONPATH`` lets the subprocess resolve the same install this
        # process is running from.  See terok-ai/terok-shield#242 by
        # Franz Pöschel — same fix pattern, different spawn site.
        env = {**os.environ, "PYTHONPATH": os.pathsep.join(sys.path)}

        # Fork into background so the vault survives shell exit.
        # stderr=PIPE only for the startup-failure detection window.
        proc = subprocess.Popen(
            cmd,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.PIPE,
            start_new_session=True,
            env=env,
        )

        broker_ok, signer_ok, broker_detail, signer_detail = self._wait_for_daemon_ready()
        if broker_ok and signer_ok:
            proc.stderr.close()
            return

        # Timed out — check whether the process crashed or is just slow.
        ret = proc.poll()
        if ret is not None:
            stderr = (proc.stderr.read() or b"").decode(errors="replace").strip()
            msg = f"Vault failed to start (exit {ret})"
            if stderr:
                msg += f":\n{stderr}"
            raise SystemExit(msg)
        proc.stderr.close()
        if not broker_ok:
            raise SystemExit(
                f"Vault process started but token-broker "
                f"{broker_detail} did not become ready within 5 s."
            )
        raise SystemExit(
            f"Vault process started but SSH signer {signer_detail} did not become ready within 5 s."
        )

    def _wait_for_daemon_ready(self) -> tuple[bool, bool, str, str]:
        """Poll broker + signer readiness; return ``(broker_ok, signer_ok, broker_detail, signer_detail)``.

        In socket mode both probes hit Unix sockets; in TCP mode the broker
        is probed via its health endpoint and the signer via a raw TCP
        connect.  The ``*_detail`` strings describe the probed endpoint for
        human-readable timeout errors.
        """
        if self._cfg.token_broker_port is None:
            broker_sock = self._cfg.vault_socket_path
            signer_sock = self._cfg.ssh_signer_socket_path
            broker_ok = self._wait_for_unix_socket(broker_sock)
            signer_ok = broker_ok and self._wait_for_unix_socket(signer_sock)
            return broker_ok, signer_ok, f"socket {broker_sock}", f"socket {signer_sock}"

        broker_ok = self._wait_for_ready(self._cfg.token_broker_port)
        signer_ok = broker_ok and self._wait_for_tcp_port(self._cfg.ssh_signer_port)
        return (
            broker_ok,
            signer_ok,
            f"port {self._cfg.token_broker_port}",
            f"port {self._cfg.ssh_signer_port}",
        )

    def stop_daemon(self) -> None:
        """Stop the vault, whether running as a systemd unit or a PID-file daemon.

        Both paths are attempted unconditionally: systemd-managed vaults
        live under ``terok-vault-socket.service`` / ``terok-vault.service``
        and have no PID file, while a manually-started daemon has a PID
        file but no active unit.  Running both paths also sweeps stray
        daemons that outlived their systemd unit.
        """
        # ``_systemctl.run_best_effort`` swallows ``TimeoutExpired`` so a
        # wedged unit can't block the PID-file path below.
        for unit in (_SOCKET_UNIT, _SERVICE_UNIT, _SOCKET_MODE_SERVICE):
            if self._is_unit_active(unit):
                _systemctl.run_best_effort("stop", unit)
        pidfile = self._cfg.vault_pid_path
        if not pidfile.is_file():
            return
        try:
            pid = int(pidfile.read_text().strip())
            if self._is_managed_vault(pid):
                os.kill(pid, signal.SIGTERM)
        except (ValueError, ProcessLookupError, PermissionError):
            pass
        finally:
            if pidfile.is_file():
                pidfile.unlink()

    def is_daemon_running(self) -> bool:
        """Check whether the managed vault daemon is alive via its PID file."""
        pidfile = self._cfg.vault_pid_path
        if not pidfile.is_file():
            return False
        try:
            pid = int(pidfile.read_text().strip())
            if not self._is_managed_vault(pid):
                return False
            os.kill(pid, 0)  # signal 0 = existence check
            return True
        except (ValueError, ProcessLookupError, PermissionError):
            return False

    # -- Private helpers -----------------------------------------------------

    def _is_managed_vault(self, pid: int) -> bool:
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
        expected = f"--pid-file={self._cfg.vault_pid_path}"
        return expected in args_str

    @staticmethod
    def _systemd_unit_dir() -> Path:
        """Return the validated systemd user unit directory."""
        from .._util import systemd_user_unit_dir

        return systemd_user_unit_dir()

    @staticmethod
    def _vault_exec_prefix() -> list[str]:
        """Return the command prefix for launching the vault server.

        Uses ``sys.executable -m terok_sandbox.vault`` so the
        server runs under the same Python that owns the installed package.
        """
        import sys as _sys

        return [_sys.executable, "-m", "terok_sandbox.vault"]

    @staticmethod
    def _probe(port: int, *, timeout: float = 2.0) -> bool:
        """Return ``True`` if the vault's health endpoint responds 200.

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
            if VaultManager._probe(port, timeout=probe_timeout):
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
