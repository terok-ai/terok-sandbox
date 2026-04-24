# SPDX-FileCopyrightText: 2026 Jiri Vyskocil
# SPDX-License-Identifier: Apache-2.0

"""Gate server lifecycle management.

Manages the ``terok-gate`` HTTP server that wraps ``git http-backend`` with
per-task token authentication.  Containers reach the gate via ``http://`` URLs
through ``host.containers.internal`` — standard HTTP protocol, no bind-mount
escape vector.

Networking across Podman versions:

- **terok-shield** handles container networking via its OCI hook.  The gate
  server port is passed as ``loopback_ports`` in :class:`ShieldConfig` so that
  shield's nftables rules allow containers to reach host loopback on that port.

**Deployment modes (ordered by preference):**

1. **Systemd socket activation** — zero-idle-cost, crash resilience, and no
   PID management.  Recommended for any Linux host with ``systemctl --user``.

2. **Managed ``terok-gate`` daemon process** — best-effort fallback.  Works
   correctly but has a simpler lifecycle (manual start/stop, PID file).

**No auto-start.**  Task creation checks reachability and fails with an
actionable error rather than silently starting a daemon.
"""

from __future__ import annotations

import os
import signal
import subprocess
from dataclasses import dataclass
from pathlib import Path

from .._util import _systemctl
from ..config import SandboxConfig

# ---------- Constants ----------

_UNIT_VERSION = 7
"""Bump when the systemd unit templates change.  ``ensure_reachable``
checks the installed version and refuses to start tasks if it is stale."""

_SOCKET_UNIT = "terok-gate.socket"
"""Name of the systemd socket unit file (TCP inetd mode)."""

_SOCKET_MODE_SERVICE = "terok-gate-socket.service"
"""Name of the systemd service unit for Unix socket mode."""

_ALL_UNIT_NAMES = (_SOCKET_UNIT, "terok-gate@.service", _SOCKET_MODE_SERVICE)
"""All unit file names across both transport modes (for cleanup)."""

_OWNED_UNIT_GLOB = "terok-gate*"
"""Glob pattern matching every name this package has ever installed.

Intentionally broader than ``_ALL_UNIT_NAMES`` so the orphan sweep can
find units from prior versions with different filenames.  Ownership is
determined by the ``# terok-gate-version:`` marker inside the file, not
by the glob match — a user-authored ``terok-gateway.service`` without
the marker survives untouched.
"""

_OWNED_MARKER_PREFIX = "# terok-gate-version:"
"""First-line marker every shipped gate unit template carries.

The orphan sweep uses this as the ownership check: only files whose
first line begins with this string were written by this package and
are safe to remove when their names no longer match the current set.
"""


# ---------- Vocabulary ----------


@dataclass(frozen=True)
class GateServerStatus:
    """Current state of the gate server."""

    mode: str
    """``"systemd"``, ``"daemon"``, or ``"none"``."""

    running: bool
    """Whether the server is currently reachable."""

    port: int | None
    """Configured TCP port, or ``None`` in socket-transport mode."""

    transport: str | None = None
    """Detected transport: ``"tcp"``, ``"socket"``, or ``None`` if not running."""


# ---------- Manager ----------


class GateServerManager:
    """Lifecycle manager for the terok-gate HTTP server.

    Encapsulates configuration, systemd unit management, and daemon
    process control behind a single object.  Construct with an optional
    :class:`SandboxConfig`; all methods use the bound configuration.
    """

    def __init__(self, cfg: SandboxConfig | None = None) -> None:
        self._cfg = cfg or SandboxConfig()

    # -- Public API ----------------------------------------------------------

    def ensure_reachable(self) -> None:
        """Verify the gate server is running and configured correctly.

        Raises ``SystemExit`` if the server is down, systemd units are outdated,
        or the installed base path diverges from the current configuration.
        Called before task creation to fail early with an actionable message.
        """
        server_status = self.get_status()
        if server_status.running:
            if server_status.mode == "systemd":
                installed = self._installed_unit_version()
                if installed is None or installed < _UNIT_VERSION:
                    installed_label = "unversioned" if installed is None else f"v{installed}"
                    raise SystemExit(
                        "Gate server systemd units are outdated "
                        f"(installed {installed_label}, expected v{_UNIT_VERSION})."
                    )
                path_warning = self._base_path_diverged()
                if path_warning:
                    raise SystemExit(path_warning)
            return

        msg = (
            "Gate server is not running.\n"
            "\n"
            "The gate server serves git repos to task containers over the network,\n"
            "replacing the previous volume-mount approach.\n"
            "\n"
        )
        if self.is_systemd_available():
            msg += "Recommended: install and start the systemd socket.\n"
        else:
            msg += "Start the gate daemon.\n"
        raise SystemExit(msg)

    def is_socket_reachable(self) -> bool:
        """Check whether the gate Unix socket accepts connections."""
        from .._util._net import probe_unix_socket

        return probe_unix_socket(self._cfg.gate_socket_path)

    def _detect_transport(self) -> str | None:
        """Detect the active transport: ``"socket"``, ``"tcp"``, or ``None``."""
        if self.is_socket_reachable():
            return "socket"
        if self.is_daemon_running():
            return "tcp"
        return None

    def get_status(self) -> GateServerStatus:
        """Return the current gate server status."""
        port = self._cfg.gate_port
        transport = self._detect_transport()

        if self.is_socket_installed():
            if self.is_socket_active():
                return GateServerStatus(
                    mode="systemd", running=True, port=port, transport=transport or "tcp"
                )
            if transport:
                return GateServerStatus(mode="daemon", running=True, port=port, transport=transport)
            return GateServerStatus(mode="systemd", running=False, port=port)

        if transport:
            return GateServerStatus(mode="daemon", running=True, port=port, transport=transport)

        return GateServerStatus(mode="none", running=False, port=port)

    def check_units_outdated(self) -> str | None:
        """Return a warning string if installed systemd units are stale, else ``None``.

        Checks both the unit version stamp and the baked ``--base-path`` against
        the current configuration.  Useful for ``gate-server status`` and
        ``sickbay`` to surface upgrade hints without blocking task creation
        (that's ``ensure_reachable``'s job).
        """
        if not self.is_socket_installed():
            return None
        installed = self._installed_unit_version()
        if installed is None or installed < _UNIT_VERSION:
            installed_label = "unversioned" if installed is None else f"v{installed}"
            return (
                f"Systemd units are outdated "
                f"(installed {installed_label}, expected v{_UNIT_VERSION})."
            )
        return self._base_path_diverged()

    @property
    def gate_base_path(self) -> Path:
        """Return the gate base path."""
        return self._cfg.gate_base_path

    @property
    def server_port(self) -> int | None:
        """Return the configured gate server TCP port, or ``None`` in socket mode."""
        return self._cfg.gate_port

    # -- Systemd lifecycle ---------------------------------------------------

    def is_systemd_available(self) -> bool:
        """Check whether ``systemctl --user`` is usable.

        Uses ``is-system-running`` which returns well-defined exit codes:
        0 = running, 1 = degraded/starting/stopping — both mean systemd is
        present.  Any other code (or missing binary) means unavailable.
        """
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
        """Check whether any gate systemd unit file exists (TCP or socket mode)."""
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

    def install_systemd_units(self, *, transport: str = "tcp") -> None:
        """Render and install systemd units, then enable+start.

        When *transport* is ``"tcp"`` (default), installs the inetd-style
        socket+service pair.  When ``"socket"``, installs a single long-running
        service that binds a Unix socket.
        """
        import shutil

        import terok_sandbox.gate

        from .._util import render_template
        from .tokens import TokenStore

        # A TCP install with no resolved port would render ``ListenStream=
        # 127.0.0.1:None`` and systemd would reject the unit.  Fail now,
        # naming the config knobs that resolve it, rather than emit a
        # broken unit file.
        if transport == "tcp" and self._cfg.gate_port is None:
            raise SystemExit(
                "Cannot install tcp-mode gate units: no gate port is set.\n"
                "Either configure ``services.mode: tcp`` (auto-allocates a port)\n"
                "or pin ``gate_server.port`` explicitly in config.yml."
            )

        gate_bin = shutil.which("terok-gate")
        if not gate_bin:
            raise SystemExit(
                "Cannot find 'terok-gate' on PATH.\n"
                "Ensure terok-sandbox is installed (pip/pipx/poetry) and the binary is accessible."
            )

        unit_dir = self._systemd_unit_dir()
        unit_dir.mkdir(parents=True, exist_ok=True)

        resource_dir = Path(terok_sandbox.gate.__file__).resolve().parent / "resources" / "systemd"
        variables = {
            "PORT": str(self._cfg.gate_port),
            "SOCKET_PATH": str(self._cfg.gate_socket_path),
            "GATE_BASE_PATH": str(self._cfg.gate_base_path),
            "TOKEN_FILE": str(TokenStore(self._cfg).file_path),
            "UNIT_VERSION": str(_UNIT_VERSION),
            "TEROK_GATE_BIN": gate_bin,
        }

        # Remove units from the *other* transport mode before installing.
        self._remove_unit_files()

        if transport == "socket":
            templates = [_SOCKET_MODE_SERVICE]
            enable_unit = _SOCKET_MODE_SERVICE
        else:
            templates = [_SOCKET_UNIT, "terok-gate@.service"]
            enable_unit = _SOCKET_UNIT

        for template_name in templates:
            template_path = resource_dir / template_name
            if not template_path.is_file():
                raise SystemExit(f"Missing systemd template: {template_path}")
            content = render_template(template_path, variables)
            (unit_dir / template_name).write_text(content, encoding="utf-8")

        # Capture the "Created symlink ..." notice — otherwise it interleaves
        # with `terok setup`'s progressive stage output.  ``_systemctl.run``
        # surfaces failures with stderr attached, since
        # CalledProcessError's default message omits captured output.
        _systemctl.run("daemon-reload")
        _systemctl.run("enable", "--now", enable_unit)

    def _stop_all_units(self) -> None:
        """Stop and disable all gate units across both transport modes."""
        unit_dir = self._systemd_unit_dir()
        for unit in (_SOCKET_UNIT, _SOCKET_MODE_SERVICE):
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
        """Disable + remove gate unit files from prior versions.

        A unit file is considered ours (and eligible for removal) when
        its first line begins with :data:`_OWNED_MARKER_PREFIX` — every
        shipped template carries that marker.  Files matching the
        :data:`_OWNED_UNIT_GLOB` but lacking the marker are left alone
        (user-authored units that happen to share the ``terok-gate*``
        name prefix).  Current-version files are skipped here too;
        they're handled by the subsequent pass in
        :meth:`_remove_unit_files`.

        This catches legacy filenames from previous releases — e.g. if
        a future rename moves ``terok-gate-socket.service`` to a new
        name, the sweep cleans up the old one on the next
        ``terok setup`` without requiring a manual uninstall step.
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
        """Disable+stop all gate units and remove unit files."""
        self._remove_unit_files()
        _systemctl.run_best_effort("daemon-reload")

    # -- Daemon lifecycle ----------------------------------------------------

    def start_daemon(self, port: int | None = None) -> None:
        """Start a ``terok-gate`` daemon process (non-systemd fallback).

        Writes a PID file to ``runtime_root() / "gate-server.pid"``.
        If ``TEROK_GATE_ADMIN_TOKEN`` is set in the environment, it is
        forwarded to the daemon for host-level access to all repos.
        """
        from .tokens import TokenStore

        effective_port = port or self._cfg.gate_port
        gate_base = self._cfg.gate_base_path
        gate_base.mkdir(parents=True, exist_ok=True)
        pidfile = self._cfg.pid_file_path
        pidfile.parent.mkdir(parents=True, exist_ok=True)

        cmd = [
            "terok-gate",
            f"--base-path={gate_base}",
            f"--token-file={TokenStore(self._cfg).file_path}",
            f"--port={effective_port}",
            "--detach",
            f"--pid-file={pidfile}",
        ]
        bind_addr = os.environ.get("TEROK_GATE_BIND")
        if bind_addr:
            cmd.append(f"--bind={bind_addr}")

        # Pass the admin token via the subprocess environment only — never on
        # the command line where it would be visible in /proc/<pid>/cmdline.
        env = os.environ.copy()
        admin_token = env.get("TEROK_GATE_ADMIN_TOKEN")
        if admin_token:
            env["TEROK_GATE_ADMIN_TOKEN"] = admin_token

        subprocess.run(cmd, check=True, timeout=10, env=env)

    def stop_daemon(self) -> None:
        """Stop the gate server, whether running as a systemd unit or a PID-file daemon.

        Both paths are attempted unconditionally: systemd-managed gates
        live under ``terok-gate-socket.service`` / ``terok-gate.socket``
        and have no PID file, while a manually-started daemon has a PID
        file but no active unit.  Running both paths also sweeps stray
        daemons that outlived their systemd unit.
        """
        # ``_systemctl.run_best_effort`` swallows ``TimeoutExpired`` so a
        # wedged unit can't block the PID-file path below.
        for unit in (_SOCKET_UNIT, _SOCKET_MODE_SERVICE):
            if self._is_unit_active(unit):
                _systemctl.run_best_effort("stop", unit)
        pidfile = self._cfg.pid_file_path
        if not pidfile.is_file():
            return
        try:
            pid = int(pidfile.read_text().strip())
            if self._is_managed_server(pid):
                os.kill(pid, signal.SIGTERM)
        except (ValueError, ProcessLookupError, PermissionError):
            pass
        finally:
            if pidfile.is_file():
                pidfile.unlink()

    def is_daemon_running(self) -> bool:
        """Check whether the managed daemon process is alive via its PID file."""
        pidfile = self._cfg.pid_file_path
        if not pidfile.is_file():
            return False
        try:
            pid = int(pidfile.read_text().strip())
            if not self._is_managed_server(pid):
                return False
            os.kill(pid, 0)  # signal 0 = existence check
            return True
        except (ValueError, ProcessLookupError, PermissionError):
            return False

    # -- Private helpers -----------------------------------------------------

    @staticmethod
    def _systemd_unit_dir() -> Path:
        """Return the validated systemd user unit directory."""
        from .._util import systemd_user_unit_dir

        return systemd_user_unit_dir()

    def _installed_unit_version(self) -> int | None:
        """Return the version stamp from the installed unit files, or ``None``.

        Checks both TCP (socket unit) and socket-mode (service unit) files.
        """
        unit_dir = self._systemd_unit_dir()
        for name in (_SOCKET_UNIT, _SOCKET_MODE_SERVICE):
            unit_file = unit_dir / name
            if not unit_file.is_file():
                continue
            try:
                for line in unit_file.read_text(encoding="utf-8").splitlines():
                    if line.startswith("# terok-gate-version:"):
                        return int(line.split(":", 1)[1].strip())
            except (ValueError, OSError):
                pass
        return None

    def _installed_base_path(self) -> Path | None:
        """Parse the ``--base-path=...`` baked into the installed service unit.

        Checks both TCP (terok-gate@.service) and socket-mode service files.
        Returns ``None`` if no service unit is found or unparseable.
        """
        unit_dir = self._systemd_unit_dir()
        for name in ("terok-gate@.service", _SOCKET_MODE_SERVICE):
            service_file = unit_dir / name
            if not service_file.is_file():
                continue
            try:
                for line in service_file.read_text(encoding="utf-8").splitlines():
                    line = line.strip()
                    if line.startswith("ExecStart=") and "--base-path=" in line:
                        for token in line.split():
                            if token.startswith("--base-path="):
                                return Path(token.split("=", 1)[1])
            except OSError:
                pass
        return None

    def _base_path_diverged(self) -> str | None:
        """Return a warning if the installed base path differs from current config.

        Returns ``None`` when paths match or when units are not installed.
        """
        installed = self._installed_base_path()
        if installed is None:
            return None
        expected = self._cfg.gate_base_path
        if installed.resolve() == expected.resolve():
            return None
        return (
            f"Installed gate base path diverges from current config.\n"
            f"  Installed: {installed}\n"
            f"  Expected:  {expected}"
        )

    def _is_managed_server(self, pid: int) -> bool:
        """Return whether *pid* was started with the expected PID file argument.

        Reads ``/proc/<pid>/cmdline`` and checks that the ``--pid-file=<path>``
        flag matches the configured PID file path.  This guards against PID
        reuse (a stale PID file pointing at an unrelated process).
        """
        cmdline_path = Path(f"/proc/{pid}/cmdline")
        if not cmdline_path.is_file():
            return False
        try:
            raw = cmdline_path.read_bytes()
        except OSError:
            return False
        args = raw.rstrip(b"\x00").split(b"\x00")
        if len(args) < 2:
            return False
        args_str = [a.decode("utf-8", errors="ignore") for a in args]
        expected_pid_flag = f"--pid-file={self._cfg.pid_file_path}"
        return expected_pid_flag in args_str
