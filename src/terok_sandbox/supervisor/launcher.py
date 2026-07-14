# SPDX-FileCopyrightText: 2026 Jiri Vyskocil
# SPDX-License-Identifier: Apache-2.0

"""How the parent supervisor spawns one child process per service.

The launcher is the seam between "which services run" (the parent's
concern) and "how a service process is isolated" (the host's concern).
Two first-class implementations share one
[`ProcessLauncher`][terok_sandbox.supervisor.launcher.ProcessLauncher]
protocol, one per host flavour:

* [`DirectLauncher`][terok_sandbox.supervisor.launcher.DirectLauncher] —
  a plain ``asyncio`` subprocess.  The supported path on hosts **without**
  systemd (Alpine / OpenRC, minimal container images) and on systemd
  hosts where no user manager is reachable (the OCI hook usually runs
  with no login session).  Every child still hardens itself
  ([`harden_self`][terok_util.harden_self]); only the cgroup resource
  ceiling is absent.
* [`SystemdRunLauncher`][terok_sandbox.supervisor.launcher.SystemdRunLauncher] —
  wraps the same argv in ``systemd-run --user --scope`` so the child also
  lands in its own transient scope with kernel-enforced resource limits
  (memory, tasks).  Used **only** when a usable user systemd manager is
  actually present; it still degrades to a direct spawn per child if a
  spawn nonetheless fails.

[`default_launcher`][terok_sandbox.supervisor.launcher.default_launcher]
picks between them by probing the host (overridable via the
``TEROK_SUPERVISOR_LAUNCHER`` env var), so a systemd box gets the scope
ceiling and a non-systemd box just works.

Both spawn the child as ``python -m terok_sandbox supervise-child
<service> <container_id> <sidecar_path>`` on the *parent's* interpreter,
so a child needs neither ``terok-sandbox`` on ``$PATH`` nor a rendered
wrapper.
"""

from __future__ import annotations

import asyncio
import os
import shutil
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Protocol

if TYPE_CHECKING:
    from collections.abc import Sequence

#: The argv that re-invokes this package on the running interpreter.
#: ``-m terok_sandbox`` resolves through
#: [`terok_sandbox.__main__`][] regardless of how the parent was
#: installed (editable, venv, system).
_SELF_ARGV: tuple[str, ...] = (sys.executable, "-m", "terok_sandbox")

#: Per-scope resource ceiling for a systemd-run child.  Deliberately
#: generous — the point is a hard backstop against a runaway service, not
#: tight tuning; a vault proxy or gate that legitimately needs more will
#: not come near these.
_SCOPE_PROPERTIES: tuple[str, ...] = (
    "MemoryMax=256M",
    "TasksMax=64",
)

#: Operator override for launcher selection.  ``auto`` (default) probes
#: the host; ``direct`` / ``systemd`` force one launcher regardless (a
#: forced ``systemd`` still degrades per child if a spawn fails).
_LAUNCHER_ENV = "TEROK_SUPERVISOR_LAUNCHER"

#: sd_booted(3): this directory exists iff the system was booted with
#: systemd as PID 1.  Absent on non-systemd inits (OpenRC, runit, s6) and
#: in minimal containers.
_SYSTEMD_INIT_MARKER = Path("/run/systemd/system")


def _user_manager_socket(runtime_dir: Path) -> Path:
    """The current user's systemd manager private socket.

    ``systemd-run --user`` talks to the per-user manager over this
    socket; its presence means a user manager is actually running (a
    lingering-enabled user, or an active login session) and ``--user``
    scopes will work.  Without it — the common OCI-hook case, spawned
    with no session — ``--user`` would fail, so we must not pick systemd.

    The ``XDG_RUNTIME_DIR`` (``/run/user/<uid>``) is taken from the
    sidecar's *runtime_dir* (``<XDG_RUNTIME_DIR>/terok/sandbox``), **not**
    from ``os.getuid()`` — the supervisor runs under crun's rootless user
    namespace where ``getuid()`` reads 0 and would misroute the probe to
    ``/run/user/0``, mis-detecting the manager as absent and silently
    dropping the systemd scope ceiling.  The launch path carries the
    correct value for exactly this reason.
    """
    return runtime_dir.parent.parent / "systemd" / "private"


@dataclass(frozen=True)
class ChildHandle:
    """A launched child: its service name and the process running it."""

    service: str
    process: asyncio.subprocess.Process

    @property
    def pid(self) -> int:
        """The child's process id."""
        return self.process.pid


class ProcessLauncher(Protocol):
    """Spawns one supervisor-child process for a named service."""

    async def launch(self, service: str, container_id: str, sidecar_path: Path) -> ChildHandle:
        """Start the *service* child and return a handle to it."""
        ...


def _child_argv(service: str, container_id: str, sidecar_path: Path) -> list[str]:
    """Build the ``supervise-child`` argv for *service*."""
    return ["supervise-child", service, container_id, str(sidecar_path)]


class DirectLauncher:
    """Spawn each child as a plain subprocess on the parent's interpreter."""

    async def launch(self, service: str, container_id: str, sidecar_path: Path) -> ChildHandle:
        """Fork+exec ``python -m terok_sandbox supervise-child …`` for *service*."""
        process = await asyncio.create_subprocess_exec(
            *_SELF_ARGV, *_child_argv(service, container_id, sidecar_path)
        )
        return ChildHandle(service=service, process=process)


class SystemdRunLauncher:
    """Wrap each child in a transient ``systemd-run --user --scope``.

    Gives every service its own cgroup with a memory/tasks ceiling.
    Whether a usable user manager exists is decided once, upfront, by
    [`default_launcher`][terok_sandbox.supervisor.launcher.default_launcher]
    via [`is_available`][terok_sandbox.supervisor.launcher.SystemdRunLauncher.is_available];
    this class is chosen only when it does (or when an operator forces
    it).  As a last-ditch guard each ``launch`` still degrades to a
    direct spawn if ``systemd-run`` has vanished from ``$PATH`` — the
    ``harden_self`` floor and SELinux transition are unaffected either
    way.
    """

    def __init__(self, *, properties: Sequence[str] = _SCOPE_PROPERTIES) -> None:
        self._properties = tuple(properties)
        self._fallback = DirectLauncher()

    @staticmethod
    def is_available(runtime_dir: Path) -> bool:
        """``True`` only when a user ``systemd-run`` scope will actually work.

        Three preconditions, cheapest first — all filesystem checks, no
        subprocess:

        1. ``systemd-run`` is on ``$PATH``.
        2. The host booted with systemd as PID 1 (``/run/systemd/system``);
           a non-systemd init fails this.
        3. A per-user systemd manager is running (its private socket,
           derived from the sidecar *runtime_dir*, exists); the OCI hook
           usually runs with no session, so this weeds out the "systemd
           installed but ``--user`` unreachable" trap the naive ``$PATH``
           check would fall into.

        Any miss means [`default_launcher`][terok_sandbox.supervisor.launcher.default_launcher]
        picks the direct path instead.
        """
        return (
            shutil.which("systemd-run") is not None
            and _SYSTEMD_INIT_MARKER.is_dir()
            and _user_manager_socket(runtime_dir).exists()
        )

    async def launch(self, service: str, container_id: str, sidecar_path: Path) -> ChildHandle:
        """Spawn *service* inside a transient user scope, or fall back to direct."""
        systemd_run = shutil.which("systemd-run")
        if systemd_run is None:
            return await self._fallback.launch(service, container_id, sidecar_path)
        scope_argv = [
            systemd_run,
            "--user",
            "--scope",
            "--quiet",
            "--collect",
            f"--unit=terok-{container_id[:12]}-{service}",
            *(f"--property={prop}" for prop in self._properties),
            *_SELF_ARGV,
            *_child_argv(service, container_id, sidecar_path),
        ]
        process = await asyncio.create_subprocess_exec(*scope_argv)
        return ChildHandle(service=service, process=process)


def default_launcher(runtime_dir: Path) -> ProcessLauncher:
    """Pick the launcher the host supports, honouring an operator override.

    ``TEROK_SUPERVISOR_LAUNCHER`` forces the choice — ``direct`` or
    ``systemd`` — for hosts the probe reads wrong or operators who want
    one behaviour everywhere.  The default, ``auto``, uses the systemd
    scope ceiling only when
    [`SystemdRunLauncher.is_available`][terok_sandbox.supervisor.launcher.SystemdRunLauncher.is_available]
    confirms a usable user manager (probed via the sidecar *runtime_dir*,
    not ``getuid()``), and the plain direct spawn otherwise (non-systemd
    hosts, sessionless hooks).  Both harden every child identically; only
    the cgroup ceiling differs.
    """
    choice = os.environ.get(_LAUNCHER_ENV, "auto").strip().lower()
    if choice == "direct":
        return DirectLauncher()
    if choice == "systemd":
        return SystemdRunLauncher()
    if SystemdRunLauncher.is_available(runtime_dir):
        return SystemdRunLauncher()
    return DirectLauncher()


__all__ = [
    "ChildHandle",
    "DirectLauncher",
    "ProcessLauncher",
    "SystemdRunLauncher",
    "default_launcher",
]
