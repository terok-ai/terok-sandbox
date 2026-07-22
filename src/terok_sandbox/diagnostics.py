# SPDX-FileCopyrightText: 2026 Jiri Vyskocil
# SPDX-License-Identifier: Apache-2.0

"""On-host supervisor + sidecar artifact paths for a container.

Single source of truth for the *human-facing* debug layout.  When an
operator needs the supervisor log, the wrapper, the PID file, or the
sidecar bundle for a container, the paths come from here rather than
being re-derived by each frontend — ``terok task status -v`` is the
first consumer, pointing a human at the file to send back instead of
making them hand-assemble it from ``podman inspect`` annotations.

The artifacts live under three different keys, which is exactly why a
shared resolver earns its keep:

* log + PID key on the immutable **container ID** — podman assigns it
  at create time, and the supervisor names both after it.
* sidecar keys on the **container name** — known before the ID exists,
  so the launch path can write it pre-``podman run``.
* the wrapper is install-global (one per state root), not per
  container.

Paths are computed, never probed: a file may be absent (the sidecar is
removed at teardown, the supervisor may never have logged).  Callers
that care about existence check it themselves.  The one exception is
[`supervisor_liveness`][terok_sandbox.diagnostics.supervisor_liveness],
which *does* probe — it answers the root-cause question "is this
container's supervisor actually running?" that the path bundle only
points at.
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path

from .paths import state_root
from .supervisor.install import _PIDS_DIR_NAME, _WRAPPER_NAME

_LOGS_DIR_NAME = "logs"
_SIDECAR_DIR_NAME = "sidecar"
_HOOK_LOG_NAME = "hook.log"

#: Where the liveness probe reads process argvs from (patchable in tests).
_PROC_DIR = Path("/proc")


@dataclass(frozen=True)
class ContainerDiagnostics:
    """Resolved on-host artifact paths for one container.

    Every field is an absolute [`Path`][pathlib.Path]; none is
    guaranteed to exist on disk (see the module docstring).
    """

    container_id: str
    log: Path
    """Persistent supervisor log: ``<state>/logs/<id>.log``."""
    pid: Path
    """Supervisor PID file: ``<state>/pids/supervisor-<id>.pid``."""
    wrapper: Path
    """Install-global supervisor wrapper: ``<state>/supervisor_wrapper.py``."""
    sidecar: Path
    """Per-container sidecar bundle: ``<state>/sidecar/<name>.json``."""
    hook_log: Path
    """Install-global OCI-hook diary: ``<state>/logs/hook.log``.

    Shared across every container (each line is container-tagged) — an
    absent or empty file means the supervisor hook never fired, which is
    the first thing to check when a container comes up unsupervised."""


def container_diagnostics(
    container_id: str,
    container_name: str,
    *,
    state_dir: Path | None = None,
) -> ContainerDiagnostics:
    """Resolve the artifact-path bundle for one container.

    *state_dir* defaults to [`state_root`][terok_sandbox.paths.state_root]
    — the operator's resolved ``paths.root`` — so the bundle moves with
    a relocated state tree just like every other terok artifact.  Pass
    an explicit *state_dir* (e.g. a caller's
    [`SandboxConfig.state_dir`][terok_sandbox.config.SandboxConfig]) to
    pin resolution to a specific config rather than the layered default.
    """
    root = state_dir or state_root()
    return ContainerDiagnostics(
        container_id=container_id,
        log=root / _LOGS_DIR_NAME / f"{container_id}.log",
        pid=root / _PIDS_DIR_NAME / f"supervisor-{container_id}.pid",
        wrapper=root / _WRAPPER_NAME,
        sidecar=root / _SIDECAR_DIR_NAME / f"{container_name}.json",
        hook_log=root / _LOGS_DIR_NAME / _HOOK_LOG_NAME,
    )


@dataclass(frozen=True)
class SupervisorLiveness:
    """Whether a container's per-container supervisor is currently running.

    The result of probing the recorded PID file and ``/proc`` — the same
    signal the OCI hook's idempotent-respawn guard uses.  ``pid`` is the
    PID the file records (kept even when the process turns out dead, so a
    caller can name it); ``detail`` is a short human phrase for a status
    line.
    """

    #: ``True`` only when the recorded PID is live *and* is our wrapper.
    alive: bool
    #: The PID recorded in the file, or ``None`` when there was no PID file.
    pid: int | None
    #: One-line reason, ready for a diagnostic status line.
    detail: str


def supervisor_liveness(
    container_id: str,
    *,
    state_dir: Path | None = None,
) -> SupervisorLiveness:
    """Probe whether the supervisor wrapper for *container_id* is running.

    Reads ``<state>/pids/supervisor-<id>.pid`` and confirms the recorded
    process is alive **and** is our wrapper for this container — the
    recorded wrapper path and the container id must both appear in its
    argv, the double-mark that guards against a recycled PID (the same
    check the OCI hook makes before deciding whether to respawn).

    Never raises: a missing PID file (the hook never spawned a supervisor)
    and a stale one (the wrapper died, or the PID was recycled) both come
    back ``alive=False`` with a distinguishing *detail*.  Pass *state_dir*
    to pin resolution to a specific config (mirrors
    [`container_diagnostics`][terok_sandbox.diagnostics.container_diagnostics]).
    """
    root = state_dir or state_root()
    pid_file = root / _PIDS_DIR_NAME / f"supervisor-{container_id}.pid"
    wrapper = root / _WRAPPER_NAME
    try:
        pid = int(pid_file.read_text().strip())
    except (OSError, ValueError):
        return SupervisorLiveness(
            alive=False,
            pid=None,
            detail="no PID file — the supervisor hook never spawned a supervisor",
        )
    if _pid_alive(pid) and _wrapper_argv_matches(pid, wrapper, container_id):
        return SupervisorLiveness(alive=True, pid=pid, detail=f"supervisor pid {pid} alive")
    return SupervisorLiveness(
        alive=False, pid=pid, detail=f"stale PID file — pid {pid} is dead or recycled"
    )


def _pid_alive(pid: int) -> bool:
    """Signal-0 liveness probe for *pid* (EPERM counts as alive)."""
    if pid <= 0:
        return False
    try:
        os.kill(pid, 0)
    except ProcessLookupError:
        return False
    except OSError:
        return True
    return True


def _wrapper_argv_matches(pid: int, wrapper: Path, container_id: str) -> bool:
    """``True`` when ``/proc/<pid>`` is our wrapper argv for *container_id*.

    Both the recorded wrapper path and the container id must appear in the
    null-separated cmdline; the wrapper path alone matches every live
    supervisor, so the id is what pins it to *this* container.
    """
    try:
        raw = (_PROC_DIR / str(pid) / "cmdline").read_bytes()
    except OSError:
        return False
    args = raw.rstrip(b"\x00").split(b"\x00")
    return str(wrapper).encode() in args and container_id.encode() in args


__all__ = [
    "ContainerDiagnostics",
    "SupervisorLiveness",
    "container_diagnostics",
    "supervisor_liveness",
]
