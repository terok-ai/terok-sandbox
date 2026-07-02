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
that care about existence check it themselves.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from .paths import state_root
from .supervisor.install import _PIDS_DIR_NAME, _WRAPPER_NAME

_LOGS_DIR_NAME = "logs"
_SIDECAR_DIR_NAME = "sidecar"


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
    )


__all__ = ["ContainerDiagnostics", "container_diagnostics"]
