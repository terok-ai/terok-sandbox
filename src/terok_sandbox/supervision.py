# SPDX-FileCopyrightText: 2026 Jiri Vyskocil
# SPDX-License-Identifier: Apache-2.0

"""Post-start supervision check — did the per-container supervisor come up?

A container start deliberately *survives* a broken supervisor: the OCI
hook soft-fails so a spawn failure never blocks the container, and
terok-shield's egress firewall is fail-closed on its own hook.  What must
not happen is that the degradation stays **silent** — a container whose
vault-routed providers and git gate are all dead, behaving normally at the
shell, with nothing said (issue #458).

[`verify_supervision`][terok_sandbox.supervision.verify_supervision] closes
that gap on the API launch path.  After the container starts, it reads the
same sidecar the supervisor reads, and — when the sidecar declares
socket-mode services — polls briefly for the sockets the supervisor binds.
On timeout it returns a [`SupervisionStatus`][terok_sandbox.supervision.SupervisionStatus]
naming the container, the unbound socket(s), and the hook diary to read;
the caller shouts it but the launch still succeeds (soft-fail preserved),
and an orchestrator may escalate on the structured result.

TCP-mode wiring binds per-container loopback ports rather than sockets, so
there is nothing on disk to poll — the check reports *skipped* there rather
than guessing.  The pure-Python file poll never runs a subprocess and never
raises.
"""

from __future__ import annotations

import stat
import sys
import time
from dataclasses import dataclass
from typing import TYPE_CHECKING

from .supervisor.sidecar import SupervisorPaths, load_sidecar

if TYPE_CHECKING:
    from pathlib import Path

    from .config import SandboxConfig

#: How long to wait for the supervisor to bind its sockets before declaring
#: it unresponsive.  The gate binds early and the vault right after, so a
#: healthy supervisor clears this in ~100 ms; the budget only elapses in
#: full when the supervisor genuinely never came up.
_DEFAULT_TIMEOUT_S = 5.0

#: Filesystem poll cadence while waiting for the sockets to appear.
_POLL_INTERVAL_S = 0.1


@dataclass(frozen=True)
class SupervisionStatus:
    """The result of a post-start supervision check for one container.

    ``missing`` is the subset of ``checked`` sockets still absent when the
    poll gave up — empty on a healthy start.  ``skipped`` marks the cases
    with nothing to verify (no sidecar, or TCP-mode wiring that binds ports
    instead of sockets), which is *not* a failure.
    """

    container_name: str
    checked: tuple[Path, ...]
    missing: tuple[Path, ...]
    hook_log: Path
    skipped: bool = False

    @property
    def ok(self) -> bool:
        """``True`` when every required socket was bound (or nothing needed checking)."""
        return not self.missing

    def warning(self) -> str:
        """A loud, multi-line operator warning naming the failure and where to look."""
        sockets = "\n".join(f"warning:     {p}" for p in self.missing)
        return (
            f"warning: container {self.container_name!r} started but its supervisor is "
            "not responding\n"
            f"{sockets}\n"
            "warning:   vault-routed providers and/or the git gate are dead in this container\n"
            f"warning:   hook diary: {self.hook_log} "
            "(absent or empty ⇒ the OCI supervisor hook never fired)"
        )


def verify_supervision(
    cfg: SandboxConfig,
    container_name: str,
    *,
    timeout: float = _DEFAULT_TIMEOUT_S,
) -> SupervisionStatus:
    """Poll for the supervisor's sockets after *container_name* has started.

    Reads ``<state>/sidecar/<container_name>.json`` — the same bundle the
    supervisor reads — and, in socket mode, waits up to *timeout* seconds
    for the vault socket (always bound) and the gate socket (when the
    sidecar wired a gate).  Returns a
    [`SupervisionStatus`][terok_sandbox.supervision.SupervisionStatus]; a
    missing socket means the supervisor is not up.  Never raises and never
    blocks a healthy start beyond the time the sockets take to appear.
    """
    sidecar_path = cfg.state_dir / "sidecar" / f"{container_name}.json"
    # The install-global hook diary the OCI hook appends to (mirrors
    # ``ContainerDiagnostics.hook_log``, which is the host-facing SSOT — but
    # that lives in the surface layer, out of reach from here).
    hook_log = cfg.state_dir / "logs" / "hook.log"
    sidecar = load_sidecar(sidecar_path) if sidecar_path.exists() else None
    if sidecar is None:
        return SupervisionStatus(container_name, (), (), hook_log, skipped=True)

    if sidecar.ipc_mode != "socket":
        # TCP-mode services bind per-container loopback ports, not sockets —
        # there is nothing on disk to poll, so don't pretend to check.
        return SupervisionStatus(container_name, (), (), hook_log, skipped=True)

    paths = SupervisorPaths.for_container(
        container_id="",  # vault/gate sockets key on the name, not the id
        container_name=container_name,
        sidecar_path=sidecar_path,
        runtime_dir=sidecar.runtime_dir,
    )
    expected = [paths.vault_socket]
    if sidecar.gate_base_path and sidecar.gate_token:
        expected.append(paths.gate_socket)

    missing = _poll_until_bound(tuple(expected), timeout)
    return SupervisionStatus(container_name, tuple(expected), missing, hook_log)


def warn_unsupervised(status: SupervisionStatus) -> None:
    """Print the loud warning for a failed check to stderr; no-op when healthy."""
    if status.missing:
        print(status.warning(), file=sys.stderr)


def _poll_until_bound(expected: tuple[Path, ...], timeout: float) -> tuple[Path, ...]:
    """Return the sockets from *expected* still unbound when *timeout* elapses."""
    deadline = time.monotonic() + timeout
    remaining = expected
    while True:
        remaining = tuple(p for p in remaining if not _is_socket(p))
        if not remaining or time.monotonic() >= deadline:
            return remaining
        time.sleep(_POLL_INTERVAL_S)


def _is_socket(path: Path) -> bool:
    """``True`` when *path* exists and is an AF_UNIX socket (best-effort)."""
    try:
        return stat.S_ISSOCK(path.stat().st_mode)
    except OSError:
        return False


__all__ = ["SupervisionStatus", "verify_supervision", "warn_unsupervised"]
