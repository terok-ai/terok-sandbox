# SPDX-FileCopyrightText: 2026 Jiri Vyskocil
# SPDX-License-Identifier: Apache-2.0

"""Shared OCI-hook ballast for the supervisor hook — stdlib only.

Shipped alongside `supervisor_hook.py`
in the installed hooks directory.  The role script adds
``Path(__file__).parent`` to ``sys.path`` and ``from _supervisor_state
import …`` resolves to this file at runtime.

Annotation-driven design: this module owns only the host-side
helpers (UID resolution, ``$PATH`` hardening, PID introspection,
logging).  All terok-specific paths come from the sidecar
JSON that the OCI annotation pins; no ``$XDG_*`` resolution lives
here.

Stdlib-only by design: OCI runtimes execute the hook with
``/usr/bin/python3`` outside any virtualenv, so an import of
``terok_sandbox`` would fail.  Mirrors the same constraint shield's
``_oci_state.py`` carries — the design rationale lives there.
"""

from __future__ import annotations

import contextlib
import os
import sys
import time
from pathlib import Path

#: Trusted ``$PATH`` for hook subprocess execution — same allowlist
#: shield's ``_oci_state.py`` pins, kept in sync deliberately.
_TRUSTED_PATH = "/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin"

#: Persistent hook diary ``log`` mirrors into, and the container tag it
#: stamps on each line.  Both stay unset until ``set_log_context`` resolves
#: the state root + container id from the sidecar annotation — an early
#: failure (bad OCI state, unusable annotation) isn't yet tied to a
#: container, so it goes to stderr alone.
_hook_log_path: Path | None = None
_log_tag: str = "-"


def outer_host_uid() -> int:
    """Return the invoking operator's host UID, even from inside ``NS_ROOTLESS``.

    Parses ``/proc/self/uid_map`` to find the outer-side UID that the
    current in-namespace UID maps to.  Each map line has the shape
    ``<inner_start> <outer_start> <length>`` — pick the mapping whose
    inner range covers ``os.getuid()`` and project through it.

    Falls back to ``os.getuid()`` on any parse trouble (init userns,
    no uid_map, unreadable, unexpected format).  Verbatim copy of
    shield's same-named helper; duplicated rather than imported
    because the hook can't depend on ``terok_shield`` either.
    """
    from pathlib import Path

    my_uid = os.getuid()
    try:
        raw = Path("/proc/self/uid_map").read_text()
    except OSError:
        return my_uid
    for line in raw.splitlines():
        parts = line.split()
        if len(parts) != 3:
            continue
        try:
            inner_start = int(parts[0])
            outer_start = int(parts[1])
            length = int(parts[2])
        except ValueError:
            continue
        if inner_start <= my_uid < inner_start + length:
            return outer_start + (my_uid - inner_start)
    return my_uid


def bootstrap_env(host_uid: int) -> None:
    """Pin ``$PATH`` and wipe linker-hijack vectors before any subprocess work.

    Same hardening shield's hook applies.  Pinning ``$PATH`` to a
    trusted constant and unsetting the dynamic-linker knobs defeats
    a malicious OCI-runtime env that might otherwise hijack child
    execs.
    """
    os.environ["PATH"] = _TRUSTED_PATH
    for var in ("LD_PRELOAD", "LD_LIBRARY_PATH", "LD_AUDIT", "PYTHONPATH", "PYTHONHOME"):
        os.environ.pop(var, None)
    if not os.environ.get("XDG_RUNTIME_DIR"):
        os.environ["XDG_RUNTIME_DIR"] = f"/run/user/{host_uid}"


def pid_exists(pid: int) -> bool:
    """Ask the kernel whether *pid* is still a running process.

    ``pid <= 0`` is a category error from a corrupt / empty PID file;
    ``os.kill(0, 0)`` would otherwise broadcast to the caller's whole
    process group, and ``os.kill(-1, 0)`` would scan every process the
    caller is allowed to signal — both nonsensical here.
    """
    if pid <= 0:
        return False
    try:
        os.kill(pid, 0)
    except ProcessLookupError:
        return False
    except OSError:
        # EPERM means the process exists but we don't own it — treat as alive.
        return True
    return True


def set_log_context(root: Path, container_id: str) -> None:
    """Point ``log`` at the persistent hook diary.

    Called by the hook the moment it has resolved both anchors — the
    state *root* (derived from the sidecar path) and the *container_id*.
    From here on every ``log`` line is *also* appended to
    ``<root>/logs/hook.log``, tagged with the container, so a degraded
    start leaves a durable trace the OCI runtime's own journal capture
    can't be trusted to keep across crun/runc/podman versions.

    Best-effort: it pre-creates the ``logs`` directory but never the log
    file — the file stays absent until something is actually logged, so
    an *empty / missing* ``hook.log`` means the hook never fired for a
    container, while any content means it fired and said why.
    """
    global _hook_log_path, _log_tag
    with contextlib.suppress(OSError):
        (root / "logs").mkdir(parents=True, exist_ok=True)
    _hook_log_path = root / "logs" / "hook.log"
    _log_tag = container_id[:12] or "-"


def log(msg: str) -> None:
    """Write *msg* to stderr, and mirror it into the hook diary when armed.

    The OCI runtime captures stderr into its journal (``journalctl
    --user _COMM=conmon``) — unreliably, across runtime versions, which
    is why ``set_log_context`` additionally routes each line to a
    persistent, container-tagged ``hook.log``.  The per-container
    supervisor keeps its own log; this
    diary is the *cross-container* record of what the hook itself did.

    The file append uses ``O_APPEND`` (one open-write-close per line), so
    concurrent hooks for different containers interleave whole lines
    without a lock, and a broken log target is swallowed — diagnostics
    must never take down container start.
    """
    print(msg, file=sys.stderr)
    if _hook_log_path is None:
        return
    stamp = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
    with contextlib.suppress(OSError), _hook_log_path.open("a", encoding="utf-8") as fh:
        fh.write(f"{stamp} [{_log_tag}] {msg}\n")
