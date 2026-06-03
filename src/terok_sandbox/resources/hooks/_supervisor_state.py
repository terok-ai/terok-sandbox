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

import os
import sys

#: Trusted ``$PATH`` for hook subprocess execution — same allowlist
#: shield's ``_oci_state.py`` pins, kept in sync deliberately.
_TRUSTED_PATH = "/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin"


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


def log(msg: str) -> None:
    """Write *msg* to stderr.

    The OCI runtime captures this into its journal (``journalctl
    --user _COMM=conmon``).  Persistent per-container log files are
    written by the supervisor itself, not by the hook — the hook
    proper has no per-container log target until after the wrapper
    is spawned, at which point its stdout / stderr inherit the log
    file the hook opened for the wrapper.
    """
    print(msg, file=sys.stderr)
