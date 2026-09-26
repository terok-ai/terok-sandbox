# SPDX-FileCopyrightText: 2026 Jiri Vyskocil
# SPDX-License-Identifier: Apache-2.0

"""Shared OCI-hook ballast for the supervisor hook — stdlib only.

Shipped alongside `supervisor_hook.py`
in the installed hooks directory.  The role script adds
``Path(__file__).parent`` to ``sys.path`` and ``from _supervisor_state
import …`` resolves to this file at runtime.

Annotation-driven design: this module owns only the host-side
helpers (UID resolution, host tool lookup, PID introspection,
logging).  All terok-specific paths come from the sidecar
JSON that the OCI annotation pins; no ``$XDG_*`` resolution lives
here.

Stdlib-only by design: OCI runtimes execute the hook with
an installation-bound Python in isolation, so an import of
``terok_sandbox`` would fail.  Mirrors the same constraint shield's
``_oci_state.py`` carries — the design rationale lives there.
"""

from __future__ import annotations

import contextlib
import os
import subprocess  # nosec B404 — the user manager's verbs, fixed argv
import sys
import time
from pathlib import Path
from typing import TYPE_CHECKING

if TYPE_CHECKING or __package__:
    from terok_util import find_host_tool, host_path
else:
    from _host_tools import find_host_tool, host_path

_SETUP_PATH = "__SETUP_PATH__"

#: Prefix of the transient user unit that hosts one container's supervisor.
_UNIT_PREFIX = "terok-supervisor-"

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
    """Preserve the launch PATH; older runtimes may need setup's fallback.

    Linker/Python injection knobs remain excluded independently of tool lookup.
    """
    os.environ.setdefault("PATH", _SETUP_PATH)
    os.environ["PATH"] = host_path()
    for var in ("LD_PRELOAD", "LD_LIBRARY_PATH", "LD_AUDIT", "PYTHONPATH", "PYTHONHOME"):
        os.environ.pop(var, None)
    if not os.environ.get("XDG_RUNTIME_DIR"):
        os.environ["XDG_RUNTIME_DIR"] = str(user_runtime_dir(host_uid))


def user_runtime_dir(host_uid: int) -> Path:
    """logind's per-user runtime directory for *host_uid*.

    The one place that spells the path: the hook pins the supervisor's
    ``XDG_RUNTIME_DIR`` to it, and the placement question below is asked
    of it, on the hook side and on the launcher side alike.
    """
    return Path(f"/run/user/{host_uid}")


def user_manager_reachable(runtime_dir: Path) -> bool:
    """Whether a per-user systemd manager answers under *runtime_dir*.

    The fact that decides where a container's supervisor runs: as a
    transient unit of that manager, in the operator's own namespaces
    where the user keyring is the operator's, or as a daemon inside the
    container runtime's user namespace, where the hook itself runs.  The
    manager's private socket is the evidence; ``systemd-run`` on the
    host PATH is what asks it.
    """
    private = runtime_dir / "systemd" / "private"
    return private.is_socket() and find_host_tool("systemd-run") is not None


def unit_name(container_id: str) -> str:
    """The transient user unit hosting *container_id*'s supervisor."""
    return f"{_UNIT_PREFIX}{container_id[:12]}.service"


def unit_pattern() -> str:
    """The glob that names every supervisor unit at once."""
    return f"{_UNIT_PREFIX}*"


def unit_active(name: str) -> bool:
    """Whether the user manager reports unit *name* active."""
    return _systemctl("is-active", name) == 0


def stop_unit(name: str) -> None:
    """Stop unit *name* through the user manager; a unit it never had is no error."""
    _systemctl("stop", name)


def kill_units(pattern: str) -> None:
    """SIGKILL every unit matching *pattern* at once — the panic path's verb."""
    _systemctl("kill", "--signal=SIGKILL", pattern)


def _systemctl(*args: str) -> int:
    """Run one quiet ``systemctl --user`` verb; its exit status is the answer."""
    binary = find_host_tool("systemctl")
    if binary is None:
        return 1
    try:
        return subprocess.run(  # nosec B603 — resolved host tool, fixed verbs
            [binary, "--user", "--quiet", *args], check=False
        ).returncode
    except OSError:
        return 1


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
