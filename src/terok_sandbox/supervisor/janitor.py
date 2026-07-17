# SPDX-FileCopyrightText: 2026 Jiri Vyskocil
# SPDX-License-Identifier: Apache-2.0

"""Reconcile stray supervisor process trees against live containers.

The per-container supervisor is meant to die with its container — the
``poststop`` hook group-kills the tree, the supervisor self-terminates
when its container's init PID dies, and ``PR_SET_PDEATHSIG`` takes the
service children down with it.
Those are the *prevention* layers.  This module is the *reconciliation*
backstop: a periodic sweep that finds supervisor trees whose container is
no longer running and kills them, no matter how they were stranded (a
``poststop`` that never fired, a host crash that left the tree, or a
supervisor built before the prevention layers existed).

Unlike the OCI hook — which crun hands the container's environment, so its
nested ``podman`` calls are unreliable — the janitor runs from an ordinary
host CLI context (``doctor``, a task launch), where ``podman ps`` answers
normally.  That is what lets it use container liveness as the ground truth
the hook cannot.

A tree is identified structurally: the wrapper is a session leader (the
OCI hook spawns it with ``start_new_session=True``), and every descendant
— supervisor, service children — inherits that process group, so one
``killpg`` per group takes the whole tree down.  Groups are matched by the
container id carried in each process's argv.
"""

from __future__ import annotations

import contextlib
import json
import os
import shutil
import signal
import subprocess  # nosec B404
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from ..doctor import DoctorCheck

#: Where the process scan reads argvs from (patchable in tests).
_PROC_DIR = Path("/proc")

#: argv marks of the supervisor family.  A wrapper carries the wrapper
#: script name; a service child carries the module invocation + verb.
_WRAPPER_MARK = b"supervisor_wrapper.py"
_CHILD_VERB = b"supervise-child"
_MODULE_MARK = b"terok_sandbox"

#: Container states in which a supervisor is *legitimately* alive — the
#: sweep never touches these.  Everything else (exited / stopped / dead /
#: removed, or a container podman no longer knows at all) means the tree
#: is stranded.  ``created`` / ``configured`` / ``initialized`` cover the
#: brief window between the createRuntime hook spawning the supervisor and
#: the container reaching ``running``.
_ALIVE_STATES = frozenset({"running", "paused", "created", "configured", "initialized", "stopping"})

#: How long a supervisor process must have existed before the sweep will
#: reap it.  A guard against the create→register race: a container mid-creation
#: may not appear in ``podman ps`` yet, so a just-spawned supervisor whose
#: container id isn't listed is given this long before it counts as stray.
_ORPHAN_GRACE_S = 90.0

#: Grace between the group SIGTERM and the escalation SIGKILL.
_KILL_GRACE_S = 5.0

_CLOCK_TICKS = os.sysconf("SC_CLK_TCK")


@dataclass
class _Group:
    """One container's supervisor tree, as found on the host."""

    container_id: str
    pgids: set[int] = field(default_factory=set)
    youngest_age_s: float = float("inf")


def reap_orphaned_supervisors(
    *, min_age_s: float = _ORPHAN_GRACE_S
) -> list[tuple[str, str | None]]:
    """Kill every supervisor tree whose container is no longer running.

    Returns one ``(container_id, error_or_None)`` row per tree reaped —
    ``None`` when it went down cleanly.  An empty list means nothing was
    stray (the steady state) *or* podman was unreachable and the sweep
    declined to guess (it never kills a tree it can't prove is orphaned).
    """
    alive = _live_container_ids()
    if alive is None:
        return []  # can't tell live from stray — do nothing, safely
    groups = _scan_supervisor_groups()
    results: list[tuple[str, str | None]] = []
    for container_id, group in sorted(groups.items()):
        if container_id in alive:
            continue
        if group.youngest_age_s < min_age_s:
            continue  # a just-spawned tree racing its container's registration
        results.append((container_id, _kill_group(group)))
    return results


def _live_container_ids() -> frozenset[str] | None:
    """Full ids of every container in a not-yet-dead state; ``None`` if unreachable.

    One ``podman ps -a`` call; reads the raw ``State`` JSON field (the
    ``{{.State}}`` template is presentation-only and version-unstable).
    """
    podman = shutil.which("podman")
    if podman is None:
        return None
    try:
        out = subprocess.run(  # noqa: S603  # nosec B603 — fixed argv, no user input
            [podman, "ps", "--all", "--no-trunc", "--format", "json"],
            capture_output=True,
            text=True,
            timeout=10,
            check=True,
        ).stdout
    except (OSError, subprocess.SubprocessError):
        return None
    try:
        rows = json.loads(out) or []
    except ValueError:
        return None
    return frozenset(
        row["Id"]
        for row in rows
        if isinstance(row.get("Id"), str) and row.get("State") in _ALIVE_STATES
    )


def _scan_supervisor_groups() -> dict[str, _Group]:
    """Map ``container_id -> _Group`` for every supervisor-family process found."""
    groups: dict[str, _Group] = {}
    for pid, args in _iter_process_argvs():
        container_id = _container_id_of(args)
        if container_id is None:
            continue
        try:
            pgid = os.getpgid(pid)
        except OSError:
            continue  # vanished mid-scan
        group = groups.setdefault(container_id, _Group(container_id))
        group.pgids.add(pgid)
        age = _process_age_s(pid)
        if age is not None:
            group.youngest_age_s = min(group.youngest_age_s, age)
    return groups


def _container_id_of(args: list[bytes]) -> str | None:
    """The container id an argv belongs to, or ``None`` if it isn't a supervisor process.

    Two shapes carry it: a service child (``… -m terok_sandbox
    supervise-child <service> <container_id> …``) and the restart-loop
    wrapper (``… supervisor_wrapper.py <container_id> <sidecar>``).
    """
    if _CHILD_VERB in args and _MODULE_MARK in args:
        idx = args.index(_CHILD_VERB)
        if idx + 2 < len(args):
            return args[idx + 2].decode(errors="replace")
    if any(arg.endswith(_WRAPPER_MARK) for arg in args):
        idx = next(i for i, arg in enumerate(args) if arg.endswith(_WRAPPER_MARK))
        if idx + 1 < len(args):
            return args[idx + 1].decode(errors="replace")
    return None


def _kill_group(group: _Group) -> str | None:
    """SIGTERM every process group of *group*, escalate to SIGKILL past the grace.

    SIGTERM first so the supervisor tears its children down gracefully
    (the vault child closes its DB connection instead of dying mid-write);
    survivors are SIGKILLed.  Returns the first error encountered, or
    ``None`` on a clean reap.
    """
    error: str | None = None
    for pgid in group.pgids:
        with contextlib.suppress(ProcessLookupError):
            os.killpg(pgid, signal.SIGTERM)
    deadline = time.monotonic() + _KILL_GRACE_S
    while any(_group_alive(pgid) for pgid in group.pgids) and time.monotonic() < deadline:
        time.sleep(0.2)
    for pgid in group.pgids:
        if not _group_alive(pgid):
            continue
        try:
            os.killpg(pgid, signal.SIGKILL)
        except ProcessLookupError:
            pass
        except OSError as exc:
            error = error or f"SIGKILL failed: {exc}"
    return error


def _group_alive(pgid: int) -> bool:
    """Signal-0 probe: does any member of process group *pgid* survive?"""
    try:
        os.killpg(pgid, 0)
    except OSError:
        return False
    return True


def _iter_process_argvs() -> list[tuple[int, list[bytes]]]:
    """Return ``(pid, argv_elements)`` for every process whose cmdline is readable.

    Bytes, not text — ``/proc/*/cmdline`` may hold arbitrary byte
    sequences for foreign processes, and a decode error must not abort
    the sweep.
    """
    found: list[tuple[int, list[bytes]]] = []
    for proc_dir in _PROC_DIR.glob("[0-9]*"):
        with contextlib.suppress(OSError, ValueError):
            raw = (proc_dir / "cmdline").read_bytes()
            found.append((int(proc_dir.name), raw.rstrip(b"\x00").split(b"\x00")))
    return found


def _process_age_s(pid: int) -> float | None:
    """Seconds since *pid* started, from ``/proc`` — ``None`` if it can't be read.

    ``starttime`` (stat field 22, in clock ticks since boot) against the
    boot time gives wall-clock age without a ``psutil`` dependency.  The
    comm field can contain spaces and parentheses, so the parse anchors
    on the final ``)`` rather than splitting naively.
    """
    try:
        stat = (_PROC_DIR / str(pid) / "stat").read_text()
        uptime = float((_PROC_DIR / "uptime").read_text().split()[0])
    except (OSError, ValueError, IndexError):
        return None
    try:
        fields = stat[stat.rindex(")") + 2 :].split()
        starttime_ticks = int(fields[19])  # field 22, 0-indexed after comm+state
    except (ValueError, IndexError):
        return None
    return max(0.0, uptime - starttime_ticks / _CLOCK_TICKS)


def make_orphan_supervisor_check() -> DoctorCheck:
    """A host-side doctor check that reaps supervisor trees without a live container.

    Host-level like the stray-sidecar sweep — one reconciliation per
    install, not per task — so top-level callers append it rather than
    it living in the per-container check list.
    """
    from ..doctor import CheckVerdict, DoctorCheck

    def _eval(_rc: int, _stdout: str, _stderr: str) -> CheckVerdict:
        reaped = reap_orphaned_supervisors()
        if not reaped:
            return CheckVerdict("ok", "no orphaned supervisor trees")
        failed = [cid for cid, err in reaped if err is not None]
        detail = f"reaped {len(reaped)} orphaned supervisor tree(s): " + ", ".join(
            cid[:12] for cid, _ in reaped
        )
        if failed:
            return CheckVerdict(
                "warn",
                detail + f" ({len(failed)} would not die: {', '.join(c[:12] for c in failed)})",
            )
        return CheckVerdict("ok", detail)

    return DoctorCheck(
        category="env",
        label="Orphaned supervisor sweep",
        probe_cmd=[],
        evaluate=_eval,
        host_side=True,
    )


__all__ = ["make_orphan_supervisor_check", "reap_orphaned_supervisors"]
