# SPDX-FileCopyrightText: 2026 Jiri Vyskocil
# SPDX-License-Identifier: Apache-2.0

"""Reconcile stray supervisor process trees against live containers.

The per-container supervisor is meant to die with its container — the
``poststop`` hook group-kills the tree, the supervisor self-terminates
when its container's init PID dies, and ``PR_SET_PDEATHSIG`` takes the
service children down with it.  Those are the *prevention* layers.  This
module is the *reconciliation* backstop: a periodic sweep that finds
supervisor trees whose container is no longer running and kills them, no
matter how they were stranded (a ``poststop`` that never fired, a host
crash that left the tree, or a supervisor built before the prevention
layers existed).

Unlike the OCI hook — which crun hands the container's environment, so its
nested ``podman`` calls are unreliable — the janitor runs from an ordinary
host CLI context (``doctor``, a task launch), where ``podman ps`` answers
normally.  That is what lets it use container liveness as the ground truth
the hook cannot.

A tree is identified structurally: the wrapper is a session leader (the
OCI hook spawns it with ``start_new_session=True``), and every descendant
— supervisor, service children — inherits that process group, so one
``killpg`` per group takes the whole tree down.  Groups are matched by the
container id carried in each process's argv, and each member's identity
(PID + start time) is revalidated immediately before signalling so a PGID
recycled between the scan and the kill is never touched.
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
from typing import TYPE_CHECKING, NamedTuple

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


class _Member(NamedTuple):
    """One supervisor-family process, with the identity used to revalidate it."""

    pid: int
    starttime_ticks: int
    pgid: int


@dataclass
class _Group:
    """One container's supervisor tree, as found on the host."""

    container_id: str
    members: list[_Member] = field(default_factory=list)
    age_known: bool = True
    """``False`` once any member's age couldn't be read — the group's true
    age is then unknown, so it is excluded from reaping (a mid-creation tree
    must not be killed on a guess)."""
    youngest_age_s: float = float("inf")


def reap_orphaned_supervisors(
    *, min_age_s: float = _ORPHAN_GRACE_S
) -> list[tuple[str, str | None]] | None:
    """Kill every supervisor tree whose container is no longer running.

    Returns one ``(container_id, error_or_None)`` row per tree reaped
    (``None`` error when it went down cleanly), or an empty list when
    nothing was stray — the steady state.  Returns **``None``** when podman
    was unreachable: liveness is then unknown, so the sweep does nothing and
    the caller must not read that as "all clean" (distinct from the empty
    list, which means podman answered and found no strays).

    A tree is reaped only when its container is absent from the live set,
    its age is known and past *min_age_s*, and — at signal time — an
    original member still occupies the PGID.
    """
    alive = _live_container_ids()
    if alive is None:
        return None  # podman unreachable — liveness unknown, don't guess
    results: list[tuple[str, str | None]] = []
    for container_id, group in sorted(_scan_supervisor_groups().items()):
        if container_id in alive:
            continue
        if not group.age_known:
            continue  # an unreadable member age → can't rule out a mid-create tree
        if group.youngest_age_s < min_age_s:
            continue  # a just-spawned tree racing its container's registration
        results.append((container_id, _kill_group(group)))
    return results


def make_orphan_supervisor_check() -> DoctorCheck:
    """A host-side doctor check that reaps supervisor trees without a live container.

    Host-level like the stray-sidecar sweep — one reconciliation per
    install, not per task — so top-level callers append it rather than
    it living in the per-container check list.  Podman being unreachable
    is surfaced as a ``warn`` (the sweep could not run), never a silent ok.
    """
    from ..doctor import CheckVerdict, DoctorCheck

    def _eval(_rc: int, _stdout: str, _stderr: str) -> CheckVerdict:
        reaped = reap_orphaned_supervisors()
        if reaped is None:
            return CheckVerdict(
                "warn", "podman unreachable — could not check for orphaned supervisor trees"
            )
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


# ── Internal helpers ────────────────────────────────────────────────────


def _live_container_ids() -> frozenset[str] | None:
    """Full ids of every container in a not-yet-dead state; ``None`` if unreachable.

    One ``podman ps -a`` call; reads the raw ``State`` JSON field (the
    ``{{.State}}`` template is presentation-only and version-unstable).
    ``None`` (podman missing, erroring, or unparsable) means liveness is
    unknown and must stay distinguishable from an empty live set.
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
        identity = _process_identity(pid)
        if identity is None:
            # No readable stat → no age and no identity anchor.  Mark the
            # group's age unknown so it is excluded from reaping.
            group.age_known = False
            continue
        starttime_ticks, age_s = identity
        group.members.append(_Member(pid, starttime_ticks, pgid))
        group.youngest_age_s = min(group.youngest_age_s, age_s)
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
    survivors are SIGKILLed.  Only PGIDs that still host an original member
    (revalidated by PID + start time) are signalled, so a recycled PGID is
    never touched.  A signal that fails for a reason other than the group
    already being gone (e.g. ``EPERM``) is recorded and the sweep continues.
    Returns the first error encountered, or ``None`` on a clean reap.
    """
    error: str | None = None
    for pgid in _live_owned_pgids(group):
        error = _signal_pgid(pgid, signal.SIGTERM, error)
    deadline = time.monotonic() + _KILL_GRACE_S
    while _live_owned_pgids(group) and time.monotonic() < deadline:
        time.sleep(0.2)
    for pgid in _live_owned_pgids(group):
        error = _signal_pgid(pgid, signal.SIGKILL, error)
    return error


def _signal_pgid(pgid: int, sig: int, error: str | None) -> str | None:
    """Send *sig* to *pgid*; fold a non-death failure into *error* (first wins)."""
    try:
        os.killpg(pgid, sig)
    except ProcessLookupError:
        pass  # the group is gone — the intended outcome
    except OSError as exc:
        error = error or f"{signal.Signals(sig).name} failed: {exc}"
    return error


def _live_owned_pgids(group: _Group) -> set[int]:
    """PGIDs of *group* that still host at least one original member.

    A PGID is only signalled while a process we scanned still occupies it
    with its recorded start time — so a PGID freed by the group's exit and
    recycled into an unrelated session is never signalled.
    """
    live: set[int] = set()
    for member in group.members:
        if member.pgid not in live and _member_present(member):
            live.add(member.pgid)
    return live


def _member_present(member: _Member) -> bool:
    """``True`` if *member*'s PID still exists in its PGID with its start time.

    Only a matching (PID, PGID, start time) counts: a recycled PID lands in
    a different PGID or carries a different start time, and either mismatch
    rejects it.
    """
    try:
        if os.getpgid(member.pid) != member.pgid:
            return False
    except OSError:
        return False  # PID gone
    identity = _process_identity(member.pid)
    return identity is not None and identity[0] == member.starttime_ticks


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


def _process_identity(pid: int) -> tuple[int, float] | None:
    """Return ``(starttime_ticks, age_s)`` for *pid* from ``/proc``, or ``None``.

    ``starttime`` (stat field 22, in clock ticks since boot) is both the
    recycle-proof identity anchor and — against the boot time — the
    wall-clock age, computed without a ``psutil`` dependency.  The comm
    field can contain spaces and parentheses, so the parse anchors on the
    final ``)`` rather than splitting naively.
    """
    try:
        stat = (_PROC_DIR / str(pid) / "stat").read_text()
        uptime = float((_PROC_DIR / "uptime").read_text().split()[0])
    except (OSError, ValueError, IndexError):
        return None
    try:
        starttime_ticks = int(stat[stat.rindex(")") + 2 :].split()[19])
    except (ValueError, IndexError):
        return None
    return starttime_ticks, max(0.0, uptime - starttime_ticks / _CLOCK_TICKS)


__all__ = ["make_orphan_supervisor_check", "reap_orphaned_supervisors"]
