#!/usr/bin/env python3

# SPDX-FileCopyrightText: 2026 Jiri Vyskocil
# SPDX-License-Identifier: Apache-2.0
"""OCI hook: spawn / reap the per-container terok supervisor.

Annotation-driven by design: the only thing the hook needs to find
its work is the ``terok.sandbox.sidecar`` OCI annotation that the
launch path emits.  The annotation value is the absolute path to the
sidecar JSON.  Every other path (logs / PID files / wrapper) is
derived from that single anchor — no ``$XDG_*`` guessing, no stamp
files, no parallel root resolution.

Layout assumed:

    <root>/sidecar/<name>.json     # the annotation value
    <root>/logs/<container_id>.log
    <root>/pids/supervisor-<container_id>.pid
    <root>/supervisor_wrapper.py

Where the supervisor runs is decided by one fact, read the same way
on the launcher side: a per-user systemd manager that answers makes it
a transient user unit of that manager, in the operator's own
namespaces; otherwise it is a daemon inside the container runtime's
user namespace, where this hook itself runs.

Soft-fails on every error path: a missing sidecar, unreachable
session bus, failed Popen all log and return normally so the
container still starts.  ``terok-shield``'s nft hook is fail-closed
independently — egress protection survives a broken supervisor.

OCI hooks fire per podman run-cycle, not per container lifetime:
createRuntime re-fires on every ``podman start`` and poststop on
every stop.  The poststop reap therefore leaves the sidecar on disk —
it is the only wiring that matches the container's immutable env on
the next start, and removing it here is what used to make restarted
containers come up unsupervised.

Stdlib-only by design, except for the sibling-module import of
``_supervisor_state`` shipped to the same hooks directory at install
time.
"""

import contextlib
import json
import os
import pwd
import signal
import stat
import subprocess  # nosec B404
import sys
import time
from pathlib import Path

# Sibling-module import — same convention shield's hooks use.
sys.path.insert(0, str(Path(__file__).resolve().parent))
import _supervisor_state  # noqa: E402 — sys.path bootstrap precedes import

#: OCI annotation key carrying the absolute sidecar JSON path.  The
#: hook descriptor's ``when`` matches on the same key; if it's absent
#: or empty crun never invokes us.
_SIDECAR_ANNOTATION = "terok.sandbox.sidecar"

#: SIGTERM→SIGKILL grace: how long to wait for the wrapper to exit
#: cleanly after poststop sends SIGTERM.  10 × 0.2s poll intervals = 2s.
_REAP_POLL_INTERVAL_S = 0.2
_REAP_POLL_TICKS = 10

#: argv element marking a process-per-service child (the launcher spawns
#: every service as ``<python> -P -m terok_sandbox supervise-child
#: <service> <container_id> <sidecar_path>``).
_CHILD_VERB_MARK = b"supervise-child"

#: Where the stray-children sweep reads process argvs from (patchable in tests).
_PROC_DIR = Path("/proc")

#: The pinned part of the spawn env a user unit receives — the rest of
#: the unit's environment is the user manager's own, never the hook's.
_UNIT_ENV = ("PATH", "XDG_RUNTIME_DIR", "HOME", "DBUS_SESSION_BUS_ADDRESS")

#: How the user manager hardens the unit.  ``KeyringMode=inherit`` is the
#: whole point of the placement: the unit reads the operator's user
#: keyring, and any private keyring would hide it.  Every filesystem
#: sandbox (``ProtectSystem``, ``PrivateTmp``, …) is out for the same
#: reason — a user service gets one only through ``PrivateUsers``, a
#: fresh user namespace where that keyring is empty again — and
#: ``RestrictNamespaces`` is out because the verdict child enters the
#: container's namespaces through podman.  What is left matches what
#: the children already set on themselves, now for the parent too.
_UNIT_PROPERTIES = (
    "KeyringMode=inherit",
    "NoNewPrivileges=yes",
    "RestrictSUIDSGID=yes",
    "RestrictRealtime=yes",
    "LockPersonality=yes",
    "SystemCallArchitectures=native",
    "UMask=0077",
    "TimeoutStopSec=15",
)


def main() -> None:
    """OCI hook entry point — soft-fail on every error path."""
    host_uid = _supervisor_state.outer_host_uid()
    _supervisor_state.bootstrap_env(host_uid)
    stage = sys.argv[1] if len(sys.argv) > 1 else "createRuntime"
    try:
        oci = json.load(sys.stdin)
    except ValueError as exc:
        _supervisor_state.log(f"terok-sandbox supervisor hook: bad OCI state: {exc}")
        return
    if not isinstance(oci, dict):
        _supervisor_state.log("terok-sandbox supervisor hook: OCI state must be a JSON object")
        return

    container_id = str(oci.get("id") or "")
    annotations = oci.get("annotations") or {}
    if not isinstance(annotations, dict):
        annotations = {}
    sidecar_raw = annotations.get(_SIDECAR_ANNOTATION) or ""
    if not container_id or not sidecar_raw:
        return  # not a terok-managed container — silent no-op

    sidecar_path = _validate_sidecar_path(sidecar_raw)
    if sidecar_path is None:
        return

    # The OCI state carries the container init's host-PID at createRuntime;
    # it's the supervisor's authoritative container-death signal.  Absent
    # or malformed → the supervisor falls back to the podman-wait watch.
    container_pid = oci.get("pid")
    if not isinstance(container_pid, int) or container_pid <= 0:
        container_pid = None

    try:
        _dispatch(stage, container_id, sidecar_path, host_uid, container_pid)
    except Exception as exc:  # noqa: BLE001 — soft-fail every path
        _supervisor_state.log(f"terok-sandbox supervisor hook: {exc}")


def _validate_sidecar_path(raw: str) -> Path | None:
    """Refuse annotation values that don't look like a terok-owned sidecar.

    The OCI annotation is operator-controlled at ``podman run`` time
    (anything in ``--annotation`` lands here verbatim), so a misuse or
    a hostile invocation could point the hook at ``/etc/passwd`` or a
    file owned by another user.  We require:

    * absolute path with no ``..`` segments;
    * resolves to an existing regular file (not a symlink chain);
    * the resolved path's parent directory is literally named
      ``sidecar``;
    * the file is owned by the in-namespace effective UID (the operator
      the hook is acting on behalf of) — see the ownership check below.

    Soft-fails (returns ``None``) on any violation — container start
    continues, just without a supervisor.
    """
    if not raw or ".." in Path(raw).parts or not Path(raw).is_absolute():
        _supervisor_state.log(
            f"terok-sandbox supervisor hook: rejecting sidecar annotation {raw!r}"
        )
        return None
    try:
        resolved = Path(raw).resolve(strict=True)
        st = resolved.lstat()
    except OSError as exc:
        _supervisor_state.log(f"terok-sandbox supervisor hook: sidecar annotation unusable: {exc}")
        return None
    if not stat.S_ISREG(st.st_mode):
        _supervisor_state.log(
            f"terok-sandbox supervisor hook: sidecar is not a regular file: {resolved}"
        )
        return None
    if resolved.parent.name != "sidecar":
        _supervisor_state.log(
            f"terok-sandbox supervisor hook: sidecar parent dir not 'sidecar': {resolved}"
        )
        return None
    # Ownership check: in NS_ROOTLESS the operator's host UID maps to
    # in-namespace UID 0, so the operator-owned sidecar stats as uid 0
    # here; files outside the user's uid_map range (e.g. host-root-owned)
    # appear as overflow uid (typically 65534) and are rejected.  In init
    # userns ``geteuid()`` is the actual operator UID and the same equality
    # holds.  ``outer_host_uid()`` (used for the spawn env) is the *wrong*
    # uid space for this comparison — it would never match under crun.
    if st.st_uid != os.geteuid():
        _supervisor_state.log(
            f"terok-sandbox supervisor hook: sidecar not owned by current uid "
            f"(uid={st.st_uid} != euid={os.geteuid()}): {resolved}"
        )
        return None
    return resolved


def _dispatch(
    stage: str,
    container_id: str,
    sidecar_path: Path,
    host_uid: int,
    container_pid: int | None = None,
) -> None:
    """Dispatch by stage — spawn at createRuntime, reap at poststop."""
    root = sidecar_path.parent.parent  # <root>/sidecar/<name>.json → <root>
    # Both anchors are now known, so arm the persistent hook diary before
    # any log() below — a soft-fail from here on leaves a durable, tagged
    # trace, not just swallowed runtime stderr.
    _supervisor_state.set_log_context(root, container_id)
    if stage == "poststop":
        _reap_supervisor(container_id, root, host_uid)
        return
    if stage != "createRuntime":
        _supervisor_state.log(f"terok-sandbox supervisor hook: unknown stage {stage!r}")
        return

    # Parse the sidecar to abort early on an unreadable / malformed file;
    # the wrapper re-reads it from the path, so the parsed dict isn't
    # threaded further.
    if _load_sidecar(sidecar_path) is None:
        return
    _spawn_supervisor(container_id, sidecar_path, root, host_uid, container_pid)


def _spawn_supervisor(
    container_id: str,
    sidecar_path: Path,
    root: Path,
    host_uid: int,
    container_pid: int | None = None,
) -> None:
    """Start the supervisor wrapper for *container_id* where this host places it.

    A user unit when the per-user manager answers, a detached daemon
    otherwise — and a daemon when the manager refuses the unit, so a
    container never starts unsupervised over a placement detail.
    *container_pid* (the container init host-PID from the OCI state) is
    passed to the wrapper as an optional 3rd positional so the supervisor
    can watch it directly; ``None`` simply omits it.
    """
    wrapper_path = Path(__file__).resolve().parent.parent / "supervisor_wrapper.py"
    if not wrapper_path.is_file():
        _supervisor_state.log(
            f"terok-sandbox supervisor hook: wrapper missing at {wrapper_path} "
            "— rerun `terok-sandbox setup`"
        )
        return

    pid_file = root / "pids" / f"supervisor-{container_id}.pid"
    log_file = root / "logs" / f"{container_id}.log"
    try:
        pid_file.parent.mkdir(parents=True, exist_ok=True)
        log_file.parent.mkdir(parents=True, exist_ok=True)
    except OSError as exc:
        _supervisor_state.log(f"terok-sandbox supervisor hook: state dir setup failed: {exc}")
        return

    if _supervisor_alive(pid_file, wrapper_path, container_id):
        _supervisor_state.log(
            f"terok-sandbox supervisor hook: supervisor already alive for {container_id}"
            " — idempotent respawn, nothing to do"
        )
        return

    env = _spawn_env(host_uid)
    wrapper_argv = [sys.executable, "-I", str(wrapper_path), container_id, str(sidecar_path)]
    if container_pid is not None:
        wrapper_argv.append(str(container_pid))

    unit: str | None = None
    pid: int | None = None
    if _supervisor_state.user_manager_reachable(_supervisor_state.user_runtime_dir(host_uid)):
        unit = _supervisor_state.unit_name(container_id)
        # No live wrapper answered above, so a unit still wearing this name
        # is a leftover; clearing it is what keeps the refusal below rare.
        _supervisor_state.stop_unit(unit)
        pid = _spawn_unit(unit, wrapper_argv, env, log_file)
        if pid == 0:
            return
    if pid is None:
        if unit is not None:
            _supervisor_state.log(
                "terok-sandbox supervisor hook: falling back to a namespace daemon — its vault"
                " and signer will not read the kernel-keyring cache this host's launcher uses"
            )
        unit = None
        pid = _spawn_daemon(wrapper_argv, env, log_file)
    if pid is None:
        return

    try:
        pid_file.write_text(f"{pid}\n")
    except OSError as exc:
        # The PID file is the only handle poststop has to reap a daemon
        # (a unit is also reachable by name).  Losing it would orphan the
        # supervisor, so take it down synchronously right here.
        _supervisor_state.log(f"terok-sandbox supervisor hook: pid file write failed: {exc}")
        if unit is not None:
            _supervisor_state.stop_unit(unit)
        else:
            _kill_daemon(pid)
    else:
        # Success — recorded so a non-empty hook.log means "the hook fired
        # and here is what it did", and an empty/absent one means it never
        # ran at all: exactly the fork a stuck-unsupervised container turns on.
        placement = f"user unit {unit}" if unit is not None else "a namespace daemon"
        _supervisor_state.log(
            f"terok-sandbox supervisor hook: spawned supervisor pid {pid} for {container_id}"
            f" as {placement}"
        )


def _spawn_unit(
    unit: str, wrapper_argv: list[str], env: dict[str, str], log_file: Path
) -> int | None:
    """Ask the user manager to run the wrapper as transient unit *unit*.

    The unit collects itself when it ends, appends its output to the same
    per-container log the daemon placement writes, is hardened by
    ``_UNIT_PROPERTIES``, and carries the effective host PATH and session identity in
    ``_UNIT_ENV`` — the manager's own environment is the rest, never the
    runtime's hook env.  Returns the unit's main PID; ``None`` when the
    manager refused, so the caller falls back to a daemon; ``0`` when the
    unit started but its process is already gone — nothing to record and
    nothing to fall back to, the wrapper's own exit is in the log.
    """
    systemd_run = _supervisor_state.find_host_tool("systemd-run")
    systemctl = _supervisor_state.find_host_tool("systemctl")
    if systemd_run is None or systemctl is None:
        return None
    argv = [
        systemd_run,
        "--user",
        "--quiet",
        "--collect",
        f"--unit={unit}",
        f"--property=StandardOutput=append:{log_file}",
        f"--property=StandardError=append:{log_file}",
        *(f"--property={prop}" for prop in _UNIT_PROPERTIES),
        *(f"--setenv={key}={env[key]}" for key in _UNIT_ENV if key in env),
        *wrapper_argv,
    ]
    try:
        run = subprocess.run(  # noqa: S603  # nosec B603 B607
            argv, capture_output=True, text=True, check=False
        )
        if run.returncode != 0:
            _supervisor_state.log(
                f"terok-sandbox supervisor hook: systemd-run refused {unit}"
                f" ({run.stderr.strip() or run.returncode})"
            )
            return None
        shown = subprocess.run(  # noqa: S603  # nosec B603 B607
            [systemctl, "--user", "show", "--property=MainPID", "--value", unit],
            capture_output=True,
            text=True,
            check=False,
        )
    except OSError as exc:
        _supervisor_state.log(f"terok-sandbox supervisor hook: user manager unreachable: {exc}")
        return None
    try:
        pid = int(shown.stdout.strip())
    except ValueError:
        pid = 0
    if pid <= 0:
        _supervisor_state.log(
            f"terok-sandbox supervisor hook: {unit} started but its wrapper is already gone"
        )
    return max(pid, 0)


def _spawn_daemon(wrapper_argv: list[str], env: dict[str, str], log_file: Path) -> int | None:
    """Detach the wrapper as its own session inside the runtime's namespace.

    The floor placement: the process group the new session opens is what
    poststop signals, and the log file gets the wrapper's output directly.
    """
    try:
        log_fh = log_file.open("ab")
    except OSError as exc:
        _supervisor_state.log(f"terok-sandbox supervisor hook: cannot open log file: {exc}")
        return None
    try:
        proc = subprocess.Popen(  # noqa: S603  # nosec B603
            wrapper_argv,
            env=env,
            stdin=subprocess.DEVNULL,
            stdout=log_fh,
            stderr=log_fh,
            start_new_session=True,
            close_fds=True,
        )
    except OSError as exc:
        _supervisor_state.log(f"terok-sandbox supervisor hook: wrapper spawn failed: {exc}")
        return None
    finally:
        log_fh.close()
    return proc.pid


def _kill_daemon(pid: int) -> None:
    """SIGTERM *pid*, poll briefly, escalate to SIGKILL if it does not oblige."""
    with contextlib.suppress(ProcessLookupError, OSError):
        os.kill(pid, signal.SIGTERM)
    for _ in range(_REAP_POLL_TICKS):
        time.sleep(_REAP_POLL_INTERVAL_S)
        if not _supervisor_state.pid_exists(pid):
            break
    else:
        with contextlib.suppress(ProcessLookupError, OSError):
            os.kill(pid, signal.SIGKILL)


def _spawn_env(host_uid: int) -> dict[str, str]:
    """Compose the env the wrapper subprocess inherits.

    Pins ``XDG_RUNTIME_DIR`` to the well-known per-user runtime dir,
    ``HOME`` to the operator's real home directory, and
    ``DBUS_SESSION_BUS_ADDRESS`` to the operator's session bus (the
    desktop notifier needs it).  Everything else flows from the sidecar
    via the wrapper's argv — the env is intentionally minimal.

    ``HOME`` must be pinned, not inherited: old OCI runtimes (crun 0.17,
    Ubuntu 22.04) hand hooks the *container's* process env, whose
    ``HOME=/root`` sends every ``~``-derived path the supervisor resolves
    (vault credentials DB, SSH signer keys) into the real root's home —
    ``EPERM`` from inside the rootless namespace.  The passwd lookup uses
    the outer host UID against the host's ``/etc/passwd`` (hooks run with
    the host filesystem view even in ``NS_ROOTLESS``).
    """
    env = dict(os.environ)
    runtime = _supervisor_state.user_runtime_dir(host_uid)
    env["XDG_RUNTIME_DIR"] = str(runtime)
    with contextlib.suppress(KeyError):
        env["HOME"] = pwd.getpwuid(host_uid).pw_dir
    if not env.get("DBUS_SESSION_BUS_ADDRESS"):
        bus_path = runtime / "bus"
        if bus_path.exists():
            env["DBUS_SESSION_BUS_ADDRESS"] = f"unix:path={bus_path}"
    return env


def _supervisor_alive(pid_file: Path, wrapper_path: Path, container_id: str) -> bool:
    """Return whether *pid_file* names a live wrapper *for this container*."""
    try:
        pid = int(pid_file.read_text().strip())
    except (OSError, ValueError):
        return False
    if not _supervisor_state.pid_exists(pid):
        return False
    return _is_our_wrapper(pid, str(wrapper_path), container_id)


def _reap_supervisor(container_id: str, root: Path, host_uid: int) -> None:
    """Take the supervisor down at poststop: its unit by name, its daemon by group.

    A user unit stops through the manager first; the group reap below then
    finds its PID gone and clears the file.  The daemon placement is the
    group reap alone.

    The createRuntime hook spawns a daemon wrapper with
    ``start_new_session=True``, so the PID it records is also the
    **process-group ID** of the container's entire supervisor tree —
    restart-loop wrapper, supervisor, service children, watcher
    subprocesses.  Signalling the group is what actually delivers
    SIGTERM to the supervisor (the wrapper's restart loop never
    forwarded signals), and it still reaches the members after the
    wrapper died: a group persists while any member lives, and its ID
    cannot be recycled meanwhile.  A final argv sweep
    (``_reap_stray_children``) nets children whose PID file is already
    gone — an earlier reap unlinked it while the kill half failed.

    The sidecar JSON deliberately survives the stop: the container's
    env (TCP ports, gate token, phantom credential tokens) is
    immutable after ``podman run``, so the preserved sidecar is the
    only wiring the supervisor may come back with when createRuntime
    re-fires on the next ``podman start``.  Sidecar removal belongs to
    real teardown —
    [`remove_container_state`][terok_sandbox.launch.remove_container_state]
    at cleanup / task delete, or the doctor's stray sweep.
    """
    pid_file = root / "pids" / f"supervisor-{container_id}.pid"
    wrapper_path = Path(__file__).resolve().parent.parent / "supervisor_wrapper.py"
    if _supervisor_state.user_manager_reachable(_supervisor_state.user_runtime_dir(host_uid)):
        _supervisor_state.stop_unit(_supervisor_state.unit_name(container_id))
    try:
        _reap_group(pid_file, wrapper_path, container_id)
    finally:
        _reap_stray_children(container_id)


def _reap_group(pid_file: Path, wrapper_path: Path, container_id: str) -> None:
    """The PID-file half of the reap: group SIGTERM → poll → group SIGKILL."""
    try:
        pgid = int(pid_file.read_text().strip())
    except (OSError, ValueError):
        pid_file.unlink(missing_ok=True)
        return
    # ``killpg(0)`` would be this hook's own group, ``killpg(-1)`` every
    # process it may signal: a file that says so is corrupt, not a target.
    if pgid <= 0 or not _is_group_ours(pgid, str(wrapper_path), container_id):
        pid_file.unlink(missing_ok=True)
        return
    try:
        os.killpg(pgid, signal.SIGTERM)
    except ProcessLookupError:
        pid_file.unlink(missing_ok=True)
        return
    except OSError as exc:
        # SIGTERM failed for a reason other than the group being gone
        # (e.g. EPERM). Members may still be live, so keep the pidfile
        # — it's the only handle a later reap has to retry.
        _supervisor_state.log(f"terok-sandbox supervisor hook: group SIGTERM failed: {exc}")
        return

    for _ in range(_REAP_POLL_TICKS):
        time.sleep(_REAP_POLL_INTERVAL_S)
        if not _group_exists(pgid):
            break
    else:
        with contextlib.suppress(ProcessLookupError, OSError):
            os.killpg(pgid, signal.SIGKILL)
    pid_file.unlink(missing_ok=True)


def _group_exists(pgid: int) -> bool:
    """Signal-0 probe: does any member of process group *pgid* survive?"""
    try:
        os.killpg(pgid, 0)
    except OSError:
        return False
    return True


def _is_group_ours(pgid: int, wrapper_path: str, container_id: str) -> bool:
    """Recycle guard for the whole group — strict on a live leader, permissive on a dead one.

    Leader alive → its argv must be our wrapper *for this container*
    (same double-mark check as ``_is_our_wrapper``).  Leader gone or a
    zombie (no ``/proc`` entry, or the empty cmdline zombies expose) →
    ``True``: a group ID stays pinned while any member lives, so a
    surviving group under this number can only be the remnant of our
    own session — and if nothing survives either, ``killpg`` lands on
    ``ProcessLookupError`` harmlessly.
    """
    try:
        raw = Path(f"/proc/{pgid}/cmdline").read_bytes()
    except OSError:
        return True
    if not raw:
        return True
    args = raw.rstrip(b"\x00").split(b"\x00")
    return wrapper_path.encode() in args and container_id.encode() in args


def _reap_stray_children(container_id: str) -> None:
    """SIGTERM → poll → SIGKILL the service children a teardown left behind.

    Normally a no-op: the wrapper's own SIGTERM handler brings its
    children down before it exits.  Children survive when the wrapper
    was SIGKILLed past the grace window, or crashed without a teardown.
    They are found by argv (``… -m terok_sandbox supervise-child
    <service> <container_id> …``), scoped to *this* container so a
    busy host never loses another container's live bundle.
    """
    strays = _find_stray_children(container_id)
    if not strays:
        return
    _supervisor_state.log(
        f"terok-sandbox supervisor hook: reaping {len(strays)} stray service"
        f" child(ren) of {container_id}: {strays}"
    )
    for pid in strays:
        with contextlib.suppress(ProcessLookupError, OSError):
            os.kill(pid, signal.SIGTERM)
    for _ in range(_REAP_POLL_TICKS):
        time.sleep(_REAP_POLL_INTERVAL_S)
        strays = [pid for pid in strays if _supervisor_state.pid_exists(pid)]
        if not strays:
            return
    for pid in strays:
        with contextlib.suppress(ProcessLookupError, OSError):
            os.kill(pid, signal.SIGKILL)


def _find_stray_children(container_id: str) -> list[int]:
    """PIDs of live ``supervise-child`` processes belonging to *container_id*.

    Bytes comparison against exact argv elements — same PID-recycle
    discipline as ``_is_our_wrapper``, and immune to arbitrary byte
    sequences in foreign processes' cmdlines.
    """
    strays: list[int] = []
    for proc_dir in _PROC_DIR.glob("[0-9]*"):
        try:
            raw = (proc_dir / "cmdline").read_bytes()
        except OSError:
            continue
        args = raw.rstrip(b"\x00").split(b"\x00")
        if _CHILD_VERB_MARK in args and container_id.encode() in args:
            strays.append(int(proc_dir.name))
    return strays


def _is_our_wrapper(pid: int, wrapper_path: str, container_id: str) -> bool:
    """``True`` if ``/proc/<pid>/cmdline`` is *our* wrapper for *this* container.

    Cross-checks the wrapper path **and** the container_id argv element
    — without the latter, a PID-recycle into a wrapper started for a
    different container would falsely match (every wrapper shares the
    same wrapper_path).
    """
    try:
        raw = Path(f"/proc/{pid}/cmdline").read_bytes()
    except OSError:
        return False
    args = raw.rstrip(b"\x00").split(b"\x00")
    return wrapper_path.encode() in args and container_id.encode() in args


def _load_sidecar(sidecar_path: Path) -> dict | None:
    """Read and JSON-parse the sidecar at *sidecar_path*; log + ``None`` on failure."""
    try:
        with sidecar_path.open(encoding="utf-8") as fh:
            data = json.load(fh)
    except (OSError, ValueError) as exc:
        _supervisor_state.log(f"terok-sandbox supervisor hook: cannot read {sidecar_path}: {exc}")
        return None
    if not isinstance(data, dict):
        _supervisor_state.log(
            f"terok-sandbox supervisor hook: sidecar must be JSON object: {sidecar_path}"
        )
        return None
    return data


if __name__ == "__main__":  # pragma: no cover
    main()  # always returns None — see ``main`` docstring
