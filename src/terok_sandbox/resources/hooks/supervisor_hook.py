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

Soft-fails on every error path: a missing sidecar, unreachable
session bus, failed Popen all log and return normally so the
container still starts.  ``terok-shield``'s nft hook is fail-closed
independently — egress protection survives a broken supervisor.

Stdlib-only by design, except for the sibling-module import of
``_supervisor_state`` shipped to the same hooks directory at install
time.
"""

import contextlib
import json
import os
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

    sidecar_path = _validate_sidecar_path(sidecar_raw, host_uid)
    if sidecar_path is None:
        return

    try:
        _dispatch(stage, container_id, sidecar_path, host_uid)
    except Exception as exc:  # noqa: BLE001 — soft-fail every path
        _supervisor_state.log(f"terok-sandbox supervisor hook: {exc}")


def _validate_sidecar_path(raw: str, host_uid: int) -> Path | None:
    """Refuse annotation values that don't look like a terok-owned sidecar.

    The OCI annotation is operator-controlled at ``podman run`` time
    (anything in ``--annotation`` lands here verbatim), so a misuse or
    a hostile invocation could point the hook at ``/etc/passwd`` or a
    file the supervisor would read with ``host_uid``'s privileges.
    We require:

    * absolute path with no ``..`` segments;
    * resolves to an existing regular file (not a symlink chain);
    * the resolved path's parent directory is literally named
      ``sidecar``;
    * the file is owned by *host_uid* (the operator the hook is acting
      on behalf of).

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
    if st.st_uid != host_uid:
        _supervisor_state.log(
            f"terok-sandbox supervisor hook: sidecar owner uid {st.st_uid} != host uid {host_uid}: "
            f"{resolved}"
        )
        return None
    return resolved


def _dispatch(stage: str, container_id: str, sidecar_path: Path, host_uid: int) -> None:
    """Dispatch by stage — spawn at createRuntime, reap at poststop."""
    root = sidecar_path.parent.parent  # <root>/sidecar/<name>.json → <root>
    if stage == "poststop":
        _reap_supervisor(container_id, root, sidecar_path)
        return
    if stage != "createRuntime":
        _supervisor_state.log(f"terok-sandbox supervisor hook: unknown stage {stage!r}")
        return

    sidecar = _load_sidecar(sidecar_path)
    if sidecar is None:
        return
    _spawn_supervisor(container_id, sidecar_path, sidecar, root, host_uid)


def _spawn_supervisor(
    container_id: str,
    sidecar_path: Path,
    sidecar: dict,
    root: Path,
    host_uid: int,
) -> None:
    """Start the supervisor wrapper for *container_id* as a detached child."""
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
        return  # idempotent respawn

    env = _spawn_env(host_uid)
    try:
        log_fh = log_file.open("ab")
    except OSError as exc:
        _supervisor_state.log(f"terok-sandbox supervisor hook: cannot open log file: {exc}")
        return

    try:
        proc = subprocess.Popen(  # noqa: S603  # nosec B603
            ["/usr/bin/python3", str(wrapper_path), container_id, str(sidecar_path)],
            env=env,
            stdin=subprocess.DEVNULL,
            stdout=log_fh,
            stderr=log_fh,
            start_new_session=True,
            close_fds=True,
        )
    except OSError as exc:
        log_fh.close()
        _supervisor_state.log(f"terok-sandbox supervisor hook: wrapper spawn failed: {exc}")
        return
    finally:
        log_fh.close()

    try:
        pid_file.write_text(f"{proc.pid}\n")
    except OSError as exc:
        _supervisor_state.log(f"terok-sandbox supervisor hook: pid file write failed: {exc}")
        with contextlib.suppress(ProcessLookupError, OSError):
            os.kill(proc.pid, signal.SIGTERM)


def _spawn_env(host_uid: int) -> dict[str, str]:
    """Compose the env the wrapper subprocess inherits.

    Pins ``DBUS_SESSION_BUS_ADDRESS`` to the operator's session bus
    (the desktop notifier needs it) and ``XDG_RUNTIME_DIR`` to the
    well-known per-user runtime dir.  Everything else flows from the
    sidecar via the wrapper's argv — the env is intentionally minimal.
    """
    env = dict(os.environ)
    runtime = Path(f"/run/user/{host_uid}")
    env["XDG_RUNTIME_DIR"] = str(runtime)
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


def _reap_supervisor(container_id: str, root: Path, sidecar_path: Path) -> None:
    """SIGTERM the wrapper at poststop, SIGKILL if it lingers past 2 s."""
    pid_file = root / "pids" / f"supervisor-{container_id}.pid"
    wrapper_path = Path(__file__).resolve().parent.parent / "supervisor_wrapper.py"
    if not pid_file.exists():
        sidecar_path.unlink(missing_ok=True)
        return
    try:
        pid = int(pid_file.read_text().strip())
    except (OSError, ValueError):
        pid_file.unlink(missing_ok=True)
        sidecar_path.unlink(missing_ok=True)
        return
    if not _is_our_wrapper(pid, str(wrapper_path), container_id):
        pid_file.unlink(missing_ok=True)
        sidecar_path.unlink(missing_ok=True)
        return
    try:
        os.kill(pid, signal.SIGTERM)
    except ProcessLookupError:
        pid_file.unlink(missing_ok=True)
        sidecar_path.unlink(missing_ok=True)
        return
    except OSError as exc:
        _supervisor_state.log(f"terok-sandbox supervisor hook: SIGTERM failed: {exc}")
        pid_file.unlink(missing_ok=True)
        sidecar_path.unlink(missing_ok=True)
        return

    for _ in range(_REAP_POLL_TICKS):
        time.sleep(_REAP_POLL_INTERVAL_S)
        if not _supervisor_state.pid_exists(pid):
            break
    else:
        with contextlib.suppress(ProcessLookupError, OSError):
            os.kill(pid, signal.SIGKILL)
    pid_file.unlink(missing_ok=True)
    sidecar_path.unlink(missing_ok=True)


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
