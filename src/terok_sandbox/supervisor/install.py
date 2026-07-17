# SPDX-FileCopyrightText: 2026 Jiri Vyskocil
# SPDX-License-Identifier: Apache-2.0

"""Install + uninstall the OCI hook that spawns the supervisor.

Single-root layout: scripts, ballast, and the JSON descriptor all
live under [`state_root()`][terok_sandbox.paths.state_root] (which
honours the operator's ``paths.root`` config).  ``containers.conf``
is patched to list ``state_root() / "hooks"`` in ``hooks_dir`` so
podman scans the canonical terok-owned directory rather than the
default ``~/.config/containers/oci/hooks.d/``.

Files written:

* ``<state_root>/hooks/supervisor_hook.py``
* ``<state_root>/hooks/_supervisor_state.py``
* ``<state_root>/hooks/terok-sandbox-supervisor-createRuntime.json``
* ``<state_root>/hooks/terok-sandbox-supervisor-poststop.json``
  (one OCI hook descriptor per stage — podman/crun reuse the same
  ``hook.args`` for every stage listed in a single descriptor, so each
  stage gets its own JSON; both match on the ``terok.sandbox.sidecar``
  annotation)
* ``<state_root>/supervisor_wrapper.py`` (templated — embeds the
  resolved ``terok-sandbox`` argv)

The supervisor flow is annotation-driven from here on: the launch
path emits ``--annotation terok.sandbox.sidecar=<abspath>`` and the
hook reads the sidecar at that path.  No ``$XDG_*`` discovery,
no stamp files, no parallel root resolution.
"""

from __future__ import annotations

import contextlib
import json
import os
import shutil
import signal
import sys
from pathlib import Path
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from collections.abc import Iterator

from ..integrations.shield import ensure_user_hooks_dir_configured
from ..paths import state_root

_HOOK_STAGES = ("createRuntime", "poststop")
_HOOK_SCRIPT_NAME = "supervisor_hook.py"
_BALLAST_NAME = "_supervisor_state.py"
_WRAPPER_NAME = "supervisor_wrapper.py"


def _descriptor_name(stage: str) -> str:
    """Per-stage filename for the supervisor hook JSON descriptor."""
    return f"terok-sandbox-supervisor-{stage}.json"


#: Per-container supervisor PID files, written by the OCI hook.  Glob
#: pattern is ``supervisor-<container_id>.pid``.
_PIDS_DIR_NAME = "pids"
_PID_GLOB = "supervisor-*.pid"

#: argv fingerprints of a process-per-service child — the launcher spawns
#: every service as ``<python> -P -m terok_sandbox supervise-child
#: <service> <container_id> <sidecar_path>``.
_CHILD_MODULE_MARK = b"terok_sandbox"
_CHILD_VERB_MARK = b"supervise-child"

#: Where the child sweep reads process argvs from (patchable in tests).
_PROC_DIR = Path("/proc")

#: Placeholder in the wrapper template ``install`` rewrites with the
#: resolved ``terok-sandbox`` argv (JSON-encoded list).
_WRAPPER_BIN_PLACEHOLDER = '["__TEROK_SANDBOX_BIN__"]'

#: OCI annotation key that triggers the hook; value is the absolute
#: per-container sidecar JSON path.
_TRIGGER_ANNOTATION = "terok.sandbox.sidecar"


def install_supervisor_hooks(*, hooks_dir: Path | None = None) -> None:
    """Lay down hook scripts, wrapper, and the OCI descriptor.

    *hooks_dir* — override for tests; defaults to
    ``state_root() / "hooks"``, where the role scripts already live.
    Scripts + ballast + descriptor share one terok-owned directory so
    a teardown is a clean ``rm -rf``.  ``containers.conf`` is patched
    to register that path.

    Idempotent — every file write overwrites silently, and the
    descriptor JSON gets re-rendered each time so a moved install
    location is picked up on the next ``terok-sandbox setup``.
    """
    install_root = state_root()
    hooks_install_dir = install_root / "hooks"
    hooks_install_dir.mkdir(parents=True, exist_ok=True)

    pkg_resources = Path(__file__).resolve().parent.parent / "resources"
    pkg_hooks = pkg_resources / "hooks"

    _copy_executable(pkg_hooks / _HOOK_SCRIPT_NAME, hooks_install_dir / _HOOK_SCRIPT_NAME)
    _copy_executable(pkg_hooks / _BALLAST_NAME, hooks_install_dir / _BALLAST_NAME)

    sandbox_argv = _resolve_sandbox_argv()
    _render_wrapper(
        src=pkg_resources / _WRAPPER_NAME,
        dst=install_root / _WRAPPER_NAME,
        sandbox_argv=sandbox_argv,
    )

    descriptor_dir = hooks_dir or hooks_install_dir
    descriptor_dir.mkdir(parents=True, exist_ok=True)
    # One JSON descriptor per stage — podman/crun reuse the same
    # ``hook.args`` for every stage in a single descriptor's ``stages``
    # list (no per-stage argv injection), so we'd lose the stage signal
    # otherwise.  The hook script self-dispatches on ``argv[1]``.
    for stage in _HOOK_STAGES:
        (descriptor_dir / _descriptor_name(stage)).write_text(
            _render_hook_descriptor(hooks_install_dir / _HOOK_SCRIPT_NAME, stage=stage),
            encoding="utf-8",
        )
    ensure_user_hooks_dir_configured(descriptor_dir)


def uninstall_supervisor_hooks(*, hooks_dir: Path | None = None) -> None:
    """Remove every file [`install_supervisor_hooks`][terok_sandbox.supervisor.install.install_supervisor_hooks] writes.

    Idempotent — missing files are tolerated.  Does **not** touch
    per-container state (``sidecar/``, ``logs/``, ``pids/`` under the
    state root) — those are sweep-able with a separate operator
    command if needed.
    """
    install_root = state_root()
    paths = [
        Path("hooks") / _HOOK_SCRIPT_NAME,
        Path("hooks") / _BALLAST_NAME,
        Path(_WRAPPER_NAME),
    ]
    paths.extend(Path("hooks") / _descriptor_name(stage) for stage in _HOOK_STAGES)
    for relative in paths:
        (install_root / relative).unlink(missing_ok=True)
    if hooks_dir is not None:
        for stage in _HOOK_STAGES:
            (hooks_dir / _descriptor_name(stage)).unlink(missing_ok=True)


def kill_all_supervisors() -> list[tuple[str, str | None]]:
    """SIGKILL every live host-side supervisor process group, then sweep strays.

    The OCI hook spawns each wrapper with ``start_new_session=True``,
    so the PID it records is also the **process-group ID** of that
    container's entire supervisor tree — restart-loop wrapper,
    supervisor, service children, watcher subprocesses.  One
    ``killpg`` per ``<state_root>/pids/supervisor-*.pid`` therefore
    takes the whole tree down atomically, and it still works after the
    wrapper itself died: a process group persists while any member
    lives, and its ID cannot be recycled meanwhile.  A second pass
    (``_kill_all_service_children``) sweeps ``supervise-child``
    processes that no PID file reaches any more — an earlier reap
    unlinked the file while the kill half failed.  Each returned row
    is ``(container_id, error_or_None)`` — ``None`` means the
    processes are no longer there, whether we killed them or they had
    already exited.

    Designed for the panic path: the OCI ``poststop`` reap does a
    graceful ``SIGTERM`` → poll → ``SIGKILL`` dance for a normal
    container stop; panic skips straight to ``SIGKILL`` because the
    whole point is to deny the supervisor any more cycles to answer
    socket calls from a misbehaving container.
    """
    results: list[tuple[str, str | None]] = []
    pids_dir = state_root() / _PIDS_DIR_NAME
    if pids_dir.is_dir():
        wrapper_path = str(state_root() / _WRAPPER_NAME)
        for pid_file in sorted(pids_dir.glob(_PID_GLOB)):
            container_id = pid_file.stem.removeprefix("supervisor-")
            results.append(
                (container_id, _kill_one_supervisor(pid_file, wrapper_path, container_id))
            )
    results.extend(_kill_all_service_children())
    return results


def _kill_all_service_children() -> list[tuple[str, str | None]]:
    """SIGKILL every ``supervise-child`` service process left on the host.

    The group pass above reaches everything a PID file names — but a
    child whose PID file is already gone (an earlier reap unlinked it
    while the kill half failed) has no group handle left, and the
    vault-daemon child then keeps the credentials DB open
    indefinitely.  So the last-resort net finds children by what they
    *are* rather than who spawned them: a ``/proc`` argv sweep for the
    ``supervise-child`` invocation.  Normally it finds nothing.  One
    row per child, keyed by its container-ID argv element.
    """
    results: list[tuple[str, str | None]] = []
    for pid, args in _iter_process_argvs():
        if _CHILD_MODULE_MARK not in args or _CHILD_VERB_MARK not in args:
            continue
        error: str | None = None
        try:
            os.kill(pid, signal.SIGKILL)
        except ProcessLookupError:
            pass
        except OSError as exc:
            error = f"SIGKILL failed: {exc}"
        results.append((_child_container_id(args), error))
    return results


def _iter_process_argvs() -> Iterator[tuple[int, list[bytes]]]:
    """Yield ``(pid, argv_elements)`` for every process whose cmdline is readable.

    Bytes, not text — ``/proc/*/cmdline`` may contain arbitrary byte
    sequences for foreign processes, and a decode error would abort
    the sweep mid-flight.
    """
    for proc_dir in _PROC_DIR.glob("[0-9]*"):
        with contextlib.suppress(OSError):
            raw = (proc_dir / "cmdline").read_bytes()
            yield int(proc_dir.name), raw.rstrip(b"\x00").split(b"\x00")


def _child_container_id(args: list[bytes]) -> str:
    """The container-ID argv element (``… supervise-child <service> <id> <sidecar>``)."""
    idx = args.index(_CHILD_VERB_MARK)
    if idx + 2 < len(args):
        return args[idx + 2].decode(errors="replace")
    return "?"


def _kill_one_supervisor(pid_file: Path, wrapper_path: str, container_id: str) -> str | None:
    """SIGKILL one supervisor's whole process group; unlink the PID file regardless."""
    try:
        pgid = int(pid_file.read_text().strip())
    except (OSError, ValueError) as exc:
        pid_file.unlink(missing_ok=True)
        return f"unreadable pid file: {exc}"
    error: str | None = None
    if _is_group_ours(pgid, wrapper_path, container_id):
        try:
            os.killpg(pgid, signal.SIGKILL)
        except ProcessLookupError:
            pass
        except OSError as exc:
            error = f"SIGKILL failed: {exc}"
    pid_file.unlink(missing_ok=True)
    return error


def _is_group_ours(pgid: int, wrapper_path: str, container_id: str) -> bool:
    """PID-recycle guard for a whole supervisor process group.

    Leader alive → its argv must be our wrapper *for this container*
    (wrapper_path alone matches every live wrapper, so a PID recycled
    into another container's wrapper would slip through).  Leader gone
    or a zombie (no ``/proc`` entry, or the empty cmdline zombies get)
    → ``True``: a group ID stays pinned while any member lives, so a
    surviving group under this number can only be the remnant of our
    own session — and if nothing survives either, the ``killpg``
    lands on ``ProcessLookupError`` harmlessly.

    Reads bytes (not text) — ``/proc/*/cmdline`` is null-separated and
    is allowed to contain arbitrary byte sequences for the recycled
    process; a ``UnicodeDecodeError`` here would otherwise abort
    ``kill_all_supervisors`` mid-sweep.
    """
    try:
        cmdline = (_PROC_DIR / str(pgid) / "cmdline").read_bytes()
    except OSError:
        return True
    if not cmdline:
        return True
    args = cmdline.rstrip(b"\x00").split(b"\x00")
    return wrapper_path.encode() in args and container_id.encode() in args


def _copy_executable(src: Path, dst: Path) -> None:
    """Copy *src* to *dst* and chmod 0755 so podman can exec the hook script."""
    shutil.copy(src, dst)
    dst.chmod(0o755)


def _resolve_sandbox_argv() -> list[str]:
    """Return the argv prefix that runs ``terok-sandbox``.

    Resolution order:

    1. ``shutil.which("terok-sandbox")`` — covers system installs
       (``apt``/``dnf``/system-Python pip) where the entry point lands
       on ``$PATH``.
    2. ``sys.executable``'s sibling bin directory — covers pipx and
       virtualenv shapes where the console script lives next to
       Python but the venv's ``bin/`` isn't on ``$PATH`` (e.g. pipx
       only exposes the primary package's binaries via ``~/.local/bin``,
       so dep-provided scripts like ``terok-sandbox`` are reachable
       only through the venv directly).

    Raises ``RuntimeError`` when neither resolves — the wrapper baked
    against a missing binary would silently fail every spawn at
    runtime, which is much harder to debug than a setup-time error.
    """
    direct = shutil.which("terok-sandbox")
    if direct:
        return [direct]
    sibling = Path(sys.executable).parent / "terok-sandbox"
    if sibling.is_file() and os.access(sibling, os.X_OK):
        return [str(sibling)]
    raise RuntimeError(
        "terok-sandbox entry point not found — checked $PATH and "
        f"{Path(sys.executable).parent}; install the wheel and re-run "
        "`terok-sandbox setup`"
    )


def _render_wrapper(*, src: Path, dst: Path, sandbox_argv: list[str]) -> None:
    """Render the wrapper template into *dst* with *sandbox_argv* baked in.

    Replaces the literal ``["__TEROK_SANDBOX_BIN__"]`` placeholder
    with the JSON-encoded argv list so the wrapper has zero runtime
    path discovery to do.
    """
    template = src.read_text(encoding="utf-8")
    rendered = template.replace(_WRAPPER_BIN_PLACEHOLDER, json.dumps(sandbox_argv))
    dst.write_text(rendered, encoding="utf-8")
    dst.chmod(0o755)


def _render_hook_descriptor(entrypoint: Path, *, stage: str) -> str:
    """Build the OCI hook JSON descriptor for one stage.

    Podman / crun reuse the same ``hook.args`` for every stage listed
    in a single descriptor's ``stages`` field — there's no per-stage
    argv injection.  The role script self-dispatches on ``argv[1]``,
    so each stage gets its own descriptor with the matching argv.

    Matching on the ``terok.sandbox.sidecar`` annotation keeps
    non-terok podman invocations unaffected — the same annotation
    that pins the sidecar JSON path also gates the hook fire-list.
    """
    hook = {
        "version": "1.0.0",
        "hook": {
            "path": str(entrypoint),
            "args": ["supervisor_hook", stage],
        },
        "when": {"annotations": {_TRIGGER_ANNOTATION: ".+"}},
        "stages": [stage],
    }
    return json.dumps(hook, indent=2) + "\n"
