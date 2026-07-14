# SPDX-FileCopyrightText: 2026 Jiri Vyskocil
# SPDX-License-Identifier: Apache-2.0

"""The parent supervisor — ``run_supervisor(container_id, sidecar_path)``.

Once a single asyncio loop composing every service, the supervisor is
now a *supervisor of processes*: it
[`launch_child`][terok_sandbox.supervisor.launcher.launch_child]s one
hardened child per service and owns only their lifecycle.  Each child
([`run_child`][terok_sandbox.supervisor.children.run_child]) hardens
itself, rebuilds its one service from the sidecar, and runs it until the
parent signals stop — so a bug in the desktop notifier can no longer
touch the address space that holds the vault's session key.

The five children, in launch order:

* ``verdict`` — [`VerdictServer`][terok_clearance.VerdictServer], the
  varlink helper that execs ``terok-shield allow|deny``.
* ``clearance`` — [`ClearanceHub`][terok_clearance.ClearanceHub] plus
  the desktop [`EventSubscriber`][terok_clearance.EventSubscriber] /
  notifier it feeds.
* ``gate`` — the git gate (only when the sidecar wired it).
* ``vault`` — [`VaultProxy`][terok_sandbox.vault.daemon.token_broker.VaultProxy].
* ``signer`` — the token-gated SSH-agent
  ([`start_ssh_signer`][terok_sandbox.vault.ssh.signer.start_ssh_signer]).

The parent awaits ``podman wait <container_id>``; when the container
exits (or every child dies first) it terminates the children in reverse
launch order and returns.

Hidden from main user help; invoked by the OCI hook chain only.
"""

from __future__ import annotations

import asyncio
import contextlib
import logging
import shutil
from pathlib import Path
from typing import TYPE_CHECKING

from .children import SERVICE_NAMES, _install_signal_handlers
from .launcher import launch_child
from .sidecar import SidecarConfig, SupervisorPaths, load_sidecar

if TYPE_CHECKING:
    from .launcher import ChildHandle

_logger = logging.getLogger("terok-supervisor")

#: Upper bound on how long the cancellation path waits for ``podman wait``
#: to exit after ``SIGTERM`` before escalating to ``SIGKILL``.  Two seconds
#: is plenty for a healthy podman; longer would block supervisor shutdown.
_PODMAN_WAIT_CANCEL_GRACE_S = 2.0

#: How long a child gets to honour ``SIGTERM`` at teardown before the
#: supervisor escalates to ``SIGKILL``.  Longer than the podman-wait
#: grace because a child may be flushing a clearance event or closing a
#: SQLCipher handle, but still bounded so a wedged service can't stall exit.
_CHILD_TERM_GRACE_S = 5.0


# ── Entry point ─────────────────────────────────────────────────────────


async def run_supervisor(container_id: str, sidecar_path: Path) -> int:
    """Launch, monitor, and tear down the per-container child processes.

    Lifecycle:

    1. Load the sidecar JSON from *sidecar_path*; bail with exit code 2
       on parse / missing.
    2. Launch one hardened child per service (skipping the git gate when
       the sidecar didn't wire it).  A child that never spawns is logged
       and skipped; when *no* child spawns, exit code 3 hands the wrapper
       its retry.
    3. Install SIGTERM / SIGINT handlers so a host-side
       ``terok-sandbox supervisor`` invocation stops cleanly with Ctrl-C.
    4. Await ``podman wait <container_id>`` racing the children's own
       liveness and the stop signal.  On any of them, terminate the
       children in reverse launch order and return.

    The function is the sole parent entry point; the CLI verb
    ``terok-sandbox supervisor`` invokes it via ``asyncio.run``.  The
    per-service work now lives in
    [`run_child`][terok_sandbox.supervisor.children.run_child], one
    process each.
    """
    cfg = load_sidecar(sidecar_path)
    if cfg is None:
        _logger.error(
            "no usable sidecar at %s — aborting supervisor for %s",
            sidecar_path,
            container_id,
        )
        return 2

    paths = SupervisorPaths.for_container(
        container_id, cfg.container_name, sidecar_path, cfg.runtime_dir
    )
    services = _select_services(cfg, SERVICE_NAMES)
    stop_event = asyncio.Event()
    _install_signal_handlers(stop_event)

    handles: list[ChildHandle] = []
    try:
        handles = await _launch_children(services, container_id, sidecar_path)
        if not handles:
            _logger.error(
                "supervisor: no child could be launched for %s — exiting so the wrapper retries",
                container_id,
            )
            return 3
        await _supervise(container_id, handles, stop_event)
        return 0
    finally:
        await _terminate_children(handles)
        # rmtree the per-container dir on every exit path — launch
        # failure included — so a half-bound socket directory can't
        # outlive the supervisor and confuse the next launch.
        shutil.rmtree(paths.container_runtime_dir, ignore_errors=True)


def _select_services(cfg: SidecarConfig, service_names: tuple[str, ...]) -> tuple[str, ...]:
    """The services to launch — every one except a gate the sidecar didn't wire."""
    gate_wired = bool(cfg.gate_base_path and cfg.gate_token)
    return tuple(name for name in service_names if name != "gate" or gate_wired)


async def _launch_children(
    services: tuple[str, ...],
    container_id: str,
    sidecar_path: Path,
) -> list[ChildHandle]:
    """Spawn one child per service; a spawn failure costs only that service.

    Mirrors the old per-service degradation: the typical restart-time
    case is a stolen TCP port, which must cost exactly the one service
    that lost it, never the whole bundle.
    """
    handles: list[ChildHandle] = []
    for service in services:
        try:
            handles.append(await launch_child(service, container_id, sidecar_path))
        except Exception:
            _logger.exception("failed to launch %s child — continuing without it", service)
    return handles


async def _supervise(
    container_id: str, handles: list[ChildHandle], stop_event: asyncio.Event
) -> None:
    """Block until the container exits, every child dies, or a stop signal.

    Races three tasks; whichever fires first unblocks teardown.  The
    ``all children exited`` arm covers the case where every service died
    on its own (a stolen port on all of them), so the parent stops
    holding a dead bundle instead of waiting forever on ``podman wait``.
    """
    wait_task = asyncio.create_task(_wait_for_container(container_id))
    stop_task = asyncio.create_task(stop_event.wait())
    children_task = asyncio.create_task(_await_all_children(handles))
    done, pending = await asyncio.wait(
        {wait_task, stop_task, children_task}, return_when=asyncio.FIRST_COMPLETED
    )
    for task in pending:
        task.cancel()
    # Await every task — cancelled included — so the ``podman wait``
    # subprocess gets a chance to terminate cleanly via its
    # CancelledError handler rather than being orphaned.
    for task in (*done, *pending):
        try:
            await task
        except asyncio.CancelledError:
            pass
        except Exception:
            _logger.exception("supervisor monitor task raised for %s", container_id)


async def _await_all_children(handles: list[ChildHandle]) -> None:
    """Return once every child process has exited (the all-died case)."""
    await asyncio.gather(*(handle.process.wait() for handle in handles))


async def _terminate_children(handles: list[ChildHandle]) -> None:
    """SIGTERM the children in reverse launch order, escalating to SIGKILL.

    Reverse order mirrors the old in-process teardown: the notifier /
    hub go down before the secret-holders (vault, signer).  All children
    then get a *single* shared grace window to exit — waiting on them
    concurrently, so a wedged bundle can't stall exit for
    ``len(handles) * grace`` — and any survivor is killed.

    An external cancellation (the enclosing task being cancelled) is
    honoured: the children are killed so none is orphaned, then the
    ``CancelledError`` is re-raised rather than swallowed as if it were
    the internal grace timeout.
    """
    live = [h for h in reversed(handles) if h.process.returncode is None]
    for handle in live:
        with contextlib.suppress(ProcessLookupError):
            handle.process.terminate()
    waits = asyncio.gather(*(h.process.wait() for h in live))
    try:
        await asyncio.wait_for(waits, timeout=_CHILD_TERM_GRACE_S)
    except TimeoutError:
        await _kill_children(live)
    except asyncio.CancelledError:
        await _kill_children(live)
        raise


async def _kill_children(handles: list[ChildHandle]) -> None:
    """SIGKILL any child still running and reap it, swallowing races."""
    for handle in handles:
        if handle.process.returncode is not None:
            continue
        with contextlib.suppress(ProcessLookupError):
            handle.process.kill()
        with contextlib.suppress(Exception):
            await handle.process.wait()


# ── Internal helpers ────────────────────────────────────────────────────


async def _wait_for_container(container_id: str) -> int:
    """Block until ``podman wait <container_id>`` returns; surface its exit code.

    Returns the container's exit code (an integer; podman prints it on
    stdout).  Treats a non-zero ``podman wait`` *invocation* exit code
    as a soft failure — that path is rare (container ID typo, podman
    crash) but never blocks the supervisor's clean shutdown.
    """
    proc = await asyncio.create_subprocess_exec(
        "podman",
        "wait",
        container_id,
        stdout=asyncio.subprocess.PIPE,
        stderr=asyncio.subprocess.PIPE,
    )
    try:
        stdout, stderr = await proc.communicate()
    except asyncio.CancelledError:
        # Stop-signal path: terminate the lingering ``podman wait``
        # before propagating cancellation, so the subprocess doesn't
        # outlive the supervisor and pin the container ID.  Bound the
        # post-SIGTERM wait so a hung podman can't stall shutdown.
        with contextlib.suppress(ProcessLookupError):
            proc.terminate()
        try:
            await asyncio.wait_for(proc.wait(), timeout=_PODMAN_WAIT_CANCEL_GRACE_S)
        except (TimeoutError, asyncio.CancelledError):
            with contextlib.suppress(ProcessLookupError):
                proc.kill()
            with contextlib.suppress(asyncio.CancelledError, Exception):
                await proc.wait()
        raise
    if proc.returncode != 0:
        _logger.warning(
            "podman wait %s exited %s: %s",
            container_id,
            proc.returncode,
            stderr.decode(errors="replace").strip(),
        )
        return proc.returncode or 0
    try:
        return int(stdout.decode().strip())
    except ValueError:
        return 0
