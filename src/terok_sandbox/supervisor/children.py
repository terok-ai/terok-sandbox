# SPDX-FileCopyrightText: 2026 Jiri Vyskocil
# SPDX-License-Identifier: Apache-2.0

"""Per-service child runners — one hardened process per supervisor service.

The supervisor used to compose every service (vault proxy, SSH signer,
git gate, clearance hub, verdict server) as coroutines in a single
asyncio loop.  That put secret-holding code (the vault's SQLCipher
session key, the signer's private keys) in the same address space as
convenience services (the desktop notifier), so a bug in any of them
exposed all of them.

Each service now runs in its own process, launched by the parent
supervisor via [`launch_child`][terok_sandbox.supervisor.launcher.launch_child].
A child does exactly one thing:

1. [`harden_self`][terok_util.harden_self] — clear the dumpable flag,
   zero the core limit, lock memory — *before* it opens the credential
   store or binds a socket.
2. Re-read the sidecar (the parent hands it the same path), rebuild the
   one service it owns, and run that service's asyncio loop.
3. Await ``SIGTERM`` from the parent, then stop the service and exit 0.

The service classes are constructed and driven the standard way — only
the process boundary is new.  IPC is unchanged because every service
already binds a per-container filesystem socket (or loopback port); a
child in a separate process binds the identical path the container
reaches.

The five children map onto the six former services: ``clearance`` owns
the hub *and* the desktop notifier/subscriber (they share the clearance
socket and the notifier only drives the subscriber), while ``verdict``,
``vault``, ``signer``, and ``gate`` are one service each.  ``gate`` only
runs when the sidecar wired it.
"""

from __future__ import annotations

import asyncio
import contextlib
import ctypes
import logging
import os
import signal
import sys
from pathlib import Path
from typing import TYPE_CHECKING

from terok_util import confine_filesystem, harden_self

from .sidecar import SupervisorPaths, load_sidecar

if TYPE_CHECKING:
    from collections.abc import Awaitable, Callable

    from .sidecar import SidecarConfig

    #: One service's runner — build the service and run it until *stop* is set.
    _Runner = Callable[[SidecarConfig, SupervisorPaths, asyncio.Event], Awaitable[None]]

_logger = logging.getLogger("terok-supervisor.child")

#: Exit codes the child hands the parent.  ``0`` = clean stop after
#: SIGTERM; ``2`` = unusable sidecar (parent won't retry a config error);
#: ``4`` = the named service failed to start (parent logs + carries on,
#: mirroring the old per-service degradation).
_EXIT_OK = 0
_EXIT_BAD_SIDECAR = 2
_EXIT_START_FAILED = 4

#: ``prctl`` option arming the parent-death signal (Linux).
_PR_SET_PDEATHSIG = 1

#: Secret-free filesystem roots every child may read and execute — the OS,
#: its shared libraries, and this interpreter.  Nothing sensitive lives here,
#: so read-confinement grants them to all services uniformly; a service's own
#: data (and any secret at rest) lives under the terok state/runtime tree and
#: is granted per-service by ``_writable_paths``, never here — which is what
#: keeps one child from reading another's.
_SYSTEM_READABLE_ROOTS: tuple[Path, ...] = (
    *(
        Path(p)
        for p in ("/usr", "/lib", "/lib64", "/bin", "/sbin", "/etc", "/proc", "/dev", "/run")
    ),
    Path(sys.prefix),
    Path(sys.base_prefix),
)


async def _run_verdict(cfg: SidecarConfig, paths: SupervisorPaths, stop: asyncio.Event) -> None:
    """Varlink verdict server — execs ``terok-shield allow|deny``.

    Started before ``clearance`` so the hub's verdict client finds the
    socket bound.  Labels its bind ``terok_socket_t`` so a confined
    ``container_t`` can ``connectto`` it under the installed policy.
    """
    from terok_sandbox._util._selinux import socket_selinux_context
    from terok_sandbox.integrations.clearance import VerdictServer

    verdict = VerdictServer(
        socket_path=paths.verdict_socket,
        socket_context=socket_selinux_context,
    )
    await verdict.start()
    try:
        await stop.wait()
    finally:
        await verdict.stop()


async def _run_clearance(cfg: SidecarConfig, paths: SupervisorPaths, stop: asyncio.Event) -> None:
    """Clearance hub plus the desktop notifier/subscriber it feeds.

    The hub subscribes to the verdict server (over ``verdict_socket``);
    the subscriber turns ``connection_blocked`` events into D-Bus popups.
    ``create_notifier`` is no-fail — it degrades to a null notifier when
    no session bus is reachable — so the hub is the only bring-up here
    that can fail the child.
    """
    from terok_sandbox._util._selinux import socket_selinux_context
    from terok_sandbox.integrations.clearance import (
        NOTIFY_BLOCKED,
        NOTIFY_VERDICT,
        ClearanceHub,
        EventSubscriber,
        VerdictClient,
        create_notifier,
    )

    hub = ClearanceHub(
        clearance_socket=paths.clearance_socket,
        reader_socket=paths.events_socket,
        verdict_client=VerdictClient(socket_path=paths.verdict_socket),
        socket_context=socket_selinux_context,
    )
    await hub.start()
    notifier = await create_notifier("terok-supervisor")
    subscriber = EventSubscriber(
        notifier,
        socket_path=paths.clearance_socket,
        enabled_categories=frozenset({NOTIFY_BLOCKED, NOTIFY_VERDICT}),
    )
    await subscriber.start()
    try:
        await stop.wait()
    finally:
        await subscriber.stop()
        await hub.stop()
        await notifier.disconnect()


async def _run_gate(cfg: SidecarConfig, paths: SupervisorPaths, stop: asyncio.Event) -> None:
    """Git gate serving ``<gate_base_path>/<project_id>.git`` on the minted token.

    Socket mode binds a per-container socket; TCP mode binds a
    per-container loopback port.  The parent only launches this child
    when the sidecar carried both ``gate_base_path`` and ``gate_token``.
    """
    from terok_sandbox.gate.hooks import hooks_dir_for, install_hooks
    from terok_sandbox.gate.server import GateServer

    if not cfg.gate_base_path or not cfg.gate_token:
        raise RuntimeError("gate child launched without gate wiring in the sidecar")
    # (Re-)render the sandbox-owned push hooks next to the gates —
    # idempotent, so every gate start converges the hook content to this
    # package version's.  The server itself only receives the path.
    hooks_path = hooks_dir_for(cfg.gate_base_path)
    install_hooks(hooks_path)
    if cfg.ipc_mode == "tcp":
        if not cfg.gate_port:
            raise RuntimeError(f"sidecar ipc_mode='tcp' but gate_port is {cfg.gate_port!r}")
        gate = GateServer(
            mirror_root=cfg.gate_base_path,
            token=cfg.gate_token,
            scope=cfg.project_id,
            host="127.0.0.1",
            port=cfg.gate_port,
            hooks_path=hooks_path,
        )
    else:
        gate = GateServer(
            mirror_root=cfg.gate_base_path,
            token=cfg.gate_token,
            scope=cfg.project_id,
            socket_path=paths.gate_socket,
            hooks_path=hooks_path,
        )
    await gate.start()
    try:
        await stop.wait()
    finally:
        await gate.stop()


async def _run_vault(cfg: SidecarConfig, paths: SupervisorPaths, stop: asyncio.Event) -> None:
    """Vault HTTP/WS proxy over the SQLCipher store; transport from ``ipc_mode``.

    The highest-value isolation target — this is the process that holds
    the decrypted credential store's session key, which the hardening
    floor keeps out of core dumps, ptrace, and swap.
    """
    from terok_sandbox.vault.daemon.token_broker import TcpBind, UnixBind, VaultProxy

    bind: UnixBind | TcpBind
    if cfg.ipc_mode == "tcp":
        if not cfg.tcp_port:
            raise RuntimeError(f"sidecar ipc_mode='tcp' but tcp_port is {cfg.tcp_port!r}")
        bind = TcpBind(host="127.0.0.1", port=cfg.tcp_port)
    else:
        bind = UnixBind(socket_path=paths.vault_socket)
    vault = VaultProxy(
        db_path=cfg.db_path,
        scope_id=cfg.scope_id,
        bind=bind,
        runtime_dir=cfg.runtime_dir,
    )
    await vault.start()
    try:
        await stop.wait()
    finally:
        await vault.stop()


async def _run_signer(cfg: SidecarConfig, paths: SupervisorPaths, stop: asyncio.Event) -> None:
    """Token-gated SSH-agent holding the container's signing keys.

    Same transport split as the vault proxy.  ``start_ssh_signer``
    returns a bare ``asyncio.Server`` (no ``.stop()``), so teardown
    closes and awaits it directly.
    """
    from terok_sandbox.vault.ssh.signer import start_ssh_signer

    if cfg.ipc_mode == "tcp":
        if not cfg.ssh_signer_port:
            raise RuntimeError(
                f"sidecar ipc_mode='tcp' but ssh_signer_port is {cfg.ssh_signer_port!r}"
            )
        server = await start_ssh_signer(
            db_path=str(cfg.db_path), host="127.0.0.1", port=cfg.ssh_signer_port
        )
    else:
        server = await start_ssh_signer(
            db_path=str(cfg.db_path), socket_path=str(paths.ssh_signer_socket)
        )
    try:
        await stop.wait()
    finally:
        server.close()
        await server.wait_closed()


#: Service name → its runner coroutine.  The parent's launch order is the
#: insertion order here: verdict before clearance (the hub connects to
#: it), gate before vault (the container clones through the gate first),
#: vault and signer last (secret-holders come up once their consumers are
#: waiting).  The keys are the wire vocabulary of ``supervise-child``.
_RUNNERS: dict[str, _Runner] = {
    "verdict": _run_verdict,
    "clearance": _run_clearance,
    "gate": _run_gate,
    "vault": _run_vault,
    "signer": _run_signer,
}

#: The service names, in launch order — consumed by the parent supervisor.
SERVICE_NAMES: tuple[str, ...] = tuple(_RUNNERS)


def _arm_parent_death_signal() -> bool:
    """Arm the kernel dead-man's switch: SIGTERM this child when its parent dies.

    A supervisor killed without teardown (crash, OOM, SIGKILL past the
    poststop grace) would otherwise strand its service children — the
    vault-daemon child then pins the credentials DB open and blocks any
    later re-encryption.  ``PR_SET_PDEATHSIG`` makes the *kernel*
    deliver the same SIGTERM the graceful teardown would have sent, so
    the child closes down cleanly with no reaper involved at all.

    Best-effort on the prctl itself (Linux-specific; the group-level
    poststop reap remains the backstop), but the return value is
    binding: ``False`` means the parent is *already* gone — the arm
    raced the supervisor's death — and the caller must exit rather
    than run as a stray the switch will never fire for.
    """
    with contextlib.suppress(OSError, AttributeError):
        libc = ctypes.CDLL(None, use_errno=True)
        libc.prctl(_PR_SET_PDEATHSIG, signal.SIGTERM, 0, 0, 0)
    return os.getppid() != 1


def run_child(service: str, container_id: str, sidecar_path: Path) -> int:
    """Harden, build the one *service*, run it until SIGTERM; return an exit code.

    The synchronous entry the ``supervise-child`` CLI verb calls via
    ``asyncio.run``.  Arms the parent-death signal first (a supervisor
    that dies without teardown must never strand a running child), then
    loads the sidecar the parent pinned (config, not secrets), hardens
    the process *before* the runner opens the credential store or binds
    a socket — honouring the sidecar's debug-mode opt-out — then drives
    the single service's lifecycle.  A start failure returns
    ``_EXIT_START_FAILED`` (4) so the parent can log it and carry on,
    degrading one service without taking the rest down.
    """
    if not _arm_parent_death_signal():
        _logger.error(
            "%s child: supervisor died before startup — refusing to run as a stray", service
        )
        return _EXIT_START_FAILED
    runner = _RUNNERS.get(service)
    if runner is None:
        _logger.error("unknown supervisor child service %r", service)
        return _EXIT_BAD_SIDECAR

    cfg = load_sidecar(sidecar_path)
    if cfg is None:
        _logger.error("%s child: no usable sidecar at %s", service, sidecar_path)
        return _EXIT_BAD_SIDECAR

    report = harden_self(allow_debugger=cfg.allow_debugger)
    if not cfg.allow_debugger and not report.fully_hardened:
        # Expected in a rootless container (mlockall needs CAP_IPC_LOCK);
        # log at debug so the operator can confirm the floor on hosts
        # where it should have taken.  (Debug mode drops no_dump on
        # purpose, so a partial report is not noteworthy there.)
        _logger.debug("%s child hardening partial: %s", service, report)

    if not cfg.allow_debugger:
        # Pin the filesystem to this service's lane: read+exec the shared
        # runtime, read+write only its own data.  A bug in a binary it runs
        # (gate → git, vault → systemd-creds) then can neither exfiltrate a
        # sibling's secret nor write outside its lane.  Debug mode keeps the
        # filesystem open for dump/trace tools.
        fs = confine_filesystem(_SYSTEM_READABLE_ROOTS, _writable_paths(service, cfg))
        if not fs.confined:
            _logger.debug("%s child filesystem-confinement not applied: %s", service, fs.reason)

    paths = SupervisorPaths.for_container(
        container_id, cfg.container_name, sidecar_path, cfg.runtime_dir
    )
    return asyncio.run(_drive(service, runner, cfg, paths))


def _writable_paths(service: str, cfg: SidecarConfig) -> list[Path]:
    """The directories *service*'s child may write into — everything else is denied.

    Every child needs its socket directory, and those live under the shared
    terok runtime root: ``cfg.runtime_dir``'s parent covers both the
    sandbox-namespaced sockets (vault/gate/ssh, under ``runtime_dir/run``)
    and the cross-package ones (clearance/verdict/control, beside it).  Only
    the two secret-holders add on-disk data — vault the SQLCipher store (its
    journal/WAL land beside it), gate the mirror tree it serves; verdict,
    clearance, and signer keep nothing on disk of their own.
    """
    writable = [cfg.runtime_dir.parent]
    if service == "vault":
        writable.append(cfg.db_path.parent)
    elif service == "gate" and cfg.gate_base_path:
        writable.append(cfg.gate_base_path)
    return writable


async def _drive(
    service: str,
    runner: _Runner,
    cfg: SidecarConfig,
    paths: SupervisorPaths,
) -> int:
    """Bind the service's socket dir, install signal handlers, run the runner."""
    _ensure_socket_dirs(service, paths)
    stop = asyncio.Event()
    _install_signal_handlers(stop)
    try:
        await runner(cfg, paths, stop)
    except Exception:
        _logger.exception("%s child failed", service)
        return _EXIT_START_FAILED
    return _EXIT_OK


def _ensure_socket_dirs(service: str, paths: SupervisorPaths) -> None:
    """Create + tighten the socket parent dirs the *service* binds under.

    ``bind_hardened`` refuses group/world-accessible parents, so each
    child mkdirs its own socket dirs at ``0o700`` — crun's rootless
    umask is too permissive to rely on.
    """
    sockets = {
        "verdict": (paths.verdict_socket,),
        "clearance": (paths.clearance_socket, paths.events_socket),
        "gate": (paths.gate_socket,),
        "vault": (paths.vault_socket,),
        "signer": (paths.ssh_signer_socket,),
    }[service]
    for sock in sockets:
        sock.parent.mkdir(parents=True, exist_ok=True)
        sock.parent.chmod(0o700)


def _install_signal_handlers(stop: asyncio.Event) -> None:
    """Set *stop* on SIGTERM/SIGINT — a no-op when called outside a loop.

    The soft-fail lets the helper be called from a synchronous context
    (its own tests) without wiring anything; under ``asyncio.run`` there
    is always a running loop.
    """
    try:
        loop = asyncio.get_running_loop()
    except RuntimeError:
        return
    for sig in (signal.SIGTERM, signal.SIGINT):
        loop.add_signal_handler(sig, stop.set)
