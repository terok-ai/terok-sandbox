# SPDX-FileCopyrightText: 2026 Jiri Vyskocil
# SPDX-License-Identifier: Apache-2.0

"""The supervisor coroutine — ``run_supervisor(container_id)``.

One asyncio loop composes:

* [`VaultProxy`][terok_sandbox.vault.daemon.token_broker.VaultProxy] — vault HTTP/WS
  proxy, transport picked from the sidecar (``socket`` → ``UnixBind``,
  ``tcp`` → ``TcpBind``).
* SSH signer ([`start_ssh_signer`][terok_sandbox.vault.ssh.signer.start_ssh_signer])
  — token-gated SSH-agent the container reaches over the same transport.
* [`VerdictServer`][terok_clearance.VerdictServer] — varlink helper
  that execs ``terok-shield allow|deny``.  Lives in its own per-
  container socket.
* [`ClearanceHub`][terok_clearance.ClearanceHub] — varlink hub the
  shield reader (and operator UIs) subscribe to.  Wired to the local
  verdict server above.
* Desktop notifier — an [`EventSubscriber`][terok_clearance.EventSubscriber]
  that turns ``connection_blocked`` events into D-Bus popups via
  [`create_notifier`][terok_clearance.create_notifier].

The supervisor awaits ``podman wait <container_id>``; when the
container exits it tears the bundle down in reverse order and
returns 0.

Hidden from main user help; invoked by the OCI hook chain only.
"""

from __future__ import annotations

import asyncio
import contextlib
import json
import logging
import shutil
import signal
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from collections.abc import Awaitable, Callable
    from typing import Any

_logger = logging.getLogger("terok-supervisor")

#: Upper bound on how long the cancellation path waits for ``podman wait``
#: to exit after ``SIGTERM`` before escalating to ``SIGKILL``.  Two seconds
#: is plenty for a healthy podman; longer would block supervisor shutdown.
_PODMAN_WAIT_CANCEL_GRACE_S = 2.0


# ── Sidecar config / paths ──────────────────────────────────────────────


@dataclass(frozen=True)
class SidecarConfig:
    """Per-container config the supervisor reads from the sidecar JSON.

    Written once at container-creation time by
    [`write_sidecar`][terok_sandbox.launch.write_sidecar] (terok-executor
    routes through the same writer) and read back by the OCI hook on
    every ``podman start``.  Keyed by container name
    (``<state>/sidecar/<name>.json``); the ``terok.sandbox.sidecar``
    annotation pins the absolute path.  The file persists across
    stop/start cycles — its ports and tokens must keep matching the
    container's immutable env — and is removed at real teardown by
    [`remove_container_state`][terok_sandbox.launch.remove_container_state].
    """

    container_name: str
    ipc_mode: str  # "socket" or "tcp"
    db_path: Path
    runtime_dir: Path
    """``/run/user/<host_uid>/terok/sandbox`` — pinned by the launch path
    because the supervisor cannot re-derive it from inside crun's
    rootless user namespace (its ``os.getuid()`` is 0 there, which
    misroutes generic resolvers to the root-only ``/run/terok``)."""
    scope_id: str | None = None
    project_id: str = ""
    task_id: str = ""
    tcp_port: int | None = None
    """Per-container TCP port for the vault proxy in TCP mode.
    ``None`` in socket mode (the path is derived from the
    container ID, not carried here)."""
    ssh_signer_port: int | None = None
    """Per-container TCP port for the SSH signer in TCP mode.
    ``None`` in socket mode."""
    gate_port: int | None = None
    """Per-container TCP port for the git gate in TCP mode.
    ``None`` in socket mode."""
    gate_base_path: Path | None = None
    """Directory holding the shared per-project bare mirrors
    (``<gate_base_path>/<project_id>.git``).  ``None`` when the gate is
    not wired for this container."""
    gate_token: str | None = None
    """The single token the gate validates.  Travels only via the
    sidecar; ``None`` when the gate is not wired."""
    dossier_path: Path | None = None


@dataclass(frozen=True)
class SupervisorPaths:
    """Resolved per-container socket / log / pid locations.

    Computed once at supervisor startup from the container ID and the
    runtime/state dirs the sidecar config doesn't carry directly.
    """

    container_id: str
    container_runtime_dir: Path
    """Per-container directory holding ``vault.sock`` and
    ``ssh-agent.sock``.  Keyed on ``container_name`` (which the
    launch path knows before ``podman run`` so it can pre-create
    the dir and bind-mount it as ``/run/terok/`` inside the
    container).  Different containers get different host dirs;
    the in-container view of these sockets is always /run/terok/."""
    vault_socket: Path
    ssh_signer_socket: Path
    gate_socket: Path
    """Per-container git-gate Unix socket inside ``container_runtime_dir``
    (= the in-container ``/run/terok``).  Used only in socket mode; in
    TCP mode the gate binds a loopback port instead."""
    clearance_socket: Path
    events_socket: Path
    """Per-container ingester socket the shield reader and ``shield
    up``/``down`` push raw line-JSON to.  Distinct from
    ``clearance_socket`` (the varlink subscriber socket operator UIs
    glob): the reader speaks line-JSON, not varlink, so the produce and
    subscribe roles need separate sockets."""
    verdict_socket: Path
    control_socket: Path
    log_path: Path

    @classmethod
    def for_container(
        cls,
        container_id: str,
        container_name: str,
        sidecar_path: Path,
        runtime_dir: Path,
    ) -> SupervisorPaths:
        """Build the per-container path bundle.

        Both anchors come from the launch path — neither is re-resolved
        inside the supervisor:

        * *runtime_dir* (``/run/user/<host_uid>/terok/sandbox``) for
          per-container sockets; carried in the sidecar because the
          rootless user namespace makes generic ``is_root``-based
          resolvers (``terok_util.namespace_runtime_dir``) misroute to
          ``/run/terok``.
        * *sidecar_path*'s grandparent for the persistent log file;
          honours whatever ``paths.root`` resolved to when the launch
          path wrote the sidecar.

        Sockets carry the **12-char short container ID** (podman's
        display convention) rather than the full UUID — AF_UNIX's
        ``sun_path`` is 108 bytes including the null terminator, and
        ``<terok-runtime>/clearance/<64-char-uuid>.sock`` lands at or
        past that limit.  Twelve characters of hex give 48 bits of
        entropy, well past the no-collisions-within-one-host bar.
        Logs keep the full UUID because they live on the filesystem
        with no AF_UNIX limit and the full UUID is easier to grep.

        Clearance / verdict / control sockets live at the
        cross-package ``<terok>/`` runtime root (parent of the
        sandbox-namespaced *runtime_dir*) because they're owned by
        terok-clearance semantically and consumed by every package
        that subscribes (terok-shield's NFLOG reader, terok-clearance
        TUI, …).  Sandbox-specific sockets (vault, ssh-agent) live in
        a per-container ``runtime_dir/run/<short_id>/`` directory the
        launch path bind-mounts at ``/run/terok/`` inside the
        container — keeping every container's sockets distinct on
        the host so concurrent containers don't collide.
        """
        short_id = container_id[:12]
        clearance_root = runtime_dir.parent  # <terok>/sandbox/  →  <terok>/
        state_anchor = sidecar_path.parent.parent  # <root>/sidecar/<name>.json → <root>
        # vault + ssh-agent are keyed on container_name (known at
        # launch time, before podman assigns the ID) so the launch
        # path can pre-create the dir and bind-mount it.  Clearance /
        # verdict / control use the 12-char container ID short prefix
        # because they're cross-package (shield's NFLOG reader keys
        # on it too).
        container_runtime_dir = runtime_dir / "run" / container_name
        return cls(
            container_id=container_id,
            container_runtime_dir=container_runtime_dir,
            vault_socket=container_runtime_dir / "vault.sock",
            ssh_signer_socket=container_runtime_dir / "ssh-agent.sock",
            gate_socket=container_runtime_dir / "gate-server.sock",
            clearance_socket=clearance_root / "clearance" / f"{short_id}.sock",
            events_socket=clearance_root / "events" / f"{short_id}.sock",
            verdict_socket=clearance_root / "verdict" / f"{short_id}.sock",
            control_socket=clearance_root / "control" / f"{short_id}.sock",
            log_path=state_anchor / "logs" / f"{container_id}.log",
        )


def load_sidecar(sidecar_path: Path) -> SidecarConfig | None:
    """Read and parse the sidecar JSON at *sidecar_path*.

    The OCI hook pinned this exact path via the
    ``terok.sandbox.sidecar`` annotation, so the supervisor never
    guesses — it opens the named file directly.  Returns ``None`` on
    any I/O / schema failure; ``run_supervisor`` surfaces that as
    exit-code 2.
    """
    try:
        with sidecar_path.open(encoding="utf-8") as fh:
            raw = json.load(fh)
    except (OSError, ValueError):
        _logger.exception("sidecar parse failure for %s", sidecar_path)
        return None
    if not isinstance(raw, dict):
        _logger.error("sidecar is not a JSON object: %s", sidecar_path)
        return None
    try:
        container_name = str(raw.get("container_name", "")).strip()
        if not container_name:
            _logger.error("sidecar missing required container_name: %s", sidecar_path)
            return None
        # ``container_name`` is interpolated into ``runtime_dir/run/<name>``
        # and that directory is mkdir'd, chmod'd, and rmtree'd by the
        # supervisor.  Reject a name that is absolute or carries a path
        # separator / parent-dir reference so a malformed sidecar can't
        # redirect those filesystem operations outside the runtime dir.
        if "/" in container_name or container_name in (".", ".."):
            _logger.error(
                "sidecar container_name is not a safe path component, got %r: %s",
                container_name,
                sidecar_path,
            )
            return None
        ipc_mode = str(raw.get("ipc_mode", "socket"))
        if ipc_mode not in ("socket", "tcp"):
            _logger.error(
                "sidecar ipc_mode must be 'socket' or 'tcp', got %r: %s",
                ipc_mode,
                sidecar_path,
            )
            return None
        # Refuse relative paths — the supervisor takes ``rmtree`` over
        # ``runtime_dir`` and binds sockets under it; a malformed sidecar
        # with a relative path would resolve against whatever cwd the
        # OCI hook spawned us with (typically ``/``), which is nowhere
        # we want to touch.
        db_path = _require_absolute_path(raw, "db_path", sidecar_path)
        runtime_dir = _require_absolute_path(raw, "runtime_dir", sidecar_path)
        if db_path is None or runtime_dir is None:
            return None
        dossier_raw = raw.get("dossier_path")
        if dossier_raw:
            dossier_path = _require_absolute_path(raw, "dossier_path", sidecar_path)
            if dossier_path is None:
                return None
        else:
            dossier_path = None
        # ``gate_base_path`` becomes ``git http-backend``'s
        # ``GIT_PROJECT_ROOT`` — refuse a relative path for the same
        # reason as ``db_path`` / ``runtime_dir``.
        if raw.get("gate_base_path"):
            gate_base_path = _require_absolute_path(raw, "gate_base_path", sidecar_path)
            if gate_base_path is None:
                return None
        else:
            gate_base_path = None
        return SidecarConfig(
            container_name=container_name,
            ipc_mode=ipc_mode,
            db_path=db_path,
            runtime_dir=runtime_dir,
            scope_id=raw.get("scope_id") or None,
            project_id=str(raw.get("project_id") or ""),
            task_id=str(raw.get("task_id") or ""),
            tcp_port=(int(raw["tcp_port"]) if raw.get("tcp_port") is not None else None),
            ssh_signer_port=(
                int(raw["ssh_signer_port"]) if raw.get("ssh_signer_port") is not None else None
            ),
            gate_port=(int(raw["gate_port"]) if raw.get("gate_port") is not None else None),
            gate_base_path=gate_base_path,
            gate_token=(str(raw["gate_token"]) if raw.get("gate_token") else None),
            dossier_path=dossier_path,
        )
    except (KeyError, TypeError, ValueError):
        _logger.exception("sidecar schema error in %s", sidecar_path)
        return None


def _require_absolute_path(raw: dict, key: str, sidecar_path: Path) -> Path | None:
    """Return ``Path(raw[key])`` if absolute, else log + ``None``."""
    value = Path(str(raw[key]))
    if not value.is_absolute():
        _logger.error(
            "sidecar %s must be an absolute path, got %r: %s", key, str(value), sidecar_path
        )
        return None
    return value


# ── Entry point ─────────────────────────────────────────────────────────


async def run_supervisor(container_id: str, sidecar_path: Path) -> int:
    """Compose + run the per-container service bundle.

    Lifecycle:

    1. Load the sidecar JSON from *sidecar_path*; bail with exit code
       2 on parse / missing.
    2. Bring the `_Services`
       bundle up in dependency order; services degrade independently
       (a single startup failure is logged and skipped), but when *no*
       service starts the bundle unwinds and exit code 3 hands the
       wrapper its retry.
    3. Install SIGTERM / SIGINT handlers that race with ``podman wait``
       so a host-side ``terok-sandbox supervisor`` invocation can be
       stopped cleanly with Ctrl-C.
    4. Await ``podman wait <container_id>``.  When it returns, tear the
       bundle down in reverse and return 0.

    The function is the sole supervisor entry point; the CLI verb
    ``terok-sandbox supervisor`` invokes it via ``asyncio.run``.
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
    for sock in (
        paths.clearance_socket,
        paths.events_socket,
        paths.verdict_socket,
        paths.control_socket,
        paths.vault_socket,
        paths.ssh_signer_socket,
        paths.gate_socket,
    ):
        sock.parent.mkdir(parents=True, exist_ok=True)
        # ``bind_hardened`` refuses group/world-accessible parents;
        # explicit chmod overrides crun's permissive rootless umask.
        sock.parent.chmod(0o700)

    services = _Services()
    stop_event = asyncio.Event()
    _install_signal_handlers(stop_event)

    try:
        if await services.start(cfg, paths) == 0:
            _logger.error(
                "supervisor: no service could start for %s — exiting so the wrapper retries",
                container_id,
            )
            await services.stop()
            return 3

        wait_task = asyncio.create_task(_wait_for_container(container_id))
        stop_task = asyncio.create_task(stop_event.wait())
        done, pending = await asyncio.wait(
            {wait_task, stop_task}, return_when=asyncio.FIRST_COMPLETED
        )
        for task in pending:
            task.cancel()
        # Await every task — including the cancelled ones — so the
        # ``podman wait`` subprocess in ``_wait_for_container`` gets a
        # chance to terminate cleanly via its CancelledError handler.
        # Skipping this would orphan the subprocess on stop-signal paths.
        for task in (*done, *pending):
            try:
                await task
            except asyncio.CancelledError:
                pass
            except Exception:
                _logger.exception("supervisor wait task raised for %s", container_id)
        await services.stop()
        return 0
    finally:
        # rmtree the per-container dir on every exit path — startup
        # failure included — so a half-bound socket directory can't
        # outlive the supervisor and confuse the next launch.
        shutil.rmtree(paths.container_runtime_dir, ignore_errors=True)


# ── Service composition (internal) ──────────────────────────────────────


class _Services:
    """The per-container service bundle plus its teardown sequence.

    Holds the clearance verdict server + hub, the vault proxy, the git
    gate, the SSH signer, and the desktop event subscriber (with the
    D-Bus notifier it drives).
    `_Services.start`
    brings them up in dependency order, degrading per service: every
    service here is convenience, not a security boundary (shield's
    fail-closed egress hook runs independently), so one failed bring-up
    must not take the rest down.  ``stop`` unwinds in reverse.

    Order matters at teardown — verdict and hub talk to each other, so
    the hub goes down first, then verdict.  The vault stops last among
    the ``.stop()``-able services because the container's outbound API
    calls may still be in flight when ``podman wait`` returns.
    """

    def __init__(self) -> None:
        self.verdict: Any | None = None
        self.hub: Any | None = None
        self.vault: Any | None = None
        self.gate: Any | None = None
        self.ssh_signer: Any | None = None
        self.subscriber: Any | None = None
        self.notifier: Any | None = None

    async def start(self, cfg: SidecarConfig, paths: SupervisorPaths) -> int:
        """Bring services online in dependency order, degrading per service.

        Each service's startup failure is logged with its traceback and
        the bring-up continues — the typical restart-time case is a TCP
        port stolen while the container sat stopped, which must cost
        exactly the one service that lost its port.  A service whose
        dependency died fails its own startup the same way (hub →
        verdict, subscriber → hub).  Imports happen per service for the
        same reason: a broken module degrades that service, not the
        bundle.

        Returns the number of services that started; the caller treats
        ``0`` as "supervisor effectively absent".
        """
        starters: list[tuple[str, Callable[[], Awaitable[None]]]] = [
            ("verdict server", lambda: self._start_verdict(paths)),
            ("clearance hub", lambda: self._start_hub(paths)),
        ]
        # Git gate — only when the launch path wired it (gate_base_path +
        # gate_token both present); an unwired gate is not a failure.
        if cfg.gate_base_path and cfg.gate_token:
            starters.append(("git gate", lambda: self._start_gate(cfg, paths)))
        starters += [
            ("vault proxy", lambda: self._start_vault(cfg, paths)),
            ("ssh signer", lambda: self._start_ssh_signer(cfg, paths)),
            ("event notifier", lambda: self._start_subscriber(paths)),
        ]

        started = 0
        for label, bring_up in starters:
            try:
                await bring_up()
                started += 1
            except Exception:
                _logger.exception("%s failed to start — continuing without it", label)
        return started

    async def _start_verdict(self, paths: SupervisorPaths) -> None:
        """Varlink verdict server — execs ``terok-shield allow|deny``.

        First up: the hub holds a client to it.  The bind gets the
        ``terok_socket_t`` SELinux type via ``setsockcreatecon`` so
        confined containers (``container_t``) can ``connectto`` it once
        the host operator has installed the bundled policy
        (``sudo bash $(terok-sandbox setup --print-selinux-script)``).
        On non-SELinux hosts the helper is a no-op.
        """
        from terok_sandbox._util._selinux import socket_selinux_context
        from terok_sandbox.integrations.clearance import VerdictServer

        self.verdict = VerdictServer(
            socket_path=paths.verdict_socket,
            socket_context=socket_selinux_context,
        )
        await self.verdict.start()

    async def _start_hub(self, paths: SupervisorPaths) -> None:
        """Clearance hub — same SELinux socket labelling as the verdict bind."""
        from terok_sandbox._util._selinux import socket_selinux_context
        from terok_sandbox.integrations.clearance import ClearanceHub, VerdictClient

        self.hub = ClearanceHub(
            clearance_socket=paths.clearance_socket,
            reader_socket=paths.events_socket,
            verdict_client=VerdictClient(socket_path=paths.verdict_socket),
            socket_context=socket_selinux_context,
        )
        await self.hub.start()

    async def _start_gate(self, cfg: SidecarConfig, paths: SupervisorPaths) -> None:
        """Git gate — before the vault because it needs no DB and is the
        first service the container touches (the entrypoint clones
        through it immediately), so binding it early keeps it off the
        vault's slower SQLCipher-open path.  Serves
        ``<gate_base_path>/<project_id>.git`` gated on the single minted
        token; scope is the project the gate serves.  Socket mode binds a
        per-container socket inside ``container_runtime_dir``; TCP mode
        binds a per-container loopback port (raising if none was allocated).
        """
        from terok_sandbox.gate.server import GateServer

        if not cfg.gate_base_path or not cfg.gate_token:
            raise RuntimeError("gate starter invoked without gate wiring in the sidecar")
        if cfg.ipc_mode == "tcp":
            if not cfg.gate_port:
                raise RuntimeError(f"sidecar ipc_mode='tcp' but gate_port is {cfg.gate_port!r}")
            self.gate = GateServer(
                mirror_root=cfg.gate_base_path,
                token=cfg.gate_token,
                scope=cfg.project_id,
                host="127.0.0.1",
                port=cfg.gate_port,
            )
        else:
            self.gate = GateServer(
                mirror_root=cfg.gate_base_path,
                token=cfg.gate_token,
                scope=cfg.project_id,
                socket_path=paths.gate_socket,
            )
        await self.gate.start()

    async def _start_vault(self, cfg: SidecarConfig, paths: SupervisorPaths) -> None:
        """Vault HTTP/WS proxy, transport picked from the sidecar's ipc_mode."""
        from terok_sandbox.vault.daemon.token_broker import TcpBind, UnixBind, VaultProxy

        bind: UnixBind | TcpBind
        if cfg.ipc_mode == "tcp":
            if not cfg.tcp_port:
                raise RuntimeError(f"sidecar ipc_mode='tcp' but tcp_port is {cfg.tcp_port!r}")
            bind = TcpBind(host="127.0.0.1", port=cfg.tcp_port)
        else:
            bind = UnixBind(socket_path=paths.vault_socket)
        self.vault = VaultProxy(
            db_path=cfg.db_path,
            scope_id=cfg.scope_id,
            bind=bind,
            runtime_dir=cfg.runtime_dir,
        )
        await self.vault.start()

    async def _start_ssh_signer(self, cfg: SidecarConfig, paths: SupervisorPaths) -> None:
        """Token-gated SSH-agent, same transport split as the vault proxy."""
        from terok_sandbox.vault.ssh.signer import start_ssh_signer

        if cfg.ipc_mode == "tcp":
            if not cfg.ssh_signer_port:
                raise RuntimeError(
                    f"sidecar ipc_mode='tcp' but ssh_signer_port is {cfg.ssh_signer_port!r}"
                )
            self.ssh_signer = await start_ssh_signer(
                db_path=str(cfg.db_path),
                host="127.0.0.1",
                port=cfg.ssh_signer_port,
            )
        else:
            self.ssh_signer = await start_ssh_signer(
                db_path=str(cfg.db_path),
                socket_path=str(paths.ssh_signer_socket),
            )

    async def _start_subscriber(self, paths: SupervisorPaths) -> None:
        """Desktop notifier + the event subscriber that drives it.

        ``create_notifier`` is no-fail by contract — it degrades to a
        NullNotifier when no session bus is reachable.
        """
        from terok_sandbox.integrations.clearance import (
            NOTIFY_BLOCKED,
            NOTIFY_VERDICT,
            EventSubscriber,
            create_notifier,
        )

        self.notifier = await create_notifier("terok-supervisor")
        self.subscriber = EventSubscriber(
            self.notifier,
            socket_path=paths.clearance_socket,
            enabled_categories=frozenset({NOTIFY_BLOCKED, NOTIFY_VERDICT}),
        )
        await self.subscriber.start()

    async def stop(self) -> None:
        """Tear services down in reverse dependency order, swallowing failures."""
        for attr in ("subscriber", "hub", "verdict", "gate", "vault"):
            svc = getattr(self, attr, None)
            if svc is None:
                continue
            with contextlib.suppress(Exception):
                await svc.stop()
            setattr(self, attr, None)
        if self.ssh_signer is not None:
            # bare asyncio.Server — no .stop() method
            with contextlib.suppress(Exception):
                self.ssh_signer.close()
                await self.ssh_signer.wait_closed()
            self.ssh_signer = None
        if self.notifier is not None:
            with contextlib.suppress(Exception):
                await self.notifier.disconnect()
            self.notifier = None


# ── Internal helpers ────────────────────────────────────────────────────


def _install_signal_handlers(stop_event: asyncio.Event) -> None:
    """Wire SIGTERM/SIGINT into *stop_event* on the running loop.

    Soft-fails when no running loop is present (e.g. when imported
    outside ``asyncio.run`` for testing) — callers handle the signal
    chain themselves in that case.
    """
    try:
        loop = asyncio.get_running_loop()
    except RuntimeError:
        return
    for sig in (signal.SIGTERM, signal.SIGINT):
        try:
            loop.add_signal_handler(sig, stop_event.set)
        except RuntimeError:
            # Windows / restricted execution environments — fall back to
            # signal.signal so at least one of the two paths fires.
            # ``NotImplementedError`` is a ``RuntimeError`` subclass, so
            # this single except still covers add_signal_handler's
            # "not implemented on this platform" path.
            signal.signal(sig, lambda *_: stop_event.set())


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
