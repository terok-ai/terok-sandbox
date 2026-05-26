# SPDX-FileCopyrightText: 2026 Jiri Vyskocil
# SPDX-License-Identifier: Apache-2.0

"""The supervisor coroutine — ``run_supervisor(container_id)``.

One asyncio loop composes:

* [`VaultProxy`][terok_sandbox.vault.daemon.token_broker.VaultProxy] — vault HTTP/WS
  proxy, transport picked from the sidecar (``socket`` → ``UnixBind``,
  ``tcp`` → ``TcpBind``).
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
container exits it tears the four services down in reverse order and
returns 0.

Hidden from main user help; invoked by the OCI hook chain only.
"""

from __future__ import annotations

import asyncio
import contextlib
import json
import logging
import signal
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from typing import Any

_logger = logging.getLogger("terok-supervisor")


# ── Sidecar config / paths ──────────────────────────────────────────────


@dataclass(frozen=True)
class SidecarConfig:
    """Per-container config the supervisor reads from the sidecar JSON.

    Written by ``terok-sandbox prepare`` (and equivalents in
    terok-executor / terok) at container-creation time.  Keyed by
    container name initially; promoted to a container-ID-keyed
    filename on first hook fire (see
    [`terok_sandbox.resources.hooks._supervisor_state.load_sidecar`][terok_sandbox.resources.hooks._supervisor_state.load_sidecar]).
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
    clearance_socket: Path
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
            clearance_socket=clearance_root / "clearance" / f"{short_id}.sock",
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
    except (OSError, ValueError) as exc:
        _logger.error("sidecar parse failure for %s: %s", sidecar_path, exc)
        return None
    if not isinstance(raw, dict):
        _logger.error("sidecar is not a JSON object: %s", sidecar_path)
        return None
    try:
        container_name = str(raw.get("container_name", "")).strip()
        if not container_name:
            _logger.error("sidecar missing required container_name: %s", sidecar_path)
            return None
        ipc_mode = str(raw.get("ipc_mode", "socket"))
        if ipc_mode not in ("socket", "tcp"):
            _logger.error(
                "sidecar ipc_mode must be 'socket' or 'tcp', got %r: %s",
                ipc_mode,
                sidecar_path,
            )
            return None
        return SidecarConfig(
            container_name=container_name,
            ipc_mode=ipc_mode,
            db_path=Path(str(raw["db_path"])),
            runtime_dir=Path(str(raw["runtime_dir"])),
            scope_id=raw.get("scope_id") or None,
            project_id=str(raw.get("project_id") or ""),
            task_id=str(raw.get("task_id") or ""),
            tcp_port=(int(raw["tcp_port"]) if raw.get("tcp_port") is not None else None),
            ssh_signer_port=(
                int(raw["ssh_signer_port"]) if raw.get("ssh_signer_port") is not None else None
            ),
            dossier_path=Path(raw["dossier_path"]) if raw.get("dossier_path") else None,
        )
    except (KeyError, TypeError, ValueError) as exc:
        _logger.error("sidecar schema error in %s: %s", sidecar_path, exc)
        return None


# ── Entry point ─────────────────────────────────────────────────────────


async def run_supervisor(container_id: str, sidecar_path: Path) -> int:
    """Compose + run the per-container service bundle.

    Lifecycle:

    1. Load the sidecar JSON from *sidecar_path*; bail with exit code
       2 on parse / missing.
    2. Bring up [`VerdictServer`][terok_clearance.VerdictServer] →
       [`ClearanceHub`][terok_clearance.ClearanceHub] →
       [`VaultProxy`][terok_sandbox.vault.daemon.token_broker.VaultProxy] →
       desktop subscriber.  Each ``start()`` is awaited in turn; a
       failure unwinds anything already started before re-raising.
    3. Install SIGTERM / SIGINT handlers that race with ``podman wait``
       so a host-side ``terok-sandbox supervisor`` invocation can be
       stopped cleanly with Ctrl-C.
    4. Await ``podman wait <container_id>``.  When it returns, tear
       everything down in reverse and return 0.

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
        paths.verdict_socket,
        paths.control_socket,
        paths.vault_socket,
        paths.ssh_signer_socket,
    ):
        sock.parent.mkdir(parents=True, exist_ok=True)
        # ``bind_hardened`` refuses group/world-accessible parents;
        # explicit chmod overrides crun's permissive rootless umask.
        sock.parent.chmod(0o700)

    services = _Services()
    stop_event = asyncio.Event()
    _install_signal_handlers(stop_event)

    try:
        await services.start(cfg, paths)
    except Exception:
        _logger.exception("supervisor failed to start services for %s", container_id)
        await services.stop()
        return 3

    try:
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
    finally:
        await services.stop()
        # rmtree the per-container dir so concurrent / future containers
        # never see a stale socket inode at the well-known basenames.
        import shutil as _shutil

        _shutil.rmtree(paths.container_runtime_dir, ignore_errors=True)

    return 0


# ── Service composition (internal) ──────────────────────────────────────


class _Services:
    """The five-service bundle plus its teardown sequence.

    Order matters at teardown — verdict and hub talk to each other, so
    the hub goes down first, then verdict.  The vault stops last
    because the container's outbound API calls may still be in flight
    when ``podman wait`` returns.
    """

    def __init__(self) -> None:
        self.verdict: Any | None = None
        self.hub: Any | None = None
        self.vault: Any | None = None
        self.ssh_signer: Any | None = None
        self.subscriber: Any | None = None
        self.notifier: Any | None = None

    async def start(self, cfg: SidecarConfig, paths: SupervisorPaths) -> None:
        """Bring all services online in dependency order."""
        from terok_sandbox._util._selinux import socket_selinux_context
        from terok_sandbox.integrations.clearance import (
            NOTIFY_BLOCKED,
            NOTIFY_VERDICT,
            ClearanceHub,
            EventSubscriber,
            VerdictServer,
            create_notifier,
        )
        from terok_sandbox.vault.daemon.token_broker import TcpBind, UnixBind, VaultProxy

        # Verdict first — the hub holds a client to it.  Both binds get
        # the ``terok_socket_t`` SELinux type via ``setsockcreatecon`` so
        # confined containers (``container_t``) can ``connectto`` them
        # once the host operator has installed the bundled policy
        # (``sudo bash $(terok-sandbox setup --print-selinux-script)``).
        # On non-SELinux hosts the helper is a no-op.
        self.verdict = VerdictServer(
            socket_path=paths.verdict_socket,
            socket_context=socket_selinux_context,
        )
        await self.verdict.start()

        from terok_sandbox.integrations.clearance import VerdictClient

        self.hub = ClearanceHub(
            clearance_socket=paths.clearance_socket,
            verdict_client=VerdictClient(socket_path=paths.verdict_socket),
            socket_context=socket_selinux_context,
        )
        await self.hub.start()

        bind: UnixBind | TcpBind
        if cfg.ipc_mode == "tcp":
            if not cfg.tcp_port:
                raise RuntimeError(f"sidecar ipc_mode='tcp' but tcp_port is {cfg.tcp_port!r}")
            bind = TcpBind(host="127.0.0.1", port=cfg.tcp_port)
        else:
            bind = UnixBind(socket_path=paths.vault_socket)
        self.vault = VaultProxy(db_path=cfg.db_path, scope_id=cfg.scope_id, bind=bind)
        await self.vault.start()

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

        # No-fail by contract — degrades to NullNotifier when no bus.
        self.notifier = await create_notifier("terok-supervisor")
        self.subscriber = EventSubscriber(
            self.notifier,
            socket_path=paths.clearance_socket,
            enabled_categories=frozenset({NOTIFY_BLOCKED, NOTIFY_VERDICT}),
        )
        await self.subscriber.start()

    async def stop(self) -> None:
        """Tear services down in reverse dependency order, swallowing failures."""
        import contextlib

        for attr in ("subscriber", "hub", "verdict", "vault"):
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
        except (NotImplementedError, RuntimeError):
            # Windows / restricted execution environments — fall back
            # to signal.signal so at least one of the two paths fires.
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
        # outlive the supervisor and pin the container ID.
        with contextlib.suppress(ProcessLookupError):
            proc.terminate()
        with contextlib.suppress(asyncio.CancelledError):
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
