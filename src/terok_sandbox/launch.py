# SPDX-FileCopyrightText: 2026 Jiri Vyskocil
# SPDX-License-Identifier: Apache-2.0

"""Per-container wiring for user-owned containers.

`prepare`/`run`/`cleanup` compose podman flags that wire a caller-owned
container into the sandbox's services (vault token broker, vault SSH
signer, git gate, shield egress firewall) and persist enough per-container
state for `cleanup` to be a no-arg reverse of `prepare`.

Container lifecycle stays with the user; sandbox owns only the services
and the per-container ancillary state (tokens, shield rules, meta JSON).
"""

from __future__ import annotations

import dataclasses
import json
import os
import shlex
import shutil
import socket as _socket
import subprocess  # nosec B404 — podman inspect helper, fixed argv
import sys
import time
from dataclasses import dataclass
from pathlib import Path

from terok_util import podman_userns_args

from .config import CONTAINER_RUNTIME_DIR, SandboxConfig
from .doctor import CheckVerdict, DoctorCheck
from .gate.tokens import mint_gate_token
from .sandbox import Sharing, VolumeSpec

# Container-side path where bridge resources are bind-mounted (runtime
# pattern) or `COPY`ed into the image (build-time pattern).  The host
# source is always the package's ``resources/bridges/`` directory.
CONTAINER_BRIDGES_DIR = "/usr/local/share/terok-sandbox/bridges"

# Loopback TCP port the in-container vault HTTP bridge listens on in
# socket-transport mode.  Constant by design — every layer (this module,
# ensure-bridges.sh, downstream tools) imports the same value so the
# wire stays in sync.
LOOPBACK_VAULT_PORT = 9419

# Podman flags sandbox owns and rejects from user-supplied trailing args
# in `run`.  Mirrors terok-shield's set and extends it with ``--userns``
# (the bind-mounted socket UIDs depend on the host UID match) and ``-v``
# targets sandbox manages — those are checked separately because flag
# values, not names, are the collision surface.
SANDBOX_MANAGED_FLAGS = frozenset(
    {
        "--name",
        "--network",
        "--hooks-dir",
        "--annotation",
        "--cap-add",
        "--cap-drop",
        "--userns",
    }
)
_FLAG_ALIASES: dict[str, str] = {"--net": "--network"}

# Volume mount targets sandbox emits.  A user `-v ...:<target>` that
# overlaps any of these — or any path under ``CONTAINER_RUNTIME_DIR``
# itself — would shadow sandbox's sockets/mounts.
_MANAGED_VOLUME_TARGETS = frozenset(
    {
        CONTAINER_BRIDGES_DIR,
        CONTAINER_RUNTIME_DIR,
        f"{CONTAINER_RUNTIME_DIR}/vault.sock",
        f"{CONTAINER_RUNTIME_DIR}/ssh-agent.sock",
        f"{CONTAINER_RUNTIME_DIR}/gate-server.sock",
    }
)


@dataclass(frozen=True)
class WiringPlan:
    """Subsystems activated for a single prepare/run invocation.

    Persisted to ``meta.json`` so `cleanup` reverses exactly what was
    activated, without re-running the flag-defaults dance.
    """

    scope: str | None
    shield: bool
    gate: bool
    broker: bool
    ssh: bool

    def to_dict(self) -> dict[str, object]:
        """Return a JSON-serialisable representation."""
        return {
            "scope": self.scope,
            "shield": self.shield,
            "gate": self.gate,
            "broker": self.broker,
            "ssh": self.ssh,
        }

    @classmethod
    def from_dict(cls, data: dict[str, object]) -> WiringPlan:
        """Construct from a previously-persisted ``to_dict`` payload."""
        return cls(
            scope=data.get("scope"),  # type: ignore[arg-type]
            shield=bool(data.get("shield")),
            gate=bool(data.get("gate")),
            broker=bool(data.get("broker")),
            ssh=bool(data.get("ssh")),
        )

    def needs_bridges(self) -> bool:
        """True when any container-side bridge will be used."""
        return self.gate or self.broker or self.ssh


@dataclass(frozen=True)
class PerContainerResources:
    """Per-container socket dir + (for TCP mode) ports.

    Allocated once per launch so the same values reach mount flags,
    env vars, and the sidecar JSON the supervisor reads.  Keeps
    concurrent containers from colliding on host-global filenames or
    ports.
    """

    container_runtime_dir: Path
    """Host-side directory that becomes ``/run/terok/`` inside the
    container.  Contains the supervisor-bound ``vault.sock`` /
    ``ssh-agent.sock``.  Created (mode 0700) before the bind mount."""

    token_broker_port: int | None
    """Per-container TCP port for the vault proxy in TCP mode; ``None``
    in socket mode."""

    ssh_signer_port: int | None
    """Per-container TCP port for the SSH signer in TCP mode; ``None``
    in socket mode."""

    gate_port: int | None
    """Per-container TCP port for the git gate in TCP mode; ``None``
    in socket mode."""


def allocate_per_container_resources(cfg: SandboxConfig, container: str) -> PerContainerResources:
    """Compute per-container paths + (for TCP mode) ports.

    Both transport modes get a per-container directory under
    ``cfg.runtime_dir/run/<container>`` (mode 0700) that the caller
    bind-mounts at ``/run/terok/`` inside the container.  In TCP mode,
    two free ports are claimed via ``bind(0)`` + ``getsockname`` +
    close so each container gets its own pair instead of fighting
    over the singleton from ``cfg``.

    The narrow window between ``bind(0)``'s close and the supervisor's
    re-bind on the same port is an EADDRINUSE-loud failure mode, not
    silent breakage.
    """
    container_runtime_dir = cfg.ensure_container_runtime_dir(container)

    if cfg.services_mode != "tcp":
        return PerContainerResources(
            container_runtime_dir=container_runtime_dir,
            token_broker_port=None,
            ssh_signer_port=None,
            gate_port=None,
        )

    # Allocate all three ports against open sockets *simultaneously* —
    # consecutive ``bind(0)`` + close pairs can legitimately hand back
    # the same port (the kernel is free to reuse the just-freed slot
    # before the next call) and that would crash one of the services on
    # startup with ``EADDRINUSE``.
    broker_port, signer_port, gate_port = _pick_free_tcp_ports(3)
    return PerContainerResources(
        container_runtime_dir=container_runtime_dir,
        token_broker_port=broker_port,
        ssh_signer_port=signer_port,
        gate_port=gate_port,
    )


def _pick_free_tcp_ports(count: int) -> tuple[int, ...]:
    """*count* distinct kernel-assigned free TCP ports.

    Holds every socket bound at the same time so the kernel can't reuse
    the same port across the ``bind(0)`` calls.
    """
    socks = [_socket.socket(_socket.AF_INET, _socket.SOCK_STREAM) for _ in range(count)]
    try:
        for s in socks:
            s.bind(("127.0.0.1", 0))
        return tuple(s.getsockname()[1] for s in socks)
    finally:
        for s in socks:
            s.close()


def bridges_resource_dir() -> Path:
    """Filesystem path to the bridge resources shipped with this package."""
    return Path(__file__).resolve().parent / "resources" / "bridges"


def run_state_dir(cfg: SandboxConfig, container: str) -> Path:
    """Per-container state directory used by `prepare`/`cleanup`."""
    return cfg.state_dir / "sandbox" / "runs" / container


def _meta_path(state_dir: Path) -> Path:
    """Return the meta JSON path for a given run state directory."""
    return state_dir / "meta.json"


def _write_meta(state_dir: Path, plan: WiringPlan) -> None:
    """Persist the wiring plan to ``state_dir/meta.json``."""
    state_dir.mkdir(parents=True, exist_ok=True)
    _meta_path(state_dir).write_text(json.dumps(plan.to_dict(), indent=2) + "\n")


def _read_meta(state_dir: Path) -> WiringPlan | None:
    """Read the wiring plan from ``state_dir/meta.json``, or ``None``."""
    path = _meta_path(state_dir)
    if not path.is_file():
        return None
    try:
        return WiringPlan.from_dict(json.loads(path.read_text()))
    except (json.JSONDecodeError, OSError):
        return None


# ---------------------------------------------------------------------------
# Args composition
# ---------------------------------------------------------------------------


def _validate_container_name(container: str) -> None:
    """Reject a container name that isn't a safe single path component.

    ``container`` is interpolated into host state + runtime directory
    paths ([`run_state_dir`][terok_sandbox.launch.run_state_dir],
    [`allocate_per_container_resources`][terok_sandbox.launch.allocate_per_container_resources])
    and the sidecar filename — all of which are created, chmod'd, and
    ``rmtree``'d.  A name carrying a path separator or parent-ref could
    redirect those filesystem operations outside their roots, so reject it
    before anything is touched.  Mirrors the supervisor-side guard in
    [`load_sidecar`][terok_sandbox.supervisor.main.load_sidecar].
    """
    if not container or "/" in container or container in (".", ".."):
        raise SystemExit(f"unsafe container name (not a single path component): {container!r}")


def _rollback_compose_state(
    cfg: SandboxConfig,
    container: str,
    plan: WiringPlan,
    per_container: PerContainerResources,
    state_dir: Path,
) -> None:
    """Best-effort teardown of the durable state ``compose`` laid down.

    Called when a launch aborts after token minting + directory creation
    but before the container starts, so the failed attempt orphans
    nothing: revokes the per-container phantom tokens and removes the
    state + runtime directories.
    """
    if plan.scope is not None:
        try:
            db = cfg.open_credential_db()
        except Exception:  # noqa: BLE001 — best-effort: nothing to revoke if the vault won't open
            db = None
        if db is not None:
            try:
                db.revoke_tokens(plan.scope, container)
            finally:
                db.close()
    shutil.rmtree(per_container.container_runtime_dir, ignore_errors=True)
    shutil.rmtree(state_dir, ignore_errors=True)


def compose(
    container: str,
    *,
    cfg: SandboxConfig,
    shield: bool,
    gate: bool,
    broker: bool,
    scope: str | None,
    profiles: tuple[str, ...] | None = None,
) -> tuple[list[str], WiringPlan]:
    """Compose podman args for one prepare/run invocation.

    Mints any tokens needed for the active subsystems (broker/gate/ssh),
    creates the per-container state directory, persists ``meta.json``,
    and returns the assembled podman flag list plus the resolved plan.

    Subsystems that require ``scope`` are silently disabled (with a
    stderr note) when ``scope`` is ``None`` — sandbox only enforces the
    fail-closed property; nudging the caller toward a useful invocation
    is the job of the CLI layer.

    Raises ``SystemExit`` if shield setup is required (propagated from
    [`ShieldManager.pre_start`][terok_sandbox.integrations.shield.ShieldManager.pre_start]).
    """
    _validate_container_name(container)

    from .integrations.shield import ShieldManager

    # Profile override flows through cfg so shield's internal builder
    # (which reads ``cfg.shield_profiles``) picks it up without a new
    # parameter on every layer.  ``__post_init__`` re-runs and skips
    # port re-allocation because every port is already concrete.
    if profiles:
        cfg = dataclasses.replace(cfg, shield_profiles=tuple(profiles))

    state_dir = run_state_dir(cfg, container)
    state_dir.mkdir(parents=True, exist_ok=True)

    # Resolve subsystem activation against the scope precondition.  Three
    # of the four subsystems are scope-bound; silently skipping them with
    # a stderr note when scope is missing keeps the default-on policy
    # honest without forcing users to type --no-broker --no-gate when
    # they just want shield.
    effective_gate = gate and scope is not None
    effective_broker = broker and scope is not None
    effective_ssh = scope is not None

    for name, requested, effective in (
        ("gate", gate, effective_gate),
        ("broker", broker, effective_broker),
    ):
        if requested and not effective:
            print(
                f"note: --{name} requires --scope; skipping (use --no-{name} to silence)",
                file=sys.stderr,
            )

    plan = WiringPlan(
        scope=scope,
        shield=shield,
        gate=effective_gate,
        broker=effective_broker,
        ssh=effective_ssh,
    )

    args: list[str] = []

    # Per-container runtime resources (host-side socket dir + TCP ports).
    # Allocated up front so shield's nft loopback-port allowlist sees
    # the actual broker/signer ports the supervisor will later bind.
    per_container = allocate_per_container_resources(cfg, container)
    loopback_ports = tuple(
        p
        for p in (
            per_container.gate_port,
            per_container.token_broker_port,
            per_container.ssh_signer_port,
        )
        if p is not None
    )

    # Shield first — its OCI hook expects to see the annotations before
    # podman processes any of the other flags.
    if shield:
        args += ShieldManager(
            state_dir, cfg, loopback_ports_override=loopback_ports or None
        ).pre_start(container)

    # User-namespace mapping ensures the host UID matches inside the
    # container, which both the bind-mounted sockets (0600 host-owned)
    # and the shield rules rely on.
    args += podman_userns_args()

    # Bridge resources: bind-mount the package's bridges/ directory so
    # `source /usr/local/share/terok-sandbox/bridges/ensure-bridges.sh`
    # works on any image with socat (and without `COPY`ing the scripts
    # in at build time).  Always emitted — harmless even when the image
    # already has its own copy.
    if plan.needs_bridges():
        args += _volume_args(
            VolumeSpec(
                bridges_resource_dir(),
                CONTAINER_BRIDGES_DIR,
                sharing=Sharing.SHARED,
                read_only=True,
                live=True,
            )
        )

    # Socket-mode: bind-mount the per-container dir at /run/terok/ so
    # the supervisor's later-bound vault.sock / ssh-agent.sock /
    # gate-server.sock appear inside the container at the well-known
    # paths.  The supervisor binds all three inside this directory.
    if cfg.services_mode == "socket" and (effective_broker or effective_ssh or effective_gate):
        args += _volume_args(
            VolumeSpec(
                per_container.container_runtime_dir,
                CONTAINER_RUNTIME_DIR,
                sharing=Sharing.SHARED,
                live=True,
            )
        )

    # Vault token broker env — the in-container bridge script reads the
    # loopback port (socket mode) or the host TCP port (TCP mode) to
    # build its forwarder.
    if effective_broker:
        if cfg.services_mode == "socket":
            args += ["-e", f"TEROK_VAULT_LOOPBACK_PORT={LOOPBACK_VAULT_PORT}"]
        elif per_container.token_broker_port is not None:
            args += ["-e", f"TEROK_TOKEN_BROKER_PORT={per_container.token_broker_port}"]

    # Gate — mint a per-container token; the gate lives in the
    # supervisor, which binds ``gate-server.sock`` inside the already-
    # mounted ``/run/terok/`` dir (socket mode) or a per-container
    # loopback port (TCP mode).  The token travels only via the sidecar
    # + the env var the in-container bridge reads.
    gate_token: str | None = None
    if effective_gate:
        gate_token = mint_gate_token()
        if cfg.services_mode == "socket":
            args += ["-e", f"TEROK_GATE_SOCKET={CONTAINER_RUNTIME_DIR}/gate-server.sock"]
        elif per_container.gate_port is not None:
            args += ["-e", f"TEROK_GATE_PORT={per_container.gate_port}"]
        args += ["-e", f"TEROK_GATE_TOKEN={gate_token}"]

    # SSH signer — mint a phantom token + tell the bridge how to reach
    # the in-supervisor signer.  Socket mode: well-known in-container
    # path (the supervisor binds it inside the per-container dir mount).
    # TCP mode: per-container port from `bind(0)`.
    if effective_ssh:
        # Non-interactive: launch runs under asyncio, where a prompt_toolkit
        # passphrase prompt cannot own the loop.  A locked vault raises
        # NoPassphraseError here — the frontend unlocks before launch.
        db = cfg.open_credential_db(prompt_on_tty=False)
        try:
            ssh_token = db.create_token(scope, container, scope, "ssh")
        finally:
            db.close()
        args += ["-e", f"TEROK_SSH_SIGNER_TOKEN={ssh_token}"]
        if cfg.services_mode == "socket":
            args += ["-e", f"TEROK_SSH_SIGNER_SOCKET={CONTAINER_RUNTIME_DIR}/ssh-agent.sock"]
        elif per_container.ssh_signer_port is not None:
            args += ["-e", f"TEROK_SSH_SIGNER_PORT={per_container.ssh_signer_port}"]

    args += ["--name", container]

    _write_meta(state_dir, plan)
    sidecar_path = write_sidecar(
        container,
        cfg=cfg,
        per_container=per_container,
        scope_id=plan.scope or "",
        # The supervisor uses ``project_id`` as the gate's scope (the repo
        # it serves is ``<project_id>.git``).  In standalone sandbox the
        # scope *is* the repo name, so carry it here when the gate is wired.
        project_id=(plan.scope or "") if (gate_token and plan.scope) else "",
        gate_base_path=str(cfg.gate_base_path) if gate_token else None,
        gate_token=gate_token,
        gate_port=per_container.gate_port if gate_token else None,
    )
    if sidecar_path is None:
        # Fail closed: a launch with no sidecar means the supervisor never
        # starts, so the container would hit dead vault/SSH/gate endpoints.
        # compose() has already minted phantom tokens and laid down the
        # state + runtime dirs, so roll those back before aborting to avoid
        # orphaning them for a container that never started.
        _rollback_compose_state(cfg, container, plan, per_container, state_dir)
        raise SystemExit(
            "sidecar write failed; aborting launch (vault/SSH endpoints would be dead)"
        )
    args += ["--annotation", f"terok.sandbox.sidecar={sidecar_path}"]
    return args, plan


def write_sidecar(
    container_name: str,
    *,
    cfg: SandboxConfig,
    per_container: PerContainerResources,
    scope_id: str = "",
    project_id: str = "",
    task_id: str = "",
    dossier_path: Path | str | None = None,
    gate_base_path: str | None = None,
    gate_token: str | None = None,
    gate_port: int | None = None,
) -> Path | None:
    """Persist the per-container sidecar config the supervisor reads.

    The canonical writer for the whole package chain — `compose` calls
    it for standalone sandbox runs and terok-executor's
    ``AgentRunner.launch_prepared`` calls it for terok tasks — so the
    schema [`load_sidecar`][terok_sandbox.supervisor.main.load_sidecar]
    parses has exactly one producer.

    Path: ``<cfg.state_dir>/sidecar/<container-name>.json``.  Returns
    the absolute path on success — the caller emits it as the
    ``terok.sandbox.sidecar`` OCI annotation that triggers the hook
    and tells it where to find this file.  No XDG guessing, no
    name-vs-id rename: one anchor, one path.  The file survives
    container stop/start cycles by design (its ports and tokens must
    keep matching the container's immutable env, and the createRuntime
    hook re-reads it on every ``podman start``); it is removed at real
    teardown by
    [`remove_container_state`][terok_sandbox.launch.remove_container_state].

    Socket paths are NOT carried in the sidecar — the supervisor
    derives them from the container name + runtime dir via
    [`SupervisorPaths.for_container`][terok_sandbox.supervisor.main.SupervisorPaths.for_container].
    TCP ports ARE carried because the launch path allocates them
    fresh per container via ``bind(0)``.  Gate config travels only
    when the gate is wired — the supervisor composes the gate iff
    both ``gate_base_path`` and ``gate_token`` are present.

    Best-effort: a write failure logs to stderr and returns ``None``;
    callers pick their own policy (`compose` rolls back and aborts the
    launch, the executor raises its ``BuildError``).

    Raises:
        ValueError: If ``container_name`` is empty or carries a path
            separator / traversal segment — such a name would let the
            write escape ``state_dir/sidecar``.
    """
    if (
        not container_name
        or container_name in (".", "..")
        or "/" in container_name
        or "\\" in container_name
    ):
        raise ValueError(f"invalid container name: {container_name!r}")

    sidecar_dir = cfg.state_dir / "sidecar"
    try:
        sidecar_dir.mkdir(parents=True, exist_ok=True)
    except OSError as exc:
        print(f"warning: sidecar dir setup failed: {exc}", file=sys.stderr)
        return None

    payload: dict[str, object] = {
        "container_name": container_name,
        "ipc_mode": cfg.services_mode,
        "db_path": str(cfg.db_path),
        "scope_id": scope_id or "",
        "project_id": project_id or "",
        "task_id": task_id or "",
        # The supervisor runs in crun's rootless userns where geteuid==0,
        # so it can't re-derive runtime_dir from path resolvers.  Pin it
        # here and read back verbatim.
        "runtime_dir": str(cfg.runtime_dir),
    }
    if cfg.services_mode == "tcp":
        payload["tcp_port"] = per_container.token_broker_port
        payload["ssh_signer_port"] = per_container.ssh_signer_port
    if dossier_path is not None:
        payload["dossier_path"] = str(dossier_path)
    if gate_base_path is not None:
        payload["gate_base_path"] = gate_base_path
    if gate_token is not None:
        payload["gate_token"] = gate_token
    if gate_port is not None:
        payload["gate_port"] = gate_port

    target = sidecar_dir / f"{container_name}.json"
    # The payload can carry a live gate_token, so the file must not be
    # world-readable.  Create it 0600 directly (the process umask would
    # otherwise leave it 0644) and fchmod to also cover the re-launch case
    # where the file already exists with looser permissions.
    #
    # ``O_NOFOLLOW`` refuses to open the final path component if it is a
    # symlink — a pre-planted symlink at ``target`` (e.g. pointing at
    # ``~/.ssh/authorized_keys``) would otherwise let this write clobber
    # an arbitrary file with the token payload.  The container_name
    # traversal guard above blocks one vector; this closes the symlink
    # vector at the open(2) level.  The ``ELOOP`` it raises is an OSError,
    # so it falls into the existing soft-fail branch and returns ``None``.
    try:
        fd = os.open(target, os.O_WRONLY | os.O_CREAT | os.O_TRUNC | os.O_NOFOLLOW, 0o600)
        with os.fdopen(fd, "w", encoding="utf-8") as fh:
            os.fchmod(fh.fileno(), 0o600)
            json.dump(payload, fh, indent=2)
    except OSError as exc:
        print(f"warning: sidecar write failed: {exc}", file=sys.stderr)
        return None
    return target


def remove_container_state(container: str, *, cfg: SandboxConfig) -> None:
    """Remove the per-container sidecar + host-side runtime directory.

    The inverse of
    [`allocate_per_container_resources`][terok_sandbox.launch.allocate_per_container_resources]
    + [`write_sidecar`][terok_sandbox.launch.write_sidecar]: deletes the
    state keyed by container *name* that deliberately outlives a stop
    (the sidecar survives poststop so restarts come back supervised).
    Call at real teardown — after the container is removed — never at
    mere stop.  Idempotent; missing state is a no-op.

    Raises:
        ValueError: If ``container`` is empty or carries a path
            separator / traversal segment — such a name would redirect
            the unlink/rmtree outside the sandbox-owned directories.
    """
    if not container or container in (".", "..") or "/" in container or "\\" in container:
        raise ValueError(f"invalid container name: {container!r}")
    (cfg.state_dir / "sidecar" / f"{container}.json").unlink(missing_ok=True)
    shutil.rmtree(cfg.container_runtime_dir(container), ignore_errors=True)


#: Minimum age before a sidecar with no matching container counts as a
#: stray.  ``terok-sandbox prepare`` writes the sidecar before the
#: operator's own ``podman run``, so a fresh file may simply not have
#: its container *yet*; an hour comfortably covers that gap.
_STRAY_GRACE_S = 3600.0


def make_stray_sidecar_check(cfg: SandboxConfig | None = None) -> DoctorCheck:
    """Sweep per-container state left behind by out-of-band removal.

    The sidecar deliberately survives ``podman stop`` — a stopped
    container must come back supervised, and the preserved file is the
    only wiring that still matches the container's immutable env (see
    [`write_sidecar`][terok_sandbox.launch.write_sidecar]).  The real
    teardown paths ([`cleanup`][terok_sandbox.launch.cleanup], terok's
    task delete) remove it; a container removed *outside* those paths
    (bare ``podman rm``) strands its sidecar.  Strays are inert —
    launches overwrite by name and the hook only fires via a live
    container's annotation — so this reconciliation is hygiene, not
    correctness.

    The check acts on what it finds: a sidecar whose container podman
    no longer knows, and which is past the prepare→run grace window, is
    swept on the spot via
    [`remove_container_state`][terok_sandbox.launch.remove_container_state],
    and the verdict reports what was done.  When podman is unreachable,
    live and stray are indistinguishable and the sweep is skipped.

    Host-level like the recovery-key check: intentionally NOT bundled
    into
    [`sandbox_doctor_checks`][terok_sandbox.doctor.sandbox_doctor_checks]
    (that list renders per container); top-level callers append it so
    it runs exactly once.
    """

    def _eval(_rc: int, _stdout: str, _stderr: str) -> CheckVerdict:
        active_cfg = cfg if cfg is not None else SandboxConfig()
        sidecar_dir = active_cfg.state_dir / "sidecar"
        candidates = sorted(sidecar_dir.glob("*.json")) if sidecar_dir.is_dir() else []
        if not candidates:
            return CheckVerdict("ok", "no sidecars on disk")

        known = _podman_container_names()
        if known is None:
            return CheckVerdict(
                "warn", "podman unreachable — cannot tell live sidecars from strays"
            )

        now = time.time()
        swept: list[str] = []
        for path in candidates:
            name = path.stem
            if name in known:
                continue
            try:
                age = now - path.stat().st_mtime
            except OSError:
                continue  # vanished mid-scan — nothing left to sweep
            if age < _STRAY_GRACE_S:
                continue  # possibly a prepare→run gap; revisit next run
            remove_container_state(name, cfg=active_cfg)
            swept.append(name)
        if swept:
            return CheckVerdict("ok", f"swept {len(swept)} stray sidecar(s): {', '.join(swept)}")
        return CheckVerdict("ok", f"no strays among {len(candidates)} sidecar(s)")

    return DoctorCheck(
        category="env",
        label="Stray sidecar sweep",
        probe_cmd=[],
        evaluate=_eval,
        host_side=True,
    )


def _podman_container_names() -> frozenset[str] | None:
    """Names of every container podman knows (any state); ``None`` if unreachable.

    One ``podman ps --all`` call covers the whole sweep — per-candidate
    ``podman container exists`` probes would cost a subprocess each.
    Mirrors `_resolve_container_id`'s soft-fail stance on a missing /
    broken podman.
    """
    podman = shutil.which("podman")
    if podman is None:
        return None
    try:
        result = subprocess.run(  # noqa: S603  # nosec B603 — fixed argv, no user input
            [podman, "ps", "--all", "--format", "{{.Names}}"],
            capture_output=True,
            text=True,
            timeout=10,
            check=True,
        )
    except (OSError, subprocess.SubprocessError):
        return None
    return frozenset(line.strip() for line in result.stdout.splitlines() if line.strip())


def _volume_args(vol: VolumeSpec) -> list[str]:
    """Return the ``-v <mount-arg>`` pair for *vol*."""
    return ["-v", vol.to_mount_arg()]


# ---------------------------------------------------------------------------
# Run — exec into podman
# ---------------------------------------------------------------------------


def exec_podman(sandbox_args: list[str], podman_args: list[str]) -> None:
    """Replace this process with ``podman run``.

    Validates that *podman_args* (everything the user typed after ``--``)
    doesn't collide with sandbox-owned flags or volume targets, then
    ``os.execv``s into podman.  Caller doesn't return.
    """
    if not podman_args:
        raise SystemExit(
            "No image specified. Usage: terok-sandbox run <container> -- <image> [cmd...]"
        )

    reject_managed_flags(podman_args)
    reject_managed_volumes(podman_args)

    podman = _find_podman()
    argv = [podman, "run", *sandbox_args, *podman_args]
    # ``argv`` is fully constructed in-process and uses an absolute path
    # to podman, so shell interpretation and PATH spoofing do not apply.
    os.execv(podman, argv)  # nosec B606


def reject_managed_flags(podman_args: list[str]) -> None:
    """Reject user-supplied flags that sandbox owns.

    Mirrors terok-shield's ``_reject_shield_managed_flags`` and adds
    sandbox-specific entries (e.g. ``--userns``).
    """
    conflicts: set[str] = set()
    for arg in podman_args:
        if not arg.startswith("--"):
            continue
        flag = arg.split("=", 1)[0]
        flag = _FLAG_ALIASES.get(flag, flag)
        if flag in SANDBOX_MANAGED_FLAGS:
            conflicts.add(flag)
    if conflicts:
        raise SystemExit(
            f"Flag(s) managed by terok-sandbox, cannot override: {', '.join(sorted(conflicts))}"
        )


def reject_managed_volumes(podman_args: list[str]) -> None:
    """Reject ``-v host:target`` whose target overlaps a sandbox mount."""
    conflicts: set[str] = set()
    iterator = iter(podman_args)
    for arg in iterator:
        spec: str | None = None
        if arg == "-v" or arg == "--volume":
            spec = next(iterator, None)
        elif arg.startswith("--volume="):
            spec = arg.split("=", 1)[1]
        if not spec:
            continue
        parts = spec.split(":")
        if len(parts) < 2:
            continue
        target = parts[1]
        # Block exact matches plus any path under ``CONTAINER_RUNTIME_DIR``
        # — sandbox owns that whole subtree, so a deeper user mount would
        # still hide a freshly-bound supervisor socket.
        if target in _MANAGED_VOLUME_TARGETS or target.startswith(f"{CONTAINER_RUNTIME_DIR}/"):
            conflicts.add(target)
    if conflicts:
        raise SystemExit(
            "Volume target(s) managed by terok-sandbox, cannot override: "
            f"{', '.join(sorted(conflicts))}"
        )


def _find_podman() -> str:
    """Locate the podman binary."""
    found = shutil.which("podman")
    if found:
        resolved = Path(found).resolve()
        if resolved.is_file() and os.access(resolved, os.X_OK):
            return str(resolved)
    raise SystemExit("podman binary not found. Install Podman to use 'terok-sandbox run'.")


# ---------------------------------------------------------------------------
# Cleanup
# ---------------------------------------------------------------------------


def cleanup(container: str, *, cfg: SandboxConfig) -> bool:
    """Reverse a prior `prepare`/`run` for *container*.

    Returns ``True`` when state was found and torn down, ``False`` when
    there was nothing to clean up.  Idempotent — safe to call repeatedly.
    """
    from .integrations.shield import ShieldManager

    state_dir = run_state_dir(cfg, container)
    plan = _read_meta(state_dir)
    if plan is None:
        return False

    # The gate token needs no revocation step: it is a stateless
    # pair-match between the container's env and the sidecar, so it
    # stops working the moment the supervisor dies, and its on-disk
    # copy goes away with the sidecar sweep below.
    #
    # Revoke vault tokens before tearing down shield: a still-running
    # container using a revoked token gets clean 401s, not stale 200s.
    if (plan.broker or plan.ssh) and plan.scope is not None:
        from .vault.store.db import (  # noqa: PLC0415
            NoPassphraseError,
            PlaintextDBFoundError,
            WrongPassphraseError,
        )

        try:
            # Non-interactive (teardown runs under asyncio): a locked vault
            # raises rather than prompting — handled below.
            db = cfg.open_credential_db(prompt_on_tty=False)
        except (
            OSError,
            NoPassphraseError,
            PlaintextDBFoundError,
            WrongPassphraseError,
        ) as exc:
            # Cleanup is best-effort: a missing DB (OSError) or a locked
            # / unencrypted / undecryptable vault (the three credential
            # exceptions) all collapse to "already revoked from the
            # caller's point of view" so shield/state teardown can still
            # proceed.  Any other RuntimeError is a real bug — let it
            # propagate.  Warn so
            # the operator knows the broker/SSH phantom tokens for this
            # container are still in the DB and should be cleaned up
            # after the next ``vault unlock``.
            print(
                f"warning: cleanup couldn't revoke broker/SSH tokens for"
                f" {plan.scope}/{container}: {type(exc).__name__}: {exc}\n"
                f"         tokens remain in the credentials DB; re-run"
                f" `terok-sandbox cleanup` after a `vault unlock` to purge them.",
                file=sys.stderr,
            )
            db = None
        if db is not None:
            try:
                db.revoke_tokens(plan.scope, container)
            finally:
                db.close()

    # Shield down is best-effort: when the container has already exited,
    # the OCI poststop hook has already removed the rules.  Resolve the
    # full container UUID via ``podman inspect`` because terok-shield's
    # per-container hub socket is keyed on the ID, not the name; a
    # vanished container surfaces as a non-zero exit and we skip the
    # call (the poststop hook handled it).
    if plan.shield:
        container_id = _resolve_container_id(container)
        if container_id is not None:
            try:
                ShieldManager(state_dir, cfg).down(container, container_id)
            except (SystemExit, OSError):
                pass

    # Sweep the per-container sidecar file and runtime dir.  This is
    # the real teardown point — the OCI poststop hook deliberately
    # leaves both in place so a stopped container can restart
    # supervised.
    remove_container_state(container, cfg=cfg)

    shutil.rmtree(state_dir, ignore_errors=True)
    return True


def _resolve_container_id(container: str) -> str | None:
    """Return the full podman container UUID for *container*, or ``None``.

    Used at cleanup time when the caller only carries the
    operator-facing container *name*: terok-shield's per-container hub
    socket is keyed on the UUID, and the API surface
    ([`ShieldManager.down`][terok_sandbox.integrations.shield.ShieldManager.down])
    requires it explicitly.

    Returns ``None`` when podman is unreachable or the container has
    already been pruned — both states mean "nothing to ask shield to
    tear down" because the OCI poststop hook has already fired by the
    time the container disappears from podman's catalogue.
    """
    podman = shutil.which("podman")
    if podman is None:
        return None
    try:
        result = subprocess.run(  # nosec B603 — argv fully constructed
            [podman, "inspect", "-f", "{{.Id}}", container],
            capture_output=True,
            text=True,
            timeout=5,
            check=False,
        )
    except (OSError, subprocess.TimeoutExpired):
        return None
    if result.returncode != 0:
        return None
    container_id = result.stdout.strip()
    return container_id or None


# ---------------------------------------------------------------------------
# Output formatting
# ---------------------------------------------------------------------------


def format_args(args: list[str], *, output_json: bool) -> str:
    """Return the printable form of an args list."""
    if output_json:
        return json.dumps(args)
    return " ".join(shlex.quote(a) for a in args)
