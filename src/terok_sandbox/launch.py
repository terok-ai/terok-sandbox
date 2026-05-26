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
from dataclasses import dataclass
from pathlib import Path

from terok_util import podman_userns_args

from .config import CONTAINER_RUNTIME_DIR, SandboxConfig
from .gate.tokens import TokenStore
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
    container_runtime_dir = cfg.runtime_dir / "run" / container
    container_runtime_dir.mkdir(parents=True, exist_ok=True)
    container_runtime_dir.chmod(0o700)

    if cfg.services_mode != "tcp":
        return PerContainerResources(
            container_runtime_dir=container_runtime_dir,
            token_broker_port=None,
            ssh_signer_port=None,
        )

    # Allocate both ports against open sockets *simultaneously* — two
    # consecutive ``bind(0)`` + close pairs can legitimately hand back
    # the same port (the kernel is free to reuse the just-freed slot
    # before the second call) and that would crash one of the two
    # services on startup with ``EADDRINUSE``.
    broker_port, signer_port = _pick_two_free_tcp_ports()
    return PerContainerResources(
        container_runtime_dir=container_runtime_dir,
        token_broker_port=broker_port,
        ssh_signer_port=signer_port,
    )


def _pick_two_free_tcp_ports() -> tuple[int, int]:
    """Two distinct kernel-assigned free TCP ports.

    Holds both sockets bound at the same time so the kernel can't
    reuse the same port across the two ``bind(0)`` calls.
    """
    with (
        _socket.socket(_socket.AF_INET, _socket.SOCK_STREAM) as a,
        _socket.socket(_socket.AF_INET, _socket.SOCK_STREAM) as b,
    ):
        a.bind(("127.0.0.1", 0))
        b.bind(("127.0.0.1", 0))
        return a.getsockname()[1], b.getsockname()[1]


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
            cfg.gate_port,
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
    # the supervisor's later-bound vault.sock / ssh-agent.sock appear
    # inside the container at the well-known paths.  Sub-mounts below
    # (gate-server.sock) land inside this directory.
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

    # Gate — mount the gate Unix socket and plant a per-container token.
    # The gate-server stays a host-singleton daemon for now (Phase B);
    # its socket mounts as a sub-mount inside the per-container
    # /run/terok/ dir.
    if effective_gate:
        token = TokenStore(cfg).create(scope, container)  # type: ignore[arg-type]
        if cfg.services_mode == "socket":
            args += _volume_args(
                VolumeSpec(
                    cfg.gate_socket_path,
                    f"{CONTAINER_RUNTIME_DIR}/gate-server.sock",
                    sharing=Sharing.SHARED,
                    live=True,
                )
            )
            args += ["-e", f"TEROK_GATE_SOCKET={CONTAINER_RUNTIME_DIR}/gate-server.sock"]
        elif cfg.gate_port is not None:
            args += ["-e", f"TEROK_GATE_PORT={cfg.gate_port}"]
        args += ["-e", f"TEROK_GATE_TOKEN={token}"]

    # SSH signer — mint a phantom token + tell the bridge how to reach
    # the in-supervisor signer.  Socket mode: well-known in-container
    # path (the supervisor binds it inside the per-container dir mount).
    # TCP mode: per-container port from `bind(0)`.
    #
    # If anything in this block raises, ``_write_meta`` is skipped and
    # cleanup() can't revoke the gate token we minted moments ago — roll
    # it back here so a locked vault doesn't leak an authorised gate
    # token.
    if effective_ssh:
        try:
            db = cfg.open_credential_db(prompt_on_tty=True)
            try:
                ssh_token = db.create_token(scope, container, scope, "ssh")
            finally:
                db.close()
        except BaseException:
            if effective_gate and scope is not None:
                TokenStore(cfg).revoke_for_task(scope, container)
            raise
        args += ["-e", f"TEROK_SSH_SIGNER_TOKEN={ssh_token}"]
        if cfg.services_mode == "socket":
            args += ["-e", f"TEROK_SSH_SIGNER_SOCKET={CONTAINER_RUNTIME_DIR}/ssh-agent.sock"]
        elif per_container.ssh_signer_port is not None:
            args += ["-e", f"TEROK_SSH_SIGNER_PORT={per_container.ssh_signer_port}"]

    args += ["--name", container]

    _write_meta(state_dir, plan)
    sidecar_path = _write_sidecar(cfg, container, plan, per_container)
    if sidecar_path is not None:
        args += ["--annotation", f"terok.sandbox.sidecar={sidecar_path}"]
    return args, plan


def _write_sidecar(
    cfg: SandboxConfig,
    container: str,
    plan: WiringPlan,
    per_container: PerContainerResources,
) -> Path | None:
    """Persist the per-container sidecar config the supervisor reads.

    Path: ``<cfg.state_dir>/sidecar/<container-name>.json``.  Returns
    the absolute path on success — the caller emits it as the
    ``terok.sandbox.sidecar`` OCI annotation that triggers the hook
    and tells it where to find this file.  No XDG guessing, no
    name-vs-id rename: one anchor, one path.

    Socket paths are NOT carried in the sidecar — the supervisor
    derives them from the container name + runtime dir via
    [`SupervisorPaths.for_container`][terok_sandbox.supervisor.main.SupervisorPaths.for_container].
    TCP ports ARE carried because the launch path allocates them
    fresh per container via ``bind(0)``.

    Best-effort: a write failure logs to stderr and returns ``None``;
    the launch proceeds but the supervisor never spawns (no annotation
    → no hook fire).
    """
    sidecar_dir = cfg.state_dir / "sidecar"
    try:
        sidecar_dir.mkdir(parents=True, exist_ok=True)
    except OSError as exc:
        print(f"warning: sidecar dir setup failed: {exc}", file=sys.stderr)
        return None

    payload: dict[str, object] = {
        "container_name": container,
        "ipc_mode": cfg.services_mode,
        "db_path": str(cfg.db_path),
        "scope_id": plan.scope or "",
        "project_id": "",
        "task_id": "",
        # The supervisor runs in crun's rootless userns where geteuid==0,
        # so it can't re-derive runtime_dir from path resolvers.  Pin it
        # here and read back verbatim.
        "runtime_dir": str(cfg.runtime_dir),
    }
    if cfg.services_mode == "tcp":
        payload["tcp_port"] = per_container.token_broker_port
        payload["ssh_signer_port"] = per_container.ssh_signer_port

    target = sidecar_dir / f"{container}.json"
    try:
        with target.open("w", encoding="utf-8") as fh:
            json.dump(payload, fh, indent=2)
    except OSError as exc:
        print(f"warning: sidecar write failed: {exc}", file=sys.stderr)
        return None
    return target


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

    # Revoke tokens before tearing down shield: a still-running container
    # using a revoked token gets clean 401s, not stale 200s.
    if plan.gate and plan.scope is not None:
        TokenStore(cfg).revoke_for_task(plan.scope, container)
    if (plan.broker or plan.ssh) and plan.scope is not None:
        from .vault.store.db import (  # noqa: PLC0415
            NoPassphraseError,
            PlaintextDBFoundError,
            WrongPassphraseError,
        )

        try:
            db = cfg.open_credential_db(prompt_on_tty=True)
        except (
            OSError,
            NoPassphraseError,
            PlaintextDBFoundError,
            WrongPassphraseError,
            SystemExit,
        ) as exc:
            # Cleanup is best-effort: a missing DB (OSError), a locked
            # / unencrypted / undecryptable vault (the three credential
            # exceptions) or a cancelled prompt (SystemExit) all collapse
            # to "already revoked from the caller's point of view" so
            # shield/state teardown can still proceed.  Any other
            # RuntimeError is a real bug — let it propagate.  Warn so
            # the operator knows the broker/SSH phantom tokens for this
            # container are still in the DB and should be cleaned up
            # after the next ``vault unlock``.
            print(
                f"warning: cleanup couldn't revoke broker/SSH tokens for"
                f" {plan.scope}/{container}: {type(exc).__name__}: {exc}\n"
                f"         tokens remain in the credentials DB until the next"
                f" `terok-sandbox credentials revoke` after a `vault unlock`.",
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

    # Sweep the per-container sidecar file and runtime dir.  The OCI
    # poststop hook would normally cover this, but ``terok-sandbox
    # cleanup`` runs out-of-band from podman lifecycle (e.g.
    # ``prepare`` without a corresponding ``podman run``).
    (cfg.state_dir / "sidecar" / f"{container}.json").unlink(missing_ok=True)
    shutil.rmtree(cfg.runtime_dir / "run" / container, ignore_errors=True)

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
