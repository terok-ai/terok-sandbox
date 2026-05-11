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
import sys
from dataclasses import dataclass
from pathlib import Path

from .config import CONTAINER_RUNTIME_DIR, SandboxConfig
from .credentials.db import CredentialDB
from .gate.tokens import TokenStore
from .runtime.podman import podman_userns_args
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
# overlaps any of these would shadow sandbox's mounts.
_MANAGED_VOLUME_TARGETS = frozenset(
    {
        CONTAINER_BRIDGES_DIR,
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
    [`shield.pre_start`][terok_sandbox.shield.pre_start]).
    """
    from . import shield as shield_module

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

    # Shield first — its OCI hook expects to see the annotations before
    # podman processes any of the other flags.
    if shield:
        args += shield_module.pre_start(container, state_dir, cfg)

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
            )
        )

    # Vault token broker — bind-mount the vault socket (socket mode) or
    # set the broker port env (TCP mode).  The in-container bridge script
    # picks up whichever is set and exposes a localhost endpoint.
    if effective_broker:
        if cfg.services_mode == "socket":
            args += _volume_args(
                VolumeSpec(
                    cfg.vault_socket_path,
                    f"{CONTAINER_RUNTIME_DIR}/vault.sock",
                    sharing=Sharing.SHARED,
                )
            )
            args += ["-e", f"TEROK_VAULT_LOOPBACK_PORT={LOOPBACK_VAULT_PORT}"]
        elif cfg.token_broker_port is not None:
            args += ["-e", f"TEROK_TOKEN_BROKER_PORT={cfg.token_broker_port}"]

    # Gate — mount the gate Unix socket and plant a per-container token.
    # The bridge script exposes the gate at http://localhost:9418/ inside
    # the container.  TCP-mode gate is not bridged here; the user reaches
    # it directly via host.containers.internal:<gate_port>.
    if effective_gate:
        token = TokenStore(cfg).create(scope, container)  # type: ignore[arg-type]
        if cfg.services_mode == "socket":
            args += _volume_args(
                VolumeSpec(
                    cfg.gate_socket_path,
                    f"{CONTAINER_RUNTIME_DIR}/gate-server.sock",
                    sharing=Sharing.SHARED,
                )
            )
            args += ["-e", f"TEROK_GATE_SOCKET={CONTAINER_RUNTIME_DIR}/gate-server.sock"]
        elif cfg.gate_port is not None:
            args += ["-e", f"TEROK_GATE_PORT={cfg.gate_port}"]
        args += ["-e", f"TEROK_GATE_TOKEN={token}"]

    # SSH signer — mint a phantom token, mount the signer socket (socket
    # mode) or set the port env (TCP mode).  The in-container bridge
    # injects the token in the agent-protocol pre-handshake and exposes
    # a plain ``SSH_AUTH_SOCK=/tmp/ssh-agent.sock`` to user-side tools.
    if effective_ssh:
        db = CredentialDB(cfg.db_path)
        try:
            ssh_token = db.create_token(scope, container, scope, "ssh")  # type: ignore[arg-type]
        finally:
            db.close()
        args += ["-e", f"TEROK_SSH_SIGNER_TOKEN={ssh_token}"]
        if cfg.services_mode == "socket":
            args += _volume_args(
                VolumeSpec(
                    cfg.ssh_signer_socket_path,
                    f"{CONTAINER_RUNTIME_DIR}/ssh-agent.sock",
                    sharing=Sharing.SHARED,
                )
            )
            args += ["-e", f"TEROK_SSH_SIGNER_SOCKET={CONTAINER_RUNTIME_DIR}/ssh-agent.sock"]
        elif cfg.ssh_signer_port is not None:
            args += ["-e", f"TEROK_SSH_SIGNER_PORT={cfg.ssh_signer_port}"]

    args += ["--name", container]

    _write_meta(state_dir, plan)
    return args, plan


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
        if target in _MANAGED_VOLUME_TARGETS:
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
    from . import shield as shield_module

    state_dir = run_state_dir(cfg, container)
    plan = _read_meta(state_dir)
    if plan is None:
        return False

    # Revoke tokens before tearing down shield: a still-running container
    # using a revoked token gets clean 401s, not stale 200s.
    if plan.gate and plan.scope is not None:
        TokenStore(cfg).revoke_for_task(plan.scope, container)
    if (plan.broker or plan.ssh) and plan.scope is not None:
        try:
            db = CredentialDB(cfg.db_path)
        except OSError:
            # Database file gone (e.g. vault uninstalled between prepare
            # and cleanup) — already revoked from the caller's point of view.
            db = None
        if db is not None:
            try:
                db.revoke_tokens(plan.scope, container)
            finally:
                db.close()

    # Shield down is best-effort: when the container has already exited,
    # the OCI poststop hook has already removed the rules.
    if plan.shield:
        try:
            shield_module.down(container, state_dir, cfg=cfg)
        except (SystemExit, OSError):
            pass

    shutil.rmtree(state_dir, ignore_errors=True)
    return True


# ---------------------------------------------------------------------------
# Output formatting
# ---------------------------------------------------------------------------


def format_args(args: list[str], *, output_json: bool) -> str:
    """Return the printable form of an args list."""
    if output_json:
        return json.dumps(args)
    return " ".join(shlex.quote(a) for a in args)
