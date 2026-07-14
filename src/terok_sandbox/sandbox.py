# SPDX-FileCopyrightText: 2026 Jiri Vyskocil
# SPDX-License-Identifier: Apache-2.0

"""High-level sandbox facade composing shield, gate, runtime, and SSH.

Convenience composition layer — delegates container lifecycle to the
injected [`ContainerRuntime`][terok_sandbox.ContainerRuntime], plus convenience wrappers for gate
and shield services.  The launch path ([`Sandbox.run`][terok_sandbox.sandbox.Sandbox.run],
[`Sandbox.create`][terok_sandbox.sandbox.Sandbox.create]) is still podman-specific and invokes the podman
CLI directly; Phase 3 will factor that through the runtime as well.
"""

from __future__ import annotations

import io
import shlex
import subprocess  # nosec B404 — container exec for ready-marker probing — container exec for ready-marker probing
import tarfile
from dataclasses import dataclass, field
from pathlib import Path, PurePosixPath
from types import MappingProxyType
from typing import TYPE_CHECKING

from terok_util import podman_userns_args

from .config import SandboxConfig
from .runtime import ContainerRuntime, PodmanRuntime
from .runtime.podman import (
    bypass_network_args,
    check_gpu_error,
    gpu_run_args,
    redact_env_args,
)

if TYPE_CHECKING:
    from collections.abc import Callable, Mapping

    from .runtime import ContainerRemoveResult
    from .vault.ssh.manager import SSHManager

# ---------------------------------------------------------------------------
# Container run specification
# ---------------------------------------------------------------------------

READY_MARKER = ">> init complete"
"""Default log line emitted by init-ssh-and-repo.sh when the container is ready."""


SAFE_RUNTIMES: frozenset[str] = frozenset({"crun", "krun"})
"""OCI runtimes the sandbox will pass to ``podman --runtime``.

Allowlist enforced at command-assembly time.  Podman's ``--runtime``
accepts either a runtime name (``crun``, ``krun``) **or a path to a
binary** — passing a path would let a caller who controls
[`RunSpec.runtime`][terok_sandbox.sandbox.RunSpec] make podman execute
an arbitrary host binary as part of container creation.  By rejecting
anything outside this set (and anything that looks path-shaped) we
keep the runtime selection a known-isolation choice rather than an
arbitrary-code-execution surface.
"""


SAFE_ANNOTATION_KEYS: frozenset[str] = frozenset(
    {
        # Per-task dossier JSON path; shield hook reads it on every
        # event to populate ClearanceEvent.dossier.
        "dossier.meta_path",
        # krun microVM vCPU count.  Required under krun runtime — the
        # standard ``--cpus`` flag only sets the cgroup CFS quota; it
        # does NOT size the VM, so without this annotation the guest
        # sees host CPU affinity (typically all host cores).  Memory
        # has an OCI fallback in crun-krun so no analogous annotation
        # is needed for ``--memory``.  See ``man crun-krun(1)``.
        "krun.cpus",
        # Absolute path to the per-container supervisor sidecar JSON.
        # Triggers the supervisor OCI hook (matched by ``when.annotations``
        # in the hook descriptor) and tells the hook where to find the
        # sidecar — one anchor, no XDG guessing.
        "terok.sandbox.sidecar",
    }
)
"""OCI annotation keys allowed on
[`RunSpec.annotations`][terok_sandbox.sandbox.RunSpec].

Annotations are privileged config — they bind a running container to
host-side state the shield (or other readers) consult on every event.
The allowlist prevents a caller-controlled
[`RunSpec`][terok_sandbox.sandbox.RunSpec] from smuggling an
unrecognised key past the sandbox.
"""

_ANNOTATION_CTRL_CHARS = "\n\r\0"
"""Characters that would split or truncate the ``--annotation k=v`` arg."""


@dataclass(frozen=True)
class LifecycleHooks:
    """Optional callbacks fired at container lifecycle transitions.

    All slots are ``None`` by default.  ``Sandbox.run()`` fires ``pre_start``
    before ``podman run`` and ``post_start`` after a successful launch.
    ``post_ready`` and ``post_stop`` are available for callers to invoke at
    the appropriate time (e.g. after log streaming or container exit).
    """

    pre_start: Callable[[], None] | None = None
    """Fired before ``podman run``."""

    post_start: Callable[[], None] | None = None
    """Fired after a successful ``podman run``."""

    post_ready: Callable[[], None] | None = None
    """Fired when the container reports ready (caller responsibility)."""

    post_stop: Callable[[], None] | None = None
    """Fired after the container exits (caller responsibility)."""


#: In-container loopback port the gate socat bridge listens on (matches
#: ``TCP-LISTEN:9418`` in ``ensure-bridges.sh``).  The bridge forwards it to
#: the supervisor's per-container gate socket (socket mode) or host TCP port
#: (TCP mode), so the gate URL the container sees uses this fixed port in
#: both modes — never a host address.
_CONTAINER_GATE_PORT = 9418


class Sharing:
    """Directory sharing semantics — expresses intent, not backend details.

    The sandbox translates these into backend-specific flags (e.g. SELinux
    relabel ``:z`` / ``:Z`` for Podman) and uses them to drive sealed-mode
    decisions (private dirs are injected, shared dirs may be skipped).
    """

    PRIVATE = "private"
    """Exclusive to one container — no other container accesses this directory."""

    SHARED = "shared"
    """Shared across multiple containers (e.g. agent auth/config directories)."""


# SELinux relabel mapping — Podman-specific, kept in the sandbox layer.
_SHARING_TO_RELABEL: dict[str, str] = {
    Sharing.SHARED: "z",
    Sharing.PRIVATE: "Z",
}


@dataclass(frozen=True)
class VolumeSpec:
    """Typed description of a host↔container directory binding.

    Replaces raw volume strings (``"host:container:z"``) with structured data
    so the sandbox can decide *how* to materialise each binding — as a bind
    mount (shared mode) or a ``podman cp`` injection (sealed mode).

    *sharing* expresses the caller's intent (private vs shared); the sandbox
    translates that into backend-specific flags (e.g. SELinux relabeling for
    Podman).  In sealed mode, sharing semantics can also drive whether a
    directory is injected (private) or skipped (shared config that the
    vault replaces).
    """

    host_path: Path
    """Absolute host-side path to mount or copy in."""

    container_path: str
    """Absolute path inside the container (e.g. ``"/workspace"``)."""

    sharing: str = Sharing.SHARED
    """Sharing semantics: [`Sharing.PRIVATE`][terok_sandbox.sandbox.Sharing.PRIVATE] or [`Sharing.SHARED`][terok_sandbox.sandbox.Sharing.SHARED]."""

    read_only: bool = False
    """When True, mount the volume read-only inside the container.

    Used to layer immutable views on top of writable directory mounts —
    e.g. exposing a credential file to the agent while preventing it from
    overwriting the host-side phantom token.
    """

    live: bool = False
    """When True, this volume is bind-mounted even in sealed mode.

    Service plumbing (per-container vault/ssh-agent socket dir, gate
    socket, sourced-at-runtime bridge scripts) must be live: sealed-mode
    ``podman cp`` would snapshot an empty dir on the container side and
    the supervisor's later-bound sockets would never appear inside.
    Operator state (workspace, agent config) leaves this False so
    sealed mode gets fresh copies as designed.
    """

    def to_mount_arg(self) -> str:
        """Format as a ``-v`` flag value for ``podman run``."""
        try:
            relabel = _SHARING_TO_RELABEL[self.sharing]
        except KeyError:
            raise ValueError(f"Unknown sharing mode: {self.sharing!r}") from None
        opts = relabel + (",ro" if self.read_only else "")
        return f"{self.host_path}:{self.container_path}:{opts}"


@dataclass(frozen=True)
class RunSpec:
    """Everything needed for a single ``podman run`` invocation."""

    container_name: str
    """Unique container name."""

    image: str
    """Image tag to run (e.g. ``terok-l1-cli:ubuntu-24.04``)."""

    env: dict[str, str]
    """Environment variables injected into the container."""

    volumes: tuple[VolumeSpec, ...]
    """Host↔container directory bindings (mounted or injected per *sealed*)."""

    command: tuple[str, ...]
    """Command to execute inside the container."""

    task_dir: Path
    """Host-side task directory (for shield state, logs, etc.)."""

    gpu_enabled: bool = False
    """Whether to pass GPU device args to podman."""

    memory: str | None = None
    """Podman ``--memory`` value (e.g. ``"4g"``, ``"512m"``).  ``None`` = unlimited."""

    cpus: str | None = None
    """Podman ``--cpus`` value (e.g. ``"2.0"``, ``"0.5"``).  ``None`` = unlimited."""

    extra_args: tuple[str, ...] = ()
    """Additional podman run arguments (e.g. port publishing)."""

    unrestricted: bool = True
    """When False, adds ``--security-opt no-new-privileges``."""

    sealed: bool = False
    """When True, volumes are injected via ``podman cp`` instead of bind-mounted."""

    ephemeral: bool = False
    """When True, podman removes the container as soon as it exits (``--rm``).

    Retained — the default — keeps the exited container and its writable
    layer in podman storage for a later
    [`start`][terok_sandbox.sandbox.Sandbox.start] or
    [`rm`][terok_sandbox.sandbox.Sandbox.rm].  Ephemeral is the opt-in for
    disposable runs whose durable output (if any) leaves through a
    mounted volume.
    """

    hostname: str | None = None
    """Override the in-container hostname (podman ``--hostname``).

    When ``None`` (default), podman assigns the short container ID as the
    hostname.  Orchestrators may set this to a value that correlates with
    their own task/container identity — e.g. so a shell prompt inside the
    container matches the name the operator sees in task lists.  Must be a
    valid DNS hostname (letters/digits/hyphens, ≤253 chars); podman enforces
    the rule when parsing the flag.
    """

    runtime: str | None = None
    """OCI runtime to use (podman ``--runtime``).

    ``None`` (default) lets podman pick — its built-in default is
    ``crun``.  Set to ``"krun"`` to launch the task inside a KVM
    microVM (Phase 3 KrunRuntime).  Backend-neutral here; the runtime
    string is passed through verbatim and any compatibility decisions
    live higher up (e.g. orchestrator config validation).
    """

    annotations: Mapping[str, str] = field(default_factory=lambda: MappingProxyType({}))
    """OCI annotations forwarded as ``podman --annotation k=v`` entries.

    Keys must be on
    [`SAFE_ANNOTATION_KEYS`][terok_sandbox.sandbox.SAFE_ANNOTATION_KEYS].
    Declared as ``Mapping`` so callers can pass plain ``dict``s;
    ``__post_init__`` snapshots into a ``MappingProxyType`` so the
    frozen-dataclass guarantee holds against caller mutation.
    """

    loopback_ports: tuple[int, ...] = ()
    """Per-container host ports shield's nft rules must allow.

    Empty falls back to the cfg-resolved
    ``(gate_port, token_broker_port, ssh_signer_port)`` triple
    (legacy / single-daemon shape).  The per-container launch path
    passes ``(gate_port, per_container.token_broker_port,
    per_container.ssh_signer_port)`` so shield allows the actual
    ports the supervisor binds — without this override, shield
    blocks the per-container broker/signer with "No route to host".
    """

    def __post_init__(self) -> None:
        """Snapshot ``annotations`` so a caller-owned dict can't mutate the spec.

        Callers may legitimately pass a plain ``dict`` (Pydantic, JSON-load,
        tests) — we'd lose the frozen guarantee if we kept the live
        reference.  Take a copy, wrap it in a ``MappingProxyType``, and
        write it back through ``object.__setattr__`` since the dataclass
        itself is ``frozen=True``.
        """
        object.__setattr__(self, "annotations", MappingProxyType(dict(self.annotations)))


# ---------------------------------------------------------------------------
# Runtime / annotation validation (security boundary for podman argv)
# ---------------------------------------------------------------------------


def _validate_runtime(runtime: str) -> str:
    """Return *runtime* if it's a known-safe OCI runtime name.

    Rejects path-shaped inputs (anything containing ``/`` or ``\\``),
    whitespace-padded strings, and any name not on the
    [`SAFE_RUNTIMES`][terok_sandbox.sandbox.SAFE_RUNTIMES] allowlist.
    Refused values become [`ValueError`][ValueError] so a caller-controlled
    [`RunSpec.runtime`][terok_sandbox.sandbox.RunSpec] can never escalate
    into "podman, please run this arbitrary binary".
    """
    if not isinstance(runtime, str):
        raise ValueError(f"runtime must be a string, got {type(runtime).__name__}")
    if "/" in runtime or "\\" in runtime or runtime != runtime.strip():
        raise ValueError(f"runtime {runtime!r}: paths and whitespace-padded names are rejected")
    if runtime not in SAFE_RUNTIMES:
        raise ValueError(
            f"runtime {runtime!r}: not in allowlist {sorted(SAFE_RUNTIMES)} — "
            "extend SAFE_RUNTIMES to enable a new backend"
        )
    return runtime


def _validate_annotations(annotations: Mapping[str, str]) -> Mapping[str, str]:
    """Return *annotations* if every key is on the allowlist and every value safe.

    Rejects keys not on
    [`SAFE_ANNOTATION_KEYS`][terok_sandbox.sandbox.SAFE_ANNOTATION_KEYS]
    and values containing control characters (``\\n``, ``\\r``, ``\\0``)
    that would split the ``--annotation k=v`` argv element.
    """
    for key, value in annotations.items():
        if key not in SAFE_ANNOTATION_KEYS:
            raise ValueError(
                f"OCI annotation {key!r}: not in allowlist "
                f"{sorted(SAFE_ANNOTATION_KEYS)} — extend SAFE_ANNOTATION_KEYS "
                "to expose a new annotation key"
            )
        if not isinstance(value, str):
            raise ValueError(
                f"OCI annotation {key!r}: value must be a string, got {type(value).__name__}"
            )
        if any(c in value for c in _ANNOTATION_CTRL_CHARS):
            raise ValueError(
                f"OCI annotation {key!r}: value contains a control character "
                "that would split the --annotation flag"
            )
    return annotations


# ---------------------------------------------------------------------------
# Facade
# ---------------------------------------------------------------------------


class Sandbox:
    """Per-task orchestrator composing runtime + services.

    Holds a [`ContainerRuntime`][terok_sandbox.ContainerRuntime] (defaulting to [`PodmanRuntime`][terok_sandbox.PodmanRuntime])
    and a [`SandboxConfig`][terok_sandbox.SandboxConfig], and exposes gate / shield / lifecycle
    verbs bundled in one place.  Container lifecycle verbs delegate to the
    runtime; the launch path ([`run`][terok_sandbox.sandbox.Sandbox.run], [`create`][terok_sandbox.sandbox.Sandbox.create]) still drives
    podman directly because shield / gate integration is podman-specific
    today.
    """

    def __init__(
        self,
        config: SandboxConfig | None = None,
        *,
        runtime: ContainerRuntime | None = None,
    ) -> None:
        # ``Sandbox`` is the facade that launches containers + composes
        # gate/vault managers; resolve TCP ports here so the same
        # registry pass covers everyone downstream.  ``cfg`` itself
        # stays pure — only the cfg ``Sandbox`` carries is allocated.
        self._cfg = (config or SandboxConfig()).with_resolved_ports()
        self._runtime: ContainerRuntime = runtime or PodmanRuntime()

    @property
    def config(self) -> SandboxConfig:
        """Return the sandbox configuration."""
        return self._cfg

    @property
    def runtime(self) -> ContainerRuntime:
        """Return the injected container runtime."""
        return self._runtime

    # -- Gate ---------------------------------------------------------------

    def mint_gate_token(self) -> str:
        """Mint a fresh per-container gate token.

        The gate lives in each container's supervisor; the token
        travels to the container via the sidecar and is validated
        in-process, so there is nothing to persist.
        """
        from .gate.tokens import mint_gate_token

        return mint_gate_token()

    def gate_url(self, repo_path: Path, token: str) -> str:
        """Build the in-container HTTP URL for gate access to *repo_path*.

        Always uses the fixed loopback bridge port (see
        `_CONTAINER_GATE_PORT`): the container reaches the per-container
        gate through the socat bridge in both transport modes, so the URL
        carries no host address (``gate_port`` is ``None`` in socket mode).
        """
        rel = repo_path.relative_to(self._cfg.gate_base_path).as_posix()
        return f"http://{token}@localhost:{_CONTAINER_GATE_PORT}/{rel}"

    # -- Shield -------------------------------------------------------------

    def pre_start_args(
        self,
        container: str,
        task_dir: Path,
        *,
        runtime: str | None = None,
        loopback_ports: tuple[int, ...] = (),
    ) -> list[str]:
        """Return extra podman args for shield integration.

        *runtime* is the podman ``--runtime`` selector — passed to
        [`ShieldRuntime.from_runtime_name`][terok_shield.ShieldRuntime.from_runtime_name]
        so shield picks the right dnsmasq bind for the krun guest's
        isolated loopback.

        *loopback_ports* overrides shield's cfg-derived allowlist
        with per-container ports (see ``RunSpec.loopback_ports``).
        """
        from .integrations.shield import ShieldManager, ShieldRuntime

        return ShieldManager(
            task_dir,
            self._cfg,
            runtime=ShieldRuntime.from_runtime_name(runtime),
            loopback_ports_override=loopback_ports or None,
        ).pre_start(container)

    def shield_down(self, container: str, container_id: str, task_dir: Path) -> None:
        """Remove shield rules for a container (allow all egress).

        *container* is the operator-facing podman name (audit-log key);
        *container_id* is the full podman UUID — terok-shield's per-
        container hub socket is keyed on it.  Both are mandatory.
        """
        from .integrations.shield import ShieldManager

        ShieldManager(task_dir, self._cfg).down(container, container_id)

    # -- Container launch ---------------------------------------------------
    #
    # The launch path still drives podman directly because shield and
    # bypass-network integration produces podman-flavoured CLI args.  A
    # future krun backend will push this down into the runtime as well.

    def _build_cmd(self, spec: RunSpec, verb: str = "run") -> list[str]:
        """Assemble the ``podman`` command line for *spec*.

        *verb* selects the podman sub-command — ``"run"`` (detached) or
        ``"create"`` (created but not started).  Everything else (userns,
        shield, GPU, volumes, env, image, entrypoint) is identical.

        In **sealed** mode the volume specs are omitted from the command
        (they are injected via [`copy_to`][terok_sandbox.sandbox.Sandbox.copy_to] between create and start).
        """
        cmd: list[str] = ["podman", verb] + (["-d"] if verb == "run" else [])
        if spec.ephemeral:
            cmd.append("--rm")
        # A real init as pid1: the spec's command runs as a *child* of
        # podman's init binary, so SIGTERM actually terminates it.
        # Without this the command itself is namespace-init, the kernel
        # ignores default-disposition signals for init, and every
        # ``podman stop`` burns the full grace period before SIGKILL.
        cmd.append("--init")
        cmd += podman_userns_args()

        # ``--runtime`` must come before the image to be honoured; emit
        # it right after the verb to keep the rest of the assembly order
        # unchanged.  ``None`` (default) lets podman pick crun itself.
        if spec.runtime is not None:
            cmd += ["--runtime", _validate_runtime(spec.runtime)]

        # OCI annotations bind the container to host-side state the
        # shield reads on every event (currently the dossier path).
        for k, v in _validate_annotations(spec.annotations).items():
            cmd += ["--annotation", f"{k}={v}"]

        if not spec.unrestricted:
            cmd += ["--security-opt", "no-new-privileges"]

        if self._cfg.shield_bypass:
            print("\n!! SHIELD BYPASSED — egress firewall DISABLED (shield_bypass is set) !!\n")
            cmd += bypass_network_args(self._cfg.gate_port)
        else:
            try:
                cmd += self.pre_start_args(
                    spec.container_name,
                    spec.task_dir,
                    runtime=spec.runtime,
                    loopback_ports=spec.loopback_ports,
                )
            except SystemExit:
                raise  # ShieldNeedsSetup; let the caller handle it
            except (OSError, FileNotFoundError) as exc:
                # Refuse to launch with silent unfiltered egress: a shielded
                # spec asked for the firewall; soft-failing past it would
                # weaken the security posture the operator explicitly chose.
                # ``SandboxConfig(shield_bypass=True)`` is the documented
                # opt-out and skips this whole branch above.
                raise SystemExit(
                    f"Shield setup failed: {exc}\n"
                    f"Refusing to launch {spec.container_name} with unfiltered "
                    f"egress. Diagnose with `terok sickbay`, or set "
                    f"SandboxConfig(shield_bypass=True) if filtering is "
                    f"intentionally disabled for this run."
                ) from None

        cmd += gpu_run_args(enabled=spec.gpu_enabled)

        if spec.memory is not None:
            cmd += ["--memory", spec.memory]
        if spec.cpus is not None:
            cmd += ["--cpus", spec.cpus]

        if spec.extra_args:
            cmd += list(spec.extra_args)

        # Sealed: operator state gets copied; ``live`` service plumbing
        # (sockets, sourced bridge scripts) stays a bind mount so the
        # supervisor's later-bound sockets appear inside the container.
        for vol in spec.volumes:
            if spec.sealed and not vol.live:
                continue
            cmd += ["-v", vol.to_mount_arg()]

        for k, v in spec.env.items():
            cmd += ["-e", f"{k}={v}"]

        cmd += ["--name", spec.container_name]
        if spec.hostname is not None:
            cmd += ["--hostname", spec.hostname]
        cmd += ["-w", "/workspace", spec.image]
        cmd += list(spec.command)
        return cmd

    def _exec_podman(self, cmd: list[str], *, input: bytes | None = None) -> None:
        """Run a podman command, translating failures to SystemExit."""
        kwargs: dict = {"check": True, "capture_output": True}
        if input is not None:
            kwargs["input"] = input
        try:
            subprocess.run(cmd, **kwargs)  # nosec B603 — argv built from fixed verbs + caller-controlled scope/container names — argv built from fixed verbs + caller-controlled scope/container names
        except FileNotFoundError:
            raise SystemExit("podman not found; please install podman")
        except subprocess.CalledProcessError as exc:
            check_gpu_error(exc)
            stderr = (exc.stderr or b"").decode(errors="replace")
            msg = f"Container launch failed:\n{stderr.strip()}" if stderr else str(exc)
            raise SystemExit(msg) from exc

    def run(self, spec: RunSpec, *, hooks: LifecycleHooks | None = None) -> None:
        """Launch a detached container from *spec*.

        In **shared** mode (default), assembles and executes a single
        ``podman run -d`` with bind mounts.

        In **sealed** mode (``spec.sealed``), splits into create → inject →
        start: the container is created without volumes, directories are
        copied in via ``podman cp``, and the container is then started.

        Fires *hooks.pre_start* before creation and *hooks.post_start*
        after a successful start.  Raises [`GpuConfigError`][terok_sandbox.GpuConfigError] when the
        launch fails due to NVIDIA CDI misconfiguration.
        """
        if spec.sealed:
            self.create(spec, hooks=hooks)
            # ``live`` volumes are bind-mounted (handled by _build_cmd);
            # only the rest get copied in here.
            present = tuple(v for v in spec.volumes if not v.live and v.host_path.exists())
            # Drop overlay file mounts (a file landing inside a sibling
            # dir mount); the dir-copy already wrote them, and podman cp
            # refuses to overwrite.
            dir_targets = tuple(v.container_path for v in present if v.host_path.is_dir())

            def _under_dir_mount(path: str) -> bool:
                return any(path == d or path.startswith(d.rstrip("/") + "/") for d in dir_targets)

            effective = tuple(
                v for v in present if v.host_path.is_dir() or not _under_dir_mount(v.container_path)
            )
            self._ensure_parents(spec.container_name, effective)
            for vol in effective:
                self.copy_to(spec.container_name, vol.host_path, vol.container_path)
            self.start(spec.container_name, hooks=hooks)
            return

        cmd = self._build_cmd(spec, verb="run")
        print("$", shlex.join(redact_env_args(cmd)))

        if hooks and hooks.pre_start:
            hooks.pre_start()

        self._exec_podman(cmd)

        if hooks and hooks.post_start:
            hooks.post_start()

    def create(self, spec: RunSpec, *, hooks: LifecycleHooks | None = None) -> str:
        """Create a container without starting it.

        Returns the container name.  Fires *hooks.pre_start* before
        ``podman create``.  The container can then receive injected files
        via [`copy_to`][terok_sandbox.sandbox.Sandbox.copy_to] before being started with [`start`][terok_sandbox.sandbox.Sandbox.start].
        """
        cmd = self._build_cmd(spec, verb="create")
        print("$", shlex.join(redact_env_args(cmd)))

        if hooks and hooks.pre_start:
            hooks.pre_start()

        self._exec_podman(cmd)
        return spec.container_name

    def start(self, container_name: str, *, hooks: LifecycleHooks | None = None) -> None:
        """Start a previously created container, re-establishing its scaffolding.

        Ensures the per-container runtime directory exists first (see
        [`ensure_container_runtime_dir`][terok_sandbox.config.SandboxConfig.ensure_container_runtime_dir]):
        it is the bind-mount source for the in-container ``/run/terok``
        socket dir, it lives on the reboot-wiped ``$XDG_RUNTIME_DIR``
        tmpfs, and the supervisor removes it on every stop — so a restart
        (most visibly after a host reboot) finds it gone and ``podman
        start`` would fail on the missing mount source.  Idempotent, so
        the ``create`` → ``copy_to`` → ``start`` launch path (where the
        dir already exists) is unaffected.  This is the single primitive
        orchestrators call to (re)start a container: the host-side
        precondition is sandbox's concern, not theirs.

        Fires *hooks.post_start* after a successful start.
        """
        self._cfg.ensure_container_runtime_dir(container_name)
        self._runtime.container(container_name).start()
        if hooks and hooks.post_start:
            hooks.post_start()

    def _ensure_parents(self, container_name: str, volumes: tuple[VolumeSpec, ...]) -> None:
        """Create parent directories inside a stopped container.

        Bind mounts auto-create mount points; ``podman cp`` does not.
        Injects a tar archive containing directory entries for every
        ancestor of every volume target, so subsequent ``copy_to`` calls
        succeed regardless of the container image layout.

        File volumes (``host_path.is_file()``) only get their *parents*
        created — pre-creating the target as a directory would force
        ``copy_to`` to land the file inside it, not replace it.
        """
        dirs: set[str] = set()
        for vol in volumes:
            target = PurePosixPath(vol.container_path)
            if vol.host_path.is_dir():
                dirs.add(str(target).lstrip("/"))
            for ancestor in target.parents:
                if str(ancestor) != "/":
                    dirs.add(str(ancestor).lstrip("/"))

        if not dirs:
            return

        buf = io.BytesIO()
        with tarfile.open(fileobj=buf, mode="w") as tar:
            for d in sorted(dirs):
                info = tarfile.TarInfo(name=d)
                info.type = tarfile.DIRTYPE
                info.mode = 0o755
                info.uid = 1000
                info.gid = 1000
                tar.addfile(info)

        self._exec_podman(
            ["podman", "cp", "-", f"{container_name}:/"],
            input=buf.getvalue(),
        )

    def copy_to(self, container_name: str, src: Path, dest: str) -> None:
        """Copy a host path into a stopped container via the runtime."""
        self._runtime.container(container_name).copy_in(src, dest)

    # -- Runtime ------------------------------------------------------------

    def stream_logs(
        self,
        container: str,
        *,
        timeout: float | None = None,
        ready_check: Callable[[str], bool] | None = None,
    ) -> bool:
        """Stream container logs until *ready_check* matches or timeout."""
        check = ready_check or (lambda line: READY_MARKER in line)
        return self._runtime.container(container).stream_initial_logs(check, timeout)

    def wait_for_exit(self, container: str, timeout: float | None = None) -> int:
        """Block until *container* exits; return its exit code."""
        return self._runtime.container(container).wait(timeout)

    def stop(self, containers: list[str], *, timeout: int = 10) -> None:
        """Stop *containers*, keeping them for a later [`start`][terok_sandbox.sandbox.Sandbox.start].

        The retain half of podman's stop/rm verb pair: the container and
        its writable layer stay in podman storage, and the per-container
        state that must outlive a stop (sidecar file) is left in place.
        A missing container surfaces the runtime's error, exactly as
        ``podman stop`` would fail on it — callers expecting absence check
        container state first (or use [`rm`][terok_sandbox.sandbox.Sandbox.rm], which tolerates it).

        Args:
            containers: Container names to stop.
            timeout: Seconds the runtime waits before escalating to SIGKILL.

        Raises:
            RuntimeError: When the runtime cannot stop a container
                (including one that does not exist).
        """
        for name in containers:
            self._runtime.container(name).stop(timeout=timeout)

    def rm(self, containers: list[str]) -> list[ContainerRemoveResult]:
        """Force-remove *containers*, stopping them first when needed.

        The teardown half of the verb pair: the container and its
        writable layer are gone afterwards.  Host-side per-container
        state (sidecar file, runtime dir) is deliberately left in place —
        it belongs to the wiring layer above, which pairs removal with
        [`remove_container_state`][terok_sandbox.launch.remove_container_state]
        at real teardown.

        Returns:
            One [`ContainerRemoveResult`][terok_sandbox.runtime.ContainerRemoveResult] per entry.
        """
        handles = [self._runtime.container(name) for name in containers]
        return self._runtime.force_remove(handles)

    # -- Per-task state -----------------------------------------------------

    def task_state_dir(self, container: str) -> Path:
        """Per-container state directory used by the launch / cleanup verbs.

        The path is consumed by the
        [`launch`][terok_sandbox.launch] module: ``compose`` writes
        the plan + readiness markers under it, and
        [`launch.cleanup`][terok_sandbox.launch.cleanup] removes it on
        teardown.  The facade owns the *derivation* — ``state_dir /
        "sandbox" / "runs" / {container}`` — so the runs subtree
        layout has a single canonical owner.
        """
        return self._cfg.state_dir / "sandbox" / "runs" / container

    # -- SSH ----------------------------------------------------------------

    def init_ssh(self, scope: str) -> SSHManager:
        """Create an SSH manager for *scope* that owns its own ``CredentialDB``.

        Callers receive an ``SSHManager`` whose DB connection is opened
        against [`SandboxConfig.db_path`][terok_sandbox.SandboxConfig.db_path].  Use it as a context
        manager (``with sandbox.init_ssh(scope) as m: ...``) or call
        [`SSHManager.close`][terok_sandbox.SSHManager.close] when done.
        """
        from .vault.ssh.manager import SSHManager

        # Library code never prompts: a locked vault raises rather than
        # spinning up a prompt_toolkit prompt (which cannot own a running
        # event loop).  The frontend unlocks before calling in.
        return SSHManager.open_for_config(scope=scope, cfg=self._cfg, prompt_on_tty=False)
