# SPDX-FileCopyrightText: 2026 Jiri Vyskocil
# SPDX-License-Identifier: Apache-2.0

"""High-level sandbox facade composing shield, gate, runtime, and SSH.

Convenience composition layer — delegates container lifecycle to the
injected :class:`ContainerRuntime`, plus convenience wrappers for gate
and shield services.  The launch path (:meth:`Sandbox.run`,
:meth:`Sandbox.create`) is still podman-specific and invokes the podman
CLI directly; Phase 3 will factor that through the runtime as well.
"""

from __future__ import annotations

import io
import shlex
import subprocess
import tarfile
from dataclasses import dataclass
from pathlib import Path, PurePosixPath
from typing import TYPE_CHECKING

from .config import SandboxConfig
from .runtime import ContainerRuntime, PodmanRuntime
from .runtime.podman import (
    bypass_network_args,
    check_gpu_error,
    gpu_run_args,
    podman_userns_args,
    redact_env_args,
)

if TYPE_CHECKING:
    from collections.abc import Callable

    from .credentials.ssh import SSHManager
    from .gate.lifecycle import GateServerStatus
    from .runtime import ContainerRemoveResult

# ---------------------------------------------------------------------------
# Container run specification
# ---------------------------------------------------------------------------

READY_MARKER = ">> init complete"
"""Default log line emitted by init-ssh-and-repo.sh when the container is ready."""


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
    """Sharing semantics: :attr:`Sharing.PRIVATE` or :attr:`Sharing.SHARED`."""

    def to_mount_arg(self) -> str:
        """Format as a ``-v`` flag value for ``podman run``."""
        try:
            relabel = _SHARING_TO_RELABEL[self.sharing]
        except KeyError:
            raise ValueError(f"Unknown sharing mode: {self.sharing!r}") from None
        return f"{self.host_path}:{self.container_path}:{relabel}"


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

    memory_limit: str | None = None
    """Podman ``--memory`` value (e.g. ``"4g"``, ``"512m"``).  ``None`` = unlimited."""

    cpu_limit: str | None = None
    """Podman ``--cpus`` value (e.g. ``"2.0"``, ``"0.5"``).  ``None`` = unlimited."""

    extra_args: tuple[str, ...] = ()
    """Additional podman run arguments (e.g. port publishing)."""

    unrestricted: bool = True
    """When False, adds ``--security-opt no-new-privileges``."""

    sealed: bool = False
    """When True, volumes are injected via ``podman cp`` instead of bind-mounted."""


# ---------------------------------------------------------------------------
# Facade
# ---------------------------------------------------------------------------


class Sandbox:
    """Per-task orchestrator composing runtime + services.

    Holds a :class:`ContainerRuntime` (defaulting to :class:`PodmanRuntime`)
    and a :class:`SandboxConfig`, and exposes gate / shield / lifecycle
    verbs bundled in one place.  Container lifecycle verbs delegate to the
    runtime; the launch path (:meth:`run`, :meth:`create`) still drives
    podman directly because shield / gate integration is podman-specific
    today.
    """

    def __init__(
        self,
        config: SandboxConfig | None = None,
        *,
        runtime: ContainerRuntime | None = None,
    ) -> None:
        self._cfg = config or SandboxConfig()
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

    def ensure_gate(self) -> None:
        """Verify the gate server is running; raise ``SystemExit`` if not."""
        from .gate.lifecycle import GateServerManager

        GateServerManager(self._cfg).ensure_reachable()

    def create_token(self, scope: str, task_id: str) -> str:
        """Create a task-scoped gate access token."""
        from .gate.tokens import TokenStore

        return TokenStore(self._cfg).create(scope, task_id)

    def gate_url(self, repo_path: Path, token: str) -> str:
        """Build an HTTP URL for gate access to *repo_path*."""
        port = self._cfg.gate_port
        base = self._cfg.gate_base_path
        rel = repo_path.relative_to(base).as_posix()
        return f"http://{token}@host.containers.internal:{port}/{rel}"

    def gate_status(self) -> GateServerStatus:
        """Return the current gate server status."""
        from .gate.lifecycle import GateServerManager

        return GateServerManager(self._cfg).get_status()

    # -- Shield -------------------------------------------------------------

    def pre_start_args(self, container: str, task_dir: Path) -> list[str]:
        """Return extra podman args for shield integration."""
        from .shield import pre_start

        return pre_start(container, task_dir, self._cfg)

    def shield_down(self, container: str, task_dir: Path) -> None:
        """Remove shield rules for a container (allow all egress)."""
        from .shield import down

        down(container, task_dir, cfg=self._cfg)

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
        (they are injected via :meth:`copy_to` between create and start).
        """
        cmd: list[str] = ["podman", verb] + (["-d"] if verb == "run" else [])
        cmd += podman_userns_args()

        if not spec.unrestricted:
            cmd += ["--security-opt", "no-new-privileges"]

        if self._cfg.shield_bypass:
            print("\n!! SHIELD BYPASSED — egress firewall DISABLED (shield_bypass is set) !!\n")
            cmd += bypass_network_args(self._cfg.gate_port)
        else:
            try:
                from .shield import pre_start

                cmd += pre_start(spec.container_name, spec.task_dir, self._cfg)
            except SystemExit:
                raise  # ShieldNeedsSetup; let the caller handle it
            except (OSError, FileNotFoundError) as exc:
                import warnings

                warnings.warn(
                    f"Shield setup failed ({exc}) — container will have unfiltered egress",
                    stacklevel=2,
                )

        cmd += gpu_run_args(enabled=spec.gpu_enabled)

        if spec.memory_limit is not None:
            cmd += ["--memory", spec.memory_limit]
        if spec.cpu_limit is not None:
            cmd += ["--cpus", spec.cpu_limit]

        if spec.extra_args:
            cmd += list(spec.extra_args)

        # Sealed containers receive their directories via podman cp, not mounts.
        if not spec.sealed:
            for vol in spec.volumes:
                cmd += ["-v", vol.to_mount_arg()]

        for k, v in spec.env.items():
            cmd += ["-e", f"{k}={v}"]

        cmd += ["--name", spec.container_name, "-w", "/workspace", spec.image]
        cmd += list(spec.command)
        return cmd

    def _exec_podman(self, cmd: list[str], *, input: bytes | None = None) -> None:
        """Run a podman command, translating failures to SystemExit."""
        kwargs: dict = {"check": True, "capture_output": True}
        if input is not None:
            kwargs["input"] = input
        try:
            subprocess.run(cmd, **kwargs)
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
        after a successful start.  Raises :class:`GpuConfigError` when the
        launch fails due to NVIDIA CDI misconfiguration.
        """
        if spec.sealed:
            self.create(spec, hooks=hooks)
            present = tuple(v for v in spec.volumes if v.host_path.exists())
            self._ensure_parents(spec.container_name, present)
            for vol in present:
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
        via :meth:`copy_to` before being started with :meth:`start`.
        """
        cmd = self._build_cmd(spec, verb="create")
        print("$", shlex.join(redact_env_args(cmd)))

        if hooks and hooks.pre_start:
            hooks.pre_start()

        self._exec_podman(cmd)
        return spec.container_name

    def start(self, container_name: str, *, hooks: LifecycleHooks | None = None) -> None:
        """Start a previously created container via the runtime.

        Fires *hooks.post_start* after a successful start.
        """
        self._runtime.container(container_name).start()
        if hooks and hooks.post_start:
            hooks.post_start()

    def _ensure_parents(self, container_name: str, volumes: tuple[VolumeSpec, ...]) -> None:
        """Create parent directories inside a stopped container.

        Bind mounts auto-create mount points; ``podman cp`` does not.
        Injects a tar archive containing directory entries for every
        ancestor of every volume target, so subsequent ``copy_to`` calls
        succeed regardless of the container image layout.
        """
        dirs: set[str] = set()
        for vol in volumes:
            target = PurePosixPath(vol.container_path)
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

    def stop(self, containers: list[str]) -> list[ContainerRemoveResult]:
        """Best-effort stop and remove *containers*.

        Returns one :class:`ContainerRemoveResult` per entry.
        """
        handles = [self._runtime.container(name) for name in containers]
        return self._runtime.force_remove(handles)

    # -- SSH ----------------------------------------------------------------

    def init_ssh(self, scope: str) -> SSHManager:
        """Create an SSH manager for *scope* that owns its own ``CredentialDB``.

        Callers receive an ``SSHManager`` whose DB connection is opened
        against :attr:`SandboxConfig.db_path`.  Use it as a context
        manager (``with sandbox.init_ssh(scope) as m: ...``) or call
        :meth:`SSHManager.close` when done.
        """
        from .credentials.ssh import SSHManager

        return SSHManager.open(scope=scope, db_path=self._cfg.db_path)
