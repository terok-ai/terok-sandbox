# SPDX-FileCopyrightText: 2026 Jiri Vyskocil
# SPDX-License-Identifier: Apache-2.0

"""High-level sandbox facade composing shield, gate, runtime, and SSH.

Convenience composition layer — delegates to the existing module-level
functions.  Callers can also use those functions directly; this class
simply groups them behind a shared :class:`SandboxConfig`.
"""

from __future__ import annotations

import shlex
import subprocess
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING

from .config import SandboxConfig

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


@dataclass(frozen=True)
class RunSpec:
    """Everything needed for a single ``podman run`` invocation."""

    container_name: str
    """Unique container name."""

    image: str
    """Image tag to run (e.g. ``terok-l1-cli:ubuntu-24.04``)."""

    env: dict[str, str]
    """Environment variables injected into the container."""

    volumes: tuple[str, ...]
    """Volume mount strings (``host:container[:opts]``)."""

    command: tuple[str, ...]
    """Command to execute inside the container."""

    task_dir: Path
    """Host-side task directory (for shield state, logs, etc.)."""

    gpu_enabled: bool = False
    """Whether to pass GPU device args to podman."""

    extra_args: tuple[str, ...] = ()
    """Additional podman run arguments (e.g. port publishing)."""

    unrestricted: bool = True
    """When False, adds ``--security-opt no-new-privileges``."""


# ---------------------------------------------------------------------------
# Facade
# ---------------------------------------------------------------------------


class Sandbox:
    """Stateless facade composing sandbox primitives.

    All methods delegate to the module-level functions in this package,
    passing the stored :class:`SandboxConfig`.  The existing function-level
    API remains the canonical interface — this class is a convenience for
    callers that manage a config instance.
    """

    def __init__(self, config: SandboxConfig | None = None) -> None:
        self._cfg = config or SandboxConfig()

    @property
    def config(self) -> SandboxConfig:
        """Return the sandbox configuration."""
        return self._cfg

    # -- Gate ---------------------------------------------------------------

    def ensure_gate(self) -> None:
        """Verify the gate server is running; raise ``SystemExit`` if not."""
        from .gate.lifecycle import ensure_server_reachable

        ensure_server_reachable(self._cfg)

    def create_token(self, scope: str, task_id: str) -> str:
        """Create a task-scoped gate access token."""
        from .gate.tokens import create_token

        return create_token(scope, task_id, self._cfg)

    def gate_url(self, repo_path: Path, token: str) -> str:
        """Build an HTTP URL for gate access to *repo_path*."""
        port = self._cfg.gate_port
        base = self._cfg.gate_base_path
        rel = repo_path.relative_to(base).as_posix()
        return f"http://{token}@host.containers.internal:{port}/{rel}"

    def gate_status(self) -> GateServerStatus:
        """Return the current gate server status."""
        from .gate.lifecycle import get_server_status

        return get_server_status(self._cfg)

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

    def run(self, spec: RunSpec, *, hooks: LifecycleHooks | None = None) -> None:
        """Launch a detached container from *spec*.

        Assembles and executes the ``podman run`` command, handling user
        namespace mapping, shield or bypass networking, GPU device args,
        environment and volume injection, CDI error detection, and lifecycle
        hook callbacks.

        Fires *hooks.pre_start* before ``podman run`` and *hooks.post_start*
        after a successful launch.  Raises :class:`~.runtime.GpuConfigError`
        when the launch fails due to NVIDIA CDI misconfiguration.
        """
        from .runtime import (
            bypass_network_args,
            check_gpu_error,
            gpu_run_args,
            podman_userns_args,
            redact_env_args,
        )

        cmd: list[str] = ["podman", "run", "-d"]
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

        if spec.extra_args:
            cmd += list(spec.extra_args)
        for vol in spec.volumes:
            cmd += ["-v", vol]
        for k, v in spec.env.items():
            cmd += ["-e", f"{k}={v}"]

        cmd += ["--name", spec.container_name, "-w", "/workspace", spec.image]
        cmd += list(spec.command)

        print("$", shlex.join(redact_env_args(cmd)))

        if hooks and hooks.pre_start:
            hooks.pre_start()

        try:
            subprocess.run(cmd, check=True, capture_output=True)
        except FileNotFoundError:
            raise SystemExit("podman not found; please install podman")
        except subprocess.CalledProcessError as exc:
            check_gpu_error(exc)
            stderr = (exc.stderr or b"").decode(errors="replace")
            msg = f"Container launch failed:\n{stderr.strip()}" if stderr else str(exc)
            raise SystemExit(msg) from exc

        if hooks and hooks.post_start:
            hooks.post_start()

    # -- Runtime ------------------------------------------------------------

    def stream_logs(
        self,
        container: str,
        *,
        timeout: float | None = None,
        ready_check: Callable[[str], bool] | None = None,
    ) -> bool:
        """Stream container logs until *ready_check* matches or timeout."""
        from .runtime import stream_initial_logs

        check = ready_check or (lambda line: READY_MARKER in line)
        return stream_initial_logs(container, timeout, check)

    def wait_for_exit(self, container: str, timeout: float | None = None) -> int:
        """Block until container exits; return exit code."""
        from .runtime import wait_for_exit

        return wait_for_exit(container, timeout)

    def stop(self, containers: list[str]) -> list[ContainerRemoveResult]:
        """Best-effort stop and remove *containers* via ``podman rm -f``.

        Returns one :class:`~.runtime.ContainerRemoveResult` per entry in
        *containers*.  Inspect ``removed`` and ``error`` on each result to
        determine whether the container was successfully removed.
        """
        from .runtime import stop_task_containers

        return stop_task_containers(containers)

    # -- SSH ----------------------------------------------------------------

    def init_ssh(self, scope: str) -> SSHManager:
        """Create an SSH manager for *scope*."""
        from .credentials.ssh import SSHManager

        return SSHManager(scope=scope)
