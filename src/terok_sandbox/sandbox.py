# SPDX-FileCopyrightText: 2026 Jiri Vyskocil
# SPDX-License-Identifier: Apache-2.0

"""High-level sandbox facade composing shield, gate, runtime, and SSH.

Convenience composition layer — delegates to the existing module-level
functions.  Callers can also use those functions directly; this class
simply groups them behind a shared :class:`SandboxConfig`.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING

from .config import SandboxConfig

if TYPE_CHECKING:
    from collections.abc import Callable

    from .gate_server import GateServerStatus
    from .ssh import SSHManager

# ---------------------------------------------------------------------------
# Container run specification
# ---------------------------------------------------------------------------

READY_MARKER = ">> init complete"
"""Default log line emitted by init-ssh-and-repo.sh when the container is ready."""


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
        from .gate_server import ensure_server_reachable

        ensure_server_reachable(self._cfg)

    def create_token(self, project_id: str, task_id: str) -> str:
        """Create a task-scoped gate access token."""
        from .gate_tokens import create_token

        return create_token(task_id, project_id, self._cfg)

    def gate_url(self, repo_path: Path, token: str) -> str:
        """Build an HTTP URL for gate access to *repo_path*."""
        port = self._cfg.gate_port
        base = self._cfg.gate_base_path
        rel = repo_path.relative_to(base).as_posix()
        return f"http://{token}@host.containers.internal:{port}/{rel}"

    def gate_status(self) -> GateServerStatus:
        """Return the current gate server status."""
        from .gate_server import get_server_status

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

    def stop(self, containers: list[str]) -> None:
        """Best-effort stop and remove containers."""
        from .runtime import stop_task_containers

        stop_task_containers(containers)

    # -- SSH ----------------------------------------------------------------

    def init_ssh(self, label: str) -> SSHManager:
        """Create an SSH manager for *label* (e.g. a project ID)."""
        from .ssh import SSHManager

        return SSHManager(label, envs_dir=self._cfg.effective_envs_dir)
