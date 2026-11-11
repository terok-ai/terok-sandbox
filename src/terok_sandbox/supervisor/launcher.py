# SPDX-FileCopyrightText: 2026 Jiri Vyskocil
# SPDX-License-Identifier: Apache-2.0

"""How the parent supervisor spawns one child process per service.

Each service runs as its own ``python -m terok_sandbox supervise-child
<service> <container_id> <sidecar_path>`` process on the *parent's* own
interpreter — ``-m`` resolves through ``terok_sandbox.__main__``
whatever the install layout, so a child needs neither ``terok-sandbox``
on ``$PATH`` nor a rendered wrapper.

Isolation is the child's own job: the instant it starts it hardens
itself ([`harden_self`][terok_util.harden_self]) and labels its own
sockets for SELinux, all *before* it opens the credential store.  Nothing
about that depends on how the process was spawned, so the launch here is
a plain fork-exec. The optional Wayland wrapper enters its dedicated SELinux
domain before Python starts, so its later parent-death signals survive exec.
The parent's package search path survives wrapper-based
Python installations, but cwd imports remain excluded.
"""

from __future__ import annotations

import asyncio
import contextlib
import logging
import os
import sys
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

from .._util._subprocess_env import child_process_env

EXPECTED_PARENT_ENV = "TEROK_SUPERVISOR_PARENT_PID"
"""Internal startup guard against reparenting during the Wayland domain transition."""

if TYPE_CHECKING:
    from pathlib import Path


@dataclass(frozen=True)
class ChildHandle:
    """A launched child: its service name and the process running it."""

    service: str
    process: asyncio.subprocess.Process
    output: asyncio.Task[None] | None = None
    """Optional pipe pump; the supervisor owns its bounded drain at teardown."""

    @property
    def pid(self) -> int:
        """The child's process id."""
        return self.process.pid


async def launch_child(service: str, container_id: str, sidecar_path: Path) -> ChildHandle:
    """Fork+exec ``python -m terok_sandbox supervise-child <service> …``.

    The parent's one spawn primitive.  The child hardens itself and binds
    its own socket. The environment preserves the parent's package paths; the
    returned [`ChildHandle`][terok_sandbox.supervisor.launcher.ChildHandle]
    is what the supervisor waits on and, at shutdown, signals.
    """
    options: dict[str, Any] = {}
    env_overrides: dict[str, str] = {}
    if service == "wayland":
        from .._util._selinux import prepare_wayland_exec_context

        # A SELinux transition clears PDEATHSIG. Transition this trusted
        # wrapper first; it arms the signal in its final domain, and its
        # same-domain wl-mitm exec can then retain the descendant signal.
        options["preexec_fn"] = prepare_wayland_exec_context
        options["stdin"] = asyncio.subprocess.DEVNULL
        options["stdout"] = asyncio.subprocess.PIPE
        options["stderr"] = asyncio.subprocess.STDOUT
        env_overrides[EXPECTED_PARENT_ENV] = str(os.getpid())
    process = await asyncio.create_subprocess_exec(
        sys.executable,
        "-P",
        "-m",
        "terok_sandbox",
        "supervise-child",
        service,
        container_id,
        str(sidecar_path),
        env=child_process_env(env_overrides),
        **options,
    )
    output = asyncio.create_task(relay_output(process, service)) if service == "wayland" else None
    return ChildHandle(service=service, process=process, output=output)


async def relay_output(process: asyncio.subprocess.Process, service: str) -> None:
    """Forward bounded service output through the existing supervisor log."""
    assert process.stdout is not None  # nosec B101 - callers create stdout with PIPE
    logger = logging.getLogger(f"terok-supervisor.{service}")
    while chunk := await process.stdout.read(4096):
        logger.info("%s", chunk.decode(errors="replace").rstrip())


async def finish_output(output: asyncio.Task[None] | None) -> None:
    """Drain a stopped service without waiting forever on an inherited pipe."""
    if output is not None:
        with contextlib.suppress(TimeoutError, asyncio.CancelledError):
            await asyncio.wait_for(output, timeout=1.0)


__all__ = ["ChildHandle", "launch_child"]
