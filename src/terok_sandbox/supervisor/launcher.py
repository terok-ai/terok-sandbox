# SPDX-FileCopyrightText: 2026 Jiri Vyskocil
# SPDX-License-Identifier: Apache-2.0

"""How the parent supervisor spawns one child process per service.

Each service runs as its own ``python -m terok_sandbox supervise-child
<service> <container_id> <sidecar_path>`` process on the *parent's* own
interpreter — ``-m`` resolves through [`terok_sandbox.__main__`][]
whatever the install layout, so a child needs neither ``terok-sandbox``
on ``$PATH`` nor a rendered wrapper.

Isolation is the child's own job: the instant it starts it hardens
itself ([`harden_self`][terok_util.harden_self]) and labels its own
sockets for SELinux, all *before* it opens the credential store.  Nothing
about that depends on how the process was spawned, so the launch here is
a plain fork-exec — identical on every host, with nothing for the parent
to configure.
"""

from __future__ import annotations

import asyncio
import sys
from dataclasses import dataclass
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from pathlib import Path

#: The argv that re-invokes this package on the running interpreter, so a
#: child is found by ``-m`` regardless of how the parent was installed
#: (editable, venv, system).
_SELF_ARGV: tuple[str, ...] = (sys.executable, "-m", "terok_sandbox")


@dataclass(frozen=True)
class ChildHandle:
    """A launched child: its service name and the process running it."""

    service: str
    process: asyncio.subprocess.Process

    @property
    def pid(self) -> int:
        """The child's process id."""
        return self.process.pid


async def launch_child(service: str, container_id: str, sidecar_path: Path) -> ChildHandle:
    """Fork+exec ``python -m terok_sandbox supervise-child <service> …``.

    The parent's one spawn primitive.  The child hardens itself and binds
    its own socket, so there is nothing to configure beyond the argv; the
    returned [`ChildHandle`][terok_sandbox.supervisor.launcher.ChildHandle]
    is what the supervisor waits on and, at shutdown, signals.
    """
    process = await asyncio.create_subprocess_exec(
        *_SELF_ARGV, "supervise-child", service, container_id, str(sidecar_path)
    )
    return ChildHandle(service=service, process=process)


__all__ = ["ChildHandle", "launch_child"]
