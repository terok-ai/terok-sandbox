# SPDX-FileCopyrightText: 2026 Jiri Vyskocil
# SPDX-License-Identifier: Apache-2.0

"""Per-container supervisor — one parent process, one child per service.

The parent ([`run_supervisor`][terok_sandbox.supervisor.main.run_supervisor])
launches the ``terok-vault`` proxy, SSH signer, git gate server,
clearance hub, and verdict server each in its own hardened child process
([`children`][terok_sandbox.supervisor.children], spawned via
[`launch_child`][terok_sandbox.supervisor.launcher.launch_child]), so a
bug in one service can't reach another's address space.  Spawned by
the OCI ``createRuntime`` hook (the
``terok_sandbox/resources/hooks/supervisor_hook.py`` script)
through the restart-loop wrapper
(``terok_sandbox/resources/supervisor_wrapper.py``);
exits when ``podman wait`` returns.

The entry point is [`run_supervisor`][terok_sandbox.supervisor.main.run_supervisor] —
``terok-sandbox supervisor <container_id> <sidecar_path>`` invokes it
under ``asyncio.run`` with both arguments.
"""

from terok_sandbox.supervisor.main import run_supervisor
from terok_sandbox.supervisor.sidecar import (
    SidecarConfig,
    SupervisorPaths,
    load_sidecar,
)

__all__ = [
    "SidecarConfig",
    "SupervisorPaths",
    "load_sidecar",
    "run_supervisor",
]
