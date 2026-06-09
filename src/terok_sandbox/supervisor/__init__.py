# SPDX-FileCopyrightText: 2026 Jiri Vyskocil
# SPDX-License-Identifier: Apache-2.0

"""Per-container supervisor — one process per container, lives for its lifetime.

Composes the ``terok-vault`` proxy, SSH signer, git gate server,
clearance hub, and verdict server into a single in-process composition
built per container.  Spawned by the OCI ``createRuntime`` hook (the
``terok_sandbox/resources/hooks/supervisor_hook.py`` script)
through the restart-loop wrapper
(``terok_sandbox/resources/supervisor_wrapper.py``);
exits when ``podman wait`` returns.

The entry point is [`run_supervisor`][terok_sandbox.supervisor.main.run_supervisor] —
``terok-sandbox supervisor <container_id> <sidecar_path>`` invokes it
under ``asyncio.run`` with both arguments.
"""

from terok_sandbox.supervisor.main import (
    SidecarConfig,
    SupervisorPaths,
    load_sidecar,
    run_supervisor,
)

__all__ = [
    "SidecarConfig",
    "SupervisorPaths",
    "load_sidecar",
    "run_supervisor",
]
