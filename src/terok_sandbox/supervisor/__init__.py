# SPDX-FileCopyrightText: 2026 Jiri Vyskocil
# SPDX-License-Identifier: Apache-2.0

"""Per-container supervisor — one process per container, lives for its lifetime.

Replaces the long-running ``terok-vault`` / clearance-hub /
verdict-server triple with a single in-process composition built per
container.  Spawned by the OCI prestart hook (see
[`terok_sandbox.resources.hooks.supervisor_hook`][terok_sandbox.resources.hooks.supervisor_hook])
through the restart-loop wrapper (see
[`terok_sandbox.resources.supervisor_wrapper`][terok_sandbox.resources.supervisor_wrapper]);
exits when ``podman wait`` returns.

The entry point is [`run_supervisor`][terok_sandbox.supervisor.main.run_supervisor] —
``terok-sandbox supervisor <container_id>`` invokes it under
``asyncio.run``.
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
