# SPDX-FileCopyrightText: 2026 Jiri Vyskocil
# SPDX-License-Identifier: Apache-2.0

"""Hidden ``supervisor`` CLI verb — the long-running per-container process.

Invoked by the OCI hook chain (via
`terok_sandbox.resources.supervisor_wrapper`)
rather than directly by operators; the ``group`` tag pins it under
the internal section so it doesn't pollute the default ``--help``.

Operators **can** still call it for debugging — ``terok-sandbox
supervisor <container-id> <sidecar-path>`` reads the same sidecar
config the hook chain prepared and composes the same service bundle.
Stops cleanly on Ctrl-C.
"""

from __future__ import annotations

import asyncio
from pathlib import Path

from terok_util import LazyHandler

from ._types import ArgDef, CommandDef


def _handle_supervisor(container_id: str, sidecar_path: str) -> int:
    """Bridge from the CLI parser to [`run_supervisor`][terok_sandbox.supervisor.main.run_supervisor]."""
    import logging
    import sys

    from ..supervisor import run_supervisor

    # Supervisor stderr is redirected to the wrapper's per-container log
    # file; configure the root logger so module loggers (terok-supervisor,
    # terok_sandbox.*, terok_clearance.*) actually write there.
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
        stream=sys.stderr,
    )
    return asyncio.run(run_supervisor(container_id, Path(sidecar_path)))


SUPERVISOR_COMMANDS: tuple[CommandDef, ...] = (
    CommandDef(
        name="supervisor",
        help="Run the per-container supervisor (internal; spawned by the OCI hook)",
        handler=LazyHandler("terok_sandbox.commands.supervisor:_handle_supervisor"),
        group="internal",
        args=(
            ArgDef(name="container_id", help="Container ID the supervisor manages"),
            ArgDef(name="sidecar_path", help="Absolute path to the per-container sidecar JSON"),
        ),
    ),
)

#: Per-verb lazy-dispatch entry point resolved by ``commands.COMMANDS``
#: via its ``source`` string (see that module).  Co-located with the
#: registry tuple above so the verb definition stays the single source.
SUPERVISOR: CommandDef = SUPERVISOR_COMMANDS[0]


__all__ = ["SUPERVISOR", "SUPERVISOR_COMMANDS"]
