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


def _configure_supervisor_logging() -> None:
    """Route supervisor / child logs through the unified facility, keeping stderr.

    The supervisor is a **host** process — the OCI hook spawns it detached,
    outside the container's namespaces — whose stderr the wrapper redirects
    to ``<state>/logs/<id>.log``.  ``stderr=True`` keeps that per-container
    log fed while also sending the ``terok-supervisor`` / ``terok_sandbox.*``
    / ``terok_clearance.*`` module loggers to the host journal when one is
    present.  On a non-systemd host it falls back to that same stderr file.
    """
    from terok_util import configure

    configure(identifier="terok-supervisor", stderr=True)


def _handle_supervisor(
    container_id: str, sidecar_path: str, container_pid: int | None = None
) -> int:
    """Bridge from the CLI parser to [`run_supervisor`][terok_sandbox.supervisor.main.run_supervisor].

    *container_pid* — the container's init host-PID, passed by the
    ``createRuntime`` hook via the wrapper — is the authoritative
    container-death signal (see
    [`run_supervisor`][terok_sandbox.supervisor.main.run_supervisor]).
    Optional so an older wrapper (positional-count 2) still dispatches.
    """
    from ..supervisor import run_supervisor

    _configure_supervisor_logging()
    return asyncio.run(run_supervisor(container_id, Path(sidecar_path), container_pid))


def _handle_supervise_child(service: str, container_id: str, sidecar_path: str) -> int:
    """Bridge from the CLI parser to [`run_child`][terok_sandbox.supervisor.children.run_child].

    One hardened service process — spawned by the parent supervisor via
    [`launch_child`][terok_sandbox.supervisor.launcher.launch_child], not
    by operators (though it is debuggable by hand the same way the
    ``supervisor`` verb is).
    """
    from ..supervisor.children import run_child

    _configure_supervisor_logging()
    return run_child(service, container_id, Path(sidecar_path))


SUPERVISOR: CommandDef = CommandDef(
    name="supervisor",
    help="Run the per-container supervisor (internal; spawned by the OCI hook)",
    handler=LazyHandler("terok_sandbox.commands.supervisor:_handle_supervisor"),
    group="internal",
    args=(
        ArgDef(name="container_id", help="Container ID the supervisor manages"),
        ArgDef(name="sidecar_path", help="Absolute path to the per-container sidecar JSON"),
        ArgDef(
            name="container_pid",
            help="Container init host-PID (from the createRuntime hook); enables the direct PID watch",
            type=int,
            nargs="?",
            default=None,
        ),
    ),
)

SUPERVISE_CHILD: CommandDef = CommandDef(
    name="supervise-child",
    help="Run one hardened supervisor service (internal; spawned by the supervisor)",
    handler=LazyHandler("terok_sandbox.commands.supervisor:_handle_supervise_child"),
    group="internal",
    args=(
        ArgDef(name="service", help="Service to run (verdict|clearance|gate|vault|signer)"),
        ArgDef(name="container_id", help="Container ID the service belongs to"),
        ArgDef(name="sidecar_path", help="Absolute path to the per-container sidecar JSON"),
    ),
)

#: The internal supervisor verbs.  ``SUPERVISOR`` and ``SUPERVISE_CHILD``
#: are the single source for each verb definition; ``commands.COMMANDS``
#: references them by their ``source`` strings for lazy dispatch.
SUPERVISOR_COMMANDS: tuple[CommandDef, ...] = (SUPERVISOR, SUPERVISE_CHILD)


__all__ = ["SUPERVISE_CHILD", "SUPERVISOR", "SUPERVISOR_COMMANDS"]
