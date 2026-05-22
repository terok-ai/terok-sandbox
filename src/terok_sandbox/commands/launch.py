# SPDX-FileCopyrightText: 2026 Jiri Vyskocil
# SPDX-License-Identifier: Apache-2.0

"""Container-wiring CLI verbs — prepare, run, cleanup.

Compose (or exec into) the podman flags that wire a user-owned
container into sandbox services.  Mirrors terok-shield's prepare/run
shape and extends it with vault SSH signer, vault token broker, gate
token, and bridge-resource volume wiring.  Container lifecycle stays
with the user; sandbox owns only the services and per-container
ancillary state.

Thin wrappers around [`terok_sandbox.launch`][terok_sandbox.launch],
which holds the actual composition logic.
"""

from __future__ import annotations

from ..config import SandboxConfig
from ._types import ArgDef, CommandDef


def _csv_list(value: str) -> list[str]:
    """Split a comma-separated CLI value into a list, stripping whitespace.

    Used as ``ArgDef.type`` for multi-value optional flags so they don't
    rely on argparse's greedy ``nargs="+"`` (which silently slurps the
    following positional, turning ``--profiles a b mycontainer`` into
    ``profiles=["a","b","mycontainer"]``).  Comma-separated single-value
    matches podman's convention (``--cap-add=A,B,C``).
    """
    return [p.strip() for p in value.split(",") if p.strip()]


_BRIDGES_EPILOG = """\
Container-side contract:
  The image must have `socat` installed and source the bridge script in
  its startup.  Two equally supported delivery paths:

    * Build-time:  COPY the bridge scripts into the image (any path);
                   RUN apt install -y socat;  source ensure-bridges.sh
                   from your entrypoint.
    * Runtime:     image already has socat; sandbox bind-mounts the
                   bridges at /usr/local/share/terok-sandbox/bridges/;
                   source ensure-bridges.sh from your entrypoint.

Without socat the container is still sandboxed (shield/userns apply)
but the broker/gate/SSH bridges cannot connect.
"""


def _handle_prepare(
    container: str,
    *,
    no_shield: bool = False,
    no_gate: bool = False,
    no_broker: bool = False,
    scope: str | None = None,
    profiles: list[str] | None = None,
    output_json: bool = False,
    cfg: SandboxConfig | None = None,
) -> None:
    """Print podman flags for sandboxing *container*.

    Mints any tokens needed for the active subsystems (broker/gate/ssh)
    and persists per-container state so
    [`_handle_cleanup`][terok_sandbox.commands.launch._handle_cleanup]
    can reverse this invocation later.

    Args:
        container: Container name; becomes ``--name`` in the emitted args.
        no_shield: Disable the egress firewall (default: on).
        no_gate: Disable the git gate (default: on).
        no_broker: Disable the vault token broker (default: on).
        scope: Credential scope.  Required for gate/broker/ssh; omit for
            a shield-only run.
        profiles: Override shield profiles for this container.
        output_json: Emit a JSON array instead of a shell-quoted string.
        cfg: Optional [`SandboxConfig`][terok_sandbox.SandboxConfig] override.
    """
    from ..launch import compose, format_args

    if cfg is None:
        cfg = SandboxConfig()
    args, _plan = compose(
        container,
        cfg=cfg,
        shield=not no_shield,
        gate=not no_gate,
        broker=not no_broker,
        scope=scope,
        profiles=tuple(profiles) if profiles else None,
    )
    print(format_args(args, output_json=output_json))


def _handle_run(
    container: str,
    *,
    no_shield: bool = False,
    no_gate: bool = False,
    no_broker: bool = False,
    scope: str | None = None,
    profiles: list[str] | None = None,
    podman_args: list[str] | None = None,
    cfg: SandboxConfig | None = None,
) -> None:
    """Launch *container* by exec-ing into ``podman run``.

    Same composition as
    [`_handle_prepare`][terok_sandbox.commands.launch._handle_prepare]
    plus a collision check on the user-supplied trailing podman args and
    an ``os.execv`` into the podman binary.  Caller does not return.
    """
    from ..launch import compose, exec_podman

    if cfg is None:
        cfg = SandboxConfig()
    sandbox_args, _plan = compose(
        container,
        cfg=cfg,
        shield=not no_shield,
        gate=not no_gate,
        broker=not no_broker,
        scope=scope,
        profiles=tuple(profiles) if profiles else None,
    )
    exec_podman(sandbox_args, podman_args or [])


def _handle_cleanup(container: str, *, cfg: SandboxConfig | None = None) -> None:
    """Reverse a prior `prepare`/`run` for *container*.

    Revokes minted tokens, calls [`shield.down`][terok_sandbox.integrations.shield.down],
    and removes the per-container state directory.  Idempotent — exits
    quietly when no state is found.
    """
    from ..launch import cleanup

    if cfg is None:
        cfg = SandboxConfig()
    found = cleanup(container, cfg=cfg)
    if found:
        print(f"Cleaned up sandbox state for {container}.")
    else:
        print(f"No sandbox state found for {container}; nothing to clean up.")


LAUNCH_COMMANDS: tuple[CommandDef, ...] = (
    CommandDef(
        name="prepare",
        help="Print podman flags for sandboxing a user-owned container",
        handler=_handle_prepare,
        epilog=_BRIDGES_EPILOG,
        args=(
            ArgDef(name="container", help="Container name (becomes --name)"),
            ArgDef(
                name="--no-shield",
                action="store_true",
                help="Disable egress firewall (default: on)",
                dest="no_shield",
            ),
            ArgDef(
                name="--no-gate",
                action="store_true",
                help="Disable git gate (default: on; requires --scope)",
                dest="no_gate",
            ),
            ArgDef(
                name="--no-broker",
                action="store_true",
                help="Disable vault token broker (default: on; requires --scope)",
                dest="no_broker",
            ),
            ArgDef(
                name="--scope",
                help="Credential scope; enables vault SSH agent and is required by gate/broker",
            ),
            ArgDef(
                name="--profiles",
                type=_csv_list,
                help="Override shield profiles for this container (comma-separated, e.g. 'dev,pypi')",
            ),
            ArgDef(
                name="--json",
                action="store_true",
                dest="output_json",
                help="Output JSON array instead of a shell-quoted string",
            ),
        ),
    ),
    CommandDef(
        name="run",
        help="Launch a sandboxed user-owned container (exec into podman run)",
        handler=_handle_run,
        epilog=_BRIDGES_EPILOG,
        args=(
            ArgDef(name="container", help="Container name (becomes --name)"),
            ArgDef(
                name="--no-shield",
                action="store_true",
                help="Disable egress firewall (default: on)",
                dest="no_shield",
            ),
            ArgDef(
                name="--no-gate",
                action="store_true",
                help="Disable git gate (default: on; requires --scope)",
                dest="no_gate",
            ),
            ArgDef(
                name="--no-broker",
                action="store_true",
                help="Disable vault token broker (default: on; requires --scope)",
                dest="no_broker",
            ),
            ArgDef(
                name="--scope",
                help="Credential scope; enables vault SSH agent and is required by gate/broker",
            ),
            ArgDef(
                name="--profiles",
                type=_csv_list,
                help="Override shield profiles for this container (comma-separated, e.g. 'dev,pypi')",
            ),
        ),
    ),
    CommandDef(
        name="cleanup",
        help="Revoke tokens and drop shield rules for a sandboxed container",
        handler=_handle_cleanup,
        args=(ArgDef(name="container", help="Container name to clean up"),),
    ),
)


__all__ = ["LAUNCH_COMMANDS"]
