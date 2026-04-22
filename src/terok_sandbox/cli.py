# SPDX-FileCopyrightText: 2025 Jiri Vyskocil
# SPDX-FileCopyrightText: 2026 Jiri Vyskocil
# SPDX-License-Identifier: Apache-2.0

"""CLI entry point for terok-sandbox.

Built from the command registry in :mod:`terok_sandbox.commands`.
No command logic lives here — just argument wiring and dispatch.
"""

from __future__ import annotations

import argparse
from importlib.metadata import PackageNotFoundError, version as _meta_version

from .commands import (
    DOCTOR_COMMANDS,
    GATE_COMMANDS,
    SETUP_COMMANDS,
    SHIELD_COMMANDS,
    SSH_COMMANDS,
    VAULT_COMMANDS,
    CommandDef,
)

try:
    __version__ = _meta_version("terok-sandbox")
except PackageNotFoundError:
    __version__ = "0.0.0"


def _wire_command(sub: argparse._SubParsersAction, cmd: CommandDef) -> None:
    """Add a :class:`CommandDef` to an argparse subparser group."""
    p = sub.add_parser(cmd.name, help=cmd.help)
    for arg in cmd.args:
        kwargs: dict = {}
        if arg.help:
            kwargs["help"] = arg.help
        if arg.type is not None:
            kwargs["type"] = arg.type
        if arg.default is not None:
            kwargs["default"] = arg.default
        if arg.action is not None:
            kwargs["action"] = arg.action
        if arg.dest is not None:
            kwargs["dest"] = arg.dest
        if arg.nargs is not None:
            kwargs["nargs"] = arg.nargs
        if arg.required and arg.name.startswith("-"):
            kwargs["required"] = True
        p.add_argument(arg.name, **kwargs)
    p.set_defaults(_cmd=cmd)


def _dispatch(args: argparse.Namespace) -> None:
    """Extract handler kwargs from parsed args and call the handler."""
    cmd: CommandDef = args._cmd
    if cmd.handler is None:
        raise SystemExit(f"Command '{cmd.name}' has no handler")
    kwargs = {
        arg.dest or arg.name.lstrip("-").replace("-", "_"): getattr(
            args, arg.dest or arg.name.lstrip("-").replace("-", "_"), arg.default
        )
        for arg in cmd.args
    }
    cmd.handler(**kwargs)


def main() -> None:
    """Entry point for the ``terok-sandbox`` command."""
    parser = argparse.ArgumentParser(
        prog="terok-sandbox",
        description="Hardened Podman container runtime with shield firewall and git gate",
    )
    parser.add_argument("--version", action="version", version=f"terok-sandbox {__version__}")
    sub = parser.add_subparsers()

    # -- top-level setup/uninstall aggregators --
    for cmd in SETUP_COMMANDS:
        _wire_command(sub, cmd)

    # -- shield --
    shield_p = sub.add_parser("shield", help="Egress firewall management")
    shield_sub = shield_p.add_subparsers()
    for cmd in SHIELD_COMMANDS:
        _wire_command(shield_sub, cmd)
    shield_p.set_defaults(_group_help=shield_p)

    # -- gate --
    gate_p = sub.add_parser("gate", help="Git gate server management")
    gate_sub = gate_p.add_subparsers()
    for cmd in GATE_COMMANDS:
        _wire_command(gate_sub, cmd)
    gate_p.set_defaults(_group_help=gate_p)

    # -- vault --
    vault_p = sub.add_parser("vault", help="Vault management")
    vault_sub = vault_p.add_subparsers()
    for cmd in VAULT_COMMANDS:
        _wire_command(vault_sub, cmd)
    vault_p.set_defaults(_group_help=vault_p)

    # -- ssh --
    ssh_p = sub.add_parser("ssh", help="SSH keypair management")
    ssh_sub = ssh_p.add_subparsers()
    for cmd in SSH_COMMANDS:
        _wire_command(ssh_sub, cmd)
    ssh_p.set_defaults(_group_help=ssh_p)

    # -- doctor --
    for cmd in DOCTOR_COMMANDS:
        _wire_command(sub, cmd)

    args = parser.parse_args()
    if hasattr(args, "_cmd"):
        _dispatch(args)
    elif hasattr(args, "_group_help"):
        args._group_help.print_help()
    else:
        parser.print_help()
        raise SystemExit(1)


if __name__ == "__main__":
    main()
