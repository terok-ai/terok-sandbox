# SPDX-FileCopyrightText: 2025 Jiri Vyskocil
# SPDX-FileCopyrightText: 2026 Jiri Vyskocil
# SPDX-License-Identifier: Apache-2.0

"""CLI entry point for terok-sandbox.

Built from the command registry in [`terok_sandbox.commands`][terok_sandbox.commands].
No command logic lives here — just argument wiring and dispatch.
"""

from __future__ import annotations

import argparse
from importlib.metadata import PackageNotFoundError, version as _meta_version

from .commands import (
    CREDENTIALS_COMMANDS,
    DOCTOR_COMMANDS,
    GATE_COMMANDS,
    LAUNCH_COMMANDS,
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
    """Add a [`CommandDef`][terok_sandbox.cli.CommandDef] to an argparse subparser group."""
    parser_kwargs: dict = {"help": cmd.help}
    if cmd.epilog:
        # ``RawDescriptionHelpFormatter`` preserves the newlines and
        # leading whitespace in the epilog so multi-line guidance
        # (e.g. delivery-pattern walkthroughs) renders verbatim.
        parser_kwargs["epilog"] = cmd.epilog
        parser_kwargs["formatter_class"] = argparse.RawDescriptionHelpFormatter
    p = sub.add_parser(cmd.name, **parser_kwargs)
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
    # Trailing args after ``--`` (currently only ``run`` consumes them).
    # ``main`` attaches ``args.podman_args`` after parsing if the split
    # was detected.  Passing it through unconditionally lets future
    # commands grow the same affordance without re-wiring dispatch.
    if hasattr(args, "podman_args"):
        kwargs["podman_args"] = args.podman_args
    cmd.handler(**kwargs)


def main(argv: list[str] | None = None) -> None:
    """Entry point for the ``terok-sandbox`` command."""
    import sys

    if argv is None:
        argv = sys.argv[1:]

    # The ``run`` subcommand uses ``--`` to separate sandbox args from
    # the trailing podman invocation.  Split before argparse to avoid
    # ``REMAINDER`` quirks with optional flags after the separator —
    # same pattern terok-shield uses.
    saw_separator = "--" in argv
    run_trailing: list[str] = []
    if saw_separator:
        sep = argv.index("--")
        run_trailing = argv[sep + 1 :]
        argv = argv[:sep]

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

    # -- credentials --
    cred_p = sub.add_parser("credentials", help="Credentials DB management")
    cred_sub = cred_p.add_subparsers()
    for cmd in CREDENTIALS_COMMANDS:
        _wire_command(cred_sub, cmd)
    cred_p.set_defaults(_group_help=cred_p)

    # -- prepare / run / cleanup (container wiring) --
    for cmd in LAUNCH_COMMANDS:
        _wire_command(sub, cmd)

    # -- doctor --
    for cmd in DOCTOR_COMMANDS:
        _wire_command(sub, cmd)

    args = parser.parse_args(argv)

    cmd_name = getattr(getattr(args, "_cmd", None), "name", None)
    if saw_separator and cmd_name != "run":
        parser.error("'--' separator is only supported by the 'run' subcommand")
    if cmd_name == "run":
        args.podman_args = run_trailing

    if hasattr(args, "_cmd"):
        _dispatch(args)
    elif hasattr(args, "_group_help"):
        args._group_help.print_help()
    else:
        parser.print_help()
        raise SystemExit(1)


if __name__ == "__main__":
    main()
