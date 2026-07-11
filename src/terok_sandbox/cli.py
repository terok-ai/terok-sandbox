# SPDX-FileCopyrightText: 2025 Jiri Vyskocil
# SPDX-License-Identifier: Apache-2.0

"""CLI entry point for terok-sandbox.

Built from the [`COMMANDS`][terok_sandbox.commands.COMMANDS] forest in
[`terok_sandbox.commands`][terok_sandbox.commands].  The tree itself
encodes the subparser nesting (``vault passphrase seal`` is a leaf
inside the ``passphrase`` group inside the ``vault`` group), so this
module is just a wrapper around
[`CommandTree.wire`][terok_util.cli_types.CommandTree.wire] and
[`CommandTree.dispatch`][terok_util.cli_types.CommandTree.dispatch].
"""

from __future__ import annotations

import argparse
from importlib.metadata import PackageNotFoundError, version as _meta_version

from .commands import COMMANDS, SUPERVISOR_COMMANDS, CommandTree

try:
    __version__ = _meta_version("terok-sandbox")
except PackageNotFoundError:
    __version__ = "0.0.0"


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
    # Fast-path the internal supervisor launcher.  `terok-sandbox supervisor
    # <id> <sidecar>` is spawned on every container start/restart and needs
    # none of the user-facing command tree — wiring the full forest would
    # materialise the shield subtree (importing terok_shield) for nothing.
    tree = CommandTree(SUPERVISOR_COMMANDS) if argv[:1] == ["supervisor"] else COMMANDS
    tree.wire(parser)

    args = parser.parse_args(argv)

    cmd_name = getattr(getattr(args, "_cmd", None), "name", None)
    if saw_separator and cmd_name != "run":
        parser.error("'--' separator is only supported by the 'run' subcommand")
    if cmd_name == "run":
        args.podman_args = run_trailing

    if hasattr(args, "_cmd"):
        CommandTree.dispatch(args)
    elif hasattr(args, "_group_help"):
        args._group_help.print_help()
    else:
        parser.print_help()
        raise SystemExit(1)


if __name__ == "__main__":
    main()
