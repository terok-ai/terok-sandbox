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

from .commands import COMMANDS, CommandTree


class _VersionAction(argparse.Action):
    """``--version`` that resolves the version string only when invoked.

    The stock ``action="version"`` wants the string at parser build
    time, which would put the ``importlib.metadata`` lookup (and the
    stdlib it drags in) back on **every** invocation — including the
    per-container supervisor and child spawns this CLI keeps slim.
    Deferring into ``__call__`` charges it only to the one run that
    actually asked.  The lookup is spelled out here rather than read
    off the package barrel because ``cli`` sits below the barrel in
    the tach layering (the barrel's lazy exports point back at
    ``commands``); [`terok_sandbox.__getattr__`][terok_sandbox.__getattr__]
    performs the same resolution for library consumers.
    """

    def __init__(self, option_strings: list[str], dest: str, **kwargs: object) -> None:
        """Fix ``nargs=0`` so the flag consumes no argument."""
        super().__init__(option_strings, dest, nargs=0, **kwargs)  # type: ignore[arg-type]

    def __call__(
        self,
        parser: argparse.ArgumentParser,
        namespace: argparse.Namespace,
        values: object,
        option_string: str | None = None,
    ) -> None:
        """Print ``terok-sandbox <version>`` to stdout and exit, mirroring ``action="version"``."""
        from importlib.metadata import PackageNotFoundError, version

        try:
            resolved = version("terok-sandbox")
        except PackageNotFoundError:
            resolved = "0.0.0"  # running from source without installed metadata
        print(f"{parser.prog} {resolved}")
        parser.exit()


def main(argv: list[str] | None = None) -> None:
    """Entry point for the ``terok-sandbox`` command."""
    import sys

    from terok_util import configure

    # One-time unified logging: routes every getLogger(__name__) to journald
    # (when present) or stderr — the non-supervisor CLI paths had no handler.
    configure(identifier="terok-sandbox")

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
    parser.add_argument(
        "--version", action=_VersionAction, help="show program's version number and exit"
    )
    # Lazy dispatch: passing ``argv`` wires only the invoked verb's module
    # in full (others stay name/help placeholders for the ``--help``
    # listing).  `terok-sandbox supervisor <id> <sidecar>` — spawned on
    # every container start/restart — therefore loads only the supervisor
    # module: no config, no SQLCipher, no terok-shield.
    COMMANDS.wire(parser, argv=argv)

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
