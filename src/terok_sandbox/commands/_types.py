# SPDX-FileCopyrightText: 2026 Jiri Vyskocil
# SPDX-License-Identifier: Apache-2.0

"""Shared vocabulary for the command registry — argument and command definitions.

Every per-subsystem command module imports from here.  Out-of-tree
consumers (terok, terok-executor) build their CLI frontends against
these types without depending on any handler implementation.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, NamedTuple

if TYPE_CHECKING:
    from collections.abc import Callable


class KeyRow(NamedTuple):
    """One registered SSH key, fully resolved for display and matching."""

    scope: str
    comment: str
    key_type: str
    fingerprint: str
    private_key: str
    public_key: str


@dataclass(frozen=True)
class ArgDef:
    """Definition of a single CLI argument."""

    name: str
    help: str = ""
    type: Callable[[str], Any] | None = None
    default: Any = None
    action: str | None = None
    dest: str | None = None
    nargs: int | str | None = None
    required: bool = False


@dataclass(frozen=True)
class CommandDef:
    """Definition of a sandbox subcommand.

    Attributes:
        name: Subcommand name (e.g. ``"gate start"``).
        help: One-line help string.
        handler: Callable implementing the command.
        args: Argument definitions.
        group: Command group (e.g. ``"gate"``, ``"shield"``).
        epilog: Optional long-form text rendered after the argparse
            argument list in ``--help`` output.
    """

    name: str
    help: str = ""
    handler: Callable[..., None] | None = None
    args: tuple[ArgDef, ...] = ()
    group: str = ""
    epilog: str = ""


__all__ = ["ArgDef", "CommandDef", "KeyRow"]
