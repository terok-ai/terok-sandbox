# SPDX-FileCopyrightText: 2026 Jiri Vyskocil
# SPDX-License-Identifier: Apache-2.0

"""Re-export shim for the CLI command-tree vocabulary.

Every per-subsystem command module imports
[`ArgDef`][terok_util.cli_types.ArgDef] /
[`CommandDef`][terok_util.cli_types.CommandDef] /
[`CommandTree`][terok_util.cli_types.CommandTree] /
[`KeyRow`][terok_util.cli_types.KeyRow] from this module's path; the
definitions themselves now live in
[`terok_util`][terok_util.cli_types].  This shim preserves the
``from ._types import …`` callsites without dragging the canonical
copies into sandbox's tree.
"""

from __future__ import annotations

from terok_util import ArgDef, CommandDef, CommandTree, KeyRow

__all__ = ["ArgDef", "CommandDef", "CommandTree", "KeyRow"]
