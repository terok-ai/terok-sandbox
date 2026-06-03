# SPDX-FileCopyrightText: 2026 Jiri Vyskocil
# SPDX-License-Identifier: Apache-2.0

"""Gate CLI verbs.

The gate lives inside each container's supervisor, so there is no host
daemon to install/start/stop.  The only verb is read-only:
``gate path <project>`` prints the ``file://`` URL of the project's bare
mirror so host tools (e.g. ``git`` invocations driven by the operator)
can address it directly.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from ._types import ArgDef, CommandDef

if TYPE_CHECKING:
    from ..config import SandboxConfig


def _handle_gate_path(*, project: str, cfg: SandboxConfig | None = None) -> None:
    """Print the ``file://`` URL of *project*'s bare mirror under the gate base path.

    *project* must be a bare project name: it is interpolated into a
    filesystem path, so path separators or parent-segment tokens
    (``/``, ``\\``, ``.``, ``..``) would let a caller escape
    ``cfg.gate_base_path`` and address a ``file://`` URL outside the
    managed mirror tree.  Reject those before composing the path.
    """
    from ..config import SandboxConfig

    cfg = cfg or SandboxConfig()
    if project in {".", ".."} or "/" in project or "\\" in project:
        raise SystemExit("project must be a bare project name without path separators")
    mirror = cfg.gate_base_path / f"{project}.git"
    print(mirror.as_uri())


#: The gate command group exposed at sandbox's top level.
GATE_COMMANDS: tuple[CommandDef, ...] = (
    CommandDef(
        name="gate",
        help="Git gate inspection",
        children=(
            CommandDef(
                name="path",
                help="Print the file:// URL of a project's bare mirror",
                handler=_handle_gate_path,
                args=(
                    ArgDef(
                        name="project",
                        help="Project name (the mirror is <project>.git)",
                    ),
                ),
            ),
        ),
    ),
)


__all__ = ["GATE_COMMANDS"]
