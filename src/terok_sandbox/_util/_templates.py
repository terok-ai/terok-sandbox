# SPDX-FileCopyrightText: 2025 Jiri Vyskocil
# SPDX-License-Identifier: Apache-2.0

"""Minimal template rendering via ``{{VAR}}`` token replacement."""

from collections.abc import Iterable
from pathlib import Path

_FORBIDDEN_CHARS = frozenset("\n\r\0")


def systemd_escape(arg: str) -> str:
    """Escape *arg* for a systemd unit-file value position (e.g. ``ExecStart=``).

    Three systemd-parser hazards a POSIX-style quoter (``shlex.join``)
    handles wrong:

    - ``%`` introduces specifier expansion — escaped as ``%%``.
    - Whitespace separates ``ExecStart=`` tokens.  Systemd treats both
      space and tab as separators (per ``systemd.syntax(7)``); escape
      both as ``\\x20`` / ``\\x09`` so a path containing either stays
      one argv element.

    Common Python paths (``/usr/bin/python3.12``) round-trip unchanged.
    Atypical paths (pipx-on-macOS framework prefixes, multi-Python
    rootfs layouts with ``%`` in directory names, custom build trees
    with tabs in names) survive instead of silently mis-parsing.
    """
    return arg.replace("%", "%%").replace(" ", "\\x20").replace("\t", "\\x09")


def systemd_exec_argv(prefix: Iterable[str]) -> str:
    """Join *prefix* into a single ``ExecStart=`` value, each element escaped.

    Use this instead of :func:`shlex.join` for unit-file ``ExecStart``,
    ``ExecStartPre``, etc. — POSIX shell quoting and systemd unit
    quoting diverge enough that ``shlex.join`` mis-renders paths that
    contain ``%`` or whitespace.
    """
    return " ".join(systemd_escape(arg) for arg in prefix)


def render_template(template_path: Path, variables: dict[str, str]) -> str:
    """Read *template_path* and replace ``{{KEY}}`` tokens with *variables* values.

    Raises [`ValueError`][ValueError] if any value contains control characters
    (newline, carriage-return, NUL) that could inject extra directives
    into the rendered systemd unit.
    """
    for key, val in variables.items():
        if _FORBIDDEN_CHARS & set(val):
            raise ValueError(f"Template variable {key!r} contains forbidden control characters")
    content = template_path.read_text()
    for k, v in variables.items():
        content = content.replace(f"{{{{{k}}}}}", v)
    return content
