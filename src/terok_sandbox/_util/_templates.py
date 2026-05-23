# SPDX-FileCopyrightText: 2025 Jiri Vyskocil
# SPDX-License-Identifier: Apache-2.0

"""Sandbox-specific template helpers — systemd argv escaping.

Unit-file rendering uses Jinja2 directly at each callsite (``gate``,
``vault.daemon``); only the ``ExecStart=`` argv-quoting helpers live
here.  They're sandbox-specific (systemd has a quoting dialect that
diverges from POSIX shell where ``shlex.join`` lives).
"""

from collections.abc import Iterable


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
