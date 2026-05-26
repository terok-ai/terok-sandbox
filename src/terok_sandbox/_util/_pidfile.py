# SPDX-FileCopyrightText: 2026 Jiri Vyskocil
# SPDX-License-Identifier: Apache-2.0

"""Safe PID-file read and unlink helpers — closes the CWE-59 symlink race.

The naive ``Path.read_text`` / ``Path.unlink`` pattern follows symlinks,
which is the [CWE-59](https://cwe.mitre.org/data/definitions/59.html)
file-clobber vector: an attacker who can write the runtime dir
(normally a 0700 ``$XDG_RUNTIME_DIR`` owned by the same UID, but
overridable via ``TEROK_SANDBOX_RUNTIME_DIR`` or pathological setups
running with elevated privileges) can swap the pidfile for a symlink
to an unrelated file and cause the wrong PID to be read, or the
symlink's *target* to be deleted.

[`read_pidfile_safely`][terok_sandbox._util._pidfile.read_pidfile_safely]
opens with ``O_NOFOLLOW`` plus a regular-file ``fstat`` check;
[`unlink_pidfile_safely`][terok_sandbox._util._pidfile.unlink_pidfile_safely]
``lstat``-verifies before ``unlink`` and refuses symlinks.  Both
short-circuit silently on every failure path — this is best-effort
cleanup, not a critical path.

Originally landed inline in the vault daemon lifecycle (PR #308) and
moved here so other service-lifecycle paths can reuse the identical
pattern (issue #311).  Public names (no leading underscore) so
service-lifecycle modules in other subpackages can import them through
the standard ``from terok_sandbox._util import …`` surface.
"""

from __future__ import annotations

import os
import stat
from pathlib import Path

# Cap the read at a generous-but-bounded size so a hostile pidfile
# can't make us read megabytes before the int parse fails anyway.
_MAX_PID_BYTES = 64


def read_pidfile_safely(pidfile: Path) -> int | None:
    """Read a PID from *pidfile* without following symlinks.

    Returns the integer PID on success, or ``None`` on any failure path
    (missing file, symlink, non-regular file, invalid contents).
    """
    nofollow = getattr(os, "O_NOFOLLOW", 0)
    try:
        fd = os.open(pidfile, os.O_RDONLY | nofollow)
    except OSError:
        # ``FileNotFoundError`` is an ``OSError`` subclass — one clause
        # covers both ENOENT and ELOOP/EACCES/etc.
        return None
    try:
        st = os.fstat(fd)
        if not stat.S_ISREG(st.st_mode):
            return None
        raw = os.read(fd, _MAX_PID_BYTES).decode("utf-8", errors="replace").strip()
    finally:
        os.close(fd)
    try:
        return int(raw)
    except ValueError:
        return None


def unlink_pidfile_safely(pidfile: Path) -> None:
    """Unlink *pidfile* only if it's a regular file (refusing symlinks).

    The ``lstat`` check is what makes the difference: a plain
    ``Path.unlink()`` happily removes a symlink target, which is the
    file-clobber vector
    ([CWE-59](https://cwe.mitre.org/data/definitions/59.html)) the
    safe-handling guidance calls out.  Any failure is silently ignored
    — this is cleanup, not a critical path.
    """
    try:
        st = os.lstat(pidfile)
    except OSError:
        # ``FileNotFoundError`` is an ``OSError`` subclass — one clause
        # covers both the "already gone" common case and pathological
        # ``EACCES`` / parent-dir failures.
        return
    if not stat.S_ISREG(st.st_mode):
        return
    try:
        os.unlink(pidfile)
    except OSError:
        # Same subclass-relationship as above; one clause swallows the
        # full TOCTOU + EBUSY range.
        pass
