# SPDX-FileCopyrightText: 2026 Jiri Vyskocil
# SPDX-License-Identifier: Apache-2.0

"""Filesystem confinement via Landlock — pin a process to the lane it needs.

A supervisor child that holds secret material (the vault's SQLCipher session
key, the signer's private keys) and shells out to third-party binaries (gate
→ ``git``, vault → ``systemd-creds``) wants its filesystem reach pinned: it
may read and execute the shared runtime, read and write its own data, and
touch nothing else — not a sibling service's secrets, not another container's
state, not the operator's home.  Then a bug in one of those binaries turned
into code execution can neither exfiltrate a secret it doesn't own nor drop a
payload outside its lane.

[`confine_filesystem`][terok_sandbox._util._landlock.confine_filesystem]
applies that to the current process and returns a
[`LandlockReport`][terok_sandbox._util._landlock.LandlockReport] of whether
the kernel took it.  It is the second self-applied floor (after
[`harden_self`][terok_util.hardening.harden_self]) every supervisor child
installs at start-up, and it needs ``no_new_privs`` from that floor already
in force.  Connecting to a unix socket is not a filesystem access Landlock
gates, so IPC to sockets outside the lane keeps working.
"""

from __future__ import annotations

import ctypes
import os
import struct
from dataclasses import dataclass
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from collections.abc import Iterable
    from pathlib import Path

#: Landlock syscall numbers.  They were allocated together on the
#: architecture-generic table, so x86-64 and arm64 — terok's targets —
#: share them; any arch where they differ simply fails the ABI probe and
#: degrades to a no-op.
_NR_CREATE_RULESET = 444
_NR_ADD_RULE = 445
_NR_RESTRICT_SELF = 446

#: ``landlock_create_ruleset`` flag that asks for the supported ABI version
#: instead of building a ruleset.
_CREATE_RULESET_VERSION = 1 << 0
#: ``enum landlock_rule_type`` — a rule over a path and everything beneath it.
_RULE_PATH_BENEATH = 1

#: Read-side filesystem access rights (Landlock ABI 1): open a file for
#: reading, list a directory, execute a file.  Granted on the read-exec lane.
_READ_ACCESS = (
    (1 << 0)  # EXECUTE
    | (1 << 2)  # READ_FILE
    | (1 << 3)  # READ_DIR
)

#: Write-side filesystem access rights (Landlock ABI 1): every way of
#: creating, changing, or removing a file.  Granted only on the read-write
#: lane, on top of the read rights.
_WRITE_ACCESS = (
    (1 << 1)  # WRITE_FILE
    | (1 << 4)  # REMOVE_DIR
    | (1 << 5)  # REMOVE_FILE
    | (1 << 6)  # MAKE_CHAR
    | (1 << 7)  # MAKE_DIR
    | (1 << 8)  # MAKE_REG
    | (1 << 9)  # MAKE_SOCK
    | (1 << 10)  # MAKE_FIFO
    | (1 << 11)  # MAKE_BLOCK
    | (1 << 12)  # MAKE_SYM
)

#: ``struct`` formats for the two Landlock attribute structs, native byte
#: order and packed (``=`` = standard sizes, no alignment padding), matching
#: the kernel's ``__attribute__((packed))`` uapi layout:
#: ``landlock_ruleset_attr { __u64 handled_access_fs; }`` and
#: ``landlock_path_beneath_attr { __u64 allowed_access; __s32 parent_fd; }``.
_RULESET_ATTR = "=Q"
_PATH_BENEATH_ATTR = "=Qi"


@dataclass(frozen=True)
class LandlockReport:
    """Whether [`confine_filesystem`][terok_sandbox._util._landlock.confine_filesystem] took hold.

    ``confined`` is ``True`` only when the kernel is now enforcing the
    restriction on this process.  ``reason`` explains a ``False`` — a kernel
    without Landlock (< 5.13) or a build lacking the syscalls degrades to a
    no-op, which the caller may log but must not treat as an error.
    """

    #: ``True`` when access outside the granted lanes is now denied by the kernel.
    confined: bool
    #: One-line explanation, ready for a diagnostic log line.
    reason: str


def confine_filesystem(read_exec: Iterable[Path], read_write: Iterable[Path]) -> LandlockReport:
    """Pin this process and its descendants to the given filesystem lane.

    After this, the process may read and execute only under *read_exec*, and
    additionally create/modify/remove only under *read_write* (each grant
    covers a directory and everything beneath it).  Every other path is
    denied even for reading.  Requires ``no_new_privs`` already set — the
    kernel gates unprivileged Landlock on it — so call
    [`harden_self`][terok_util.hardening.harden_self] first.

    Best-effort and irreversible: never raises; a kernel or build without
    Landlock returns ``confined=False`` and changes nothing.  A path that
    does not exist is skipped (there is nothing to reach until it is created,
    and a parent grant covers that creation).
    """
    libc = ctypes.CDLL(None, use_errno=True)
    if _landlock_abi(libc) < 1:
        return LandlockReport(False, "landlock unavailable (kernel < 5.13 or no syscall)")
    ruleset = _create_ruleset(libc, _READ_ACCESS | _WRITE_ACCESS)
    if ruleset < 0:
        return LandlockReport(False, f"create_ruleset failed (errno {ctypes.get_errno()})")
    try:
        for path in read_exec:
            _grant_beneath(libc, ruleset, path, _READ_ACCESS)
        for path in read_write:
            _grant_beneath(libc, ruleset, path, _READ_ACCESS | _WRITE_ACCESS)
        if libc.syscall(_NR_RESTRICT_SELF, ruleset, 0) != 0:
            return LandlockReport(False, f"restrict_self failed (errno {ctypes.get_errno()})")
    finally:
        os.close(ruleset)
    return LandlockReport(True, "filesystem confined")


def _landlock_abi(libc: ctypes.CDLL) -> int:
    """Return the kernel's Landlock ABI version, or a negative value when unsupported."""
    return libc.syscall(_NR_CREATE_RULESET, None, 0, _CREATE_RULESET_VERSION)


def _create_ruleset(libc: ctypes.CDLL, handled_access: int) -> int:
    """Create a ruleset governing *handled_access*; return its fd or a negative errno."""
    attr = struct.pack(_RULESET_ATTR, handled_access)
    return libc.syscall(_NR_CREATE_RULESET, attr, len(attr), 0)


def _grant_beneath(libc: ctypes.CDLL, ruleset: int, path: Path, access: int) -> None:
    """Grant *access* on *path* and everything beneath it (best-effort)."""
    try:
        parent_fd = os.open(path, os.O_PATH | os.O_CLOEXEC)
    except OSError:
        return
    try:
        attr = struct.pack(_PATH_BENEATH_ATTR, access, parent_fd)
        libc.syscall(_NR_ADD_RULE, ruleset, _RULE_PATH_BENEATH, attr, 0)
    finally:
        os.close(parent_fd)


__all__ = ["LandlockReport", "confine_filesystem"]
