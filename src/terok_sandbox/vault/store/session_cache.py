# SPDX-FileCopyrightText: 2026 Jiri Vyskocil
# SPDX-License-Identifier: Apache-2.0

"""The volatile unlock cache — one tier, two backings.

The cache tier of the passphrase chain holds the vault passphrase for
reuse between processes.  The cache has no timeout; ``vault lock`` clears it;
a reboot removes it.  Which backing holds it follows where this host
runs the supervisor, the reader that has to find it
([`supervisor_placement`][terok_sandbox._util._placement.supervisor_placement]):
a user-unit supervisor reads the operator's kernel keyring
([`kernel_keyring`][terok_sandbox.vault.store.kernel_keyring]), so that
is the backing; a supervisor inside the container runtime's namespace
sees an empty kernel keyring there, so the tier is a tmpfs session file
([`session_file`][terok_sandbox.vault.store.session_file]), a path
being a path in any namespace.  The file also stands in where the
kernel facility itself is unusable.  The status surfaces name the
backing and the reason, so a degradation is never silent.

Callers use this module, not a backing, for every cache operation.
``forget`` clears both backings: facility availability can change
between boots, and ``vault lock`` must not leave a live cache in the
backing this boot does not prefer.
"""

from __future__ import annotations

import os
from typing import TYPE_CHECKING

from ..._util._placement import SupervisorPlacement, supervisor_placement
from . import kernel_keyring as _kernel_keyring, session_file as _session_file

if TYPE_CHECKING:
    from types import ModuleType


def store(passphrase: str, db_path: str | os.PathLike[str]) -> bool:
    """Cache *passphrase* for *db_path* in the session's preferred backing."""
    return _backend().store(passphrase, db_path)


def load(db_path: str | os.PathLike[str]) -> str | None:
    """Return the cached passphrase for *db_path*, or ``None`` on any miss."""
    return _backend().load(db_path)


def forget(db_path: str | os.PathLike[str]) -> bool:
    """Clear the cache for *db_path* from both backings.

    Returns:
        True when no backing can still hold the cache, False when either
        removal failed and a live cache may remain.
    """
    kernel_cleared = _kernel_keyring.forget(db_path)
    file_cleared = _session_file.forget(db_path)
    return kernel_cleared and file_cleared


def is_cached(db_path: str | os.PathLike[str]) -> bool:
    """Return ``True`` when the preferred backing holds a cache for *db_path*."""
    return _backend().is_cached(db_path)


def unavailable_reason() -> str | None:
    """Explain why no backing can hold the cache here, or ``None`` when one can.

    A file backing that fails names why the kernel keyring was not the choice
    first, so the operator reads both facts in one line.
    """
    backend = _backend()
    reason = backend.unavailable_reason()
    if reason is None or backend is _kernel_keyring:
        return reason
    return f"{_file_reason()}; {reason}"


def backing_detail(*, cached: bool) -> str:
    """Human detail for the cache tier in the ``vault status`` chain.

    Separates the states an operator acts on differently: which backing
    serves this session and why, whether it holds a passphrase, and why
    the tier cannot run at all.
    """
    if (reason := unavailable_reason()) is not None:
        return f"unusable here: {reason}"
    if _backend() is _kernel_keyring:
        return (
            "cached in the kernel keyring (Linux kernel user keyring, @u)"
            if cached
            else "no passphrase cached"
        )
    where = f"tmpfs session file ({_file_reason()})"
    return f"cached in a {where}" if cached else f"no passphrase cached — {where}"


def _backend() -> ModuleType:
    """The backing the supervisor this host would start can read."""
    kernel_keyring_readable = (
        supervisor_placement() is SupervisorPlacement.USER_UNIT
        and _kernel_keyring.unavailable_reason() is None
    )
    return _kernel_keyring if kernel_keyring_readable else _session_file


def _file_reason() -> str:
    """Why the session file, not the kernel keyring, is this host's backing."""
    if supervisor_placement() is SupervisorPlacement.NAMESPACE_DAEMON:
        return "the supervisor runs in the container namespace, without a user manager"
    return f"kernel keyring unusable here: {_kernel_keyring.unavailable_reason()}"


__all__ = [
    "backing_detail",
    "forget",
    "is_cached",
    "load",
    "store",
    "unavailable_reason",
]
