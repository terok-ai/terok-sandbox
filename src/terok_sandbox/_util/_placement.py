# SPDX-FileCopyrightText: 2026 Jiri Vyskocil
# SPDX-License-Identifier: Apache-2.0

"""Where this host runs a container's supervisor — and so, where its readers live.

One fact decides it, and the OCI hook reads the same fact through the
same predicate: a per-user systemd manager that answers.  With one, the
supervisor is a transient user unit in the operator's own namespaces,
and the kernel user keyring it reads is the operator's.  Without one, it
is a daemon inside the container runtime's user namespace, where that
kernel keyring is an empty stranger and only a path can carry the passphrase
across.  The cache tier follows the placement for exactly that reason
(see [`session_cache`][terok_sandbox.vault.store.session_cache]).
"""

from __future__ import annotations

import os
from enum import StrEnum

from ..resources.hooks import _supervisor_state


class SupervisorPlacement(StrEnum):
    """The two homes of a per-container supervisor."""

    USER_UNIT = "user unit"
    """A transient unit of the per-user systemd manager."""

    NAMESPACE_DAEMON = "namespace daemon"
    """A detached daemon inside the container runtime's user namespace."""


def supervisor_placement() -> SupervisorPlacement:
    """Where a supervisor started from this process would run."""
    runtime = _supervisor_state.user_runtime_dir(os.getuid())
    if _supervisor_state.user_manager_reachable(runtime):
        return SupervisorPlacement.USER_UNIT
    return SupervisorPlacement.NAMESPACE_DAEMON


__all__ = ["SupervisorPlacement", "supervisor_placement"]
