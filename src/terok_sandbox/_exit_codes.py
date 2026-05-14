# SPDX-FileCopyrightText: 2026 Jiri Vyskocil
# SPDX-License-Identifier: Apache-2.0

"""Shared exit codes for the sandbox CLI.

Foundation module — zero internal imports — so every layer (the
``_setup`` orchestrator that raises the codes, the package init that
re-exports them as public API, downstream CLIs that branch on them)
can depend on this without dragging the surface layer back into the
foundation.
"""

from __future__ import annotations

#: Manual host configuration is required to finish setup.
#:
#: Currently fires when all install phases succeed but the SELinux
#: ``terok_socket_t`` policy is still missing on a socket-mode host
#: — the sockets are bound but containers can't reach them, so the
#: install is functionally incomplete.  Distinct from ``1`` (a phase
#: failure) so scripts and the TUI can offer the specific fix.
EXIT_MANUAL_STEP_NEEDED = 5


__all__ = ["EXIT_MANUAL_STEP_NEEDED"]
