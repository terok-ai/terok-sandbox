# SPDX-FileCopyrightText: 2026 Jiri Vyskocil
# SPDX-License-Identifier: Apache-2.0

"""Sandbox's default-path logging shorthand over the shared logger.

The [`BestEffortLogger`][terok_util.logging.BestEffortLogger] implementation
lives once in terok-util, at the bottom of the dependency chain.  This module
keeps only what is sandbox-specific: the default log path
([`_sandbox_log_path`][terok_sandbox._util._logging._sandbox_log_path]) and the
module-level ``log_debug`` / ``log_warning`` / ``warn_user`` shorthand that
in-tree call sites use — they delegate to a singleton bound to the sandbox
state log.
"""

from __future__ import annotations

from pathlib import Path

from terok_util import BestEffortLogger


def _sandbox_log_path() -> Path:
    """Return the sandbox's own default log path under the user state dir."""
    from platformdirs import user_state_path

    return user_state_path("terok-sandbox", ensure_exists=True) / "terok-sandbox.log"


# Default singleton bound to the sandbox's own state path.  The
# module-level shorthand below preserves the call shape every in-tree
# call site already uses.
_default_logger = BestEffortLogger(_sandbox_log_path)


def log_debug(message: str) -> None:
    """Append a DEBUG line to the sandbox's default log."""
    _default_logger.debug(message)


def log_warning(message: str) -> None:
    """Append a WARNING line to the sandbox's default log."""
    _default_logger.warning(message)


def warn_user(component: str, message: str) -> None:
    """Print a structured warning to stderr and log it to the sandbox's default log."""
    _default_logger.warn_user(component, message)
