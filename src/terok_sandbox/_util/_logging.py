# SPDX-FileCopyrightText: 2026 Jiri Vyskocil
# SPDX-License-Identifier: Apache-2.0

"""Sandbox's default-path logging shorthand over the shared logger.

[`BestEffortLogger`][terok_util.logging.BestEffortLogger] provides the
implementation; this module binds it to the sandbox state log
(``_sandbox_log_path``) and exposes the module-level ``log_debug`` /
``log_warning`` / ``warn_user`` shorthand that in-tree call sites use.
"""

from __future__ import annotations

from pathlib import Path

from terok_util import BestEffortLogger


def _sandbox_log_path() -> Path:
    """Return the sandbox's own default log path under the user state dir."""
    from platformdirs import user_state_path

    return user_state_path("terok-sandbox", ensure_exists=True) / "terok-sandbox.log"


# Singleton bound to the sandbox state path; the module-level shorthand
# below delegates to it.
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
