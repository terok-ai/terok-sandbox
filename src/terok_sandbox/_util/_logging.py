# SPDX-FileCopyrightText: 2026 Jiri Vyskocil
# SPDX-License-Identifier: Apache-2.0

"""File-backed best-effort logger — never raises, never disrupts callers.

Two surfaces share one implementation:

* The [`BestEffortLogger`][terok_sandbox._util._logging.BestEffortLogger]
  class binds a destination path on construction so any subsystem can
  spin up its own log file.  Public symbol — re-exported from
  ``terok_sandbox.__init__`` so out-of-tree consumers (notably terok
  itself) drop their own near-identical copy and pick this up instead.
* Module-level ``log_debug`` / ``log_warning`` / ``warn_user`` are
  the sandbox's own default-path shorthand, kept for legacy call
  sites — they delegate to a singleton bound at first call to the
  sandbox state log.
"""

from __future__ import annotations

import sys
import time
from collections.abc import Callable
from pathlib import Path


class BestEffortLogger:
    """Append timestamped lines to a state-file log; soft-fail on any error.

    The destination is supplied as a *callable* rather than an eager
    ``Path`` so XDG / env-var overrides applied between construction
    and write time still take effect — that mirrors the lazy-path
    behaviour of the original module-level helper this class replaces.

    Args:
        log_path_fn: Zero-arg callable returning the destination path.
            Called on every write so tests overriding ``HOME`` /
            ``XDG_STATE_HOME`` see their override applied even when the
            logger was constructed under the previous environment.
    """

    def __init__(self, log_path_fn: Callable[[], Path]) -> None:
        """Bind the destination resolver."""
        self._log_path_fn = log_path_fn

    def log(self, message: str, *, level: str = "DEBUG") -> None:
        """Append one ``[timestamp] LEVEL: message`` line.  Never raises."""
        try:
            log_path = self._log_path_fn()
            log_path.parent.mkdir(parents=True, exist_ok=True)
            timestamp = time.strftime("%Y-%m-%d %H:%M:%S", time.localtime())
            with log_path.open("a", encoding="utf-8") as f:
                f.write(f"[{timestamp}] {level}: {message}\n")
        except Exception:  # nosec B110 — intentionally silent
            pass

    def debug(self, message: str) -> None:
        """Append a DEBUG-level line."""
        self.log(message, level="DEBUG")

    def warning(self, message: str) -> None:
        """Append a WARNING-level line."""
        self.log(message, level="WARNING")

    def warn_user(self, component: str, message: str) -> None:
        """Print a structured warning to stderr and append it to the log file.

        Stderr output is run through ``sanitize_tty`` so attacker-bytes
        in *component* / *message* (e.g. originating from foreign config
        files) can't smuggle terminal escapes into the operator's
        terminal.  The file-side write is unsanitised so the log keeps
        the original bytes for forensic review.
        """
        from ._sanitize import sanitize_tty

        try:
            print(
                f"Warning [{sanitize_tty(component)}]: {sanitize_tty(message)}",
                file=sys.stderr,
            )
        except Exception:  # nosec B110 — intentionally silent
            pass
        self.warning(f"[{component}] {message}")


def _sandbox_log_path() -> Path:
    """Return the sandbox's own default log path under the user state dir."""
    from platformdirs import user_state_path

    return user_state_path("terok-sandbox", ensure_exists=True) / "terok-sandbox.log"


# Default singleton bound to the sandbox's own state path.  The
# module-level shorthand below preserves the legacy call shape for
# every in-tree call site.
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
