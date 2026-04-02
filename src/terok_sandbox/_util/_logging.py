# SPDX-FileCopyrightText: 2026 Jiri Vyskocil
# SPDX-License-Identifier: Apache-2.0

"""Logging utilities for terok-sandbox."""


def _log(message: str, *, level: str = "DEBUG") -> None:
    """Append a timestamped line to the terok-sandbox log.

    Best-effort, exception-safe: any IO error is silently ignored so this
    function never raises or affects callers.

    Writes to ``<state_root>/terok-sandbox.log``.
    """
    try:
        import time

        from platformdirs import user_state_path

        log_path = user_state_path("terok-sandbox", ensure_exists=True) / "terok-sandbox.log"
        timestamp = time.strftime("%Y-%m-%d %H:%M:%S", time.localtime())
        with open(log_path, "a", encoding="utf-8") as f:
            f.write(f"[{timestamp}] {level}: {message}\n")
    except Exception:
        pass


def log_debug(message: str) -> None:
    """Append a DEBUG line to the terok-sandbox log."""
    _log(message, level="DEBUG")


def log_warning(message: str) -> None:
    """Append a WARNING line to the terok-sandbox log."""
    _log(message, level="WARNING")


def warn_user(component: str, message: str) -> None:
    """Print a structured warning to stderr and log it."""
    import sys

    try:
        print(f"Warning [{component}]: {message}", file=sys.stderr)
    except Exception:
        pass
    log_warning(f"[{component}] {message}")
