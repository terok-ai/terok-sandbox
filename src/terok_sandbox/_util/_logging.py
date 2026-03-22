# SPDX-FileCopyrightText: 2026 Jiri Vyskocil
# SPDX-License-Identifier: Apache-2.0

"""Debug logging utility for terok-sandbox."""


def log_debug(message: str) -> None:
    """Append a simple debug line to the terok-sandbox log.

    Best-effort, exception-safe: any IO error is silently ignored so this
    function never raises or affects callers.

    Writes timestamped lines to ``<state_root>/terok-sandbox.log``.
    """
    try:
        import time

        from platformdirs import user_state_path

        log_path = user_state_path("terok-sandbox", ensure_exists=True) / "terok-sandbox.log"
        timestamp = time.strftime("%Y-%m-%d %H:%M:%S", time.localtime())
        with open(log_path, "a", encoding="utf-8") as f:
            f.write(f"[{timestamp}] {message}\n")
    except Exception:
        pass
