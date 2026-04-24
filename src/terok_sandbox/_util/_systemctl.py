# SPDX-FileCopyrightText: 2026 Jiri Vyskocil
# SPDX-License-Identifier: Apache-2.0

"""Shared ``systemctl --user`` invocation helpers.

Two flavours, both targeting the user session bus:

* :func:`run` — the authoritative variant.  Raises :class:`SystemExit`
  with captured stderr on failure so setup phases that depend on the
  call succeeding (e.g. enabling a freshly rendered unit) surface the
  real ``Failed to connect to bus`` / ``Unit X not loaded`` line
  rather than the bare exit code ``subprocess.CalledProcessError``
  prints by default.

* :func:`run_best_effort` — the idempotent variant.  Swallows every
  error, including a missing ``systemctl`` binary and a
  :class:`subprocess.TimeoutExpired` on a wedged unit, so cleanup
  passes (stop, disable, sweep orphans) can't turn a non-failure
  into a raised exception.

Pick ``run`` when the call sits on the critical install path;
``run_best_effort`` when the call is cleanup-shaped.
"""

from __future__ import annotations

import shutil
import subprocess  # nosec B404 — systemctl is a trusted host binary

_TIMEOUT_SECONDS = 10


def run(verb: str, *args: str) -> None:
    """Run ``systemctl --user <verb> <args…>``; raise on failure with captured stderr.

    ``subprocess.run(check=True, capture_output=True)`` otherwise
    swallows stderr inside :class:`subprocess.CalledProcessError` —
    its ``str()`` only includes the exit status, so failures read as
    "command returned 1" with no hint of the underlying cause.
    """
    argv = ["systemctl", "--user", verb, *args]
    try:
        subprocess.run(argv, check=True, capture_output=True, timeout=_TIMEOUT_SECONDS)  # nosec B603
    except subprocess.CalledProcessError as exc:
        stderr = (exc.stderr or b"").decode("utf-8", errors="replace").strip()
        raise SystemExit(
            f"{' '.join(argv)} failed (exit {exc.returncode}){': ' + stderr if stderr else ''}"
        ) from exc


def run_best_effort(verb: str, *args: str) -> None:
    """Run ``systemctl --user <verb> <args…>``, swallowing every error path.

    Returns silently when ``systemctl`` isn't on PATH (containerised
    hosts, CI), when the unit doesn't exist, or when the call times
    out against a wedged unit.  Suitable for stop / disable / reload
    passes where the absence of state is the expected shape.
    """
    if not shutil.which("systemctl"):
        return
    argv = ["systemctl", "--user", verb, *args]
    try:
        subprocess.run(argv, check=False, capture_output=True, timeout=_TIMEOUT_SECONDS)  # nosec B603
    except subprocess.TimeoutExpired:
        pass
