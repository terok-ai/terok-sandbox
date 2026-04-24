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

    Every known failure mode is normalised to :class:`SystemExit` with a
    human-readable message, so the setup aggregator's error row points
    the operator at the real cause rather than a raw Python traceback:

    * ``CalledProcessError`` — captured stderr is attached (default
      ``str()`` only includes the exit code).
    * ``TimeoutExpired`` — include the timeout value and any captured
      stdout/stderr so ``Failed to connect to bus`` surfaces through
      even a wedged call.
    * ``FileNotFoundError`` — name the missing binary rather than leak
      a ``[Errno 2] No such file or directory: 'systemctl'`` line.
    """
    argv = ["systemctl", "--user", verb, *args]
    try:
        subprocess.run(argv, check=True, capture_output=True, timeout=_TIMEOUT_SECONDS)  # nosec B603
    except subprocess.CalledProcessError as exc:
        stderr = (exc.stderr or b"").decode("utf-8", errors="replace").strip()
        raise SystemExit(
            f"{' '.join(argv)} failed (exit {exc.returncode}){': ' + stderr if stderr else ''}"
        ) from exc
    except subprocess.TimeoutExpired as exc:
        captured = _format_captured(exc.stdout, exc.stderr)
        raise SystemExit(f"{' '.join(argv)} timed out after {exc.timeout}s{captured}") from exc
    except FileNotFoundError as exc:
        raise SystemExit(f"{argv[0]}: command not found on PATH") from exc


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
    except (subprocess.TimeoutExpired, FileNotFoundError):
        # ``FileNotFoundError`` catches the TOCTOU window between the
        # ``which`` probe above and ``subprocess.run`` — the binary
        # could theoretically vanish under us on a live pipx upgrade.
        pass


def _format_captured(stdout: bytes | str | None, stderr: bytes | str | None) -> str:
    """Return a ``"; <details>"`` suffix from captured output, or ``""`` if empty."""
    parts = [_coerce(stream).strip() for stream in (stderr, stdout)]
    rendered = " ".join(p for p in parts if p)
    return f"; {rendered}" if rendered else ""


def _coerce(stream: bytes | str | None) -> str:
    """Decode bytes leniently, pass strings through, treat ``None`` as empty."""
    if stream is None:
        return ""
    if isinstance(stream, bytes):
        return stream.decode("utf-8", errors="replace")
    return stream
