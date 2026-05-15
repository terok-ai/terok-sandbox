# SPDX-FileCopyrightText: 2026 Jiri Vyskocil
# SPDX-License-Identifier: Apache-2.0

"""Shared ``systemctl --user`` invocation helpers.

Three flavours, all targeting the user session bus:

* [`run`][terok_sandbox._util._systemctl.run] — the authoritative variant.  Raises [`SystemExit`][SystemExit]
  with captured stderr on failure so setup phases that depend on the
  call succeeding (e.g. enabling a freshly rendered unit) surface the
  real ``Failed to connect to bus`` / ``Unit X not loaded`` line
  rather than the bare exit code ``subprocess.CalledProcessError``
  prints by default.

* [`run_best_effort`][terok_sandbox._util._systemctl.run_best_effort] — the idempotent variant.  Swallows every
  error, including a missing ``systemctl`` binary and a
  [`subprocess.TimeoutExpired`][subprocess.TimeoutExpired] on a wedged unit, so cleanup
  passes (stop, disable, sweep orphans) can't turn a non-failure
  into a raised exception.

* [`query`][terok_sandbox._util._systemctl.query] — the read-only variant.  Returns the captured
  [`subprocess.CompletedProcess`][subprocess.CompletedProcess] so callers can inspect ``returncode``
  and ``stdout`` (``is-active``, ``is-system-running``, …).  A
  missing ``systemctl`` binary or a timeout is normalised into a
  synthetic non-zero result so callers never need their own
  try/except for the cross-platform "systemd absent" path.

Pick ``run`` when the call sits on the critical install path;
``run_best_effort`` when the call is cleanup-shaped; ``query`` when
the caller cares about the output, not the success/failure outcome.
"""

from __future__ import annotations

import shutil
import subprocess  # nosec B404 — systemctl is a trusted host binary

_TIMEOUT_SECONDS = 10

# Resolve the ``systemctl`` binary once at module import so every later
# invocation runs the same absolute path.  A second resolution from
# ``subprocess.run("systemctl", …)`` would re-walk ``PATH`` on every
# call, exposing every consumer to a PATH-hijack race
# ([CWE-426](https://cwe.mitre.org/data/definitions/426.html)): if
# ``PATH`` contains an attacker-writable directory before the real
# ``/usr/bin``, the wrong binary runs under the user's identity.
# Holding the resolved path freezes the lookup at the moment the
# package is loaded and reused everywhere downstream.
_SYSTEMCTL_PATH: str | None = shutil.which("systemctl")


def run(verb: str, *args: str) -> None:
    """Run ``systemctl --user <verb> <args…>``; raise on failure with captured stderr.

    Every known failure mode is normalised to [`SystemExit`][SystemExit] with a
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
    if _SYSTEMCTL_PATH is None:
        raise SystemExit("systemctl: command not found on PATH")
    argv = [_SYSTEMCTL_PATH, "--user", verb, *args]
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
        # The resolved path vanished under us between module load and
        # call (e.g. a live pipx upgrade).  Preserve the existing exit
        # shape so callers don't grow a separate branch for this.
        raise SystemExit(f"{argv[0]}: command not found on PATH") from exc


def query(verb: str, *args: str, timeout: float = 5.0) -> subprocess.CompletedProcess[str]:
    """Run ``systemctl --user <verb> <args…>`` and return the captured result.

    Read-only variant for status probes (``is-active``,
    ``is-system-running``, …) where the caller inspects ``returncode``
    and ``stdout`` rather than treating any non-zero exit as fatal.

    A missing ``systemctl`` binary (containerised host, CI) and a
    [`subprocess.TimeoutExpired`][subprocess.TimeoutExpired] are normalised into a synthetic
    non-zero [`CompletedProcess`][subprocess.CompletedProcess] (returncodes ``127`` and
    ``124`` respectively, mirroring the shell convention) so callers
    never need their own try/except for the cross-platform "systemd
    absent" path.
    """
    if _SYSTEMCTL_PATH is None:
        return subprocess.CompletedProcess(args=[], returncode=127, stdout="", stderr="")
    argv = [_SYSTEMCTL_PATH, "--user", verb, *args]
    try:
        return subprocess.run(  # nosec B603 — argv is a fixed prefix + caller-controlled verb/args
            argv,
            capture_output=True,
            text=True,
            timeout=timeout,
            check=False,
        )
    except FileNotFoundError:
        return subprocess.CompletedProcess(argv, returncode=127, stdout="", stderr="")
    except subprocess.TimeoutExpired:
        return subprocess.CompletedProcess(argv, returncode=124, stdout="", stderr="")


def run_best_effort(verb: str, *args: str) -> None:
    """Run ``systemctl --user <verb> <args…>``, swallowing every error path.

    Returns silently when ``systemctl`` isn't on PATH (containerised
    hosts, CI), when the unit doesn't exist, or when the call times
    out against a wedged unit.  Suitable for stop / disable / reload
    passes where the absence of state is the expected shape.
    """
    if _SYSTEMCTL_PATH is None:
        return
    argv = [_SYSTEMCTL_PATH, "--user", verb, *args]
    try:
        subprocess.run(argv, check=False, capture_output=True, timeout=_TIMEOUT_SECONDS)  # nosec B603
    except (subprocess.TimeoutExpired, FileNotFoundError):
        # ``FileNotFoundError`` catches the TOCTOU window between the
        # module-load resolution above and this call — the binary
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
