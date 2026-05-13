# SPDX-FileCopyrightText: 2026 Jiri Vyskocil
# SPDX-License-Identifier: Apache-2.0

"""Wrap ``systemd-creds(1)`` for the machine-bound passphrase tier.

[`systemd-creds`](https://www.freedesktop.org/software/systemd/man/systemd-creds.html)
seals a credential against the local machine's TPM2 and/or the host
key under ``/var/lib/systemd/credential.secret``.  Decryption only
works on the same machine — moving the encrypted blob to another host
yields ``Failed to decrypt: Operation not supported``.

The sandbox uses systemd-creds as the tier just below session-unlock
in the SQLCipher passphrase resolution chain: machine-bound, no OS
keyring required, survives reboots, no plaintext-on-disk.

**Why this works for a non-root user.** Both ``encrypt`` and
``decrypt`` are always called with ``--user``.  In that mode a
regular shell invocation that detects ``geteuid() != 0`` transparently
delegates the privileged half (reading ``credential.secret``, talking
to the TPM) to PID 1 over the ``io.systemd.Credentials`` Varlink
interface; PID 1 returns the plaintext to the caller.  No setuid,
no ``tss`` group, no ``sudo``.

This requires systemd ≥ 257 (PR systemd/systemd#35536, merged
2024-12-20).  Earlier releases lack the Varlink endpoint and the
``--user`` decrypt path fails for non-root with ``Failed to determine
local credential key: Permission denied``.  [`is_available`][terok_sandbox.credentials.systemd_creds.is_available]
gates on the version so the tier reports as unavailable rather than
failing at runtime.

**Design choices that follow systemd-creds' intent.**

- ``--name=terok-sandbox.vault-passphrase`` is always set.  systemd
  embeds the name in the sealed blob to *prevent cross-purpose reuse*
  ("a credential sealed for X must not decrypt as if it were Y"); an
  explicit namespaced name is the documented production pattern.
- The default key mode (``KeyMode.AUTO`` → ``--with-key=auto``)
  delegates the host-vs-TPM choice to systemd, which yields
  ``host+tpm2`` on TPM-equipped systems (defense in depth) and falls
  back to host alone on TPM-less hosts.  We don't second-guess that
  decision — duplicating systemd's auto-detection in Python would
  drift over time and weaken the dual-factor default.
- No PCR policy.  Application-level credentials that need to survive
  kernel / UKI updates without operator intervention shouldn't bind
  to PCR values — Lennart's writing on PCR sealing targets disk
  encryption, not application secrets, where PCR brittleness costs
  too much (every kernel update would require re-sealing).  The
  attacker that can boot another kernel can also read the encrypted
  DB plaintext when the legitimate operator unlocks the vault, so the
  PCR policy doesn't move the needle for our threat model.

The module is a thin subprocess shim — no Python binding exists for
systemd-creds and the CLI is the supported entry point.  Tests stub
``subprocess.run`` directly; production code stays trivial.
"""

from __future__ import annotations

import contextlib
import functools
import os
import re
import shutil
import subprocess  # nosec: B404 — sealed credential lifecycle requires the systemd-creds CLI
import tempfile
from pathlib import Path
from typing import Literal

# ── Vocabulary ──────────────────────────────────────────────────────

_BINARY = "systemd-creds"

#: Namespaced credential name embedded in the encrypted blob.  systemd
#: refuses to decrypt a credential whose stored name doesn't match the
#: one supplied at decrypt time — the sandbox-prefixed value here
#: prevents the sealed file from being re-read by any other consumer
#: of systemd-creds on the same machine.
_CREDENTIAL_NAME = "terok-sandbox.vault-passphrase"

#: Minimum systemd version with the non-root Varlink delegation path —
#: PR systemd/systemd#35536, released in v257 (2024-12-20).  Below this
#: the tier is silently unusable for a non-root caller, so we surface
#: that as "tier unavailable" rather than letting subprocess errors
#: leak through.
_MIN_SYSTEMD_VERSION = 257

#: Subset of systemd-creds' ``--with-key=`` values we expose.  These map
#: 1:1 to the systemd flag so the wrapper doesn't reinvent the choice
#: (or invite the choice to drift): ``auto`` is "let systemd pick" —
#: host+tpm2 if a TPM is present, host alone otherwise.  ``host+tpm2``
#: pins the dual-factor combination explicitly; ``tpm2`` and ``host``
#: pin the single-factor flavours.  We don't expose ``tpm2-absent`` or
#: ``auto-initrd`` — those exist for boot-time / no-state environments
#: that don't match the sandbox use case.
KeyMode = Literal["auto", "host", "tpm2", "host+tpm2"]

_SEAL_TIMEOUT = 10.0
_UNSEAL_TIMEOUT = 10.0
_PROBE_TIMEOUT = 5.0


# ── Sealing ─────────────────────────────────────────────────────────


def seal(passphrase: str, credential_path: Path, *, key_mode: KeyMode = "auto") -> None:
    """Encrypt *passphrase* into *credential_path* under the namespaced
    credential name ``terok-sandbox.vault-passphrase``.

    *key_mode* maps 1:1 onto ``systemd-creds --with-key=…``:

    - ``"auto"`` (default) — systemd chooses ``host+tpm2`` on
      TPM-equipped hosts and ``host`` otherwise.  Defense in depth on
      hardware that supports it, graceful fallback on hardware that
      doesn't.
    - ``"host+tpm2"`` — pin the dual-factor combination explicitly.
    - ``"tpm2"`` — TPM-only.  Refuses to seal on a host without a TPM.
    - ``"host"`` — host-key only, no TPM dependency.

    All seals go through ``--user`` so the credential is bound to the
    calling UID + username + machine-id; PID 1's Varlink interface
    handles the privileged half.

    The encrypted blob is captured from systemd-creds' stdout and
    written atomically via ``tempfile.mkstemp`` + ``os.replace`` — the
    leaf is materialised at ``0o600`` from creation (no umask window)
    and the rename never follows a symlink at the destination.
    Symlinks at the parent or the leaf are refused outright.

    Empty *passphrase* is rejected — sealing nothing produces a
    credential that decrypts to nothing, which the chain would treat
    as "tier empty" and silently skip.

    Raises:
        ValueError: *passphrase* is empty.
        RuntimeError: the binary is missing, too old, times out, or
            its subprocess fails for any other reason; the caller
            surfaces actionable hints.
    """
    if not passphrase:
        raise ValueError("refusing to seal an empty passphrase")
    if not is_available():
        raise RuntimeError(
            f"{_BINARY} unavailable: needs systemd ≥ {_MIN_SYSTEMD_VERSION} for non-root --user mode"
        )
    exe = _require_exe()

    parent = credential_path.parent
    # Refuse symlinked parent or leaf — both would let an attacker
    # who can pre-create the path redirect the write to an arbitrary
    # location.  ``tempfile.mkstemp`` below also prevents racy
    # leaf-replacement, but the parent check has to happen before we
    # touch the filesystem.
    if parent.exists() and parent.is_symlink():
        raise RuntimeError(f"refusing to seal credential under symlinked parent: {parent}")
    if credential_path.is_symlink():
        raise RuntimeError(f"refusing to overwrite symlinked credential path: {credential_path}")
    parent.mkdir(parents=True, exist_ok=True, mode=0o700)

    try:
        result = subprocess.run(  # nosec: B603 — fixed argv shape, absolute path, captured output
            [
                exe,
                "encrypt",
                "--user",
                f"--name={_CREDENTIAL_NAME}",
                f"--with-key={key_mode}",
                "-",
                "-",
            ],
            input=passphrase.encode("utf-8"),
            check=True,
            capture_output=True,
            timeout=_SEAL_TIMEOUT,
        )
    except FileNotFoundError as exc:
        raise RuntimeError(f"{_BINARY} binary not found on PATH") from exc
    except subprocess.CalledProcessError as exc:
        stderr = (exc.stderr or b"").decode("utf-8", errors="replace").strip()
        raise RuntimeError(f"{_BINARY} encrypt failed (exit {exc.returncode}): {stderr}") from exc
    except subprocess.TimeoutExpired as exc:
        raise RuntimeError(f"{_BINARY} encrypt timed out after {_SEAL_TIMEOUT:.0f}s") from exc
    except OSError as exc:
        raise RuntimeError(f"{_BINARY} encrypt failed: {exc}") from exc

    # Write the sealed blob atomically via a same-dir tempfile.
    # ``mkstemp`` creates with mode 0600 by design — no umask window
    # between create and chmod — and ``os.replace`` swaps the entry
    # atomically without following any symlink the destination might
    # acquire mid-operation.
    sealed_blob = result.stdout
    try:
        fd, tmp_path = tempfile.mkstemp(prefix=credential_path.name + ".", dir=parent)
    except OSError as exc:
        raise RuntimeError(f"failed to stage sealed credential at {parent}: {exc}") from exc
    try:
        with os.fdopen(fd, "wb") as f:
            f.write(sealed_blob)
        os.replace(tmp_path, credential_path)
    except OSError as exc:
        with contextlib.suppress(OSError):
            os.unlink(tmp_path)
        raise RuntimeError(
            f"failed to materialise sealed credential at {credential_path}: {exc}"
        ) from exc


def unseal(credential_path: Path) -> str | None:
    """Return the decrypted passphrase, or ``None`` if the credential isn't usable here.

    Passes ``--user`` and the same ``--name=`` the credential was
    sealed with, so a sealed blob can only be unsealed for its
    intended purpose — systemd refuses cross-purpose decrypts even on
    the same host.

    Returns ``None`` rather than raising so the resolver can fall
    through to the next tier on every failure mode: file missing,
    ``systemd-creds`` absent or too old, host can't decrypt (e.g.
    credential moved from another machine, TPM state changed), Varlink
    socket unreachable, timeout, name mismatch.

    Empty decrypt output is also collapsed to ``None`` — SQLCipher's
    no-encryption sentinel must never reach the connection.
    """
    if not credential_path.is_file():
        return None
    exe = _systemd_creds_exe()
    if exe is None:
        return None
    try:
        result = subprocess.run(  # nosec: B603 — fixed argv, absolute path, internally-owned target
            [
                exe,
                "decrypt",
                "--user",
                f"--name={_CREDENTIAL_NAME}",
                str(credential_path),
                "-",
            ],
            capture_output=True,
            text=True,
            check=True,
            timeout=_UNSEAL_TIMEOUT,
        )
    except (OSError, subprocess.CalledProcessError, subprocess.TimeoutExpired):
        return None
    return result.stdout.rstrip("\n") or None


# ── Capability probes ───────────────────────────────────────────────


def is_available() -> bool:
    """Return ``True`` when ``systemd-creds`` is usable from a non-root caller.

    Requires the binary on ``PATH`` and a host systemd ≥
    ``_MIN_SYSTEMD_VERSION`` so the non-root Varlink delegation path
    is present.  An older systemd is reported as unavailable rather
    than left to fail at decrypt with the opaque ``Failed to determine
    local credential key`` error.
    """
    version = _systemd_creds_version()
    return version is not None and version >= _MIN_SYSTEMD_VERSION


def has_tpm2() -> bool:
    """Return ``True`` when the host has a TPM2 device usable by systemd-creds.

    Mirrors ``systemd-creds has-tpm2``'s exit code.  A *preference*
    probe, not a precondition: a missing TPM doesn't break the tier —
    host-key sealing still works in ``--user`` mode — so callers use
    this to choose between TPM2 and host-key, not to gate availability.
    """
    if not is_available():
        return False
    try:
        result = subprocess.run(  # nosec: B603 — absolute path, fixed argv, no user input
            [_require_exe(), "has-tpm2"],
            capture_output=True,
            timeout=_PROBE_TIMEOUT,
            check=False,
        )
    except (OSError, subprocess.TimeoutExpired):
        return False
    return result.returncode == 0


# ── Version detection ───────────────────────────────────────────────


@functools.cache
def _systemd_creds_version() -> int | None:
    """Return the major version of the installed ``systemd-creds``, or ``None``.

    Parses the first integer out of ``systemd-creds --version``.
    Returns ``None`` if the binary is absent, the call fails, or the
    output doesn't contain an integer — callers treat all three as
    "tier unavailable" and fall through.

    Cached for the process lifetime: the host's systemd version doesn't
    change between invocations, and a single ``vault seal`` already
    funnels through ``is_available`` → ``has_tpm2`` → ``seal``.  Tests
    clear the cache between cases via the
    ``_isolate_systemd_creds_version_cache`` autouse fixture in
    ``conftest.py``.
    """
    exe = _systemd_creds_exe()
    if exe is None:
        return None
    try:
        result = subprocess.run(  # nosec: B603 — absolute path, fixed argv, no user input
            [exe, "--version"],
            capture_output=True,
            text=True,
            check=True,
            timeout=_PROBE_TIMEOUT,
        )
    except (OSError, subprocess.CalledProcessError, subprocess.TimeoutExpired):
        return None
    match = re.search(r"\b(\d+)\b", result.stdout.splitlines()[0] if result.stdout else "")
    if match is None:
        return None
    return int(match.group(1))


@functools.cache
def _systemd_creds_exe() -> str | None:
    """Return the absolute path of ``systemd-creds``, or ``None`` if absent.

    Resolved once via ``shutil.which`` and reused for every subprocess
    call; this pins us to the binary that was on ``PATH`` at first
    look, defending against later ``PATH``-shuffling that would
    otherwise allow a same-UID attacker to substitute the binary
    between calls.

    Cached for the process lifetime — ``PATH`` resolution is stable
    while the process runs.  Tests clear the cache via the
    ``_isolate_systemd_creds_version_cache`` conftest fixture.
    """
    return shutil.which(_BINARY)


def _require_exe() -> str:
    """Return the resolved absolute path; raise if the binary is absent.

    Used at every code-path that must run the CLI (rather than just
    probe its presence): ``seal`` / ``unseal`` / ``has_tpm2``.
    """
    exe = _systemd_creds_exe()
    if exe is None:
        raise RuntimeError(f"{_BINARY} binary not found on PATH")
    return exe


__all__ = ["KeyMode", "has_tpm2", "is_available", "seal", "unseal"]
