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

This module is a thin subprocess shim — no Python binding exists for
systemd-creds and the CLI is the supported entry point.  Tests stub
``subprocess.run`` directly; production code stays trivial.
"""

from __future__ import annotations

import re
import shutil
import subprocess  # nosec: B404 — sealed credential lifecycle requires the systemd-creds CLI
from pathlib import Path

_BINARY = "systemd-creds"

#: Minimum systemd version with the non-root Varlink delegation path —
#: PR systemd/systemd#35536, released in v257 (2024-12-20).  Below this
#: the tier is silently unusable for a non-root caller, so we surface
#: that as "tier unavailable" rather than letting subprocess errors
#: leak through.
_MIN_SYSTEMD_VERSION = 257

_SEAL_TIMEOUT = 10.0
_UNSEAL_TIMEOUT = 10.0
_PROBE_TIMEOUT = 5.0


def _systemd_creds_version() -> int | None:
    """Return the major version of the installed ``systemd-creds``, or ``None``.

    Parses the first integer out of ``systemd-creds --version``.
    Returns ``None`` if the binary is absent, the call fails, or the
    output doesn't contain an integer — callers treat all three as
    "tier unavailable" and fall through.
    """
    if shutil.which(_BINARY) is None:
        return None
    try:
        result = subprocess.run(  # nosec: B603 — fixed argv, no user input
            [_BINARY, "--version"],
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


def is_available() -> bool:
    """Return ``True`` if ``systemd-creds`` is usable from this process.

    Requires the binary on ``PATH`` *and* a host systemd ≥ 257 so the
    non-root Varlink delegation path is present.  Older systemd is
    treated as "tier unavailable" — see the module docstring.
    """
    version = _systemd_creds_version()
    return version is not None and version >= _MIN_SYSTEMD_VERSION


def has_tpm2() -> bool:
    """Return ``True`` if the host has a TPM2 device usable by systemd-creds.

    Mirrors ``systemd-creds has-tpm2``'s exit code.  Used by ``vault
    seal --key=auto`` to choose between TPM2 and host-key sealing — a
    missing TPM doesn't break the tier (host-key fallback still works
    in ``--user`` mode), so this is a *preference* probe, not a
    precondition.
    """
    if not is_available():
        return False
    try:
        result = subprocess.run(  # nosec: B603 — fixed argv, no shell, no user input
            [_BINARY, "has-tpm2"],
            capture_output=True,
            timeout=_PROBE_TIMEOUT,
            check=False,
        )
    except (OSError, subprocess.TimeoutExpired):
        return False
    return result.returncode == 0


def seal(passphrase: str, credential_path: Path, *, tpm: bool = True) -> None:
    """Encrypt *passphrase* into *credential_path* (user-scoped).

    *tpm=True* seals against TPM2; *tpm=False* falls through to the
    host key.  Both go through ``--user`` so the credential is bound
    to the calling UID + username + machine-id; PID 1's Varlink
    interface handles the privileged half.

    The file is materialised at ``0o600`` after the seal so its
    contents stay inaccessible to other local users while still being
    machine-decryptable.

    Empty *passphrase* is rejected — sealing nothing produces a
    credential that decrypts to nothing, which the chain would treat
    as "tier empty" and silently skip.

    Raises ``RuntimeError`` if the binary fails; the caller surfaces
    actionable hints.
    """
    if not passphrase:
        raise ValueError("refusing to seal an empty passphrase")
    if not is_available():
        raise RuntimeError(
            f"{_BINARY} unavailable: needs systemd ≥ {_MIN_SYSTEMD_VERSION} for non-root --user mode"
        )
    credential_path.parent.mkdir(parents=True, exist_ok=True)
    key_arg = "--with-key=tpm2" if tpm else "--with-key=host"
    try:
        subprocess.run(  # nosec: B603 — fixed argv shape; credential_path is internally owned
            [_BINARY, "encrypt", "--user", key_arg, "-", str(credential_path)],
            input=passphrase,
            text=True,
            check=True,
            capture_output=True,
            timeout=_SEAL_TIMEOUT,
        )
    except FileNotFoundError as exc:
        raise RuntimeError(f"{_BINARY} binary not found on PATH") from exc
    except subprocess.CalledProcessError as exc:
        raise RuntimeError(
            f"{_BINARY} encrypt failed (exit {exc.returncode}): {(exc.stderr or '').strip()}"
        ) from exc
    credential_path.chmod(0o600)


def unseal(credential_path: Path) -> str | None:
    """Return the decrypted passphrase, or ``None`` if the credential isn't usable here.

    Always passes ``--user`` so the Varlink delegation handles the
    privileged decrypt (see module docstring).

    Returns ``None`` rather than raising so the resolver can fall
    through to the next tier on every failure mode: file missing,
    ``systemd-creds`` absent or too old, host can't decrypt (e.g.
    credential moved from another machine, TPM state changed), Varlink
    socket unreachable, timeout.

    Empty decrypt output is also collapsed to ``None`` — SQLCipher's
    no-encryption sentinel must never reach the connection.
    """
    if not credential_path.is_file():
        return None
    try:
        result = subprocess.run(  # nosec: B603 — fixed argv, credential_path internally owned
            [_BINARY, "decrypt", "--user", str(credential_path), "-"],
            capture_output=True,
            text=True,
            check=True,
            timeout=_UNSEAL_TIMEOUT,
        )
    except (OSError, subprocess.CalledProcessError, subprocess.TimeoutExpired):
        return None
    return result.stdout.rstrip("\n") or None


__all__ = ["has_tpm2", "is_available", "seal", "unseal"]
