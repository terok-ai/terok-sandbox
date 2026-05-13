# SPDX-FileCopyrightText: 2026 Jiri Vyskocil
# SPDX-License-Identifier: Apache-2.0

"""Wrap ``systemd-creds(1)`` for the machine-bound passphrase tier.

[`systemd-creds`](https://www.freedesktop.org/software/systemd/man/systemd-creds.html)
seals a credential against the local machine's TPM2 (or, when no TPM
is reachable, the host key derived from ``/var/lib/systemd/random-seed``).
Decryption only works on the same machine — moving the encrypted blob
to another host yields ``Failed to decrypt: Operation not supported``.

The sandbox uses systemd-creds as the tier just below session-unlock
in the SQLCipher passphrase resolution chain: machine-bound, no OS
keyring required, survives reboots, no plaintext-on-disk.

This module is a thin subprocess shim — no Python binding exists for
systemd-creds and the CLI is the supported entry point.  Tests stub
``subprocess.run`` directly; production code stays trivial.
"""

from __future__ import annotations

import shutil
import subprocess  # nosec: B404 — sealed credential lifecycle requires the systemd-creds CLI
from pathlib import Path

_BINARY = "systemd-creds"

_SEAL_TIMEOUT = 10.0
_UNSEAL_TIMEOUT = 10.0
_PROBE_TIMEOUT = 5.0


def is_available() -> bool:
    """Return ``True`` if the ``systemd-creds`` CLI is reachable on ``PATH``."""
    return shutil.which(_BINARY) is not None


def has_tpm2() -> bool:
    """Return ``True`` if the host has a TPM2 device usable by systemd-creds.

    Mirrors ``systemd-creds has-tpm2``'s exit code.  Used by ``vault
    seal --key=auto`` to choose between TPM2 and host-key sealing — a
    missing TPM doesn't break the tier (host-key fallback still works),
    so this is a *preference* probe, not a precondition.
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
    """Encrypt *passphrase* into *credential_path*.

    *tpm=True* seals against TPM2; *tpm=False* falls through to the
    host key.  The file is materialised at ``0o600`` after the seal so
    its contents stay inaccessible to other local users while still
    being machine-decryptable.

    Empty *passphrase* is rejected — sealing nothing produces a
    credential that decrypts to nothing, which the chain would treat
    as "tier empty" and silently skip.

    Raises ``RuntimeError`` if the binary fails; the caller surfaces
    actionable hints.
    """
    if not passphrase:
        raise ValueError("refusing to seal an empty passphrase")
    credential_path.parent.mkdir(parents=True, exist_ok=True)
    key_arg = "--with-key=tpm2" if tpm else "--with-key=host"
    try:
        subprocess.run(  # nosec: B603 — fixed argv shape; credential_path is internally owned
            [_BINARY, "encrypt", key_arg, "-", str(credential_path)],
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

    Returns ``None`` rather than raising so the resolver can fall
    through to the next tier on every failure mode: file missing,
    ``systemd-creds`` absent, host can't decrypt (e.g. credential
    moved from another machine, TPM state changed), timeout.

    Empty decrypt output is also collapsed to ``None`` — SQLCipher's
    no-encryption sentinel must never reach the connection.
    """
    if not credential_path.is_file():
        return None
    try:
        result = subprocess.run(  # nosec: B603 — fixed argv, credential_path internally owned
            [_BINARY, "decrypt", str(credential_path), "-"],
            capture_output=True,
            text=True,
            check=True,
            timeout=_UNSEAL_TIMEOUT,
        )
    except (OSError, subprocess.CalledProcessError, subprocess.TimeoutExpired):
        return None
    return result.stdout.rstrip("\n") or None


__all__ = ["has_tpm2", "is_available", "seal", "unseal"]
