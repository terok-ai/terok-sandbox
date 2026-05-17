# SPDX-FileCopyrightText: 2026 Jiri Vyskocil
# SPDX-License-Identifier: Apache-2.0

"""Track operator-confirmed possession of the vault recovery passphrase.

Whenever the sandbox mints a passphrase the operator never typed
(auto-detected systemd-creds tier, fresh keyring entry, or the
session-file fallback), they need a written copy stashed off-host —
every keystore tier is bound to *this* machine, account, or boot, so
a disk failure or TPM transplant strands the vault.

This module persists a small attestation that the operator has saved
the current passphrase, keyed by a stable fingerprint of the
passphrase itself, so a re-key naturally re-prompts.  The marker is a
sidecar file under ``vault_dir`` (mode ``0o600``); its contents are
SHA-256 of ``_SALT || passphrase``, hex-encoded.  Storing the raw
passphrase here would defeat its own purpose; storing nothing at all
would mean we couldn't distinguish "operator clicked I-have-saved-it"
from "operator opened the reveal modal and walked away".  The salt
is constant — the goal is "same passphrase → same digest", not
brute-force hardening (the keystore tier holds the cleartext anyway).
"""

from __future__ import annotations

import hashlib
from pathlib import Path

_SALT = b"terok-recovery-acknowledged-v1"


def fingerprint(passphrase: str) -> str:
    """Return a stable hex digest identifying *passphrase*.

    A re-key changes the digest, so an acknowledgement keyed by digest
    self-invalidates on rotation.  Not a security primitive — the
    keystore tier holds the cleartext and is the actual trust boundary.
    """
    return hashlib.sha256(_SALT + passphrase.encode("utf-8")).hexdigest()


def acknowledged_fingerprint(marker_path: Path) -> str | None:
    """Return the recorded fingerprint, or ``None`` if no marker is set."""
    try:
        text = marker_path.read_text(encoding="utf-8").strip()
    except FileNotFoundError:
        return None
    return text or None


def is_acknowledged(marker_path: Path, passphrase: str) -> bool:
    """Return ``True`` iff *passphrase*'s fingerprint matches the on-disk marker."""
    recorded = acknowledged_fingerprint(marker_path)
    return recorded is not None and recorded == fingerprint(passphrase)


def acknowledge(marker_path: Path, passphrase: str) -> None:
    """Persist the marker for *passphrase*.  Idempotent."""
    # Lazy import keeps the foundation layer free of an eager
    # ``_yaml`` (round-trip YAML editor) load; this code path only
    # runs at setup / ack time, not on every chain walk.
    from terok_sandbox._yaml import write_secret_text  # noqa: PLC0415

    write_secret_text(marker_path, fingerprint(passphrase) + "\n")


def forget(marker_path: Path) -> None:
    """Remove the marker; no-op if absent."""
    marker_path.unlink(missing_ok=True)


__all__ = [
    "acknowledge",
    "acknowledged_fingerprint",
    "fingerprint",
    "forget",
    "is_acknowledged",
]
