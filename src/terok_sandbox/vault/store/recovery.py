# SPDX-FileCopyrightText: 2026 Jiri Vyskocil
# SPDX-License-Identifier: Apache-2.0

"""Track operator-confirmed possession of the vault recovery passphrase.

Whenever the sandbox mints a passphrase the operator never typed
(auto-detected systemd-creds tier, fresh keyring entry, or the
session-file fallback), they need a written copy stashed off-host —
every keystore tier is bound to *this* machine, account, or boot, so
a disk failure or TPM transplant strands the vault.

The marker is a **zero-byte sidecar file** under ``vault_dir`` (mode
``0o600``).  Presence means "operator confirmed they saved the
recovery key"; absence means "unconfirmed".  An earlier iteration
stored a SHA-256 fingerprint of the passphrase here so a re-key
would auto-invalidate the marker, but that turned the sidecar into
an offline-guessing oracle: anyone with read access to the file
(e.g. a leaked backup) could brute-force the passphrase by hashing
candidates and comparing.  An empty file leaks nothing.

The trade-off: a passphrase rotation does NOT auto-invalidate the
marker.  Operators who rotate their key should re-ack (interactive
``vault passphrase reveal`` or silent ``vault passphrase
acknowledge``).  The destructive ``vault passphrase destroy`` flow
clears the marker for them.
"""

from __future__ import annotations

from pathlib import Path


def acknowledged(marker_path: Path) -> bool:
    """Return ``True`` iff the marker file exists.

    A bare ``stat`` — never reads the file content because the file
    has none.  Any ``OSError`` (permission denied, broken symlink,
    busy mount) degrades to ``False`` so the unconfirmed-warning
    surfaces can keep going on a host with an unreadable marker.
    """
    try:
        marker_path.stat()
    except OSError:
        return False
    return True


def acknowledge(marker_path: Path) -> None:
    """Create the empty marker file (owner-only, ``0o600``).  Idempotent."""
    # Lazy import keeps the foundation layer free of an eager
    # ``_yaml`` (round-trip YAML editor) load; this code path only
    # runs at setup / ack time, not on every chain walk.
    from terok_sandbox._yaml import write_secret_text  # noqa: PLC0415

    write_secret_text(marker_path, "")


def forget(marker_path: Path) -> None:
    """Remove the marker; no-op if absent."""
    marker_path.unlink(missing_ok=True)


__all__ = [
    "acknowledge",
    "acknowledged",
    "forget",
]
