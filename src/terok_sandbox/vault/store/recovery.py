# SPDX-FileCopyrightText: 2026 Jiri Vyskocil
# SPDX-License-Identifier: Apache-2.0

"""Track operator-confirmed possession of the vault recovery passphrase.

Whenever the sandbox mints a passphrase the operator never typed
(auto-detected systemd-creds tier, fresh keyring entry, or the
kernel-keyring cache), they need a written copy stashed off-host —
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
acknowledge``).  ``vault lock`` clears the marker for them when it
purges every tier.
"""

from __future__ import annotations

import dataclasses
from pathlib import Path
from typing import TYPE_CHECKING

from .tiers import PassphraseTier

if TYPE_CHECKING:
    from ...config import SandboxConfig


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


@dataclasses.dataclass(frozen=True)
class RecoveryStatus:
    """Combined marker + resolved-source view for the recovery-key warning surfaces.

    Returned by [`RecoveryStatus.load`][terok_sandbox.vault.store.recovery.RecoveryStatus.load]
    so sickbay / doctor / TUI / post-launch CLI all paint the same picture
    of "is the operator one reboot away from losing their vault?".
    """

    acknowledged: bool
    """``True`` iff the zero-byte marker file is present."""

    source: PassphraseTier | None
    """Whichever resolver tier unlocked the chain right now, or ``None`` if locked."""

    resolve_error: str | None = None
    """Diagnostic when the chain walk itself *raised* rather than yielded nothing.

    A configured tier that fails closed (sealed systemd-creds credential
    that won't unseal — machine identity changed; a ``passphrase_command``
    that produced nothing) is a different operator problem from "no
    passphrase anywhere": re-typing the passphrase won't fix a broken
    seal, and status surfaces must say which one they're looking at.
    ``None`` when resolution completed (with or without a source)."""

    @property
    def volatile_only(self) -> bool:
        """``True`` iff the passphrase lives only in a volatile tier.

        The kernel-keyring cache dies at logout (and never survives a
        reboot) — and the writer only populates it when no durable tier
        holds the passphrase, so a volatile resolved source means there
        is no reboot-surviving copy anywhere.  Without an off-host copy
        the vault becomes unrecoverable the moment the login session
        ends; severity escalates accordingly on every surface that
        renders this status.
        """
        return self.source is not None and not self.source.durable

    @property
    def urgent(self) -> bool:
        """``True`` iff unacknowledged AND volatile-only (one logout away from loss)."""
        return not self.acknowledged and self.volatile_only

    @classmethod
    def load(cls, cfg: SandboxConfig | None = None) -> RecoveryStatus:
        """Resolve marker + passphrase source for *cfg* (defaults if ``None``).

        Single seam for every "recovery key unconfirmed" surface —
        doctor, sickbay, TUI pill, post-task-launch CLI footer.
        Walking the resolver chain to find the source is cheap (no DB
        open, just tier knobs) and bundling it with the marker check
        here means no caller has to repeat the "is this session-only?"
        lookup.
        """
        from ...config import SandboxConfig  # noqa: PLC0415
        from .encryption import NoPassphraseError, WrongPassphraseError  # noqa: PLC0415

        cfg = cfg or SandboxConfig()
        resolve_error: str | None = None
        try:
            _passphrase, source = cfg.resolve_passphrase_with_source()
        except WrongPassphraseError as exc:
            # A tier is present but broken (unsealable credential, dead
            # helper) — fail-closed by design.  Record the message so
            # status surfaces can distinguish this from a plain lock.
            source = None
            resolve_error = str(exc)
        except NoPassphraseError:
            source = None
        return cls(
            acknowledged=acknowledged(cfg.vault_recovery_marker_file),
            source=source,
            resolve_error=resolve_error,
        )

    @staticmethod
    def is_acknowledged(cfg: SandboxConfig | None = None) -> bool:
        """Cheap marker-only check (no passphrase resolution).

        The vault's resolver tiers (systemd-creds, keyring,
        kernel-keyring) are all bound to *this* machine, account, or
        boot — a hardware failure or TPM transplant strands the vault
        without an off-host copy of the passphrase.  This check is
        what surfaces the "unconfirmed recovery key" warning in
        sickbay / doctor / the TUI pill: presence of a zero-byte
        marker file at
        [`vault_recovery_marker_file`][terok_sandbox.SandboxConfig.vault_recovery_marker_file]
        means the operator has acknowledged at some point.  Absence
        (or an unreadable marker) reports ``False`` — the warning is
        conservative by design.
        """
        from ...config import SandboxConfig  # noqa: PLC0415

        cfg = cfg or SandboxConfig()
        return acknowledged(cfg.vault_recovery_marker_file)

    @staticmethod
    def acknowledge(cfg: SandboxConfig | None = None) -> None:
        """Mark the recovery key as saved (writes the zero-byte sidecar marker).

        Always succeeds — the marker is independent of the passphrase
        resolver, so a locked vault doesn't block acknowledgement.
        Idempotent; safe to call on an already-acknowledged vault.
        """
        from ...config import SandboxConfig  # noqa: PLC0415

        cfg = cfg or SandboxConfig()
        acknowledge(cfg.vault_recovery_marker_file)


__all__ = [
    "RecoveryStatus",
    "acknowledge",
    "acknowledged",
    "forget",
]
