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

import dataclasses
from pathlib import Path
from typing import TYPE_CHECKING

from .encryption import PassphraseSource

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

    source: PassphraseSource | None
    """Whichever resolver tier unlocked the chain right now, or ``None`` if locked."""

    @property
    def session_only(self) -> bool:
        """``True`` iff the passphrase lives only in the tmpfs session-unlock file.

        That tier dies on the next reboot — without an off-host copy
        the vault becomes unrecoverable the moment the machine
        restarts.  Severity should escalate accordingly on every
        surface that renders this status.
        """
        return self.source == "session-file"

    @property
    def urgent(self) -> bool:
        """``True`` iff unacknowledged AND session-only (one reboot away from loss)."""
        return not self.acknowledged and self.session_only

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
        try:
            _passphrase, source = cfg.resolve_passphrase_with_source()
        except (NoPassphraseError, WrongPassphraseError):
            source = None
        return cls(
            acknowledged=acknowledged(cfg.vault_recovery_marker_file),
            source=source,
        )

    @staticmethod
    def is_acknowledged(cfg: SandboxConfig | None = None) -> bool:
        """Cheap marker-only check (no passphrase resolution).

        The vault's resolver tiers (systemd-creds, keyring,
        session-file) are all bound to *this* machine, account, or
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
