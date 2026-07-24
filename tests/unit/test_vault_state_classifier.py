# SPDX-FileCopyrightText: 2026 Jiri Vyskocil
# SPDX-License-Identifier: Apache-2.0

"""Tests for the [`VaultStatus`][terok_sandbox.vault.store.status.VaultStatus] snapshot.

``VaultStatus.load`` is the one vault-state picture every frontend
renders; these tests pin its two load-time guarantees — the
UNPROVISIONED / UNLOCKED classification on a fresh install, and the
read-only promise that a status *read* never creates the credentials
DB (SQLite would otherwise mint it as a side effect, keying the vault
to whatever tier happened to resolve) — plus the shared warning
catalog whose wording every surface re-prints verbatim.  The rendering
of the snapshot is covered in ``test_vault_status.py``.
"""

from __future__ import annotations

from pathlib import Path
from unittest.mock import MagicMock

import pytest

import terok_sandbox.vault.store.kernel_keyring as _kk
from terok_sandbox.vault.store import encryption
from terok_sandbox.vault.store.recovery import RecoveryStatus
from terok_sandbox.vault.store.status import (
    VaultState,
    VaultStatus,
    VaultWarningKind,
    _build_warnings,
)
from terok_sandbox.vault.store.tiers import PassphraseTier


def _load_cfg(
    tmp_path: Path,
    *,
    resolved: tuple[str | None, PassphraseTier | None] = (None, None),
) -> MagicMock:
    """A mock ``SandboxConfig`` for ``VaultStatus.load`` over a fresh install.

    ``db_path`` points into *tmp_path* but is never created — the tests
    assert it stays that way.  *resolved* is what the config's resolver
    seam reports, feeding ``RecoveryStatus.load`` without a stub.
    """
    cfg = MagicMock()
    cfg.vault_systemd_creds_file = tmp_path / "no-sealed.cred"
    cfg.credentials_use_keyring = False
    cfg.credentials_passphrase_command = None
    cfg.db_path = tmp_path / "vault" / "credentials.db"
    cfg.vault_recovery_marker_file = tmp_path / "no-marker"
    cfg.resolve_passphrase_with_source.return_value = resolved
    return cfg


class TestVaultStatusLoad:
    """``VaultStatus.load`` classifies fresh installs without touching the DB."""

    def test_fresh_install_is_unprovisioned_and_creates_nothing(self, tmp_path: Path) -> None:
        """No DB + empty chain → UNPROVISIONED, and the DB file stays absent."""
        cfg = _load_cfg(tmp_path)
        status = VaultStatus.load(cfg)
        assert status.state is VaultState.UNPROVISIONED
        assert status.db_exists is False
        assert status.source is None
        assert status.providers is None
        assert not cfg.db_path.exists()  # the read minted no DB on disk
        cfg.open_credential_db.assert_not_called()

    def test_ready_kernel_keyring_tier_without_db_is_unlocked(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """A cached kernel-keyring key + no DB → UNLOCKED ("the key is ready"), still no DB open."""
        monkeypatch.setattr(_kk, "is_cached", lambda: True)
        cfg = _load_cfg(tmp_path, resolved=("hunter2", PassphraseTier.KERNEL_KEYRING))
        status = VaultStatus.load(cfg)
        assert status.state is VaultState.UNLOCKED
        assert status.db_exists is False
        assert status.source is PassphraseTier.KERNEL_KEYRING
        assert status.providers == ()  # nothing stored yet, but readable-by-key
        # The kernel-keyring row (index 2) is the active one.
        assert status.chain[2].active is True
        assert not cfg.db_path.exists()  # unlocked-by-classification, not by opening
        cfg.open_credential_db.assert_not_called()

    def test_broken_tier_without_db_is_locked_not_unprovisioned(self, tmp_path: Path) -> None:
        """A fail-closed tier on a DB-less host is a fault to fix, not a fresh install."""
        cfg = _load_cfg(tmp_path)
        cfg.resolve_passphrase_with_source.side_effect = encryption.WrongPassphraseError(
            "sealed credential present but could not be unsealed"
        )
        status = VaultStatus.load(cfg)
        assert status.state is VaultState.LOCKED
        assert status.lock_reason is not None and "could not be unsealed" in status.lock_reason
        assert any(w.kind is VaultWarningKind.BROKEN_TIER for w in status.warnings)


def _recovery(
    source: PassphraseTier | None,
    *,
    acknowledged: bool = False,
    resolve_error: str | None = None,
) -> RecoveryStatus:
    """A real ``RecoveryStatus`` with the catalog-relevant fields pinned."""
    return RecoveryStatus(acknowledged=acknowledged, source=source, resolve_error=resolve_error)


class TestBuildWarnings:
    """The warning catalog authors each situation's wording exactly once."""

    def test_unacked_volatile_only_is_urgent(self) -> None:
        """Kernel-keyring-only + unacknowledged → the logout-loss error, nothing softer."""
        warnings = _build_warnings(_recovery(PassphraseTier.KERNEL_KEYRING))
        assert [w.kind for w in warnings] == [VaultWarningKind.RECOVERY_VOLATILE]
        (warning,) = warnings
        assert warning.severity == "error"
        assert (
            "the only copy of the vault passphrase is the kernel-keyring cache" in warning.message
        )
        assert "unrecoverable" in warning.message

    def test_unacked_durable_tier_is_unconfirmed(self) -> None:
        """A durable tier without an off-host copy gets the softer machine-bound warning."""
        warnings = _build_warnings(_recovery(PassphraseTier.KEYRING))
        assert [w.kind for w in warnings] == [VaultWarningKind.RECOVERY_UNCONFIRMED]
        (warning,) = warnings
        assert warning.severity == "warning"
        assert warning.message == (
            "the vault passphrase is not confirmed saved off-host — every"
            " storage tier is bound to this machine, account, or boot, so a"
            " hardware failure strands the vault without a written copy"
        )

    def test_acknowledged_or_unresolved_emits_no_recovery_warning(self) -> None:
        """An acked vault (or one with nothing resolved) has nothing to nag about."""
        assert _build_warnings(_recovery(PassphraseTier.KERNEL_KEYRING, acknowledged=True)) == ()
        assert _build_warnings(_recovery(PassphraseTier.KEYRING, acknowledged=True)) == ()
        assert _build_warnings(_recovery(None)) == ()
