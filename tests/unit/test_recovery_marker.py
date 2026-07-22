# SPDX-FileCopyrightText: 2026 Jiri Vyskocil
# SPDX-License-Identifier: Apache-2.0

"""Tests for the vault recovery-key acknowledgement marker.

The marker is a zero-byte sidecar file — presence = acknowledged,
absence = unconfirmed.  An earlier iteration stored a
passphrase-derived fingerprint here, but that turned the file into
an offline-guessing oracle if it leaked via backup or misconfigured
permissions (audit finding #2 on PR #325).  An empty file leaks
nothing.

Trade-off: a passphrase rotation no longer auto-invalidates the
marker.  Operators who rotate should re-ack via ``vault passphrase
reveal`` (or use the destructive ``destroy`` flow, which clears the
marker for them).
"""

from __future__ import annotations

from pathlib import Path

import pytest

from terok_sandbox import RecoveryStatus, SandboxConfig
from terok_sandbox.vault.store.recovery import acknowledge, acknowledged, forget
from terok_sandbox.vault.store.tiers import PassphraseTier


def _cfg(tmp_path: Path, *, unlocked: bool = True) -> SandboxConfig:
    """Sandbox config rooted under *tmp_path*.

    The marker is independent of the passphrase; the *unlocked* keyword
    switches the keyring tier (which the conftest stub resolves to a
    deterministic passphrase) on or off, so the locked-vault branch
    (where the rest of the chain would be inoperative anyway) can be
    exercised explicitly.
    """
    return SandboxConfig(
        state_dir=tmp_path / "state",
        runtime_dir=tmp_path / "rt",
        config_dir=tmp_path / "cfg",
        vault_dir=tmp_path / "vault",
        services_mode="socket",
        credentials_use_keyring=unlocked,
    )


class TestSidecarPrimitives:
    """Sidecar marker read/write/forget — the file-level contract."""

    def test_acknowledge_creates_zero_byte_marker(self, tmp_path: Path) -> None:
        """Marker file is created empty — leaks no information about the passphrase."""
        marker = tmp_path / "vault.recovery_acknowledged"
        acknowledge(marker)
        assert marker.exists()
        assert marker.stat().st_size == 0

    def test_acknowledge_writes_secret_perms(self, tmp_path: Path) -> None:
        """The marker file is owner-only (0o600) — same level as the DB."""
        marker = tmp_path / "vault.recovery_acknowledged"
        acknowledge(marker)
        assert (marker.stat().st_mode & 0o777) == 0o600

    def test_acknowledge_then_acknowledged_true(self, tmp_path: Path) -> None:
        """The round-trip: write then check — matches."""
        marker = tmp_path / "vault.recovery_acknowledged"
        acknowledge(marker)
        assert acknowledged(marker)

    def test_acknowledged_false_when_marker_missing(self, tmp_path: Path) -> None:
        """A pristine install starts with no marker — must read as unconfirmed."""
        assert not acknowledged(tmp_path / "vault.recovery_acknowledged")

    def test_acknowledge_is_idempotent(self, tmp_path: Path) -> None:
        """Calling twice doesn't error and leaves the file in place."""
        marker = tmp_path / "vault.recovery_acknowledged"
        acknowledge(marker)
        acknowledge(marker)
        assert acknowledged(marker)

    def test_forget_removes_marker(self, tmp_path: Path) -> None:
        """``forget`` clears the marker so the unconfirmed state surfaces again."""
        marker = tmp_path / "vault.recovery_acknowledged"
        acknowledge(marker)
        forget(marker)
        assert not acknowledged(marker)

    def test_forget_is_idempotent(self, tmp_path: Path) -> None:
        """Forgetting a missing marker is a silent no-op."""
        marker = tmp_path / "absent"
        forget(marker)  # must not raise
        assert not acknowledged(marker)


class TestRecoveryStatusMarkerMethods:
    """``RecoveryStatus.is_acknowledged`` / ``.acknowledge`` (terok consumes these)."""

    def test_is_acknowledged_false_when_marker_missing(self, tmp_path: Path) -> None:
        """Fresh install — marker missing → False."""
        assert RecoveryStatus.is_acknowledged(_cfg(tmp_path)) is False

    def test_acknowledge_writes_marker(self, tmp_path: Path) -> None:
        """Writing through the classmethod lands the marker on disk."""
        cfg = _cfg(tmp_path)
        RecoveryStatus.acknowledge(cfg)
        assert RecoveryStatus.is_acknowledged(cfg) is True

    def test_acknowledge_works_on_locked_vault(self, tmp_path: Path) -> None:
        """Marker is independent of the passphrase resolver — locked vault ack works.

        Closing the loop quickly matters more than gating on a working
        resolver chain; the marker just records "operator confirmed".
        """
        cfg = _cfg(tmp_path, unlocked=False)
        RecoveryStatus.acknowledge(cfg)
        assert RecoveryStatus.is_acknowledged(cfg) is True


class TestRecoveryStatusDefaultConfig:
    """``cfg=None`` branches in the classmethods."""

    def test_is_acknowledged_constructs_default_cfg(
        self,
        tmp_path: Path,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        """No-arg call lazy-builds ``SandboxConfig()`` (the documented default)."""
        sentinel = _cfg(tmp_path)
        RecoveryStatus.acknowledge(sentinel)  # pre-seed the marker
        monkeypatch.setattr("terok_sandbox.config.SandboxConfig", lambda: sentinel)
        assert RecoveryStatus.is_acknowledged() is True

    def test_acknowledge_constructs_default_cfg(
        self,
        tmp_path: Path,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        """No-arg call lazy-builds ``SandboxConfig()`` and lands the marker."""
        sentinel = _cfg(tmp_path)
        monkeypatch.setattr("terok_sandbox.config.SandboxConfig", lambda: sentinel)
        RecoveryStatus.acknowledge()
        assert sentinel.vault_recovery_marker_file.exists()


class TestRecoveryStatus:
    """``RecoveryStatus.load`` bundles the marker + resolved source for every surface."""

    def test_acked_durable_tier_not_urgent(self, tmp_path: Path) -> None:
        """Acknowledged + keyring tier → not urgent (durable, ack present)."""
        cfg = _cfg(tmp_path)
        RecoveryStatus.acknowledge(cfg)
        status = RecoveryStatus.load(cfg)
        assert status.acknowledged is True
        assert status.source is PassphraseTier.KEYRING
        assert status.volatile_only is False
        assert status.urgent is False

    def test_unacked_durable_tier_not_urgent(self, tmp_path: Path) -> None:
        """Unacknowledged + durable tier → warn but NOT urgent (no reboot loss risk)."""
        status = RecoveryStatus.load(_cfg(tmp_path))
        assert status.acknowledged is False
        assert status.volatile_only is False
        assert status.urgent is False

    def test_unacked_volatile_only_is_urgent(
        self,
        tmp_path: Path,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        """Unacknowledged + kernel-keyring → urgent (one logout away from loss)."""
        from terok_sandbox.vault.store import encryption as enc

        # Spoof the chain so it resolves via the kernel-keyring tier.  We
        # bypass the conftest stub that nulls the kernel-keyring tier
        # by patching the resolver entry point directly.
        cfg = _cfg(tmp_path)
        monkeypatch.setattr(
            enc,
            "resolve_passphrase_with_source",
            lambda **_kw: ("p4ss", PassphraseTier.KERNEL_KEYRING),
        )
        status = RecoveryStatus.load(cfg)
        assert status.volatile_only is True
        assert status.urgent is True

    def test_acked_volatile_only_not_urgent(
        self,
        tmp_path: Path,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        """Acknowledged + kernel-keyring → not urgent (operator saved it; rebooting just
        means re-unlock + the ack stays valid because the marker is independent)."""
        from terok_sandbox.vault.store import encryption as enc

        cfg = _cfg(tmp_path)
        RecoveryStatus.acknowledge(cfg)
        monkeypatch.setattr(
            enc,
            "resolve_passphrase_with_source",
            lambda **_kw: ("p4ss", PassphraseTier.KERNEL_KEYRING),
        )
        assert RecoveryStatus.load(cfg).urgent is False

    def test_locked_vault_source_is_none(self, tmp_path: Path) -> None:
        """No resolvable passphrase → source=None, not urgent (locked-vault check owns it)."""
        status = RecoveryStatus.load(_cfg(tmp_path, unlocked=False))
        assert status.source is None
        assert status.volatile_only is False
        assert status.urgent is False

    def test_resolver_exception_collapses_to_locked(
        self,
        tmp_path: Path,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        """``WrongPassphraseError`` from the resolver → source=None, not urgent.

        Pins the except branch in
        [`RecoveryStatus.load`][terok_sandbox.vault.store.recovery.RecoveryStatus.load]
        — a vault that fails to resolve (mismatched passphrase tier,
        corrupt session file, denied keyring) should surface the same
        "source unknown" shape as a fresh locked install rather than
        propagating the resolver exception into every doctor / sickbay /
        pill caller.
        """
        from terok_sandbox.config import SandboxConfig as _SandboxConfig
        from terok_sandbox.vault.store.encryption import WrongPassphraseError

        cfg = _cfg(tmp_path)

        def _raise(self, **_kw: object) -> object:
            raise WrongPassphraseError("decryption failed under test")

        monkeypatch.setattr(_SandboxConfig, "resolve_passphrase_with_source", _raise)
        status = RecoveryStatus.load(cfg)
        assert status.source is None
        assert status.urgent is False
        # The fail-closed message is preserved so status surfaces can say
        # "a configured tier is unreadable" instead of a bare "locked".
        assert status.resolve_error == "decryption failed under test"

    def test_resolver_no_passphrase_exception_is_plain_lock(
        self,
        tmp_path: Path,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        """``NoPassphraseError`` from the resolver → plain lock, no resolve_error.

        The resolver normally *returns* ``(None, None)`` rather than
        raising, so this branch is defensive — but a raise must collapse
        to the same "locked, nothing broken" shape, not be confused with
        a fail-closed broken tier.
        """
        from terok_sandbox.config import SandboxConfig as _SandboxConfig
        from terok_sandbox.vault.store.encryption import NoPassphraseError

        cfg = _cfg(tmp_path)

        def _raise(self, **_kw: object) -> object:
            raise NoPassphraseError("nothing resolvable under test")

        monkeypatch.setattr(_SandboxConfig, "resolve_passphrase_with_source", _raise)
        status = RecoveryStatus.load(cfg)
        assert status.source is None
        assert status.resolve_error is None

    def test_clean_resolution_has_no_resolve_error(self, tmp_path: Path) -> None:
        """A chain that resolves (or yields nothing) reports resolve_error=None."""
        assert RecoveryStatus.load(_cfg(tmp_path)).resolve_error is None
        assert RecoveryStatus.load(_cfg(tmp_path, unlocked=False)).resolve_error is None


class TestNoOfflineOracle:
    """The audit-driven contract: the marker file leaks no passphrase signal.

    Before fix: marker contained ``sha256(constant_salt || passphrase)`` —
    an attacker reading the file could brute-force the passphrase
    offline by hashing candidates.  After fix: marker is zero bytes,
    so no oracle is possible regardless of who reads it.
    """

    def test_marker_content_is_empty(self, tmp_path: Path) -> None:
        """A read of the on-disk marker yields no data — nothing to guess against."""
        cfg = _cfg(tmp_path)
        RecoveryStatus.acknowledge(cfg)
        assert cfg.vault_recovery_marker_file.read_bytes() == b""

    def test_marker_does_not_change_with_passphrase(self, tmp_path: Path) -> None:
        """Two independent installs produce identical (empty) marker bytes.

        Nothing passphrase-derived is written, so the bytes cannot vary
        with the vault's passphrase — there is no signal to compare.
        """
        cfg_a = _cfg(tmp_path / "a")
        cfg_b = _cfg(tmp_path / "b")
        RecoveryStatus.acknowledge(cfg_a)
        RecoveryStatus.acknowledge(cfg_b)
        assert (
            cfg_a.vault_recovery_marker_file.read_bytes()
            == cfg_b.vault_recovery_marker_file.read_bytes()
            == b""
        )
