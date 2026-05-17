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

from terok_sandbox import (
    SandboxConfig,
    acknowledge_recovery,
    is_recovery_acknowledged,
    recovery_status,
)
from terok_sandbox.vault.store.recovery import acknowledge, acknowledged, forget


def _cfg(tmp_path: Path, *, passphrase: str | None = "any-value") -> SandboxConfig:
    """Sandbox config rooted under *tmp_path*.

    The marker is independent of the passphrase; the *passphrase*
    keyword is kept on the helper signature only so the locked-vault
    branch (where the rest of the chain would be inoperative anyway)
    can be exercised explicitly.
    """
    return SandboxConfig(
        state_dir=tmp_path / "state",
        runtime_dir=tmp_path / "rt",
        config_dir=tmp_path / "cfg",
        vault_dir=tmp_path / "vault",
        services_mode="socket",
        credentials_passphrase=passphrase,
        credentials_use_keyring=False,
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


class TestTopLevelWrappers:
    """``is_recovery_acknowledged`` / ``acknowledge_recovery`` (terok consumes these)."""

    def test_is_recovery_acknowledged_false_when_marker_missing(self, tmp_path: Path) -> None:
        """Fresh install — marker missing → False."""
        assert is_recovery_acknowledged(_cfg(tmp_path)) is False

    def test_acknowledge_recovery_writes_marker(self, tmp_path: Path) -> None:
        """Writing through the wrapper lands the marker on disk."""
        cfg = _cfg(tmp_path)
        assert acknowledge_recovery(cfg) is True
        assert is_recovery_acknowledged(cfg) is True

    def test_acknowledge_recovery_works_on_locked_vault(self, tmp_path: Path) -> None:
        """Marker is independent of the passphrase resolver — locked vault ack works.

        Closing the loop quickly matters more than gating on a working
        resolver chain; the marker just records "operator confirmed".
        """
        cfg = _cfg(tmp_path, passphrase=None)
        assert acknowledge_recovery(cfg) is True
        assert is_recovery_acknowledged(cfg) is True


class TestTopLevelWrappersDefaultConfig:
    """``cfg=None`` branches in the top-level wrappers."""

    def test_is_recovery_acknowledged_constructs_default_cfg(
        self,
        tmp_path: Path,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        """No-arg call lazy-builds ``SandboxConfig()`` (the documented default)."""
        sentinel = _cfg(tmp_path)
        acknowledge_recovery(sentinel)  # pre-seed the marker
        monkeypatch.setattr("terok_sandbox.SandboxConfig", lambda: sentinel)
        assert is_recovery_acknowledged() is True

    def test_acknowledge_recovery_constructs_default_cfg(
        self,
        tmp_path: Path,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        """No-arg call lazy-builds ``SandboxConfig()`` and lands the marker."""
        sentinel = _cfg(tmp_path)
        monkeypatch.setattr("terok_sandbox.SandboxConfig", lambda: sentinel)
        assert acknowledge_recovery() is True
        assert sentinel.vault_recovery_marker_file.exists()


class TestRecoveryStatus:
    """``recovery_status`` bundles the marker + resolved source for every surface."""

    def test_acked_durable_tier_not_urgent(self, tmp_path: Path) -> None:
        """Acknowledged + config tier → not urgent (durable, ack present)."""
        cfg = _cfg(tmp_path)
        acknowledge_recovery(cfg)
        status = recovery_status(cfg)
        assert status.acknowledged is True
        assert status.source == "config"
        assert status.session_only is False
        assert status.urgent is False

    def test_unacked_durable_tier_not_urgent(self, tmp_path: Path) -> None:
        """Unacknowledged + durable tier → warn but NOT urgent (no reboot loss risk)."""
        status = recovery_status(_cfg(tmp_path))
        assert status.acknowledged is False
        assert status.session_only is False
        assert status.urgent is False

    def test_unacked_session_only_is_urgent(
        self,
        tmp_path: Path,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        """Unacknowledged + session-file → urgent (one reboot away from loss)."""
        from terok_sandbox.vault.store import encryption as enc

        # Spoof the chain so it resolves via the session-file tier.  We
        # bypass the conftest stub that nulls ``load_passphrase_from_file``
        # by patching the resolver entry point directly.
        cfg = _cfg(tmp_path)
        monkeypatch.setattr(
            enc, "resolve_passphrase_with_source", lambda **_kw: ("p4ss", "session-file")
        )
        status = recovery_status(cfg)
        assert status.session_only is True
        assert status.urgent is True

    def test_acked_session_only_not_urgent(
        self,
        tmp_path: Path,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        """Acknowledged + session-file → not urgent (operator saved it; rebooting just
        means re-unlock + the ack stays valid because the marker is independent)."""
        from terok_sandbox.vault.store import encryption as enc

        cfg = _cfg(tmp_path)
        acknowledge_recovery(cfg)
        monkeypatch.setattr(
            enc, "resolve_passphrase_with_source", lambda **_kw: ("p4ss", "session-file")
        )
        assert recovery_status(cfg).urgent is False

    def test_locked_vault_source_is_none(self, tmp_path: Path) -> None:
        """No resolvable passphrase → source=None, not urgent (locked-vault check owns it)."""
        status = recovery_status(_cfg(tmp_path, passphrase=None))
        assert status.source is None
        assert status.session_only is False
        assert status.urgent is False


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
        acknowledge_recovery(cfg)
        assert cfg.vault_recovery_marker_file.read_bytes() == b""

    def test_marker_does_not_change_with_passphrase(self, tmp_path: Path) -> None:
        """Two installs with different passphrases produce identical marker bytes."""
        cfg_a = _cfg(tmp_path / "a", passphrase="alpha")
        cfg_b = _cfg(tmp_path / "b", passphrase="beta")
        acknowledge_recovery(cfg_a)
        acknowledge_recovery(cfg_b)
        assert (
            cfg_a.vault_recovery_marker_file.read_bytes()
            == cfg_b.vault_recovery_marker_file.read_bytes()
            == b""
        )
