# SPDX-FileCopyrightText: 2026 Jiri Vyskocil
# SPDX-License-Identifier: Apache-2.0

"""Tests for the vault recovery-key acknowledgement marker.

Covers both the low-level sidecar primitives in
[`terok_sandbox.vault.store.recovery`][terok_sandbox.vault.store.recovery]
and the top-level convenience wrappers
([`is_recovery_acknowledged`][terok_sandbox.is_recovery_acknowledged],
[`acknowledge_recovery`][terok_sandbox.acknowledge_recovery]) so the
chain-walking behaviour terok depends on stays pinned.
"""

from __future__ import annotations

from pathlib import Path

from terok_sandbox import (
    SandboxConfig,
    acknowledge_recovery,
    is_recovery_acknowledged,
)
from terok_sandbox.vault.store.recovery import (
    acknowledge,
    acknowledged_fingerprint,
    fingerprint,
    forget,
    is_acknowledged,
)

_PASSPHRASE = "correct-horse-battery-staple"
_OTHER = "tr0ub4dor&3"


def _cfg(tmp_path: Path, *, passphrase: str | None = _PASSPHRASE) -> SandboxConfig:
    """Return a SandboxConfig pinned under *tmp_path*; passphrase via config tier.

    ``credentials_use_keyring=False`` keeps the resolver away from the
    host's real keyring so the test outcome doesn't depend on whatever
    happens to be stored there.  ``credentials_passphrase=None`` lets
    a caller exercise the "locked vault" branch where no tier resolves.
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


class TestFingerprint:
    """The fingerprint is the foundation everything else compares against."""

    def test_stable_across_calls(self) -> None:
        """Same input → same digest, every time."""
        assert fingerprint(_PASSPHRASE) == fingerprint(_PASSPHRASE)

    def test_different_passphrase_different_digest(self) -> None:
        """A re-key naturally invalidates a stored marker."""
        assert fingerprint(_PASSPHRASE) != fingerprint(_OTHER)

    def test_digest_does_not_contain_passphrase(self) -> None:
        """SHA-256 is one-way; the fingerprint is safe to persist as-is."""
        digest = fingerprint(_PASSPHRASE)
        assert _PASSPHRASE not in digest


class TestSidecarPrimitives:
    """Sidecar marker read/write/forget — the file-level contract."""

    def test_acknowledge_writes_secret_perms(self, tmp_path: Path) -> None:
        """The marker file is owner-only (0o600) — same level as the DB."""
        marker = tmp_path / "vault.recovery_acknowledged"
        acknowledge(marker, _PASSPHRASE)
        assert (marker.stat().st_mode & 0o777) == 0o600

    def test_acknowledge_then_is_acknowledged_true(self, tmp_path: Path) -> None:
        """The round-trip: write then check — matches."""
        marker = tmp_path / "vault.recovery_acknowledged"
        acknowledge(marker, _PASSPHRASE)
        assert is_acknowledged(marker, _PASSPHRASE)

    def test_is_acknowledged_false_when_marker_missing(self, tmp_path: Path) -> None:
        """A pristine install starts with no marker — must read as unconfirmed."""
        marker = tmp_path / "vault.recovery_acknowledged"
        assert not is_acknowledged(marker, _PASSPHRASE)

    def test_rotation_invalidates_marker(self, tmp_path: Path) -> None:
        """Re-keying must re-prompt — fingerprint mismatch is enough."""
        marker = tmp_path / "vault.recovery_acknowledged"
        acknowledge(marker, _PASSPHRASE)
        assert not is_acknowledged(marker, _OTHER)

    def test_acknowledge_is_idempotent(self, tmp_path: Path) -> None:
        """Calling twice doesn't change the on-disk contents."""
        marker = tmp_path / "vault.recovery_acknowledged"
        acknowledge(marker, _PASSPHRASE)
        first = marker.read_text()
        acknowledge(marker, _PASSPHRASE)
        assert marker.read_text() == first

    def test_acknowledged_fingerprint_returns_recorded_value(self, tmp_path: Path) -> None:
        """Diagnostic helper for surfaces that show the marker state."""
        marker = tmp_path / "vault.recovery_acknowledged"
        acknowledge(marker, _PASSPHRASE)
        assert acknowledged_fingerprint(marker) == fingerprint(_PASSPHRASE)

    def test_acknowledged_fingerprint_none_when_missing(self, tmp_path: Path) -> None:
        """No marker → no fingerprint; the absence is the answer."""
        assert acknowledged_fingerprint(tmp_path / "absent") is None

    def test_forget_is_idempotent(self, tmp_path: Path) -> None:
        """Forgetting a missing marker is a silent no-op."""
        marker = tmp_path / "absent"
        forget(marker)  # must not raise
        acknowledge(marker, _PASSPHRASE)
        forget(marker)
        assert not marker.exists()


class TestTopLevelWrappers:
    """``is_recovery_acknowledged`` / ``acknowledge_recovery`` (terok consumes these)."""

    def test_is_recovery_acknowledged_false_when_marker_missing(self, tmp_path: Path) -> None:
        """Fresh install — vault resolves, marker missing → False."""
        cfg = _cfg(tmp_path)
        assert is_recovery_acknowledged(cfg) is False

    def test_acknowledge_recovery_writes_marker(self, tmp_path: Path) -> None:
        """Writing through the wrapper lands the right fingerprint on disk."""
        cfg = _cfg(tmp_path)
        assert acknowledge_recovery(cfg) is True
        assert is_recovery_acknowledged(cfg) is True

    def test_is_recovery_acknowledged_false_when_vault_locked(self, tmp_path: Path) -> None:
        """No resolvable passphrase → False (the warning is conservative by design)."""
        assert is_recovery_acknowledged(_cfg(tmp_path, passphrase=None)) is False

    def test_acknowledge_recovery_false_when_vault_locked(self, tmp_path: Path) -> None:
        """Locked vault → no fingerprint to write; the wrapper short-circuits."""
        cfg = _cfg(tmp_path, passphrase=None)
        assert acknowledge_recovery(cfg) is False
        assert not cfg.vault_recovery_marker_file.exists()


class TestRotationFlow:
    """End-to-end: ack the current key, rotate, see the warning come back."""

    def test_re_key_re_triggers_warning(self, tmp_path: Path) -> None:
        """After re-keying we expect ``is_recovery_acknowledged`` to flip back to False."""
        cfg = _cfg(tmp_path)
        acknowledge_recovery(cfg)
        assert is_recovery_acknowledged(cfg)

        # Simulate a rotation by swapping in a different config-tier
        # passphrase — the resolver fingerprint changes, the marker
        # doesn't, mismatch → False.
        assert is_recovery_acknowledged(_cfg(tmp_path, passphrase=_OTHER)) is False
