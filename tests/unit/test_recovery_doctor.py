# SPDX-FileCopyrightText: 2026 Jiri Vyskocil
# SPDX-License-Identifier: Apache-2.0

"""Tests for the ``Recovery key acknowledged`` doctor check.

Exercises every branch of [`_make_recovery_acknowledged_check`][terok_sandbox.doctor._make_recovery_acknowledged_check]:
locked vault (deferred ``ok``), missing marker (``warn``), fingerprint
mismatch after rotation (``warn``), and the happy path (``ok``).
"""

from __future__ import annotations

from pathlib import Path
from unittest.mock import patch

from terok_sandbox import SandboxConfig
from terok_sandbox.doctor import sandbox_doctor_checks
from terok_sandbox.vault.store.recovery import acknowledge as _acknowledge

_PASSPHRASE = "correct-horse-battery-staple"


def _cfg(tmp_path: Path, *, passphrase: str | None = _PASSPHRASE) -> SandboxConfig:
    """Sandbox config rooted under tmp_path with the keyring tier disabled."""
    return SandboxConfig(
        state_dir=tmp_path / "state",
        runtime_dir=tmp_path / "rt",
        config_dir=tmp_path / "cfg",
        vault_dir=tmp_path / "vault",
        services_mode="socket",
        credentials_passphrase=passphrase,
        credentials_use_keyring=False,
    )


def _find_check(checks: list[object], label: str) -> object:
    """Pull the ``DoctorCheck`` with the matching label out of the list."""
    matches = [c for c in checks if getattr(c, "label", "") == label]
    assert len(matches) == 1, f"expected one {label!r} check, found {len(matches)}"
    return matches[0]


def _eval_recovery(cfg: SandboxConfig) -> object:
    """Build the doctor checks under *cfg* and evaluate the recovery one."""
    check = _find_check(
        sandbox_doctor_checks(
            token_broker_port=None, ssh_signer_port=None, desired_shield_state=None
        ),
        "Recovery key acknowledged",
    )
    with patch("terok_sandbox.config.SandboxConfig", return_value=cfg):
        return check.evaluate(0, "", "")


class TestRecoveryAcknowledgedCheck:
    """End-to-end behaviour of the doctor check across vault states."""

    def test_unlocked_with_marker_returns_ok(self, tmp_path: Path) -> None:
        """Marker matches resolver fingerprint → ``ok``."""
        cfg = _cfg(tmp_path)
        _acknowledge(cfg.vault_recovery_marker_file, _PASSPHRASE)
        verdict = _eval_recovery(cfg)
        assert verdict.severity == "ok"
        assert "acknowledged" in verdict.detail

    def test_unlocked_no_marker_returns_warn(self, tmp_path: Path) -> None:
        """Vault unlocks but marker is absent → ``warn`` with the reveal hint."""
        verdict = _eval_recovery(_cfg(tmp_path))
        assert verdict.severity == "warn"
        assert "unconfirmed" in verdict.detail
        assert "vault passphrase reveal" in verdict.detail

    def test_marker_stale_after_rotation_returns_warn(self, tmp_path: Path) -> None:
        """Marker for a different passphrase → ``warn`` (re-key invalidates ack)."""
        cfg = _cfg(tmp_path)
        _acknowledge(cfg.vault_recovery_marker_file, "old-passphrase")
        verdict = _eval_recovery(cfg)
        assert verdict.severity == "warn"

    def test_locked_vault_defers_to_other_check(self, tmp_path: Path) -> None:
        """Unresolvable passphrase → ``ok`` (locked-vault check above is the right surface)."""
        verdict = _eval_recovery(_cfg(tmp_path, passphrase=None))
        assert verdict.severity == "ok"
        assert "deferred" in verdict.detail

    def test_fix_description_mentions_both_paths(self) -> None:
        """The remediation hint covers both interactive and CI flows."""
        check = _find_check(
            sandbox_doctor_checks(
                token_broker_port=None, ssh_signer_port=None, desired_shield_state=None
            ),
            "Recovery key acknowledged",
        )
        assert "vault passphrase reveal" in check.fix_description
        assert "vault passphrase acknowledge" in check.fix_description
