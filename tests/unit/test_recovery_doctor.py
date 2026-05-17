# SPDX-FileCopyrightText: 2026 Jiri Vyskocil
# SPDX-License-Identifier: Apache-2.0

"""Tests for the ``Recovery key acknowledged`` doctor check.

Exercises every branch of [`_make_recovery_acknowledged_check`][terok_sandbox.doctor._make_recovery_acknowledged_check]:
marker absent (``warn``), marker present (``ok``), and the
fix-description's dual-flow remediation hint.

The post-audit marker is a zero-byte sidecar — independent of the
passphrase resolver — so the check no longer has a "locked vault
defers" branch.  Pre-audit behaviour treated a missing fingerprint
as deferred; that's gone.
"""

from __future__ import annotations

from pathlib import Path
from unittest.mock import patch

from terok_sandbox import SandboxConfig
from terok_sandbox.doctor import make_recovery_acknowledged_check
from terok_sandbox.vault.store.recovery import acknowledge as _acknowledge


def _cfg(tmp_path: Path) -> SandboxConfig:
    """Sandbox config rooted under tmp_path."""
    return SandboxConfig(
        state_dir=tmp_path / "state",
        runtime_dir=tmp_path / "rt",
        config_dir=tmp_path / "cfg",
        vault_dir=tmp_path / "vault",
        services_mode="socket",
        credentials_passphrase="any-value",
        credentials_use_keyring=False,
    )


def _eval_recovery(cfg: SandboxConfig) -> object:
    """Build the recovery check under *cfg* and evaluate it."""
    check = make_recovery_acknowledged_check()
    with patch("terok_sandbox.config.SandboxConfig", return_value=cfg):
        return check.evaluate(0, "", "")


class TestRecoveryAcknowledgedCheck:
    """End-to-end behaviour of the doctor check across marker states."""

    def test_marker_present_returns_ok(self, tmp_path: Path) -> None:
        """Marker file exists → ``ok``."""
        cfg = _cfg(tmp_path)
        _acknowledge(cfg.vault_recovery_marker_file)
        verdict = _eval_recovery(cfg)
        assert verdict.severity == "ok"
        assert "acknowledged" in verdict.detail

    def test_marker_missing_returns_warn(self, tmp_path: Path) -> None:
        """Marker absent → ``warn`` naming both remediations."""
        verdict = _eval_recovery(_cfg(tmp_path))
        assert verdict.severity == "warn"
        assert "unconfirmed" in verdict.detail
        # Pin BOTH remediation verbs — the interactive one (reveal +
        # type SAVED) and the silent one (acknowledge, used by CI / TUI
        # after the value was captured out-of-band).
        assert "vault passphrase reveal" in verdict.detail
        assert "vault passphrase acknowledge" in verdict.detail

    def test_check_does_not_depend_on_passphrase_resolver(self, tmp_path: Path) -> None:
        """Locked vault still produces a meaningful answer — marker check is independent.

        Pre-audit the check called ``cfg.resolve_passphrase()`` to compute
        a fingerprint to compare against; a locked vault returned an
        unhelpful "deferred ok".  The decoupled marker check just looks
        at the sidecar, so a locked vault with no marker is still a
        clear ``warn`` (and with a marker is still a clear ``ok``).
        """
        cfg = _cfg(tmp_path)
        # Booby-trap the resolver: a regression that re-introduces the
        # passphrase dependency would surface as an exception here.
        with patch.object(
            type(cfg),
            "resolve_passphrase",
            lambda self, **_kw: (_ for _ in ()).throw(
                RuntimeError("resolver must not be called from the recovery check")
            ),
        ):
            assert _eval_recovery(cfg).severity == "warn"
            _acknowledge(cfg.vault_recovery_marker_file)
            assert _eval_recovery(cfg).severity == "ok"

    def test_fix_description_mentions_both_paths(self) -> None:
        """The remediation hint covers both interactive and CI flows."""
        check = make_recovery_acknowledged_check()
        assert "vault passphrase reveal" in check.fix_description
        assert "vault passphrase acknowledge" in check.fix_description
