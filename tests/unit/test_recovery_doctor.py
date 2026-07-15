# SPDX-FileCopyrightText: 2026 Jiri Vyskocil
# SPDX-License-Identifier: Apache-2.0

"""Tests for the ``Recovery key acknowledged`` doctor check.

Three severity bands:

* marker present → ``ok``
* marker absent + session-file resolver → ``error`` (one reboot away
  from losing the vault — the session tier is wiped on restart)
* marker absent + any durable tier → ``warn`` (machine-bound; needs
  an off-host copy for hardware-failure DR)

Plus the fix-description's dual-flow remediation hint.
"""

from __future__ import annotations

from pathlib import Path
from unittest.mock import patch

from terok_sandbox import SandboxConfig
from terok_sandbox.doctor import make_recovery_acknowledged_check
from terok_sandbox.vault.store.recovery import acknowledge as _acknowledge
from terok_sandbox.vault.store.tiers import PassphraseTier


def _cfg(tmp_path: Path) -> SandboxConfig:
    """Sandbox config rooted under tmp_path.

    The keyring tier is on, so the chain resolves via the conftest
    stub's deterministic keyring passphrase — a durable tier.
    """
    return SandboxConfig(
        state_dir=tmp_path / "state",
        runtime_dir=tmp_path / "rt",
        config_dir=tmp_path / "cfg",
        vault_dir=tmp_path / "vault",
        services_mode="socket",
        credentials_use_keyring=True,
    )


def _eval_recovery(cfg: SandboxConfig) -> object:
    """Build the recovery check under *cfg* and evaluate it.

    Patches ``terok_sandbox.config.SandboxConfig`` because the doctor
    check resolves ``from .config import SandboxConfig`` at call time
    (foundation-layer reach-around to keep the check off the package's
    surface-layer ``recovery_status`` wrapper).
    """
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

    def test_marker_missing_with_durable_tier_returns_warn(self, tmp_path: Path) -> None:
        """Marker absent + non-session-file source → ``warn`` naming both remediations."""
        # ``_cfg`` resolves via the keyring tier (a durable, machine-bound
        # store) — missing marker is "warn", not "error".
        verdict = _eval_recovery(_cfg(tmp_path))
        assert verdict.severity == "warn"
        assert "unconfirmed" in verdict.detail
        # Pin BOTH remediation verbs — the interactive one (reveal +
        # type SAVED) and the silent one (acknowledge, used by CI / TUI
        # after the value was captured out-of-band).
        assert "vault passphrase reveal" in verdict.detail
        assert "vault passphrase acknowledge" in verdict.detail

    def test_marker_missing_with_session_only_returns_error(self, tmp_path: Path) -> None:
        """Marker absent + session-file source → ``error`` with loud "next reboot" text.

        The session-unlock tmpfs file is wiped on every reboot, so an
        unconfirmed session-only key means the vault becomes
        unrecoverable on the next restart — a genuinely higher-severity
        state than the generic machine-bound warning.
        """
        cfg = _cfg(tmp_path)
        # Spoof the chain so it resolves via the session-file tier.
        # We don't need a real session-file on disk — the doctor only
        # reads the returned source string.
        with patch.object(
            type(cfg),
            "resolve_passphrase_with_source",
            lambda self, **_kw: ("p4ss", PassphraseTier.SESSION_FILE),
        ):
            verdict = _eval_recovery(cfg)
        assert verdict.severity == "error"
        # The loud text must call out the reboot lifetime explicitly,
        # not just generic "machine-bound" — the operator needs to
        # understand the difference between "save it eventually" and
        # "save it NOW or lose it on the next reboot".
        assert "session-unlock" in verdict.detail
        assert "reboot" in verdict.detail.lower()
        assert "UNRECOVERABLE" in verdict.detail
        # Remediation verbs still surface for the operator to act.
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
