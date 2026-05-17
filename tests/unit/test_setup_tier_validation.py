# SPDX-FileCopyrightText: 2026 Jiri Vyskocil
# SPDX-License-Identifier: Apache-2.0

"""Tests for the upfront ``--passphrase-tier`` validation in setup.

A typo in ``--passphrase-tier`` must hard-fail *before* any host-
mutating phase runs (shield install in particular).  Without the
upfront check, the credentials phase rejection would land too late
to back out cleanly.
"""

from __future__ import annotations

from unittest.mock import patch

import pytest

from terok_sandbox.commands.sandbox import _handle_sandbox_setup, _validate_passphrase_tier


class TestValidatePassphraseTier:
    """Direct coverage for the small validator helper."""

    def test_known_tier_passes(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """Each member of ``_EXPLICIT_TIERS`` survives validation (with systemd-creds patched)."""
        monkeypatch.setattr("terok_sandbox.vault.store.systemd_creds.is_available", lambda: True)
        for tier in ("session-file", "keyring", "config", "systemd-creds"):
            _validate_passphrase_tier(tier)  # must not raise

    def test_unknown_tier_raises(self) -> None:
        """A typo fails closed with the allowed vocabulary in the message."""
        with pytest.raises(SystemExit, match="unknown --passphrase-tier"):
            _validate_passphrase_tier("session-fiel")

    def test_systemd_creds_unavailable_raises(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """Explicit ``systemd-creds`` on a host that doesn't have it fails closed."""
        monkeypatch.setattr("terok_sandbox.vault.store.systemd_creds.is_available", lambda: False)
        with pytest.raises(SystemExit, match="systemd-creds is unavailable"):
            _validate_passphrase_tier("systemd-creds")


class TestSetupRejectsBeforeMutation:
    """``_handle_sandbox_setup`` validates the tier before shield runs."""

    def test_bad_tier_rejected_before_shield_install(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """``run_shield_install_phase`` must never see a bogus tier."""
        # Boobytrap every host-mutating phase — if any of them runs we
        # know validation was too late.
        called = {"shield": 0, "vault": 0, "gate": 0, "clearance": 0}

        def _trap(name: str) -> object:
            def _inner(*_a: object, **_kw: object) -> bool:
                called[name] += 1
                return True

            return _inner

        monkeypatch.setattr(
            "terok_sandbox.commands.sandbox.run_shield_install_phase", _trap("shield")
        )
        monkeypatch.setattr(
            "terok_sandbox.commands.sandbox.run_vault_install_phase", _trap("vault")
        )
        monkeypatch.setattr("terok_sandbox.commands.sandbox.run_gate_install_phase", _trap("gate"))
        monkeypatch.setattr(
            "terok_sandbox.commands.sandbox.run_clearance_install_phase", _trap("clearance")
        )
        # The prereq report is read-only but it still talks to selinux —
        # patch it to a no-op so the test doesn't depend on the host.
        monkeypatch.setattr(
            "terok_sandbox.commands.sandbox.run_prereq_report",
            lambda _cfg: type("R", (), {"status": "OK"})(),
        )
        monkeypatch.setattr(
            "terok_sandbox.commands.sandbox.print_selinux_install_hint",
            lambda _result: None,
        )

        with pytest.raises(SystemExit, match="unknown --passphrase-tier"):
            _handle_sandbox_setup(passphrase_tier="bogus")

        assert called == {"shield": 0, "vault": 0, "gate": 0, "clearance": 0}

    def test_no_vault_skips_tier_check(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """``--no-vault`` makes the tier irrelevant — don't gate on it."""
        # When no_vault is set, the credentials phase never runs, so a
        # bogus tier value is just noise.  Refusing here would block the
        # documented "install everything *except* vault" escape hatch.
        monkeypatch.setattr(
            "terok_sandbox.commands.sandbox.run_shield_install_phase", lambda **_kw: True
        )
        monkeypatch.setattr(
            "terok_sandbox.commands.sandbox.run_gate_install_phase", lambda _cfg: True
        )
        monkeypatch.setattr(
            "terok_sandbox.commands.sandbox.run_clearance_install_phase", lambda: True
        )
        monkeypatch.setattr(
            "terok_sandbox.commands.sandbox.run_prereq_report",
            lambda _cfg: type("R", (), {"status": "OK"})(),
        )
        monkeypatch.setattr(
            "terok_sandbox.commands.sandbox.print_selinux_install_hint",
            lambda _result: None,
        )
        # write_stamp is imported lazily inside the handler; patch its
        # source module so the import resolves to a no-op.
        monkeypatch.setattr("terok_sandbox.setup_stamp.write_stamp", lambda: "fake-stamp")

        # Boobytrap the vault phases.  ``--no-vault`` MUST short-circuit
        # them entirely; if either fires it means the gate didn't hold
        # and the test should fail loudly instead of just appearing to
        # pass because we stubbed them benign.
        def _boom(*_a: object, **_kw: object) -> bool:
            raise RuntimeError("vault phase ran despite --no-vault")

        monkeypatch.setattr("terok_sandbox.commands.sandbox._run_credentials_setup_phase", _boom)
        monkeypatch.setattr("terok_sandbox.commands.sandbox.run_vault_install_phase", _boom)

        # Should not raise — the bogus tier is ignored under --no-vault.
        with patch("terok_sandbox.commands.sandbox.SandboxConfig"):
            _handle_sandbox_setup(passphrase_tier="bogus", no_vault=True)
