# SPDX-FileCopyrightText: 2026 Jiri Vyskocil
# SPDX-License-Identifier: Apache-2.0

"""Tests for the container health check protocol and sandbox-level diagnostics."""

from __future__ import annotations

import pytest

from terok_sandbox.doctor import (
    CheckVerdict,
    DoctorCheck,
    _make_plaintext_passphrase_warning_check,
    _make_shield_check,
    _make_ssh_signer_check,
    _make_token_broker_check,
    _make_vault_unlocked_check,
    sandbox_doctor_checks,
)

TOKEN_BROKER_PORT = 18731
SSH_SIGNER_PORT = 18732


class TestCheckVerdict:
    """CheckVerdict dataclass basics."""

    def test_default_fixable_is_false(self) -> None:
        v = CheckVerdict("ok", "all good")
        assert v.fixable is False

    def test_fixable_flag(self) -> None:
        v = CheckVerdict("error", "broken", fixable=True)
        assert v.fixable is True

    def test_frozen(self) -> None:
        v = CheckVerdict("ok", "fine")
        with pytest.raises(AttributeError):
            v.severity = "error"  # type: ignore[misc]


class TestDoctorCheck:
    """DoctorCheck dataclass basics."""

    def test_defaults(self) -> None:
        c = DoctorCheck(
            category="test",
            label="Test",
            probe_cmd=["true"],
            evaluate=lambda rc, out, err: CheckVerdict("ok", "ok"),
        )
        assert c.fix_cmd is None
        assert c.fix_description == ""
        assert c.host_side is False

    def test_host_side_check(self) -> None:
        c = DoctorCheck(
            category="shield",
            label="Shield",
            probe_cmd=[],
            evaluate=lambda rc, out, err: CheckVerdict("ok", "ok"),
            host_side=True,
        )
        assert c.host_side is True


class TestTokenBrokerCheck:
    """Token broker TCP reachability check."""

    def test_ok_on_success(self) -> None:
        check = _make_token_broker_check(TOKEN_BROKER_PORT)
        verdict = check.evaluate(0, "", "")
        assert verdict.severity == "ok"
        assert str(TOKEN_BROKER_PORT) in verdict.detail

    def test_error_on_failure(self) -> None:
        check = _make_token_broker_check(TOKEN_BROKER_PORT)
        verdict = check.evaluate(4, "", "connection refused")
        assert verdict.severity == "error"
        assert "unreachable" in verdict.detail

    def test_probe_cmd_uses_health_endpoint(self) -> None:
        check = _make_token_broker_check(TOKEN_BROKER_PORT)
        cmd_str = " ".join(check.probe_cmd)
        assert str(TOKEN_BROKER_PORT) in cmd_str
        assert "/-/health" in cmd_str
        assert "wget" in cmd_str

    def test_category_is_network(self) -> None:
        check = _make_token_broker_check(TOKEN_BROKER_PORT)
        assert check.category == "network"


class TestSSHSignerCheck:
    """SSH signer TCP reachability check."""

    def test_ok_on_success(self) -> None:
        check = _make_ssh_signer_check(SSH_SIGNER_PORT)
        verdict = check.evaluate(0, "", "")
        assert verdict.severity == "ok"
        assert str(SSH_SIGNER_PORT) in verdict.detail

    def test_error_on_failure(self) -> None:
        check = _make_ssh_signer_check(SSH_SIGNER_PORT)
        verdict = check.evaluate(1, "", "timeout")
        assert verdict.severity == "error"

    def test_probe_cmd_uses_nc(self) -> None:
        check = _make_ssh_signer_check(SSH_SIGNER_PORT)
        cmd_str = " ".join(check.probe_cmd)
        assert "nc" in cmd_str
        assert str(SSH_SIGNER_PORT) in cmd_str


class TestShieldCheck:
    """Shield state verification check.

    These tests exercise the ``evaluate`` callable in isolation by passing
    state strings via the *stdout* parameter.  This matches how the
    orchestrator (terok's ``container_doctor``) calls evaluate after
    resolving the actual shield state on the host.  The host_side flag
    means the orchestrator bypasses ``podman exec`` — it does NOT mean
    the evaluate function itself performs a side-effect.
    """

    def test_no_desired_state(self) -> None:
        check = _make_shield_check(None)
        verdict = check.evaluate(0, "", "")
        assert verdict.severity == "ok"
        assert "not managed" in verdict.detail

    def test_matching_state(self) -> None:
        check = _make_shield_check("up")
        verdict = check.evaluate(0, "up", "")
        assert verdict.severity == "ok"
        assert "matches" in verdict.detail

    def test_mismatched_state(self) -> None:
        check = _make_shield_check("up")
        verdict = check.evaluate(0, "down", "")
        assert verdict.severity == "warn"
        assert verdict.fixable is True
        assert "mismatch" in verdict.detail

    def test_host_side_flag(self) -> None:
        check = _make_shield_check("up")
        assert check.host_side is True

    def test_empty_probe_cmd(self) -> None:
        check = _make_shield_check("up")
        assert check.probe_cmd == []


class TestVaultUnlockedCheck:
    """Host-side check: passphrase resolves through *some* tier or vault stays locked."""

    def test_ok_when_resolution_chain_yields(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """Any tier returning a passphrase → ok verdict."""
        from terok_sandbox.credentials import encryption as enc

        monkeypatch.setattr(enc, "resolve_passphrase", lambda **_kw: "found-it")
        check = _make_vault_unlocked_check()
        verdict = check.evaluate(0, "", "")
        assert verdict.severity == "ok"
        assert "available" in verdict.detail

    def test_error_when_chain_empty(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """Every tier empty → actionable error verdict with the unlock hint."""
        from terok_sandbox.credentials import encryption as enc

        monkeypatch.setattr(enc, "resolve_passphrase", lambda **_kw: None)
        check = _make_vault_unlocked_check()
        verdict = check.evaluate(0, "", "")
        assert verdict.severity == "error"
        assert "vault is locked" in verdict.detail
        assert "vault unlock" in verdict.detail


class TestPlaintextPassphraseWarningCheck:
    """Host-side check (sandbox#282): visibility for plaintext-on-disk passphrase."""

    def test_ok_when_field_absent(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """No ``credentials.passphrase`` anywhere → silent ``ok`` verdict."""
        from terok_sandbox import paths

        monkeypatch.setattr(paths, "plaintext_passphrase_config_path", lambda: None)
        check = _make_plaintext_passphrase_warning_check()
        verdict = check.evaluate(0, "", "")
        assert verdict.severity == "ok"

    def test_warn_when_field_configured(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """Configured field → ``warn`` verdict naming the file + the safer-tier hint."""
        from pathlib import Path

        from terok_sandbox import paths

        seeded = Path("/etc/terok/config.yml")
        monkeypatch.setattr(paths, "plaintext_passphrase_config_path", lambda: seeded)
        check = _make_plaintext_passphrase_warning_check()
        verdict = check.evaluate(0, "", "")
        assert verdict.severity == "warn"
        assert str(seeded) in verdict.detail
        assert "plaintext" in verdict.detail
        # The fix description points the operator at a safer-tier verb.
        assert "vault unlock" in check.fix_description or "vault seal" in check.fix_description


class TestSandboxDoctorChecks:
    """Integration: sandbox_doctor_checks() assembly."""

    def test_all_checks_present(self) -> None:
        checks = sandbox_doctor_checks(
            token_broker_port=TOKEN_BROKER_PORT,
            ssh_signer_port=SSH_SIGNER_PORT,
            desired_shield_state="up",
        )
        labels = {c.label for c in checks}
        assert "Credentials DB passphrase" in labels
        assert "Plaintext passphrase" in labels
        assert "Token broker (TCP)" in labels
        assert "SSH signer (TCP)" in labels
        assert "Shield state" in labels
        assert len(checks) == 5

    def test_skips_broker_when_none(self) -> None:
        checks = sandbox_doctor_checks(
            token_broker_port=None,
            ssh_signer_port=SSH_SIGNER_PORT,
            desired_shield_state=None,
        )
        labels = {c.label for c in checks}
        assert "Token broker (TCP)" not in labels
        assert "SSH signer (TCP)" in labels

    def test_skips_ssh_signer_when_none(self) -> None:
        checks = sandbox_doctor_checks(
            token_broker_port=TOKEN_BROKER_PORT,
            ssh_signer_port=None,
            desired_shield_state=None,
        )
        labels = {c.label for c in checks}
        assert "SSH signer (TCP)" not in labels
        assert "Token broker (TCP)" in labels

    def test_minimal(self) -> None:
        """With no ports, only the always-on vault and shield checks remain."""
        checks = sandbox_doctor_checks(
            token_broker_port=None,
            ssh_signer_port=None,
            desired_shield_state=None,
        )
        categories = [c.category for c in checks]
        # Two vault checks now: unlocked-passphrase + plaintext warning.
        assert categories == ["vault", "vault", "shield"]

    def test_all_checks_are_doctor_check_instances(self) -> None:
        checks = sandbox_doctor_checks(
            token_broker_port=TOKEN_BROKER_PORT,
            ssh_signer_port=SSH_SIGNER_PORT,
            desired_shield_state="down",
        )
        for check in checks:
            assert isinstance(check, DoctorCheck)
