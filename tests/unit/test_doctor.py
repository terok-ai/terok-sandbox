# SPDX-FileCopyrightText: 2026 Jiri Vyskocil
# SPDX-License-Identifier: Apache-2.0

"""Tests for the container health check protocol and sandbox-level diagnostics."""

from __future__ import annotations

import pytest

from terok_sandbox.doctor import (
    CheckVerdict,
    DoctorCheck,
    _make_shield_check,
    _make_ssh_signer_check,
    _make_token_broker_check,
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


class TestSandboxDoctorChecks:
    """Integration: sandbox_doctor_checks() assembly."""

    def test_all_checks_present(self) -> None:
        checks = sandbox_doctor_checks(
            token_broker_port=TOKEN_BROKER_PORT,
            ssh_signer_port=SSH_SIGNER_PORT,
            desired_shield_state="up",
        )
        labels = {c.label for c in checks}
        assert "Token broker (TCP)" in labels
        assert "SSH signer (TCP)" in labels
        assert "Shield state" in labels
        assert len(checks) == 3

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
        """With no ports, only shield check remains."""
        checks = sandbox_doctor_checks(
            token_broker_port=None,
            ssh_signer_port=None,
            desired_shield_state=None,
        )
        assert len(checks) == 1
        assert checks[0].category == "shield"

    def test_all_checks_are_doctor_check_instances(self) -> None:
        checks = sandbox_doctor_checks(
            token_broker_port=TOKEN_BROKER_PORT,
            ssh_signer_port=SSH_SIGNER_PORT,
            desired_shield_state="down",
        )
        for check in checks:
            assert isinstance(check, DoctorCheck)
