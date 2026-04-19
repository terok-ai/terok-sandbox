# SPDX-FileCopyrightText: 2026 Jiri Vyskocil
# SPDX-License-Identifier: Apache-2.0

"""Tests for command handlers: gate-stop branches, vault install/uninstall,
shield-status setup hint, doctor, and SSH key removal helpers.
"""

from __future__ import annotations

import subprocess
from collections.abc import Iterator
from contextlib import contextmanager
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from terok_sandbox.commands import (
    _handle_doctor,
    _handle_gate_stop,
    _handle_shield_status,
    _handle_vault_install,
    _handle_vault_uninstall,
)
from terok_sandbox.config import SandboxConfig
from terok_sandbox.doctor import CheckVerdict, DoctorCheck
from terok_sandbox.gate.lifecycle import GateServerManager, GateServerStatus
from terok_sandbox.vault.lifecycle import VaultManager

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _gate_status(mode: str) -> GateServerStatus:
    """Build a minimal GateServerStatus with the given mode."""
    return GateServerStatus(mode=mode, running=False, port=9418)


def _make_cfg(tmp_path: Path) -> SandboxConfig:
    """Build a SandboxConfig rooted at *tmp_path* (TCP mode → ports auto-allocated)."""
    return SandboxConfig(
        state_dir=tmp_path / "state",
        runtime_dir=tmp_path / "run",
        config_dir=tmp_path / "config",
        vault_dir=tmp_path / "vault",
    )


# ---------------------------------------------------------------------------
# _handle_gate_stop "not running" branch (line 106)
# ---------------------------------------------------------------------------


class TestHandleGateStopNotRunning:
    """When status.mode is neither 'systemd' nor 'daemon', print the idle message."""

    def test_prints_not_running(self, capsys: pytest.CaptureFixture[str]) -> None:
        with patch.object(GateServerManager, "get_status", return_value=_gate_status("none")):
            _handle_gate_stop()
        out = capsys.readouterr().out
        assert "not running" in out


# ---------------------------------------------------------------------------
# _handle_shield_status setup-hint branch (line 155)
# ---------------------------------------------------------------------------


class TestHandleShieldStatusHint:
    """When shield needs setup, the hint is written to stderr."""

    def test_setup_hint_emitted_on_stderr(self, capsys: pytest.CaptureFixture[str]) -> None:
        env = MagicMock(hooks="missing", health="degraded", needs_setup=True, setup_hint="Run X.")
        cfg = {"mode": "hook", "profiles": ["dev-standard"], "audit_enabled": True}
        # The handler imports check_environment + status from .shield at runtime,
        # so patching the shield module is what takes effect.
        with (
            patch("terok_sandbox.shield.check_environment", return_value=env),
            patch("terok_sandbox.shield.status", return_value=cfg),
        ):
            _handle_shield_status()
        captured = capsys.readouterr()
        assert "Run X." in captured.err

    def test_no_hint_when_not_needed(self, capsys: pytest.CaptureFixture[str]) -> None:
        env = MagicMock(hooks="ok", health="ok", needs_setup=False, setup_hint="should not show")
        cfg = {"mode": "hook", "profiles": [], "audit_enabled": False}
        with (
            patch("terok_sandbox.shield.check_environment", return_value=env),
            patch("terok_sandbox.shield.status", return_value=cfg),
        ):
            _handle_shield_status()
        assert "should not show" not in capsys.readouterr().err


# ---------------------------------------------------------------------------
# _handle_vault_install / _handle_vault_uninstall — systemd-unavailable branch
# ---------------------------------------------------------------------------


class TestHandleVaultInstall:
    """Installer fails loudly when the systemd user session is unavailable."""

    def test_install_systemd_unavailable_exits_1(self, capsys: pytest.CaptureFixture[str]) -> None:
        with patch.object(VaultManager, "is_systemd_available", return_value=False):
            with pytest.raises(SystemExit) as exc:
                _handle_vault_install()
        assert exc.value.code == 1
        assert "systemd" in capsys.readouterr().out

    def test_install_systemd_available_calls_install(
        self, capsys: pytest.CaptureFixture[str]
    ) -> None:
        with (
            patch.object(VaultManager, "is_systemd_available", return_value=True),
            patch.object(VaultManager, "install_systemd_units") as install,
        ):
            _handle_vault_install()
        install.assert_called_once()
        assert "installed" in capsys.readouterr().out.lower()


class TestHandleVaultUninstall:
    """Uninstaller mirrors install: fail loudly when systemd missing."""

    def test_uninstall_systemd_unavailable_exits_1(
        self, capsys: pytest.CaptureFixture[str]
    ) -> None:
        with patch.object(VaultManager, "is_systemd_available", return_value=False):
            with pytest.raises(SystemExit) as exc:
                _handle_vault_uninstall()
        assert exc.value.code == 1
        assert "Nothing to uninstall" in capsys.readouterr().out

    def test_uninstall_systemd_available_calls_uninstall(
        self, capsys: pytest.CaptureFixture[str]
    ) -> None:
        with (
            patch.object(VaultManager, "is_systemd_available", return_value=True),
            patch.object(VaultManager, "uninstall_systemd_units") as un,
        ):
            _handle_vault_uninstall()
        un.assert_called_once()
        assert "removed" in capsys.readouterr().out.lower()


# ---------------------------------------------------------------------------
# _handle_doctor — the largest uncovered chunk (lines 975-1023)
# ---------------------------------------------------------------------------


def _make_check(
    *,
    label: str = "test",
    severity: str,
    detail: str = "",
    host_side: bool = False,
    probe_cmd: list[str] | None = None,
) -> DoctorCheck:
    """Build a DoctorCheck whose evaluate returns a fixed verdict."""
    return DoctorCheck(
        category="test",
        label=label,
        probe_cmd=probe_cmd or ["true"],
        evaluate=lambda rc, out, err: CheckVerdict(severity=severity, detail=detail),
        host_side=host_side,
    )


@contextmanager
def _doctor_patches(
    checks: list[DoctorCheck],
    *,
    subprocess_side_effect: object = None,
) -> Iterator[MagicMock]:
    """Patch ``sandbox_doctor_checks``, ``subprocess.run``, and VaultManager ports.

    Yields the ``subprocess.run`` mock so tests can inspect calls or override
    return_value.  By default the mock returns a successful CompletedProcess.
    Pass *subprocess_side_effect* (e.g. ``FileNotFoundError``, a
    ``TimeoutExpired`` instance) to simulate probe failures.
    """
    with (
        patch("terok_sandbox.doctor.sandbox_doctor_checks", return_value=checks),
        patch("subprocess.run") as run,
        patch.object(VaultManager, "token_broker_port", new=1),
        patch.object(VaultManager, "ssh_signer_port", new=2),
    ):
        if subprocess_side_effect is not None:
            run.side_effect = subprocess_side_effect
        else:
            run.return_value = subprocess.CompletedProcess(
                args=[], returncode=0, stdout="", stderr=""
            )
        yield run


class TestHandleDoctor:
    """The standalone doctor runs each check and exits per worst severity."""

    def test_all_ok_exits_normally(self, capsys: pytest.CaptureFixture[str]) -> None:
        checks = [_make_check(label="A", severity="ok", detail="fine")]
        with _doctor_patches(checks):
            _handle_doctor()  # must not raise
        out = capsys.readouterr().out
        assert "A" in out and "ok" in out

    def test_warn_exits_1(self, capsys: pytest.CaptureFixture[str]) -> None:
        checks = [_make_check(label="W", severity="warn", detail="be careful")]
        with _doctor_patches(checks), pytest.raises(SystemExit) as exc:
            _handle_doctor()
        assert exc.value.code == 1
        assert "WARN" in capsys.readouterr().out

    def test_error_exits_2(self, capsys: pytest.CaptureFixture[str]) -> None:
        checks = [
            _make_check(label="A", severity="ok"),
            _make_check(label="B", severity="error", detail="boom"),
        ]
        with _doctor_patches(checks), pytest.raises(SystemExit) as exc:
            _handle_doctor()
        assert exc.value.code == 2
        assert "ERROR" in capsys.readouterr().out

    def test_host_side_check_skips_subprocess(self) -> None:
        """host_side=True checks call evaluate(0,'','') directly, no subprocess."""
        checks = [_make_check(label="H", severity="ok", host_side=True)]
        with _doctor_patches(checks) as run:
            _handle_doctor()
        run.assert_not_called()

    def test_probe_unavailable_yields_unavailable_verdict(self) -> None:
        """FileNotFoundError from probe → evaluate is called with rc=1 and 'unavailable'."""
        captured: dict = {}

        def evaluate(rc: int, out: str, err: str) -> CheckVerdict:
            captured["rc"], captured["err"] = rc, err
            return CheckVerdict(severity="warn", detail="probe missing")

        check = DoctorCheck(
            category="net", label="P", probe_cmd=["nonexistent-binary"], evaluate=evaluate
        )
        with (
            _doctor_patches([check], subprocess_side_effect=FileNotFoundError),
            pytest.raises(SystemExit),
        ):
            _handle_doctor()
        assert captured["rc"] == 1
        assert "unavailable" in captured["err"]

    def test_probe_timeout_yields_unavailable_verdict(self) -> None:
        """TimeoutExpired from probe → evaluate is called with rc=1 and 'unavailable'."""
        captured: dict = {}

        def evaluate(rc: int, out: str, err: str) -> CheckVerdict:
            captured["rc"], captured["err"] = rc, err
            return CheckVerdict(severity="warn", detail="t/o")

        check = DoctorCheck(category="net", label="P", probe_cmd=["sleep", "9"], evaluate=evaluate)
        timeout = subprocess.TimeoutExpired("sleep", 5)
        with _doctor_patches([check], subprocess_side_effect=timeout), pytest.raises(SystemExit):
            _handle_doctor()
        assert captured["rc"] == 1
        assert "unavailable" in captured["err"]

    def test_check_without_probe_cmd_calls_evaluate_directly(self) -> None:
        """A check with no probe_cmd and host_side=False still gets evaluate(0,'','')."""
        called: list[tuple] = []

        def evaluate(rc: int, out: str, err: str) -> CheckVerdict:
            called.append((rc, out, err))
            return CheckVerdict(severity="ok", detail="")

        check = DoctorCheck(category="x", label="L", probe_cmd=[], evaluate=evaluate)
        with _doctor_patches([check]) as run:
            _handle_doctor()
        run.assert_not_called()
        assert called == [(0, "", "")]
