# SPDX-FileCopyrightText: 2026 Jiri Vyskocil
# SPDX-License-Identifier: Apache-2.0

"""Tests for command handlers: doctor and SSH key removal helpers."""

from __future__ import annotations

import subprocess
from collections.abc import Iterator
from contextlib import contextmanager
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from terok_sandbox.commands import _handle_doctor
from terok_sandbox.config import SandboxConfig
from terok_sandbox.doctor import CheckVerdict, DoctorCheck

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_cfg(tmp_path: Path) -> SandboxConfig:
    """Build a SandboxConfig rooted at *tmp_path* (TCP mode → ports auto-allocated)."""
    return SandboxConfig(
        state_dir=tmp_path / "state",
        runtime_dir=tmp_path / "run",
        config_dir=tmp_path / "config",
        vault_dir=tmp_path / "vault",
    )


# The gate-stop / shield-status handler tests that lived here covered the
# retired host gate daemon and a hand-rolled shield-status verb that no
# longer exist; sandbox consumes shield's registry verb directly now.


# The per-container-supervisor refactor (May 2026) removed the
# host-global vault daemon, so ``vault install`` / ``vault uninstall``
# are gone too — the surrounding install/uninstall tests live in
# test_setup_phases.py for the supervisor / shield / gate / legacy
# cleanup phases that remain.


# ---------------------------------------------------------------------------
# _handle_doctor — the largest uncovered chunk
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
    """Patch ``sandbox_doctor_checks`` and ``subprocess.run``.

    Yields the ``subprocess.run`` mock so tests can inspect calls or override
    return_value.  By default the mock returns a successful CompletedProcess.
    Pass *subprocess_side_effect* (e.g. ``FileNotFoundError``, a
    ``TimeoutExpired`` instance) to simulate probe failures.

    ``make_recovery_acknowledged_check`` is appended to the standalone
    doctor list outside the ``sandbox_doctor_checks`` bundle (it's
    host-only and would otherwise duplicate per-task under terok's
    sickbay).  We stub it to a benign ok check so tests that pin the
    iteration loop's behaviour aren't perturbed by the marker's host
    state.
    """
    benign_recovery_check = _make_check(
        label="Recovery key acknowledged", severity="ok", host_side=True
    )
    with (
        patch("terok_sandbox.doctor.sandbox_doctor_checks", return_value=checks),
        patch(
            "terok_sandbox.doctor.make_recovery_acknowledged_check",
            return_value=benign_recovery_check,
        ),
        patch("subprocess.run") as run,
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
