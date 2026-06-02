# SPDX-FileCopyrightText: 2026 Jiri Vyskocil
# SPDX-License-Identifier: Apache-2.0

"""Unit tests for terok_sandbox._util._apparmor (dnsmasq profile detection)."""

from __future__ import annotations

from pathlib import Path

import pytest

from terok_sandbox import _setup
from terok_sandbox._util import _apparmor
from terok_sandbox._util._apparmor import (
    AppArmorCheckResult,
    AppArmorStatus,
    check_status,
    install_command,
)
from terok_sandbox.paths import namespace_state_dir


def _arrange(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    *,
    apparmor: bool = True,
    dnsmasq: bool = True,
    profile: bool = True,
    addendum: bool = False,
) -> None:
    """Point the module's sysfs/profile probes at a tmp fixture tree."""
    enabled = tmp_path / "enabled"
    enabled.write_text("Y\n" if apparmor else "N\n")
    monkeypatch.setattr(_apparmor, "_APPARMOR_ENABLED", enabled)
    monkeypatch.setattr(
        _apparmor.shutil, "which", lambda _n: "/usr/sbin/dnsmasq" if dnsmasq else None
    )
    prof = tmp_path / "etc" / "apparmor.d" / "dnsmasq"
    if profile:
        prof.parent.mkdir(parents=True, exist_ok=True)
        prof.write_text("# dnsmasq profile\n")
        if addendum:
            local = prof.parent / "local" / "dnsmasq"
            local.parent.mkdir(parents=True, exist_ok=True)
            local.write_text("# >>> terok-shield apparmor (managed) >>>\nowner x r,\n")
    monkeypatch.setattr(_apparmor, "_DNSMASQ_PROFILES", (prof,))


def test_not_applicable_without_apparmor(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    """AppArmor disabled → NOT_APPLICABLE."""
    _arrange(monkeypatch, tmp_path, apparmor=False)
    assert check_status().status is AppArmorStatus.NOT_APPLICABLE


def test_not_applicable_without_dnsmasq(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    """dnsmasq absent → NOT_APPLICABLE (the dnsmasq tier wouldn't be used)."""
    _arrange(monkeypatch, tmp_path, dnsmasq=False)
    assert check_status().status is AppArmorStatus.NOT_APPLICABLE


def test_not_applicable_without_profile(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    """No stock dnsmasq AppArmor profile → NOT_APPLICABLE."""
    _arrange(monkeypatch, tmp_path, profile=False)
    assert check_status().status is AppArmorStatus.NOT_APPLICABLE


def test_profile_missing_without_addendum(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    """AppArmor + dnsmasq + stock profile but no terok addendum → PROFILE_MISSING."""
    _arrange(monkeypatch, tmp_path, addendum=False)
    assert check_status().status is AppArmorStatus.PROFILE_MISSING


def test_ok_when_addendum_installed(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    """The terok addendum present in the local include → OK."""
    _arrange(monkeypatch, tmp_path, addendum=True)
    assert check_status().status is AppArmorStatus.OK


def test_install_command_shape(tmp_path: Path) -> None:
    """install_command renders a sudo invocation with the script and the state root."""
    root = tmp_path / "sandbox-live"
    cmd = install_command(root)
    assert cmd.startswith("sudo bash ")
    assert "install_profile.sh" in cmd
    assert str(root) in cmd


def test_is_apparmor_enabled_false_when_sysfs_absent(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """A missing sysfs node (AppArmor not in the kernel) reads as disabled, not an error."""
    monkeypatch.setattr(_apparmor, "_APPARMOR_ENABLED", tmp_path / "absent")
    assert _apparmor.is_apparmor_enabled() is False


# ── setup reporting glue (terok_sandbox._setup) ─────────────────────────


def _patch_status(monkeypatch: pytest.MonkeyPatch, status: AppArmorStatus) -> None:
    """Force ``_setup.check_apparmor_status`` to report *status*."""
    monkeypatch.setattr(_setup, "check_apparmor_status", lambda: AppArmorCheckResult(status))


def test_report_apparmor_silent_when_not_applicable(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    """Hosts with no dnsmasq AppArmor profile get no stage line at all."""
    _patch_status(monkeypatch, AppArmorStatus.NOT_APPLICABLE)
    result = _setup._report_apparmor()
    assert result.status is AppArmorStatus.NOT_APPLICABLE
    assert capsys.readouterr().out == ""


def test_report_apparmor_ok_marks_installed(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    """With the addendum present the stage line reports it installed."""
    _patch_status(monkeypatch, AppArmorStatus.OK)
    _setup._report_apparmor()
    out = capsys.readouterr().out
    assert "AppArmor profile" in out
    assert "installed" in out


def test_report_apparmor_missing_shows_install_command(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    """A confined-but-unpatched host gets the installer invocation in its stage line."""
    _patch_status(monkeypatch, AppArmorStatus.PROFILE_MISSING)
    _setup._report_apparmor()
    assert "install_profile.sh" in capsys.readouterr().out


def test_install_hint_only_when_profile_missing(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    """The end-of-setup hint block fires for PROFILE_MISSING and stays quiet otherwise."""
    _patch_status(monkeypatch, AppArmorStatus.PROFILE_MISSING)
    _setup.print_apparmor_install_hint()
    out = capsys.readouterr().out
    assert "AppArmor profile recommended" in out
    assert "install_profile.sh" in out

    for quiet in (AppArmorStatus.OK, AppArmorStatus.NOT_APPLICABLE):
        _patch_status(monkeypatch, quiet)
        _setup.print_apparmor_install_hint()
        assert capsys.readouterr().out == ""


def test_apparmor_state_root_is_sandbox_live() -> None:
    """The reported state root is the conventional sandbox-live namespace dir."""
    assert _setup._apparmor_state_root() == namespace_state_dir("sandbox-live")
