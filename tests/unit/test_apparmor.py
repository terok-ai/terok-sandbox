# SPDX-FileCopyrightText: 2026 Jiri Vyskocil
# SPDX-License-Identifier: Apache-2.0

"""Tests for AppArmor profile-state probes and the status decision tree."""

from __future__ import annotations

import unittest.mock
from pathlib import Path

import pytest

from terok_sandbox._util._apparmor import (
    CONFINED_PROFILES,
    ApparmorStatus,
    check_status,
    is_apparmor_enabled,
    is_profile_loaded,
    loaded_confined_profiles,
    loaded_profiles,
    missing_policy_tools,
    profile_mode,
)


@pytest.fixture(autouse=True)
def _reset_check_status_cache() -> None:
    """Clear the lru_cache on `check_status` between tests so each one
    sees the patched /sys state independently.
    """
    check_status.cache_clear()


def _stage_apparmor(tmp_path: Path, profiles_text: str | None) -> dict[str, Path]:
    """Build a fake securityfs layout under *tmp_path* and return the
    paths to use as patch targets.  *profiles_text* of ``None`` means
    the ``profiles`` file does not exist (non-AppArmor host)."""
    secfs = tmp_path / "apparmor"
    secfs.mkdir()
    profiles = secfs / "profiles"
    if profiles_text is not None:
        profiles.write_text(profiles_text)
    return {"secfs": secfs, "profiles": profiles}


def _patch_paths(secfs: Path, profiles: Path) -> tuple:
    """Return the two `unittest.mock.patch` context managers used to
    redirect the module-level path constants — same shape as the
    existing _selinux tests use for ``_ENFORCE_PATH``."""
    return (
        unittest.mock.patch("terok_sandbox._util._apparmor._SECFS_ROOT", secfs),
        unittest.mock.patch("terok_sandbox._util._apparmor._PROFILES_FILE", profiles),
    )


# ---------- Detection ----------


class TestIsApparmorEnabled:
    """Presence of ``/sys/kernel/security/apparmor`` is the canonical signal."""

    def test_true_when_secfs_dir_present(self, tmp_path: Path) -> None:
        paths = _stage_apparmor(tmp_path, "")
        with unittest.mock.patch("terok_sandbox._util._apparmor._SECFS_ROOT", paths["secfs"]):
            assert is_apparmor_enabled() is True

    def test_false_on_non_apparmor_host(self, tmp_path: Path) -> None:
        nonexistent = tmp_path / "no-such-dir"
        with unittest.mock.patch("terok_sandbox._util._apparmor._SECFS_ROOT", nonexistent):
            assert is_apparmor_enabled() is False


class TestMissingPolicyTools:
    """Only ``apparmor_parser`` is required; the others are optional helpers."""

    def test_empty_when_present(self) -> None:
        with unittest.mock.patch("shutil.which", return_value="/usr/sbin/apparmor_parser"):
            assert missing_policy_tools() == []

    def test_lists_apparmor_parser_when_absent(self) -> None:
        with unittest.mock.patch("shutil.which", return_value=None):
            assert missing_policy_tools() == ["apparmor_parser"]


# ---------- Profile-state parsing ----------


_PROFILES_SAMPLE = (
    "terok-gate (complain)\nterok-vault (enforce)\n/usr/bin/man (enforce)\nlsb_release (complain)\n"
)


class TestLoadedProfiles:
    """``/sys/kernel/security/apparmor/profiles`` is line-oriented:
    ``<name> (<mode>)`` per loaded profile."""

    def test_parses_name_and_mode(self, tmp_path: Path) -> None:
        paths = _stage_apparmor(tmp_path, _PROFILES_SAMPLE)
        with unittest.mock.patch("terok_sandbox._util._apparmor._PROFILES_FILE", paths["profiles"]):
            assert loaded_profiles() == {
                "terok-gate": "complain",
                "terok-vault": "enforce",
                "/usr/bin/man": "enforce",
                "lsb_release": "complain",
            }

    def test_handles_path_named_profiles(self, tmp_path: Path) -> None:
        """Profile names may contain slashes (path-attached profiles)."""
        paths = _stage_apparmor(tmp_path, "/usr/bin/foo (enforce)\n")
        with unittest.mock.patch("terok_sandbox._util._apparmor._PROFILES_FILE", paths["profiles"]):
            assert loaded_profiles() == {"/usr/bin/foo": "enforce"}

    def test_returns_empty_on_missing_file(self, tmp_path: Path) -> None:
        """Non-AppArmor host or absent securityfs node — degrade to empty."""
        nonexistent = tmp_path / "no-such-file"
        with unittest.mock.patch("terok_sandbox._util._apparmor._PROFILES_FILE", nonexistent):
            assert loaded_profiles() == {}

    def test_skips_malformed_lines(self, tmp_path: Path) -> None:
        """Lines that don't match ``<name> (<mode>)`` are silently dropped —
        defensive against future kernel format changes."""
        paths = _stage_apparmor(tmp_path, "terok-gate (enforce)\nbogus_line_no_paren\n")
        with unittest.mock.patch("terok_sandbox._util._apparmor._PROFILES_FILE", paths["profiles"]):
            assert loaded_profiles() == {"terok-gate": "enforce"}


class TestProfileQueries:
    """`is_profile_loaded` and `profile_mode` are thin convenience wrappers."""

    def test_is_profile_loaded_true(self, tmp_path: Path) -> None:
        paths = _stage_apparmor(tmp_path, _PROFILES_SAMPLE)
        with unittest.mock.patch("terok_sandbox._util._apparmor._PROFILES_FILE", paths["profiles"]):
            assert is_profile_loaded("terok-gate") is True

    def test_is_profile_loaded_false(self, tmp_path: Path) -> None:
        paths = _stage_apparmor(tmp_path, _PROFILES_SAMPLE)
        with unittest.mock.patch("terok_sandbox._util._apparmor._PROFILES_FILE", paths["profiles"]):
            assert is_profile_loaded("nonexistent") is False

    def test_profile_mode_returns_mode(self, tmp_path: Path) -> None:
        paths = _stage_apparmor(tmp_path, _PROFILES_SAMPLE)
        with unittest.mock.patch("terok_sandbox._util._apparmor._PROFILES_FILE", paths["profiles"]):
            assert profile_mode("terok-vault") == "enforce"

    def test_profile_mode_returns_none_when_absent(self, tmp_path: Path) -> None:
        paths = _stage_apparmor(tmp_path, _PROFILES_SAMPLE)
        with unittest.mock.patch("terok_sandbox._util._apparmor._PROFILES_FILE", paths["profiles"]):
            assert profile_mode("nonexistent") is None


class TestLoadedConfinedProfiles:
    """Filters `loaded_profiles` against `CONFINED_PROFILES`."""

    def test_returns_only_terok_profiles(self, tmp_path: Path) -> None:
        paths = _stage_apparmor(tmp_path, _PROFILES_SAMPLE)
        with unittest.mock.patch("terok_sandbox._util._apparmor._PROFILES_FILE", paths["profiles"]):
            assert set(loaded_confined_profiles()) == {"terok-gate", "terok-vault"}

    def test_returns_empty_when_none_loaded(self, tmp_path: Path) -> None:
        paths = _stage_apparmor(tmp_path, "/usr/bin/man (enforce)\n")
        with unittest.mock.patch("terok_sandbox._util._apparmor._PROFILES_FILE", paths["profiles"]):
            assert loaded_confined_profiles() == ()


# ---------- Decision-tree status ----------


class TestCheckStatus:
    """Each branch of the `check_status` decision tree, in order."""

    def test_not_applicable_when_apparmor_off(self, tmp_path: Path) -> None:
        nonexistent = tmp_path / "no-secfs"
        with unittest.mock.patch("terok_sandbox._util._apparmor._SECFS_ROOT", nonexistent):
            assert check_status().status == ApparmorStatus.NOT_APPLICABLE

    def test_profiles_missing_when_apparmor_on_and_terok_absent(self, tmp_path: Path) -> None:
        paths = _stage_apparmor(tmp_path, "/usr/bin/man (enforce)\n")
        with (
            _patch_paths(paths["secfs"], paths["profiles"])[0],
            _patch_paths(paths["secfs"], paths["profiles"])[1],
        ):
            result = check_status()
        assert result.status == ApparmorStatus.PROFILES_MISSING

    def test_profiles_partial_when_only_some_loaded(self, tmp_path: Path) -> None:
        """Loaded subset of CONFINED_PROFILES → PROFILES_PARTIAL.
        Catches a half-broken install (one profile load failed)."""
        paths = _stage_apparmor(tmp_path, "terok-gate (complain)\n")
        with (
            _patch_paths(paths["secfs"], paths["profiles"])[0],
            _patch_paths(paths["secfs"], paths["profiles"])[1],
        ):
            result = check_status()
        assert result.status == ApparmorStatus.PROFILES_PARTIAL
        assert result.loaded == ("terok-gate",)

    def test_ok_complain_when_all_loaded_in_complain(self, tmp_path: Path) -> None:
        paths = _stage_apparmor(tmp_path, "terok-gate (complain)\nterok-vault (complain)\n")
        with (
            _patch_paths(paths["secfs"], paths["profiles"])[0],
            _patch_paths(paths["secfs"], paths["profiles"])[1],
        ):
            assert check_status().status == ApparmorStatus.OK_COMPLAIN

    def test_ok_enforce_when_all_loaded_in_enforce(self, tmp_path: Path) -> None:
        paths = _stage_apparmor(tmp_path, "terok-gate (enforce)\nterok-vault (enforce)\n")
        with (
            _patch_paths(paths["secfs"], paths["profiles"])[0],
            _patch_paths(paths["secfs"], paths["profiles"])[1],
        ):
            assert check_status().status == ApparmorStatus.OK_ENFORCE

    def test_ok_complain_when_any_in_complain(self, tmp_path: Path) -> None:
        """Mixed enforce + complain → not yet OK_ENFORCE.  A single
        complain profile downgrades the aggregate verdict — the soak
        posture is "all enforce, or it's still soaking"."""
        paths = _stage_apparmor(tmp_path, "terok-gate (complain)\nterok-vault (enforce)\n")
        with (
            _patch_paths(paths["secfs"], paths["profiles"])[0],
            _patch_paths(paths["secfs"], paths["profiles"])[1],
        ):
            assert check_status().status == ApparmorStatus.OK_COMPLAIN


# ---------- Constants ----------


class TestConfinedProfiles:
    """The shipped profile names match what terok hardening install writes."""

    def test_exposes_gate_and_vault(self) -> None:
        assert set(CONFINED_PROFILES) == {"terok-gate", "terok-vault"}
