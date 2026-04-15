# SPDX-FileCopyrightText: 2026 Jiri Vyskocil
# SPDX-License-Identifier: Apache-2.0

"""Tests for SELinux socket labeling and policy management."""

from __future__ import annotations

import subprocess
import unittest.mock
from pathlib import Path

import pytest

from terok_sandbox._util._selinux import (
    SELINUX_SOCKET_TYPE,
    _try_getsockcreatecon,
    _try_setsockcreatecon,
    install_policy,
    is_policy_installed,
    is_selinux_enabled,
    is_selinux_enforcing,
    policy_source_path,
    socket_selinux_context,
    uninstall_policy,
)
from tests.constants import MOCK_BASE

MOCK_ENFORCE_PATH = MOCK_BASE / "selinux" / "enforce"


# ---------- Detection ----------


class TestIsSelinuxEnforcing:
    """Verify SELinux enforcement detection via sysfs."""

    def test_enforcing(self, tmp_path: Path) -> None:
        """Enforcing = sysfs node reads '1'."""
        enforce = tmp_path / "enforce"
        enforce.write_text("1\n")
        with unittest.mock.patch("terok_sandbox._util._selinux._ENFORCE_PATH", enforce):
            assert is_selinux_enforcing() is True

    def test_permissive(self, tmp_path: Path) -> None:
        """Permissive = sysfs node reads '0'."""
        enforce = tmp_path / "enforce"
        enforce.write_text("0\n")
        with unittest.mock.patch("terok_sandbox._util._selinux._ENFORCE_PATH", enforce):
            assert is_selinux_enforcing() is False

    def test_missing(self) -> None:
        """Non-SELinux system = sysfs node absent."""
        with unittest.mock.patch(
            "terok_sandbox._util._selinux._ENFORCE_PATH",
            MOCK_ENFORCE_PATH,
        ):
            assert is_selinux_enforcing() is False


class TestIsSelinuxEnabled:
    """Verify SELinux availability detection."""

    def test_enabled(self, tmp_path: Path) -> None:
        """SELinux present when sysfs node exists."""
        enforce = tmp_path / "enforce"
        enforce.write_text("0\n")
        with unittest.mock.patch("terok_sandbox._util._selinux._ENFORCE_PATH", enforce):
            assert is_selinux_enabled() is True

    def test_disabled(self) -> None:
        """SELinux absent when sysfs node missing."""
        with unittest.mock.patch(
            "terok_sandbox._util._selinux._ENFORCE_PATH",
            MOCK_ENFORCE_PATH,
        ):
            assert is_selinux_enabled() is False


# ---------- Policy detection ----------


class TestIsPolicyInstalled:
    """Verify policy module detection."""

    def test_installed(self) -> None:
        """Module present in semodule -l output."""
        result = unittest.mock.MagicMock(stdout="terok_socket\nother_module\n")
        with unittest.mock.patch("subprocess.run", return_value=result):
            assert is_policy_installed() is True

    def test_not_installed(self) -> None:
        """Module absent from semodule output."""
        result = unittest.mock.MagicMock(stdout="other_module\n")
        with unittest.mock.patch("subprocess.run", return_value=result):
            assert is_policy_installed() is False

    def test_semodule_missing(self) -> None:
        """semodule binary not found."""
        with unittest.mock.patch("subprocess.run", side_effect=FileNotFoundError):
            assert is_policy_installed() is False

    def test_partial_name_no_match(self) -> None:
        """A module whose name contains the target as a substring must not match."""
        result = unittest.mock.MagicMock(stdout="terok_socket_extra\n")
        with unittest.mock.patch("subprocess.run", return_value=result):
            assert is_policy_installed() is False


# ---------- Policy installation ----------


class TestInstallPolicy:
    """Verify the compile-and-install flow."""

    def test_missing_tool_fails(self) -> None:
        """Raise SystemExit when checkmodule is not on PATH."""
        with unittest.mock.patch("shutil.which", return_value=None):
            with pytest.raises(SystemExit, match="checkmodule"):
                install_policy()

    def test_install_calls_semodule(self, tmp_path: Path) -> None:
        """Full compile + install sequence."""
        te_file = tmp_path / "terok_socket.te"
        te_file.write_text("policy_module(terok_socket, 1.0)\n")

        calls: list[list[str]] = []

        def _fake_run(cmd: list[str], **_kw: object) -> subprocess.CompletedProcess[str]:
            calls.append(cmd)
            # Create expected output files so unlink works
            if cmd[0] == "checkmodule":
                Path(cmd[3]).touch()
            elif cmd[0] == "semodule_package":
                Path(cmd[2]).touch()
            return subprocess.CompletedProcess(cmd, 0)

        with (
            unittest.mock.patch("shutil.which", return_value="/usr/bin/tool"),
            unittest.mock.patch(
                "terok_sandbox._util._selinux.policy_source_path",
                return_value=te_file,
            ),
            unittest.mock.patch("subprocess.run", side_effect=_fake_run),
        ):
            install_policy()

        # checkmodule → semodule_package → semodule -i
        assert calls[0][0] == "checkmodule"
        assert calls[1][0] == "semodule_package"
        assert calls[2][:2] == ["semodule", "-i"]

    def test_readonly_source_uses_tempdir(self, tmp_path: Path) -> None:
        """When .te lives in read-only dir, artifacts go to a temp dir."""
        ro_dir = tmp_path / "readonly"
        ro_dir.mkdir()
        te_file = ro_dir / "terok_socket.te"
        te_file.write_text("policy_module(terok_socket, 1.0)\n")
        ro_dir.chmod(0o555)

        artifact_dirs: list[str] = []

        def _fake_run(cmd: list[str], **_kw: object) -> subprocess.CompletedProcess[str]:
            if cmd[0] == "checkmodule":
                artifact_dirs.append(str(Path(cmd[3]).parent))
                Path(cmd[3]).touch()
            elif cmd[0] == "semodule_package":
                Path(cmd[2]).touch()
            return subprocess.CompletedProcess(cmd, 0)

        try:
            with (
                unittest.mock.patch("shutil.which", return_value="/usr/bin/tool"),
                unittest.mock.patch(
                    "terok_sandbox._util._selinux.policy_source_path",
                    return_value=te_file,
                ),
                unittest.mock.patch("subprocess.run", side_effect=_fake_run),
            ):
                install_policy()
        finally:
            ro_dir.chmod(0o755)

        # Artifacts must NOT be in the read-only source dir
        assert artifact_dirs, "checkmodule was not called"
        assert artifact_dirs[0] != str(ro_dir)


class TestUninstallPolicy:
    """Verify policy removal."""

    def test_uninstall_when_installed(self) -> None:
        """Calls semodule -r when policy is installed."""
        with (
            unittest.mock.patch(
                "terok_sandbox._util._selinux.is_policy_installed",
                return_value=True,
            ),
            unittest.mock.patch("subprocess.run") as mock_run,
        ):
            uninstall_policy()
        mock_run.assert_called_once()
        assert mock_run.call_args[0][0][:2] == ["semodule", "-r"]

    def test_noop_when_not_installed(self) -> None:
        """No-op when policy is not installed."""
        with (
            unittest.mock.patch(
                "terok_sandbox._util._selinux.is_policy_installed",
                return_value=False,
            ),
            unittest.mock.patch("subprocess.run") as mock_run,
        ):
            uninstall_policy()
        mock_run.assert_not_called()


# ---------- Socket context manager ----------


class TestSocketSelinuxContext:
    """Verify setsockcreatecon context manager."""

    def test_sets_and_restores_context(self, tmp_path: Path) -> None:
        """Context manager calls setsockcreatecon before yield and restores after."""
        enforce = tmp_path / "enforce"
        enforce.write_text("1\n")

        calls: list[str | None] = []

        def _fake_set(ctx: str | None) -> None:
            calls.append(ctx)

        mock_selinux = unittest.mock.MagicMock()
        mock_selinux.getsockcreatecon.return_value = (0, None)
        mock_selinux.setsockcreatecon.side_effect = _fake_set

        with (
            unittest.mock.patch("terok_sandbox._util._selinux._ENFORCE_PATH", enforce),
            unittest.mock.patch.dict("sys.modules", {"selinux": mock_selinux}),
        ):
            with socket_selinux_context():
                pass

        # First call: set terok_socket_t context
        assert calls[0] is not None
        assert SELINUX_SOCKET_TYPE in calls[0]
        # Second call: restore original (None)
        assert calls[1] is None

    def test_noop_when_selinux_absent(self) -> None:
        """Context manager is a no-op on non-SELinux systems."""
        with unittest.mock.patch(
            "terok_sandbox._util._selinux._ENFORCE_PATH",
            MOCK_ENFORCE_PATH,
        ):
            # Should not raise
            with socket_selinux_context():
                pass

    def test_restores_on_exception(self, tmp_path: Path) -> None:
        """Context restores even when the body raises."""
        enforce = tmp_path / "enforce"
        enforce.write_text("1\n")

        calls: list[str | None] = []

        mock_selinux = unittest.mock.MagicMock()
        mock_selinux.getsockcreatecon.return_value = (0, "old_ctx")
        mock_selinux.setsockcreatecon.side_effect = lambda ctx: calls.append(ctx)

        with (
            unittest.mock.patch("terok_sandbox._util._selinux._ENFORCE_PATH", enforce),
            unittest.mock.patch.dict("sys.modules", {"selinux": mock_selinux}),
        ):
            with pytest.raises(RuntimeError, match="boom"):
                with socket_selinux_context():
                    raise RuntimeError("boom")

        # Restore must happen even on exception
        assert calls[-1] == "old_ctx"


# ---------- Low-level helpers ----------


class TestTrySetsockcreatecon:
    """Verify graceful degradation of setsockcreatecon wrapper."""

    def test_returns_false_without_selinux(self) -> None:
        """Returns False when selinux module is unavailable."""
        with unittest.mock.patch.dict("sys.modules", {"selinux": None}):
            assert _try_setsockcreatecon("some_ctx") is False

    def test_returns_true_with_selinux(self) -> None:
        """Returns True when selinux module is available."""
        mock_selinux = unittest.mock.MagicMock()
        with unittest.mock.patch.dict("sys.modules", {"selinux": mock_selinux}):
            assert _try_setsockcreatecon("some_ctx") is True
        mock_selinux.setsockcreatecon.assert_called_once_with("some_ctx")


class TestTryGetsockcreatecon:
    """Verify graceful degradation of getsockcreatecon wrapper."""

    def test_returns_none_without_selinux(self) -> None:
        """Returns None when selinux module is unavailable."""
        with unittest.mock.patch.dict("sys.modules", {"selinux": None}):
            assert _try_getsockcreatecon() is None

    def test_returns_context_with_selinux(self) -> None:
        """Returns the context string from getsockcreatecon."""
        mock_selinux = unittest.mock.MagicMock()
        mock_selinux.getsockcreatecon.return_value = (0, "old_context")
        with unittest.mock.patch.dict("sys.modules", {"selinux": mock_selinux}):
            assert _try_getsockcreatecon() == "old_context"


# ---------- Policy source path ----------


class TestPolicySourcePath:
    """Verify bundled policy file is discoverable."""

    def test_path_exists(self) -> None:
        """The bundled .te file must exist in the package resources."""
        path = policy_source_path()
        assert path.is_file()
        assert path.name == "terok_socket.te"
