# SPDX-FileCopyrightText: 2026 Jiri Vyskocil
# SPDX-License-Identifier: Apache-2.0

"""Tests for SELinux socket labeling and policy management."""

from __future__ import annotations

import ctypes
import unittest.mock
from pathlib import Path

import pytest

from terok_sandbox._util import _selinux
from terok_sandbox._util._selinux import (
    SELINUX_SOCKET_TYPE,
    _try_getsockcreatecon,
    _try_setsockcreatecon,
    install_script_path,
    is_libselinux_available,
    is_policy_installed,
    is_selinux_enabled,
    is_selinux_enforcing,
    missing_policy_tools,
    policy_source_path,
    socket_selinux_context,
)
from tests.constants import MOCK_BASE

MOCK_ENFORCE_PATH = MOCK_BASE / "selinux" / "enforce"


@pytest.fixture(autouse=True)
def _reset_libselinux_cache() -> None:
    """Clear the cached ``_load_libselinux`` result between tests."""
    _selinux._load_libselinux.cache_clear()


def _mock_libselinux(
    *,
    set_rc: int = 0,
    get_rc: int = 0,
    get_value: bytes | None = None,
) -> unittest.mock.MagicMock:
    """Build a MagicMock that mimics the libselinux CDLL handle."""
    lib = unittest.mock.MagicMock()
    lib.setsockcreatecon.return_value = set_rc

    def _fake_get(ptr: ctypes.c_char_p) -> int:
        # ptr is a POINTER(c_char_p); emulate the C out-param assignment.
        ptr._obj.value = get_value  # type: ignore[attr-defined]
        return get_rc

    lib.getsockcreatecon.side_effect = _fake_get
    lib.freecon.return_value = None
    return lib


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


# ---------- libselinux availability ----------


class TestIsLibselinuxAvailable:
    """Verify libselinux.so.1 load check."""

    def test_available(self) -> None:
        """Returns True when CDLL load succeeds."""
        with unittest.mock.patch("ctypes.CDLL", return_value=_mock_libselinux()):
            assert is_libselinux_available() is True

    def test_missing(self) -> None:
        """Returns False when CDLL load raises OSError."""
        with unittest.mock.patch("ctypes.CDLL", side_effect=OSError("not found")):
            assert is_libselinux_available() is False


class TestMissingPolicyTools:
    """Verify policy-tool availability probe."""

    def test_none_missing(self) -> None:
        """Empty list when every tool is on PATH."""
        with unittest.mock.patch("shutil.which", return_value="/usr/bin/any"):
            assert missing_policy_tools() == []

    def test_reports_absent_tools_in_order(self) -> None:
        """Tools are listed in the order install_policy would call them."""
        present = {"semodule": "/usr/sbin/semodule"}
        with unittest.mock.patch("shutil.which", side_effect=present.get):
            assert missing_policy_tools() == ["checkmodule", "semodule_package"]


# ---------- Vendored installer script ----------


class TestInstallScriptPath:
    """Verify the bundled installer shell script is discoverable."""

    def test_script_exists(self) -> None:
        """install_policy.sh must ship in the resource directory."""
        path = install_script_path()
        assert path.is_file()
        assert path.name == "install_policy.sh"


# ---------- Socket context manager ----------


class TestSocketSelinuxContext:
    """Verify setsockcreatecon context manager."""

    def test_sets_and_restores_context(self, tmp_path: Path) -> None:
        """Context manager sets terok_socket_t on entry, restores on exit."""
        enforce = tmp_path / "enforce"
        enforce.write_text("1\n")

        lib = _mock_libselinux(get_value=None)

        with (
            unittest.mock.patch("terok_sandbox._util._selinux._ENFORCE_PATH", enforce),
            unittest.mock.patch("ctypes.CDLL", return_value=lib),
        ):
            with socket_selinux_context():
                pass

        # First: set terok_socket_t context
        set_calls = lib.setsockcreatecon.call_args_list
        assert len(set_calls) == 2
        first_arg = set_calls[0].args[0]
        assert first_arg is not None
        assert SELINUX_SOCKET_TYPE.encode() in first_arg
        # Second: restore (None since original was unset)
        assert set_calls[1].args[0] is None

    def test_noop_when_selinux_absent(self) -> None:
        """Context manager is a no-op on non-SELinux systems."""
        with unittest.mock.patch(
            "terok_sandbox._util._selinux._ENFORCE_PATH",
            MOCK_ENFORCE_PATH,
        ):
            with socket_selinux_context():
                pass  # must not raise

    def test_restores_on_exception(self, tmp_path: Path) -> None:
        """Context restores even when the body raises."""
        enforce = tmp_path / "enforce"
        enforce.write_text("1\n")

        lib = _mock_libselinux(get_value=b"old_ctx")

        with (
            unittest.mock.patch("terok_sandbox._util._selinux._ENFORCE_PATH", enforce),
            unittest.mock.patch("ctypes.CDLL", return_value=lib),
        ):
            with pytest.raises(RuntimeError, match="boom"):
                with socket_selinux_context():
                    raise RuntimeError("boom")

        # Restore call sent the captured prior context back (encoded).
        set_calls = lib.setsockcreatecon.call_args_list
        assert set_calls[-1].args[0] == b"old_ctx"


# ---------- Low-level helpers ----------


class TestTrySetsockcreatecon:
    """Verify graceful degradation of setsockcreatecon wrapper."""

    def test_returns_false_without_libselinux(self) -> None:
        """Returns False when libselinux.so.1 fails to load."""
        with unittest.mock.patch("ctypes.CDLL", side_effect=OSError("missing")):
            assert _try_setsockcreatecon("some_ctx") is False

    def test_returns_true_on_success(self) -> None:
        """Returns True when libselinux accepts the context."""
        lib = _mock_libselinux(set_rc=0)
        with unittest.mock.patch("ctypes.CDLL", return_value=lib):
            assert _try_setsockcreatecon("some_ctx") is True
        lib.setsockcreatecon.assert_called_once_with(b"some_ctx")

    def test_returns_false_on_nonzero_rc(self) -> None:
        """Returns False when libselinux returns a non-zero status."""
        lib = _mock_libselinux(set_rc=-1)
        with unittest.mock.patch("ctypes.CDLL", return_value=lib):
            assert _try_setsockcreatecon("some_ctx") is False

    def test_passes_none_to_clear(self) -> None:
        """Passing None forwards NULL to libselinux (clear context)."""
        lib = _mock_libselinux(set_rc=0)
        with unittest.mock.patch("ctypes.CDLL", return_value=lib):
            assert _try_setsockcreatecon(None) is True
        lib.setsockcreatecon.assert_called_once_with(None)


class TestTryGetsockcreatecon:
    """Verify graceful degradation of getsockcreatecon wrapper."""

    def test_returns_none_without_libselinux(self) -> None:
        """Returns None when libselinux.so.1 fails to load."""
        with unittest.mock.patch("ctypes.CDLL", side_effect=OSError("missing")):
            assert _try_getsockcreatecon() is None

    def test_returns_context_string(self) -> None:
        """Returns the context as a str when libselinux reports one."""
        lib = _mock_libselinux(get_rc=0, get_value=b"old_context")
        with unittest.mock.patch("ctypes.CDLL", return_value=lib):
            assert _try_getsockcreatecon() == "old_context"

    def test_returns_none_when_unset(self) -> None:
        """Returns None when libselinux reports no current context."""
        lib = _mock_libselinux(get_rc=0, get_value=None)
        with unittest.mock.patch("ctypes.CDLL", return_value=lib):
            assert _try_getsockcreatecon() is None


# ---------- Policy source path ----------


class TestPolicySourcePath:
    """Verify bundled policy file is discoverable."""

    def test_path_exists(self) -> None:
        """The bundled .te file must exist in the package resources."""
        path = policy_source_path()
        assert path.is_file()
        assert path.name == "terok_socket.te"
