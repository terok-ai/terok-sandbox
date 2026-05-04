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
    SelinuxStatus,
    _try_getsockcreatecon,
    _try_setsockcreatecon,
    check_status,
    install_command,
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
    check_context_rc: int = 0,
) -> unittest.mock.MagicMock:
    """Build a MagicMock that mimics the libselinux CDLL handle.

    *check_context_rc* controls ``security_check_context`` — the
    userspace probe used by `is_policy_installed` and the new
    `is_socket_type_loaded` / `is_domain_loaded` helpers.  Default
    ``0`` (success) means probed types are treated as loaded; pass
    a non-zero rc to model a host where the policy module is absent.
    """
    lib = unittest.mock.MagicMock()
    lib.setsockcreatecon.return_value = set_rc
    lib.security_check_context.return_value = check_context_rc

    def _fake_get(ptr: ctypes.c_char_p) -> int:
        # Deliberate test scaffolding: ``ctypes.byref(x)`` returns a
        # ``CArgObject`` whose private ``_obj`` attribute points back at
        # the original ``c_char_p`` instance.  We mutate it here to
        # emulate the C out-parameter pattern that ``getsockcreatecon``
        # uses.  This relies on a ctypes CPython implementation detail
        # — if a future Python version changes the internal attribute,
        # this test will fail fast and we'll switch to a narrower stub.
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
    """Verify policy type detection via libselinux security_check_context."""

    def test_installed(self) -> None:
        """security_check_context returning 0 means the type is known."""
        lib = _mock_libselinux()
        lib.security_check_context.return_value = 0
        with unittest.mock.patch("ctypes.CDLL", return_value=lib):
            assert is_policy_installed() is True
        lib.security_check_context.assert_called_once()
        # Must pass a context containing the terok_socket_t type as bytes.
        ctx = lib.security_check_context.call_args.args[0]
        assert b"terok_socket_t" in ctx

    def test_not_installed(self) -> None:
        """security_check_context returning -1 means the type is not in policy."""
        lib = _mock_libselinux()
        lib.security_check_context.return_value = -1
        with unittest.mock.patch("ctypes.CDLL", return_value=lib):
            assert is_policy_installed() is False

    def test_libselinux_missing(self) -> None:
        """No libselinux.so.1 → no way to tell, treat as not installed."""
        with unittest.mock.patch("ctypes.CDLL", side_effect=OSError("missing")):
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


# ---------- Hardening install command ----------


class TestInstallCommand:
    """Verify the user-facing install command string helper."""

    def test_returns_terok_hardening_install(self) -> None:
        """Returns the orchestrator's CLI invocation, not a sudo-bash path.

        The bash install script was retired in favour of
        ``terok hardening install`` (a Python orchestrator that runs
        as the user and shells out to sudo only for privileged
        operations).  This helper is the single source of the
        invocation string; sickbay hints and setup tips render the
        same value.
        """
        assert install_command() == "terok hardening install"


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

    def test_picks_first_loadable_candidate(self, tmp_path: Path) -> None:
        """Variadic call uses the first candidate whose type is loaded.

        Per-service hardening rolls out incrementally: services pass the
        per-service type *and* the legacy fallback, so a mixed-version
        host (per-service module loaded → use new type; legacy-only
        host → fall back to terok_socket_t) lands the right context
        without service-side branching.
        """
        enforce = tmp_path / "enforce"
        enforce.write_text("1\n")

        # check_context returns 0 for any probed type — both candidates
        # are "loaded".  First in candidate order wins.
        lib = _mock_libselinux(get_value=None, check_context_rc=0)

        with (
            unittest.mock.patch("terok_sandbox._util._selinux._ENFORCE_PATH", enforce),
            unittest.mock.patch("ctypes.CDLL", return_value=lib),
        ):
            with socket_selinux_context("terok_gate_sock_t", SELINUX_SOCKET_TYPE):
                pass

        first_ctx = lib.setsockcreatecon.call_args_list[0].args[0]
        assert b"terok_gate_sock_t" in first_ctx
        assert SELINUX_SOCKET_TYPE.encode() not in first_ctx

    def test_falls_back_when_first_candidate_not_loaded(self, tmp_path: Path) -> None:
        """If the first candidate type isn't in policy, try the next.

        The legacy-only-policy case: per-service modules not installed,
        legacy ``terok_socket`` allow rule still works.
        """
        enforce = tmp_path / "enforce"
        enforce.write_text("1\n")

        lib = _mock_libselinux(get_value=None)
        # Reject the per-service type, accept the legacy type.
        lib.security_check_context.side_effect = lambda ctx: (
            0 if SELINUX_SOCKET_TYPE.encode() in ctx else 1
        )

        with (
            unittest.mock.patch("terok_sandbox._util._selinux._ENFORCE_PATH", enforce),
            unittest.mock.patch("ctypes.CDLL", return_value=lib),
        ):
            with socket_selinux_context("terok_gate_sock_t", SELINUX_SOCKET_TYPE):
                pass

        first_ctx = lib.setsockcreatecon.call_args_list[0].args[0]
        assert SELINUX_SOCKET_TYPE.encode() in first_ctx

    def test_no_labelling_when_no_candidate_loaded(self, tmp_path: Path) -> None:
        """If no candidate type is in policy, yield without labelling.

        Container connectto will be denied — but we let the bind go
        through so the operator surface (sickbay row, setup WARN) tells
        them what's missing instead of crashing the service.
        """
        enforce = tmp_path / "enforce"
        enforce.write_text("1\n")

        # Reject every probed type.
        lib = _mock_libselinux(get_value=None)
        lib.security_check_context.return_value = 1

        with (
            unittest.mock.patch("terok_sandbox._util._selinux._ENFORCE_PATH", enforce),
            unittest.mock.patch("ctypes.CDLL", return_value=lib),
        ):
            with socket_selinux_context("terok_gate_sock_t", SELINUX_SOCKET_TYPE):
                pass

        # No setsockcreatecon call at all — the bind happens with the
        # process default context, container connectto will be denied.
        assert lib.setsockcreatecon.call_args_list == []

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

    def test_returns_none_on_nonzero_rc(self) -> None:
        """Returns None when ``getsockcreatecon`` itself fails (rc != 0)."""
        lib = _mock_libselinux(get_rc=-1, get_value=b"ignored")
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


class TestCheckStatus:
    """Verify the single decision tree exposed as ``check_selinux_status``."""

    def test_tcp_mode_is_not_applicable(self) -> None:
        """Transport ``tcp`` short-circuits to NOT_APPLICABLE_TCP_MODE."""
        result = check_status(services_mode="tcp")
        assert result.status is SelinuxStatus.NOT_APPLICABLE_TCP_MODE
        assert result.missing_policy_tools == ()

    def test_permissive_host_is_not_applicable(self, tmp_path: Path) -> None:
        """Socket mode but non-enforcing host → NOT_APPLICABLE_PERMISSIVE."""
        enforce = tmp_path / "enforce"
        enforce.write_text("0\n")
        with unittest.mock.patch("terok_sandbox._util._selinux._ENFORCE_PATH", enforce):
            result = check_status(services_mode="socket")
        assert result.status is SelinuxStatus.NOT_APPLICABLE_PERMISSIVE

    def test_policy_missing_includes_missing_tools(self, tmp_path: Path) -> None:
        """POLICY_MISSING carries the list of missing compile tools."""
        enforce = tmp_path / "enforce"
        enforce.write_text("1\n")
        lib = _mock_libselinux()
        lib.security_check_context.return_value = -1  # policy not loaded
        which_map = {"semodule": "/usr/sbin/semodule"}
        with (
            unittest.mock.patch("terok_sandbox._util._selinux._ENFORCE_PATH", enforce),
            unittest.mock.patch("ctypes.CDLL", return_value=lib),
            unittest.mock.patch("shutil.which", side_effect=which_map.get),
        ):
            result = check_status(services_mode="socket")
        assert result.status is SelinuxStatus.POLICY_MISSING
        assert result.missing_policy_tools == ("checkmodule", "semodule_package")

    def test_libselinux_missing_fires_only_when_policy_installed(self, tmp_path: Path) -> None:
        """LIBSELINUX_MISSING only makes sense when policy *is* installed."""
        enforce = tmp_path / "enforce"
        enforce.write_text("1\n")
        # Two interacting paths: is_policy_installed needs libselinux (uses
        # ctypes.CDLL via _load_libselinux).  is_libselinux_available also
        # uses _load_libselinux.  When libselinux fails to load, both
        # predicates return False, so the "policy missing" branch fires
        # first and we never reach LIBSELINUX_MISSING.  That's the right
        # semantic: no libselinux → can't even verify policy.
        with (
            unittest.mock.patch("terok_sandbox._util._selinux._ENFORCE_PATH", enforce),
            unittest.mock.patch("ctypes.CDLL", side_effect=OSError("missing")),
            unittest.mock.patch("shutil.which", return_value="/usr/bin/x"),
        ):
            result = check_status(services_mode="socket")
        assert result.status is SelinuxStatus.POLICY_MISSING

    def test_ok_when_everything_ready(self, tmp_path: Path) -> None:
        """Enforcing + policy loaded + libselinux loadable → OK."""
        enforce = tmp_path / "enforce"
        enforce.write_text("1\n")
        lib = _mock_libselinux()
        lib.security_check_context.return_value = 0
        with (
            unittest.mock.patch("terok_sandbox._util._selinux._ENFORCE_PATH", enforce),
            unittest.mock.patch("ctypes.CDLL", return_value=lib),
        ):
            result = check_status(services_mode="socket")
        assert result.status is SelinuxStatus.OK
