# SPDX-FileCopyrightText: 2025 Jiri Vyskocil
# SPDX-FileCopyrightText: 2026 Jiri Vyskocil
# SPDX-License-Identifier: Apache-2.0

"""Tests for path resolution functions in terok_sandbox.paths."""

from __future__ import annotations

import os
import unittest.mock
from pathlib import Path

import pytest

import terok_sandbox.paths as _paths_mod
from terok_sandbox.paths import (
    config_root,
    credentials_root,
    namespace_config_dir,
    namespace_config_root,
    namespace_runtime_dir,
    namespace_state_dir,
    runtime_root,
    state_root,
)
from tests.constants import MOCK_BASE

MOCK_CREDENTIALS_DIR = MOCK_BASE / "credentials"

_NAMESPACE_SEP = "terok" + os.sep + "sandbox"
"""Expected namespace path segment for sandbox roots (``terok/sandbox``)."""

_CRED_NAMESPACE_SEP = "terok" + os.sep + "credentials"
"""Expected namespace path segment for credentials root (``terok/credentials``)."""


class TestConfigRoot:
    """Tests for config_root()."""

    def test_returns_path(self):
        """config_root() returns a Path instance."""
        assert isinstance(config_root(), Path)

    def test_env_override(self):
        """TEROK_SANDBOX_CONFIG_DIR env var takes precedence."""
        with unittest.mock.patch.dict(
            os.environ, {"TEROK_SANDBOX_CONFIG_DIR": str(MOCK_BASE / "cfg")}
        ):
            assert config_root() == MOCK_BASE / "cfg"

    def test_default_uses_namespace(self):
        """Default path nests under the terok/ namespace."""
        with unittest.mock.patch.dict(os.environ, {}, clear=True):
            result = config_root()
            assert _NAMESPACE_SEP in str(result)

    def test_root_user_fallback(self):
        """When running as root, falls back to /etc/terok/sandbox."""
        with (
            unittest.mock.patch.dict(os.environ, {}, clear=True),
            unittest.mock.patch("terok_sandbox.paths._is_root", return_value=True),
        ):
            assert config_root() == Path("/etc/terok/sandbox")


class TestStateRoot:
    """Tests for state_root()."""

    def test_returns_path(self):
        """state_root() returns a Path instance."""
        assert isinstance(state_root(), Path)

    def test_env_override(self):
        """TEROK_SANDBOX_STATE_DIR env var takes precedence."""
        with unittest.mock.patch.dict(
            os.environ, {"TEROK_SANDBOX_STATE_DIR": str(MOCK_BASE / "state")}
        ):
            assert state_root() == MOCK_BASE / "state"

    def test_default_uses_namespace(self):
        """Default path nests under the terok/ namespace."""
        with unittest.mock.patch.dict(os.environ, {}, clear=True):
            result = state_root()
            assert _NAMESPACE_SEP in str(result)

    def test_root_user_fallback(self):
        """When running as root, falls back to /var/lib/terok/sandbox."""
        with (
            unittest.mock.patch.dict(os.environ, {}, clear=True),
            unittest.mock.patch("terok_sandbox.paths._is_root", return_value=True),
        ):
            assert state_root() == Path("/var/lib/terok/sandbox")


class TestRuntimeRoot:
    """Tests for runtime_root()."""

    def test_returns_path(self):
        """runtime_root() returns a Path instance."""
        assert isinstance(runtime_root(), Path)

    def test_env_override(self):
        """TEROK_SANDBOX_RUNTIME_DIR env var takes precedence."""
        with unittest.mock.patch.dict(
            os.environ, {"TEROK_SANDBOX_RUNTIME_DIR": str(MOCK_BASE / "run")}
        ):
            assert runtime_root() == MOCK_BASE / "run"

    def test_default_uses_namespace(self):
        """Default path nests under the terok/ namespace."""
        with unittest.mock.patch.dict(os.environ, {}, clear=True):
            result = runtime_root()
            assert _NAMESPACE_SEP in str(result)

    def test_root_user_fallback(self):
        """When running as root, falls back to /run/terok/sandbox."""
        with (
            unittest.mock.patch.dict(os.environ, {}, clear=True),
            unittest.mock.patch("terok_sandbox.paths._is_root", return_value=True),
        ):
            assert runtime_root() == Path("/run/terok/sandbox")


class TestCredentialsRoot:
    """Tests for credentials_root()."""

    def test_returns_path(self):
        """credentials_root() returns a Path instance."""
        assert isinstance(credentials_root(), Path)

    def test_env_override(self):
        """TEROK_CREDENTIALS_DIR env var takes precedence."""
        with unittest.mock.patch.dict(
            os.environ, {"TEROK_CREDENTIALS_DIR": str(MOCK_CREDENTIALS_DIR)}
        ):
            assert credentials_root() == MOCK_CREDENTIALS_DIR

    def test_default_uses_namespace(self):
        """Default path nests under the terok/ namespace."""
        with unittest.mock.patch.dict(os.environ, {}, clear=True):
            result = credentials_root()
            assert _CRED_NAMESPACE_SEP in str(result)

    def test_env_override_with_tilde(self):
        """TEROK_CREDENTIALS_DIR supports ~ expansion."""
        with unittest.mock.patch.dict(os.environ, {"TEROK_CREDENTIALS_DIR": "~/my-creds"}):
            result = credentials_root()
            assert "~" not in str(result)
            assert result.is_absolute()

    def test_root_user_fallback(self):
        """When running as root, falls back to /var/lib/terok/credentials."""
        with (
            unittest.mock.patch.dict(os.environ, {}, clear=True),
            unittest.mock.patch("terok_sandbox.paths._is_root", return_value=True),
        ):
            assert credentials_root() == Path("/var/lib/terok/credentials")


class TestNamespaceConfigRoot:
    """Tests for namespace_config_root()."""

    def test_returns_path(self):
        """namespace_config_root() returns a Path instance."""
        assert isinstance(namespace_config_root(), Path)

    def test_env_override(self):
        """TEROK_CONFIG_DIR env var takes precedence."""
        with unittest.mock.patch.dict(
            os.environ, {"TEROK_CONFIG_DIR": str(MOCK_BASE / "namespace-cfg")}
        ):
            assert namespace_config_root() == MOCK_BASE / "namespace-cfg"

    def test_default_ends_with_terok(self):
        """Default path ends with the 'terok' namespace directory."""
        with unittest.mock.patch.dict(os.environ, {}, clear=True):
            result = namespace_config_root()
            assert result.name == "terok"

    def test_root_user_fallback(self):
        """When running as root, falls back to /etc/terok."""
        with (
            unittest.mock.patch.dict(os.environ, {}, clear=True),
            unittest.mock.patch("terok_sandbox.paths._is_root", return_value=True),
        ):
            assert namespace_config_root() == Path("/etc/terok")


# ---------------------------------------------------------------------------
# Namespace resolver tests
# ---------------------------------------------------------------------------

_NAMESPACE_ROOT = "terok"


@pytest.fixture(autouse=True)
def _reset_config_cache():
    """Clear the config paths cache between tests."""
    _paths_mod._config_section_cache.clear()
    yield
    _paths_mod._config_section_cache.clear()


class TestConfigFilePaths:
    """Tests for _config_file_paths() layered config discovery."""

    def test_env_override_returns_single_path(self):
        """TEROK_CONFIG_FILE returns a single-element list (no layering)."""
        with unittest.mock.patch.dict(
            os.environ, {"TEROK_CONFIG_FILE": str(MOCK_BASE / "custom.yml")}, clear=True
        ):
            result = _paths_mod._config_file_paths()
            assert len(result) == 1
            assert result[0][0] == "override"
            assert result[0][1] == MOCK_BASE / "custom.yml"

    def test_non_root_returns_system_and_user(self):
        """Non-root user gets system (/etc) → user (~/.config) layers."""
        with (
            unittest.mock.patch.dict(os.environ, {}, clear=True),
            unittest.mock.patch("terok_sandbox.paths._is_root", return_value=False),
        ):
            result = _paths_mod._config_file_paths()
            assert len(result) == 2
            labels = [label for label, _ in result]
            assert labels == ["system", "user"]
            assert result[0][1] == Path("/etc/terok/config.yml")

    def test_root_returns_system_only(self):
        """Root user gets only the system path (no user override layer)."""
        with (
            unittest.mock.patch.dict(os.environ, {}, clear=True),
            unittest.mock.patch("terok_sandbox.paths._is_root", return_value=True),
        ):
            result = _paths_mod._config_file_paths()
            assert len(result) == 1
            assert result[0] == ("system", Path("/etc/terok/config.yml"))


class TestLayeredConfigMerge:
    """Tests for _read_config_paths() merging system + user config files."""

    def test_user_overrides_system_paths(self, tmp_path: Path):
        """User config.yml overrides system config.yml at the leaf level."""
        sys_cfg = tmp_path / "system" / "config.yml"
        usr_cfg = tmp_path / "user" / "config.yml"
        sys_cfg.parent.mkdir()
        usr_cfg.parent.mkdir()
        sys_cfg.write_text(
            "paths:\n  root: /srv/terok\n  build_dir: /srv/terok/build\n", encoding="utf-8"
        )
        usr_cfg.write_text("paths:\n  build_dir: /home/me/build\n", encoding="utf-8")
        with unittest.mock.patch.object(
            _paths_mod,
            "_config_file_paths",
            return_value=[("system", sys_cfg), ("user", usr_cfg)],
        ):
            result = _paths_mod._read_config_paths()
            assert result["root"] == "/srv/terok"
            assert result["build_dir"] == "/home/me/build"

    def test_system_only_when_no_user(self, tmp_path: Path):
        """System config is used as-is when no user config exists."""
        sys_cfg = tmp_path / "config.yml"
        sys_cfg.write_text("paths:\n  root: /opt/terok\n", encoding="utf-8")
        with unittest.mock.patch.object(
            _paths_mod,
            "_config_file_paths",
            return_value=[("system", sys_cfg), ("user", tmp_path / "missing.yml")],
        ):
            result = _paths_mod._read_config_paths()
            assert result["root"] == "/opt/terok"

    def test_user_delete_via_none(self, tmp_path: Path):
        """User can remove a system-set key by setting it to null."""
        sys_cfg = tmp_path / "system.yml"
        usr_cfg = tmp_path / "user.yml"
        sys_cfg.write_text("paths:\n  root: /srv/terok\n  build_dir: /x\n", encoding="utf-8")
        usr_cfg.write_text("paths:\n  build_dir: null\n", encoding="utf-8")
        with unittest.mock.patch.object(
            _paths_mod,
            "_config_file_paths",
            return_value=[("system", sys_cfg), ("user", usr_cfg)],
        ):
            result = _paths_mod._read_config_paths()
            assert result["root"] == "/srv/terok"
            assert "build_dir" not in result


class TestNamespaceRoot:
    """Tests for TEROK_ROOT env var and config.yml paths.root reading."""

    def test_terok_root_env_overrides_platform_default(self):
        """TEROK_ROOT moves the namespace state root for all subdirs."""
        with unittest.mock.patch.dict(os.environ, {"TEROK_ROOT": str(MOCK_BASE / "custom")}):
            assert namespace_state_dir("sandbox") == MOCK_BASE / "custom" / "sandbox"
            assert namespace_state_dir("agent") == MOCK_BASE / "custom" / "agent"
            assert namespace_state_dir() == MOCK_BASE / "custom"

    def test_config_yml_paths_root(self, tmp_path: Path):
        """config.yml paths.root is honored when no env var is set."""
        cfg = tmp_path / "config.yml"
        custom = tmp_path / "from-config"
        cfg.write_text(f"paths:\n  root: {custom}\n", encoding="utf-8")
        with unittest.mock.patch.dict(os.environ, {"TEROK_CONFIG_FILE": str(cfg)}, clear=True):
            assert namespace_state_dir("sandbox") == custom / "sandbox"

    def test_package_env_overrides_terok_root(self):
        """Package-specific env var beats TEROK_ROOT."""
        with unittest.mock.patch.dict(
            os.environ,
            {"TEROK_ROOT": str(MOCK_BASE / "root"), "MY_PKG": str(MOCK_BASE / "pkg")},
        ):
            assert namespace_state_dir("x", "MY_PKG") == MOCK_BASE / "pkg"

    def test_terok_root_overrides_config_yml(self, tmp_path: Path):
        """TEROK_ROOT env var beats config.yml paths.root."""
        cfg = tmp_path / "config.yml"
        cfg.write_text(f"paths:\n  root: {tmp_path / 'from-config'}\n", encoding="utf-8")
        with unittest.mock.patch.dict(
            os.environ,
            {"TEROK_CONFIG_FILE": str(cfg), "TEROK_ROOT": str(MOCK_BASE / "from-env")},
        ):
            assert namespace_state_dir("sandbox") == MOCK_BASE / "from-env" / "sandbox"

    def test_missing_config_yml_falls_through(self):
        """Missing config.yml is silently ignored."""
        with unittest.mock.patch.dict(
            os.environ, {"TEROK_CONFIG_FILE": "/nonexistent/config.yml"}, clear=True
        ):
            result = namespace_state_dir("sandbox")
            assert isinstance(result, Path)
            assert result.name == "sandbox"

    def test_malformed_config_yml_falls_through(self, tmp_path: Path):
        """Malformed config.yml is silently ignored."""
        cfg = tmp_path / "config.yml"
        cfg.write_text("not: [valid: yaml: {{{\n", encoding="utf-8")
        with unittest.mock.patch.dict(os.environ, {"TEROK_CONFIG_FILE": str(cfg)}, clear=True):
            result = namespace_state_dir("sandbox")
            assert isinstance(result, Path)


class TestNamespaceStateDir:
    """Tests for namespace_state_dir()."""

    def test_no_subdir_returns_namespace_root(self):
        """Empty subdir returns the bare terok/ data root."""
        with unittest.mock.patch.dict(os.environ, {}, clear=True):
            result = namespace_state_dir()
            assert result.name == _NAMESPACE_ROOT

    def test_subdir_appended(self):
        """Subdir is appended to the namespace root."""
        with unittest.mock.patch.dict(os.environ, {}, clear=True):
            result = namespace_state_dir("agent")
            assert result.name == "agent"
            assert result.parent.name == _NAMESPACE_ROOT

    def test_env_var_override(self):
        """Specific env var takes precedence over platform default."""
        with unittest.mock.patch.dict(os.environ, {"MY_STATE": str(MOCK_BASE / "custom-state")}):
            assert namespace_state_dir("agent", "MY_STATE") == MOCK_BASE / "custom-state"

    def test_env_var_none_ignored(self):
        """When env_var is None, no env lookup is performed."""
        with unittest.mock.patch.dict(os.environ, {}, clear=True):
            result = namespace_state_dir("sandbox")
            assert isinstance(result, Path)

    def test_root_user(self):
        """Root user gets /var/lib/terok/<subdir>."""
        with (
            unittest.mock.patch.dict(os.environ, {}, clear=True),
            unittest.mock.patch("terok_sandbox.paths._is_root", return_value=True),
        ):
            assert namespace_state_dir("sandbox") == Path("/var/lib/terok/sandbox")
            assert namespace_state_dir() == Path("/var/lib/terok")

    def test_tilde_expansion(self):
        """Env var values with ~ are expanded."""
        with unittest.mock.patch.dict(os.environ, {"MY_STATE": "~/terok-data"}):
            result = namespace_state_dir("x", "MY_STATE")
            assert "~" not in str(result)
            assert result.is_absolute()

    def test_absolute_subdir_rejected(self):
        """Absolute subdir paths are rejected to prevent namespace escape."""
        with pytest.raises(ValueError, match="relative"):
            namespace_state_dir("/tmp/evil")

    def test_parent_traversal_rejected(self):
        """Parent traversal in subdir is rejected."""
        with pytest.raises(ValueError, match="relative"):
            namespace_state_dir("../escape")


class TestNamespaceConfigDir:
    """Tests for namespace_config_dir()."""

    def test_no_subdir_returns_namespace_root(self):
        """Empty subdir returns the bare terok/ config root."""
        with unittest.mock.patch.dict(os.environ, {}, clear=True):
            result = namespace_config_dir()
            assert result.name == _NAMESPACE_ROOT

    def test_subdir_appended(self):
        """Subdir is appended to the config namespace root."""
        with unittest.mock.patch.dict(os.environ, {}, clear=True):
            result = namespace_config_dir("agent")
            assert result.name == "agent"
            assert result.parent.name == _NAMESPACE_ROOT

    def test_env_var_override(self):
        """Specific env var takes precedence over platform default."""
        with unittest.mock.patch.dict(os.environ, {"MY_CFG": str(MOCK_BASE / "custom-cfg")}):
            assert namespace_config_dir("agent", "MY_CFG") == MOCK_BASE / "custom-cfg"

    def test_root_user(self):
        """Root user gets /etc/terok/<subdir>."""
        with (
            unittest.mock.patch.dict(os.environ, {}, clear=True),
            unittest.mock.patch("terok_sandbox.paths._is_root", return_value=True),
        ):
            assert namespace_config_dir("sandbox") == Path("/etc/terok/sandbox")
            assert namespace_config_dir() == Path("/etc/terok")


class TestNamespaceRuntimeDir:
    """Tests for namespace_runtime_dir()."""

    def test_no_subdir_returns_namespace_root(self):
        """Empty subdir returns the bare terok/ runtime root."""
        with unittest.mock.patch.dict(os.environ, {}, clear=True):
            result = namespace_runtime_dir()
            assert result.name == _NAMESPACE_ROOT

    def test_subdir_appended(self):
        """Subdir is appended to the runtime namespace root."""
        with unittest.mock.patch.dict(os.environ, {}, clear=True):
            result = namespace_runtime_dir("sandbox")
            assert result.name == "sandbox"
            assert result.parent.name == _NAMESPACE_ROOT

    def test_env_var_override(self):
        """Specific env var takes precedence."""
        with unittest.mock.patch.dict(os.environ, {"MY_RUN": str(MOCK_BASE / "custom-run")}):
            assert namespace_runtime_dir("x", "MY_RUN") == MOCK_BASE / "custom-run"

    def test_root_user(self):
        """Root user gets /run/terok/<subdir>."""
        with (
            unittest.mock.patch.dict(os.environ, {}, clear=True),
            unittest.mock.patch("terok_sandbox.paths._is_root", return_value=True),
        ):
            assert namespace_runtime_dir("sandbox") == Path("/run/terok/sandbox")
            assert namespace_runtime_dir() == Path("/run/terok")

    def test_xdg_runtime_dir_fallback(self):
        """XDG_RUNTIME_DIR is used when available (non-root)."""
        with (
            unittest.mock.patch.dict(os.environ, {"XDG_RUNTIME_DIR": "/run/user/1000"}, clear=True),
            unittest.mock.patch("terok_sandbox.paths._is_root", return_value=False),
        ):
            assert namespace_runtime_dir("sandbox") == Path("/run/user/1000/terok/sandbox")

    def test_xdg_state_home_fallback(self):
        """XDG_STATE_HOME is used when XDG_RUNTIME_DIR is absent."""
        with (
            unittest.mock.patch.dict(
                os.environ, {"XDG_STATE_HOME": str(MOCK_BASE / "state-home")}, clear=True
            ),
            unittest.mock.patch("terok_sandbox.paths._is_root", return_value=False),
        ):
            assert (
                namespace_runtime_dir("sandbox") == MOCK_BASE / "state-home" / "terok" / "sandbox"
            )
