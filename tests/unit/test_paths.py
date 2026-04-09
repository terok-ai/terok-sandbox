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
    runtime_root,
    state_root,
    umbrella_config_dir,
    umbrella_config_root,
    umbrella_runtime_dir,
    umbrella_state_dir,
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

    def test_default_uses_umbrella_namespace(self):
        """Default path nests under the terok/ umbrella."""
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

    def test_default_uses_umbrella_namespace(self):
        """Default path nests under the terok/ umbrella."""
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

    def test_default_uses_umbrella_namespace(self):
        """Default path nests under the terok/ umbrella."""
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

    def test_default_uses_umbrella_namespace(self):
        """Default path nests under the terok/ umbrella."""
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


class TestUmbrellaConfigRoot:
    """Tests for umbrella_config_root()."""

    def test_returns_path(self):
        """umbrella_config_root() returns a Path instance."""
        assert isinstance(umbrella_config_root(), Path)

    def test_env_override(self):
        """TEROK_CONFIG_DIR env var takes precedence."""
        with unittest.mock.patch.dict(
            os.environ, {"TEROK_CONFIG_DIR": str(MOCK_BASE / "umbrella-cfg")}
        ):
            assert umbrella_config_root() == MOCK_BASE / "umbrella-cfg"

    def test_default_ends_with_terok(self):
        """Default path ends with the 'terok' umbrella directory."""
        with unittest.mock.patch.dict(os.environ, {}, clear=True):
            result = umbrella_config_root()
            assert result.name == "terok"

    def test_root_user_fallback(self):
        """When running as root, falls back to /etc/terok."""
        with (
            unittest.mock.patch.dict(os.environ, {}, clear=True),
            unittest.mock.patch("terok_sandbox.paths._is_root", return_value=True),
        ):
            assert umbrella_config_root() == Path("/etc/terok")


# ---------------------------------------------------------------------------
# Umbrella resolver tests
# ---------------------------------------------------------------------------

_NAMESPACE_ROOT = "terok"


@pytest.fixture(autouse=True)
def _reset_config_cache():
    """Clear the config paths cache between tests."""
    _paths_mod._config_paths_cache = None
    yield
    _paths_mod._config_paths_cache = None


class TestUmbrellaRoot:
    """Tests for TEROK_ROOT env var and config.yml paths.root reading."""

    def test_terok_root_env_overrides_platform_default(self):
        """TEROK_ROOT moves the umbrella state root for all subdirs."""
        with unittest.mock.patch.dict(os.environ, {"TEROK_ROOT": str(MOCK_BASE / "custom")}):
            assert umbrella_state_dir("sandbox") == MOCK_BASE / "custom" / "sandbox"
            assert umbrella_state_dir("agent") == MOCK_BASE / "custom" / "agent"
            assert umbrella_state_dir() == MOCK_BASE / "custom"

    def test_config_yml_paths_root(self, tmp_path: Path):
        """config.yml paths.root is honored when no env var is set."""
        cfg = tmp_path / "config.yml"
        custom = tmp_path / "from-config"
        cfg.write_text(f"paths:\n  root: {custom}\n", encoding="utf-8")
        with unittest.mock.patch.dict(os.environ, {"TEROK_CONFIG_FILE": str(cfg)}, clear=True):
            assert umbrella_state_dir("sandbox") == custom / "sandbox"

    def test_config_yml_state_dir_as_deprecated_alias(self, tmp_path: Path):
        """config.yml paths.state_dir is accepted as a deprecated alias for paths.root."""
        cfg = tmp_path / "config.yml"
        custom = tmp_path / "legacy-state"
        cfg.write_text(f"paths:\n  state_dir: {custom}\n", encoding="utf-8")
        with unittest.mock.patch.dict(os.environ, {"TEROK_CONFIG_FILE": str(cfg)}, clear=True):
            assert umbrella_state_dir("core") == custom / "core"

    def test_paths_root_preferred_over_state_dir(self, tmp_path: Path):
        """When both paths.root and paths.state_dir are set, root wins."""
        cfg = tmp_path / "config.yml"
        cfg.write_text(
            f"paths:\n  root: {tmp_path / 'new'}\n  state_dir: {tmp_path / 'old'}\n",
            encoding="utf-8",
        )
        with unittest.mock.patch.dict(os.environ, {"TEROK_CONFIG_FILE": str(cfg)}, clear=True):
            assert umbrella_state_dir("x") == (tmp_path / "new" / "x").resolve()

    def test_package_env_overrides_terok_root(self):
        """Package-specific env var beats TEROK_ROOT."""
        with unittest.mock.patch.dict(
            os.environ,
            {"TEROK_ROOT": str(MOCK_BASE / "root"), "MY_PKG": str(MOCK_BASE / "pkg")},
        ):
            assert umbrella_state_dir("x", "MY_PKG") == MOCK_BASE / "pkg"

    def test_terok_root_overrides_config_yml(self, tmp_path: Path):
        """TEROK_ROOT env var beats config.yml paths.root."""
        cfg = tmp_path / "config.yml"
        cfg.write_text(f"paths:\n  root: {tmp_path / 'from-config'}\n", encoding="utf-8")
        with unittest.mock.patch.dict(
            os.environ,
            {"TEROK_CONFIG_FILE": str(cfg), "TEROK_ROOT": str(MOCK_BASE / "from-env")},
        ):
            assert umbrella_state_dir("sandbox") == MOCK_BASE / "from-env" / "sandbox"

    def test_missing_config_yml_falls_through(self):
        """Missing config.yml is silently ignored."""
        with unittest.mock.patch.dict(
            os.environ, {"TEROK_CONFIG_FILE": "/nonexistent/config.yml"}, clear=True
        ):
            result = umbrella_state_dir("sandbox")
            assert isinstance(result, Path)
            assert result.name == "sandbox"

    def test_malformed_config_yml_falls_through(self, tmp_path: Path):
        """Malformed config.yml is silently ignored."""
        cfg = tmp_path / "config.yml"
        cfg.write_text("not: [valid: yaml: {{{\n", encoding="utf-8")
        with unittest.mock.patch.dict(os.environ, {"TEROK_CONFIG_FILE": str(cfg)}, clear=True):
            result = umbrella_state_dir("sandbox")
            assert isinstance(result, Path)


class TestUmbrellaStateDir:
    """Tests for umbrella_state_dir()."""

    def test_no_subdir_returns_umbrella_root(self):
        """Empty subdir returns the bare terok/ data root."""
        with unittest.mock.patch.dict(os.environ, {}, clear=True):
            result = umbrella_state_dir()
            assert result.name == _NAMESPACE_ROOT

    def test_subdir_appended(self):
        """Subdir is appended to the umbrella root."""
        with unittest.mock.patch.dict(os.environ, {}, clear=True):
            result = umbrella_state_dir("agent")
            assert result.name == "agent"
            assert result.parent.name == _NAMESPACE_ROOT

    def test_env_var_override(self):
        """Specific env var takes precedence over platform default."""
        with unittest.mock.patch.dict(os.environ, {"MY_STATE": str(MOCK_BASE / "custom-state")}):
            assert umbrella_state_dir("agent", "MY_STATE") == MOCK_BASE / "custom-state"

    def test_env_var_none_ignored(self):
        """When env_var is None, no env lookup is performed."""
        with unittest.mock.patch.dict(os.environ, {}, clear=True):
            result = umbrella_state_dir("sandbox")
            assert isinstance(result, Path)

    def test_root_user(self):
        """Root user gets /var/lib/terok/<subdir>."""
        with (
            unittest.mock.patch.dict(os.environ, {}, clear=True),
            unittest.mock.patch("terok_sandbox.paths._is_root", return_value=True),
        ):
            assert umbrella_state_dir("sandbox") == Path("/var/lib/terok/sandbox")
            assert umbrella_state_dir() == Path("/var/lib/terok")

    def test_tilde_expansion(self):
        """Env var values with ~ are expanded."""
        with unittest.mock.patch.dict(os.environ, {"MY_STATE": "~/terok-data"}):
            result = umbrella_state_dir("x", "MY_STATE")
            assert "~" not in str(result)
            assert result.is_absolute()

    def test_absolute_subdir_rejected(self):
        """Absolute subdir paths are rejected to prevent namespace escape."""
        with pytest.raises(ValueError, match="relative"):
            umbrella_state_dir("/tmp/evil")

    def test_parent_traversal_rejected(self):
        """Parent traversal in subdir is rejected."""
        with pytest.raises(ValueError, match="relative"):
            umbrella_state_dir("../escape")


class TestUmbrellaConfigDir:
    """Tests for umbrella_config_dir()."""

    def test_no_subdir_returns_umbrella_root(self):
        """Empty subdir returns the bare terok/ config root."""
        with unittest.mock.patch.dict(os.environ, {}, clear=True):
            result = umbrella_config_dir()
            assert result.name == _NAMESPACE_ROOT

    def test_subdir_appended(self):
        """Subdir is appended to the config umbrella root."""
        with unittest.mock.patch.dict(os.environ, {}, clear=True):
            result = umbrella_config_dir("agent")
            assert result.name == "agent"
            assert result.parent.name == _NAMESPACE_ROOT

    def test_env_var_override(self):
        """Specific env var takes precedence over platform default."""
        with unittest.mock.patch.dict(os.environ, {"MY_CFG": str(MOCK_BASE / "custom-cfg")}):
            assert umbrella_config_dir("agent", "MY_CFG") == MOCK_BASE / "custom-cfg"

    def test_root_user(self):
        """Root user gets /etc/terok/<subdir>."""
        with (
            unittest.mock.patch.dict(os.environ, {}, clear=True),
            unittest.mock.patch("terok_sandbox.paths._is_root", return_value=True),
        ):
            assert umbrella_config_dir("sandbox") == Path("/etc/terok/sandbox")
            assert umbrella_config_dir() == Path("/etc/terok")


class TestUmbrellaRuntimeDir:
    """Tests for umbrella_runtime_dir()."""

    def test_no_subdir_returns_umbrella_root(self):
        """Empty subdir returns the bare terok/ runtime root."""
        with unittest.mock.patch.dict(os.environ, {}, clear=True):
            result = umbrella_runtime_dir()
            assert result.name == _NAMESPACE_ROOT

    def test_subdir_appended(self):
        """Subdir is appended to the runtime umbrella root."""
        with unittest.mock.patch.dict(os.environ, {}, clear=True):
            result = umbrella_runtime_dir("sandbox")
            assert result.name == "sandbox"
            assert result.parent.name == _NAMESPACE_ROOT

    def test_env_var_override(self):
        """Specific env var takes precedence."""
        with unittest.mock.patch.dict(os.environ, {"MY_RUN": str(MOCK_BASE / "custom-run")}):
            assert umbrella_runtime_dir("x", "MY_RUN") == MOCK_BASE / "custom-run"

    def test_root_user(self):
        """Root user gets /run/terok/<subdir>."""
        with (
            unittest.mock.patch.dict(os.environ, {}, clear=True),
            unittest.mock.patch("terok_sandbox.paths._is_root", return_value=True),
        ):
            assert umbrella_runtime_dir("sandbox") == Path("/run/terok/sandbox")
            assert umbrella_runtime_dir() == Path("/run/terok")

    def test_xdg_runtime_dir_fallback(self):
        """XDG_RUNTIME_DIR is used when available (non-root)."""
        with (
            unittest.mock.patch.dict(os.environ, {"XDG_RUNTIME_DIR": "/run/user/1000"}, clear=True),
            unittest.mock.patch("terok_sandbox.paths._is_root", return_value=False),
        ):
            assert umbrella_runtime_dir("sandbox") == Path("/run/user/1000/terok/sandbox")

    def test_xdg_state_home_fallback(self):
        """XDG_STATE_HOME is used when XDG_RUNTIME_DIR is absent."""
        with (
            unittest.mock.patch.dict(
                os.environ, {"XDG_STATE_HOME": str(MOCK_BASE / "state-home")}, clear=True
            ),
            unittest.mock.patch("terok_sandbox.paths._is_root", return_value=False),
        ):
            assert umbrella_runtime_dir("sandbox") == MOCK_BASE / "state-home" / "terok" / "sandbox"
