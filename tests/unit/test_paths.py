# SPDX-FileCopyrightText: 2025 Jiri Vyskocil
# SPDX-License-Identifier: Apache-2.0

"""Tests for path resolution functions in terok_sandbox.paths."""

from __future__ import annotations

import os
import unittest.mock
from pathlib import Path

from terok_sandbox.paths import config_root, credentials_root, runtime_root, state_root
from tests.constants import MOCK_BASE

MOCK_CREDENTIALS_DIR = MOCK_BASE / "credentials"


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

    def test_default_contains_app_name(self):
        """Default path uses the terok-sandbox namespace."""
        with unittest.mock.patch.dict(os.environ, {}, clear=True):
            result = config_root()
            assert "terok-sandbox" in str(result)


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

    def test_default_contains_app_name(self):
        """Default path uses the terok-sandbox namespace."""
        with unittest.mock.patch.dict(os.environ, {}, clear=True):
            result = state_root()
            assert "terok-sandbox" in str(result)


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

    def test_default_contains_app_name(self):
        """Default path uses the terok-sandbox namespace."""
        with unittest.mock.patch.dict(os.environ, {}, clear=True):
            result = runtime_root()
            assert "terok-sandbox" in str(result)


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

    def test_default_contains_credentials_app_name(self):
        """Default path uses the terok-credentials namespace."""
        with unittest.mock.patch.dict(os.environ, {}, clear=True):
            result = credentials_root()
            assert "terok-credentials" in str(result)

    def test_env_override_with_tilde(self):
        """TEROK_CREDENTIALS_DIR supports ~ expansion."""
        with unittest.mock.patch.dict(os.environ, {"TEROK_CREDENTIALS_DIR": "~/my-creds"}):
            result = credentials_root()
            assert "~" not in str(result)
            assert result.is_absolute()

    def test_root_user_fallback(self):
        """When running as root, falls back to /var/lib/terok-credentials."""
        with (
            unittest.mock.patch.dict(os.environ, {}, clear=True),
            unittest.mock.patch("terok_sandbox.paths._is_root", return_value=True),
        ):
            assert credentials_root() == Path("/var/lib/terok-credentials")
