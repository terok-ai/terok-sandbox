# SPDX-FileCopyrightText: 2025 Jiri Vyskocil
# SPDX-FileCopyrightText: 2026 Jiri Vyskocil
# SPDX-License-Identifier: Apache-2.0

"""Tests for the sandbox-specific path helpers in ``terok_sandbox.paths``.

Generic namespace resolvers (``namespace_state_dir`` and friends) and
the layered-config readers (``read_config_section``,
``read_config_top_level``) are exercised in ``terok-util``'s own test
suite — they live there now.  This file only covers what stays in
sandbox: the thin wrappers (``config_root``, ``state_root``, …) and
the ``plaintext_passphrase_config_path`` walker.
"""

from __future__ import annotations

import os
import unittest.mock
from pathlib import Path

import pytest

from terok_sandbox.paths import (
    config_root,
    namespace_config_root,
    runtime_root,
    state_root,
    vault_root,
)
from tests.constants import MOCK_BASE

MOCK_VAULT_DIR = MOCK_BASE / "vault"

_NAMESPACE_SEP = "terok" + os.sep + "sandbox"
"""Expected namespace path segment for sandbox roots (``terok/sandbox``)."""

_VAULT_NAMESPACE_SEP = "terok" + os.sep + "vault"
"""Expected namespace path segment for vault root (``terok/vault``)."""


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
            unittest.mock.patch("terok_util.paths._is_root", return_value=True),
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
            unittest.mock.patch("terok_util.paths._is_root", return_value=True),
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
            unittest.mock.patch("terok_util.paths._is_root", return_value=True),
        ):
            assert runtime_root() == Path("/run/terok/sandbox")


class TestVaultRoot:
    """Tests for vault_root()."""

    def test_returns_path(self):
        """vault_root() returns a Path instance."""
        assert isinstance(vault_root(), Path)

    def test_env_override(self):
        """TEROK_VAULT_DIR env var takes precedence."""
        with unittest.mock.patch.dict(os.environ, {"TEROK_VAULT_DIR": str(MOCK_VAULT_DIR)}):
            assert vault_root() == MOCK_VAULT_DIR

    def test_default_uses_namespace(self):
        """Default path nests under the terok/ namespace."""
        with unittest.mock.patch.dict(os.environ, {}, clear=True):
            result = vault_root()
            assert _VAULT_NAMESPACE_SEP in str(result)

    def test_env_override_with_tilde(self):
        """TEROK_VAULT_DIR supports ~ expansion."""
        with unittest.mock.patch.dict(os.environ, {"TEROK_VAULT_DIR": "~/my-vault"}):
            result = vault_root()
            assert "~" not in str(result)
            assert result.is_absolute()

    def test_root_user_fallback(self):
        """When running as root, falls back to /var/lib/terok/vault."""
        with (
            unittest.mock.patch.dict(os.environ, {}, clear=True),
            unittest.mock.patch("terok_util.paths._is_root", return_value=True),
        ):
            assert vault_root() == Path("/var/lib/terok/vault")

    def test_legacy_credentials_env_var_fallback(self, tmp_path: Path) -> None:
        """``TEROK_CREDENTIALS_DIR`` is honoured (with deprecation warning) when
        ``TEROK_VAULT_DIR`` is unset — preserves pre-rename installations."""
        legacy = tmp_path / "old-credentials"
        env = {k: v for k, v in os.environ.items() if k != "TEROK_VAULT_DIR"}
        env["TEROK_CREDENTIALS_DIR"] = str(legacy)
        with unittest.mock.patch.dict(os.environ, env, clear=True):
            result = vault_root()
        assert result == legacy

    def test_legacy_credentials_env_var_expands_tilde(self) -> None:
        """``TEROK_CREDENTIALS_DIR`` supports ``~`` expansion like the new var."""
        env = {k: v for k, v in os.environ.items() if k != "TEROK_VAULT_DIR"}
        env["TEROK_CREDENTIALS_DIR"] = "~/legacy-creds"
        with unittest.mock.patch.dict(os.environ, env, clear=True):
            result = vault_root()
        assert "~" not in str(result)
        assert result.is_absolute()

    def test_new_var_takes_precedence_over_legacy(self, tmp_path: Path) -> None:
        """When both env vars are set, the new ``TEROK_VAULT_DIR`` wins."""
        new_path = tmp_path / "new-vault"
        legacy_path = tmp_path / "old-credentials"
        env = {
            "TEROK_VAULT_DIR": str(new_path),
            "TEROK_CREDENTIALS_DIR": str(legacy_path),
        }
        with unittest.mock.patch.dict(os.environ, env, clear=True):
            result = vault_root()
        assert result == new_path

    def test_legacy_credentials_dir_emits_warning(
        self, tmp_path: Path, caplog: pytest.LogCaptureFixture
    ) -> None:
        """A pre-0.8 ``credentials`` sibling next to the new vault path emits a
        warning pointing at the migration tool but still returns the new path."""
        # The default vault path nests under namespace_state_dir(); point both
        # vars at a tmp dir so we can plant a legacy "credentials" sibling.
        new_path = tmp_path / "vault"
        legacy_sibling = tmp_path / "credentials"
        legacy_sibling.mkdir()
        with (
            unittest.mock.patch.dict(os.environ, {}, clear=True),
            unittest.mock.patch("terok_sandbox.paths.namespace_state_dir", return_value=new_path),
            caplog.at_level("WARNING", logger="terok_sandbox.paths"),
        ):
            result = vault_root()
        assert result == new_path
        # Warning mentions the legacy directory and the migration tool.
        joined = " ".join(rec.message for rec in caplog.records)
        assert "credentials" in joined
        assert "terok-migrate-vault" in joined


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
            unittest.mock.patch("terok_util.paths._is_root", return_value=True),
        ):
            assert namespace_config_root() == Path("/etc/terok")


class TestPlaintextPassphraseConfigPath:
    """sandbox#282 helper that locates ``credentials.passphrase`` in the layered config."""

    def test_returns_none_when_field_absent(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """No ``credentials.passphrase`` anywhere → ``None``."""
        from terok_sandbox.paths import plaintext_passphrase_config_path

        cfg = tmp_path / "config.yml"
        cfg.write_text("credentials:\n  use_keyring: true\n", encoding="utf-8")
        monkeypatch.setattr(
            "terok_sandbox.paths.config_file_paths",
            lambda: [("user", cfg)],
        )
        assert plaintext_passphrase_config_path() is None

    def test_finds_field_in_user_config(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Field set in user config → that path."""
        from terok_sandbox.paths import plaintext_passphrase_config_path

        cfg = tmp_path / "user" / "config.yml"
        cfg.parent.mkdir(parents=True)
        cfg.write_text("credentials:\n  passphrase: hunter2\n", encoding="utf-8")
        monkeypatch.setattr(
            "terok_sandbox.paths.config_file_paths",
            lambda: [("user", cfg)],
        )
        assert plaintext_passphrase_config_path() == cfg

    def test_user_wins_over_system(self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
        """Both layers set the field → the higher-priority (last-walked) one wins."""
        from terok_sandbox.paths import plaintext_passphrase_config_path

        system = tmp_path / "system.yml"
        user = tmp_path / "user.yml"
        system.write_text("credentials:\n  passphrase: sys-value\n", encoding="utf-8")
        user.write_text("credentials:\n  passphrase: user-value\n", encoding="utf-8")
        monkeypatch.setattr(
            "terok_sandbox.paths.config_file_paths",
            lambda: [("system", system), ("user", user)],
        )
        assert plaintext_passphrase_config_path() == user

    def test_bad_layer_is_swallowed(self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
        """A malformed YAML layer doesn't kill the walk — visibility surfaces must not crash."""
        from terok_sandbox.paths import plaintext_passphrase_config_path

        broken = tmp_path / "broken.yml"
        good = tmp_path / "good.yml"
        # ``: : :`` is unambiguously not valid YAML; the parse error must
        # NOT propagate out of the helper.
        broken.write_text(": : :\n", encoding="utf-8")
        good.write_text("credentials:\n  passphrase: visible\n", encoding="utf-8")
        monkeypatch.setattr(
            "terok_sandbox.paths.config_file_paths",
            lambda: [("system", broken), ("user", good)],
        )
        assert plaintext_passphrase_config_path() == good
