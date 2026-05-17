# SPDX-FileCopyrightText: 2026 Jiri Vyskocil
# SPDX-License-Identifier: Apache-2.0

"""Tests for the explicit ``--passphrase-tier`` override.

The chooser auto-detect path stays untouched for desktop installs.
The new explicit-tier knob exists for headless / CI bootstraps where
the silent ``session-file`` fall-through was removed (mint-without-
reveal → lost recovery key on the first reboot).
"""

from __future__ import annotations

from pathlib import Path

import pytest

from terok_sandbox import SandboxConfig
from terok_sandbox.commands import _handle_credentials_encrypt_db


def _real_load_from_file(path: Path) -> str | None:
    """Bypass the conftest stub of ``load_passphrase_from_file``.

    The autouse fixture in ``conftest.py`` replaces the function with a
    null-returning stub for the entire test session; tests that need to
    exercise the file tier restore the real reader via monkeypatch.
    Inlining the implementation here keeps the test independent of
    private symbols on the encryption module.
    """
    try:
        return path.read_text(encoding="utf-8").rstrip("\n") or None
    except OSError:
        return None


def _cfg(tmp_path: Path, *, passphrase: str | None = None) -> SandboxConfig:
    """Sandbox config with the keyring tier turned off (avoid host keyring leakage)."""
    return SandboxConfig(
        state_dir=tmp_path / "state",
        runtime_dir=tmp_path / "rt",
        config_dir=tmp_path / "cfg",
        vault_dir=tmp_path / "vault",
        services_mode="socket",
        credentials_passphrase=passphrase,
        credentials_use_keyring=False,
    )


class TestExplicitTier:
    """Plumbing for the ``--passphrase-tier`` knob."""

    def test_unknown_tier_exits(self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
        """A typo must hard-fail with the allowed vocabulary in the message."""
        monkeypatch.setattr("terok_sandbox.vault.store.systemd_creds.is_available", lambda: False)
        with pytest.raises(SystemExit, match="unknown --passphrase-tier"):
            _handle_credentials_encrypt_db(cfg=_cfg(tmp_path), passphrase_tier="bogus")

    def test_systemd_creds_refused_when_unavailable(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Explicit systemd-creds choice still requires the host to have it."""
        monkeypatch.setattr("terok_sandbox.vault.store.systemd_creds.is_available", lambda: False)
        with pytest.raises(SystemExit, match="systemd-creds is\\s*unavailable"):
            _handle_credentials_encrypt_db(cfg=_cfg(tmp_path), passphrase_tier="systemd-creds")

    def test_session_tier_uses_existing_file(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """An explicit ``session-file`` tier picks up an existing tmpfs file silently."""
        from terok_sandbox.vault.store import encryption as enc

        monkeypatch.setattr("terok_sandbox.vault.store.systemd_creds.is_available", lambda: False)
        # Undo conftest's autouse stub so the real reader runs against
        # the file we just dropped on disk.
        monkeypatch.setattr(
            enc,
            "load_passphrase_from_file",
            enc.load_passphrase_from_file.__wrapped__
            if hasattr(enc.load_passphrase_from_file, "__wrapped__")
            else _real_load_from_file,
        )
        cfg = _cfg(tmp_path)
        cfg.vault_passphrase_file.parent.mkdir(parents=True, exist_ok=True, mode=0o700)
        cfg.vault_passphrase_file.write_text("preset-passphrase\n")
        # No DB → handler short-circuits after provisioning; no ack
        # required because the value pre-existed (not auto-generated).
        _handle_credentials_encrypt_db(cfg=cfg, passphrase_tier="session-file")
        assert cfg.vault_passphrase_file.read_text() == "preset-passphrase\n"
        assert not cfg.vault_recovery_marker_file.exists()


class TestNonTtyRefusalFromSetup:
    """End-to-end: setup without TTY and without --passphrase-tier hard-fails."""

    def test_non_tty_chooser_path_refuses(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """The historical silent ``session-file`` default is gone."""
        monkeypatch.setattr("terok_sandbox.vault.store.systemd_creds.is_available", lambda: False)
        monkeypatch.setattr("sys.stdin.isatty", lambda: False)
        with pytest.raises(SystemExit, match="--passphrase-tier"):
            _handle_credentials_encrypt_db(cfg=_cfg(tmp_path))


class TestExplicitConfigTier:
    """``--passphrase-tier=config`` still gates on the plaintext-on-disk confirmation."""

    def test_config_tier_requires_yes_confirmation(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Operator who says ``no`` at the plaintext gate is told to pick another tier."""
        monkeypatch.setattr("terok_sandbox.vault.store.systemd_creds.is_available", lambda: False)
        # Operator declines the plaintext warning — pipe ``no`` into stdin.
        monkeypatch.setattr("sys.stdin.readline", lambda: "no\n")
        with pytest.raises(SystemExit, match="config tier not confirmed"):
            _handle_credentials_encrypt_db(
                cfg=_cfg(tmp_path, passphrase="hunter2"), passphrase_tier="config"
            )

    def test_config_tier_accepts_yes(self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
        """``yes`` lands the config tier; pre-existing passphrase is reused."""
        monkeypatch.setattr("terok_sandbox.vault.store.systemd_creds.is_available", lambda: False)
        monkeypatch.setattr("sys.stdin.readline", lambda: "yes\n")
        cfg = _cfg(tmp_path, passphrase="hunter2")
        # No DB → handler short-circuits after provisioning.  No ack
        # required because the config value pre-existed.
        _handle_credentials_encrypt_db(cfg=cfg, passphrase_tier="config")
        assert not cfg.vault_recovery_marker_file.exists()
