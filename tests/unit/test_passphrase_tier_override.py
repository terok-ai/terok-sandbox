# SPDX-FileCopyrightText: 2026 Jiri Vyskocil
# SPDX-License-Identifier: Apache-2.0

"""Tests for the explicit ``--passphrase-tier`` override.

The chooser auto-detect path stays untouched for desktop installs.
The new explicit-tier knob exists for headless / CI bootstraps where
the silent volatile-tier fall-through was removed (mint-without-
reveal → lost recovery key on the first logout).
"""

from __future__ import annotations

from pathlib import Path

import pytest

import terok_sandbox.vault.store.kernel_keyring as _kk
from terok_sandbox import SandboxConfig
from terok_sandbox.commands import _handle_credentials_encrypt_db


def _cfg(tmp_path: Path) -> SandboxConfig:
    """Sandbox config with the keyring tier turned off (avoid host keyring leakage)."""
    return SandboxConfig(
        state_dir=tmp_path / "state",
        runtime_dir=tmp_path / "rt",
        config_dir=tmp_path / "cfg",
        vault_dir=tmp_path / "vault",
        services_mode="socket",
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

    def test_kernel_keyring_tier_uses_existing_cache(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """An explicit ``kernel-keyring`` tier picks up an existing cached value silently."""
        monkeypatch.setattr("terok_sandbox.vault.store.systemd_creds.is_available", lambda: False)
        # Back the kernel-keyring tier with an in-memory cache already
        # holding a value, undoing conftest's autouse blank.
        cache = {"pw": "preset-passphrase"}
        monkeypatch.setattr(_kk, "load", lambda: cache["pw"])
        monkeypatch.setattr(_kk, "store", lambda pw, **_kw: cache.__setitem__("pw", pw) or True)
        cfg = _cfg(tmp_path)
        # No DB → handler short-circuits after provisioning; no ack
        # required because the value pre-existed (not auto-generated).
        _handle_credentials_encrypt_db(cfg=cfg, passphrase_tier="kernel-keyring")
        assert cache["pw"] == "preset-passphrase"
        assert not cfg.vault_recovery_marker_file.exists()


class TestNonTtyRefusalFromSetup:
    """End-to-end: setup without TTY and without --passphrase-tier hard-fails."""

    def test_non_tty_chooser_path_refuses(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """The historical silent volatile-tier default is gone."""
        monkeypatch.setattr("terok_sandbox.vault.store.systemd_creds.is_available", lambda: False)
        monkeypatch.setattr("sys.stdin.isatty", lambda: False)
        with pytest.raises(SystemExit, match="--passphrase-tier"):
            _handle_credentials_encrypt_db(cfg=_cfg(tmp_path))


class TestExplicitSystemdCredsTier:
    """``--passphrase-tier=systemd-creds`` with the host actually supporting it."""

    def test_systemd_creds_tier_mints_and_seals(
        self,
        tmp_path: Path,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        """When systemd-creds IS available, explicit choice mints + seals + acks via TTY."""
        from terok_sandbox.vault.store import systemd_creds as _sc

        monkeypatch.setattr(_sc, "is_available", lambda: True)
        sealed: list[tuple[str, object, str]] = []

        def _seal(passphrase: str, path: object, *, key_mode: str) -> None:
            sealed.append((passphrase, path, key_mode))

        monkeypatch.setattr(_sc, "seal", _seal)
        # Block both directions of /dev/tty so the ack flow degrades to
        # "no controlling TTY" rather than fishing for SAVED input.
        from pathlib import Path as _Path

        real_open = _Path.open

        def _no_tty(self, *args, **kwargs):
            if str(self) == "/dev/tty":
                raise OSError("no /dev/tty in test")
            return real_open(self, *args, **kwargs)

        monkeypatch.setattr(_Path, "open", _no_tty)

        cfg = _cfg(tmp_path)
        _handle_credentials_encrypt_db(
            cfg=cfg, passphrase_tier="systemd-creds", echo_passphrase=True
        )
        # One mint → one seal call with --with-key=auto.
        assert len(sealed) == 1
        assert sealed[0][1] == cfg.vault_systemd_creds_file
        assert sealed[0][2] == "auto"
