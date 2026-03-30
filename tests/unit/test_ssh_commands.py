# SPDX-FileCopyrightText: 2026 Jiri Vyskocil
# SPDX-License-Identifier: Apache-2.0

"""Tests for SSH CLI command handlers."""

from __future__ import annotations

import json
from pathlib import Path
from unittest.mock import patch

import pytest
from cryptography.hazmat.primitives.asymmetric.ed25519 import Ed25519PrivateKey
from cryptography.hazmat.primitives.serialization import (
    Encoding,
    NoEncryption,
    PrivateFormat,
    PublicFormat,
)

from terok_sandbox.commands import _handle_ssh_import


@pytest.fixture()
def keypair(tmp_path: Path) -> tuple[Path, Path]:
    """Generate an ed25519 keypair and return (priv_path, pub_path)."""
    key = Ed25519PrivateKey.generate()
    priv = tmp_path / "id_ed25519_proj"
    pub = tmp_path / "id_ed25519_proj.pub"
    priv.write_bytes(key.private_bytes(Encoding.PEM, PrivateFormat.OpenSSH, NoEncryption()))
    pub_raw = key.public_key().public_bytes(Encoding.OpenSSH, PublicFormat.OpenSSH)
    pub.write_text(f"{pub_raw.decode()} proj-comment\n")
    return priv, pub


class TestHandleSshImport:
    """Verify _handle_ssh_import registers the keypair in ssh-keys.json."""

    def test_registers_key(
        self, tmp_path: Path, keypair: tuple[Path, Path], capsys: pytest.CaptureFixture[str]
    ) -> None:
        """Registers key and prints confirmation."""
        priv, pub = keypair
        keys_path = tmp_path / "ssh-keys.json"

        with patch("terok_sandbox.config.SandboxConfig") as mock_cfg:
            mock_cfg.return_value.ssh_keys_json_path = keys_path
            _handle_ssh_import(project="myproj", private_key=str(priv))

        data = json.loads(keys_path.read_text())
        assert data == {"myproj": [{"private_key": str(priv), "public_key": str(pub)}]}
        assert "myproj" in capsys.readouterr().out

    def test_explicit_public_key(self, tmp_path: Path, keypair: tuple[Path, Path]) -> None:
        """Accepts an explicit --public-key path."""
        priv, pub = keypair
        keys_path = tmp_path / "ssh-keys.json"

        with patch("terok_sandbox.config.SandboxConfig") as mock_cfg:
            mock_cfg.return_value.ssh_keys_json_path = keys_path
            _handle_ssh_import(project="proj", private_key=str(priv), public_key=str(pub))

        data = json.loads(keys_path.read_text())
        assert data["proj"][0]["public_key"] == str(pub)

    def test_missing_private_key_exits(self, tmp_path: Path) -> None:
        """Missing private key raises SystemExit."""
        with pytest.raises(SystemExit, match="not found"):
            _handle_ssh_import(project="proj", private_key=str(tmp_path / "no-such"))

    def test_missing_public_key_exits(self, tmp_path: Path, keypair: tuple[Path, Path]) -> None:
        """Private key exists but .pub is missing raises SystemExit."""
        priv, pub = keypair
        pub.unlink()
        with pytest.raises(SystemExit, match="not found"):
            _handle_ssh_import(project="proj", private_key=str(priv))

    def test_second_key_appends(self, tmp_path: Path, keypair: tuple[Path, Path]) -> None:
        """Importing a second key for the same project appends to the list."""
        priv, pub = keypair
        keys_path = tmp_path / "ssh-keys.json"

        key2 = Ed25519PrivateKey.generate()
        priv2 = tmp_path / "id_ed25519_proj_alt"
        pub2 = tmp_path / "id_ed25519_proj_alt.pub"
        priv2.write_bytes(key2.private_bytes(Encoding.PEM, PrivateFormat.OpenSSH, NoEncryption()))
        pub_raw2 = key2.public_key().public_bytes(Encoding.OpenSSH, PublicFormat.OpenSSH)
        pub2.write_text(f"{pub_raw2.decode()} alt\n")

        with patch("terok_sandbox.config.SandboxConfig") as mock_cfg:
            mock_cfg.return_value.ssh_keys_json_path = keys_path
            _handle_ssh_import(project="proj", private_key=str(priv))
            _handle_ssh_import(project="proj", private_key=str(priv2))

        data = json.loads(keys_path.read_text())
        assert len(data["proj"]) == 2
