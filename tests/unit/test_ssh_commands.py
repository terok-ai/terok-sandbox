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


def _mock_cfg(tmp_path: Path, keys_subdir: str = "ssh-keys") -> object:
    """Return a mock SandboxConfig with paths in *tmp_path*."""
    from unittest.mock import MagicMock

    cfg = MagicMock()
    cfg.ssh_keys_dir = tmp_path / keys_subdir
    cfg.ssh_keys_json_path = tmp_path / "proxy" / "ssh-keys.json"
    return cfg


class TestHandleSshImport:
    """Verify _handle_ssh_import copies keypairs and registers them in ssh-keys.json."""

    def test_copies_and_registers_key(
        self, tmp_path: Path, keypair: tuple[Path, Path], capsys: pytest.CaptureFixture[str]
    ) -> None:
        """Copies both key files to the managed dir and registers them."""
        priv_src, pub_src = keypair
        cfg = _mock_cfg(tmp_path)

        with patch("terok_sandbox.config.SandboxConfig", return_value=cfg):
            _handle_ssh_import(project="myproj", private_key=str(priv_src))

        # Files must exist in the managed directory
        dest_dir = cfg.ssh_keys_dir / "myproj"
        priv_dst = dest_dir / priv_src.name
        pub_dst = dest_dir / pub_src.name
        assert priv_dst.is_file(), "Private key must be copied to managed dir"
        assert pub_dst.is_file(), "Public key must be copied to managed dir"

        # JSON must reference the copies, not the originals
        data = json.loads(cfg.ssh_keys_json_path.read_text())
        assert data == {"myproj": [{"private_key": str(priv_dst), "public_key": str(pub_dst)}]}

        assert "myproj" in capsys.readouterr().out

    def test_private_key_copy_has_600_permissions(
        self, tmp_path: Path, keypair: tuple[Path, Path]
    ) -> None:
        """Copied private key must have 0o600 permissions."""
        priv_src, _pub = keypair
        cfg = _mock_cfg(tmp_path)

        with patch("terok_sandbox.config.SandboxConfig", return_value=cfg):
            _handle_ssh_import(project="myproj", private_key=str(priv_src))

        priv_dst = cfg.ssh_keys_dir / "myproj" / priv_src.name
        assert oct(priv_dst.stat().st_mode & 0o777) == oct(0o600)

    def test_explicit_public_key(self, tmp_path: Path, keypair: tuple[Path, Path]) -> None:
        """Accepts an explicit --public-key path and copies it."""
        priv_src, pub_src = keypair
        cfg = _mock_cfg(tmp_path)

        with patch("terok_sandbox.config.SandboxConfig", return_value=cfg):
            _handle_ssh_import(project="proj", private_key=str(priv_src), public_key=str(pub_src))

        pub_dst = cfg.ssh_keys_dir / "proj" / pub_src.name
        assert pub_dst.is_file()
        data = json.loads(cfg.ssh_keys_json_path.read_text())
        assert data["proj"][0]["public_key"] == str(pub_dst)

    def test_original_files_not_modified(self, tmp_path: Path, keypair: tuple[Path, Path]) -> None:
        """Source key files must remain unchanged after import."""
        priv_src, pub_src = keypair
        priv_orig = priv_src.read_bytes()
        pub_orig = pub_src.read_text()
        cfg = _mock_cfg(tmp_path)

        with patch("terok_sandbox.config.SandboxConfig", return_value=cfg):
            _handle_ssh_import(project="proj", private_key=str(priv_src))

        assert priv_src.read_bytes() == priv_orig
        assert pub_src.read_text() == pub_orig

    def test_missing_private_key_exits(self, tmp_path: Path) -> None:
        """Missing private key raises SystemExit."""
        cfg = _mock_cfg(tmp_path)
        with patch("terok_sandbox.config.SandboxConfig", return_value=cfg):
            with pytest.raises(SystemExit, match="not found"):
                _handle_ssh_import(project="proj", private_key=str(tmp_path / "no-such"))

    def test_missing_public_key_exits(self, tmp_path: Path, keypair: tuple[Path, Path]) -> None:
        """Private key exists but .pub is missing raises SystemExit."""
        priv, pub = keypair
        pub.unlink()
        cfg = _mock_cfg(tmp_path)
        with patch("terok_sandbox.config.SandboxConfig", return_value=cfg):
            with pytest.raises(SystemExit, match="not found"):
                _handle_ssh_import(project="proj", private_key=str(priv))

    def test_second_key_appends(self, tmp_path: Path, keypair: tuple[Path, Path]) -> None:
        """Importing a second key for the same project appends to the list."""
        priv, pub = keypair
        cfg = _mock_cfg(tmp_path)

        key2 = Ed25519PrivateKey.generate()
        # Use a different source directory so filenames don't clash
        src2 = tmp_path / "keys2"
        src2.mkdir()
        priv2 = src2 / "id_ed25519_proj_alt"
        pub2 = src2 / "id_ed25519_proj_alt.pub"
        priv2.write_bytes(key2.private_bytes(Encoding.PEM, PrivateFormat.OpenSSH, NoEncryption()))
        pub_raw2 = key2.public_key().public_bytes(Encoding.OpenSSH, PublicFormat.OpenSSH)
        pub2.write_text(f"{pub_raw2.decode()} alt\n")

        with patch("terok_sandbox.config.SandboxConfig", return_value=cfg):
            _handle_ssh_import(project="proj", private_key=str(priv))
            _handle_ssh_import(project="proj", private_key=str(priv2))

        data = json.loads(cfg.ssh_keys_json_path.read_text())
        assert len(data["proj"]) == 2
        # Both entries must point into the managed dir
        dest_dir = cfg.ssh_keys_dir / "proj"
        for entry in data["proj"]:
            assert Path(entry["private_key"]).parent == dest_dir
            assert Path(entry["public_key"]).parent == dest_dir

    def test_same_basename_gets_numeric_suffix(self, tmp_path: Path) -> None:
        """Two keys with the same filename don't overwrite each other in the managed dir."""
        cfg = _mock_cfg(tmp_path)

        src_a = tmp_path / "a"
        src_b = tmp_path / "b"
        src_a.mkdir()
        src_b.mkdir()

        # Both source dirs have a key called id_ed25519_proj (same basename)
        for src_dir in (src_a, src_b):
            key = Ed25519PrivateKey.generate()
            priv = src_dir / "id_ed25519_proj"
            pub = src_dir / "id_ed25519_proj.pub"
            priv.write_bytes(key.private_bytes(Encoding.PEM, PrivateFormat.OpenSSH, NoEncryption()))
            pub_raw = key.public_key().public_bytes(Encoding.OpenSSH, PublicFormat.OpenSSH)
            pub.write_text(f"{pub_raw.decode()} comment\n")

        with patch("terok_sandbox.config.SandboxConfig", return_value=cfg):
            _handle_ssh_import(project="proj", private_key=str(src_a / "id_ed25519_proj"))
            _handle_ssh_import(project="proj", private_key=str(src_b / "id_ed25519_proj"))

        data = json.loads(cfg.ssh_keys_json_path.read_text())
        assert len(data["proj"]) == 2
        priv_paths = [Path(e["private_key"]).name for e in data["proj"]]
        # Second key must have a unique name (numeric suffix)
        assert len(set(priv_paths)) == 2, f"Expected distinct dest names, got: {priv_paths}"
        assert "id_ed25519_proj" in priv_paths
        assert "id_ed25519_proj_1" in priv_paths

    def test_reimport_same_key_is_idempotent(
        self, tmp_path: Path, keypair: tuple[Path, Path]
    ) -> None:
        """Re-importing the same key by path does not duplicate the entry."""
        priv_src, _pub = keypair
        cfg = _mock_cfg(tmp_path)

        with patch("terok_sandbox.config.SandboxConfig", return_value=cfg):
            _handle_ssh_import(project="proj", private_key=str(priv_src))
            _handle_ssh_import(project="proj", private_key=str(priv_src))

        data = json.loads(cfg.ssh_keys_json_path.read_text())
        assert len(data["proj"]) == 1

    @pytest.mark.parametrize(
        "bad_project",
        [
            "../other-dir",
            "dir/subdir",
            "/absolute",
            "..",
            "",
            "has space",
            "has\x00null",
        ],
    )
    def test_invalid_project_id_exits(
        self, tmp_path: Path, keypair: tuple[Path, Path], bad_project: str
    ) -> None:
        """Project IDs with path-traversal or invalid characters raise SystemExit."""
        priv_src, _pub = keypair
        cfg = _mock_cfg(tmp_path)
        with patch("terok_sandbox.config.SandboxConfig", return_value=cfg):
            with pytest.raises(SystemExit, match="Invalid project ID"):
                _handle_ssh_import(project=bad_project, private_key=str(priv_src))
