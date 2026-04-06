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

from terok_sandbox.commands import _handle_ssh_add_key, _handle_ssh_import, _handle_ssh_list
from terok_sandbox.credentials.ssh import _next_key_number, generate_keypair


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

        _handle_ssh_import(project="myproj", private_key=str(priv_src), cfg=cfg)

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

        _handle_ssh_import(project="myproj", private_key=str(priv_src), cfg=cfg)

        priv_dst = cfg.ssh_keys_dir / "myproj" / priv_src.name
        assert oct(priv_dst.stat().st_mode & 0o777) == oct(0o600)

    def test_explicit_public_key(self, tmp_path: Path, keypair: tuple[Path, Path]) -> None:
        """Accepts an explicit --public-key path and copies it."""
        priv_src, pub_src = keypair
        cfg = _mock_cfg(tmp_path)

        _handle_ssh_import(
            project="proj", private_key=str(priv_src), public_key=str(pub_src), cfg=cfg
        )

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

        _handle_ssh_import(project="proj", private_key=str(priv_src), cfg=cfg)

        assert priv_src.read_bytes() == priv_orig
        assert pub_src.read_text() == pub_orig

    def test_missing_private_key_exits(self, tmp_path: Path) -> None:
        """Missing private key raises SystemExit."""
        cfg = _mock_cfg(tmp_path)
        with pytest.raises(SystemExit, match="not found"):
            _handle_ssh_import(project="proj", private_key=str(tmp_path / "no-such"), cfg=cfg)

    def test_missing_public_key_exits(self, tmp_path: Path, keypair: tuple[Path, Path]) -> None:
        """Private key exists but .pub is missing raises SystemExit."""
        priv, pub = keypair
        pub.unlink()
        cfg = _mock_cfg(tmp_path)
        with pytest.raises(SystemExit, match="not found"):
            _handle_ssh_import(project="proj", private_key=str(priv), cfg=cfg)

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

        _handle_ssh_import(project="proj", private_key=str(priv), cfg=cfg)
        _handle_ssh_import(project="proj", private_key=str(priv2), cfg=cfg)

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

        _handle_ssh_import(project="proj", private_key=str(src_a / "id_ed25519_proj"), cfg=cfg)
        _handle_ssh_import(project="proj", private_key=str(src_b / "id_ed25519_proj"), cfg=cfg)

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

        _handle_ssh_import(project="proj", private_key=str(priv_src), cfg=cfg)
        _handle_ssh_import(project="proj", private_key=str(priv_src), cfg=cfg)

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
        with pytest.raises(SystemExit, match="Invalid project ID"):
            _handle_ssh_import(project=bad_project, private_key=str(priv_src), cfg=cfg)

    def test_standalone_fallback_without_cfg(
        self, tmp_path: Path, keypair: tuple[Path, Path]
    ) -> None:
        """Handler creates SandboxConfig internally when cfg=None (standalone)."""
        priv_src, _pub = keypair
        cfg = _mock_cfg(tmp_path)

        with patch("terok_sandbox.config.SandboxConfig", return_value=cfg):
            _handle_ssh_import(project="proj", private_key=str(priv_src))

        dest_dir = cfg.ssh_keys_dir / "proj"
        assert (dest_dir / priv_src.name).is_file()


# ---------------------------------------------------------------------------
# _next_key_number
# ---------------------------------------------------------------------------


class TestNextKeyNumber:
    """Verify auto-numbering logic for side keys."""

    def test_empty_dir_returns_1(self, tmp_path: Path) -> None:
        """Empty project directory starts numbering at 1."""
        assert _next_key_number(tmp_path, "ed25519") == 1

    def test_nonexistent_dir_returns_1(self, tmp_path: Path) -> None:
        """Non-existent directory starts numbering at 1."""
        assert _next_key_number(tmp_path / "nope", "ed25519") == 1

    def test_single_key_returns_2(self, tmp_path: Path) -> None:
        """With key-1 present, next is key-2."""
        (tmp_path / "id_ed25519_key-1").touch()
        assert _next_key_number(tmp_path, "ed25519") == 2

    def test_non_contiguous_returns_max_plus_1(self, tmp_path: Path) -> None:
        """With key-1 and key-5, next is key-6 (no gap-filling)."""
        (tmp_path / "id_ed25519_key-1").touch()
        (tmp_path / "id_ed25519_key-5").touch()
        assert _next_key_number(tmp_path, "ed25519") == 6

    def test_ignores_other_algo(self, tmp_path: Path) -> None:
        """RSA numbered keys are ignored when scanning for ed25519."""
        (tmp_path / "id_rsa_key-3").touch()
        assert _next_key_number(tmp_path, "ed25519") == 1

    def test_ignores_non_matching_files(self, tmp_path: Path) -> None:
        """Config files and main keys are ignored."""
        (tmp_path / "config").touch()
        (tmp_path / "id_ed25519_myproject").touch()
        (tmp_path / "id_ed25519_key-2.pub").touch()  # pub suffix doesn't match
        assert _next_key_number(tmp_path, "ed25519") == 1

    def test_rsa_scan(self, tmp_path: Path) -> None:
        """Scans RSA keys when algo is 'rsa'."""
        (tmp_path / "id_rsa_key-1").touch()
        (tmp_path / "id_rsa_key-2").touch()
        assert _next_key_number(tmp_path, "rsa") == 3


# ---------------------------------------------------------------------------
# ssh add-key
# ---------------------------------------------------------------------------


def _fake_keygen(tmp_path: Path):
    """Return a side_effect for subprocess.run that creates fake key files."""

    def _side_effect(cmd, **_kwargs):
        if cmd[0] != "ssh-keygen":
            return None
        args = dict(zip(cmd[1::2], cmd[2::2], strict=False))
        priv = Path(args["-f"])
        comment = args.get("-C", "")
        priv.write_text("FAKE-PRIVATE-KEY\n")
        Path(f"{priv}.pub").write_text(f"ssh-ed25519 AAAA... {comment}\n")
        return None

    return _side_effect


class TestHandleSshAddKey:
    """Verify _handle_ssh_add_key generates keypairs and registers them."""

    def test_generates_with_explicit_name(
        self, tmp_path: Path, capsys: pytest.CaptureFixture[str]
    ) -> None:
        """Generates a key with the given --name and correct comment."""
        cfg = _mock_cfg(tmp_path)
        with patch(
            "terok_sandbox.credentials.ssh.subprocess.run", side_effect=_fake_keygen(tmp_path)
        ):
            _handle_ssh_add_key(project="myproj", name="deploy-gitlab", cfg=cfg)

        dest_dir = cfg.ssh_keys_dir / "myproj"
        assert (dest_dir / "id_ed25519_deploy-gitlab").is_file()
        assert (dest_dir / "id_ed25519_deploy-gitlab.pub").is_file()

        data = json.loads(cfg.ssh_keys_json_path.read_text())
        assert len(data["myproj"]) == 1
        assert data["myproj"][0]["private_key"] == str(dest_dir / "id_ed25519_deploy-gitlab")

        out = capsys.readouterr().out
        assert "deploy-gitlab" in out
        assert "tk-side:myproj:deploy-gitlab" in out

    def test_auto_numbering_key_1(self, tmp_path: Path) -> None:
        """Without --name, first key is key-1."""
        cfg = _mock_cfg(tmp_path)
        with patch(
            "terok_sandbox.credentials.ssh.subprocess.run", side_effect=_fake_keygen(tmp_path)
        ):
            _handle_ssh_add_key(project="proj", cfg=cfg)

        dest_dir = cfg.ssh_keys_dir / "proj"
        assert (dest_dir / "id_ed25519_key-1").is_file()

    def test_auto_numbering_increments(self, tmp_path: Path) -> None:
        """Second call auto-generates key-2."""
        cfg = _mock_cfg(tmp_path)
        with patch(
            "terok_sandbox.credentials.ssh.subprocess.run", side_effect=_fake_keygen(tmp_path)
        ):
            _handle_ssh_add_key(project="proj", cfg=cfg)
            _handle_ssh_add_key(project="proj", cfg=cfg)

        dest_dir = cfg.ssh_keys_dir / "proj"
        assert (dest_dir / "id_ed25519_key-1").is_file()
        assert (dest_dir / "id_ed25519_key-2").is_file()

    def test_rsa_key_type(self, tmp_path: Path) -> None:
        """RSA key type produces id_rsa_ filename prefix."""
        cfg = _mock_cfg(tmp_path)
        with patch(
            "terok_sandbox.credentials.ssh.subprocess.run", side_effect=_fake_keygen(tmp_path)
        ):
            _handle_ssh_add_key(project="proj", key_type="rsa", name="my-key", cfg=cfg)

        assert (cfg.ssh_keys_dir / "proj" / "id_rsa_my-key").is_file()

    def test_comment_format(self, tmp_path: Path) -> None:
        """ssh-keygen is called with the correct tk-side: comment."""
        cfg = _mock_cfg(tmp_path)
        captured_cmds: list[list[str]] = []

        def _capture(cmd, **_kwargs):
            captured_cmds.append(list(cmd))
            _fake_keygen(tmp_path)(cmd)

        with patch("terok_sandbox.credentials.ssh.subprocess.run", side_effect=_capture):
            _handle_ssh_add_key(project="proj", name="deploy", cfg=cfg)

        assert len(captured_cmds) == 1
        args = dict(zip(captured_cmds[0][1::2], captured_cmds[0][2::2], strict=False))
        assert args["-C"] == "tk-side:proj:deploy"

    def test_existing_key_refuses_overwrite(self, tmp_path: Path) -> None:
        """Refuses to overwrite an existing key file."""
        cfg = _mock_cfg(tmp_path)
        dest_dir = cfg.ssh_keys_dir / "proj"
        dest_dir.mkdir(parents=True)
        (dest_dir / "id_ed25519_my-key").touch()

        with patch(
            "terok_sandbox.credentials.ssh.subprocess.run", side_effect=_fake_keygen(tmp_path)
        ):
            with pytest.raises(SystemExit, match="already exists"):
                _handle_ssh_add_key(project="proj", name="my-key", cfg=cfg)

    def test_existing_pub_key_refuses_overwrite(self, tmp_path: Path) -> None:
        """A lone .pub file also prevents generation."""
        cfg = _mock_cfg(tmp_path)
        dest_dir = cfg.ssh_keys_dir / "proj"
        dest_dir.mkdir(parents=True)
        (dest_dir / "id_ed25519_my-key.pub").touch()

        with patch(
            "terok_sandbox.credentials.ssh.subprocess.run", side_effect=_fake_keygen(tmp_path)
        ):
            with pytest.raises(SystemExit, match="already exists"):
                _handle_ssh_add_key(project="proj", name="my-key", cfg=cfg)

    def test_registers_in_ssh_keys_json(self, tmp_path: Path) -> None:
        """Generated key is registered in ssh-keys.json."""
        cfg = _mock_cfg(tmp_path)
        with patch(
            "terok_sandbox.credentials.ssh.subprocess.run", side_effect=_fake_keygen(tmp_path)
        ):
            _handle_ssh_add_key(project="proj", name="extra", cfg=cfg)

        data = json.loads(cfg.ssh_keys_json_path.read_text())
        entry = data["proj"][0]
        assert entry["private_key"].endswith("id_ed25519_extra")
        assert entry["public_key"].endswith("id_ed25519_extra.pub")

    @pytest.mark.parametrize("bad_name", ["has space", "with.dot", "123", "a/b", ""])
    def test_invalid_name_exits(self, tmp_path: Path, bad_name: str) -> None:
        """Names with invalid characters raise SystemExit."""
        cfg = _mock_cfg(tmp_path)
        with patch(
            "terok_sandbox.credentials.ssh.subprocess.run", side_effect=_fake_keygen(tmp_path)
        ):
            with pytest.raises(SystemExit, match="Invalid key name"):
                _handle_ssh_add_key(project="proj", name=bad_name, cfg=cfg)

    @pytest.mark.parametrize("good_name", ["deploy", "my-key", "DEPLOY", "my_key", "_private"])
    def test_valid_name_accepted(self, tmp_path: Path, good_name: str) -> None:
        """Alphanumeric, underscores, and hyphens are accepted."""
        cfg = _mock_cfg(tmp_path)
        with patch(
            "terok_sandbox.credentials.ssh.subprocess.run", side_effect=_fake_keygen(tmp_path)
        ):
            _handle_ssh_add_key(project="proj", name=good_name, cfg=cfg)

        assert (cfg.ssh_keys_dir / "proj" / f"id_ed25519_{good_name}").is_file()

    @pytest.mark.parametrize(
        "bad_project",
        ["../other", "dir/sub", "/absolute", "..", "", "has space"],
    )
    def test_invalid_project_exits(self, tmp_path: Path, bad_project: str) -> None:
        """Invalid project IDs raise SystemExit."""
        cfg = _mock_cfg(tmp_path)
        with patch(
            "terok_sandbox.credentials.ssh.subprocess.run", side_effect=_fake_keygen(tmp_path)
        ):
            with pytest.raises(SystemExit, match="Invalid project ID"):
                _handle_ssh_add_key(project=bad_project, name="key", cfg=cfg)

    def test_invalid_key_type_exits(self, tmp_path: Path) -> None:
        """Unsupported key type raises SystemExit."""
        cfg = _mock_cfg(tmp_path)
        with patch(
            "terok_sandbox.credentials.ssh.subprocess.run", side_effect=_fake_keygen(tmp_path)
        ):
            with pytest.raises(SystemExit, match="Unsupported --key-type"):
                _handle_ssh_add_key(project="proj", name="k", key_type="dsa", cfg=cfg)

    def test_permission_error_exits(self, tmp_path: Path) -> None:
        """OSError during permission hardening raises SystemExit."""
        cfg = _mock_cfg(tmp_path)
        with (
            patch(
                "terok_sandbox.credentials.ssh.subprocess.run", side_effect=_fake_keygen(tmp_path)
            ),
            patch(
                "terok_sandbox.credentials.ssh._harden_permissions",
                side_effect=OSError("perm denied"),
            ),
        ):
            with pytest.raises(SystemExit, match="Failed to set permissions"):
                _handle_ssh_add_key(project="proj", name="k", cfg=cfg)

    def test_pub_key_read_error_is_silent(
        self, tmp_path: Path, capsys: pytest.CaptureFixture[str]
    ) -> None:
        """Exception reading the public key for display does not abort."""
        cfg = _mock_cfg(tmp_path)

        def _keygen_then_remove_pub(cmd, **_kwargs):
            _fake_keygen(tmp_path)(cmd)
            # Remove the pub file after generation so the read fails
            pub = Path(f"{cmd[4]}.pub")
            pub.unlink()

        with patch(
            "terok_sandbox.credentials.ssh.subprocess.run", side_effect=_keygen_then_remove_pub
        ):
            _handle_ssh_add_key(project="proj", name="k", cfg=cfg)

        out = capsys.readouterr().out
        assert "SSH key generated" in out
        assert "Public key (add as deploy key)" not in out

    def test_standalone_fallback_without_cfg(self, tmp_path: Path) -> None:
        """Handler creates SandboxConfig internally when cfg=None (standalone)."""
        cfg = _mock_cfg(tmp_path)
        with (
            patch("terok_sandbox.config.SandboxConfig", return_value=cfg),
            patch(
                "terok_sandbox.credentials.ssh.subprocess.run", side_effect=_fake_keygen(tmp_path)
            ),
        ):
            _handle_ssh_add_key(project="proj", name="standalone")

        assert (cfg.ssh_keys_dir / "proj" / "id_ed25519_standalone").is_file()


# ---------------------------------------------------------------------------
# generate_keypair
# ---------------------------------------------------------------------------


class TestGenerateKeypair:
    """Verify generate_keypair error handling."""

    def test_missing_ssh_keygen_exits(self, tmp_path: Path) -> None:
        """FileNotFoundError from missing ssh-keygen raises SystemExit."""
        with patch("terok_sandbox.credentials.ssh.subprocess.run", side_effect=FileNotFoundError):
            with pytest.raises(SystemExit, match="ssh-keygen not found"):
                generate_keypair("ed25519", tmp_path / "k", tmp_path / "k.pub", "comment")

    def test_ssh_keygen_failure_exits(self, tmp_path: Path) -> None:
        """Non-zero exit from ssh-keygen raises SystemExit."""
        import subprocess as sp

        err = sp.CalledProcessError(1, "ssh-keygen")
        with patch("terok_sandbox.credentials.ssh.subprocess.run", side_effect=err):
            with pytest.raises(SystemExit, match="ssh-keygen failed"):
                generate_keypair("ed25519", tmp_path / "k", tmp_path / "k.pub", "comment")

    def test_removes_stale_files(self, tmp_path: Path) -> None:
        """Stale key files are removed before generation."""
        priv = tmp_path / "stale_key"
        pub = tmp_path / "stale_key.pub"
        priv.write_text("old")
        pub.write_text("old")

        with patch(
            "terok_sandbox.credentials.ssh.subprocess.run", side_effect=_fake_keygen(tmp_path)
        ):
            generate_keypair("ed25519", priv, pub, "fresh")

        assert priv.read_text() != "old"
        assert pub.read_text() != "old"


# ---------------------------------------------------------------------------
# ssh list
# ---------------------------------------------------------------------------


def _write_keys_json(cfg: object, data: dict) -> None:
    """Write *data* as the ssh-keys.json for *cfg*."""
    p = cfg.ssh_keys_json_path  # type: ignore[union-attr]
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps(data))


def _make_pub_file(path: Path, *, comment: str = "test-key") -> None:
    """Create a minimal ed25519 .pub file at *path*."""
    key = Ed25519PrivateKey.generate()
    pub_raw = key.public_key().public_bytes(Encoding.OpenSSH, PublicFormat.OpenSSH)
    path.write_text(f"{pub_raw.decode()} {comment}\n")


class TestHandleSshList:
    """Verify _handle_ssh_list output and filtering."""

    def test_no_json_file(self, tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
        """Missing ssh-keys.json prints 'No SSH keys registered.'."""
        cfg = _mock_cfg(tmp_path)
        _handle_ssh_list(cfg=cfg)
        assert "No SSH keys registered." in capsys.readouterr().out

    def test_empty_json(self, tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
        """Empty JSON object prints 'No SSH keys registered.'."""
        cfg = _mock_cfg(tmp_path)
        _write_keys_json(cfg, {})
        _handle_ssh_list(cfg=cfg)
        assert "No SSH keys registered." in capsys.readouterr().out

    def test_single_project(self, tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
        """One project with one key prints the table."""
        cfg = _mock_cfg(tmp_path)
        priv = tmp_path / "id_ed25519"
        pub = tmp_path / "id_ed25519.pub"
        priv.touch()
        _make_pub_file(pub, comment="tk-main:myproj")

        _write_keys_json(cfg, {"myproj": [{"private_key": str(priv), "public_key": str(pub)}]})
        _handle_ssh_list(cfg=cfg)

        out = capsys.readouterr().out
        assert "myproj" in out
        assert "tk-main:myproj" in out
        assert "ed25519" in out
        assert "SHA256:" in out

    def test_multiple_projects(self, tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
        """Two projects both appear in output."""
        cfg = _mock_cfg(tmp_path)
        rows = {}
        for pid in ("alpha", "beta"):
            d = tmp_path / pid
            d.mkdir()
            priv, pub = d / "id_ed25519", d / "id_ed25519.pub"
            priv.touch()
            _make_pub_file(pub, comment=f"tk-main:{pid}")
            rows[pid] = [{"private_key": str(priv), "public_key": str(pub)}]

        _write_keys_json(cfg, rows)
        _handle_ssh_list(cfg=cfg)

        out = capsys.readouterr().out
        assert "alpha" in out
        assert "beta" in out

    def test_filter_project(self, tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
        """--project filters to a single project."""
        cfg = _mock_cfg(tmp_path)
        rows = {}
        for pid in ("show-me", "hide-me"):
            d = tmp_path / pid
            d.mkdir()
            priv, pub = d / "id_ed25519", d / "id_ed25519.pub"
            priv.touch()
            _make_pub_file(pub, comment=f"tk-main:{pid}")
            rows[pid] = [{"private_key": str(priv), "public_key": str(pub)}]

        _write_keys_json(cfg, rows)
        _handle_ssh_list(project="show-me", cfg=cfg)

        out = capsys.readouterr().out
        assert "show-me" in out
        assert "hide-me" not in out

    def test_filter_nonexistent_project_exits(self, tmp_path: Path) -> None:
        """--project for unknown project raises SystemExit."""
        cfg = _mock_cfg(tmp_path)
        _write_keys_json(cfg, {"other": []})
        with pytest.raises(SystemExit, match="No keys registered"):
            _handle_ssh_list(project="ghost", cfg=cfg)

    def test_side_key_comment(self, tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
        """Side-key comments like 'tk-side:proj:keyname' are shown correctly."""
        cfg = _mock_cfg(tmp_path)
        priv = tmp_path / "id_ed25519_dbus"
        pub = tmp_path / "id_ed25519_dbus.pub"
        priv.touch()
        _make_pub_file(pub, comment="tk-side:terok:terok-dbus")

        _write_keys_json(cfg, {"terok": [{"private_key": str(priv), "public_key": str(pub)}]})
        _handle_ssh_list(cfg=cfg)

        out = capsys.readouterr().out
        assert "tk-side:terok:terok-dbus" in out

    def test_missing_pub_file(self, tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
        """Missing .pub file shows '(pub missing)' instead of crashing."""
        cfg = _mock_cfg(tmp_path)
        priv = tmp_path / "id_ed25519"
        priv.touch()

        _write_keys_json(
            cfg, {"proj": [{"private_key": str(priv), "public_key": str(tmp_path / "gone.pub")}]}
        )
        _handle_ssh_list(cfg=cfg)

        out = capsys.readouterr().out
        assert "(pub missing)" in out
        assert "proj" in out

    def test_corrupt_json_exits(self, tmp_path: Path) -> None:
        """Corrupt ssh-keys.json raises SystemExit."""
        cfg = _mock_cfg(tmp_path)
        p = cfg.ssh_keys_json_path
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text("{bad json")
        with pytest.raises(SystemExit, match="Cannot read"):
            _handle_ssh_list(cfg=cfg)

    def test_malformed_pub_file(self, tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
        """Pub file with invalid base64 shows '(error)' gracefully."""
        cfg = _mock_cfg(tmp_path)
        priv = tmp_path / "id_ed25519"
        pub = tmp_path / "id_ed25519.pub"
        priv.touch()
        pub.write_text("ssh-ed25519 !!!not-base64!!!\n")

        _write_keys_json(cfg, {"proj": [{"private_key": str(priv), "public_key": str(pub)}]})
        _handle_ssh_list(cfg=cfg)

        out = capsys.readouterr().out
        assert "(error)" in out

    def test_standalone_fallback(self, tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
        """Handler creates SandboxConfig when cfg=None."""
        cfg = _mock_cfg(tmp_path)
        _write_keys_json(cfg, {})
        with patch("terok_sandbox.config.SandboxConfig", return_value=cfg):
            _handle_ssh_list()
        assert "No SSH keys registered." in capsys.readouterr().out

    def test_all_empty_lists(self, tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
        """Projects with empty key lists print 'No SSH keys registered.'."""
        cfg = _mock_cfg(tmp_path)
        _write_keys_json(cfg, {"proj": []})
        _handle_ssh_list(cfg=cfg)
        assert "No SSH keys registered." in capsys.readouterr().out
