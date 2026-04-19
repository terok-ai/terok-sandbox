# SPDX-FileCopyrightText: 2026 Jiri Vyskocil
# SPDX-License-Identifier: Apache-2.0

"""Tests for :mod:`terok_sandbox.credentials.ssh_keypair` — generate, import, export."""

from __future__ import annotations

import os
import stat
from pathlib import Path

import pytest

from terok_sandbox.credentials.db import CredentialDB
from terok_sandbox.credentials.ssh_keypair import (
    KeypairMismatchError,
    PasswordProtectedKeyError,
    export_ssh_keypair,
    generate_keypair,
    import_ssh_keypair,
    parse_openssh_keypair,
)


@pytest.fixture()
def db(tmp_path: Path) -> CredentialDB:
    """Return a fresh DB rooted under a per-test tmp dir."""
    return CredentialDB(tmp_path / "vault" / "credentials.db")


@pytest.fixture()
def disk_keypair(tmp_path: Path) -> tuple[Path, Path]:
    """Generate a keypair in memory and write it to files; return (priv, pub)."""
    kp = generate_keypair("ed25519", comment="disk-comment")
    priv = tmp_path / "id_ed25519_disk"
    pub = tmp_path / "id_ed25519_disk.pub"
    priv.write_bytes(kp.private_pem)
    pub.write_text(kp.public_line + "\n")
    return priv, pub


class TestGenerate:
    """Verify in-memory keypair generation."""

    def test_ed25519_shape(self) -> None:
        """Generated ed25519 keypair carries PEM + wire blob + fingerprint."""
        kp = generate_keypair("ed25519", comment="hello")
        assert kp.key_type == "ed25519"
        assert b"OPENSSH PRIVATE KEY" in kp.private_pem
        assert len(kp.public_blob) > 0
        assert kp.public_line.startswith("ssh-ed25519 ")
        assert kp.public_line.endswith(" hello")
        assert len(kp.fingerprint) == 64  # sha256 hex
        assert kp.comment == "hello"

    def test_rsa_shape(self) -> None:
        """RSA generation works end-to-end (slower; smoke only)."""
        kp = generate_keypair("rsa", comment="rsa-c")
        assert kp.key_type == "rsa"
        assert kp.public_line.startswith("ssh-rsa ")

    def test_unknown_key_type_rejected(self) -> None:
        """Anything other than ed25519/rsa is rejected."""
        with pytest.raises(ValueError):
            generate_keypair("dsa", comment="c")

    def test_distinct_keys_have_distinct_fingerprints(self) -> None:
        """Two calls produce independent keys."""
        a = generate_keypair("ed25519", comment="a")
        b = generate_keypair("ed25519", comment="b")
        assert a.fingerprint != b.fingerprint


class TestImport:
    """Verify :func:`import_ssh_keypair` end-to-end behaviour."""

    def test_round_trip_with_public_file(
        self, db: CredentialDB, disk_keypair: tuple[Path, Path]
    ) -> None:
        """Importing priv+pub stores the key and assigns it to the scope."""
        priv, pub = disk_keypair
        result = import_ssh_keypair(db, "proj", priv, pub_path=pub)
        assert result.already_present is False
        assert len(result.fingerprint) == 64
        rows = db.list_ssh_keys_for_scope("proj")
        assert len(rows) == 1
        assert rows[0].id == result.key_id

    def test_derives_public_when_omitted(
        self, db: CredentialDB, disk_keypair: tuple[Path, Path]
    ) -> None:
        """Without a public file the public key is derived from the private key."""
        priv, _pub = disk_keypair
        result = import_ssh_keypair(db, "proj", priv)
        assert result.already_present is False
        assert db.get_ssh_key_by_fingerprint(result.fingerprint) is not None

    def test_second_import_marks_already_present(
        self, db: CredentialDB, disk_keypair: tuple[Path, Path]
    ) -> None:
        """Re-importing the same keypair reports ``already_present=True``."""
        priv, pub = disk_keypair
        first = import_ssh_keypair(db, "proj", priv, pub_path=pub)
        second = import_ssh_keypair(db, "proj", priv, pub_path=pub)
        assert first.key_id == second.key_id
        assert second.already_present is True

    def test_mismatched_pub_rejected(self, tmp_path: Path, db: CredentialDB) -> None:
        """Priv and pub from unrelated keypairs raises KeypairMismatchError."""
        kp1 = generate_keypair("ed25519", comment="a")
        kp2 = generate_keypair("ed25519", comment="b")
        priv = tmp_path / "priv"
        pub = tmp_path / "pub"
        priv.write_bytes(kp1.private_pem)
        pub.write_text(kp2.public_line + "\n")
        with pytest.raises(KeypairMismatchError):
            import_ssh_keypair(db, "proj", priv, pub_path=pub)


class TestPasswordProtected:
    """Verify password-protected imports raise a clear error."""

    def test_rejected_with_actionable_message(self, tmp_path: Path, db: CredentialDB) -> None:
        """Passphrase-protected private keys cannot be imported (yet).

        cryptography.load_ssh_private_key raises ``TypeError`` when fed a
        passphrase-protected key without a password; :func:`import_ssh_keypair`
        catches that and re-raises :class:`PasswordProtectedKeyError` with an
        actionable hint.  We mock the decoder directly to stay independent
        of optional runtime dependencies (like ``bcrypt``).
        """
        from unittest.mock import patch

        priv = tmp_path / "protected"
        priv.write_bytes(b"dummy-encrypted-bytes")
        with (
            patch(
                "terok_sandbox.credentials.ssh_keypair.load_ssh_private_key",
                side_effect=TypeError("Password was not given but private key is encrypted"),
            ),
            pytest.raises(PasswordProtectedKeyError) as excinfo,
        ):
            import_ssh_keypair(db, "proj", priv)
        assert "ssh-keygen -p" in str(excinfo.value)


class TestExport:
    """Verify :func:`export_ssh_keypair` O_EXCL + permission semantics."""

    def test_writes_both_files_with_permissions(
        self, db: CredentialDB, disk_keypair: tuple[Path, Path], tmp_path: Path
    ) -> None:
        """Export creates 0600 private + 0644 public with fingerprint-suffixed names."""
        priv, pub = disk_keypair
        import_ssh_keypair(db, "proj", priv, pub_path=pub)
        out_dir = tmp_path / "out"
        result = export_ssh_keypair(db, "proj", out_dir)
        assert result.private_path.exists()
        assert result.public_path.exists()
        assert result.public_path.suffix == ".pub"
        assert f"_{result.fingerprint[:8]}" in result.private_path.name

        priv_mode = stat.S_IMODE(os.lstat(result.private_path).st_mode)
        pub_mode = stat.S_IMODE(os.lstat(result.public_path).st_mode)
        assert priv_mode == 0o600
        assert pub_mode == 0o644

    def test_refuses_to_clobber_existing_file(
        self, db: CredentialDB, disk_keypair: tuple[Path, Path], tmp_path: Path
    ) -> None:
        """O_EXCL on the output files prevents silent overwrite; other files OK."""
        priv, pub = disk_keypair
        import_ssh_keypair(db, "proj", priv, pub_path=pub)
        out_dir = tmp_path / "out"
        export_ssh_keypair(db, "proj", out_dir, out_name="custom")
        # Unrelated files in the dir are fine.
        (out_dir / "unrelated.txt").write_text("hi")
        with pytest.raises(FileExistsError):
            export_ssh_keypair(db, "proj", out_dir, out_name="custom")

    def test_round_trip_fingerprints_match(
        self, db: CredentialDB, disk_keypair: tuple[Path, Path], tmp_path: Path
    ) -> None:
        """Export + re-parse yields the same fingerprint as the stored key."""
        priv, pub = disk_keypair
        imp = import_ssh_keypair(db, "proj", priv, pub_path=pub)
        result = export_ssh_keypair(db, "proj", tmp_path / "out")
        parsed = parse_openssh_keypair(
            result.private_path.read_bytes(),
            result.public_path.read_bytes(),
        )
        assert parsed.fingerprint == imp.fingerprint

    def test_scope_with_no_keys_raises(self, db: CredentialDB, tmp_path: Path) -> None:
        """Exporting a known scope that owns no keys fails explicitly."""
        with pytest.raises(ValueError):
            export_ssh_keypair(db, "proj", tmp_path / "out")

    def test_empty_scope_raises(self, db: CredentialDB, tmp_path: Path) -> None:
        """An empty-string scope is never a legitimate caller."""
        with pytest.raises(ValueError):
            export_ssh_keypair(db, "", tmp_path / "out")

    @pytest.mark.parametrize("bad_name", ["../escape", "/etc/passwd", "sub/file", ".", ".."])
    def test_rejects_path_like_out_name(
        self,
        db: CredentialDB,
        disk_keypair: tuple[Path, Path],
        tmp_path: Path,
        bad_name: str,
    ) -> None:
        """``out_name`` must be a bare stem — path-ish values are rejected up front."""
        priv, pub = disk_keypair
        import_ssh_keypair(db, "proj", priv, pub_path=pub)
        with pytest.raises(ValueError):
            export_ssh_keypair(db, "proj", tmp_path / "out", out_name=bad_name)
