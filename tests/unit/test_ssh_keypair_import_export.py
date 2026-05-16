# SPDX-FileCopyrightText: 2026 Jiri Vyskocil
# SPDX-License-Identifier: Apache-2.0

"""Tests for [`terok_sandbox.vault.ssh.keypair`][terok_sandbox.vault.ssh.keypair] — generate, import, export."""

from __future__ import annotations

import os
import stat
from pathlib import Path

import pytest

from terok_sandbox.vault.ssh.keypair import (
    InfraKeypair,
    KeypairMismatchError,
    PasswordProtectedKeyError,
    ensure_infra_keypair,
    export_ssh_keypair,
    generate_keypair,
    import_ssh_keypair,
    openssh_pem_of,
    parse_openssh_keypair,
)
from terok_sandbox.vault.store.db import CredentialDB, UnsafeCommentError


@pytest.fixture()
def db(tmp_path: Path) -> CredentialDB:
    """Return a fresh DB rooted under a per-test tmp dir."""
    return CredentialDB(tmp_path / "vault" / "credentials.db", passphrase="test")


@pytest.fixture()
def disk_keypair(tmp_path: Path) -> tuple[Path, Path]:
    """Generate a keypair in memory and write it to files; return (priv, pub)."""
    kp = generate_keypair("ed25519", comment="disk-comment")
    priv = tmp_path / "id_ed25519_disk"
    pub = tmp_path / "id_ed25519_disk.pub"
    priv.write_bytes(openssh_pem_of(kp.private_der))
    pub.write_text(kp.public_line + "\n")
    return priv, pub


class TestGenerate:
    """Verify in-memory keypair generation."""

    def test_ed25519_shape(self) -> None:
        """Generated ed25519 keypair carries DER + wire blob + fingerprint."""
        kp = generate_keypair("ed25519", comment="hello")
        assert kp.key_type == "ed25519"
        # PKCS#8 DER: opaque binary, no banner strings.
        assert b"PRIVATE KEY" not in kp.private_der
        assert len(kp.public_blob) > 0
        assert kp.public_line.startswith("ssh-ed25519 ")
        assert kp.public_line.endswith(" hello")
        # Standard OpenSSH fingerprint: ``SHA256:<43-char unpadded base64>``.
        assert kp.fingerprint.startswith("SHA256:")
        assert len(kp.fingerprint) == len("SHA256:") + 43
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
    """Verify [`import_ssh_keypair`][terok_sandbox.import_ssh_keypair] end-to-end behaviour."""

    def test_round_trip_with_public_file(
        self, db: CredentialDB, disk_keypair: tuple[Path, Path]
    ) -> None:
        """Importing priv+pub stores the key and assigns it to the scope."""
        priv, pub = disk_keypair
        result = import_ssh_keypair(db, "proj", priv, pub_path=pub)
        assert result.already_present is False
        assert result.fingerprint.startswith("SHA256:")
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

    def test_newly_imported_key_reports_fresh(self, tmp_path: Path, db: CredentialDB) -> None:
        """First-time import: not already present, scope wasn't assigned."""
        kp = generate_keypair("ed25519", comment="")
        priv = tmp_path / "id"
        pub = tmp_path / "id.pub"
        priv.write_bytes(openssh_pem_of(kp.private_der))
        pub.write_text(kp.public_line + "\n")
        result = import_ssh_keypair(db, "proj-a", priv, pub_path=pub)
        assert result.already_present is False
        assert result.scope_was_assigned is False

    def test_second_scope_reports_key_present_scope_new(
        self, tmp_path: Path, db: CredentialDB
    ) -> None:
        """Re-importing the same key under a new scope: key present, scope fresh."""
        kp = generate_keypair("ed25519", comment="")
        priv = tmp_path / "id"
        pub = tmp_path / "id.pub"
        priv.write_bytes(openssh_pem_of(kp.private_der))
        pub.write_text(kp.public_line + "\n")
        import_ssh_keypair(db, "proj-a", priv, pub_path=pub)
        result = import_ssh_keypair(db, "proj-b", priv, pub_path=pub)
        assert result.already_present is True
        assert result.scope_was_assigned is False

    def test_redundant_import_reports_both_flags(self, tmp_path: Path, db: CredentialDB) -> None:
        """Importing the same key twice for the same scope: fully redundant."""
        kp = generate_keypair("ed25519", comment="")
        priv = tmp_path / "id"
        pub = tmp_path / "id.pub"
        priv.write_bytes(openssh_pem_of(kp.private_der))
        pub.write_text(kp.public_line + "\n")
        import_ssh_keypair(db, "proj-a", priv, pub_path=pub)
        result = import_ssh_keypair(db, "proj-a", priv, pub_path=pub)
        assert result.already_present is True
        assert result.scope_was_assigned is True

    def test_rsa_import_round_trip(self, tmp_path: Path, db: CredentialDB) -> None:
        """RSA keys flow through import just like ed25519 — classifier picks the right algo."""
        kp = generate_keypair("rsa", comment="rsa-import")
        priv = tmp_path / "id_rsa"
        pub = tmp_path / "id_rsa.pub"
        priv.write_bytes(openssh_pem_of(kp.private_der))
        pub.write_text(kp.public_line + "\n")
        result = import_ssh_keypair(db, "proj", priv, pub_path=pub)
        [row] = db.list_ssh_keys_for_scope("proj")
        assert row.key_type == "rsa"
        assert row.fingerprint == result.fingerprint

    def test_mismatched_pub_rejected(self, tmp_path: Path, db: CredentialDB) -> None:
        """Priv and pub from unrelated keypairs raises KeypairMismatchError."""
        kp1 = generate_keypair("ed25519", comment="a")
        kp2 = generate_keypair("ed25519", comment="b")
        priv = tmp_path / "priv"
        pub = tmp_path / "pub"
        priv.write_bytes(openssh_pem_of(kp1.private_der))
        pub.write_text(kp2.public_line + "\n")
        with pytest.raises(KeypairMismatchError):
            import_ssh_keypair(db, "proj", priv, pub_path=pub)


class TestPasswordProtected:
    """Verify password-protected imports raise a clear error."""

    def test_rejected_with_actionable_message(self, tmp_path: Path, db: CredentialDB) -> None:
        """Passphrase-protected private keys cannot be imported (yet).

        cryptography.load_ssh_private_key raises ``TypeError`` when fed a
        passphrase-protected key without a password; [`import_ssh_keypair`][terok_sandbox.import_ssh_keypair]
        catches that and re-raises [`PasswordProtectedKeyError`][terok_sandbox.PasswordProtectedKeyError] with an
        actionable hint.  We mock the decoder directly to stay independent
        of optional runtime dependencies (like ``bcrypt``).
        """
        from unittest.mock import patch

        priv = tmp_path / "protected"
        priv.write_bytes(b"dummy-encrypted-bytes")
        with (
            patch(
                "terok_sandbox.vault.ssh.keypair.load_ssh_private_key",
                side_effect=TypeError("Password was not given but private key is encrypted"),
            ),
            pytest.raises(PasswordProtectedKeyError) as excinfo,
        ):
            import_ssh_keypair(db, "proj", priv)
        # Library exception is diagnostic; the CLI handler appends the
        # ``ssh-keygen -p`` remediation hint itself.
        assert "passphrase-protected" in str(excinfo.value)


class TestExport:
    """Verify [`export_ssh_keypair`][terok_sandbox.export_ssh_keypair] O_EXCL + permission semantics."""

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
        # Filename stem embeds an 8-hex-char short id of the public blob —
        # stable across display-format changes to the fingerprint string.
        import hashlib

        [record] = db.load_ssh_keys_for_scope("proj")
        short_id = hashlib.sha256(record.public_blob).hexdigest()[:8]
        assert f"_{short_id}" in result.private_path.name

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

    def test_public_write_failure_rolls_back_private(
        self,
        db: CredentialDB,
        disk_keypair: tuple[Path, Path],
        tmp_path: Path,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        """If the .pub write fails, the private file must not linger on disk."""
        from terok_sandbox.vault.ssh import keypair as mod

        priv, pub = disk_keypair
        import_ssh_keypair(db, "proj", priv, pub_path=pub)

        real = mod._write_exclusive
        calls: list[Path] = []

        def _fault_on_pub(path: Path, data: bytes, mode: int) -> None:
            calls.append(path)
            if path.suffix == ".pub":
                raise OSError("synthetic pub-write failure")
            real(path, data, mode)

        monkeypatch.setattr(mod, "_write_exclusive", _fault_on_pub)

        out_dir = tmp_path / "out"
        with pytest.raises(OSError, match="synthetic pub-write failure"):
            export_ssh_keypair(db, "proj", out_dir)

        # Private file exists briefly, then gets unlinked during rollback.
        survivors = list(out_dir.glob("*"))
        assert survivors == [], f"no files should remain, got {survivors}"

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

    def test_picks_key_by_id(
        self, db: CredentialDB, disk_keypair: tuple[Path, Path], tmp_path: Path
    ) -> None:
        """Explicit ``key_id`` exports that specific key, not the most-recent."""
        priv, pub = disk_keypair
        first = import_ssh_keypair(db, "proj", priv, pub_path=pub)
        kp2 = generate_keypair("ed25519", comment="second")
        second_priv = tmp_path / "id2"
        second_pub = tmp_path / "id2.pub"
        second_priv.write_bytes(openssh_pem_of(kp2.private_der))
        second_pub.write_text(kp2.public_line + "\n")
        import_ssh_keypair(db, "proj", second_priv, pub_path=second_pub)

        result = export_ssh_keypair(db, "proj", tmp_path / "out", key_id=first.key_id)
        assert result.key_id == first.key_id
        assert result.fingerprint == first.fingerprint

    def test_unknown_key_id_raises(
        self, db: CredentialDB, disk_keypair: tuple[Path, Path], tmp_path: Path
    ) -> None:
        """An unassigned key_id fails with a clear message."""
        priv, pub = disk_keypair
        import_ssh_keypair(db, "proj", priv, pub_path=pub)
        with pytest.raises(ValueError, match="key_id .* not assigned"):
            export_ssh_keypair(db, "proj", tmp_path / "out", key_id=99999)


class TestWriteExclusive:
    """Verify ``_write_exclusive`` handles short writes and rollback."""

    def test_short_writes_are_looped(self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
        """``os.write`` short-writes of 1 byte at a time still produce the full file."""
        import os

        from terok_sandbox.vault.ssh import keypair as mod

        real_write = os.write

        def _one_byte_at_a_time(fd: int, data):
            return real_write(fd, bytes(memoryview(data))[:1])

        monkeypatch.setattr(mod.os, "write", _one_byte_at_a_time)

        target = tmp_path / "small.bin"
        mod._write_exclusive(target, b"hello-world", 0o600)
        assert target.read_bytes() == b"hello-world"

    def test_chmod_failure_unlinks_file(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """A failure after bytes landed still rolls back — no truncated-mode leftover."""
        from terok_sandbox.vault.ssh import keypair as mod

        def _chmod_boom(_path, _mode):
            raise OSError("synthetic chmod failure")

        monkeypatch.setattr(mod.os, "chmod", _chmod_boom)

        target = tmp_path / "partial.bin"
        with pytest.raises(OSError, match="chmod failure"):
            mod._write_exclusive(target, b"data", 0o600)
        assert not target.exists()


class TestCommentGuard:
    """Unsafe comments (control chars, newlines, oversize) are rejected at entry."""

    @pytest.mark.parametrize(
        "bad",
        [
            "line1\nline2",  # LF — breaks authorized_keys / public-line contract
            "has\rCR",  # CR — same
            "esc\x1b[31mred",  # ANSI escape — terminal spoofing
            "null\x00byte",  # C0
            "del\x7fchar",  # DEL
        ],
    )
    def test_generate_rejects_control_chars(self, bad: str) -> None:
        """Every C0 control char or DEL in a comment fails fast at generation time."""
        with pytest.raises(UnsafeCommentError, match="disallowed control character"):
            generate_keypair("ed25519", comment=bad)

    def test_generate_rejects_oversize(self) -> None:
        """A 201-char comment is over the 200-char bound."""
        with pytest.raises(UnsafeCommentError, match="character limit"):
            generate_keypair("ed25519", comment="x" * 201)

    def test_generate_accepts_plain_ascii(self) -> None:
        """Printable ASCII passes through untouched."""
        kp = generate_keypair("ed25519", comment="tk-main:myproj")
        assert kp.comment == "tk-main:myproj"

    def test_import_rejects_poisoned_pub_file_comment(
        self, tmp_path: Path, db: CredentialDB
    ) -> None:
        """A ``.pub`` file whose comment smuggles an ANSI escape is rejected."""
        kp = generate_keypair("ed25519", comment="clean")
        priv = tmp_path / "id"
        pub = tmp_path / "id.pub"
        priv.write_bytes(openssh_pem_of(kp.private_der))
        # Replace the clean comment with a malicious one in the .pub file.
        pub.write_text(kp.public_line.rsplit(" ", 1)[0] + " bad\x1b[31m\n")
        with pytest.raises(UnsafeCommentError):
            import_ssh_keypair(db, "proj", priv, pub_path=pub)


class TestPublicLine:
    """Verify [`public_line_of`][terok_sandbox.public_line_of] covers both algos and rejects unknowns."""

    def test_ed25519_line_format(self) -> None:
        """Line starts with ``ssh-ed25519`` and ends with the comment."""
        from terok_sandbox.vault.ssh.keypair import public_line_of

        kp = generate_keypair("ed25519", comment="hi")
        rec = _record_from(kp, id=1)
        line = public_line_of(rec)
        assert line.startswith("ssh-ed25519 ")
        assert line.endswith(" hi")

    def test_rsa_line_format(self) -> None:
        """RSA records render as ``ssh-rsa <b64> <comment>``."""
        from terok_sandbox.vault.ssh.keypair import public_line_of

        kp = generate_keypair("rsa", comment="rsa-c")
        rec = _record_from(kp, id=1)
        line = public_line_of(rec)
        assert line.startswith("ssh-rsa ")
        assert line.endswith(" rsa-c")

    def test_unknown_algo_rejected(self) -> None:
        """Corrupt DB with an unexpected key_type surfaces as ValueError."""
        from terok_sandbox.vault.ssh.keypair import public_line_of
        from terok_sandbox.vault.store.db import SSHKeyRecord

        rec = SSHKeyRecord(
            id=1,
            key_type="dsa",
            private_der=b"",
            public_blob=b"x",
            comment="",
            fingerprint="SHA256:deadbeef",
        )
        with pytest.raises(ValueError, match="unsupported key type"):
            public_line_of(rec)


def _record_from(kp, *, id: int):
    """Build an [`SSHKeyRecord`][terok_sandbox.SSHKeyRecord] from a [`GeneratedKeypair`][terok_sandbox.GeneratedKeypair]."""
    from terok_sandbox.vault.store.db import SSHKeyRecord

    return SSHKeyRecord(
        id=id,
        key_type=kp.key_type,
        private_der=kp.private_der,
        public_blob=kp.public_blob,
        comment=kp.comment,
        fingerprint=kp.fingerprint,
    )


class TestParseErrors:
    """Verify [`parse_openssh_keypair`][terok_sandbox.parse_openssh_keypair] failure modes."""

    def test_malformed_pub_line_raises(self, tmp_path: Path) -> None:
        """A ``.pub`` file with fewer than two whitespace-separated fields is rejected."""
        kp = generate_keypair("ed25519", comment="")
        priv = tmp_path / "k"
        priv.write_bytes(openssh_pem_of(kp.private_der))
        with pytest.raises(ValueError, match="malformed public key file"):
            parse_openssh_keypair(priv.read_bytes(), b"only-one-field")

    def test_malformed_private_key_raises(self, tmp_path: Path) -> None:
        """A garbage private key bubbles up as a plain ``ValueError``, not the passphrase one."""
        with pytest.raises(ValueError):
            parse_openssh_keypair(b"not-a-real-pem")


class TestEnsureInfraKeypair:
    """[`ensure_infra_keypair`][terok_sandbox.vault.ssh.keypair.ensure_infra_keypair]
    is the load-or-mint single entry point for ``%scope`` slots."""

    def test_first_call_generates_and_persists(self, db: CredentialDB) -> None:
        """Empty DB → mint + store + assign, ``created=True``."""
        result = ensure_infra_keypair("%host", db=db)
        assert isinstance(result, InfraKeypair)
        assert result.scope == "%host"
        assert result.created is True
        assert result.key_type == "ed25519"
        assert result.private_pem.startswith(b"-----BEGIN OPENSSH PRIVATE KEY-----")
        assert result.public_line.startswith("ssh-ed25519 ")
        # Persisted under the infra scope.
        assert len(db.list_ssh_keys_for_scope("%host")) == 1

    def test_second_call_loads_without_rotating(self, db: CredentialDB) -> None:
        """Subsequent calls return the same key with ``created=False``.

        The OpenSSH PEM format embeds a random ``checkint`` that's fresh
        on every serialisation, so the *bytes* differ across calls even
        for the same underlying private key.  Fingerprint + public line
        are the stable identity comparators.
        """
        first = ensure_infra_keypair("%host", db=db)
        second = ensure_infra_keypair("%host", db=db)
        assert second.fingerprint == first.fingerprint
        assert second.public_line == first.public_line
        assert second.created is False
        # Still just one assignment — no duplicate keys.
        assert len(db.list_ssh_keys_for_scope("%host")) == 1
        # Both PEMs round-trip to the *same* private key material.
        from cryptography.hazmat.primitives import serialization

        k1 = serialization.load_ssh_private_key(first.private_pem, password=None)
        k2 = serialization.load_ssh_private_key(second.private_pem, password=None)
        assert k1.private_bytes(
            encoding=serialization.Encoding.Raw,
            format=serialization.PrivateFormat.Raw,
            encryption_algorithm=serialization.NoEncryption(),
        ) == k2.private_bytes(
            encoding=serialization.Encoding.Raw,
            format=serialization.PrivateFormat.Raw,
            encryption_algorithm=serialization.NoEncryption(),
        )

    def test_rejects_non_infra_scope(self, db: CredentialDB) -> None:
        """User scopes (no ``%`` prefix) go through ``import_ssh_keypair``."""
        with pytest.raises(ValueError, match="must start with '%'"):
            ensure_infra_keypair("myproject", db=db)

    def test_uses_allow_infra_path(self, db: CredentialDB) -> None:
        """The store/assign uses ``allow_infra=True`` so the
        ``%``-prefix gate doesn't refuse the write.  Verified
        indirectly: a successful generation must have called
        ``assign_ssh_key`` with the gate satisfied; if it had hit the
        user-scope guard, the call would have raised
        ``InvalidScopeName``."""
        ensure_infra_keypair("%host", db=db)  # no raise

    def test_default_comment_names_the_scope(self, db: CredentialDB) -> None:
        """Default comment reads ``terok-infra:%scope`` so operators
        can tell the key apart from user-imported ones."""
        result = ensure_infra_keypair("%host", db=db)
        assert result.public_line.rstrip().endswith("terok-infra:%host")

    def test_explicit_comment_overrides_default(self, db: CredentialDB) -> None:
        """A caller can supply their own comment on first generation."""
        result = ensure_infra_keypair("%host", db=db, comment="krun-host (terok)")
        assert result.public_line.rstrip().endswith("krun-host (terok)")

    def test_pem_round_trips_to_a_loadable_private_key(self, db: CredentialDB) -> None:
        """The returned PEM is what ``ssh -i`` reads — load it back."""
        from cryptography.hazmat.primitives import serialization

        result = ensure_infra_keypair("%host", db=db)
        # Parses without exception → valid OpenSSH PEM.
        serialization.load_ssh_private_key(result.private_pem, password=None)

    def test_existing_assignment_made_outside_helper_is_reused(self, db: CredentialDB) -> None:
        """If sandbox internals seeded ``%host`` directly, the helper
        re-serialises that key rather than minting a new one."""
        seeded = generate_keypair("ed25519", comment="legacy")
        key_id = db.store_ssh_key(
            key_type=seeded.key_type,
            private_der=seeded.private_der,
            public_blob=seeded.public_blob,
            comment=seeded.comment,
            fingerprint=seeded.fingerprint,
        )
        db.assign_ssh_key("%host", key_id, allow_infra=True)

        result = ensure_infra_keypair("%host", db=db)
        assert result.fingerprint == seeded.fingerprint
        assert result.created is False

    def test_multi_assigned_scope_returns_newest_not_oldest(self, db: CredentialDB) -> None:
        """If the scope has multiple assigned keys, pick the newest.

        ``load_ssh_keys_for_scope`` orders by ``assigned_at`` ascending,
        so the naive ``existing[0]`` would resurrect the oldest key —
        bad if an additive rotation ever leaves stale material under
        the scope.  We assert the newest assignment wins.
        """
        old_kp = generate_keypair("ed25519", comment="old-key")
        new_kp = generate_keypair("ed25519", comment="new-key")

        old_id = db.store_ssh_key(
            key_type=old_kp.key_type,
            private_der=old_kp.private_der,
            public_blob=old_kp.public_blob,
            comment=old_kp.comment,
            fingerprint=old_kp.fingerprint,
        )
        db.assign_ssh_key("%host", old_id, allow_infra=True)

        new_id = db.store_ssh_key(
            key_type=new_kp.key_type,
            private_der=new_kp.private_der,
            public_blob=new_kp.public_blob,
            comment=new_kp.comment,
            fingerprint=new_kp.fingerprint,
        )
        db.assign_ssh_key("%host", new_id, allow_infra=True)

        result = ensure_infra_keypair("%host", db=db)
        assert result.fingerprint == new_kp.fingerprint
        assert result.fingerprint != old_kp.fingerprint
