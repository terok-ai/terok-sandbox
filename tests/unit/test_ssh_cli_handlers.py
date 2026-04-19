# SPDX-FileCopyrightText: 2026 Jiri Vyskocil
# SPDX-License-Identifier: Apache-2.0

"""Tests for terok-sandbox's ``ssh`` CLI handler functions.

Exercises handlers directly (not through argparse) so every branch can
be pinned with a compact assertion against captured stdout or DB state.
"""

from __future__ import annotations

from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from terok_sandbox.commands import (
    _handle_ssh_import,
    _handle_ssh_link,
    _handle_ssh_pub,
)
from terok_sandbox.credentials.db import CredentialDB
from terok_sandbox.credentials.ssh_keypair import generate_keypair


@pytest.fixture()
def db_path(tmp_path: Path) -> Path:
    """Return the path of a not-yet-opened vault DB."""
    return tmp_path / "vault" / "credentials.db"


@pytest.fixture()
def mock_cfg(db_path: Path) -> MagicMock:
    """Stub ``SandboxConfig`` carrying only ``db_path``."""
    cfg = MagicMock()
    cfg.db_path = db_path
    return cfg


@pytest.fixture()
def patched_open_db(db_path: Path):
    """Patch ``_open_db`` to yield a fresh connection on every call.

    Each handler closes its DB in a ``finally``, so tests must get a new
    connection per invocation — and a follow-up check from the test body
    needs *another* new connection.
    """

    def _factory(_cfg):
        return CredentialDB(db_path)

    with patch("terok_sandbox.commands._open_db", side_effect=_factory):
        yield


def _seed_disk_pair(tmp_path: Path, scope_label: str) -> tuple[Path, Path]:
    """Materialize a keypair to disk and return ``(priv, pub)``."""
    kp = generate_keypair("ed25519", comment=f"tk-main:{scope_label}")
    priv = tmp_path / f"id_ed25519_{scope_label}"
    pub = priv.with_name(priv.name + ".pub")
    priv.write_bytes(kp.private_pem)
    pub.write_text(kp.public_line + "\n")
    return priv, pub


def _seed_in_db(db: CredentialDB, scope: str) -> int:
    """Generate + store + assign a key, returning its id."""
    kp = generate_keypair("ed25519", comment=f"tk-main:{scope}")
    key_id = db.store_ssh_key(
        key_type=kp.key_type,
        private_pem=kp.private_pem,
        public_blob=kp.public_blob,
        comment=kp.comment,
        fingerprint=kp.fingerprint,
    )
    db.assign_ssh_key(scope, key_id)
    return key_id


def _seed(db_path: Path, scope: str) -> int:
    """Seed one key for *scope* through a fresh DB connection; return its id."""
    db = CredentialDB(db_path)
    try:
        return _seed_in_db(db, scope)
    finally:
        db.close()


class TestPubAll:
    """``ssh-pub --all`` prints every key assigned to the scope."""

    def test_prints_single_key_by_default(
        self,
        db_path: Path,
        mock_cfg: MagicMock,
        patched_open_db,
        capsys: pytest.CaptureFixture[str],
    ) -> None:
        """Without ``--all``, the most recent key alone is printed."""
        _seed(db_path, "proj")
        _seed(db_path, "proj")
        _handle_ssh_pub(scope="proj", cfg=mock_cfg)
        assert capsys.readouterr().out.count("\n") == 1

    def test_all_prints_every_key(
        self,
        db_path: Path,
        mock_cfg: MagicMock,
        patched_open_db,
        capsys: pytest.CaptureFixture[str],
    ) -> None:
        """With ``--all``, one line per assigned key."""
        for _ in range(3):
            _seed(db_path, "proj")
        _handle_ssh_pub(scope="proj", all_keys=True, cfg=mock_cfg)
        lines = [ln for ln in capsys.readouterr().out.splitlines() if ln]
        assert len(lines) == 3
        assert all(ln.startswith("ssh-ed25519 ") for ln in lines)

    def test_all_conflicts_with_key_id(
        self, db_path: Path, mock_cfg: MagicMock, patched_open_db
    ) -> None:
        """``--all`` and ``--key-id`` together is a user error."""
        _seed(db_path, "proj")
        with pytest.raises(SystemExit, match="mutually exclusive"):
            _handle_ssh_pub(scope="proj", key_id=1, all_keys=True, cfg=mock_cfg)


class TestLink:
    """``ssh-link`` adds a scope → key_id row without re-importing material."""

    def test_links_existing_key_to_new_scope(
        self,
        db_path: Path,
        mock_cfg: MagicMock,
        patched_open_db,
        capsys: pytest.CaptureFixture[str],
    ) -> None:
        """The scope gains the assignment; the original scope keeps it too."""
        key_id = _seed(db_path, "proj-a")
        _handle_ssh_link(key_id=key_id, scope="proj-b", cfg=mock_cfg)
        assert f"Linked key id={key_id}" in capsys.readouterr().out

        verify = CredentialDB(db_path)
        try:
            assert [r.id for r in verify.list_ssh_keys_for_scope("proj-a")] == [key_id]
            assert [r.id for r in verify.list_ssh_keys_for_scope("proj-b")] == [key_id]
        finally:
            verify.close()

    def test_unknown_key_id_is_rejected(
        self, db_path: Path, mock_cfg: MagicMock, patched_open_db
    ) -> None:
        """A non-existent key_id fails with a clear message (no FK error leak)."""
        # Create an empty DB file so ``_open_db`` has something to open.
        CredentialDB(db_path).close()
        with pytest.raises(SystemExit, match="No ssh_keys row with id=999"):
            _handle_ssh_link(key_id=999, scope="proj-b", cfg=mock_cfg)

    def test_idempotent(
        self,
        db_path: Path,
        mock_cfg: MagicMock,
        patched_open_db,
        capsys: pytest.CaptureFixture[str],
    ) -> None:
        """Re-linking the same pair is a reported no-op, not a duplicate row."""
        key_id = _seed(db_path, "proj-a")
        db = CredentialDB(db_path)
        try:
            db.assign_ssh_key("proj-b", key_id)
        finally:
            db.close()
        _handle_ssh_link(key_id=key_id, scope="proj-b", cfg=mock_cfg)
        assert "already linked" in capsys.readouterr().out


class TestImportErrorMapping:
    """``_handle_ssh_import`` turns library-level errors into clean SystemExits."""

    def test_mismatched_pub_file_raises_system_exit(
        self, tmp_path: Path, db_path: Path, mock_cfg: MagicMock, patched_open_db
    ) -> None:
        """A pub file from a different keypair surfaces as a user-facing error."""
        kp_a = generate_keypair("ed25519", comment="a")
        kp_b = generate_keypair("ed25519", comment="b")
        priv = tmp_path / "priv"
        pub = tmp_path / "pub"
        priv.write_bytes(kp_a.private_pem)
        pub.write_text(kp_b.public_line + "\n")
        with pytest.raises(SystemExit, match="Import failed:"):
            _handle_ssh_import(
                scope="proj", private_key=str(priv), public_key=str(pub), cfg=mock_cfg
            )

    def test_poisoned_comment_raises_system_exit(
        self, tmp_path: Path, db_path: Path, mock_cfg: MagicMock, patched_open_db
    ) -> None:
        """ANSI escapes in the comment come back as ``Import failed``, not a traceback."""
        kp = generate_keypair("ed25519", comment="clean")
        priv = tmp_path / "id"
        pub = tmp_path / "id.pub"
        priv.write_bytes(kp.private_pem)
        pub.write_text(kp.public_line.rsplit(" ", 1)[0] + " bad\x1b[31m\n")
        with pytest.raises(SystemExit, match="Import failed:"):
            _handle_ssh_import(
                scope="proj", private_key=str(priv), public_key=str(pub), cfg=mock_cfg
            )

    def test_garbage_pem_raises_system_exit(
        self, tmp_path: Path, db_path: Path, mock_cfg: MagicMock, patched_open_db
    ) -> None:
        """A non-PEM private-key file surfaces as a clean error."""
        priv = tmp_path / "id"
        priv.write_bytes(b"not a real PEM")
        with pytest.raises(SystemExit, match="Import failed:"):
            _handle_ssh_import(scope="proj", private_key=str(priv), cfg=mock_cfg)


class TestScopeNameValidation:
    """``_validate_scope_name`` now covers length, not just charset."""

    def test_oversize_scope_rejected(
        self, db_path: Path, mock_cfg: MagicMock, patched_open_db
    ) -> None:
        """A 65-char scope exceeds the unix-socket-budget bound and fails at CLI time."""
        from terok_sandbox.commands import _validate_scope_name

        with pytest.raises(SystemExit, match="exceeds"):
            _validate_scope_name("x" * 65)


class TestImportMessaging:
    """``ssh-import`` emits different messages depending on DB + scope state."""

    def test_fresh_import_reports_imported(
        self,
        tmp_path: Path,
        db_path: Path,
        mock_cfg: MagicMock,
        patched_open_db,
        capsys: pytest.CaptureFixture[str],
    ) -> None:
        """First-time import → ``Imported new key``."""
        priv, pub = _seed_disk_pair(tmp_path, "fresh")
        _handle_ssh_import(scope="proj-a", private_key=str(priv), public_key=str(pub), cfg=mock_cfg)
        assert "Imported new key to scope 'proj-a'" in capsys.readouterr().out

    def test_key_already_in_db_new_scope_reports_linked(
        self,
        tmp_path: Path,
        db_path: Path,
        mock_cfg: MagicMock,
        patched_open_db,
        capsys: pytest.CaptureFixture[str],
    ) -> None:
        """Existing key, new scope → ``Linked existing vault key …`` (not "already")."""
        priv, pub = _seed_disk_pair(tmp_path, "share")
        _handle_ssh_import(scope="proj-a", private_key=str(priv), public_key=str(pub), cfg=mock_cfg)
        capsys.readouterr()
        _handle_ssh_import(scope="proj-b", private_key=str(priv), public_key=str(pub), cfg=mock_cfg)
        out = capsys.readouterr().out
        assert "Linked existing vault key to scope 'proj-b'" in out

    def test_redundant_import_reports_already_linked(
        self,
        tmp_path: Path,
        db_path: Path,
        mock_cfg: MagicMock,
        patched_open_db,
        capsys: pytest.CaptureFixture[str],
    ) -> None:
        """Second import for the same scope → ``Key already linked…`` (true no-op)."""
        priv, pub = _seed_disk_pair(tmp_path, "dup")
        _handle_ssh_import(scope="proj-a", private_key=str(priv), public_key=str(pub), cfg=mock_cfg)
        capsys.readouterr()
        _handle_ssh_import(scope="proj-a", private_key=str(priv), public_key=str(pub), cfg=mock_cfg)
        assert "Key already linked to scope" in capsys.readouterr().out
