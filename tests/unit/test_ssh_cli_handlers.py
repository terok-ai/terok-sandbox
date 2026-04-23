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
    _handle_ssh_add,
    _handle_ssh_export,
    _handle_ssh_import,
    _handle_ssh_link,
    _handle_ssh_list,
    _handle_ssh_pub,
    _handle_ssh_remove,
)
from terok_sandbox.credentials.db import CredentialDB
from terok_sandbox.credentials.ssh_keypair import generate_keypair, openssh_pem_of


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
    priv.write_bytes(openssh_pem_of(kp.private_der))
    pub.write_text(kp.public_line + "\n")
    return priv, pub


def _seed_in_db(db: CredentialDB, scope: str) -> int:
    """Generate + store + assign a key, returning its id."""
    kp = generate_keypair("ed25519", comment=f"tk-main:{scope}")
    key_id = db.store_ssh_key(
        key_type=kp.key_type,
        private_der=kp.private_der,
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
        priv.write_bytes(openssh_pem_of(kp_a.private_der))
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
        priv.write_bytes(openssh_pem_of(kp.private_der))
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


class TestAdd:
    """``ssh-add`` mints a new keypair and prints the summary."""

    def test_generates_and_reports(
        self,
        db_path: Path,
        mock_cfg: MagicMock,
        patched_open_db,
        capsys: pytest.CaptureFixture[str],
    ) -> None:
        """The happy path stores one row and prints id/type/fingerprint/pub."""
        _handle_ssh_add(scope="proj", cfg=mock_cfg)
        out = capsys.readouterr().out
        assert "SSH key ready for scope 'proj'" in out
        assert "type:        ed25519" in out
        assert "ssh-ed25519 " in out
        db = CredentialDB(db_path)
        try:
            assert len(db.list_ssh_keys_for_scope("proj")) == 1
        finally:
            db.close()

    def test_rejects_unsupported_key_type(
        self, db_path: Path, mock_cfg: MagicMock, patched_open_db
    ) -> None:
        """Anything other than ed25519/rsa fails before touching the DB."""
        with pytest.raises(SystemExit, match="Unsupported --key-type"):
            _handle_ssh_add(scope="proj", key_type="dsa", cfg=mock_cfg)


class TestExport:
    """``ssh-export`` writes OpenSSH files and maps library errors cleanly."""

    def test_writes_both_files(
        self,
        tmp_path: Path,
        db_path: Path,
        mock_cfg: MagicMock,
        patched_open_db,
        capsys: pytest.CaptureFixture[str],
    ) -> None:
        """Success prints paths for both files and both land on disk."""
        _seed(db_path, "proj")
        out_dir = tmp_path / "out"
        _handle_ssh_export(scope="proj", out_dir=str(out_dir), cfg=mock_cfg)
        out = capsys.readouterr().out
        assert "Exported key id=" in out
        assert "private key:" in out
        assert "public key:" in out
        privs = list(out_dir.glob("id_*"))
        assert any(p.suffix == ".pub" for p in privs)
        assert any(p.suffix == "" for p in privs)

    def test_empty_scope_surfaces_as_system_exit(
        self, tmp_path: Path, db_path: Path, mock_cfg: MagicMock, patched_open_db
    ) -> None:
        """Exporting a scope with no keys bubbles the library ValueError as SystemExit."""
        CredentialDB(db_path).close()  # ensure DB exists
        with pytest.raises(SystemExit, match="has no SSH keys"):
            _handle_ssh_export(scope="proj", out_dir=str(tmp_path / "out"), cfg=mock_cfg)

    def test_clobber_surfaces_as_system_exit(
        self,
        tmp_path: Path,
        db_path: Path,
        mock_cfg: MagicMock,
        patched_open_db,
    ) -> None:
        """Refusing to overwrite a pre-existing file reports the offender path."""
        _seed(db_path, "proj")
        out_dir = tmp_path / "out"
        _handle_ssh_export(scope="proj", out_dir=str(out_dir), out_name="custom", cfg=mock_cfg)
        with pytest.raises(SystemExit, match="Refusing to overwrite"):
            _handle_ssh_export(scope="proj", out_dir=str(out_dir), out_name="custom", cfg=mock_cfg)


class TestList:
    """``ssh-list`` renders the full key table or a single-scope subset."""

    def test_lists_all_when_no_filter(
        self,
        db_path: Path,
        mock_cfg: MagicMock,
        patched_open_db,
        capsys: pytest.CaptureFixture[str],
    ) -> None:
        """Without ``--scope``, every registered key shows up in the table."""
        _seed(db_path, "proj-a")
        _seed(db_path, "proj-b")
        _handle_ssh_list(cfg=mock_cfg)
        out = capsys.readouterr().out
        assert "proj-a" in out
        assert "proj-b" in out

    def test_scope_filter_narrows_rows(
        self,
        db_path: Path,
        mock_cfg: MagicMock,
        patched_open_db,
        capsys: pytest.CaptureFixture[str],
    ) -> None:
        """A matching ``--scope`` narrows the table to that scope's keys only."""
        _seed(db_path, "proj-a")
        _seed(db_path, "proj-b")
        _handle_ssh_list(scope="proj-a", cfg=mock_cfg)
        out = capsys.readouterr().out
        assert "proj-a" in out
        assert "proj-b" not in out

    def test_unknown_scope_filter_errors(
        self, db_path: Path, mock_cfg: MagicMock, patched_open_db
    ) -> None:
        """``--scope`` that matches nothing exits non-zero."""
        _seed(db_path, "proj-a")
        with pytest.raises(SystemExit, match="No keys registered for scope"):
            _handle_ssh_list(scope="nowhere", cfg=mock_cfg)


class TestRemove:
    """``ssh-remove`` covers filter + confirmation + interactive selection."""

    def test_empty_vault_errors(self, db_path: Path, mock_cfg: MagicMock, patched_open_db) -> None:
        """With no registered keys, remove refuses up front."""
        CredentialDB(db_path).close()
        with pytest.raises(SystemExit, match="No SSH keys registered"):
            _handle_ssh_remove(cfg=mock_cfg)

    def test_filter_with_yes_removes_silently(
        self,
        db_path: Path,
        mock_cfg: MagicMock,
        patched_open_db,
        capsys: pytest.CaptureFixture[str],
    ) -> None:
        """``--scope X --yes`` drops X's keys without prompting."""
        _seed(db_path, "proj-a")
        _seed(db_path, "proj-b")
        _handle_ssh_remove(scope="proj-a", yes=True, cfg=mock_cfg)
        out = capsys.readouterr().out
        assert "Unassigned 1 key from their scope(s)" in out
        db = CredentialDB(db_path)
        try:
            assert db.list_ssh_keys_for_scope("proj-a") == []
            assert len(db.list_ssh_keys_for_scope("proj-b")) == 1
        finally:
            db.close()

    def test_filter_no_match_errors(
        self, db_path: Path, mock_cfg: MagicMock, patched_open_db
    ) -> None:
        """A filter that matches nothing exits with a clear message."""
        _seed(db_path, "proj-a")
        with pytest.raises(SystemExit, match="No keys match"):
            _handle_ssh_remove(scope="nowhere", yes=True, cfg=mock_cfg)

    def test_yes_without_filters_errors(
        self, db_path: Path, mock_cfg: MagicMock, patched_open_db
    ) -> None:
        """``--yes`` alone refuses to wipe everything silently."""
        _seed(db_path, "proj-a")
        with pytest.raises(SystemExit, match="without at least one filter"):
            _handle_ssh_remove(yes=True, cfg=mock_cfg)

    def test_filter_prompt_decline(
        self,
        db_path: Path,
        mock_cfg: MagicMock,
        patched_open_db,
    ) -> None:
        """A non-yes answer to the confirmation prompt aborts."""
        _seed(db_path, "proj-a")
        with patch("builtins.input", return_value="n"):
            with pytest.raises(SystemExit, match="Aborted"):
                _handle_ssh_remove(scope="proj-a", cfg=mock_cfg)

    def test_filter_prompt_accept(
        self,
        db_path: Path,
        mock_cfg: MagicMock,
        patched_open_db,
        capsys: pytest.CaptureFixture[str],
    ) -> None:
        """A ``y`` answer proceeds with the removal."""
        _seed(db_path, "proj-a")
        with patch("builtins.input", return_value="y"):
            _handle_ssh_remove(scope="proj-a", cfg=mock_cfg)
        assert "Unassigned 1 key" in capsys.readouterr().out

    def test_interactive_all(
        self,
        db_path: Path,
        mock_cfg: MagicMock,
        patched_open_db,
        capsys: pytest.CaptureFixture[str],
    ) -> None:
        """Without filters, typing ``all`` at the prompt removes every row."""
        _seed(db_path, "proj-a")
        _seed(db_path, "proj-b")
        with patch("builtins.input", return_value="all"):
            _handle_ssh_remove(cfg=mock_cfg)
        assert "Unassigned 2 keys" in capsys.readouterr().out

    def test_interactive_numeric_selection(
        self,
        db_path: Path,
        mock_cfg: MagicMock,
        patched_open_db,
        capsys: pytest.CaptureFixture[str],
    ) -> None:
        """Comma-separated indices remove only the chosen rows."""
        _seed(db_path, "proj-a")
        _seed(db_path, "proj-b")
        with patch("builtins.input", return_value="1"):
            _handle_ssh_remove(cfg=mock_cfg)
        assert "Unassigned 1 key" in capsys.readouterr().out

    def test_interactive_empty_selection_aborts(
        self, db_path: Path, mock_cfg: MagicMock, patched_open_db
    ) -> None:
        """An empty line at the prompt counts as ``Aborted``."""
        _seed(db_path, "proj-a")
        with patch("builtins.input", return_value=""):
            with pytest.raises(SystemExit, match="Aborted"):
                _handle_ssh_remove(cfg=mock_cfg)

    def test_interactive_invalid_index_errors(
        self, db_path: Path, mock_cfg: MagicMock, patched_open_db
    ) -> None:
        """Non-numeric or out-of-range input is rejected with a clear message."""
        _seed(db_path, "proj-a")
        with patch("builtins.input", return_value="42"):
            with pytest.raises(SystemExit, match="Invalid selection"):
                _handle_ssh_remove(cfg=mock_cfg)
