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
    _handle_ssh_rename,
)
from terok_sandbox.vault.ssh.keypair import generate_keypair, openssh_pem_of
from terok_sandbox.vault.store.db import CredentialDB


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
        return CredentialDB(db_path, passphrase="test")

    with patch("terok_sandbox.commands.ssh._open_db", side_effect=_factory):
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
    db = CredentialDB(db_path, passphrase="test")
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

        verify = CredentialDB(db_path, passphrase="test")
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
        CredentialDB(db_path, passphrase="test").close()
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
        db = CredentialDB(db_path, passphrase="test")
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
        db = CredentialDB(db_path, passphrase="test")
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
        CredentialDB(db_path, passphrase="test").close()  # ensure DB exists
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
        CredentialDB(db_path, passphrase="test").close()
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
        db = CredentialDB(db_path, passphrase="test")
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


def _fingerprint_prefix(db_path: Path, scope: str, *, length: int = 12) -> str:
    """Read back the seeded key's fingerprint and return a stable prefix."""
    db = CredentialDB(db_path, passphrase="test")
    try:
        fp = db.list_ssh_keys_for_scope(scope)[0].fingerprint
    finally:
        db.close()
    return fp.removeprefix("SHA256:")[:length]


class TestRename:
    """``ssh-rename`` edits a key's comment, identified by fingerprint prefix."""

    def test_renames_by_fingerprint_prefix(
        self,
        db_path: Path,
        mock_cfg: MagicMock,
        patched_open_db,
        capsys: pytest.CaptureFixture[str],
    ) -> None:
        """A prefix matching exactly one key rewrites that row's comment."""
        _seed(db_path, "proj")
        prefix = _fingerprint_prefix(db_path, "proj")
        _handle_ssh_rename(fingerprint=prefix, comment="renamed", cfg=mock_cfg)
        assert "Renamed" in capsys.readouterr().out
        verify = CredentialDB(db_path, passphrase="test")
        try:
            assert verify.list_ssh_keys_for_scope("proj")[0].comment == "renamed"
        finally:
            verify.close()

    def test_no_match_errors(self, db_path: Path, mock_cfg: MagicMock, patched_open_db) -> None:
        """A prefix that matches nothing exits with a clear message."""
        _seed(db_path, "proj")
        with pytest.raises(SystemExit, match="No SSH key matches"):
            _handle_ssh_rename(fingerprint="zzzz-nope-zzzz", comment="x", cfg=mock_cfg)

    def test_ambiguous_prefix_errors_without_writing(
        self,
        db_path: Path,
        mock_cfg: MagicMock,
        patched_open_db,
        capsys: pytest.CaptureFixture[str],
    ) -> None:
        """A prefix that matches >1 distinct fingerprint exits without writing."""
        _seed(db_path, "proj-a")
        _seed(db_path, "proj-b")
        # Single-char prefix that happens to match both is unlikely, so we
        # force ambiguity by feeding the truncated common prefix "SHA256".
        # ``_filter_key_rows`` strips that prefix, so every key matches.
        with pytest.raises(SystemExit, match="Refine the prefix"):
            _handle_ssh_rename(fingerprint="SHA256:", comment="x", cfg=mock_cfg)
        assert "Ambiguous" in capsys.readouterr().out
        verify = CredentialDB(db_path, passphrase="test")
        try:
            for scope in ("proj-a", "proj-b"):
                assert verify.list_ssh_keys_for_scope(scope)[0].comment.startswith("tk-main:")
        finally:
            verify.close()

    def test_unsafe_comment_rejected(
        self, db_path: Path, mock_cfg: MagicMock, patched_open_db
    ) -> None:
        """A comment with control characters surfaces as a user-facing error."""
        _seed(db_path, "proj")
        prefix = _fingerprint_prefix(db_path, "proj")
        with pytest.raises(SystemExit, match="Invalid comment"):
            _handle_ssh_rename(fingerprint=prefix, comment="bad\x01comment", cfg=mock_cfg)

    def test_renames_across_all_linked_scopes(
        self,
        db_path: Path,
        mock_cfg: MagicMock,
        patched_open_db,
    ) -> None:
        """A key linked to multiple scopes has one comment row — rename hits all."""
        key_id = _seed(db_path, "proj-a")
        db = CredentialDB(db_path, passphrase="test")
        try:
            db.assign_ssh_key("proj-b", key_id)
        finally:
            db.close()
        prefix = _fingerprint_prefix(db_path, "proj-a")
        _handle_ssh_rename(fingerprint=prefix, comment="shared-new", cfg=mock_cfg)
        verify = CredentialDB(db_path, passphrase="test")
        try:
            for scope in ("proj-a", "proj-b"):
                assert verify.list_ssh_keys_for_scope(scope)[0].comment == "shared-new"
        finally:
            verify.close()

    def test_omitted_cfg_falls_back_to_default(
        self,
        db_path: Path,
        mock_cfg: MagicMock,
        patched_open_db,
        capsys: pytest.CaptureFixture[str],
    ) -> None:
        """Argparse-dispatched calls omit ``cfg``; handler builds a default ``SandboxConfig``."""
        _seed(db_path, "proj")
        prefix = _fingerprint_prefix(db_path, "proj")
        with patch("terok_sandbox.config.SandboxConfig", return_value=mock_cfg):
            _handle_ssh_rename(fingerprint=prefix, comment="defaulted")
        assert "Renamed" in capsys.readouterr().out
        verify = CredentialDB(db_path, passphrase="test")
        try:
            assert verify.list_ssh_keys_for_scope("proj")[0].comment == "defaulted"
        finally:
            verify.close()


class TestCoverageGaps:
    """Hit a handful of branches the main test classes don't exercise."""

    def test_print_key_table_empty_says_so(self, capsys: pytest.CaptureFixture[str]) -> None:
        """No keys → friendly "No SSH keys registered." (early-return path)."""
        from terok_sandbox.commands.ssh import _print_key_table

        _print_key_table([])
        assert capsys.readouterr().out.strip() == "No SSH keys registered."

    def test_filter_by_comment_glob(self, db_path: Path) -> None:
        """``_filter_key_rows`` accepts shell-style globs against the comment field."""
        from terok_sandbox.commands.ssh import KeyRow, _filter_key_rows

        rows = [
            KeyRow("a", "deploy-prod", "ed25519", "fp1", "p1", "p1"),
            KeyRow("a", "deploy-dev", "ed25519", "fp2", "p2", "p2"),
            KeyRow("a", "personal", "ed25519", "fp3", "p3", "p3"),
        ]
        out = _filter_key_rows(rows, comment="deploy-*")
        assert [r.comment for r in out] == ["deploy-prod", "deploy-dev"]

    def test_ssh_pub_no_keys(self, db_path: Path, mock_cfg: MagicMock, patched_open_db) -> None:
        """Empty scope → SystemExit instead of an IndexError on ``records[-1]``."""
        CredentialDB(db_path, passphrase="test").close()
        with pytest.raises(SystemExit, match="has no SSH keys assigned"):
            _handle_ssh_pub(scope="empty", cfg=mock_cfg)

    def test_ssh_pub_unknown_key_id(
        self, db_path: Path, mock_cfg: MagicMock, patched_open_db
    ) -> None:
        """``--key-id`` pointing at a row that isn't assigned to *scope* exits cleanly."""
        _seed(db_path, "proj")
        with pytest.raises(SystemExit, match="not assigned to scope"):
            _handle_ssh_pub(scope="proj", key_id=9999, cfg=mock_cfg)

    def test_ssh_pub_specific_key_id(
        self,
        db_path: Path,
        mock_cfg: MagicMock,
        patched_open_db,
        capsys: pytest.CaptureFixture[str],
    ) -> None:
        """``--key-id`` of an assigned key prints just that one line."""
        key_id = _seed(db_path, "proj")
        _handle_ssh_pub(scope="proj", key_id=key_id, cfg=mock_cfg)
        out = capsys.readouterr().out.splitlines()
        assert len(out) == 1
        assert out[0].startswith("ssh-ed25519 ")

    def test_ssh_import_missing_private_key(
        self, tmp_path: Path, mock_cfg: MagicMock, patched_open_db
    ) -> None:
        """Non-existent private-key path → SystemExit before any DB work."""
        with pytest.raises(SystemExit, match="Private key not found"):
            _handle_ssh_import(
                scope="proj",
                private_key=str(tmp_path / "nowhere"),
                cfg=mock_cfg,
            )

    def test_ssh_import_missing_public_key(
        self, tmp_path: Path, mock_cfg: MagicMock, patched_open_db
    ) -> None:
        """Private key present but the ``--public-key`` path is missing."""
        kp = generate_keypair("ed25519", comment="x")
        priv = tmp_path / "priv"
        priv.write_bytes(openssh_pem_of(kp.private_der))
        with pytest.raises(SystemExit, match="Public key not found"):
            _handle_ssh_import(
                scope="proj",
                private_key=str(priv),
                public_key=str(tmp_path / "nowhere.pub"),
                cfg=mock_cfg,
            )

    def test_ssh_import_password_protected_key(
        self, tmp_path: Path, mock_cfg: MagicMock, patched_open_db
    ) -> None:
        """Encrypted private keys are rejected up front with a ``ssh-keygen -p`` hint."""
        from terok_sandbox.vault.ssh import keypair

        priv = tmp_path / "priv"
        priv.write_text("dummy")  # contents don't matter — the loader is patched.
        with (
            patch.object(
                keypair,
                "import_ssh_keypair",
                side_effect=keypair.PasswordProtectedKeyError("encrypted"),
            ),
            pytest.raises(SystemExit, match="ssh-keygen -p -f"),
        ):
            _handle_ssh_import(scope="proj", private_key=str(priv), cfg=mock_cfg)

    def test_ssh_remove_interactive_empty_selection_aborts(
        self,
        db_path: Path,
        mock_cfg: MagicMock,
        patched_open_db,
    ) -> None:
        """Pressing Enter at the interactive selection prompt exits without changes."""
        _seed(db_path, "proj")
        with patch("builtins.input", return_value=""), pytest.raises(SystemExit, match="Aborted"):
            _handle_ssh_remove(cfg=mock_cfg)

    def test_ssh_remove_interactive_eof_aborts(
        self,
        db_path: Path,
        mock_cfg: MagicMock,
        patched_open_db,
    ) -> None:
        """Ctrl-D at the selection prompt aborts cleanly (no traceback)."""
        _seed(db_path, "proj")
        with (
            patch("builtins.input", side_effect=EOFError),
            pytest.raises(SystemExit, match="Aborted"),
        ):
            _handle_ssh_remove(cfg=mock_cfg)

    def test_ssh_remove_confirmation_eof_aborts(
        self,
        db_path: Path,
        mock_cfg: MagicMock,
        patched_open_db,
    ) -> None:
        """Ctrl-D at the per-match confirmation prompt aborts cleanly."""
        _seed(db_path, "proj")
        with (
            patch("builtins.input", side_effect=KeyboardInterrupt),
            pytest.raises(SystemExit, match="Aborted"),
        ):
            _handle_ssh_remove(scope="proj", cfg=mock_cfg)

    def test_ssh_remove_multi_match_listing(
        self,
        db_path: Path,
        mock_cfg: MagicMock,
        patched_open_db,
        capsys: pytest.CaptureFixture[str],
    ) -> None:
        """``Multiple keys match (N):`` precedes the table when more than one match."""
        _seed(db_path, "proj")
        _seed(db_path, "proj")
        with patch("builtins.input", return_value="no"), pytest.raises(SystemExit):
            _handle_ssh_remove(scope="proj", cfg=mock_cfg)
        assert "Multiple keys match (2)" in capsys.readouterr().out


@pytest.mark.parametrize(
    ("handler", "kwargs"),
    [
        (_handle_ssh_list, {}),
        (_handle_ssh_pub, {"scope": "absent"}),
        (_handle_ssh_link, {"key_id": 9999, "scope": "absent"}),
        (
            _handle_ssh_import,
            {"scope": "absent", "private_key": "/nowhere/priv"},
        ),
        (_handle_ssh_add, {"scope": "absent"}),
        (_handle_ssh_export, {"scope": "absent", "out_dir": "/tmp/terok-testing/x"}),
        (_handle_ssh_rename, {"fingerprint": "nope", "comment": "x"}),
        (_handle_ssh_remove, {"scope": "absent"}),
    ],
)
def test_each_ssh_handler_defaults_cfg(
    handler,
    kwargs,
    db_path: Path,
    patched_open_db,
) -> None:
    """Argparse calls every handler without ``cfg=`` — drives past the default-factory line.

    The aim is coverage of ``if cfg is None: cfg = SandboxConfig()``;
    whether the handler then succeeds, raises, or no-ops depends on the
    DB state it sees, which is unrelated to the line under test.
    """
    CredentialDB(db_path, passphrase="test").close()
    try:
        handler(**kwargs)
    except SystemExit:
        pass  # handler-specific exit shape is exercised by its dedicated tests


def test_open_db_threads_through_sandbox_config(tmp_path: Path) -> None:
    """``_open_db`` delegates to ``cfg.open_credential_db(prompt_on_tty=True)``.

    Exercised here because the handler-level tests patch ``_open_db``
    out — that fixture trades coverage of this one-liner for not needing
    a real SQLCipher DB per handler.  This test pays the cost once.
    """
    from terok_sandbox.commands.ssh import _open_db
    from terok_sandbox.config import SandboxConfig

    # All resolver tiers explicitly set so the host's layered config /
    # keyring can't slip a different passphrase in ahead of ours.
    cfg = SandboxConfig(
        state_dir=tmp_path / "state",
        runtime_dir=tmp_path / "rt",
        config_dir=tmp_path / "cfg",
        vault_dir=tmp_path / "vault",
        services_mode="socket",
        credentials_passphrase="test-pass",
        credentials_use_keyring=False,
        credentials_passphrase_command=None,
    )
    cfg.vault_dir.mkdir(parents=True, exist_ok=True)
    CredentialDB(cfg.db_path, passphrase="test-pass").close()

    db = _open_db(cfg)
    try:
        assert db.list_scopes_with_ssh_keys() == []
    finally:
        db.close()
