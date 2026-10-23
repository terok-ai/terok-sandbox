# SPDX-FileCopyrightText: 2026 Jiri Vyskocil
# SPDX-License-Identifier: Apache-2.0

"""Tests for [`SSHManager`][terok_sandbox.SSHManager] with the DB-backed storage."""

from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from threading import Barrier

import pytest

from terok_sandbox.vault.ssh.manager import SSHManager
from terok_sandbox.vault.store.db import CredentialDB


@pytest.fixture()
def db(tmp_path: Path) -> CredentialDB:
    """Return a fresh DB."""
    return CredentialDB(tmp_path / "vault" / "credentials.db", passphrase="test")


class TestInit:
    """Verify [`SSHManager.init`][terok_sandbox.SSHManager.init] contract."""

    def test_creates_key_and_assignment(self, db: CredentialDB) -> None:
        """First init on an empty scope generates + assigns a new key."""
        result = SSHManager(scope="proj", db=db).init()
        assert result["key_id"] >= 1
        assert result["key_type"] == "ed25519"
        assert result["public_line"].startswith("ssh-ed25519 ")
        rows = db.list_ssh_keys_for_scope("proj")
        assert len(rows) == 1
        assert rows[0].id == result["key_id"]

    def test_default_comment_uses_scope(self, db: CredentialDB) -> None:
        """First key on a scope gets the ``<scope>-1`` comment."""
        result = SSHManager(scope="myproj", db=db).init()
        assert result["comment"] == "myproj-1"

    def test_explicit_comment_overrides(self, db: CredentialDB) -> None:
        """An explicit comment lands verbatim in the stored record."""
        result = SSHManager(scope="myproj", db=db).init(comment="custom")
        assert result["comment"] == "custom"

    def test_idempotent_without_force_or_comment(self, db: CredentialDB) -> None:
        """Bare re-``init`` on a scope with a primary returns the same key — no fresh mint.

        The operator's mental model is "show me the key for this
        project"; minting a side key on every re-run forced a second
        upstream-registration step that surprised users.  Force and an
        explicit comment still opt back into additive behaviour.
        """
        first = SSHManager(scope="proj", db=db).init()
        second = SSHManager(scope="proj", db=db).init()
        assert first["key_id"] == second["key_id"]
        assert first["public_line"] == second["public_line"]
        rows = db.list_ssh_keys_for_scope("proj")
        assert [r.id for r in rows] == [first["key_id"]]

    def test_explicit_comment_still_mints_side_key(self, db: CredentialDB) -> None:
        """Passing ``comment`` is the explicit signal for "make another key" (e.g. GitHub + GitLab)."""
        first = SSHManager(scope="proj", db=db).init()
        assert first["comment"] == "proj-1"
        second = SSHManager(scope="proj", db=db).init(comment="gitlab-deploy")
        assert second["key_id"] != first["key_id"]
        assert second["comment"] == "gitlab-deploy"
        rows = db.list_ssh_keys_for_scope("proj")
        assert {r.id for r in rows} == {first["key_id"], second["key_id"]}

    def test_force_rotates_after_new_key_assigned(self, db: CredentialDB) -> None:
        """force=True assigns the new key *before* revoking the old ones."""
        first = SSHManager(scope="proj", db=db).init()
        second = SSHManager(scope="proj", db=db).init(force=True)
        assert first["key_id"] != second["key_id"]
        rows = db.list_ssh_keys_for_scope("proj")
        assert [r.id for r in rows] == [second["key_id"]]

    def test_force_rotation_uses_next_unused_comment(self, db: CredentialDB) -> None:
        """Rotation gets the next unused name without reusing a previous label."""
        SSHManager(scope="proj", db=db).init()
        rotated = SSHManager(scope="proj", db=db).init(force=True)
        assert rotated["comment"] == "proj-2"
        assert db.list_ssh_key_defaults() == {"proj": rotated["key_id"]}

    def test_force_rotation_drops_orphaned_ssh_keys_rows(self, db: CredentialDB) -> None:
        """Atomic replace deletes unassigned ``ssh_keys`` rows too (no stale secrets)."""
        first = SSHManager(scope="proj", db=db).init()
        second = SSHManager(scope="proj", db=db).init(force=True)
        ids_in_db = {r[0] for r in db._conn.execute("SELECT id FROM ssh_keys").fetchall()}
        assert ids_in_db == {second["key_id"]}
        assert first["key_id"] not in ids_in_db

    def test_empty_comment_is_preserved_not_defaulted(self, db: CredentialDB) -> None:
        """An explicit ``comment=""`` is passed through verbatim."""
        result = SSHManager(scope="proj", db=db).init(comment="")
        assert result["comment"] == ""

    def test_init_reuses_selected_default_regardless_of_comment(self, db: CredentialDB) -> None:
        """Changing the default assignment changes initialization, not its label."""
        manager = SSHManager(scope="proj", db=db)
        manager.init(comment="tk-main:legacy")
        chosen = manager.mint(comment="deploy")
        db.set_default_ssh_key("proj", chosen["key_id"])
        assert manager.init() == chosen

    def test_invalid_scope_rejected_before_key_material_is_persisted(
        self, db: CredentialDB
    ) -> None:
        """An unsafe scope fails fast — ``ssh_keys`` stays empty."""
        from terok_sandbox.vault.store.db import InvalidScopeName

        with pytest.raises(InvalidScopeName):
            SSHManager(scope="../evil", db=db).init()
        # The private key must not have been stored.
        assert db.list_ssh_keys_for_scope("../evil") == []
        # And no orphaned rows crept into ``ssh_keys`` under any scope.
        orphaned = db._conn.execute("SELECT COUNT(*) FROM ssh_keys").fetchone()[0]
        assert orphaned == 0


class TestMint:
    """Minting always creates a key without promoting it over an existing default."""

    def test_mint_is_additive_and_preserves_default(self, db: CredentialDB) -> None:
        """Unnamed keys receive sequential labels; the first remains default."""
        manager = SSHManager(scope="proj", db=db)
        first, second = manager.mint(), manager.mint()
        assert first["comment"] == "proj-1"
        assert second["comment"] == "proj-2"
        assert first["key_id"] != second["key_id"]
        assert db.list_ssh_key_defaults() == {"proj": first["key_id"]}

    def test_suggested_comment_fills_gap_in_actual_comments(self, db: CredentialDB) -> None:
        """Custom labels and numbering gaps do not make count-based collisions."""
        manager = SSHManager(scope="proj", db=db)
        manager.mint(comment="proj-1")
        manager.mint(comment="proj-3")
        manager.mint(comment="custom")
        assert manager.suggested_comment() == "proj-2"
        assert manager.mint()["comment"] == "proj-2"
        assert manager.suggested_comment() == "proj-4"

    def test_suggested_comment_uses_assigned_shared_keys(self, db: CredentialDB) -> None:
        """A label on a key minted elsewhere still reserves that scope's name."""
        shared = SSHManager(scope="other", db=db).mint(comment="proj-1")
        db.assign_ssh_key("proj", shared["key_id"])
        assert SSHManager(scope="proj", db=db).mint()["comment"] == "proj-2"

    @pytest.mark.parametrize("comment", ["custom", ""])
    def test_explicit_comment_is_preserved(self, db: CredentialDB, comment: str) -> None:
        """Explicit labels, including empty strings, are not replaced."""
        assert SSHManager(scope="proj", db=db).mint(comment=comment)["comment"] == comment

    def test_failed_assignment_rolls_back_stored_key(
        self, db: CredentialDB, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """A failed assignment cannot strand private material in the registry."""

        def fail_assignment(*args, **kwargs):
            raise RuntimeError("assignment failed")

        monkeypatch.setattr(db, "assign_ssh_key", fail_assignment)
        with pytest.raises(RuntimeError, match="assignment failed"):
            SSHManager(scope="proj", db=db).mint()
        assert db.count_ssh_keys() == 0


@pytest.mark.parametrize("initialize", [False, True])
def test_concurrent_creation_preserves_names_and_default(tmp_path: Path, initialize: bool) -> None:
    """Concurrent mints get unique names; concurrent initialization reuses one key."""
    path = tmp_path / "vault.db"
    db = CredentialDB(path, passphrase="test")
    barrier = Barrier(2)

    def create_key():
        """Open an independent writer and race the other caller."""
        writer = CredentialDB(path, passphrase="test")
        try:
            manager = SSHManager(scope="proj", db=writer)
            barrier.wait(timeout=10)
            return manager.init() if initialize else manager.mint()
        finally:
            writer.close()

    try:
        with ThreadPoolExecutor(max_workers=2) as executor:
            futures = [executor.submit(create_key) for _ in range(2)]
            results = [future.result(timeout=15) for future in futures]
        expected = {"proj-1"} if initialize else {"proj-1", "proj-2"}
        assert {result["comment"] for result in results} == expected
        assert db.count_ssh_keys() == len(expected)
        first = next(result for result in results if result["comment"] == "proj-1")
        assert db.list_ssh_key_defaults() == {"proj": first["key_id"]}
    finally:
        db.close()


class TestOwnership:
    """``SSHManager`` owns its DB iff constructed via [`SSHManager.open_for_config`][terok_sandbox.SSHManager.open_for_config]."""

    def test_context_manager_closes_owned_db(self, tmp_path, monkeypatch) -> None:
        """``SSHManager.open_for_config`` + ``with`` closes the DB at block exit."""
        import sqlcipher3.dbapi2 as _sqlcipher_dbapi

        from terok_sandbox import SandboxConfig

        # Pin the chain to a deterministic test passphrase via the desktop keyring
        # tier so we don't depend on any session-file / sealed-cred state
        # on the developer host.
        monkeypatch.setattr(
            "terok_sandbox.vault.store.encryption.load_passphrase_from_desktop_keyring",
            lambda **_kw: "test",
        )
        cfg = SandboxConfig(credentials_use_desktop_keyring=True)
        db_path = tmp_path / "owned.db"
        with SSHManager.open_for_config(scope="proj", cfg=cfg, db_path=db_path) as m:
            m.init()  # proves the DB is usable inside the block
        # Any read on the closed connection must raise — proves __exit__ really closed it.
        with pytest.raises(_sqlcipher_dbapi.ProgrammingError):
            m._db.list_ssh_keys_for_scope("proj")
        # A second close() must be a no-op (idempotent).
        m.close()

    def test_does_not_close_caller_owned_db(self, db: CredentialDB) -> None:
        """Direct constructor = caller-owned DB; survives the manager's exit."""
        with SSHManager(scope="proj", db=db):
            pass
        # If the manager had closed it, this would raise ProgrammingError.
        db.store_credential("default", "probe", {"v": "1"})
        assert db.load_credential("default", "probe") == {"v": "1"}

    def test_rsa_keytype(self, db: CredentialDB) -> None:
        """RSA keytype flows through end-to-end."""
        result = SSHManager(scope="proj", db=db).init(key_type="rsa")
        assert result["key_type"] == "rsa"
        assert result["public_line"].startswith("ssh-rsa ")
