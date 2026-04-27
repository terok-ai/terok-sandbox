# SPDX-FileCopyrightText: 2026 Jiri Vyskocil
# SPDX-License-Identifier: Apache-2.0

"""Tests for [`SSHManager`][terok_sandbox.SSHManager] with the DB-backed storage."""

from __future__ import annotations

from pathlib import Path

import pytest

from terok_sandbox.credentials.db import CredentialDB
from terok_sandbox.credentials.ssh import SSHManager


@pytest.fixture()
def db(tmp_path: Path) -> CredentialDB:
    """Return a fresh DB."""
    return CredentialDB(tmp_path / "vault" / "credentials.db")


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

    def test_default_comment_is_tk_main(self, db: CredentialDB) -> None:
        """First key on a scope gets the ``tk-main:<scope>`` comment."""
        result = SSHManager(scope="myproj", db=db).init()
        assert result["comment"] == "tk-main:myproj"

    def test_explicit_comment_overrides(self, db: CredentialDB) -> None:
        """An explicit comment lands verbatim in the stored record."""
        result = SSHManager(scope="myproj", db=db).init(comment="custom")
        assert result["comment"] == "custom"

    def test_additive_without_force(self, db: CredentialDB) -> None:
        """force=False is additive: each init adds a new key alongside existing ones."""
        first = SSHManager(scope="proj", db=db).init()
        second = SSHManager(scope="proj", db=db).init()
        assert first["key_id"] != second["key_id"]
        rows = db.list_ssh_keys_for_scope("proj")
        assert {r.id for r in rows} == {first["key_id"], second["key_id"]}

    def test_second_key_gets_tk_side_comment(self, db: CredentialDB) -> None:
        """Additional keys use ``tk-side:`` so only one ``tk-main:`` leads the agent."""
        first = SSHManager(scope="proj", db=db).init()
        assert first["comment"].startswith("tk-main:")
        second = SSHManager(scope="proj", db=db).init()
        assert second["comment"].startswith("tk-side:")

    def test_force_rotates_after_new_key_assigned(self, db: CredentialDB) -> None:
        """force=True assigns the new key *before* revoking the old ones."""
        first = SSHManager(scope="proj", db=db).init()
        second = SSHManager(scope="proj", db=db).init(force=True)
        assert first["key_id"] != second["key_id"]
        rows = db.list_ssh_keys_for_scope("proj")
        assert [r.id for r in rows] == [second["key_id"]]

    def test_force_rotation_reseeds_primary_comment(self, db: CredentialDB) -> None:
        """The survivor of a force-rotation is the new primary — ``tk-main:``."""
        SSHManager(scope="proj", db=db).init()
        rotated = SSHManager(scope="proj", db=db).init(force=True)
        assert rotated["comment"] == "tk-main:proj"

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

    def test_invalid_scope_rejected_before_key_material_is_persisted(
        self, db: CredentialDB
    ) -> None:
        """An unsafe scope fails fast — ``ssh_keys`` stays empty."""
        from terok_sandbox.credentials.db import InvalidScopeName

        with pytest.raises(InvalidScopeName):
            SSHManager(scope="../evil", db=db).init()
        # The private key must not have been stored.
        assert db.list_ssh_keys_for_scope("../evil") == []
        # And no orphaned rows crept into ``ssh_keys`` under any scope.
        orphaned = db._conn.execute("SELECT COUNT(*) FROM ssh_keys").fetchone()[0]
        assert orphaned == 0


class TestOwnership:
    """``SSHManager`` owns its DB iff constructed via [`SSHManager.open`][terok_sandbox.SSHManager.open]."""

    def test_context_manager_closes_owned_db(self, tmp_path) -> None:
        """``SSHManager.open`` + ``with`` closes the DB at block exit."""
        import sqlite3

        with SSHManager.open(scope="proj", db_path=tmp_path / "owned.db") as m:
            m.init()  # proves the DB is usable inside the block
        # Any read on the closed connection must raise — proves __exit__ really closed it.
        with pytest.raises(sqlite3.ProgrammingError):
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
