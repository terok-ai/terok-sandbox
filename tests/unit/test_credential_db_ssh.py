# SPDX-FileCopyrightText: 2026 Jiri Vyskocil
# SPDX-License-Identifier: Apache-2.0

"""Tests for :class:`CredentialDB` SSH key + assignment methods."""

from __future__ import annotations

from pathlib import Path

import pytest

from terok_sandbox.credentials.db import CredentialDB, InvalidScopeName, _require_safe_scope


@pytest.fixture()
def db(tmp_path: Path) -> CredentialDB:
    """Return a fresh DB rooted under a per-test tmp dir."""
    return CredentialDB(tmp_path / "vault" / "credentials.db")


def _store_key(db: CredentialDB, fp: str, *, comment: str = "c") -> int:
    """Insert an SSH key with the given fingerprint; return its id."""
    return db.store_ssh_key(
        key_type="ed25519",
        private_pem=b"priv-pem-" + fp.encode(),
        public_blob=b"pub-blob-" + fp.encode(),
        comment=comment,
        fingerprint=fp,
    )


class TestStoreAndDedup:
    """Verify store_ssh_key honours the fingerprint uniqueness constraint."""

    def test_insert_new_key_returns_id(self, db: CredentialDB) -> None:
        """First insert of a fingerprint returns a fresh autoincrement id."""
        key_id = _store_key(db, "fp-1")
        assert key_id >= 1

    def test_duplicate_fingerprint_is_no_op_and_returns_same_id(self, db: CredentialDB) -> None:
        """Second INSERT OR IGNORE on the same fingerprint returns the original id."""
        first = _store_key(db, "fp-1", comment="first")
        second = _store_key(db, "fp-1", comment="SECOND")
        assert first == second
        # The original comment is preserved — re-assertion doesn't mutate stored bytes.
        row = db.get_ssh_key_by_fingerprint("fp-1")
        assert row is not None
        assert row.comment == "first"

    def test_version_bumps_on_new_insert(self, db: CredentialDB) -> None:
        """Every novel key insertion advances ssh_keys_version."""
        v0 = db.ssh_keys_version()
        _store_key(db, "fp-1")
        v1 = db.ssh_keys_version()
        assert v1 > v0

    def test_version_unchanged_on_duplicate_insert(self, db: CredentialDB) -> None:
        """A no-op INSERT OR IGNORE does not bump the version."""
        _store_key(db, "fp-1")
        v = db.ssh_keys_version()
        _store_key(db, "fp-1")
        assert db.ssh_keys_version() == v


class TestAssignments:
    """Verify scope → key_id assignment invariants."""

    def test_assign_and_list(self, db: CredentialDB) -> None:
        """Assigned keys appear in list_ssh_keys_for_scope."""
        key_id = _store_key(db, "fp-1")
        db.assign_ssh_key("proj", key_id)
        rows = db.list_ssh_keys_for_scope("proj")
        assert len(rows) == 1
        assert rows[0].fingerprint == "fp-1"

    def test_multi_scope_assignment(self, db: CredentialDB) -> None:
        """A single key can be assigned to multiple scopes."""
        key_id = _store_key(db, "fp-1")
        db.assign_ssh_key("proj-a", key_id)
        db.assign_ssh_key("proj-b", key_id)
        assert {r.id for r in db.list_ssh_keys_for_scope("proj-a")} == {key_id}
        assert {r.id for r in db.list_ssh_keys_for_scope("proj-b")} == {key_id}

    def test_list_scopes_covers_only_nonempty(self, db: CredentialDB) -> None:
        """list_scopes_with_ssh_keys skips unassigned scopes."""
        k1 = _store_key(db, "fp-1")
        _store_key(db, "fp-2")
        db.assign_ssh_key("has-keys", k1)
        assert db.list_scopes_with_ssh_keys() == ["has-keys"]

    def test_unassign_is_idempotent(self, db: CredentialDB) -> None:
        """Unassigning a non-existent assignment is a no-op."""
        key_id = _store_key(db, "fp-1")
        db.unassign_ssh_key("nope", key_id)  # never assigned → idempotent
        assert db.list_ssh_keys_for_scope("nope") == []


class TestCascadeOrphan:
    """Verify orphan cleanup — a key with no remaining assignments is dropped."""

    def test_removing_last_assignment_drops_key(self, db: CredentialDB) -> None:
        """Unassigning the last scope deletes the ssh_keys row too."""
        key_id = _store_key(db, "fp-1")
        db.assign_ssh_key("proj", key_id)
        db.unassign_ssh_key("proj", key_id)
        assert db.get_ssh_key_by_fingerprint("fp-1") is None

    def test_removing_one_of_many_keeps_key(self, db: CredentialDB) -> None:
        """The key survives while any scope still references it."""
        key_id = _store_key(db, "fp-1")
        db.assign_ssh_key("proj-a", key_id)
        db.assign_ssh_key("proj-b", key_id)
        db.unassign_ssh_key("proj-a", key_id)
        assert db.get_ssh_key_by_fingerprint("fp-1") is not None
        # proj-b can still load the key.
        assert {r.id for r in db.list_ssh_keys_for_scope("proj-b")} == {key_id}

    def test_unassign_all_for_scope(self, db: CredentialDB) -> None:
        """unassign_all_ssh_keys wipes every assignment for a scope."""
        k1 = _store_key(db, "fp-1")
        k2 = _store_key(db, "fp-2")
        db.assign_ssh_key("proj", k1)
        db.assign_ssh_key("proj", k2)
        count = db.unassign_all_ssh_keys("proj")
        assert count == 2
        assert db.list_ssh_keys_for_scope("proj") == []


class TestLoadRecords:
    """Verify load_ssh_keys_for_scope returns full records with raw bytes."""

    def test_carries_bytes(self, db: CredentialDB) -> None:
        """The raw PEM and public blob round-trip through the DB."""
        key_id = _store_key(db, "fp-raw", comment="hello")
        db.assign_ssh_key("proj", key_id)
        records = db.load_ssh_keys_for_scope("proj")
        assert len(records) == 1
        r = records[0]
        assert r.id == key_id
        assert r.private_pem.startswith(b"priv-pem-fp-raw")
        assert r.public_blob == b"pub-blob-fp-raw"
        assert r.comment == "hello"
        assert r.fingerprint == "fp-raw"


class TestScopeNameGuard:
    """``_require_safe_scope`` blocks path-unsafe and oversized scope names."""

    @pytest.mark.parametrize(
        "bad",
        [
            "",  # empty
            "../escape",  # traversal
            "with/slash",  # slash
            "with\\backslash",  # backslash
            "-starts-with-dash",  # leading dash
            ".hidden",  # leading dot
            "_underscore",  # leading underscore
            "space in name",  # whitespace
            "null\x00char",  # NUL byte
            "a" * 65,  # over the 64-char cap
        ],
    )
    def test_rejects_unsafe_names(self, bad: str) -> None:
        """A representative set of hostile inputs are refused."""
        with pytest.raises(InvalidScopeName):
            _require_safe_scope(bad)

    @pytest.mark.parametrize(
        "good",
        ["proj", "My-Project", "alpha.beta", "0xdeadbeef", "a" * 64],
    )
    def test_accepts_reasonable_names(self, good: str) -> None:
        """Valid scope identifiers pass the guard silently."""
        _require_safe_scope(good)  # must not raise

    def test_assign_rejects_unsafe_scope(self, db: CredentialDB) -> None:
        """``assign_ssh_key`` refuses to persist a hostile scope name."""
        key_id = _store_key(db, "fp-x")
        with pytest.raises(InvalidScopeName):
            db.assign_ssh_key("../evil", key_id)
