# SPDX-FileCopyrightText: 2026 Jiri Vyskocil
# SPDX-License-Identifier: Apache-2.0

"""Tests for [`CredentialDB`][terok_sandbox.CredentialDB] SSH key + assignment methods."""

from __future__ import annotations

from pathlib import Path

import pytest

from terok_sandbox.vault.store.db import (
    CredentialDB,
    InvalidScopeName,
    UnsafeCommentError,
    _require_safe_scope,
)


@pytest.fixture()
def db(tmp_path: Path) -> CredentialDB:
    """Return a fresh DB rooted under a per-test tmp dir."""
    return CredentialDB(tmp_path / "vault" / "credentials.db", passphrase="test")


def _store_key(db: CredentialDB, fp: str, *, comment: str = "c") -> int:
    """Insert an SSH key with the given fingerprint; return its id."""
    return db.store_ssh_key(
        key_type="ed25519",
        private_der=b"priv-der-" + fp.encode(),
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


class TestCountSshKeys:
    """Verify count_ssh_keys feeds VaultStatus.ssh_keys_stored."""

    def test_zero_when_empty(self, db: CredentialDB) -> None:
        """Fresh DB → no keypairs → 0."""
        assert db.count_ssh_keys() == 0

    def test_counts_distinct_fingerprints(self, db: CredentialDB) -> None:
        """One row per fingerprint, regardless of how many scopes hold the key."""
        k1 = _store_key(db, "fp-1")
        _store_key(db, "fp-2")
        db.assign_ssh_key("proj-a", k1)
        db.assign_ssh_key("proj-b", k1)
        assert db.count_ssh_keys() == 2


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


class TestRoutingProjection:
    """Verify the scope-independent listings that feed a routing matrix."""

    def test_list_all_ssh_keys_is_empty_on_fresh_db(self, db: CredentialDB) -> None:
        """No keys stored → empty row axis."""
        assert db.list_all_ssh_keys() == []

    def test_list_all_ssh_keys_spans_scopes_without_duplication(self, db: CredentialDB) -> None:
        """A key shared across scopes appears once; ordering is by id."""
        k1 = _store_key(db, "fp-1", comment="one")
        k2 = _store_key(db, "fp-2", comment="two")
        db.assign_ssh_key("proj-a", k1)
        db.assign_ssh_key("proj-b", k1)  # k1 shared — still one row
        db.assign_ssh_key("proj-b", k2)
        rows = db.list_all_ssh_keys()
        assert [r.id for r in rows] == [k1, k2]
        assert {r.fingerprint for r in rows} == {"fp-1", "fp-2"}

    def test_list_assignments_returns_full_edge_set(self, db: CredentialDB) -> None:
        """Every (scope, key_id) pair is returned, sorted deterministically."""
        k1 = _store_key(db, "fp-1")
        k2 = _store_key(db, "fp-2")
        db.assign_ssh_key("proj-b", k1)
        db.assign_ssh_key("proj-a", k1)
        db.assign_ssh_key("proj-a", k2)
        assert db.list_ssh_key_assignments() == [
            ("proj-a", k1),
            ("proj-a", k2),
            ("proj-b", k1),
        ]


class TestDeleteSshKey:
    """Verify delete_ssh_key drops a key and cascades its assignments."""

    def test_delete_removes_key_and_all_edges(self, db: CredentialDB) -> None:
        """The key vanishes from every scope in one call."""
        key_id = _store_key(db, "fp-del")
        db.assign_ssh_key("proj-a", key_id)
        db.assign_ssh_key("proj-b", key_id)
        assert db.delete_ssh_key(key_id) is True
        assert db.get_ssh_key_by_fingerprint("fp-del") is None
        assert db.list_ssh_key_assignments() == []

    def test_delete_unknown_key_returns_false(self, db: CredentialDB) -> None:
        """Deleting an absent id is a no-op that reports nothing happened."""
        assert db.delete_ssh_key(9999) is False

    def test_delete_rejects_infra_assigned_key_by_default(self, db: CredentialDB) -> None:
        """A key bound to a ``%`` scope is protected from caller-driven delete."""
        key_id = _store_key(db, "fp-infra-del")
        db.assign_ssh_key("%host", key_id, allow_infra=True)
        with pytest.raises(InvalidScopeName, match="reserved for sandbox"):
            db.delete_ssh_key(key_id)
        assert db.get_ssh_key_by_fingerprint("fp-infra-del") is not None

    def test_delete_accepts_infra_assigned_key_with_flag(self, db: CredentialDB) -> None:
        """Infra-aware callers can decommission a key bound to a reserved scope."""
        key_id = _store_key(db, "fp-infra-del-ok")
        db.assign_ssh_key("%host", key_id, allow_infra=True)
        assert db.delete_ssh_key(key_id, allow_infra=True) is True
        assert db.get_ssh_key_by_fingerprint("fp-infra-del-ok") is None


class TestLoadRecords:
    """Verify load_ssh_keys_for_scope returns full records with raw bytes."""

    def test_carries_bytes(self, db: CredentialDB) -> None:
        """The raw DER and public blob round-trip through the DB."""
        key_id = _store_key(db, "fp-raw", comment="hello")
        db.assign_ssh_key("proj", key_id)
        records = db.load_ssh_keys_for_scope("proj")
        assert len(records) == 1
        r = records[0]
        assert r.id == key_id
        assert r.private_der.startswith(b"priv-der-fp-raw")
        assert r.public_blob == b"pub-blob-fp-raw"
        assert r.comment == "hello"
        assert r.fingerprint == "fp-raw"


class TestSetSshKeyComment:
    """Verify set_ssh_key_comment edits the stored comment in place."""

    def test_updates_existing_row(self, db: CredentialDB) -> None:
        """A matching fingerprint has its comment rewritten and id preserved."""
        key_id = _store_key(db, "fp-rename", comment="old@host")
        assert db.set_ssh_key_comment("fp-rename", "new@host") is True
        row = db.get_ssh_key_by_fingerprint("fp-rename")
        assert row is not None
        assert row.id == key_id
        assert row.comment == "new@host"

    def test_unknown_fingerprint_returns_false(self, db: CredentialDB) -> None:
        """Missing fingerprint → returns False, no row touched."""
        _store_key(db, "fp-other", comment="kept")
        assert db.set_ssh_key_comment("fp-missing", "irrelevant") is False
        row = db.get_ssh_key_by_fingerprint("fp-other")
        assert row is not None
        assert row.comment == "kept"

    def test_rejects_unsafe_input(self, db: CredentialDB) -> None:
        """Control characters are refused and the existing comment is preserved."""
        _store_key(db, "fp-safe", comment="safe")
        with pytest.raises(UnsafeCommentError):
            db.set_ssh_key_comment("fp-safe", "bad\x01comment")
        row = db.get_ssh_key_by_fingerprint("fp-safe")
        assert row is not None
        assert row.comment == "safe"


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
            "%",  # bare sigil, no infra name
            "%Host",  # infra form is lowercase only
            "%host-suffix",  # infra form allows only [a-z]+
            "%host.x",  # ditto
            "prefix%host",  # sigil only valid at position 0
        ],
    )
    def test_rejects_unsafe_names(self, bad: str) -> None:
        """A representative set of hostile inputs are refused."""
        with pytest.raises(InvalidScopeName):
            _require_safe_scope(bad)

    @pytest.mark.parametrize(
        "good",
        [
            "proj",
            "My-Project",
            "alpha.beta",
            "0xdeadbeef",
            "a" * 64,
            "%host",  # reserved infra scope: krun host-side keypair
            "%vault",  # any %[a-z]+ form is structurally accepted
        ],
    )
    def test_accepts_reasonable_names(self, good: str) -> None:
        """Valid scope identifiers pass the guard silently."""
        _require_safe_scope(good)  # must not raise

    def test_assign_rejects_unsafe_scope(self, db: CredentialDB) -> None:
        """[`CredentialDB.assign_ssh_key`][terok_sandbox.vault.store.db.CredentialDB.assign_ssh_key]
        refuses to persist a hostile scope name."""
        key_id = _store_key(db, "fp-x")
        with pytest.raises(InvalidScopeName):
            db.assign_ssh_key("../evil", key_id)

    def test_assign_rejects_infra_scope_by_default(self, db: CredentialDB) -> None:
        """Caller-driven writes can't target the ``%`` infrastructure space.

        The DB-layer validator is structurally permissive (both forms
        round-trip through it), so a separate user-vs-infra guard at the
        write entry point is what keeps a non-CLI bypass from creating
        keys under ``%host`` or any future reserved name.
        """
        key_id = _store_key(db, "fp-infra")
        with pytest.raises(InvalidScopeName, match="reserved for sandbox"):
            db.assign_ssh_key("%host", key_id)

    def test_assign_accepts_infra_scope_with_explicit_flag(self, db: CredentialDB) -> None:
        """Sandbox internals provisioning ``%host`` opt in explicitly."""
        key_id = _store_key(db, "fp-infra-ok")
        db.assign_ssh_key("%host", key_id, allow_infra=True)  # no raise
        rows = db.list_ssh_keys_for_scope("%host")
        assert len(rows) == 1

    def test_unassign_rejects_infra_scope_by_default(self, db: CredentialDB) -> None:
        """User-facing remove paths can't decommission infra-reserved keys."""
        key_id = _store_key(db, "fp-infra-2")
        db.assign_ssh_key("%host", key_id, allow_infra=True)
        with pytest.raises(InvalidScopeName, match="reserved for sandbox"):
            db.unassign_ssh_key("%host", key_id)

    def test_unassign_accepts_infra_scope_with_explicit_flag(self, db: CredentialDB) -> None:
        """Infra-aware callers can decommission a reserved-scope key."""
        key_id = _store_key(db, "fp-infra-unassign")
        db.assign_ssh_key("%host", key_id, allow_infra=True)
        db.unassign_ssh_key("%host", key_id, allow_infra=True)
        assert db.list_ssh_keys_for_scope("%host") == []

    def test_replace_rejects_infra_scope_by_default(self, db: CredentialDB) -> None:
        """[`CredentialDB.replace_ssh_keys_for_scope`][terok_sandbox.vault.store.db.CredentialDB.replace_ssh_keys_for_scope]
        is gated the same as assign."""
        key_id = _store_key(db, "fp-infra-3")
        with pytest.raises(InvalidScopeName, match="reserved for sandbox"):
            db.replace_ssh_keys_for_scope("%host", keep_key_id=key_id)

    def test_replace_accepts_infra_scope_with_explicit_flag(self, db: CredentialDB) -> None:
        """Infra-aware rotation: old infra keys are revoked, new one survives."""
        old = _store_key(db, "fp-infra-old")
        new = _store_key(db, "fp-infra-new")
        db.assign_ssh_key("%host", old, allow_infra=True)
        db.replace_ssh_keys_for_scope("%host", keep_key_id=new, allow_infra=True)
        rows = db.list_ssh_keys_for_scope("%host")
        assert [r.fingerprint for r in rows] == ["fp-infra-new"]

    def test_unassign_all_rejects_infra_scope_by_default(self, db: CredentialDB) -> None:
        """[`CredentialDB.unassign_all_ssh_keys`][terok_sandbox.vault.store.db.CredentialDB.unassign_all_ssh_keys]
        is gated the same as the single-key form."""
        with pytest.raises(InvalidScopeName, match="reserved for sandbox"):
            db.unassign_all_ssh_keys("%host")

    def test_unassign_all_accepts_infra_scope_with_explicit_flag(self, db: CredentialDB) -> None:
        """``allow_infra=True`` lets sandbox internals drop every infra key."""
        k1 = _store_key(db, "fp-infra-a")
        k2 = _store_key(db, "fp-infra-b")
        db.assign_ssh_key("%host", k1, allow_infra=True)
        db.assign_ssh_key("%host", k2, allow_infra=True)
        removed = db.unassign_all_ssh_keys("%host", allow_infra=True)
        assert removed == 2
        assert db.list_ssh_keys_for_scope("%host") == []
