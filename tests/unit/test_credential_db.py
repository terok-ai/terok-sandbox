# SPDX-FileCopyrightText: 2026 Jiri Vyskocil
# SPDX-License-Identifier: Apache-2.0

"""Tests for the credential store and token registry."""

from __future__ import annotations

from pathlib import Path

import pytest

from terok_sandbox.credentials.db import CredentialDB


@pytest.fixture()
def db(tmp_path: Path) -> CredentialDB:
    """Return a fresh CredentialDB backed by a temp file."""
    return CredentialDB(tmp_path / "proxy" / "credentials.db")


class TestCredentialCRUD:
    """Verify credential store operations."""

    def test_store_and_load(self, db: CredentialDB) -> None:
        """Round-trip: store a credential and load it back."""
        db.store_credential("default", "claude", {"type": "oauth", "access_token": "abc"})
        cred = db.load_credential("default", "claude")
        assert cred == {"type": "oauth", "access_token": "abc"}

    def test_load_missing_returns_none(self, db: CredentialDB) -> None:
        """Loading a non-existent credential returns None."""
        assert db.load_credential("default", "nonexistent") is None

    def test_upsert_replaces(self, db: CredentialDB) -> None:
        """Storing the same (set, provider) replaces the previous entry."""
        db.store_credential("default", "claude", {"token": "old"})
        db.store_credential("default", "claude", {"token": "new"})
        assert db.load_credential("default", "claude") == {"token": "new"}

    def test_list_credentials(self, db: CredentialDB) -> None:
        """list_credentials returns all providers for a credential set."""
        db.store_credential("default", "claude", {"k": "1"})
        db.store_credential("default", "codex", {"k": "2"})
        db.store_credential("work", "claude", {"k": "3"})
        assert db.list_credentials("default") == ["claude", "codex"]
        assert db.list_credentials("work") == ["claude"]
        assert db.list_credentials("empty") == []

    def test_delete_credential(self, db: CredentialDB) -> None:
        """Deleting a credential removes it; deleting again is a no-op."""
        db.store_credential("default", "claude", {"k": "1"})
        db.delete_credential("default", "claude")
        assert db.load_credential("default", "claude") is None
        db.delete_credential("default", "claude")  # idempotent

    def test_credential_sets_are_isolated(self, db: CredentialDB) -> None:
        """Different credential sets don't interfere."""
        db.store_credential("default", "claude", {"v": "default"})
        db.store_credential("work", "claude", {"v": "work"})
        assert db.load_credential("default", "claude")["v"] == "default"
        assert db.load_credential("work", "claude")["v"] == "work"


class TestTokens:
    """Verify token lifecycle."""

    def test_create_and_lookup(self, db: CredentialDB) -> None:
        """Create a token, look it up, verify fields."""
        token = db.create_token("proj1", "task-42", "default", "claude")
        assert token.startswith("terok-p-")
        assert len(token) == 8 + 32  # prefix + hex(16 bytes)
        info = db.lookup_token(token)
        assert info == {
            "scope": "proj1",
            "task": "task-42",
            "credential_set": "default",
            "provider": "claude",
        }

    def test_create_with_provider(self, db: CredentialDB) -> None:
        """Per-provider tokens encode the provider name."""
        token = db.create_token("proj1", "task-1", "default", "claude")
        info = db.lookup_token(token)
        assert info is not None
        assert info["provider"] == "claude"

    def test_per_provider_tokens_are_independent(self, db: CredentialDB) -> None:
        """Different providers get different tokens for the same task."""
        t_claude = db.create_token("p", "t", "default", "claude")
        t_vibe = db.create_token("p", "t", "default", "vibe")
        assert t_claude != t_vibe
        assert db.lookup_token(t_claude)["provider"] == "claude"
        assert db.lookup_token(t_vibe)["provider"] == "vibe"

    def test_lookup_invalid_returns_none(self, db: CredentialDB) -> None:
        """Looking up a non-existent token returns None."""
        assert db.lookup_token("nonexistent") is None

    def test_tokens_are_unique(self, db: CredentialDB) -> None:
        """Multiple tokens for the same task are distinct."""
        t1 = db.create_token("p", "t", "default", "claude")
        t2 = db.create_token("p", "t", "default", "claude")
        assert t1 != t2
        assert db.lookup_token(t1) is not None
        assert db.lookup_token(t2) is not None

    def test_revoke_removes_all_for_task(self, db: CredentialDB) -> None:
        """revoke_tokens removes all tokens for a scope+task pair."""
        t1 = db.create_token("proj", "task-1", "default", "claude")
        t2 = db.create_token("proj", "task-1", "default", "claude")
        t3 = db.create_token("proj", "task-2", "default", "claude")
        count = db.revoke_tokens("proj", "task-1")
        assert count == 2
        assert db.lookup_token(t1) is None
        assert db.lookup_token(t2) is None
        assert db.lookup_token(t3) is not None  # different task

    def test_revoke_idempotent(self, db: CredentialDB) -> None:
        """Revoking tokens for a task with none is a no-op."""
        assert db.revoke_tokens("nonexistent", "task") == 0


class TestDBLifecycle:
    """Verify database creation and cleanup."""

    def test_creates_parent_dirs(self, tmp_path: Path) -> None:
        """CredentialDB creates parent directories if missing."""
        deep_path = tmp_path / "a" / "b" / "c" / "db.sqlite3"
        db = CredentialDB(deep_path)
        db.store_credential("s", "p", {"k": "v"})
        assert deep_path.exists()
        db.close()

    def test_wal_mode_enabled(self, tmp_path: Path) -> None:
        """Database uses WAL journal mode for concurrent access."""
        db = CredentialDB(tmp_path / "test.db")
        mode = db._conn.execute("PRAGMA journal_mode").fetchone()[0]
        assert mode == "wal"
        db.close()
