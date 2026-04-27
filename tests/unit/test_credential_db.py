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


def _seed_legacy_v0_db(db_path: Path) -> bytes:
    """Hand-build a v0 ``ssh_keys`` table with one row; return the public blob."""
    import base64
    import hashlib
    import sqlite3

    from cryptography.hazmat.primitives.asymmetric import ed25519
    from cryptography.hazmat.primitives.serialization import (
        Encoding,
        NoEncryption,
        PrivateFormat,
        PublicFormat,
    )

    priv = ed25519.Ed25519PrivateKey.generate()
    pem = priv.private_bytes(Encoding.PEM, PrivateFormat.OpenSSH, NoEncryption())
    pub_wire = priv.public_key().public_bytes(Encoding.OpenSSH, PublicFormat.OpenSSH)
    pub_blob = base64.b64decode(pub_wire.decode("ascii").split()[1])
    hex_fp = hashlib.sha256(pub_blob).hexdigest()

    conn = sqlite3.connect(str(db_path))
    conn.executescript("""
        CREATE TABLE ssh_keys (
            id           INTEGER PRIMARY KEY AUTOINCREMENT,
            key_type     TEXT    NOT NULL,
            private_pem  BLOB    NOT NULL,
            public_blob  BLOB    NOT NULL,
            comment      TEXT    NOT NULL DEFAULT '',
            fingerprint  TEXT    NOT NULL UNIQUE,
            created_at   TEXT    NOT NULL DEFAULT (datetime('now'))
        );
        CREATE TABLE ssh_key_assignments (
            scope        TEXT    NOT NULL,
            key_id       INTEGER NOT NULL REFERENCES ssh_keys(id) ON DELETE CASCADE,
            assigned_at  TEXT    NOT NULL DEFAULT (datetime('now')),
            PRIMARY KEY (scope, key_id)
        );
        CREATE TABLE ssh_keys_version (
            id      INTEGER PRIMARY KEY CHECK (id = 0),
            version INTEGER NOT NULL
        );
        INSERT OR IGNORE INTO ssh_keys_version (id, version) VALUES (0, 0);
    """)
    conn.execute(
        "INSERT INTO ssh_keys (key_type, private_pem, public_blob, fingerprint)"
        " VALUES (?, ?, ?, ?)",
        ("ed25519", pem, pub_blob, hex_fp),
    )
    conn.execute(
        "INSERT INTO ssh_key_assignments (scope, key_id) VALUES (?, ?)",
        ("proj", 1),
    )
    conn.commit()
    conn.close()
    return pub_blob


class TestSchemaMigration:
    """Verify legacy v0 rows (OpenSSH PEM + hex fingerprint) migrate cleanly."""

    def test_credential_db_open_migrates_v0_to_v1(self, tmp_path: Path) -> None:
        """Opening a v0 DB through [`CredentialDB`][terok_sandbox.CredentialDB] rewrites every row.

        The old shape stored OpenSSH PEM in a ``private_pem`` column with
        64-char hex fingerprints.  The new shape stores PKCS#8 DER in
        ``private_der`` with ``SHA256:<base64>`` fingerprints.
        """
        from cryptography.hazmat.primitives.asymmetric import ed25519
        from cryptography.hazmat.primitives.serialization import load_der_private_key

        db_path = tmp_path / "legacy.db"
        _seed_legacy_v0_db(db_path)

        db = CredentialDB(db_path)
        try:
            cols = {r[1] for r in db._conn.execute("PRAGMA table_info(ssh_keys)").fetchall()}
            assert "private_der" in cols
            assert "private_pem" not in cols
            (fp,) = db._conn.execute("SELECT fingerprint FROM ssh_keys").fetchone()
            assert fp.startswith("SHA256:")
            (der_blob,) = db._conn.execute("SELECT private_der FROM ssh_keys").fetchone()
            # Decode round-trips: the migrated bytes still produce the same key.
            reloaded = load_der_private_key(bytes(der_blob), password=None)
            assert isinstance(reloaded, ed25519.Ed25519PrivateKey)
            (version,) = db._conn.execute("PRAGMA user_version").fetchone()
            assert version == 1
        finally:
            db.close()

        # Re-opening the same DB is a pure no-op.
        db2 = CredentialDB(db_path)
        try:
            (version,) = db2._conn.execute("PRAGMA user_version").fetchone()
            assert version == 1
        finally:
            db2.close()

    def test_token_db_open_also_migrates_v0(self, tmp_path: Path) -> None:
        """The vault daemon's ``_TokenDB`` also runs the migration on open.

        Guards against the upgrade race where the vault daemon restarts
        (via systemd) after a package bump and hits the DB before any
        user-facing CLI command has had a chance to go through
        ``CredentialDB``.  Without this path the daemon would crash on
        its first key lookup with "no such column: private_der".
        """
        from terok_sandbox.vault.token_broker import _TokenDB

        db_path = tmp_path / "legacy.db"
        pub_blob = _seed_legacy_v0_db(db_path)

        token_db = _TokenDB(str(db_path))
        try:
            (version,) = token_db._conn.execute("PRAGMA user_version").fetchone()
            assert version == 1
            [record] = token_db.load_ssh_keys_for_scope("proj")
            assert record.public_blob == pub_blob
            assert record.fingerprint.startswith("SHA256:")
            assert len(record.private_der) > 0
        finally:
            token_db.close()

    def test_token_db_bootstraps_schema_on_empty_db(self, tmp_path: Path) -> None:
        """Fresh-install path: ``_TokenDB`` opens an untouched DB without crashing.

        On a clean host the vault daemon is the *first* thing to open
        ``credentials.db`` — no CLI ``terok auth`` has run yet, so the
        file either does not exist or exists as an empty sqlite3 image
        with no tables.  Without schema bootstrap, the OAuth refresh
        loop's ``SELECT ... FROM credentials`` and the SSH reconciler's
        ``SELECT ... FROM ssh_keys_version`` both raise
        ``OperationalError: no such table`` and the systemd unit dies
        on startup.
        """
        from terok_sandbox.vault.token_broker import _TokenDB

        db_path = tmp_path / "fresh.db"
        # Path doesn't exist yet — sqlite3.connect() creates an empty file.
        token_db = _TokenDB(str(db_path))
        try:
            assert token_db.list_refreshable() == []
            assert token_db.ssh_keys_version() == 0
            assert token_db.load_ssh_keys_for_scope("any") == []
            assert token_db.lookup_token("nonexistent") is None
        finally:
            token_db.close()

    def test_ensure_credentials_schema_is_idempotent(self, tmp_path: Path) -> None:
        """Calling the bootstrap helper twice on the same connection is harmless."""
        import sqlite3

        from terok_sandbox.credentials.db import ensure_credentials_schema

        conn = sqlite3.connect(str(tmp_path / "twice.db"))
        try:
            ensure_credentials_schema(conn)
            ensure_credentials_schema(conn)
            tables = {
                r[0] for r in conn.execute("SELECT name FROM sqlite_master WHERE type='table'")
            }
            assert {
                "credentials",
                "ssh_keys",
                "ssh_key_assignments",
                "ssh_keys_version",
                "proxy_tokens",
            } <= tables
            (version,) = conn.execute("SELECT version FROM ssh_keys_version WHERE id=0").fetchone()
            assert version == 0
        finally:
            conn.close()
