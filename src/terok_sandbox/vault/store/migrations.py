# SPDX-FileCopyrightText: 2026 Jiri Vyskocil
# SPDX-License-Identifier: Apache-2.0

"""Credential-DB schema bootstrap + forward migrations.

Two functions, both idempotent, both called by every opener of the
sqlite3 file ([`CredentialDB`][terok_sandbox.vault.store.db.CredentialDB]
for writers, the vault daemon's read-only ``_TokenDB`` for readers):

* [`ensure_credentials_schema`][terok_sandbox.vault.store.migrations.ensure_credentials_schema]
  declares the *current* shape via ``CREATE TABLE IF NOT EXISTS`` so
  fresh installs land at the latest schema in one shot.
* [`migrate_credential_db_schema`][terok_sandbox.vault.store.migrations.migrate_credential_db_schema]
  walks legacy DBs forward step by step, gated by ``PRAGMA user_version``
  so already-upgraded files are a no-op.

Splitting these out of ``db.py`` keeps the data-access layer free of
``ALTER TABLE`` machinery and gives schema changes a focused review
target — every future bump touches one file.
"""

from __future__ import annotations

import sqlite3

#: Current credential-DB schema version.  Bump when the on-disk shape
#: changes; add a matching ``current < N`` block in
#: [`migrate_credential_db_schema`][terok_sandbox.vault.store.migrations.migrate_credential_db_schema].
#:
#: Version history:
#:
#: * **v0 → v1** — reshape ``ssh_keys`` (OpenSSH PEM in ``private_pem`` +
#:   hex fingerprints  →  PKCS#8 DER in ``private_der`` +
#:   ``SHA256:<base64>`` fingerprints).
#: * **v1 → v2** — rename ``proxy_tokens.task`` → ``proxy_tokens.subject``
#:   to reflect the column's opaque-label semantics at the
#:   sandbox-orchestrator boundary.
#: * **v2 → v3** — re-key credential / phantom-token ``provider`` values from
#:   agent names (``claude``, ``codex``, …) to the provider names they
#:   authenticate to (``anthropic``, ``openai``, …) so one provider's
#:   credential can serve many agents.
SCHEMA_VERSION = 3


def ensure_credentials_schema(conn: sqlite3.Connection) -> None:
    """Create the credential / SSH-key / phantom-token tables if missing.

    Idempotent — every statement is ``IF NOT EXISTS``.  Exposed at module
    level so every opener of the DB file runs it before issuing queries.
    Without this, a daemon that opens an empty DB on a fresh install
    (before any CLI command has touched the file) hits ``no such table:
    credentials`` on the first query and crashes the unit.
    """
    conn.executescript("""
        CREATE TABLE IF NOT EXISTS credentials (
            credential_set TEXT NOT NULL,
            provider       TEXT NOT NULL,
            data           TEXT NOT NULL,
            PRIMARY KEY (credential_set, provider)
        );
        CREATE TABLE IF NOT EXISTS ssh_keys (
            id           INTEGER PRIMARY KEY AUTOINCREMENT,
            key_type     TEXT    NOT NULL CHECK (key_type IN ('ed25519','rsa')),
            private_der  BLOB    NOT NULL,
            public_blob  BLOB    NOT NULL,
            comment      TEXT    NOT NULL DEFAULT '',
            fingerprint  TEXT    NOT NULL UNIQUE,
            created_at   TEXT    NOT NULL DEFAULT (datetime('now'))
        );
        CREATE TABLE IF NOT EXISTS ssh_key_assignments (
            scope        TEXT    NOT NULL,
            key_id       INTEGER NOT NULL REFERENCES ssh_keys(id) ON DELETE CASCADE,
            assigned_at  TEXT    NOT NULL DEFAULT (datetime('now')),
            PRIMARY KEY (scope, key_id)
        );
        CREATE TABLE IF NOT EXISTS proxy_tokens (
            token          TEXT PRIMARY KEY,
            scope          TEXT NOT NULL,
            subject        TEXT NOT NULL,
            credential_set TEXT NOT NULL,
            provider       TEXT NOT NULL
        );
    """)
    conn.commit()


def migrate_credential_db_schema(conn: sqlite3.Connection) -> None:
    """Walk legacy credential-DB rows forward to the current schema.

    Tracked via ``PRAGMA user_version`` so the whole function is a no-op
    on already-upgraded DBs.  Each ``current < N`` branch handles one
    forward step; the final ``PRAGMA user_version`` set commits the
    whole upgrade in one go.

    Exposed at module level so every opener of the DB file
    ([`CredentialDB`][terok_sandbox.vault.store.db.CredentialDB] for
    writers, ``_TokenDB`` in the vault daemon for readers) runs it
    before issuing queries — otherwise a daemon that restarts before any
    CLI command has touched the DB would hit "no such column: …" on a
    freshly-upgraded host.

    The ``cryptography`` import is scoped to the v0 → v1 branch so
    already-migrated DBs (the common case) don't pay an import cost,
    and the storage module keeps tach-clean at import time.
    """
    (current,) = conn.execute("PRAGMA user_version").fetchone()
    if current >= SCHEMA_VERSION:
        return

    if current < 1:
        _migrate_v0_to_v1(conn)

    if current < 2:
        _migrate_v1_to_v2(conn)

    if current < 3:
        _migrate_v2_to_v3(conn)

    conn.execute(f"PRAGMA user_version = {SCHEMA_VERSION}")
    conn.commit()


def _migrate_v0_to_v1(conn: sqlite3.Connection) -> None:
    """Reshape ``ssh_keys``: OpenSSH PEM + hex fp  →  PKCS#8 DER + ``SHA256:<b64>``.

    ``cryptography`` is imported inside the branch so DBs that never
    held a v0 row (the common case post-rollout) don't pay the import.
    """
    cols = {r[1] for r in conn.execute("PRAGMA table_info(ssh_keys)").fetchall()}
    if not ("private_pem" in cols and "private_der" not in cols):
        return

    import base64 as _b64
    import hashlib as _sha

    from cryptography.hazmat.primitives.serialization import (
        Encoding,
        NoEncryption,
        PrivateFormat,
        load_ssh_private_key,
    )

    def _fp(pub: bytes) -> str:
        """Re-format a public blob's fingerprint as ``SHA256:<base64>``."""
        digest = _sha.sha256(pub).digest()
        return f"SHA256:{_b64.b64encode(digest).decode('ascii').rstrip('=')}"

    rows = conn.execute("SELECT id, private_pem, public_blob FROM ssh_keys").fetchall()
    for row_id, priv_pem, pub_blob in rows:
        key = load_ssh_private_key(bytes(priv_pem), password=None)
        der = key.private_bytes(Encoding.DER, PrivateFormat.PKCS8, NoEncryption())
        conn.execute(
            "UPDATE ssh_keys SET private_pem = ?, fingerprint = ? WHERE id = ?",
            (der, _fp(bytes(pub_blob)), row_id),
        )
    conn.execute("ALTER TABLE ssh_keys RENAME COLUMN private_pem TO private_der")


def _migrate_v1_to_v2(conn: sqlite3.Connection) -> None:
    """Rename ``proxy_tokens.task`` → ``proxy_tokens.subject``.

    The column was always opaque to the sandbox; the new name makes
    that contract explicit at the schema boundary.  ``RENAME COLUMN``
    is supported on SQLite ≥ 3.25 (Ubuntu 22.04 ships 3.37+).
    """
    proxy_cols = {r[1] for r in conn.execute("PRAGMA table_info(proxy_tokens)").fetchall()}
    if "task" in proxy_cols and "subject" not in proxy_cols:
        conn.execute("ALTER TABLE proxy_tokens RENAME COLUMN task TO subject")


#: Agent-name → provider-name re-key applied by the v2 → v3 migration.  The
#: provider is the upstream API a credential authenticates to; keying by it
#: (rather than by the CLI/agent that happens to use it) lets many agents share
#: one provider's credential.  Providers already named for their service
#: (``openrouter``, ``blablador``, ``kisski``, ``coderabbit``) need no rename.
_PROVIDER_RENAMES = {
    "claude": "anthropic",
    "codex": "openai",
    "vibe": "mistral",
    "gh": "github",
    "glab": "gitlab",
    "sonar": "sonarcloud",
}


def _migrate_v2_to_v3(conn: sqlite3.Connection) -> None:
    """Re-key ``provider`` values from agent names to provider names.

    Touches both ``credentials`` (persistent) and ``proxy_tokens`` (ephemeral)
    so a host mid-task stays consistent.  ``UPDATE OR IGNORE`` skips the
    degenerate case where the target name already exists in the same set —
    leaving the legacy row rather than violating the primary key.  Idempotent:
    rows already on a provider name match no ``WHERE`` clause.
    """
    for old_name, new_name in _PROVIDER_RENAMES.items():
        conn.execute(
            "UPDATE OR IGNORE credentials SET provider = ? WHERE provider = ?",
            (new_name, old_name),
        )
        conn.execute(
            "UPDATE OR IGNORE proxy_tokens SET provider = ? WHERE provider = ?",
            (new_name, old_name),
        )
