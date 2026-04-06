# SPDX-FileCopyrightText: 2026 Jiri Vyskocil
# SPDX-License-Identifier: Apache-2.0

"""SQLite-backed credential store and phantom token registry.

Provides host-side storage for captured credentials (API keys, OAuth tokens)
and per-task phantom tokens used by the credential proxy.  The database is
**never** mounted into task containers — only the proxy daemon reads it.

Uses sqlite3 in WAL mode for lock-free concurrent reads across multiple
terok processes (CLI commands, proxy daemon, task runners).  Zero external
dependencies.

Encryption upgrade path: wrap the ``data`` column with
``cryptography.fernet`` before INSERT, or swap ``sqlite3`` for
``sqlcipher3`` (drop-in API replacement).
"""

from __future__ import annotations

import json
import secrets
import sqlite3
from pathlib import Path


class CredentialDB:
    """SQLite-backed credential store and phantom token registry.

    Args:
        db_path: Path to the sqlite3 database file.  Parent directories
            are created automatically.
    """

    def __init__(self, db_path: Path) -> None:
        db_path.parent.mkdir(parents=True, exist_ok=True)
        self._conn = sqlite3.connect(str(db_path), isolation_level="DEFERRED")
        self._conn.execute("PRAGMA journal_mode=WAL")
        self._conn.execute("PRAGMA foreign_keys=ON")
        self._create_tables()

    def _create_tables(self) -> None:
        """Ensure both tables exist (idempotent)."""
        self._conn.executescript("""
            CREATE TABLE IF NOT EXISTS credentials (
                credential_set TEXT NOT NULL,
                provider       TEXT NOT NULL,
                data           TEXT NOT NULL,
                PRIMARY KEY (credential_set, provider)
            );
            CREATE TABLE IF NOT EXISTS proxy_tokens (
                token          TEXT PRIMARY KEY,
                project        TEXT NOT NULL,
                task           TEXT NOT NULL,
                credential_set TEXT NOT NULL,
                provider       TEXT NOT NULL
            );
        """)

    # ── Credentials ──────────────────────────────────────────────────────

    def store_credential(self, credential_set: str, provider: str, data: dict) -> None:
        """Insert or replace a credential entry."""
        self._conn.execute(
            "INSERT OR REPLACE INTO credentials (credential_set, provider, data) VALUES (?, ?, ?)",
            (credential_set, provider, json.dumps(data)),
        )
        self._conn.commit()

    def load_credential(self, credential_set: str, provider: str) -> dict | None:
        """Return the credential dict, or ``None`` if not found."""
        row = self._conn.execute(
            "SELECT data FROM credentials WHERE credential_set = ? AND provider = ?",
            (credential_set, provider),
        ).fetchone()
        return json.loads(row[0]) if row else None

    def list_credentials(self, credential_set: str) -> list[str]:
        """Return provider names that have stored credentials."""
        rows = self._conn.execute(
            "SELECT provider FROM credentials WHERE credential_set = ? ORDER BY provider",
            (credential_set,),
        ).fetchall()
        return [r[0] for r in rows]

    def delete_credential(self, credential_set: str, provider: str) -> None:
        """Remove a credential entry (idempotent)."""
        self._conn.execute(
            "DELETE FROM credentials WHERE credential_set = ? AND provider = ?",
            (credential_set, provider),
        )
        self._conn.commit()

    # ── Phantom tokens ───────────────────────────────────────────────────

    def create_proxy_token(
        self, project: str, task: str, credential_set: str, provider: str
    ) -> str:
        """Create a per-task, per-provider phantom token.

        Token format: ``terok-p-<32 hex chars>``.
        """
        token = f"terok-p-{secrets.token_hex(16)}"
        self._conn.execute(
            "INSERT INTO proxy_tokens (token, project, task, credential_set, provider)"
            " VALUES (?, ?, ?, ?, ?)",
            (token, project, task, credential_set, provider),
        )
        self._conn.commit()
        return token

    def lookup_proxy_token(self, token: str) -> dict | None:
        """Return ``{project, task, credential_set, provider}`` or ``None``."""
        row = self._conn.execute(
            "SELECT project, task, credential_set, provider FROM proxy_tokens WHERE token = ?",
            (token,),
        ).fetchone()
        if row is None:
            return None
        return {"project": row[0], "task": row[1], "credential_set": row[2], "provider": row[3]}

    def revoke_proxy_tokens(self, project: str, task: str) -> int:
        """Revoke all tokens for a project/task pair.  Returns count revoked."""
        cur = self._conn.execute(
            "DELETE FROM proxy_tokens WHERE project = ? AND task = ?",
            (project, task),
        )
        self._conn.commit()
        return cur.rowcount

    def close(self) -> None:
        """Close the database connection."""
        self._conn.close()

    def __del__(self) -> None:
        """Best-effort close on garbage collection."""
        try:
            self._conn.close()
        except Exception:  # noqa: BLE001
            pass
