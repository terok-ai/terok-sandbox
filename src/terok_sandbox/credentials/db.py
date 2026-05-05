# SPDX-FileCopyrightText: 2026 Jiri Vyskocil
# SPDX-License-Identifier: Apache-2.0

"""SQLite-backed credential store, SSH key registry, and phantom token registry.

Provides host-side storage for the three kinds of secret material the vault
mediates:

- **Provider credentials** (API keys, OAuth tokens) stored as JSON blobs keyed
  by ``(credential_set, provider)``.
- **SSH keys** stored as unencrypted PKCS#8 DER + SSH wire-format public blob,
  deduplicated by standard-format fingerprint, linked to project scopes through
  an assignments join table.
- **Phantom tokens** minted per-``(scope, subject)`` so containers can
  authenticate to the vault without ever seeing real credentials.  ``subject``
  is an opaque caller-supplied correlation label — the sandbox stores it
  verbatim and never interprets its contents.  Callers (the orchestrator)
  decide what it identifies; today terok puts the task id there, but the
  sandbox treats it as a string label.

The database is **never** mounted into task containers — only the vault daemon
reads it.  sqlite3 in WAL mode gives lock-free concurrent reads across multiple
terok processes (CLI commands, vault daemon, task runners).  Zero external
dependencies.

Schema declarations and forward migrations live in
[`terok_sandbox.credentials.migrations`][terok_sandbox.credentials.migrations]
— this module is the data-access layer only.

Encryption upgrade path: wrap the ``data`` / ``private_der`` columns with
``cryptography.fernet`` before INSERT, or swap ``sqlite3`` for ``sqlcipher3``
(drop-in API replacement).  A single wrap applies uniformly to both API
credentials and SSH keys.
"""

from __future__ import annotations

import dataclasses
import json
import re
import secrets
import sqlite3
from pathlib import Path

from .migrations import ensure_credentials_schema, migrate_credential_db_schema

__all__ = [
    "CredentialDB",
    "InvalidScopeName",
    "SSHKeyRecord",
    "SSHKeyRow",
    "ensure_credentials_schema",
    "migrate_credential_db_schema",
]


# ── Scope-name guard ────────────────────────────────────────────────────────

_SCOPE_NAME_RE = re.compile(r"[A-Za-z0-9][A-Za-z0-9._-]*")
_MAX_SCOPE_LEN = 64
"""Bound below the 108-byte AF_UNIX path limit once combined with
``ssh-agent-local-<scope>.sock`` + ``runtime_dir``."""


class InvalidScopeName(ValueError):
    """Raised when a scope name would be unsafe as a filesystem path segment.

    Scopes are embedded verbatim in per-scope Unix-socket paths
    (``ssh-agent-local-<scope>.sock``), so unrestricted input could lead
    to traversal (``../``) or oversized sockaddr strings.  Every write
    path that persists a scope validates through this helper first, so
    a malicious or buggy caller can't slip a hostile name past the CLI.
    """


def _require_safe_scope(scope: str) -> None:
    """Reject scope names that would be unsafe as a filename fragment."""
    if not isinstance(scope, str) or not scope:
        raise InvalidScopeName("scope must be a non-empty string")
    if len(scope) > _MAX_SCOPE_LEN:
        raise InvalidScopeName(f"scope {scope!r} exceeds the {_MAX_SCOPE_LEN}-character limit")
    if not _SCOPE_NAME_RE.fullmatch(scope):
        raise InvalidScopeName(
            f"invalid scope {scope!r}: must start with alphanumeric and match "
            "[A-Za-z0-9][A-Za-z0-9._-]*"
        )


# ── Domain types ────────────────────────────────────────────────────────────


@dataclasses.dataclass(frozen=True, slots=True)
class SSHKeyRow:
    """SSH key metadata — everything except the private material.

    Returned from listing operations where the caller wants to render
    information about what is stored without decoding the private key.
    """

    id: int
    key_type: str
    fingerprint: str
    comment: str
    created_at: str


@dataclasses.dataclass(frozen=True, slots=True)
class SSHKeyRecord:
    """SSH key record carrying both metadata and raw key bytes.

    Returned from loading operations that feed the signer.  The raw bytes
    are *not* decoded here — decoding is the signer's responsibility so the
    storage layer stays free of cryptography imports.
    """

    id: int
    key_type: str
    private_der: bytes
    public_blob: bytes
    comment: str
    fingerprint: str


class CredentialDB:
    """SQLite-backed store for provider credentials, SSH keys, and phantom tokens.

    Stores captured provider credentials, SSH keypairs (private DER + public
    blob, assigned to scopes), and issues per-task phantom tokens consumed
    by the vault's token broker and SSH signer.

    Args:
        db_path: Path to the sqlite3 database file.  Parent directories
            are created automatically.
    """

    def __init__(self, db_path: Path) -> None:
        db_path.parent.mkdir(parents=True, exist_ok=True)
        self._conn = sqlite3.connect(str(db_path), isolation_level="DEFERRED")
        self._conn.execute("PRAGMA journal_mode=WAL")
        self._conn.execute("PRAGMA foreign_keys=ON")
        ensure_credentials_schema(self._conn)
        migrate_credential_db_schema(self._conn)

    # ── Provider credentials ────────────────────────────────────────────

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

    # ── SSH keys ────────────────────────────────────────────────────────

    def store_ssh_key(
        self,
        key_type: str,
        private_der: bytes,
        public_blob: bytes,
        comment: str,
        fingerprint: str,
    ) -> int:
        """Register a keypair, dedup-by-fingerprint; return the ``ssh_keys.id``.

        When a row with the same fingerprint already exists the stored bytes
        and comment are left untouched (the caller is re-asserting an
        already-known key, which is expected on repeat ``ssh-import``).
        """
        cur = self._conn.execute(
            "INSERT OR IGNORE INTO ssh_keys"
            " (key_type, private_der, public_blob, comment, fingerprint)"
            " VALUES (?, ?, ?, ?, ?)",
            (key_type, private_der, public_blob, comment, fingerprint),
        )
        if cur.rowcount:
            self._bump_ssh_keys_version()
        self._conn.commit()
        row = self._conn.execute(
            "SELECT id FROM ssh_keys WHERE fingerprint = ?",
            (fingerprint,),
        ).fetchone()
        return row[0]

    def get_ssh_key_by_fingerprint(self, fingerprint: str) -> SSHKeyRow | None:
        """Look up a key by fingerprint; returns metadata only."""
        row = self._conn.execute(
            "SELECT id, key_type, fingerprint, comment, created_at"
            " FROM ssh_keys WHERE fingerprint = ?",
            (fingerprint,),
        ).fetchone()
        return SSHKeyRow(*row) if row else None

    def assign_ssh_key(self, scope: str, key_id: int) -> None:
        """Grant *scope* access to *key_id* (idempotent).

        Rejects unsafe scope names with [`InvalidScopeName`][terok_sandbox.credentials.db.InvalidScopeName] — the
        value is later embedded in per-scope Unix-socket paths, so
        traversal-like strings (``../``, ``/``) must not be persisted.
        """
        _require_safe_scope(scope)
        cur = self._conn.execute(
            "INSERT OR IGNORE INTO ssh_key_assignments (scope, key_id) VALUES (?, ?)",
            (scope, key_id),
        )
        if cur.rowcount:
            self._bump_ssh_keys_version()
        self._conn.commit()

    def unassign_ssh_key(self, scope: str, key_id: int) -> None:
        """Revoke *scope*'s access to *key_id*; drop the key row if orphaned."""
        cur = self._conn.execute(
            "DELETE FROM ssh_key_assignments WHERE scope = ? AND key_id = ?",
            (scope, key_id),
        )
        if cur.rowcount:
            self._conn.execute(
                "DELETE FROM ssh_keys WHERE id = ? AND NOT EXISTS ("
                "  SELECT 1 FROM ssh_key_assignments WHERE key_id = ?"
                ")",
                (key_id, key_id),
            )
            self._bump_ssh_keys_version()
        self._conn.commit()

    def replace_ssh_keys_for_scope(self, scope: str, *, keep_key_id: int) -> None:
        """Atomically make *keep_key_id* the scope's sole assigned key.

        Wraps the "assign new + revoke every other" sequence in a single
        SQLite transaction so two concurrent ``init(force=True)`` calls
        can't both leave their own keys assigned — whichever transaction
        commits last wins the scope, and exactly one primary survives.
        Orphaned ``ssh_keys`` rows for revoked keys are cleaned up in the
        same step via ``unassign_ssh_key`` semantics.
        """
        _require_safe_scope(scope)
        with self._conn:
            self._conn.execute(
                "INSERT OR IGNORE INTO ssh_key_assignments (scope, key_id) VALUES (?, ?)",
                (scope, keep_key_id),
            )
            stale_ids = [
                r[0]
                for r in self._conn.execute(
                    "SELECT key_id FROM ssh_key_assignments WHERE scope = ? AND key_id != ?",
                    (scope, keep_key_id),
                ).fetchall()
            ]
            if stale_ids:
                # ``placeholders`` is a fixed-length string of ``?`` marks,
                # never user input — the variadic IN() clause is the reason
                # we build the SQL with f-string instead of plain params.
                placeholders = ",".join("?" * len(stale_ids))
                self._conn.execute(
                    f"DELETE FROM ssh_key_assignments"  # nosec B608
                    f" WHERE scope = ? AND key_id IN ({placeholders})",
                    (scope, *stale_ids),
                )
                self._conn.execute(
                    f"DELETE FROM ssh_keys WHERE id IN ({placeholders})"  # nosec B608
                    f" AND NOT EXISTS ("
                    f"  SELECT 1 FROM ssh_key_assignments WHERE key_id = ssh_keys.id"
                    f")",
                    tuple(stale_ids),
                )
            self._bump_ssh_keys_version()

    def unassign_all_ssh_keys(self, scope: str) -> int:
        """Revoke every key currently assigned to *scope*.  Returns count removed."""
        key_ids = [
            r[0]
            for r in self._conn.execute(
                "SELECT key_id FROM ssh_key_assignments WHERE scope = ?",
                (scope,),
            ).fetchall()
        ]
        for kid in key_ids:
            self.unassign_ssh_key(scope, kid)
        return len(key_ids)

    def list_ssh_keys_for_scope(self, scope: str) -> list[SSHKeyRow]:
        """Return metadata rows for every key assigned to *scope*."""
        rows = self._conn.execute(
            "SELECT k.id, k.key_type, k.fingerprint, k.comment, k.created_at"
            " FROM ssh_keys k"
            " JOIN ssh_key_assignments a ON a.key_id = k.id"
            " WHERE a.scope = ?"
            " ORDER BY a.assigned_at",
            (scope,),
        ).fetchall()
        return [SSHKeyRow(*r) for r in rows]

    def load_ssh_keys_for_scope(self, scope: str) -> list[SSHKeyRecord]:
        """Return full records (with raw bytes) for every key assigned to *scope*."""
        rows = self._conn.execute(
            "SELECT k.id, k.key_type, k.private_der, k.public_blob,"
            " k.comment, k.fingerprint"
            " FROM ssh_keys k"
            " JOIN ssh_key_assignments a ON a.key_id = k.id"
            " WHERE a.scope = ?"
            " ORDER BY a.assigned_at",
            (scope,),
        ).fetchall()
        return [SSHKeyRecord(*r) for r in rows]

    def list_scopes_with_ssh_keys(self) -> list[str]:
        """Return every scope that currently has at least one assigned key."""
        rows = self._conn.execute(
            "SELECT DISTINCT scope FROM ssh_key_assignments ORDER BY scope",
        ).fetchall()
        return [r[0] for r in rows]

    def ssh_keys_version(self) -> int:
        """Return the monotonic version counter for the SSH key tables.

        Bumped on every successful insert, assignment, or unassignment.
        Readers compare against a cached value to decide whether to reload.
        """
        row = self._conn.execute(
            "SELECT version FROM ssh_keys_version WHERE id = 0",
        ).fetchone()
        return row[0] if row else 0

    def _bump_ssh_keys_version(self) -> None:
        """Increment the SSH key version counter."""
        self._conn.execute(
            "UPDATE ssh_keys_version SET version = version + 1 WHERE id = 0",
        )

    # ── Phantom tokens ──────────────────────────────────────────────────

    def create_token(self, scope: str, subject: str, credential_set: str, provider: str) -> str:
        """Mint a phantom token bound to ``(scope, subject, credential_set, provider)``.

        ``subject`` is an opaque caller-supplied correlation label — the
        sandbox stores it verbatim and never interprets its contents.
        Today terok puts the orchestrator's task id there; the sandbox
        treats the value as a string.

        Token format: ``terok-p-<32 hex chars>``.
        """
        token = f"terok-p-{secrets.token_hex(16)}"
        self._conn.execute(
            "INSERT INTO proxy_tokens (token, scope, subject, credential_set, provider)"
            " VALUES (?, ?, ?, ?, ?)",
            (token, scope, subject, credential_set, provider),
        )
        self._conn.commit()
        return token

    def lookup_token(self, token: str) -> dict | None:
        """Return ``{scope, subject, credential_set, provider}`` or ``None``."""
        row = self._conn.execute(
            "SELECT scope, subject, credential_set, provider FROM proxy_tokens WHERE token = ?",
            (token,),
        ).fetchone()
        if row is None:
            return None
        return {
            "scope": row[0],
            "subject": row[1],
            "credential_set": row[2],
            "provider": row[3],
        }

    def revoke_tokens(self, scope: str, subject: str) -> int:
        """Revoke every phantom token bound to ``(scope, subject)``.

        Returns the number of rows removed.  The sandbox makes no claim
        about what ``subject`` identifies; callers (the orchestrator) pass
        whatever opaque label they used at
        [`create_token`][terok_sandbox.credentials.db.CredentialDB.create_token]
        time.
        """
        cur = self._conn.execute(
            "DELETE FROM proxy_tokens WHERE scope = ? AND subject = ?",
            (scope, subject),
        )
        self._conn.commit()
        return cur.rowcount

    # ── Lifecycle ───────────────────────────────────────────────────────

    def close(self) -> None:
        """Close the database connection."""
        self._conn.close()

    def __del__(self) -> None:
        """Best-effort close on garbage collection."""
        try:
            self._conn.close()
        except Exception:  # noqa: BLE001
            pass
