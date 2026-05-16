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
terok processes (CLI commands, vault daemon, task runners).

Schema declarations and forward migrations live in
[`terok_sandbox.vault.store.migrations`][terok_sandbox.vault.store.migrations]
— this module is the data-access layer only.

The on-disk file is always SQLCipher-encrypted; the passphrase
resolution chain (keyring → ``credentials.passphrase`` config field)
and the SQLCipher open helpers live in
[`terok_sandbox.vault.store.encryption`][terok_sandbox.vault.store.encryption].
"""

from __future__ import annotations

import contextlib
import dataclasses
import json
import re
import secrets
import sqlite3
from collections.abc import Iterator
from pathlib import Path
from typing import Any

import sqlcipher3.dbapi2 as _sqlcipher_dbapi

from .encryption import NoPassphraseError, PassphraseSource, WrongPassphraseError
from .migrations import ensure_credentials_schema, migrate_credential_db_schema

# sqlcipher3 has its own DatabaseError class disjoint from stdlib sqlite3's;
# bad-passphrase failures raise the sqlcipher3 flavour.
_DB_ERRORS: tuple[type[Exception], ...] = (sqlite3.DatabaseError, _sqlcipher_dbapi.DatabaseError)

__all__ = [
    "CredentialDB",
    "InvalidScopeName",
    "NoPassphraseError",
    "PlaintextDBFoundError",
    "SSHKeyRecord",
    "SSHKeyRow",
    "UnsafeCommentError",
    "WrongPassphraseError",
    "ensure_credentials_schema",
    "migrate_credential_db_schema",
    "open_credential_db",
    "open_credential_db_with_source",
]


# ── Scope-name guard ────────────────────────────────────────────────────────

_SCOPE_NAME_RE = re.compile(r"%[a-z]+|[A-Za-z0-9][A-Za-z0-9._-]*")
"""Two parallel forms: ``%name`` for infrastructure scopes reserved by the
sandbox itself (e.g. ``%host`` for the host-side krun keypair), and the
plain ``name`` form for user/project scopes.  The leading ``%`` is a
sigil that lets every call site distinguish operator-controlled scopes
from caller-controlled ones at a glance, without a separate lookup."""

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
    """Reject scope names that would be unsafe as a filename fragment.

    Structural-only — accepts both the user form
    (``[A-Za-z0-9][A-Za-z0-9._-]*``) and the infrastructure form
    (``%[a-z]+``).  Use [`_require_user_scope`][terok_sandbox.vault.store.db._require_user_scope]
    on any write path driven by caller-supplied input so a non-CLI bypass
    can't persist or modify a sandbox-reserved ``%name`` scope.
    """
    if not isinstance(scope, str) or not scope:
        raise InvalidScopeName("scope must be a non-empty string")
    if len(scope) > _MAX_SCOPE_LEN:
        raise InvalidScopeName(f"scope {scope!r} exceeds the {_MAX_SCOPE_LEN}-character limit")
    if not _SCOPE_NAME_RE.fullmatch(scope):
        raise InvalidScopeName(
            f"invalid scope {scope!r}: must match either user form "
            "[A-Za-z0-9][A-Za-z0-9._-]* or infrastructure form %[a-z]+"
        )


def _require_user_scope(scope: str) -> None:
    """Reject scopes that are unsafe **or** reserved for infrastructure use.

    Layered on top of [`_require_safe_scope`][terok_sandbox.vault.store.db._require_safe_scope]:
    same structural checks, plus a refusal of the ``%``-prefixed
    infrastructure form.  Use this on any write-path API that takes a
    caller-controlled scope so a non-CLI bypass (library import,
    automation, future plugin) can't persist to ``%host`` and collide
    with sandbox-reserved credentials.

    Callers that legitimately need to write an infrastructure scope —
    sandbox internals provisioning ``%host`` for the krun host-side
    keypair, future ``%name`` slots — pass ``allow_infra=True`` to the
    underlying DB write method instead.
    """
    _require_safe_scope(scope)
    if scope.startswith("%"):
        raise InvalidScopeName(
            f"scope {scope!r}: '%' prefix is reserved for sandbox "
            "infrastructure scopes and not allowed for caller-driven writes"
        )


# ── Comment-safety guard ────────────────────────────────────────────────────

_UNSAFE_COMMENT_CHARS = re.compile(r"[\x00-\x1F\x7F]")
"""Any C0 control character or DEL.  Newlines break the one-line public-key
contract; ESC (``\\x1B``) enables terminal-escape output spoofing (CWE-150);
the rest have no legitimate place in an SSH key comment."""

_MAX_COMMENT_LEN = 200
"""Bound embedded comments so a pathological input can't bloat every
listing/export/stream indefinitely."""


class UnsafeCommentError(ValueError):
    """Raised when a comment contains control characters or is too long.

    Comments flow into SSH ``authorized_keys`` lines, public-line rendering,
    ``ssh-add -L`` output, and terminal summaries — so embedded newlines or
    escape sequences could break the wire format or spoof terminal output.
    Rejection happens at the storage entry points; every display site then
    trusts the DB to hold only safe strings.
    """


def _require_safe_comment(comment: str) -> str:
    """Validate *comment* and return it unchanged; raise on unsafe input."""
    if not isinstance(comment, str):
        raise UnsafeCommentError(f"comment must be a string, got {type(comment).__name__}")
    if len(comment) > _MAX_COMMENT_LEN:
        raise UnsafeCommentError(
            f"comment exceeds {_MAX_COMMENT_LEN}-character limit ({len(comment)} chars)"
        )
    match = _UNSAFE_COMMENT_CHARS.search(comment)
    if match:
        raise UnsafeCommentError(
            f"comment contains disallowed control character "
            f"\\x{ord(match.group(0)):02x} at position {match.start()}"
        )
    return comment


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


class PlaintextDBFoundError(RuntimeError):
    """A legacy plaintext sqlite DB was found where an encrypted one was expected."""


class CredentialDB:
    """SQLite-backed store for provider credentials, SSH keys, and phantom tokens.

    The on-disk file is always SQLCipher-encrypted.  Callers either
    supply *passphrase* explicitly or leave it ``None`` to walk the
    runtime resolution chain (keyring → ``credentials.passphrase``).
    A missing passphrase raises [`NoPassphraseError`][terok_sandbox.vault.store.db.NoPassphraseError];
    a stale plaintext file raises [`PlaintextDBFoundError`][terok_sandbox.vault.store.db.PlaintextDBFoundError]
    — both are diagnostic-only.  Operator-facing remediation (which CLI
    verb to run, which doc page to read) is the caller's job: library
    code shouldn't bake one frontend's verbs into its exception text.
    """

    def __init__(self, db_path: Path, *, passphrase: str) -> None:
        if not passphrase:
            raise NoPassphraseError(f"no SQLCipher passphrase available for {db_path}")
        db_path.parent.mkdir(parents=True, exist_ok=True)
        self._conn = _open_connection(db_path, passphrase)
        try:
            self._conn.execute("PRAGMA journal_mode=WAL")
            self._conn.execute("PRAGMA foreign_keys=ON")
            ensure_credentials_schema(self._conn)
            migrate_credential_db_schema(self._conn)
        except _DB_ERRORS as exc:
            self._conn.close()
            if _looks_like_plaintext_db(db_path):
                raise PlaintextDBFoundError(
                    f"{db_path} is a legacy plaintext sqlite DB — run "
                    "`terok-sandbox credentials encrypt-db` to migrate it"
                ) from exc
            raise WrongPassphraseError(
                f"could not decrypt {db_path} — wrong passphrase, or the DB was"
                " created with a different key"
            ) from exc

    @contextlib.contextmanager
    def transaction(self) -> Iterator[Any]:
        """Run the body in an explicit ``BEGIN IMMEDIATE`` transaction.

        The bare ``with self._conn`` form is no good for caller-driven
        atomicity: most write methods on this class commit eagerly via
        ``self._conn.commit()`` at the end, which silently ends the
        outer scope mid-block.  This context manager takes the write
        lock up front (``BEGIN IMMEDIATE``) and only commits/rolls
        back on its own exit, so callers can compose read-then-write
        sequences and trust the whole thing serialises against
        concurrent writers.

        Inner write methods that need to participate accept a
        ``commit: bool = True`` opt-out (currently
        [`store_ssh_key`][terok_sandbox.vault.store.db.CredentialDB.store_ssh_key]
        and [`assign_ssh_key`][terok_sandbox.vault.store.db.CredentialDB.assign_ssh_key]);
        pass ``commit=False`` for each call inside the ``transaction()``
        scope so the outer block owns the commit.

        On exit: ``COMMIT`` on clean exit, ``ROLLBACK`` on any
        exception (including ``BaseException`` like
        ``KeyboardInterrupt`` — leaving a half-written ``%scope``
        keypair around would be worse than a re-mint on retry).
        """
        self._conn.execute("BEGIN IMMEDIATE")
        try:
            yield self._conn
        except BaseException:
            self._conn.execute("ROLLBACK")
            raise
        else:
            self._conn.execute("COMMIT")

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
        *,
        commit: bool = True,
    ) -> int:
        """Register a keypair, dedup-by-fingerprint; return the ``ssh_keys.id``.

        When a row with the same fingerprint already exists the stored bytes
        and comment are left untouched (the caller is re-asserting an
        already-known key, which is expected on repeat ``ssh-import``).

        ``commit=False`` skips the per-call commit so the caller can
        compose multiple writes inside a
        [`transaction()`][terok_sandbox.vault.store.db.CredentialDB.transaction]
        scope without each inner write ending the outer atomic block.
        """
        cur = self._conn.execute(
            "INSERT OR IGNORE INTO ssh_keys"
            " (key_type, private_der, public_blob, comment, fingerprint)"
            " VALUES (?, ?, ?, ?, ?)",
            (key_type, private_der, public_blob, comment, fingerprint),
        )
        if cur.rowcount:
            self._bump_ssh_keys_version()
        if commit:
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

    def set_ssh_key_comment(self, fingerprint: str, comment: str) -> bool:
        """Update the comment of the key with *fingerprint*.

        Returns ``True`` if a row was updated, ``False`` if the fingerprint
        is unknown.  The comment is validated by the same safety helper
        that gates ``import_ssh_keypair`` — control characters and
        overlong strings raise
        [`UnsafeCommentError`][terok_sandbox.vault.store.db.UnsafeCommentError]
        so the storage-entry-point invariant holds for this path too.

        Bumps ``ssh_keys_version`` on success so the scope-socket
        reconciler and ssh-signer drop their cached resolved-key state,
        surfacing the new comment to subsequent ``ssh-add -L`` queries
        from the container.
        """
        _require_safe_comment(comment)
        cur = self._conn.execute(
            "UPDATE ssh_keys SET comment = ? WHERE fingerprint = ?",
            (comment, fingerprint),
        )
        if cur.rowcount:
            self._bump_ssh_keys_version()
        self._conn.commit()
        return bool(cur.rowcount)

    def assign_ssh_key(
        self,
        scope: str,
        key_id: int,
        *,
        allow_infra: bool = False,
        commit: bool = True,
    ) -> None:
        """Grant *scope* access to *key_id* (idempotent).

        Rejects unsafe scope names with [`InvalidScopeName`][terok_sandbox.vault.store.db.InvalidScopeName] — the
        value is later embedded in per-scope Unix-socket paths, so
        traversal-like strings (``../``, ``/``) must not be persisted.

        By default also rejects ``%``-prefixed infrastructure scopes so
        callers driven by user input can't write to sandbox-reserved
        names (``%host`` for the krun host-side keypair, future
        ``%name`` slots).  Sandbox internals that legitimately provision
        infrastructure scopes pass ``allow_infra=True``.

        ``commit=False`` skips the per-call commit so the caller can
        compose multiple writes inside a
        [`transaction()`][terok_sandbox.vault.store.db.CredentialDB.transaction]
        scope without each inner write ending the outer atomic block.
        """
        if allow_infra:
            _require_safe_scope(scope)
        else:
            _require_user_scope(scope)
        cur = self._conn.execute(
            "INSERT OR IGNORE INTO ssh_key_assignments (scope, key_id) VALUES (?, ?)",
            (scope, key_id),
        )
        if cur.rowcount:
            self._bump_ssh_keys_version()
        if commit:
            self._conn.commit()

    def unassign_ssh_key(self, scope: str, key_id: int, *, allow_infra: bool = False) -> None:
        """Revoke *scope*'s access to *key_id*; drop the key row if orphaned.

        Refuses ``%``-prefixed infrastructure scopes by default — pair
        with ``allow_infra=True`` for sandbox internals that need to
        decommission a reserved scope.
        """
        if allow_infra:
            _require_safe_scope(scope)
        else:
            _require_user_scope(scope)
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

    def replace_ssh_keys_for_scope(
        self, scope: str, *, keep_key_id: int, allow_infra: bool = False
    ) -> None:
        """Atomically make *keep_key_id* the scope's sole assigned key.

        Wraps the "assign new + revoke every other" sequence in a single
        SQLite transaction so two concurrent ``init(force=True)`` calls
        can't both leave their own keys assigned — whichever transaction
        commits last wins the scope, and exactly one primary survives.
        Orphaned ``ssh_keys`` rows for revoked keys are cleaned up in the
        same step via ``unassign_ssh_key`` semantics.

        Refuses ``%``-prefixed infrastructure scopes by default; sandbox
        internals provisioning infra keys pass ``allow_infra=True``.
        """
        if allow_infra:
            _require_safe_scope(scope)
        else:
            _require_user_scope(scope)
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

    def unassign_all_ssh_keys(self, scope: str, *, allow_infra: bool = False) -> int:
        """Revoke every key currently assigned to *scope*.  Returns count removed.

        Refuses ``%``-prefixed infrastructure scopes by default — pair
        with ``allow_infra=True`` for sandbox internals.
        """
        if allow_infra:
            _require_safe_scope(scope)
        else:
            _require_user_scope(scope)
        key_ids = [
            r[0]
            for r in self._conn.execute(
                "SELECT key_id FROM ssh_key_assignments WHERE scope = ?",
                (scope,),
            ).fetchall()
        ]
        for kid in key_ids:
            self.unassign_ssh_key(scope, kid, allow_infra=allow_infra)
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

    def count_ssh_keys(self) -> int:
        """Return the number of distinct keypairs stored in the DB.

        Counts ``ssh_keys`` rows (deduplicated by fingerprint) rather
        than ``ssh_key_assignments`` rows — a single key shared across
        scopes is one stored key, not N.  Surfaces through
        [`VaultStatus.ssh_keys_stored`][terok_sandbox.VaultStatus] so
        TUI/CLI consumers can show a count without opening the DB
        themselves.
        """
        row = self._conn.execute("SELECT count(*) FROM ssh_keys").fetchone()
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
        [`create_token`][terok_sandbox.vault.store.db.CredentialDB.create_token]
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
        except Exception:  # noqa: BLE001  # nosec B110 — best-effort __del__ close on GC
            pass


# ── Configured-open entry point ─────────────────────────────────────────────


def _open_connection(db_path: Path, passphrase: str) -> Any:
    """Return a sqlcipher3 connection — the only flavour the encrypted DB takes."""
    from .encryption import open_sqlcipher  # noqa: PLC0415

    return open_sqlcipher(db_path, passphrase, isolation_level="DEFERRED")


def _looks_like_plaintext_db(db_path: Path) -> bool:
    """Best-effort post-failure check used to translate sqlcipher errors.

    Only called from the [`CredentialDB`][terok_sandbox.vault.store.db.CredentialDB]
    error path when a SQLCipher open fails — never on the success
    path.  Delegates to the setup-time probe.
    """
    from .encryption import is_plaintext_sqlite  # noqa: PLC0415

    return is_plaintext_sqlite(db_path)


def open_credential_db_with_source(
    db_path: Path,
    *,
    passphrase_file: Path | None = None,
    systemd_creds_file: Path | None = None,
    use_keyring: bool = False,
    passphrase_command: str | None = None,
    config_fallback: str | None = None,
    prompt_on_tty: bool = False,
) -> tuple[CredentialDB, PassphraseSource]:
    """Same as [`open_credential_db`][terok_sandbox.vault.store.db.open_credential_db]
    but also returns which tier the passphrase came from.

    Used by [`VaultStatus`][terok_sandbox.VaultStatus] so the TUI status
    pill can label the unlocked vault by its source without re-walking
    the chain itself.
    """
    from .encryption import resolve_passphrase_with_source  # noqa: PLC0415

    passphrase, source = resolve_passphrase_with_source(
        passphrase_file=passphrase_file,
        systemd_creds_file=systemd_creds_file,
        use_keyring=use_keyring,
        passphrase_command=passphrase_command,
        config_fallback=config_fallback,
        prompt_on_tty=prompt_on_tty,
    )
    if passphrase is None or source is None:
        raise NoPassphraseError(f"no SQLCipher passphrase available for {db_path}")
    return CredentialDB(db_path, passphrase=passphrase), source


def open_credential_db(
    db_path: Path,
    *,
    passphrase_file: Path | None = None,
    systemd_creds_file: Path | None = None,
    use_keyring: bool = False,
    passphrase_command: str | None = None,
    config_fallback: str | None = None,
    prompt_on_tty: bool = False,
) -> CredentialDB:
    """Open the credential DB, resolving the passphrase via the runtime chain.

    Walks: *passphrase_file* (tmpfs session-unlock) → *systemd_creds_file*
    (sealed credential decrypted via ``systemd-creds(1)``) → OS keyring
    (when *use_keyring*) → *passphrase_command* (operator-supplied
    helper, e.g. ``pass show …`` / ``op read …``) → *config_fallback*
    → (when *prompt_on_tty* and a TTY is attached) interactive prompt.
    CLI consumers pass ``prompt_on_tty=True``; daemons leave it
    ``False`` so they fail fast instead of blocking on stdin.
    """
    db, _source = open_credential_db_with_source(
        db_path,
        passphrase_file=passphrase_file,
        systemd_creds_file=systemd_creds_file,
        use_keyring=use_keyring,
        passphrase_command=passphrase_command,
        config_fallback=config_fallback,
        prompt_on_tty=prompt_on_tty,
    )
    return db
