# SPDX-FileCopyrightText: 2025 Jiri Vyskocil
# SPDX-License-Identifier: Apache-2.0

"""SSH keypair generation for a project scope.

:class:`SSHManager` generates an SSH keypair in memory, stores the private
material in the credential DB, and assigns it to a project scope.  The
generated key never touches the filesystem — the signer serves it over the
per-scope agent socket managed by the vault.

See :mod:`.ssh_keypair` for import/export against OpenSSH files and for the
bytes-level keypair vocabulary (``GeneratedKeypair``, fingerprint helpers).
"""

from __future__ import annotations

from pathlib import Path
from typing import TypedDict

from .db import CredentialDB
from .ssh_keypair import DEFAULT_RSA_BITS, GeneratedKeypair, generate_keypair


class SSHInitResult(TypedDict):
    """Public summary of an ``ssh-init`` invocation."""

    key_id: int
    key_type: str
    fingerprint: str
    comment: str
    public_line: str


class SSHManager:
    """Generates SSH keypairs for a scope and stores them in the vault.

    Each scope may hold multiple keys (e.g. GitHub + GitLab), each with a
    distinct fingerprint.  ``init`` is **additive** by default: every call
    generates a new keypair and assigns it alongside any existing keys.
    ``force=True`` **rotates** — the new key is created and assigned
    *before* the scope's previous assignments are revoked, so a mid-run
    crash can leave stale keys (harmless, to be manually cleaned) but
    never leaves the scope with no key at all.

    Two constructors for two ownership stories:

    - ``SSHManager(scope=..., db=...)`` binds the manager to a
      caller-owned :class:`CredentialDB`.  The manager uses it and
      never closes it.  Right shape for tests and pooled connections.
    - :meth:`SSHManager.open` opens its own DB against a path and
      closes it on :meth:`close` / context exit / garbage collection.
      Right shape for one-shot CLI commands.
    """

    def __init__(self, *, scope: str, db: CredentialDB) -> None:
        """Bind the manager to a caller-provided :class:`CredentialDB`."""
        self._scope = scope
        self._db = db
        self._owned_db: CredentialDB | None = None

    @classmethod
    def open(cls, *, scope: str, db_path: Path | str) -> SSHManager:
        """Return a manager that owns its own DB connection.

        The connection is opened against *db_path* and closed when the
        manager exits its context or is garbage-collected.
        """
        db = CredentialDB(Path(db_path))
        manager = cls(scope=scope, db=db)
        manager._owned_db = db
        return manager

    def close(self) -> None:
        """Close the DB connection if this manager opened it (idempotent)."""
        if self._owned_db is not None:
            self._owned_db.close()
            self._owned_db = None

    def __enter__(self) -> SSHManager:
        """Enter the runtime context; returns self."""
        return self

    def __exit__(self, *exc: object) -> None:
        """Close the owned DB on exit."""
        self.close()

    def __del__(self) -> None:
        """Best-effort close on garbage collection."""
        try:
            self.close()
        except Exception:  # noqa: BLE001
            pass

    def init(
        self,
        key_type: str = "ed25519",
        comment: str | None = None,
        force: bool = False,
    ) -> SSHInitResult:
        """Provision a keypair for the scope.

        Args:
            key_type: ``"ed25519"`` (default) or ``"rsa"``.
            comment: Comment to embed in the public key.  Defaults to
                ``tk-main:<scope>`` for the first key, ``tk-side:<scope>:<n>``
                for additional keys (so the signer's ``tk-main:`` promotion
                still picks the primary deploy key).
            force: When ``True``, rotate — drop every *other* key assigned
                to the scope after the new one is stored and assigned.

        Returns:
            Metadata sufficient to display the key to the user or register
            it with a remote.  No filesystem paths.
        """
        existing = self._db.list_ssh_keys_for_scope(self._scope)
        effective_comment = comment or self._default_comment(existing)

        keypair = generate_keypair(key_type, comment=effective_comment)
        key_id = self._db.store_ssh_key(
            key_type=keypair.key_type,
            private_pem=keypair.private_pem,
            public_blob=keypair.public_blob,
            comment=keypair.comment,
            fingerprint=keypair.fingerprint,
        )
        self._db.assign_ssh_key(self._scope, key_id)

        if force:
            # New key is already durable — tear down the old assignments.
            for row in existing:
                if row.id != key_id:
                    self._db.unassign_ssh_key(self._scope, row.id)

        return SSHInitResult(
            key_id=key_id,
            key_type=keypair.key_type,
            fingerprint=keypair.fingerprint,
            comment=keypair.comment,
            public_line=keypair.public_line,
        )

    def _default_comment(self, existing) -> str:
        """Pick a default comment based on whether the scope already has keys.

        The signer's ``tk-main:`` promotion heuristic expects exactly one
        primary key per scope; additional keys use ``tk-side:`` so they
        don't compete for the front of the identity list.
        """
        if existing:
            return f"tk-side:{self._scope}:{len(existing) + 1}"
        return f"tk-main:{self._scope}"


__all__ = ["SSHInitResult", "SSHManager", "DEFAULT_RSA_BITS", "GeneratedKeypair"]
