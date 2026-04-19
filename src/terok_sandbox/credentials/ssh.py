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

    Ownership: the manager constructs its own :class:`CredentialDB` from
    the given ``db_path`` and closes it on :meth:`close` / context exit /
    garbage collection — matching the pattern ``CredentialDB`` itself
    uses.  Callers get a clean "open ssh manager, use it, toss it" API
    with no separate resource to track.  For tests and advanced callers
    that already hold a live ``CredentialDB``, construct with the ``db``
    keyword instead; the manager will *not* close a caller-owned DB.
    """

    def __init__(
        self,
        *,
        scope: str,
        db_path: Path | str | None = None,
        db: CredentialDB | None = None,
    ) -> None:
        """Open a manager backed by *db* (caller-owned) or *db_path* (we own it).

        Exactly one of ``db`` and ``db_path`` must be provided.
        """
        if (db is None) == (db_path is None):
            raise ValueError("SSHManager needs exactly one of `db` or `db_path`")
        self._scope = scope
        if db is not None:
            self._db = db
            self._owns_db = False
        else:
            self._db = CredentialDB(Path(db_path))
            self._owns_db = True

    def close(self) -> None:
        """Close the DB connection if this manager opened it (idempotent)."""
        if self._owns_db:
            self._db.close()
            self._owns_db = False

    def __enter__(self) -> "SSHManager":
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
