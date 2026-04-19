# SPDX-FileCopyrightText: 2026 Jiri Vyskocil
# SPDX-License-Identifier: Apache-2.0

"""Tests for :class:`ScopeSocketReconciler` — the per-scope socket manager."""

from __future__ import annotations

from pathlib import Path

import pytest

from terok_sandbox.credentials.db import CredentialDB
from terok_sandbox.credentials.ssh_keypair import generate_keypair
from terok_sandbox.vault.scope_sockets import ScopeSocketReconciler


def _seed(db: CredentialDB, scope: str, comment: str = "c") -> int:
    """Generate a key, store it, and assign it to *scope*; return key id."""
    kp = generate_keypair("ed25519", comment=comment)
    key_id = db.store_ssh_key(
        key_type=kp.key_type,
        private_pem=kp.private_pem,
        public_blob=kp.public_blob,
        comment=kp.comment,
        fingerprint=kp.fingerprint,
    )
    db.assign_ssh_key(scope, key_id)
    return key_id


@pytest.mark.asyncio
class TestReconciler:
    """Verify socket lifecycle tracks assignment state."""

    async def test_initial_bind_creates_sockets(self, tmp_path: Path) -> None:
        """Scopes with assigned keys get a socket each on ``start``."""
        db_path = tmp_path / "vault.db"
        db = CredentialDB(db_path)
        _seed(db, "proj-a")
        _seed(db, "proj-b")
        db.close()

        runtime_dir = tmp_path / "runtime"
        reconciler = ScopeSocketReconciler(db_path=str(db_path), runtime_dir=runtime_dir)
        try:
            await reconciler.start()
            assert (runtime_dir / "ssh-agent-local-proj-a.sock").exists()
            assert (runtime_dir / "ssh-agent-local-proj-b.sock").exists()
        finally:
            await reconciler.stop()

    async def test_unassign_removes_socket_after_poll(self, tmp_path: Path) -> None:
        """Dropping the last assignment cleans up the socket on the next reconcile."""
        import asyncio

        db_path = tmp_path / "vault.db"
        db = CredentialDB(db_path)
        key_id = _seed(db, "proj")

        runtime_dir = tmp_path / "runtime"
        reconciler = ScopeSocketReconciler(db_path=str(db_path), runtime_dir=runtime_dir)
        try:
            await reconciler.start()
            sock = runtime_dir / "ssh-agent-local-proj.sock"
            assert sock.exists()

            db.unassign_ssh_key("proj", key_id)
            # Force a reconciliation without waiting the full poll interval.
            await reconciler._reconcile()  # noqa: SLF001  (test probe)
            # Give the event loop a turn so the server close / unlink completes.
            await asyncio.sleep(0)
            assert not sock.exists()
        finally:
            db.close()
            await reconciler.stop()

    async def test_stop_unlinks_all_sockets(self, tmp_path: Path) -> None:
        """stop() closes every server and unlinks its socket file."""
        db_path = tmp_path / "vault.db"
        db = CredentialDB(db_path)
        _seed(db, "proj")
        db.close()

        runtime_dir = tmp_path / "runtime"
        reconciler = ScopeSocketReconciler(db_path=str(db_path), runtime_dir=runtime_dir)
        await reconciler.start()
        sock = runtime_dir / "ssh-agent-local-proj.sock"
        assert sock.exists()
        await reconciler.stop()
        assert not sock.exists()
