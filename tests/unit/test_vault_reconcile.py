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
        private_der=kp.private_der,
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

    async def test_socket_path_is_public_accessor(self, tmp_path: Path) -> None:
        """``socket_path(scope)`` renders the canonical path without side effects."""
        runtime_dir = tmp_path / "runtime"
        reconciler = ScopeSocketReconciler(
            db_path=str(tmp_path / "unused.db"), runtime_dir=runtime_dir
        )
        assert reconciler.socket_path("alpha") == runtime_dir / "ssh-agent-local-alpha.sock"

    async def test_unchanged_version_is_a_noop(self, tmp_path: Path) -> None:
        """A second reconcile with the same version doesn't rebind anything."""
        db_path = tmp_path / "vault.db"
        db = CredentialDB(db_path)
        _seed(db, "proj")
        db.close()

        runtime_dir = tmp_path / "runtime"
        reconciler = ScopeSocketReconciler(db_path=str(db_path), runtime_dir=runtime_dir)
        try:
            await reconciler.start()
            existing_server = reconciler._servers["proj"]
            await reconciler._reconcile()
            # Same server object — no unbind/rebind cycle happened.
            assert reconciler._servers["proj"] is existing_server
        finally:
            await reconciler.stop()

    async def test_bind_failure_keeps_version_pinned(self, tmp_path: Path) -> None:
        """A bind failure leaves ``_last_version`` behind so the next tick retries."""
        db_path = tmp_path / "vault.db"
        db = CredentialDB(db_path)
        _seed(db, "proj")
        db.close()

        runtime_dir = tmp_path / "runtime"
        reconciler = ScopeSocketReconciler(db_path=str(db_path), runtime_dir=runtime_dir)

        async def _boom(**_kwargs):
            raise RuntimeError("synthetic bind failure")

        import terok_sandbox.vault.scope_sockets as mod

        original = mod.start_ssh_signer_local
        mod.start_ssh_signer_local = _boom
        try:
            await reconciler._reconcile()
            assert reconciler._last_version == -1  # did not advance
            assert "proj" not in reconciler._servers
        finally:
            mod.start_ssh_signer_local = original
            await reconciler.stop()

    async def test_unbind_on_unknown_scope_is_noop(self, tmp_path: Path) -> None:
        """``_unbind_scope`` returns True when the scope was never bound."""
        runtime_dir = tmp_path / "runtime"
        runtime_dir.mkdir()
        reconciler = ScopeSocketReconciler(
            db_path=str(tmp_path / "unused.db"), runtime_dir=runtime_dir
        )
        assert await reconciler._unbind_scope("ghost") is True

    async def test_unbind_close_failure_keeps_scope_tracked(self, tmp_path: Path) -> None:
        """A ``server.close`` error leaves the scope in ``_servers`` for retry."""
        import unittest.mock as mock

        runtime_dir = tmp_path / "runtime"
        runtime_dir.mkdir()
        reconciler = ScopeSocketReconciler(
            db_path=str(tmp_path / "unused.db"), runtime_dir=runtime_dir
        )
        bad_server = mock.MagicMock()
        bad_server.close.side_effect = RuntimeError("close boom")
        reconciler._servers["proj"] = bad_server

        assert await reconciler._unbind_scope("proj") is False
        assert "proj" in reconciler._servers  # still tracked — retried next tick

    async def test_unbind_unlink_failure_keeps_scope_tracked(self, tmp_path: Path) -> None:
        """An ``unlink`` OSError leaves the scope tracked so the next pass retries."""
        import unittest.mock as mock

        runtime_dir = tmp_path / "runtime"
        runtime_dir.mkdir()
        reconciler = ScopeSocketReconciler(
            db_path=str(tmp_path / "unused.db"), runtime_dir=runtime_dir
        )
        server = mock.AsyncMock()
        server.close = mock.MagicMock()
        server.wait_closed = mock.AsyncMock()
        reconciler._servers["proj"] = server

        def _boom(*_a, **_kw):
            raise OSError("unlink boom")

        with mock.patch("pathlib.Path.unlink", _boom):
            assert await reconciler._unbind_scope("proj") is False
        assert "proj" in reconciler._servers

    async def test_reconcile_unbind_failure_keeps_version_pinned(self, tmp_path: Path) -> None:
        """One failing unbind in a pass stops ``_last_version`` from advancing."""
        import unittest.mock as mock

        db_path = tmp_path / "vault.db"
        db = CredentialDB(db_path)
        key_id = _seed(db, "proj")
        runtime_dir = tmp_path / "runtime"
        reconciler = ScopeSocketReconciler(db_path=str(db_path), runtime_dir=runtime_dir)
        try:
            await reconciler.start()
            # Force the next unbind to fail.
            reconciler._servers["proj"] = mock.MagicMock(
                close=mock.MagicMock(side_effect=RuntimeError("boom"))
            )
            db.unassign_ssh_key("proj", key_id)
            version_before = reconciler._last_version
            await reconciler._reconcile()
            assert reconciler._last_version == version_before  # did not advance
        finally:
            db.close()
            # Clear the poisoned server so stop() doesn't try to close it again.
            reconciler._servers.clear()
            await reconciler.stop()
