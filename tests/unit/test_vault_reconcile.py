# SPDX-FileCopyrightText: 2026 Jiri Vyskocil
# SPDX-License-Identifier: Apache-2.0

"""Tests for [`ScopeSocketReconciler`][terok_sandbox.vault.ssh.scope_sockets.ScopeSocketReconciler] — the per-scope socket manager."""

from __future__ import annotations

import asyncio
from pathlib import Path

import pytest

from terok_sandbox.vault.ssh.keypair import generate_keypair
from terok_sandbox.vault.ssh.scope_sockets import ScopeSocketReconciler
from terok_sandbox.vault.store.db import CredentialDB


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
        db = CredentialDB(db_path, passphrase="test")
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
        db = CredentialDB(db_path, passphrase="test")
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
        db = CredentialDB(db_path, passphrase="test")
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
        db = CredentialDB(db_path, passphrase="test")
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
        db = CredentialDB(db_path, passphrase="test")
        _seed(db, "proj")
        db.close()

        runtime_dir = tmp_path / "runtime"
        reconciler = ScopeSocketReconciler(db_path=str(db_path), runtime_dir=runtime_dir)

        async def _boom(**_kwargs):
            raise RuntimeError("synthetic bind failure")

        import terok_sandbox.vault.ssh.scope_sockets as mod

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
        db = CredentialDB(db_path, passphrase="test")
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

    async def test_socket_path_rejects_unsafe_scope(self, tmp_path: Path) -> None:
        """``socket_path`` validates the scope to block path-traversal vectors."""
        from terok_sandbox.vault.store.db import InvalidScopeName

        runtime_dir = tmp_path / "runtime"
        runtime_dir.mkdir()
        reconciler = ScopeSocketReconciler(
            db_path=str(tmp_path / "unused.db"), runtime_dir=runtime_dir
        )
        with pytest.raises(InvalidScopeName):
            reconciler.socket_path("../escape")

    async def test_poll_swallows_inner_reconcile_exceptions(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """A failing reconcile inside ``_poll`` logs and continues, never crashes the task.

        The poller is the long-running heartbeat that keeps scope sockets
        in sync; an exception during one tick (DB locked, file system
        hiccup) must not silently kill it.  Drives the loop by patching
        the 2-second sleep to ``sleep(0)`` so a single tick fires
        immediately.
        """
        import unittest.mock as mock

        # Replace the long inter-tick wait with an immediate yield so the
        # loop actually enters its reconcile branch under test.
        monkeypatch.setattr("terok_sandbox.vault.ssh.scope_sockets._POLL_INTERVAL_SECONDS", 0)
        db_path = tmp_path / "vault.db"
        CredentialDB(db_path, passphrase="test").close()
        reconciler = ScopeSocketReconciler(db_path=str(db_path), runtime_dir=tmp_path / "runtime")
        try:
            # Let ``start()`` complete its initial bind pass before we
            # inject the failure — otherwise the boot reconcile itself
            # would propagate the RuntimeError and stop() would never run.
            await reconciler.start()
            with mock.patch.object(
                reconciler, "_reconcile", side_effect=RuntimeError("simulated tick failure")
            ) as failing_reconcile:
                # Yield enough times for the loop's sleep + reconcile +
                # except cycle to fire at least once.
                for _ in range(5):
                    await asyncio.sleep(0)
                assert failing_reconcile.call_count >= 1
            # The poller task is still alive (the exception arm logged
            # and continued instead of dying).
            assert reconciler._task is not None and not reconciler._task.done()
        finally:
            await reconciler.stop()

    async def test_stop_swallows_wait_closed_errors_on_each_server(self, tmp_path: Path) -> None:
        """``stop()`` keeps tearing down sockets even when one ``wait_closed`` raises.

        The blanket-except around ``await server.wait_closed()`` is best-effort
        cleanup — a thrown shutdown error on one scope must not orphan the
        sockets of every other scope.  Pins the except-arm coverage AND
        the surviving-loop semantic.
        """
        import unittest.mock as mock

        runtime_dir = tmp_path / "runtime"
        runtime_dir.mkdir()
        reconciler = ScopeSocketReconciler(
            db_path=str(tmp_path / "unused.db"), runtime_dir=runtime_dir
        )
        # Materialise two socket files so .unlink(missing_ok=True) has
        # something to remove (and we can verify both are gone after stop).
        for scope in ("proj-a", "proj-b"):
            (runtime_dir / f"ssh-agent-local-{scope}.sock").touch()
        # One scope raises, the other succeeds — both must be cleaned up.
        bad_server = mock.AsyncMock()
        bad_server.close = mock.MagicMock()
        bad_server.wait_closed = mock.AsyncMock(side_effect=RuntimeError("shutdown boom"))
        good_server = mock.AsyncMock()
        good_server.close = mock.MagicMock()
        good_server.wait_closed = mock.AsyncMock()
        reconciler._servers = {"proj-a": bad_server, "proj-b": good_server}
        await reconciler.stop()
        for scope in ("proj-a", "proj-b"):
            assert not (runtime_dir / f"ssh-agent-local-{scope}.sock").exists()
