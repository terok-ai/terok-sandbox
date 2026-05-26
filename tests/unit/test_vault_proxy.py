# SPDX-FileCopyrightText: 2026 Jiri Vyskocil
# SPDX-License-Identifier: Apache-2.0

"""Tests for [`VaultProxy`][terok_sandbox.vault.daemon.token_broker.VaultProxy].

Pins:

- ``VaultProxy.start()`` brings up the right transport for the given
  bind type and tears it down on ``stop()``.
- The cross-supervisor refresh ``flock`` returns ``None`` when the
  lock is contended and the held fd otherwise.
"""

from __future__ import annotations

import os
from pathlib import Path

import pytest


class TestRefreshLock:
    """``acquire_refresh_lock`` / ``release_refresh_lock`` contract."""

    def test_first_caller_gets_the_lock(self, tmp_path: Path) -> None:
        """The first acquire returns a non-None fd."""
        from terok_sandbox.vault.daemon.token_broker import (
            acquire_refresh_lock,
            release_refresh_lock,
        )

        fd = acquire_refresh_lock(tmp_path / "locks", "default", "claude")
        try:
            assert fd is not None
            assert fd >= 0
        finally:
            if fd is not None:
                release_refresh_lock(fd)

    def test_second_concurrent_caller_returns_none(self, tmp_path: Path) -> None:
        """A second acquire while the first is held returns ``None`` (non-blocking)."""
        from terok_sandbox.vault.daemon.token_broker import (
            acquire_refresh_lock,
            release_refresh_lock,
        )

        first = acquire_refresh_lock(tmp_path / "locks", "default", "claude")
        try:
            assert first is not None
            second = acquire_refresh_lock(tmp_path / "locks", "default", "claude")
            assert second is None
        finally:
            if first is not None:
                release_refresh_lock(first)

    def test_release_unblocks_the_next_acquire(self, tmp_path: Path) -> None:
        """After release, a fresh acquire succeeds again."""
        from terok_sandbox.vault.daemon.token_broker import (
            acquire_refresh_lock,
            release_refresh_lock,
        )

        first = acquire_refresh_lock(tmp_path / "locks", "default", "claude")
        assert first is not None
        release_refresh_lock(first)
        second = acquire_refresh_lock(tmp_path / "locks", "default", "claude")
        assert second is not None
        release_refresh_lock(second)

    def test_traversal_components_stay_inside_lock_dir(self, tmp_path: Path) -> None:
        """A ``credential_set`` / ``provider`` with ``/`` or ``..`` can't escape.

        The composed lock file must land inside *lock_dir* regardless of
        what the credential row carries.
        """
        from terok_sandbox.vault.daemon.token_broker import (
            acquire_refresh_lock,
            release_refresh_lock,
        )

        lock_dir = tmp_path / "locks"
        fd = acquire_refresh_lock(lock_dir, "../../etc", "p/../../../q")
        try:
            assert fd is not None
            created = list(lock_dir.iterdir())
            assert created, "no lock file created"
            for entry in created:
                assert entry.parent == lock_dir
                assert "/" not in entry.name
                assert entry.resolve().parent == lock_dir.resolve()
        finally:
            if fd is not None:
                release_refresh_lock(fd)

    def test_unwritable_dir_returns_none(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """A non-writable lock dir collapses to ``None`` (soft-fail)."""
        from terok_sandbox.vault.daemon import token_broker

        sentinel = tmp_path / "definitely-not-writable"
        sentinel.mkdir()
        sentinel.chmod(0o500)

        def _fake_open(*_args: object, **_kwargs: object) -> int:
            raise OSError("planted")

        # Patch ``os.open`` to simulate the EACCES path without juggling
        # filesystem permissions across the rest of the suite.
        with monkeypatch.context() as patch_ctx:
            patch_ctx.setattr(os, "open", _fake_open)
            assert token_broker.acquire_refresh_lock(sentinel, "x", "y") is None

    def test_mkdir_failure_returns_none(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """A lock-dir mkdir failure (e.g. a file in the way) soft-fails to ``None``."""
        from pathlib import Path as _Path

        from terok_sandbox.vault.daemon import token_broker

        def _fake_mkdir(*_args: object, **_kwargs: object) -> None:
            raise OSError("planted mkdir failure")

        with monkeypatch.context() as patch_ctx:
            patch_ctx.setattr(_Path, "mkdir", _fake_mkdir)
            assert token_broker.acquire_refresh_lock(tmp_path / "locks", "x", "y") is None

    def test_release_swallows_close_error(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """Releasing a bad fd (``OSError`` on close) is swallowed, not raised."""
        from terok_sandbox.vault.daemon import token_broker

        def _fake_close(_fd: int) -> None:
            raise OSError("bad file descriptor")

        with monkeypatch.context() as patch_ctx:
            patch_ctx.setattr(os, "close", _fake_close)
            token_broker.release_refresh_lock(-1)  # must not raise


class TestBindShape:
    """``UnixBind`` / ``TcpBind`` are simple data carriers."""

    def test_unix_bind_carries_socket_path(self, tmp_path: Path) -> None:
        from terok_sandbox.vault.daemon.token_broker import UnixBind

        bind = UnixBind(socket_path=tmp_path / "vault.sock")
        assert bind.socket_path == tmp_path / "vault.sock"

    def test_tcp_bind_pins_host_and_port(self) -> None:
        from terok_sandbox.vault.daemon.token_broker import TcpBind

        bind = TcpBind(host="127.0.0.1", port=43210)
        assert bind.host == "127.0.0.1"
        assert bind.port == 43210


class TestVaultProxyConstruction:
    """``VaultProxy`` wiring — construction stores config without binding.

    Exercises the pure surface (``__init__`` + properties +
    ``_describe_bind`` + ``_compute_refresh_lock_dir``) without standing
    up an aiohttp listener (that's integration territory)."""

    def test_unix_bind_describe_and_lock_dir(self, tmp_path: Path) -> None:
        """A socket-mode proxy describes its unix bind and derives the
        refresh-lock dir from the injected ``runtime_dir``."""
        from terok_sandbox.vault.daemon.token_broker import UnixBind, VaultProxy

        runtime = tmp_path / "run"
        bind = UnixBind(socket_path=tmp_path / "vault.sock")
        proxy = VaultProxy(
            db_path=tmp_path / "vault.db",
            scope_id="proj-a",
            bind=bind,
            runtime_dir=runtime,
        )

        assert proxy.bind is bind
        assert proxy.scope_id == "proj-a"
        assert proxy._describe_bind() == f"unix:{tmp_path / 'vault.sock'}"
        # The lock dir is anchored on the injected runtime_dir (not ambient
        # env), the only reliable anchor under crun's rootless userns.
        assert proxy._compute_refresh_lock_dir() == runtime / "terok" / "vault" / "locks"

    def test_tcp_bind_describe(self, tmp_path: Path) -> None:
        """A TCP-mode proxy describes its ``tcp://host:port`` bind."""
        from terok_sandbox.vault.daemon.token_broker import TcpBind, VaultProxy

        proxy = VaultProxy(
            db_path=tmp_path / "vault.db",
            scope_id=None,
            bind=TcpBind(host="127.0.0.1", port=43210),
            runtime_dir=tmp_path / "run",
        )
        assert proxy.scope_id is None
        assert proxy._describe_bind() == "tcp://127.0.0.1:43210"

    def test_lock_dir_falls_back_to_ambient_without_runtime_dir(self, tmp_path: Path) -> None:
        """Without an injected ``runtime_dir`` the lock dir comes from the
        ambient runtime env (the standalone-caller fallback)."""
        from terok_sandbox.vault.daemon import token_broker
        from terok_sandbox.vault.daemon.token_broker import UnixBind, VaultProxy

        proxy = VaultProxy(
            db_path=tmp_path / "vault.db",
            scope_id=None,
            bind=UnixBind(socket_path=tmp_path / "vault.sock"),
        )
        assert proxy._compute_refresh_lock_dir() == token_broker._ambient_lock_dir()


class TestBindUnixSocket:
    """``_bind_unix_socket`` is the hardened AF_UNIX listener bind.

    Tested directly — it binds a real Unix socket in ``tmp_path`` (no
    aiohttp runner / thread), so the stale-collision / non-socket /
    dead-socket-unlink branches are unit-reachable without standing up
    the full proxy listener.
    """

    @staticmethod
    def _proxy(tmp_path: Path):
        """A construction-only proxy whose ``_bind_unix_socket`` we exercise."""
        from terok_sandbox.vault.daemon.token_broker import UnixBind, VaultProxy

        return VaultProxy(
            db_path=tmp_path / "vault.db",
            scope_id=None,
            bind=UnixBind(socket_path=tmp_path / "sockets" / "vault.sock"),
            runtime_dir=tmp_path / "run",
        )

    def test_binds_fresh_socket_mode_0600(self, tmp_path: Path) -> None:
        """A fresh bind creates a 0600 AF_UNIX socket and makes the parent dir."""
        import socket as _socket
        import stat

        proxy = self._proxy(tmp_path)
        path = tmp_path / "sockets" / "vault.sock"
        sock = proxy._bind_unix_socket(path)
        try:
            assert path.exists()
            assert stat.S_ISSOCK(path.lstat().st_mode)
            # harden_socket pins the inode to owner-only rw.
            assert stat.S_IMODE(path.lstat().st_mode) == 0o600
            assert isinstance(sock, _socket.socket)
        finally:
            sock.close()

    def test_rejects_non_socket_path(self, tmp_path: Path) -> None:
        """A regular file where the socket should go is refused (never unlinked)."""
        proxy = self._proxy(tmp_path)
        path = tmp_path / "sockets" / "vault.sock"
        path.parent.mkdir(parents=True)
        path.write_text("i am a regular file")
        with pytest.raises(RuntimeError, match="Refusing to remove non-socket"):
            proxy._bind_unix_socket(path)
        # The intruder file is left in place.
        assert path.read_text() == "i am a regular file"

    def test_unlinks_stale_dead_socket_and_rebinds(self, tmp_path: Path) -> None:
        """A leftover socket inode with no live peer is unlinked, then re-bound."""
        import socket as _socket

        proxy = self._proxy(tmp_path)
        path = tmp_path / "sockets" / "vault.sock"
        path.parent.mkdir(parents=True)
        # Bind + close a socket to leave a stale (dead) socket inode behind.
        stale = _socket.socket(_socket.AF_UNIX, _socket.SOCK_STREAM)
        stale.bind(str(path))
        stale.close()
        assert path.exists()

        sock = proxy._bind_unix_socket(path)
        try:
            assert path.exists()
        finally:
            sock.close()

    def test_rejects_live_socket_in_use(self, tmp_path: Path) -> None:
        """A live listener already on the path is refused (no unlink, no steal)."""
        import socket as _socket

        proxy = self._proxy(tmp_path)
        path = tmp_path / "sockets" / "vault.sock"
        path.parent.mkdir(parents=True)
        live = _socket.socket(_socket.AF_UNIX, _socket.SOCK_STREAM)
        live.bind(str(path))
        live.listen(1)
        try:
            with pytest.raises(RuntimeError, match="already in use"):
                proxy._bind_unix_socket(path)
        finally:
            live.close()

    def test_closes_socket_when_bind_fails(self, tmp_path: Path) -> None:
        """A bind failure closes the freshly-created FD before re-raising.

        AF_UNIX ``sun_path`` tops out near 108 bytes, so an overlong name
        fails at ``bind`` — exercising the cleanup branch that would
        otherwise leak a descriptor across retries.
        """
        proxy = self._proxy(tmp_path)
        overlong = tmp_path / "sockets" / ("x" * 200 + ".sock")
        with pytest.raises(OSError):
            proxy._bind_unix_socket(overlong)


@pytest.mark.asyncio
class TestVaultProxyLifecycle:
    """``start()`` brings the listener up; ``stop()`` tears it down idempotently.

    Stands up a real (empty) credential DB + routes table and binds an
    actual AF_UNIX listener under ``tmp_path`` — no upstream network, so
    the startup refresh over the empty route table is a no-op.
    """

    @staticmethod
    def _proxy(tmp_path: Path, sock_path: Path):
        from terok_sandbox.vault.daemon.token_broker import UnixBind, VaultProxy
        from terok_sandbox.vault.store.db import CredentialDB

        routes = tmp_path / "routes.json"
        routes.write_text("{}")
        db_path = tmp_path / "creds.db"
        CredentialDB(db_path, passphrase="test").close()
        return VaultProxy(
            db_path=str(db_path),
            scope_id=None,
            bind=UnixBind(socket_path=sock_path),
            routes_path=str(routes),
            runtime_dir=tmp_path / "run",
        )

    async def test_start_binds_socket_then_stop_is_idempotent(self, tmp_path: Path) -> None:
        """start() binds the unix socket; stop() is safe to call twice."""
        import stat

        sock_path = tmp_path / "sockets" / "vault.sock"
        proxy = self._proxy(tmp_path, sock_path)
        await proxy.start()
        try:
            assert stat.S_ISSOCK(sock_path.lstat().st_mode)
        finally:
            await proxy.stop()
        # A second stop on the torn-down proxy is a no-op, not a crash.
        await proxy.stop()
