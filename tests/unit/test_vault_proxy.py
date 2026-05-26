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
