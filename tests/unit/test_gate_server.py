# SPDX-FileCopyrightText: 2026 Jiri Vyskocil
# SPDX-License-Identifier: Apache-2.0

"""Tests for the per-container [`GateServer`][terok_sandbox.gate.server.GateServer]
component and its single-token store."""

from __future__ import annotations

import asyncio
import base64
import socket as _socket
import subprocess
import urllib.error
import urllib.request
from pathlib import Path

import pytest

from terok_sandbox.gate.server import GateServer, _SingleTokenStore
from tests.constants import LOCALHOST

# ---------------------------------------------------------------------------
# _SingleTokenStore
# ---------------------------------------------------------------------------


class TestSingleTokenStore:
    """The per-container store validates exactly one token."""

    def test_matching_token_returns_scope(self) -> None:
        store = _SingleTokenStore("terok-g-abc", "proj-a")
        assert store.validate("terok-g-abc") == "proj-a"


class TestGateServerStopIdempotent:
    """``stop`` on a never-started gate is a no-op, not a crash."""

    def test_stop_without_start_is_noop(self, tmp_path: Path) -> None:
        """A gate that never bound a listener tolerates ``stop``.

        Covers the teardown path the supervisor takes when a partial
        start aborted before the gate's listener came up.
        """
        gate = GateServer(
            mirror_root=tmp_path,
            token="t",
            scope="p",
            socket_path=tmp_path / "never-bound.sock",
        )

        async def _run() -> None:
            await gate.stop()  # must not raise

        asyncio.run(_run())
        assert gate._server is None

    def test_wrong_token_returns_none(self) -> None:
        store = _SingleTokenStore("terok-g-abc", "proj-a")
        assert store.validate("terok-g-other") is None

    def test_empty_token_returns_none(self) -> None:
        store = _SingleTokenStore("terok-g-abc", "proj-a")
        assert store.validate("") is None


# ---------------------------------------------------------------------------
# GateServer lifecycle — start → serve → stop
# ---------------------------------------------------------------------------


def _git_available() -> bool:
    """Return True when ``git`` is on PATH (the CGI backend needs it)."""
    try:
        subprocess.run(["git", "--version"], capture_output=True, check=True)  # nosec B603 B607
        return True
    except (OSError, subprocess.CalledProcessError):
        return False


def _basic_auth(token: str) -> str:
    """Build a ``Basic`` auth header value with *token* as the username."""
    creds = base64.b64encode(f"{token}:x".encode()).decode()
    return f"Basic {creds}"


def _free_tcp_port() -> int:
    """Return a free loopback TCP port."""
    with _socket.socket(_socket.AF_INET, _socket.SOCK_STREAM) as s:
        s.bind((LOCALHOST, 0))
        return s.getsockname()[1]


def _make_bare_repo(root: Path, name: str) -> None:
    """Create an empty bare repo ``<root>/<name>.git`` for the gate to serve."""
    subprocess.run(  # nosec B603 B607
        ["git", "init", "--bare", str(root / f"{name}.git")],
        check=True,
        capture_output=True,
    )


@pytest.mark.skipif(not _git_available(), reason="needs git for http-backend")
class TestGateServerLifecycle:
    """start → serve a real request → stop, over both transports."""

    def test_requires_a_transport(self, tmp_path: Path) -> None:
        """No socket and no host/port → ValueError at start."""
        gate = GateServer(mirror_root=tmp_path, token="t", scope="p")

        async def _run() -> None:
            with pytest.raises(ValueError, match="socket_path or host"):
                await gate.start()

        asyncio.run(_run())

    def test_tcp_serves_and_stops(self, tmp_path: Path) -> None:
        """A TCP gate authenticates the minted token and shuts down cleanly."""
        _make_bare_repo(tmp_path, "proj-a")
        port = _free_tcp_port()
        gate = GateServer(
            mirror_root=tmp_path,
            token="terok-g-tok",
            scope="proj-a",
            host=LOCALHOST,
            port=port,
        )

        async def _run() -> None:
            await gate.start()
            try:
                url = f"http://{LOCALHOST}:{port}/proj-a.git/info/refs?service=git-upload-pack"
                req = urllib.request.Request(url)
                req.add_header("Authorization", _basic_auth("terok-g-tok"))
                with urllib.request.urlopen(req, timeout=5) as resp:  # nosec B310
                    assert resp.status == 200
                # Wrong token → 403.
                bad = urllib.request.Request(url)
                bad.add_header("Authorization", _basic_auth("terok-g-wrong"))
                with pytest.raises(urllib.error.HTTPError) as exc:
                    urllib.request.urlopen(bad, timeout=5)  # nosec B310
                assert exc.value.code == 403
            finally:
                await gate.stop()

        asyncio.run(_run())
        # After stop, the serve loop has exited and the thread is joined.
        assert gate._thread is None

    def test_socket_serves_and_stops(self, tmp_path: Path) -> None:
        """A Unix-socket gate binds, serves an authenticated request, and stops."""
        mirror = tmp_path / "mirror"
        mirror.mkdir()
        _make_bare_repo(mirror, "proj-a")
        sock_path = tmp_path / "gate-server.sock"
        gate = GateServer(
            mirror_root=mirror,
            token="terok-g-tok",
            scope="proj-a",
            socket_path=sock_path,
        )

        async def _run() -> None:
            await gate.start()
            try:
                assert sock_path.is_socket()
                # Drive a request straight over the Unix socket.
                conn = _socket.socket(_socket.AF_UNIX, _socket.SOCK_STREAM)
                conn.settimeout(5)
                conn.connect(str(sock_path))
                auth = _basic_auth("terok-g-tok")
                request = (
                    "GET /proj-a.git/info/refs?service=git-upload-pack HTTP/1.1\r\n"
                    f"Host: localhost\r\nAuthorization: {auth}\r\nConnection: close\r\n\r\n"
                ).encode()
                conn.sendall(request)
                data = b""
                while True:
                    chunk = conn.recv(4096)
                    if not chunk:
                        break
                    data += chunk
                conn.close()
                assert b"200" in data.split(b"\r\n", 1)[0]
            finally:
                await gate.stop()

        asyncio.run(_run())
        assert gate._thread is None
