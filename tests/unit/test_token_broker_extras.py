# SPDX-FileCopyrightText: 2026 Jiri Vyskocil
# SPDX-License-Identifier: Apache-2.0

"""Tests for token broker: argparse + main(), refresh-loop error handling,
``_run_multi`` socket validation, edge cases in handler + OAuth refresh.
"""

from __future__ import annotations

import asyncio
import json
import socket as _socket
import stat
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest
from aiohttp import web
from aiohttp.test_utils import TestClient, TestServer

from terok_sandbox.credentials.db import CredentialDB
from terok_sandbox.vault.token_broker import (
    _KEY_CLIENT,
    _KEY_REFRESH_TASK,  # noqa: PLC2701
    _build_app,  # noqa: PLC2701
    _do_oauth_refresh,  # noqa: PLC2701
    _refresh_loop,  # noqa: PLC2701
    _run_multi,  # noqa: PLC2701
    _TokenDB,  # noqa: PLC2701
    main,
)

# ---------------------------------------------------------------------------
# _TokenDB.__del__ exception swallow
# ---------------------------------------------------------------------------


class TestTokenDBDel:
    """The destructor must never propagate exceptions."""

    def test_del_swallows_close_error(self, tmp_path: Path) -> None:
        db_path = tmp_path / "creds.db"
        CredentialDB(db_path).close()
        db = _TokenDB(str(db_path))
        # sqlite3.Connection.close is read-only — swap the whole _conn
        # for a stand-in whose close() raises.
        bad = MagicMock()
        bad.close.side_effect = RuntimeError("boom")
        db._conn = bad  # noqa: SLF001
        db.__del__()  # must not raise


# ---------------------------------------------------------------------------
# _do_oauth_refresh: client_secret + scope branches
# ---------------------------------------------------------------------------


class _FakeJsonResponse:
    """Minimal aiohttp-like response stand-in for OAuth refresh tests."""

    def __init__(self, *, status: int, payload: dict | None = None, text: str = "") -> None:
        self.status = status
        self._payload = payload or {}
        self._text = text

    async def __aenter__(self) -> _FakeJsonResponse:
        return self

    async def __aexit__(self, *_exc) -> None:
        return None

    async def json(self) -> dict:
        return self._payload

    async def text(self) -> str:
        return self._text


@pytest.mark.asyncio
class TestDoOAuthRefreshExtras:
    """Cover client_secret + scope inclusion and HTTP error branches."""

    async def test_includes_scope_and_client_secret_when_provided(self) -> None:
        captured: dict = {}

        def post(url, json=None, timeout=None):  # noqa: A002
            captured["url"] = url
            captured["json"] = json
            return _FakeJsonResponse(
                status=200,
                payload={"access_token": "new-tok", "expires_in": 3600},
            )

        session = MagicMock()
        session.post = post
        cred = {"refresh_token": "rt"}
        oauth_cfg = {
            "token_url": "https://oauth.example/token",
            "client_id": "cid",
            "client_secret": "csec",
            "scope": "read write",
        }
        new = await _do_oauth_refresh(session, "p", oauth_cfg, cred)
        assert new["access_token"] == "new-tok"
        assert captured["url"] == "https://oauth.example/token"
        # Both scope and client_secret were included in the payload
        assert captured["json"]["client_secret"] == "csec"
        assert captured["json"]["scope"] == "read write"

    async def test_non_200_response_raises(self) -> None:
        def post(url, json=None, timeout=None):  # noqa: A002
            return _FakeJsonResponse(status=400, text="bad request")

        session = MagicMock()
        session.post = post
        oauth_cfg = {"token_url": "u", "client_id": "c"}
        with pytest.raises(RuntimeError, match="status=400") as exc:
            await _do_oauth_refresh(session, "p", oauth_cfg, {"refresh_token": "rt"})
        # The body must NOT leak into the exception (could contain secrets).
        assert "bad request" not in str(exc.value)

    async def test_preserves_old_refresh_token_when_response_omits_it(self) -> None:
        def post(url, json=None, timeout=None):  # noqa: A002
            return _FakeJsonResponse(
                status=200,
                payload={"access_token": "new-tok"},  # no refresh_token in response
            )

        session = MagicMock()
        session.post = post
        cred = {"refresh_token": "old-rt", "extra": "preserved"}
        new = await _do_oauth_refresh(session, "p", {"token_url": "u", "client_id": "c"}, cred)
        assert new["refresh_token"] == "old-rt"
        # Pre-existing fields are preserved (spread of cred)
        assert new["extra"] == "preserved"


# ---------------------------------------------------------------------------
# _refresh_loop: errors are swallowed; cancellation propagates
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
class TestRefreshLoop:
    """The refresh loop must never crash on errors but must respect cancellation."""

    async def test_cancellation_propagates(self, tmp_path: Path) -> None:
        # Build a minimal app with a short refresh interval; cancel after one iteration.
        app = web.Application()
        with patch("terok_sandbox.vault.token_broker._refresh_all", side_effect=asyncio.sleep):
            task = asyncio.create_task(_refresh_loop(app))
            await asyncio.sleep(0.01)  # let it start
            task.cancel()
            with pytest.raises(asyncio.CancelledError):
                await task

    async def test_exception_in_refresh_does_not_crash_loop(self) -> None:
        """If _refresh_all raises, the loop logs and continues to the next sleep."""
        app = web.Application()
        # First call raises, then we cancel.
        call_count = {"n": 0}

        async def faulty(*_args, **_kwargs) -> None:
            call_count["n"] += 1
            raise RuntimeError("simulated failure")

        with (
            patch("terok_sandbox.vault.token_broker._refresh_all", side_effect=faulty),
            patch("terok_sandbox.vault.token_broker._REFRESH_INTERVAL", 0.01),
        ):
            task = asyncio.create_task(_refresh_loop(app))
            await asyncio.sleep(0.05)  # a few iterations should attempt
            task.cancel()
            with pytest.raises(asyncio.CancelledError):
                await task
        assert call_count["n"] >= 1  # we kept trying despite the exception


# ---------------------------------------------------------------------------
# _run_multi: socket validation paths
# ---------------------------------------------------------------------------


def _bare_app() -> web.Application:
    """Return an app with no on_startup handlers (so AppRunner.setup() is cheap)."""
    return web.Application()


@pytest.mark.asyncio
class TestRunMultiSocketValidation:
    """``_run_multi`` validates the requested socket path before binding."""

    async def test_refuses_regular_file_at_socket_path(self, tmp_path: Path) -> None:
        # Plant a regular file where the socket would go
        sock_path = tmp_path / "vault.sock"
        sock_path.write_text("not-a-socket")

        with pytest.raises(RuntimeError, match="non-socket"):
            await _run_multi(
                _bare_app(),
                sock_path=str(sock_path),
                port=None,
            )

    async def test_refuses_in_use_socket(self, tmp_path: Path) -> None:
        """If a real listener already holds the socket, _run_multi refuses."""
        sock_path = tmp_path / "vault.sock"
        srv = _socket.socket(_socket.AF_UNIX, _socket.SOCK_STREAM)
        srv.bind(str(sock_path))
        srv.listen(1)
        try:
            assert stat.S_ISSOCK(sock_path.lstat().st_mode)
            with pytest.raises(RuntimeError, match="already in use"):
                await _run_multi(
                    _bare_app(),
                    sock_path=str(sock_path),
                    port=None,
                )
        finally:
            srv.close()


# ---------------------------------------------------------------------------
# main() — argparse + dispatch
# ---------------------------------------------------------------------------


def _argv_basics(tmp_path: Path) -> list[str]:
    """Return a minimal valid argv for terok-vault main()."""
    db = tmp_path / "creds.db"
    CredentialDB(db).close()
    routes = tmp_path / "routes.json"
    routes.write_text("{}")
    return [
        "terok-vault",
        f"--socket-path={tmp_path / 'vault.sock'}",
        f"--db-path={db}",
        f"--routes-file={routes}",
    ]


class TestMainCli:
    """main() validates flag combinations and writes the pid file."""

    def test_dispatches_to_run_multi(self, tmp_path: Path) -> None:
        with (
            patch("sys.argv", _argv_basics(tmp_path)),
            patch("terok_sandbox.vault.token_broker.asyncio.run") as run,
        ):
            main()
        run.assert_called_once()

    def test_writes_pid_file(self, tmp_path: Path) -> None:
        pidfile = tmp_path / "vault.pid"
        argv = _argv_basics(tmp_path) + [f"--pid-file={pidfile}"]
        with (
            patch("sys.argv", argv),
            patch("terok_sandbox.vault.token_broker.asyncio.run"),
        ):
            main()
        import os

        assert int(pidfile.read_text()) == os.getpid()

    def test_log_file_handler_used(self, tmp_path: Path) -> None:
        log = tmp_path / "vault.log"
        argv = _argv_basics(tmp_path) + [f"--log-file={log}", "--log-level=DEBUG"]
        with (
            patch("sys.argv", argv),
            patch("terok_sandbox.vault.token_broker.asyncio.run"),
            patch("terok_sandbox.vault.token_broker.logging.basicConfig") as basic_config,
        ):
            main()
        kwargs = basic_config.call_args.kwargs
        # FileHandler appended (not a StreamHandler)
        import logging

        assert any(isinstance(h, logging.FileHandler) for h in kwargs["handlers"])

    def test_ssh_signer_port_and_socket_mutually_exclusive(self, tmp_path: Path) -> None:
        argv = _argv_basics(tmp_path) + [
            "--ssh-signer-port=18732",
            "--ssh-signer-socket-path=/tmp/terok-testing/ssh.sock",
            "--ssh-keys-file=/tmp/terok-testing/keys.json",
        ]
        with patch("sys.argv", argv):
            with pytest.raises(SystemExit):
                main()

    def test_unknown_flag_is_an_error(self, tmp_path: Path) -> None:
        """A legacy flag like the old ``--ssh-keys-file`` is rejected by argparse."""
        argv = _argv_basics(tmp_path) + ["--ssh-keys-file=/tmp/nope.json"]
        with patch("sys.argv", argv), pytest.raises(SystemExit):
            main()

    def test_invalid_log_level_falls_back_to_info(self, tmp_path: Path) -> None:
        """Unknown --log-level value falls through to logging.INFO."""
        argv = _argv_basics(tmp_path) + ["--log-level=NOTALEVEL"]
        with (
            patch("sys.argv", argv),
            patch("terok_sandbox.vault.token_broker.asyncio.run"),
            patch("terok_sandbox.vault.token_broker.logging.basicConfig") as basic_config,
        ):
            main()
        import logging

        assert basic_config.call_args.kwargs["level"] == logging.INFO

    def test_keyboard_interrupt_caught_silently(self, tmp_path: Path) -> None:
        with (
            patch("sys.argv", _argv_basics(tmp_path)),
            patch("terok_sandbox.vault.token_broker.asyncio.run", side_effect=KeyboardInterrupt),
        ):
            main()  # must not raise

    def test_systemexit_caught_silently(self, tmp_path: Path) -> None:
        with (
            patch("sys.argv", _argv_basics(tmp_path)),
            patch("terok_sandbox.vault.token_broker.asyncio.run", side_effect=SystemExit(0)),
        ):
            main()  # must not raise


# ---------------------------------------------------------------------------
# _tcp_port validator (defined inside main)
# ---------------------------------------------------------------------------


class TestTcpPortValidator:
    """The argparse port validator rejects out-of-range numbers."""

    def test_rejects_zero(self, tmp_path: Path) -> None:
        argv = _argv_basics(tmp_path) + ["--port=0"]
        with patch("sys.argv", argv):
            with pytest.raises(SystemExit):
                main()

    def test_rejects_too_large(self, tmp_path: Path) -> None:
        argv = _argv_basics(tmp_path) + ["--port=70000"]
        with patch("sys.argv", argv):
            with pytest.raises(SystemExit):
                main()

    def test_rejects_non_integer(self, tmp_path: Path) -> None:
        argv = _argv_basics(tmp_path) + ["--port=abc"]
        with patch("sys.argv", argv):
            with pytest.raises(SystemExit):
                main()


# ---------------------------------------------------------------------------
# _handle_request: API-key edge cases not covered elsewhere
# ---------------------------------------------------------------------------


def _populate_db_and_routes(
    tmp_path: Path,
    *,
    cred: dict,
    route: dict | None = None,
) -> tuple[Path, Path, str]:
    """Write a credentials DB + routes file and return (db_path, routes_path, phantom)."""
    db_path = tmp_path / "creds.db"
    db = CredentialDB(db_path)
    db.store_credential("default", "myprovider", cred)
    phantom = db.create_token("myproj", "task-1", "default", "myprovider")
    db.close()

    routes_path = tmp_path / "routes.json"
    routes_path.write_text(
        json.dumps(
            {
                "myprovider": route
                or {"upstream": "https://upstream.example", "auth_header": "Authorization"}
            }
        )
    )
    return db_path, routes_path, phantom


@pytest.mark.asyncio
class TestHandleRequestApiKeyEdges:
    """Cover dynamic auth_header and missing-token-field branches."""

    async def test_dynamic_auth_header_uses_x_api_key(self, tmp_path: Path) -> None:
        """auth_header = 'dynamic' for an API-key credential becomes 'x-api-key'."""
        db_path, routes_path, token = _populate_db_and_routes(
            tmp_path,
            cred={"type": "api_key", "key": "real-key"},
            route={"upstream": "http://localhost:1", "auth_header": "dynamic"},
        )
        app = _build_app(str(db_path), str(routes_path))
        # Replace the on-startup-created session with one that captures the request.
        captured: dict = {}

        class _Session:
            def request(self, method, url, *, headers, data, allow_redirects, timeout):
                captured["headers"] = headers

                # Return a context manager mimicking aiohttp's session.request().
                async def _aenter():
                    raise RuntimeError("forced — we only care about headers")

                class _Ctx:
                    async def __aenter__(self):
                        await _aenter()

                    async def __aexit__(self, *_exc):
                        return False

                return _Ctx()

            async def close(self) -> None:
                return None

        # Skip the on_startup handler so we can inject our fake session.
        app.on_startup.clear()
        app[_KEY_CLIENT] = _Session()
        app[_KEY_REFRESH_TASK] = asyncio.create_task(asyncio.sleep(0))

        async with TestClient(TestServer(app)) as client:
            resp = await client.get("/path", headers={"Authorization": f"Bearer {token}"})
            # The forced exception is caught and returns 502
            assert resp.status == 502
        # Verify dynamic resolution → x-api-key
        assert captured["headers"].get("x-api-key") == "real-key"
        assert "Authorization" not in captured["headers"]

    async def test_api_key_missing_token_field_returns_502(self, tmp_path: Path) -> None:
        """API-key credential with neither 'token' nor 'key' → 502."""
        db_path, routes_path, token = _populate_db_and_routes(
            tmp_path, cred={"type": "api_key", "irrelevant": "x"}
        )
        app = _build_app(str(db_path), str(routes_path))
        async with TestClient(TestServer(app)) as client:
            resp = await client.get("/p", headers={"Authorization": f"Bearer {token}"})
            assert resp.status == 502
            text = await resp.text()
            assert "misconfigured" in text.lower()

    async def test_oauth_credential_missing_access_token_returns_502(self, tmp_path: Path) -> None:
        """OAuth credential without access_token → 502."""
        db_path, routes_path, token = _populate_db_and_routes(
            tmp_path, cred={"type": "oauth", "refresh_token": "r"}
        )
        app = _build_app(str(db_path), str(routes_path))
        async with TestClient(TestServer(app)) as client:
            resp = await client.get("/p", headers={"Authorization": f"Bearer {token}"})
            assert resp.status == 502
