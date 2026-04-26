# SPDX-FileCopyrightText: 2026 Jiri Vyskocil
# SPDX-License-Identifier: Apache-2.0

"""Tests for the token broker server -- token routing, auth, and forwarding."""

from __future__ import annotations

import json
from pathlib import Path
from unittest.mock import MagicMock

import pytest
from aiohttp import web

from terok_sandbox.credentials.db import CredentialDB
from terok_sandbox.vault.token_broker import (
    _KEY_CLIENT,
    _KEY_ROUTES,
    _KEY_TOKEN_DB,
    _build_app,
    _do_oauth_refresh,
    _extract_phantom_token,
    _refresh_all,
    _RouteTable,
    _run_multi,
    _TokenDB,
)

# ── Health endpoint ──────────────────────────────────────────────────────


@pytest.mark.asyncio
class TestHealthEndpoint:
    """Verify the /-/health readiness probe."""

    @pytest.fixture()
    def _app(self, tmp_path: Path) -> web.Application:
        """Build a minimal app with a valid (empty) routes file and DB."""
        routes_file = tmp_path / "routes.json"
        routes_file.write_text("{}")
        db_path = tmp_path / "creds.db"
        CredentialDB(db_path).close()
        return _build_app(str(db_path), str(routes_file))

    async def test_health_returns_200(self, _app) -> None:
        """GET /-/health returns 200 with status ok."""
        from aiohttp.test_utils import TestClient, TestServer

        async with TestClient(TestServer(_app)) as client:
            resp = await client.get("/-/health")
            assert resp.status == 200
            body = await resp.json()
            assert body == {"status": "ok"}

    async def test_health_no_auth_required(self, _app) -> None:
        """Health endpoint succeeds without any authentication headers."""
        from aiohttp.test_utils import TestClient, TestServer

        async with TestClient(TestServer(_app)) as client:
            resp = await client.get("/-/health")
            assert resp.status == 200


# ── Route table ──────────────────────────────────────────────────────────


class TestRouteTable:
    """Verify provider-keyed route lookup."""

    @pytest.fixture()
    def routes(self, tmp_path: Path) -> _RouteTable:
        """Write a routes file and return a RouteTable."""
        routes_file = tmp_path / "routes.json"
        routes_file.write_text(
            json.dumps(
                {
                    "claude": {
                        "upstream": "https://api.anthropic.com",
                        "auth_header": "Authorization",
                    },
                    "gh": {"upstream": "https://api.github.com", "auth_header": "Authorization"},
                }
            )
        )
        return _RouteTable(str(routes_file))

    def test_get_known_provider(self, routes: _RouteTable) -> None:
        """Known provider returns route config."""
        route = routes.get("claude")
        assert route is not None
        assert route["upstream"] == "https://api.anthropic.com"

    def test_get_unknown_provider(self, routes: _RouteTable) -> None:
        """Unknown provider returns None."""
        assert routes.get("nonexistent") is None

    def test_rejects_invalid_path_upstreams(self, tmp_path: Path) -> None:
        """Path-prefix upstream overrides must be a ``/``-prefixed mapping."""
        routes_file = tmp_path / "bad-routes.json"
        routes_file.write_text(
            json.dumps(
                {"codex": {"upstream": "https://api.openai.com", "path_upstreams": {"x": ""}}}
            )
        )

        with pytest.raises(ValueError, match="path_upstreams"):
            _RouteTable(str(routes_file))

    def test_rejects_invalid_oauth_extra_headers(self, tmp_path: Path) -> None:
        """OAuth extra headers must be a string-to-string mapping."""
        routes_file = tmp_path / "bad-routes.json"
        routes_file.write_text(
            json.dumps(
                {
                    "claude": {
                        "upstream": "https://api.anthropic.com",
                        "oauth_extra_headers": {"anthropic-beta": 123},
                    }
                }
            )
        )

        with pytest.raises(ValueError, match="oauth_extra_headers"):
            _RouteTable(str(routes_file))


# ── Token extraction ─────────────────────────────────────────────────────


class TestExtractPhantomToken:
    """Verify phantom token extraction from request headers."""

    def _make_request(self, **headers: str) -> MagicMock:
        """Create a mock request with given headers."""
        req = MagicMock()
        req.headers = headers
        return req

    def test_bearer_token(self) -> None:
        """Extract token from 'Bearer <token>' Authorization header."""
        req = self._make_request(authorization="Bearer abc123")
        assert _extract_phantom_token(req) == "abc123"

    def test_bare_api_key(self) -> None:
        """Extract bare token from X-Api-Key header."""
        req = self._make_request(**{"x-api-key": "sk-phantom-xyz"})
        assert _extract_phantom_token(req) == "sk-phantom-xyz"

    def test_private_token(self) -> None:
        """Extract token from PRIVATE-TOKEN header (glab)."""
        req = self._make_request(**{"private-token": "glpat-abc"})
        assert _extract_phantom_token(req) == "glpat-abc"

    def test_no_auth_headers(self) -> None:
        """Return None when no auth headers present."""
        req = self._make_request(**{"content-type": "application/json"})
        assert _extract_phantom_token(req) is None

    def test_token_prefix_stripped(self) -> None:
        """Extract token from 'token <value>' format (gh style)."""
        req = self._make_request(authorization="token ghp_abc")
        assert _extract_phantom_token(req) == "ghp_abc"


# ── Token DB ─────────────────────────────────────────────────────────────


class TestTokenDB:
    """Verify the read-only DB accessor used by the token broker server."""

    @pytest.fixture()
    def token_db(self, tmp_path: Path):
        """Create a DB with test data and return a _TokenDB accessor."""
        db = CredentialDB(tmp_path / "test.db")
        db.store_credential("default", "claude", {"access_token": "sk-real-123"})
        token = db.create_token("proj", "task-1", "default", "claude")
        db.close()
        accessor = _TokenDB(str(tmp_path / "test.db"))
        accessor._test_token = token  # stash for test use
        yield accessor
        accessor.close()

    def test_lookup_valid_token(self, token_db: _TokenDB) -> None:
        """Valid token returns scope/task/credential_set/provider."""
        info = token_db.lookup_token(token_db._test_token)
        assert info["scope"] == "proj"
        assert info["credential_set"] == "default"
        assert info["provider"] == "claude"

    def test_lookup_invalid_token(self, token_db: _TokenDB) -> None:
        """Invalid token returns None."""
        assert token_db.lookup_token("nonexistent") is None

    def test_load_credential(self, token_db: _TokenDB) -> None:
        """Loads credential data for a provider."""
        cred = token_db.load_credential("default", "claude")
        assert cred["access_token"] == "sk-real-123"

    def test_load_missing_credential(self, token_db: _TokenDB) -> None:
        """Missing credential returns None."""
        assert token_db.load_credential("default", "nonexistent") is None


# ── Application construction ─────────────────────────────────────────────


class TestBuildApp:
    """Verify the aiohttp application is constructed correctly."""

    def test_app_has_required_keys(self, tmp_path: Path) -> None:
        """_build_app sets up routes, token_db, and lifecycle hooks."""
        db = CredentialDB(tmp_path / "test.db")
        db.close()
        routes_file = tmp_path / "routes.json"
        routes_file.write_text(json.dumps({"claude": {"upstream": "https://example.com"}}))

        app = _build_app(str(tmp_path / "test.db"), str(routes_file))

        assert isinstance(app[_KEY_ROUTES], _RouteTable)
        assert isinstance(app[_KEY_TOKEN_DB], _TokenDB)
        assert len(app.on_cleanup) > 0
        app[_KEY_TOKEN_DB].close()

    def test_route_validation_rejects_missing_upstream(self, tmp_path: Path) -> None:
        """Route config without 'upstream' field raises ValueError."""
        routes_file = tmp_path / "bad_routes.json"
        routes_file.write_text(json.dumps({"claude": {"auth_header": "Authorization"}}))

        with pytest.raises(ValueError, match="missing required 'upstream'"):
            _RouteTable(str(routes_file))


# ── Handler-level tests via aiohttp TestClient ───────────────────────────


@pytest.fixture()
def _broker_env(tmp_path: Path):
    """Set up a DB with per-provider tokens and routes, return (app, tokens)."""
    db = CredentialDB(tmp_path / "test.db")
    db.store_credential("default", "claude", {"access_token": "sk-real"})
    db.store_credential("default", "empty", {"type": "oauth"})  # no token fields
    db.store_credential("default", "orphan", {"access_token": "sk-orphan"})
    claude_token = db.create_token("proj", "t1", "default", "claude")
    empty_token = db.create_token("proj", "t1", "default", "empty")
    orphan_token = db.create_token("proj", "t1", "default", "orphan")
    db.close()

    routes = tmp_path / "routes.json"
    routes.write_text(
        json.dumps(
            {
                "claude": {
                    "upstream": "http://127.0.0.1:1",
                    "auth_header": "Authorization",
                    "auth_prefix": "Bearer ",
                },
                "empty": {"upstream": "http://127.0.0.1:1"},
                # "orphan" intentionally omitted — no route
            }
        )
    )

    app = _build_app(str(tmp_path / "test.db"), str(routes))
    return app, {
        "claude": claude_token,
        "empty": empty_token,
        "orphan": orphan_token,
    }


@pytest.mark.asyncio
class TestHandlerEdgeCases:
    """Exercise _handle_request edge cases via aiohttp TestClient."""

    async def test_missing_auth_401(self, _broker_env) -> None:
        """Request without auth header returns 401."""
        from aiohttp.test_utils import TestClient, TestServer

        app, _tokens = _broker_env
        async with TestClient(TestServer(app)) as client:
            resp = await client.get("/v1/messages")
            assert resp.status == 401

    async def test_missing_auth_logs_warning(self, _broker_env, caplog) -> None:
        """Missing auth header logs a WARNING with method and path."""
        import logging

        from aiohttp.test_utils import TestClient, TestServer

        app, _tokens = _broker_env
        with caplog.at_level(logging.WARNING, logger="terok-vault"):
            async with TestClient(TestServer(app)) as client:
                await client.get("/v1/messages")
        assert any("GET /v1/messages" in r.message and "401" in r.message for r in caplog.records)

    async def test_invalid_token_401(self, _broker_env) -> None:
        """Request with bad phantom token returns 401."""
        from aiohttp.test_utils import TestClient, TestServer

        app, _tokens = _broker_env
        async with TestClient(TestServer(app)) as client:
            resp = await client.get("/v1/messages", headers={"Authorization": "Bearer fake"})
            assert resp.status == 401

    async def test_invalid_token_logs_warning(self, _broker_env, caplog) -> None:
        """Invalid phantom token logs a WARNING with method and path."""
        import logging

        from aiohttp.test_utils import TestClient, TestServer

        app, _tokens = _broker_env
        with caplog.at_level(logging.WARNING, logger="terok-vault"):
            async with TestClient(TestServer(app)) as client:
                await client.get("/v1/messages", headers={"Authorization": "Bearer fake"})
        assert any("GET /v1/messages" in r.message and "401" in r.message for r in caplog.records)

    async def test_no_route_logs_warning(self, _broker_env, caplog) -> None:
        """Valid token with no matching route logs a WARNING."""
        import logging

        from aiohttp.test_utils import TestClient, TestServer

        app, tokens = _broker_env
        with caplog.at_level(logging.WARNING, logger="terok-vault"):
            async with TestClient(TestServer(app)) as client:
                resp = await client.get(
                    "/v1/x", headers={"Authorization": f"Bearer {tokens['orphan']}"}
                )
        assert resp.status == 404
        assert any("404" in r.message and "orphan" in r.message for r in caplog.records)

    async def test_empty_credential_returns_502(self, _broker_env) -> None:
        """Credential with no usable token field returns 502."""
        from aiohttp.test_utils import TestClient, TestServer

        app, tokens = _broker_env
        async with TestClient(TestServer(app)) as client:
            resp = await client.get(
                "/v1/x",
                headers={"Authorization": f"Bearer {tokens['empty']}"},
            )
            assert resp.status == 502
            assert "misconfigured" in (await resp.text()).lower()


@pytest.fixture()
def _static_marker_env(tmp_path: Path):
    """Set up DB with shared OAuth credentials and routes for static marker tests."""
    db = CredentialDB(tmp_path / "test.db")
    db.store_credential(
        "default", "claude", {"type": "oauth", "access_token": "sk-real-oauth-static"}
    )
    db.store_credential(
        "default", "codex", {"type": "oauth", "access_token": "sk-real-codex-static"}
    )
    db.close()

    routes = tmp_path / "routes.json"
    routes.write_text(
        json.dumps(
            {
                "claude": {"upstream": "http://127.0.0.1:1", "auth_header": "dynamic"},
                "codex": {"upstream": "http://127.0.0.1:1", "auth_header": "Authorization"},
            }
        )
    )
    return _build_app(str(tmp_path / "test.db"), str(routes))


@pytest.mark.asyncio
class TestStaticPhantomMarker:
    """Verify the static PHANTOM_CREDENTIALS_MARKER is accepted by the token broker."""

    async def test_static_marker_routes_to_claude(self, _static_marker_env) -> None:
        """Static marker resolves to Claude credential (returns 502 since upstream is down)."""
        from aiohttp.test_utils import TestClient, TestServer

        from terok_sandbox.vault.constants import PHANTOM_CREDENTIALS_MARKER

        app = _static_marker_env
        async with TestClient(TestServer(app)) as client:
            resp = await client.get(
                "/v1/messages",
                headers={"Authorization": f"Bearer {PHANTOM_CREDENTIALS_MARKER}"},
            )
            # 502 = upstream unreachable, which means auth succeeded and routing worked
            assert resp.status == 502

    async def test_codex_static_marker_routes_to_codex(self, _static_marker_env) -> None:
        """Codex shared auth.json marker resolves to the Codex credential."""
        from aiohttp.test_utils import TestClient, TestServer

        from terok_sandbox.vault.constants import CODEX_SHARED_OAUTH_MARKER

        app = _static_marker_env
        async with TestClient(TestServer(app)) as client:
            resp = await client.get(
                "/v1/responses",
                headers={"Authorization": f"Bearer {CODEX_SHARED_OAUTH_MARKER}"},
            )
            assert resp.status == 502

    async def test_static_marker_rejected_without_claude_credential(self, tmp_path: Path) -> None:
        """Static marker returns 502 (no credential) when Claude has no stored credentials."""
        from aiohttp.test_utils import TestClient, TestServer

        from terok_sandbox.vault.constants import PHANTOM_CREDENTIALS_MARKER

        db = CredentialDB(tmp_path / "test.db")
        db.close()  # empty DB

        routes = tmp_path / "routes.json"
        routes.write_text(json.dumps({"claude": {"upstream": "http://127.0.0.1:1"}}))
        app = _build_app(str(tmp_path / "test.db"), str(routes))

        async with TestClient(TestServer(app)) as client:
            resp = await client.get(
                "/v1/messages",
                headers={"Authorization": f"Bearer {PHANTOM_CREDENTIALS_MARKER}"},
            )
            assert resp.status == 502
            assert "not configured" in (await resp.text()).lower()

    async def test_random_token_still_rejected(self, _static_marker_env) -> None:
        """Non-marker, non-registered tokens are still rejected with 401."""
        from aiohttp.test_utils import TestClient, TestServer

        app = _static_marker_env
        async with TestClient(TestServer(app)) as client:
            resp = await client.get(
                "/v1/messages",
                headers={"Authorization": "Bearer not-a-valid-token"},
            )
            assert resp.status == 401


@pytest.fixture()
def _forwarding_env(tmp_path: Path):
    """Set up token broker + mock upstream to test the full forwarding path."""
    from aiohttp import web as _web

    # Mock upstream that echoes back auth and path
    async def _echo(request: _web.Request) -> _web.Response:
        return _web.json_response(
            {
                "auth": request.headers.get("Authorization", ""),
                "x_api_key": request.headers.get("x-api-key", ""),
                "private_token": request.headers.get("PRIVATE-TOKEN", ""),
                "path": request.path,
                "qs": request.query_string,
                "beta": request.headers.get("anthropic-beta", ""),
            }
        )

    upstream_app = _web.Application()
    upstream_app.router.add_route("*", "/{tail:.*}", _echo)

    db = CredentialDB(tmp_path / "test.db")
    db.store_credential("default", "claude", {"type": "oauth", "access_token": "sk-real-oauth"})
    db.store_credential("default", "vibe", {"type": "api_key", "key": "mistral-real-key"})
    db.store_credential("default", "glab", {"type": "pat", "token": "glpat-real"})
    claude_token = db.create_token("proj", "t1", "default", "claude")
    vibe_token = db.create_token("proj", "t1", "default", "vibe")
    glab_token = db.create_token("proj", "t1", "default", "glab")
    db.close()

    return upstream_app, tmp_path, {"claude": claude_token, "vibe": vibe_token, "glab": glab_token}


@pytest.mark.asyncio
class TestForwardingPath:
    """Exercise the full request forwarding with a mock upstream."""

    async def test_oauth_forwards_with_bearer_and_beta(self, _forwarding_env) -> None:
        """OAuth credential forwards as Bearer + anthropic-beta header."""
        from aiohttp.test_utils import TestClient, TestServer

        upstream_app, tmp_path, tokens = _forwarding_env
        upstream_server = TestServer(upstream_app)
        await upstream_server.start_server()

        routes = tmp_path / "routes.json"
        routes.write_text(
            json.dumps(
                {
                    "claude": {
                        "upstream": f"http://127.0.0.1:{upstream_server.port}",
                        "auth_header": "dynamic",
                        "oauth_extra_headers": {"anthropic-beta": "oauth-2025-04-20"},
                    },
                }
            )
        )

        broker_app = _build_app(str(tmp_path / "test.db"), str(routes))
        async with TestClient(TestServer(broker_app)) as client:
            resp = await client.post(
                "/v1/messages",
                headers={
                    "Authorization": f"Bearer {tokens['claude']}",
                    "anthropic-beta": "some-feature",
                },
                json={"model": "test"},
            )
            assert resp.status == 200
            body = await resp.json()
            assert body["auth"] == "Bearer sk-real-oauth"
            assert "oauth-2025-04-20" in body["beta"]
            assert "some-feature" in body["beta"]
            assert body["path"] == "/v1/messages"

        await upstream_server.close()

    async def test_codex_oauth_does_not_inherit_anthropic_beta(self, _forwarding_env) -> None:
        """OAuth extra headers are route-scoped; Codex does not get Claude's beta header."""
        from aiohttp.test_utils import TestClient, TestServer

        upstream_app, tmp_path, _tokens = _forwarding_env
        upstream_server = TestServer(upstream_app)
        await upstream_server.start_server()

        db = CredentialDB(tmp_path / "test.db")
        db.store_credential("default", "codex", {"type": "oauth", "access_token": "sk-codex"})
        codex_token = db.create_token("proj", "t1", "default", "codex")
        db.close()

        routes = tmp_path / "routes.json"
        routes.write_text(
            json.dumps({"codex": {"upstream": f"http://127.0.0.1:{upstream_server.port}"}})
        )

        broker_app = _build_app(str(tmp_path / "test.db"), str(routes))
        async with TestClient(TestServer(broker_app)) as client:
            resp = await client.post(
                "/v1/responses",
                headers={"Authorization": f"Bearer {codex_token}"},
                json={"model": "test"},
            )
            assert resp.status == 200
            body = await resp.json()
            assert body["auth"] == "Bearer sk-codex"
            assert body["beta"] == ""

        await upstream_server.close()

    async def test_api_key_forwards_correctly(self, _forwarding_env) -> None:
        """API key credential uses route-configured auth header."""
        from aiohttp.test_utils import TestClient, TestServer

        upstream_app, tmp_path, tokens = _forwarding_env
        upstream_server = TestServer(upstream_app)
        await upstream_server.start_server()

        routes = tmp_path / "routes.json"
        routes.write_text(
            json.dumps(
                {
                    "vibe": {
                        "upstream": f"http://127.0.0.1:{upstream_server.port}",
                        "auth_header": "Authorization",
                        "auth_prefix": "Bearer ",
                    },
                }
            )
        )

        broker_app = _build_app(str(tmp_path / "test.db"), str(routes))
        async with TestClient(TestServer(broker_app)) as client:
            resp = await client.post(
                "/v1/chat/completions",
                headers={"Authorization": f"Bearer {tokens['vibe']}"},
                json={"model": "test"},
            )
            assert resp.status == 200
            body = await resp.json()
            assert body["auth"] == "Bearer mistral-real-key"

        await upstream_server.close()

    async def test_pat_forwards_with_private_token(self, _forwarding_env) -> None:
        """PAT credential uses PRIVATE-TOKEN header."""
        from aiohttp.test_utils import TestClient, TestServer

        upstream_app, tmp_path, tokens = _forwarding_env
        upstream_server = TestServer(upstream_app)
        await upstream_server.start_server()

        routes = tmp_path / "routes.json"
        routes.write_text(
            json.dumps(
                {
                    "glab": {
                        "upstream": f"http://127.0.0.1:{upstream_server.port}",
                        "auth_header": "PRIVATE-TOKEN",
                        "auth_prefix": "",
                    },
                }
            )
        )

        broker_app = _build_app(str(tmp_path / "test.db"), str(routes))
        async with TestClient(TestServer(broker_app)) as client:
            resp = await client.get(
                "/api/v4/projects",
                headers={"PRIVATE-TOKEN": tokens["glab"]},
            )
            assert resp.status == 200
            body = await resp.json()
            assert body["private_token"] == "glpat-real"
            assert body["path"] == "/api/v4/projects"

        await upstream_server.close()

    async def test_path_upstreams_route_backend_api_to_chatgpt(self, tmp_path: Path) -> None:
        """Codex ``/backend-api`` traffic can target a distinct upstream host."""
        from aiohttp import web as _web
        from aiohttp.test_utils import TestClient, TestServer

        async def _api_echo(request: _web.Request) -> _web.Response:
            return _web.json_response({"service": "api", "path": request.path})

        async def _chatgpt_echo(request: _web.Request) -> _web.Response:
            return _web.json_response({"service": "chatgpt", "path": request.path})

        api_app = _web.Application()
        api_app.router.add_route("*", "/{tail:.*}", _api_echo)
        chatgpt_app = _web.Application()
        chatgpt_app.router.add_route("*", "/{tail:.*}", _chatgpt_echo)

        api_server = TestServer(api_app)
        chatgpt_server = TestServer(chatgpt_app)
        await api_server.start_server()
        await chatgpt_server.start_server()

        db = CredentialDB(tmp_path / "test.db")
        db.store_credential("default", "codex", {"type": "oauth", "access_token": "sk-real-codex"})
        codex_token = db.create_token("proj", "t1", "default", "codex")
        db.close()

        routes = tmp_path / "routes.json"
        routes.write_text(
            json.dumps(
                {
                    "codex": {
                        "upstream": f"http://127.0.0.1:{api_server.port}",
                        "path_upstreams": {
                            "/backend-api/": f"http://127.0.0.1:{chatgpt_server.port}"
                        },
                    }
                }
            )
        )

        broker_app = _build_app(str(tmp_path / "test.db"), str(routes))
        async with TestClient(TestServer(broker_app)) as client:
            api_resp = await client.get(
                "/v1/responses",
                headers={"Authorization": f"Bearer {codex_token}"},
            )
            backend_resp = await client.get(
                "/backend-api/me",
                headers={"Authorization": f"Bearer {codex_token}"},
            )
            assert api_resp.status == 200
            assert backend_resp.status == 200
            assert (await api_resp.json())["service"] == "api"
            assert (await backend_resp.json())["service"] == "chatgpt"

        await api_server.close()
        await chatgpt_server.close()

    async def test_websocket_proxy_injects_real_bearer(self, tmp_path: Path) -> None:
        """Realtime websocket traffic proxies through the broker with real auth."""
        from aiohttp import web as _web
        from aiohttp.test_utils import TestClient, TestServer

        async def _ws_echo(request: _web.Request) -> _web.WebSocketResponse:
            ws = _web.WebSocketResponse()
            await ws.prepare(request)
            await ws.send_json(
                {
                    "auth": request.headers.get("Authorization", ""),
                    "path": request.path,
                }
            )
            msg = await ws.receive()
            await ws.send_str(f"echo:{msg.data}")
            await ws.close()
            return ws

        upstream_app = _web.Application()
        upstream_app.router.add_route("*", "/{tail:.*}", _ws_echo)
        upstream_server = TestServer(upstream_app)
        await upstream_server.start_server()

        db = CredentialDB(tmp_path / "test.db")
        db.store_credential(
            "default",
            "codex",
            {"type": "oauth", "access_token": "sk-real-codex-ws"},
        )
        codex_token = db.create_token("proj", "t1", "default", "codex")
        db.close()

        routes = tmp_path / "routes.json"
        routes.write_text(
            json.dumps({"codex": {"upstream": f"http://127.0.0.1:{upstream_server.port}"}})
        )

        broker_app = _build_app(str(tmp_path / "test.db"), str(routes))
        async with TestClient(TestServer(broker_app)) as client:
            async with client.ws_connect(
                "/v1/realtime",
                headers={"Authorization": f"Bearer {codex_token}"},
            ) as ws:
                first = await ws.receive_json()
                assert first["auth"] == "Bearer sk-real-codex-ws"
                assert first["path"] == "/v1/realtime"
                await ws.send_str("hello")
                assert await ws.receive_str() == "echo:hello"

        await upstream_server.close()

    async def test_query_string_preserved(self, _forwarding_env) -> None:
        """Query string is preserved in the upstream request."""
        from aiohttp.test_utils import TestClient, TestServer

        upstream_app, tmp_path, tokens = _forwarding_env
        upstream_server = TestServer(upstream_app)
        await upstream_server.start_server()

        routes = tmp_path / "routes.json"
        routes.write_text(
            json.dumps({"claude": {"upstream": f"http://127.0.0.1:{upstream_server.port}"}})
        )

        broker_app = _build_app(str(tmp_path / "test.db"), str(routes))
        async with TestClient(TestServer(broker_app)) as client:
            resp = await client.get(
                "/v1/check?foo=bar&baz=1",
                headers={"Authorization": f"Bearer {tokens['claude']}"},
            )
            assert resp.status == 200
            body = await resp.json()
            assert "foo=bar" in body["qs"]

        await upstream_server.close()

    async def test_upstream_error_returns_502(self, _forwarding_env) -> None:
        """Connection refused to upstream returns generic 502."""
        from aiohttp.test_utils import TestClient, TestServer

        _, tmp_path, tokens = _forwarding_env

        routes = tmp_path / "routes.json"
        routes.write_text(
            json.dumps(
                {"claude": {"upstream": "http://127.0.0.1:1"}}  # port 1 = refused
            )
        )

        broker_app = _build_app(str(tmp_path / "test.db"), str(routes))
        async with TestClient(TestServer(broker_app)) as client:
            resp = await client.get(
                "/v1/messages",
                headers={"Authorization": f"Bearer {tokens['claude']}"},
            )
            assert resp.status == 502
            text = await resp.text()
            assert "127.0.0.1" not in text


# ── TokenDB refresh methods ──────────────────────────────────────────────


class TestTokenDBRefresh:
    """Verify list_refreshable() and update_credential() on _TokenDB."""

    @pytest.fixture()
    def db_with_creds(self, tmp_path: Path):
        """Create a DB with mixed credential types."""
        db = CredentialDB(tmp_path / "test.db")
        db.store_credential(
            "default",
            "claude",
            {
                "type": "oauth",
                "access_token": "sk-old",
                "refresh_token": "rt-abc",
                "expires_at": 1000,
            },
        )
        db.store_credential(
            "default",
            "vibe",
            {
                "type": "api_key",
                "key": "mistral-key",
            },
        )
        db.store_credential(
            "default",
            "codex",
            {
                "type": "oauth",
                "access_token": "sk-codex",
                # no refresh_token
            },
        )
        db.close()
        accessor = _TokenDB(str(tmp_path / "test.db"))
        yield accessor
        accessor.close()

    def test_list_refreshable_returns_only_oauth_with_refresh(self, db_with_creds) -> None:
        """Only OAuth credentials with refresh_token are returned."""
        result = db_with_creds.list_refreshable()
        assert len(result) == 1
        cs, prov, data = result[0]
        assert cs == "default"
        assert prov == "claude"
        assert data["refresh_token"] == "rt-abc"

    def test_update_credential_persists(self, db_with_creds) -> None:
        """update_credential writes new data that load_credential can read back."""
        db_with_creds.update_credential(
            "default",
            "claude",
            {
                "type": "oauth",
                "access_token": "sk-new",
                "refresh_token": "rt-new",
                "expires_at": 9999,
            },
        )
        cred = db_with_creds.load_credential("default", "claude")
        assert cred["access_token"] == "sk-new"
        assert cred["refresh_token"] == "rt-new"


# ── OAuth refresh logic ──────────────────────────────────────────────────


@pytest.mark.asyncio
class TestDoOAuthRefresh:
    """Exercise _do_oauth_refresh against a mock token endpoint."""

    async def test_successful_refresh(self) -> None:
        """A 200 response updates access_token, refresh_token, and expires_at."""
        import time

        from aiohttp import web as _web
        from aiohttp.test_utils import TestServer

        async def _token_handler(request: _web.Request) -> _web.Response:
            body = await request.json()
            assert body["grant_type"] == "refresh_token"
            assert body["refresh_token"] == "rt-old"
            assert body["client_id"] == "test-client"
            return _web.json_response(
                {
                    "access_token": "sk-fresh",
                    "refresh_token": "rt-rotated",
                    "expires_in": 3600,
                }
            )

        token_app = _web.Application()
        token_app.router.add_post("/v1/oauth/token", _token_handler)
        server = TestServer(token_app)
        await server.start_server()

        from aiohttp import ClientSession

        async with ClientSession() as session:
            oauth_cfg = {
                "token_url": f"http://127.0.0.1:{server.port}/v1/oauth/token",
                "client_id": "test-client",
                "scope": "user:inference",
            }
            cred = {"type": "oauth", "access_token": "sk-old", "refresh_token": "rt-old"}
            result = await _do_oauth_refresh(session, "claude", oauth_cfg, cred)

        assert result["access_token"] == "sk-fresh"
        assert result["refresh_token"] == "rt-rotated"
        assert result["expires_at"] > time.time()
        await server.close()

    async def test_refresh_failure_raises(self) -> None:
        """Non-200 response raises RuntimeError."""
        from aiohttp import web as _web
        from aiohttp.test_utils import TestServer

        async def _fail_handler(_request: _web.Request) -> _web.Response:
            return _web.Response(status=400, text="bad_request")

        token_app = _web.Application()
        token_app.router.add_post("/v1/oauth/token", _fail_handler)
        server = TestServer(token_app)
        await server.start_server()

        from aiohttp import ClientSession

        async with ClientSession() as session:
            oauth_cfg = {
                "token_url": f"http://127.0.0.1:{server.port}/v1/oauth/token",
                "client_id": "test-client",
            }
            cred = {"type": "oauth", "access_token": "sk-old", "refresh_token": "rt-old"}
            with pytest.raises(RuntimeError, match="Token refresh failed"):
                await _do_oauth_refresh(session, "claude", oauth_cfg, cred)

        await server.close()

    async def test_missing_rotated_token_keeps_old(self) -> None:
        """If response omits refresh_token, the old one is preserved."""
        from aiohttp import web as _web
        from aiohttp.test_utils import TestServer

        async def _no_rotate_handler(_request: _web.Request) -> _web.Response:
            return _web.json_response({"access_token": "sk-new", "expires_in": 7200})

        token_app = _web.Application()
        token_app.router.add_post("/v1/oauth/token", _no_rotate_handler)
        server = TestServer(token_app)
        await server.start_server()

        from aiohttp import ClientSession

        async with ClientSession() as session:
            oauth_cfg = {
                "token_url": f"http://127.0.0.1:{server.port}/v1/oauth/token",
                "client_id": "c",
            }
            cred = {"type": "oauth", "access_token": "sk-old", "refresh_token": "rt-keep"}
            result = await _do_oauth_refresh(session, "claude", oauth_cfg, cred)

        assert result["refresh_token"] == "rt-keep"
        await server.close()


@pytest.mark.asyncio
class TestRefreshAll:
    """Exercise _refresh_all with real DB and mock token endpoint."""

    async def test_refreshes_expired_skips_valid(self, tmp_path: Path) -> None:
        """Only expired/expiring credentials are refreshed."""
        import time

        from aiohttp import web as _web
        from aiohttp.test_utils import TestServer

        refreshed_providers: list[str] = []

        async def _token_handler(request: _web.Request) -> _web.Response:
            body = await request.json()
            refreshed_providers.append(body.get("client_id", ""))
            return _web.json_response(
                {
                    "access_token": "sk-refreshed",
                    "refresh_token": "rt-new",
                    "expires_in": 3600,
                }
            )

        token_app = _web.Application()
        token_app.router.add_post("/v1/oauth/token", _token_handler)
        server = TestServer(token_app)
        await server.start_server()

        token_url = f"http://127.0.0.1:{server.port}/v1/oauth/token"

        # DB: claude expired, codex still valid
        db = CredentialDB(tmp_path / "test.db")
        db.store_credential(
            "default",
            "claude",
            {
                "type": "oauth",
                "access_token": "sk-expired",
                "refresh_token": "rt-c",
                "expires_at": 1000,
            },
        )
        db.store_credential(
            "default",
            "codex",
            {
                "type": "oauth",
                "access_token": "sk-valid",
                "refresh_token": "rt-x",
                "expires_at": time.time() + 7200,
            },
        )
        db.close()

        routes_file = tmp_path / "routes.json"
        routes_file.write_text(
            json.dumps(
                {
                    "claude": {
                        "upstream": "https://api.anthropic.com",
                        "oauth_refresh": {"token_url": token_url, "client_id": "claude-id"},
                    },
                    "codex": {
                        "upstream": "https://api.openai.com",
                        "oauth_refresh": {"token_url": token_url, "client_id": "codex-id"},
                    },
                }
            )
        )

        app = _build_app(str(tmp_path / "test.db"), str(routes_file))
        from aiohttp import ClientSession

        app[_KEY_CLIENT] = ClientSession()
        try:
            await _refresh_all(app)
        finally:
            await app[_KEY_CLIENT].close()

        # Only claude (expired) should have been refreshed
        assert refreshed_providers == ["claude-id"]

        # Verify DB was updated
        accessor = _TokenDB(str(tmp_path / "test.db"))
        cred = accessor.load_credential("default", "claude")
        assert cred["access_token"] == "sk-refreshed"
        accessor.close()

        await server.close()

    async def test_skips_provider_without_oauth_refresh(self, tmp_path: Path) -> None:
        """Providers without oauth_refresh in routes are skipped."""
        db = CredentialDB(tmp_path / "test.db")
        db.store_credential(
            "default",
            "gh",
            {
                "type": "oauth",
                "access_token": "ghp-old",
                "refresh_token": "rt-gh",
                "expires_at": 1000,
            },
        )
        db.close()

        routes_file = tmp_path / "routes.json"
        routes_file.write_text(
            json.dumps(
                {
                    "gh": {"upstream": "https://api.github.com"},
                }
            )
        )

        app = _build_app(str(tmp_path / "test.db"), str(routes_file))
        from aiohttp import ClientSession

        app[_KEY_CLIENT] = ClientSession()
        try:
            await _refresh_all(app)  # should not raise
        finally:
            await app[_KEY_CLIENT].close()

        # Credential unchanged
        accessor = _TokenDB(str(tmp_path / "test.db"))
        assert accessor.load_credential("default", "gh")["access_token"] == "ghp-old"
        accessor.close()


# ── ServerDisconnectedError retry paths ─────────────────────────────────


@pytest.mark.asyncio
class TestServerDisconnectRetry:
    """Verify ServerDisconnectedError retry logic and the resulting 502 path."""

    @pytest.fixture()
    def _app_and_token(self, tmp_path: Path):
        """Minimal broker app with a single claude credential; upstream URL is irrelevant (session mocked)."""
        db = CredentialDB(tmp_path / "test.db")
        db.store_credential("default", "claude", {"type": "oauth", "access_token": "sk-real"})
        token = db.create_token("proj", "t1", "default", "claude")
        db.close()

        routes = tmp_path / "routes.json"
        routes.write_text(
            json.dumps(
                {"claude": {"upstream": "http://127.0.0.1:1", "auth_header": "Authorization"}}
            )
        )
        return _build_app(str(tmp_path / "test.db"), str(routes)), token

    async def test_retries_and_succeeds_after_first_disconnect(self, _app_and_token) -> None:
        """ServerDisconnectedError on first attempt triggers a retry; second attempt succeeds."""
        from aiohttp import ServerDisconnectedError
        from aiohttp.test_utils import TestClient, TestServer

        app, token = _app_and_token

        class _Content:
            async def iter_any(self):
                yield b'{"ok": true}'

        class _OkCM:
            status = 200
            headers: dict = {}
            content = _Content()

            async def __aenter__(self):
                return self

            async def __aexit__(self, *args: object) -> None:
                pass

        class _DisconnectCM:
            async def __aenter__(self):
                raise ServerDisconnectedError()

            async def __aexit__(self, *args: object) -> None:
                pass

        call_count = 0

        class _MockSession:
            def request(self, *_args: object, **_kwargs: object) -> object:
                nonlocal call_count
                call_count += 1
                return _DisconnectCM() if call_count == 1 else _OkCM()

            async def close(self) -> None:
                pass

        async with TestClient(TestServer(app)) as client:
            app[_KEY_CLIENT] = _MockSession()
            resp = await client.get("/v1/messages", headers={"Authorization": f"Bearer {token}"})
            assert resp.status == 200

        assert call_count == 2  # confirms retry occurred

    async def test_both_attempts_disconnect_returns_502(self, _app_and_token) -> None:
        """ServerDisconnectedError on both attempts returns 502 with 'Upstream disconnected'."""
        from aiohttp import ServerDisconnectedError
        from aiohttp.test_utils import TestClient, TestServer

        app, token = _app_and_token

        class _DisconnectCM:
            async def __aenter__(self):
                raise ServerDisconnectedError()

            async def __aexit__(self, *args: object) -> None:
                pass

        class _MockSession:
            def request(self, *_args: object, **_kwargs: object) -> _DisconnectCM:
                return _DisconnectCM()

            async def close(self) -> None:
                pass

        async with TestClient(TestServer(app)) as client:
            app[_KEY_CLIENT] = _MockSession()
            resp = await client.get("/v1/messages", headers={"Authorization": f"Bearer {token}"})
            assert resp.status == 502
            assert await resp.text() == "Upstream disconnected"


# ── _run_multi site selection ────────────────────────────────────────────


class TestRunMultiSiteSelection:
    """Verify _run_multi uses inherited sockets or creates its own listeners."""

    @staticmethod
    def _make_app(tmp_path: Path) -> web.Application:
        """Build a minimal app with required keys."""
        routes_file = tmp_path / "routes.json"
        routes_file.write_text("{}")
        db = CredentialDB(tmp_path / "creds.db")
        db.close()
        return _build_app(
            routes_path=str(routes_file),
            db_path=str(tmp_path / "creds.db"),
        )

    @pytest.mark.asyncio
    async def test_systemd_inherited_sockets(self, tmp_path: Path) -> None:
        """When _systemd_sockets returns both FDs, SockSite is used for each."""
        import asyncio
        import socket as _socket
        from unittest.mock import patch

        app = self._make_app(tmp_path)
        mock_unix = MagicMock(spec=_socket.socket)
        mock_tcp = MagicMock(spec=_socket.socket)
        sock_sites: list = []

        class _TrackingSockSite:
            def __init__(self, runner, sock, **kw):
                sock_sites.append(sock)

            async def start(self):
                pass

        with (
            patch(
                "terok_sandbox.vault.token_broker._systemd_sockets",
                return_value=(mock_unix, mock_tcp),
            ),
            patch("aiohttp.web_runner.SockSite", _TrackingSockSite),
        ):
            task = asyncio.create_task(
                _run_multi(app, sock_path=str(tmp_path / "vault.sock"), port=18731)
            )
            await asyncio.sleep(0.05)
            task.cancel()
            with pytest.raises(asyncio.CancelledError):
                await task

        assert mock_unix in sock_sites
        assert mock_tcp in sock_sites

    @pytest.mark.asyncio
    async def test_daemon_mode_creates_own_listeners(self, tmp_path: Path) -> None:
        """When _systemd_sockets returns (None, None), SockSite (Unix) and TCPSite are used."""
        import asyncio
        import socket
        from unittest.mock import patch

        app = self._make_app(tmp_path)
        used_sites: list[str] = []

        class _TrackingSockSite:
            def __init__(self, runner, sock, **kw):
                kind = "unix" if sock.family == socket.AF_UNIX else "sock"
                used_sites.append(f"{kind}:{sock.getsockname()}")
                sock.close()  # prevent leak — we don't actually serve here

            async def start(self):
                pass

        class _TrackingTCPSite:
            def __init__(self, runner, host, port, **kw):
                used_sites.append(f"tcp:{host}:{port}")

            async def start(self):
                pass

        with (
            patch(
                "terok_sandbox.vault.token_broker._systemd_sockets",
                return_value=(None, None),
            ),
            patch("aiohttp.web_runner.SockSite", _TrackingSockSite),
            patch("aiohttp.web_runner.TCPSite", _TrackingTCPSite),
        ):
            task = asyncio.create_task(
                _run_multi(app, sock_path=str(tmp_path / "vault.sock"), port=18731)
            )
            await asyncio.sleep(0.05)
            task.cancel()
            with pytest.raises(asyncio.CancelledError):
                await task

        assert any(s.startswith("unix:") for s in used_sites)
        assert any("tcp:127.0.0.1:18731" in s for s in used_sites)
