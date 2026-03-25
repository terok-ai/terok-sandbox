# SPDX-FileCopyrightText: 2026 Jiri Vyskocil
# SPDX-License-Identifier: Apache-2.0

"""Tests for the credential proxy server — routing, auth, and forwarding."""

from __future__ import annotations

import json
from pathlib import Path
from unittest.mock import MagicMock

import pytest

from terok_sandbox.credential_db import CredentialDB
from terok_sandbox.credential_proxy.server import (
    _KEY_ROUTES,
    _KEY_TOKEN_DB,
    _build_app,
    _extract_phantom_token,
    _RouteTable,
    _TokenDB,
)

# ── Route resolution ─────────────────────────────────────────────────────


class TestRouteTable:
    """Verify path-prefix routing."""

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

    def test_resolve_known_prefix(self, routes: _RouteTable) -> None:
        """Known prefix extracts provider + rest."""
        prefix, rest, route = routes.resolve("/claude/v1/messages")
        assert prefix == "claude"
        assert rest == "/v1/messages"
        assert route["upstream"] == "https://api.anthropic.com"

    def test_resolve_unknown_prefix(self, routes: _RouteTable) -> None:
        """Unknown prefix returns None and passes through the original path."""
        prefix, rest, _ = routes.resolve("/unknown/v1/chat")
        assert prefix is None
        assert rest == "/unknown/v1/chat"

    def test_resolve_root_only(self, routes: _RouteTable) -> None:
        """Prefix with no rest gives '/' as rest."""
        prefix, rest, _ = routes.resolve("/claude")
        assert prefix == "claude"
        assert rest == "/"


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
    """Verify the read-only DB accessor used by the proxy server."""

    @pytest.fixture()
    def token_db(self, tmp_path: Path):
        """Create a DB with test data and return a _TokenDB accessor."""
        db = CredentialDB(tmp_path / "test.db")
        db.store_credential("default", "claude", {"access_token": "sk-real-123"})
        token = db.create_proxy_token("proj", "task-1", "default")
        db.close()
        accessor = _TokenDB(str(tmp_path / "test.db"))
        accessor._test_token = token  # stash for test use
        yield accessor
        accessor.close()

    def test_lookup_valid_token(self, token_db: _TokenDB) -> None:
        """Valid token returns project/task/credential_set."""
        info = token_db.lookup_token(token_db._test_token)
        assert info["project"] == "proj"
        assert info["credential_set"] == "default"

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
def _proxy_env(tmp_path: Path):
    """Set up a DB with credentials + routes, return (app, token)."""
    db = CredentialDB(tmp_path / "test.db")
    db.store_credential("default", "claude", {"access_token": "sk-real"})
    db.store_credential("default", "empty", {"type": "oauth"})  # no token fields
    token = db.create_proxy_token("proj", "t1", "default")
    db.close()

    routes = tmp_path / "routes.json"
    # Use a placeholder upstream — tests that hit the handler won't
    # reach it because we're testing auth/routing failures only.
    routes.write_text(
        json.dumps(
            {
                "claude": {
                    "upstream": "http://127.0.0.1:1",
                    "auth_header": "Authorization",
                    "auth_prefix": "Bearer ",
                },
                "empty": {"upstream": "http://127.0.0.1:1"},
            }
        )
    )

    app = _build_app(str(tmp_path / "test.db"), str(routes))
    return app, token


@pytest.mark.asyncio
class TestHandlerEdgeCases:
    """Exercise _handle_request edge cases via aiohttp TestClient."""

    async def test_unknown_route_404(self, _proxy_env) -> None:
        """Request to unknown prefix returns 404."""
        from aiohttp.test_utils import TestClient, TestServer

        app, token = _proxy_env
        async with TestClient(TestServer(app)) as client:
            resp = await client.get("/bogus/v1/x", headers={"Authorization": f"Bearer {token}"})
            assert resp.status == 404

    async def test_missing_auth_401(self, _proxy_env) -> None:
        """Request without auth header returns 401."""
        from aiohttp.test_utils import TestClient, TestServer

        app, _token = _proxy_env
        async with TestClient(TestServer(app)) as client:
            resp = await client.get("/claude/v1/messages")
            assert resp.status == 401

    async def test_invalid_token_401(self, _proxy_env) -> None:
        """Request with bad phantom token returns 401."""
        from aiohttp.test_utils import TestClient, TestServer

        app, _token = _proxy_env
        async with TestClient(TestServer(app)) as client:
            resp = await client.get("/claude/v1/messages", headers={"Authorization": "Bearer fake"})
            assert resp.status == 401

    async def test_empty_credential_returns_502(self, _proxy_env) -> None:
        """Credential with no usable token field returns 502."""
        from aiohttp.test_utils import TestClient, TestServer

        app, token = _proxy_env
        async with TestClient(TestServer(app)) as client:
            resp = await client.get("/empty/v1/x", headers={"Authorization": f"Bearer {token}"})
            assert resp.status == 502
            assert "misconfigured" in (await resp.text()).lower()
