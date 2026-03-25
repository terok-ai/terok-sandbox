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
        """Unknown prefix returns None."""
        prefix, rest, _ = routes.resolve("/unknown/v1/chat")
        assert prefix is None

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
    def token_db(self, tmp_path: Path) -> _TokenDB:
        """Create a DB with test data and return a _TokenDB accessor."""
        db = CredentialDB(tmp_path / "test.db")
        db.store_credential("default", "claude", {"access_token": "sk-real-123"})
        token = db.create_proxy_token("proj", "task-1", "default")
        db.close()
        accessor = _TokenDB(str(tmp_path / "test.db"))
        accessor._test_token = token  # stash for test use
        return accessor

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

        assert isinstance(app["routes"], _RouteTable)
        assert isinstance(app["token_db"], _TokenDB)
        # Cleanup hook registered
        assert len(app.on_cleanup) > 0
