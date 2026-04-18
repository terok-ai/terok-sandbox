# SPDX-FileCopyrightText: 2026 Jiri Vyskocil
# SPDX-License-Identifier: Apache-2.0

"""End-to-end story tests for the vault token broker.

Each test exercises a full user journey against a real aiohttp server
backed by real sqlite — no mocks.  A mock upstream (also aiohttp) stands
in for the real API servers.

Marker: ``needs_vault`` — use to select/skip in CI.
"""

from __future__ import annotations

import json
from pathlib import Path

import aiohttp
import pytest
from aiohttp import web
from aiohttp.test_utils import TestServer

from terok_sandbox.credentials.db import CredentialDB
from terok_sandbox.vault.token_broker import _build_app

pytestmark = pytest.mark.needs_vault


# ---------------------------------------------------------------------------
# Mock upstream API server
# ---------------------------------------------------------------------------


def _make_upstream_app() -> web.Application:
    """Create a trivial upstream that echoes back auth headers as JSON."""

    async def echo_auth(request: web.Request) -> web.Response:
        return web.json_response(
            {
                "path": request.path,
                "method": request.method,
                "auth": request.headers.get("Authorization", ""),
                "x_api_key": request.headers.get("X-Api-Key", ""),
            }
        )

    async def stream_sse(request: web.Request) -> web.StreamResponse:
        """Simulate an SSE endpoint (like /v1/messages with streaming)."""
        resp = web.StreamResponse(
            status=200,
            headers={"Content-Type": "text/event-stream"},
        )
        await resp.prepare(request)
        for i in range(3):
            await resp.write(f"data: chunk-{i}\n\n".encode())
        await resp.write_eof()
        return resp

    app = web.Application()
    app.router.add_route("*", "/v1/messages", echo_auth)
    app.router.add_route("*", "/v1/chat/completions", echo_auth)
    app.router.add_get("/v1/stream", stream_sse)
    app.router.add_route("*", "/{tail:.*}", echo_auth)
    return app


# ---------------------------------------------------------------------------
# Fixture: proxy + upstream wired together
# ---------------------------------------------------------------------------


@pytest.fixture()
async def proxy_env(tmp_path: Path, populated_db: CredentialDB):
    """Start a mock upstream and a real broker, yield (proxy_url, db, upstream_url).

    The broker routes are rewritten to point at the mock upstream instead of
    the real API servers, so the full forwarding path is exercised without
    touching the internet.
    """
    # 1. Start mock upstream
    upstream_app = _make_upstream_app()
    upstream_server = TestServer(upstream_app)
    await upstream_server.start_server()
    upstream_base = f"http://127.0.0.1:{upstream_server.port}"

    # 2. Write routes pointing to mock upstream
    routes = {
        "claude": {
            "upstream": upstream_base,
            "auth_header": "Authorization",
            "auth_prefix": "Bearer ",
        },
        "codex": {
            "upstream": upstream_base,
            "auth_header": "Authorization",
            "auth_prefix": "Bearer ",
        },
        "gh": {
            "upstream": upstream_base,
            "auth_header": "Authorization",
            "auth_prefix": "token ",
        },
    }
    routes_file = tmp_path / "routes.json"
    routes_file.write_text(json.dumps(routes), encoding="utf-8")

    # 3. Build and start proxy app
    db_path = str(tmp_path / "proxy" / "credentials.db")
    proxy_app = _build_app(db_path, str(routes_file))
    proxy_server = TestServer(proxy_app)
    await proxy_server.start_server()

    # 4. Create per-provider tokens (broker routes by token's provider field)
    claude_token = populated_db.create_token("test-project", "task-1", "default", "claude")
    gh_token = populated_db.create_token("test-project", "task-1", "default", "gh")
    no_route_token = populated_db.create_token(
        "test-project", "task-1", "default", "unknown-provider"
    )

    yield {
        "proxy_url": f"http://127.0.0.1:{proxy_server.port}",
        "upstream_url": upstream_base,
        "db": populated_db,
        "token": claude_token,
        "gh_token": gh_token,
        "no_route_token": no_route_token,
    }

    await proxy_server.close()
    await upstream_server.close()


# ---------------------------------------------------------------------------
# Stories
# ---------------------------------------------------------------------------


class TestStoreAndRetrieveCredentials:
    """Story: operator stores credentials, verifies they're queryable."""

    def test_store_claude_oauth_and_list(self, db: CredentialDB) -> None:
        """Operator stores Claude OAuth tokens and lists available providers."""
        db.store_credential(
            "default",
            "claude",
            {
                "type": "oauth",
                "access_token": "sk-ant-test",
                "refresh_token": "rt-test",
            },
        )
        db.store_credential(
            "default",
            "gh",
            {
                "type": "oauth_token",
                "token": "ghp_test",
            },
        )

        providers = db.list_credentials("default")
        assert "claude" in providers
        assert "gh" in providers

    def test_per_scope_credential_isolation(self, db: CredentialDB) -> None:
        """Different scopes have independent credential sets."""
        db.store_credential("default", "claude", {"access_token": "default-key"})
        db.store_credential("work", "claude", {"access_token": "work-key"})

        assert db.load_credential("default", "claude")["access_token"] == "default-key"
        assert db.load_credential("work", "claude")["access_token"] == "work-key"


class TestTokenLifecycle:
    """Story: task creates token, uses it, task ends and token is revoked."""

    def test_full_token_lifecycle(self, db: CredentialDB) -> None:
        """Token is created, resolvable, then revoked on task end."""
        # Task starts -> token created
        token = db.create_token("myproject", "task-42", "default", "claude")
        assert token.startswith("terok-p-")
        assert len(token) == 8 + 32  # prefix + hex

        # Token resolves to the right credential set and provider
        info = db.lookup_token(token)
        assert info["scope"] == "myproject"
        assert info["credential_set"] == "default"
        assert info["provider"] == "claude"

        # Task ends → tokens revoked
        db.revoke_tokens("myproject", "task-42")
        assert db.lookup_token(token) is None

    def test_revoked_token_rejected(self, populated_db: CredentialDB) -> None:
        """After revocation the token is no longer valid."""
        token = populated_db.create_token("proj", "task-1", "default", "claude")
        populated_db.revoke_tokens("proj", "task-1")
        assert populated_db.lookup_token(token) is None


@pytest.mark.asyncio
class TestBrokerForwardsWithRealAuth:
    """Story: container sends request with token → broker injects real auth."""

    async def test_claude_api_call(self, proxy_env: dict) -> None:
        """Claude API call through broker gets real Bearer token injected."""

        async with aiohttp.ClientSession() as session:
            async with session.post(
                f"{proxy_env['proxy_url']}/v1/messages",
                headers={"Authorization": f"Bearer {proxy_env['token']}"},
                json={"model": "claude-sonnet-4-20250514", "messages": []},
            ) as resp:
                assert resp.status == 200
                body = await resp.json()

        # The upstream saw the REAL token, not the scoped one
        assert body["auth"] == "Bearer sk-ant-real-secret-token"
        assert proxy_env["token"] not in body["auth"]

    async def test_gh_api_call(self, proxy_env: dict) -> None:
        """GitHub API call through broker gets 'token <real>' injected."""

        async with aiohttp.ClientSession() as session:
            async with session.get(
                f"{proxy_env['proxy_url']}/user",
                headers={"Authorization": f"token {proxy_env['gh_token']}"},
            ) as resp:
                assert resp.status == 200
                body = await resp.json()

        assert body["auth"] == "token ghp_realGitHubToken123"

    async def test_unknown_route_returns_404(self, proxy_env: dict) -> None:
        """Token for an unregistered provider returns 404."""

        async with aiohttp.ClientSession() as session:
            async with session.get(
                f"{proxy_env['proxy_url']}/v1/foo",
                headers={"Authorization": f"Bearer {proxy_env['no_route_token']}"},
            ) as resp:
                assert resp.status == 404

    async def test_missing_auth_returns_401(self, proxy_env: dict) -> None:
        """Request without any auth header returns 401."""

        async with aiohttp.ClientSession() as session:
            async with session.get(
                f"{proxy_env['proxy_url']}/v1/messages",
            ) as resp:
                assert resp.status == 401

    async def test_invalid_token_returns_401(self, proxy_env: dict) -> None:
        """Request with a fake/revoked token returns 401."""

        async with aiohttp.ClientSession() as session:
            async with session.get(
                f"{proxy_env['proxy_url']}/v1/messages",
                headers={"Authorization": "Bearer totally-fake-token"},
            ) as resp:
                assert resp.status == 401


@pytest.mark.asyncio
class TestBrokerStreaming:
    """Story: broker streams SSE responses without buffering."""

    async def test_sse_chunks_forwarded(self, proxy_env: dict) -> None:
        """SSE response from upstream is streamed through the broker."""

        chunks: list[bytes] = []
        async with aiohttp.ClientSession() as session:
            async with session.get(
                f"{proxy_env['proxy_url']}/v1/stream",
                headers={"Authorization": f"Bearer {proxy_env['token']}"},
            ) as resp:
                assert resp.status == 200
                async for chunk in resp.content.iter_any():
                    chunks.append(chunk)

        combined = b"".join(chunks).decode()
        assert "chunk-0" in combined
        assert "chunk-2" in combined


@pytest.mark.asyncio
class TestBrokerAfterTokenRevocation:
    """Story: task ends, tokens revoked, broker rejects subsequent requests."""

    async def test_revoked_token_rejected_by_broker(self, proxy_env: dict) -> None:
        """After revoking, the broker returns 401 for the same token."""

        db = proxy_env["db"]
        token = proxy_env["token"]

        # Verify it works before revocation
        async with aiohttp.ClientSession() as session:
            async with session.post(
                f"{proxy_env['proxy_url']}/v1/messages",
                headers={"Authorization": f"Bearer {token}"},
            ) as resp:
                assert resp.status == 200

        # Revoke
        db.revoke_tokens("test-project", "task-1")

        # Now it's rejected
        async with aiohttp.ClientSession() as session:
            async with session.post(
                f"{proxy_env['proxy_url']}/v1/messages",
                headers={"Authorization": f"Bearer {token}"},
            ) as resp:
                assert resp.status == 401
