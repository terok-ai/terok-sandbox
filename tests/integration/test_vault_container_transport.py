# SPDX-FileCopyrightText: 2026 Jiri Vyskocil
# SPDX-License-Identifier: Apache-2.0

"""The vault's socket transport, exercised from inside a real podman container.

``test_vault_stories`` drives the broker with a same-process HTTP client; this adds
the transport hop the agent actually uses — the broker binds a UNIX socket, a real
container bind-mounts it, and ``curl --unix-socket`` inside the container drives the
request from a separate process and user namespace.  This is the layer sandbox owns
(``VaultProxy`` / ``UnixBind`` / the socket path), so its container-level guarantee
belongs here rather than up in terok.

No real API keys: a phantom token in the SQLCipher DB resolves to a fake "real"
credential, and a local aiohttp mock stands in for the provider.  The security
invariant under test is that the phantom swaps to the real key *and never leaves the
host* — asserted against what the mock upstream saw.
"""

from __future__ import annotations

import asyncio
import json
import shutil
import subprocess
import uuid
from collections.abc import AsyncIterator
from dataclasses import dataclass, field
from pathlib import Path

import pytest
from aiohttp import web

from terok_sandbox.vault.store.db import CredentialDB
from tests.constants import PODMAN_BASE_IMAGE

pytestmark = [pytest.mark.needs_vault, pytest.mark.needs_podman]

_CURL_IMAGE = "terok-sandbox-itest:latest"
_CONTAINER_PREFIX = "terok-sandbox-itest"
_REAL_KEY = "sk-ant-real-secret-001"  # nosec B105 — fixture value, not a real credential
# Generous build ceiling: a slow runner (a krun microVM, or a loaded crun
# host) building this image can take minutes; only a hung build should trip it.
_BUILD_TIMEOUT_S = 900


@dataclass
class _VaultSocket:
    """A broker bound to a UNIX socket, its phantom token, and the mock it fronts."""

    socket_path: Path
    db_path: Path
    phantom: str
    real_key: str
    upstream_requests: list[dict[str, str]] = field(default_factory=list)


@pytest.fixture(scope="session")
def curl_image() -> str:
    """Build (once) an Alpine image with ``curl`` — busybox wget can't do --unix-socket."""
    if not shutil.which("podman"):
        pytest.skip("podman not on PATH")
    if (
        subprocess.run(
            ["podman", "image", "exists", _CURL_IMAGE], capture_output=True, check=False
        ).returncode
        != 0
    ):
        subprocess.run(
            ["podman", "build", "-t", _CURL_IMAGE, "-f", "-", "."],
            input=f"FROM {PODMAN_BASE_IMAGE}\nRUN apk add --no-cache curl\n",
            check=True,
            text=True,
            timeout=_BUILD_TIMEOUT_S,
        )
    return _CURL_IMAGE


@pytest.fixture
async def vault_socket(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> AsyncIterator[_VaultSocket]:
    """Run a mock upstream + a broker on a UNIX socket, wired to a seeded phantom.

    Self-contained: seals a throwaway DB with a pinned ``"test"`` passphrase (and
    pins the resolution chain the broker's ``_TokenDB`` walks to the same value), so
    the operator's real vault and passphrase chain are never touched.
    """
    from terok_sandbox import config as _config
    from terok_sandbox.vault.daemon.token_broker import _build_app
    from terok_sandbox.vault.store import (
        encryption as _enc,
        kernel_keyring as _kk,
        systemd_creds as _sc,
    )

    # Pin every passphrase tier to a deterministic "test" so the broker's _TokenDB
    # (which re-opens the DB through the production chain) matches how it was sealed.
    # Blank the upper tiers (kernel-keyring cache, sealed systemd-creds) so the chain
    # falls through to the OS-keyring tier, which we pin — mirrors sandbox's own `db`
    # fixture (the session-file tier was replaced by kernel-keyring in #461).
    monkeypatch.setattr(_kk, "load", lambda _db=None: None)
    monkeypatch.setattr(_sc, "unseal", lambda _path: None)
    # ``**_kw`` absorbs ``allow_prompt``, which the encryption chain now passes.
    monkeypatch.setattr(_enc, "load_passphrase_from_keyring", lambda **_kw: "test")
    monkeypatch.setattr(_config, "credentials_use_keyring", lambda: True)

    state = _VaultSocket(
        socket_path=tmp_path / "vault.sock",
        db_path=tmp_path / "vault" / "credentials.db",
        phantom="",
        real_key=_REAL_KEY,
    )

    async def _echo(request: web.Request) -> web.Response:
        state.upstream_requests.append(dict(request.headers))
        return web.json_response(
            {"path": request.path, "authorization": request.headers.get("Authorization", "")}
        )

    upstream_app = web.Application()
    upstream_app.router.add_route("*", "/{tail:.*}", _echo)
    upstream_runner = web.AppRunner(upstream_app)
    await upstream_runner.setup()
    upstream_site = web.TCPSite(upstream_runner, host="127.0.0.1", port=0)
    await upstream_site.start()
    up_sock = upstream_site._server.sockets[0]  # type: ignore[attr-defined]
    upstream_url = f"http://{up_sock.getsockname()[0]}:{up_sock.getsockname()[1]}"

    routes_path = tmp_path / "routes.json"
    routes_path.write_text(
        json.dumps(
            {
                "claude": {
                    "upstream": upstream_url,
                    "auth_header": "Authorization",
                    "auth_prefix": "Bearer ",
                }
            }
        )
    )

    db = CredentialDB(state.db_path, passphrase="test")
    db.store_credential("default", "claude", {"type": "api_key", "key": _REAL_KEY})
    state.phantom = db.create_token("project-x", "task-42", "default", "claude")
    db.close()

    broker_runner = web.AppRunner(_build_app(str(state.db_path), str(routes_path)))
    await broker_runner.setup()
    broker_site = web.UnixSite(broker_runner, path=str(state.socket_path))
    await broker_site.start()
    # Loosen perms so a container peer with a different mapped UID reaches the socket
    # through the bind-mount; the file lives in a per-test tmp dir either way.
    state.socket_path.chmod(0o666)

    try:
        yield state
    finally:
        await broker_runner.cleanup()
        await upstream_runner.cleanup()


def _qualified(image: str) -> str:
    """Qualify a local-store image so podman's strict short-name policy resolves it."""
    return image if "/" in image else f"localhost/{image}"


async def _run_keepalive(name: str, image: str, host_socket: Path) -> None:
    """Start a keepalive container with the broker socket bind-mounted at /vault.sock."""
    result = await asyncio.to_thread(
        subprocess.run,
        [
            "podman", "run", "-d", "--rm",
            "--pull", "never",
            "--security-opt", "label=disable",
            "--name", name,
            "-v", f"{host_socket}:/vault.sock",
            _qualified(image),
            "sleep", "60",
        ],
        check=False,
        capture_output=True,
        text=True,
        timeout=30,
    )  # fmt: skip
    assert result.returncode == 0, (
        f"podman run failed (exit {result.returncode}): {result.stderr!r}"
    )


async def _curl(name: str, phantom: str) -> subprocess.CompletedProcess[str]:
    """Drive one request from inside the container through the bind-mounted socket."""
    return await asyncio.to_thread(
        subprocess.run,
        [
            "podman", "exec", name,
            "curl", "-sS", "--fail",
            "--unix-socket", "/vault.sock",
            "-H", f"Authorization: Bearer {phantom}",
            "http://localhost/v1/messages",
        ],
        check=False,
        capture_output=True,
        text=True,
        timeout=15,
    )  # fmt: skip


async def test_phantom_swaps_to_real_key_through_container_socket(
    vault_socket: _VaultSocket,
    curl_image: str,
) -> None:
    """Phantom in from the container → real key out at the upstream, no leak."""
    if not shutil.which("podman"):
        pytest.skip("podman not on PATH")

    name = f"{_CONTAINER_PREFIX}-swap-{uuid.uuid4().hex[:8]}"
    try:
        await _run_keepalive(name, curl_image, vault_socket.socket_path)
        result = await _curl(name, vault_socket.phantom)
        assert result.returncode == 0, f"in-container curl failed: {result.stderr!r}"

        assert len(vault_socket.upstream_requests) == 1
        seen = vault_socket.upstream_requests[-1]
        assert seen.get("Authorization") == f"Bearer {vault_socket.real_key}"
        assert vault_socket.phantom not in " ".join(seen.values()), "phantom leaked upstream"
    finally:
        await asyncio.to_thread(
            subprocess.run, ["podman", "rm", "-f", name], capture_output=True, timeout=30
        )


async def test_revoked_phantom_is_rejected_at_the_socket(
    vault_socket: _VaultSocket,
    curl_image: str,
) -> None:
    """After revocation the container's token 401s at the broker; the upstream is untouched."""
    if not shutil.which("podman"):
        pytest.skip("podman not on PATH")

    # Revoke through a fresh handle (the fixture already closed its writer).
    db = CredentialDB(vault_socket.db_path, passphrase="test")
    assert db.revoke_tokens("project-x", "task-42") >= 1
    db.close()

    name = f"{_CONTAINER_PREFIX}-revoked-{uuid.uuid4().hex[:8]}"
    try:
        await _run_keepalive(name, curl_image, vault_socket.socket_path)
        result = await _curl(name, vault_socket.phantom)
        # curl --fail exits non-zero on the broker's 401.
        assert result.returncode != 0, "revoked phantom was not rejected"
        assert vault_socket.upstream_requests == [], "revoked phantom reached the upstream"
    finally:
        await asyncio.to_thread(
            subprocess.run, ["podman", "rm", "-f", name], capture_output=True, timeout=30
        )
