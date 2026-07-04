# SPDX-FileCopyrightText: 2026 Jiri Vyskocil
# SPDX-License-Identifier: Apache-2.0

"""Shared fixtures for vault integration tests.

Each fixture creates disposable resources (DB, config, sockets) within
``tmp_path`` -- no host-state pollution.
"""

from __future__ import annotations

import os
import shutil
import socket
from pathlib import Path

import pytest

from terok_sandbox.vault.store.db import CredentialDB
from tests.constants import PUBLIC_DNS_PROBE


@pytest.fixture()
def db(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    """Yield a fresh credential DB, closing the connection on teardown.

    Pins every chain tier to a deterministic "test" outcome so
    downstream code that re-opens the DB via
    ``SandboxConfig().open_sqlcipher_connection()`` — notably
    [`_TokenDB`][terok_sandbox.vault.daemon.token_broker._TokenDB], which
    builds a fresh config and walks the production chain — resolves
    the same ``"test"`` value the DB was sealed with.

    Without this, the chain reaches into host state: a residual
    session-unlock tmpfs file or sealed systemd-creds credential on
    the developer machine would silently make ``_TokenDB`` open the
    DB with the *wrong* passphrase, surfacing as an opaque HMAC
    failure under the SQLCipher first-query path.
    """
    from terok_sandbox import config as _config
    from terok_sandbox.vault.store import encryption as _enc, systemd_creds as _sc

    # Blank the upper tiers so the chain falls through to keyring.
    monkeypatch.setattr(_enc, "load_passphrase_from_file", lambda _path: None)
    monkeypatch.setattr(_sc, "unseal", lambda _path: None)
    # Pin keyring to the test passphrase.
    monkeypatch.setattr(_enc, "load_passphrase_from_keyring", lambda: "test")
    monkeypatch.setattr(_config, "credentials_use_keyring", lambda: True)

    database = CredentialDB(tmp_path / "proxy" / "credentials.db", passphrase="test")
    yield database
    database.close()


@pytest.fixture()
def populated_db(db: CredentialDB) -> CredentialDB:
    """Return a DB pre-populated with test credentials for all routed providers."""
    db.store_credential(
        "default",
        "claude",
        {
            "type": "oauth",
            "access_token": "sk-ant-real-secret-token",
            "refresh_token": "rt-ant-refresh",
        },
    )
    db.store_credential(
        "default",
        "codex",
        {
            "type": "oauth",
            "access_token": "sk-openai-real-secret",
        },
    )
    db.store_credential(
        "default",
        "gh",
        {
            "type": "oauth_token",
            "token": "ghp_realGitHubToken123",
        },
    )
    return db


# ── Matrix capability contract ───────────────────────────────────────
# On a dev machine a missing binary is a host limitation and skipping is
# the right degradation.  Inside the matrix the harness BUILT the image,
# so every capability it declares (TEROK_EXPECT, exported by the matrix
# engine) is a contract: absence means the slot is broken and must fail
# at session start — not dissolve into skips that read as green.

# dnsmasq/nft install into sbin on several distros while the test user's
# PATH may omit those dirs — probe with them appended.
_SBIN_AWARE_PATH = os.pathsep.join(
    [os.environ.get("PATH", ""), "/usr/sbin", "/sbin", "/usr/local/sbin"]
)

_CAPABILITY_PROBES = {
    "podman": lambda: _has("podman"),
    "nft": lambda: _has("nft"),
    "dnsmasq": lambda: _has("dnsmasq"),
    "dig": lambda: _has("dig"),
    "getent": lambda: _has("getent"),
    "git": lambda: _has("git"),
    "internet": lambda: _tcp_reachable(*PUBLIC_DNS_PROBE),
}


def _has(binary: str) -> bool:
    """Whether *binary* resolves on the sbin-extended PATH."""
    return shutil.which(binary, path=_SBIN_AWARE_PATH) is not None


def _tcp_reachable(ip: str, port: int, timeout: float = 5.0) -> bool:
    """Whether a TCP connection to ``ip:port`` succeeds within *timeout*."""
    try:
        with socket.create_connection((ip, port), timeout=timeout):
            return True
    except OSError:
        return False


def pytest_sessionstart(session: pytest.Session) -> None:
    """Fail the whole session when the matrix capability contract is broken."""
    expected = {c for c in os.environ.get("TEROK_EXPECT", "").split(",") if c}
    if not expected:
        return
    unknown = expected - _CAPABILITY_PROBES.keys()
    if unknown:
        pytest.exit(f"TEROK_EXPECT names unknown capabilities: {sorted(unknown)}", returncode=3)
    missing = sorted(cap for cap in expected if not _CAPABILITY_PROBES[cap]())
    if missing:
        pytest.exit(
            "matrix capability contract broken — expected but missing: " + ", ".join(missing),
            returncode=3,
        )
