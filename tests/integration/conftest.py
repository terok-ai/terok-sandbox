# SPDX-FileCopyrightText: 2026 Jiri Vyskocil
# SPDX-License-Identifier: Apache-2.0

"""Shared fixtures for vault integration tests.

Each fixture creates disposable resources (DB, config, sockets) within
``tmp_path`` -- no host-state pollution.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from terok_sandbox.credentials.db import CredentialDB


@pytest.fixture()
def db(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    """Yield a fresh credential DB, closing the connection on teardown.

    Pins every chain tier to a deterministic "test" outcome so
    downstream code that re-opens the DB via
    ``SandboxConfig().open_sqlcipher_connection()`` — notably
    [`_TokenDB`][terok_sandbox.vault.token_broker._TokenDB], which
    builds a fresh config and walks the production chain — resolves
    the same ``"test"`` value the DB was sealed with.

    Without this, the chain reaches into host state: a residual
    session-unlock tmpfs file or sealed systemd-creds credential on
    the developer machine would silently make ``_TokenDB`` open the
    DB with the *wrong* passphrase, surfacing as an opaque HMAC
    failure under the SQLCipher first-query path.
    """
    from terok_sandbox import config as _config
    from terok_sandbox.credentials import encryption as _enc, systemd_creds as _sc

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
