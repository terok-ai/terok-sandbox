# SPDX-FileCopyrightText: 2026 Jiri Vyskocil
# SPDX-License-Identifier: Apache-2.0

"""Shared fixtures for credential proxy integration tests.

Each fixture creates disposable resources (DB, config, sockets) within
``tmp_path`` — no host-state pollution.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from terok_sandbox.credentials.db import CredentialDB


@pytest.fixture()
def db(tmp_path: Path):
    """Yield a fresh credential DB, closing the connection on teardown."""
    database = CredentialDB(tmp_path / "proxy" / "credentials.db")
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
