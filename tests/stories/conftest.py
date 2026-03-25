# SPDX-FileCopyrightText: 2026 Jiri Vyskocil
# SPDX-License-Identifier: Apache-2.0

"""Shared fixtures for credential proxy story tests.

Each fixture creates disposable resources (DB, config, sockets) within
``tmp_path`` — no host-state pollution.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from terok_sandbox.credential_db import CredentialDB


@pytest.fixture()
def db(tmp_path: Path) -> CredentialDB:
    """Return a fresh credential DB at a temp path."""
    return CredentialDB(tmp_path / "proxy" / "credentials.db")


@pytest.fixture()
def routes_file(tmp_path: Path) -> Path:
    """Write a standard routes config and return the path."""
    routes = {
        "claude": {
            "upstream": "https://api.anthropic.com",
            "auth_header": "Authorization",
            "auth_prefix": "Bearer ",
        },
        "codex": {
            "upstream": "https://api.openai.com",
            "auth_header": "Authorization",
            "auth_prefix": "Bearer ",
        },
        "gh": {
            "upstream": "https://api.github.com",
            "auth_header": "Authorization",
            "auth_prefix": "token ",
        },
    }
    path = tmp_path / "routes.json"
    path.write_text(json.dumps(routes), encoding="utf-8")
    return path


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
