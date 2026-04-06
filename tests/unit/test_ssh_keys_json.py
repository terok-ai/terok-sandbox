# SPDX-FileCopyrightText: 2026 Jiri Vyskocil
# SPDX-License-Identifier: Apache-2.0

"""Tests for SSH key JSON sidecar management (update_ssh_keys_json)."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from terok_sandbox.credentials.ssh import SSHInitResult, update_ssh_keys_json


def _result(priv: str = "/k/priv", pub: str = "/k/pub") -> SSHInitResult:
    """Build a minimal SSHInitResult for testing."""
    return SSHInitResult(
        dir="/keys", private_key=priv, public_key=pub, config_path="/keys/config", key_name="id"
    )


class TestUpdateSshKeysJson:
    """Verify update_ssh_keys_json read-modify-write behavior."""

    def test_creates_file_if_missing(self, tmp_path: Path) -> None:
        """Creates ssh-keys.json (and parent dirs) on first call."""
        keys_path = tmp_path / "proxy" / "ssh-keys.json"
        update_ssh_keys_json(keys_path, "proj-a", _result("/a/priv", "/a/pub"))

        data = json.loads(keys_path.read_text())
        assert data == {"proj-a": [{"private_key": "/a/priv", "public_key": "/a/pub"}]}

    def test_appends_to_existing(self, tmp_path: Path) -> None:
        """Adds a new project without overwriting existing entries."""
        keys_path = tmp_path / "ssh-keys.json"
        keys_path.write_text(json.dumps({"old": [{"private_key": "/o", "public_key": "/o.pub"}]}))

        update_ssh_keys_json(keys_path, "new", _result("/n", "/n.pub"))

        data = json.loads(keys_path.read_text())
        assert "old" in data
        assert data["new"] == [{"private_key": "/n", "public_key": "/n.pub"}]

    def test_same_private_key_path_is_idempotent(self, tmp_path: Path) -> None:
        """Re-running ssh-init with the same private key path updates in-place."""
        keys_path = tmp_path / "ssh-keys.json"
        update_ssh_keys_json(keys_path, "proj", _result("/v1/priv", "/v1/pub"))
        update_ssh_keys_json(keys_path, "proj", _result("/v1/priv", "/v1/pub-updated"))

        data = json.loads(keys_path.read_text())
        assert isinstance(data["proj"], list)
        assert len(data["proj"]) == 1
        assert data["proj"][0]["public_key"] == "/v1/pub-updated"

    def test_different_path_expands_to_list(self, tmp_path: Path) -> None:
        """A second key with a different private_key path creates a list for the project."""
        keys_path = tmp_path / "ssh-keys.json"
        update_ssh_keys_json(keys_path, "proj", _result("/k1/priv", "/k1/pub"))
        update_ssh_keys_json(keys_path, "proj", _result("/k2/priv", "/k2/pub"))

        data = json.loads(keys_path.read_text())
        assert isinstance(data["proj"], list)
        assert len(data["proj"]) == 2
        assert data["proj"][0]["private_key"] == "/k1/priv"
        assert data["proj"][1]["private_key"] == "/k2/priv"

    def test_list_replaces_matching_entry(self, tmp_path: Path) -> None:
        """Updating a key already in a list replaces the entry with the matching path."""
        keys_path = tmp_path / "ssh-keys.json"
        keys_path.write_text(
            json.dumps(
                {
                    "proj": [
                        {"private_key": "/k1/priv", "public_key": "/k1/pub"},
                        {"private_key": "/k2/priv", "public_key": "/k2/pub"},
                    ]
                }
            )
        )
        update_ssh_keys_json(keys_path, "proj", _result("/k1/priv", "/k1/pub-new"))

        data = json.loads(keys_path.read_text())
        assert len(data["proj"]) == 2
        assert data["proj"][0]["public_key"] == "/k1/pub-new"
        assert data["proj"][1]["private_key"] == "/k2/priv"

    def test_list_appends_new_path(self, tmp_path: Path) -> None:
        """A new private_key path is appended to an existing list entry."""
        keys_path = tmp_path / "ssh-keys.json"
        keys_path.write_text(
            json.dumps({"proj": [{"private_key": "/k1/priv", "public_key": "/k1/pub"}]})
        )
        update_ssh_keys_json(keys_path, "proj", _result("/k3/priv", "/k3/pub"))

        data = json.loads(keys_path.read_text())
        assert len(data["proj"]) == 2
        assert data["proj"][1]["private_key"] == "/k3/priv"

    def test_handles_empty_file(self, tmp_path: Path) -> None:
        """Handles a pre-existing empty file (e.g. from lifecycle.start_daemon)."""
        keys_path = tmp_path / "ssh-keys.json"
        keys_path.write_text("")

        update_ssh_keys_json(keys_path, "proj", _result())

        data = json.loads(keys_path.read_text())
        assert "proj" in data

    def test_malformed_json_raises(self, tmp_path: Path) -> None:
        """Malformed JSON raises JSONDecodeError — caller is responsible for recovery."""
        keys_path = tmp_path / "ssh-keys.json"
        keys_path.write_text("{not valid json")

        with pytest.raises(json.JSONDecodeError):
            update_ssh_keys_json(keys_path, "proj", _result())
