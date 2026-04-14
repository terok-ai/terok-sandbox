# SPDX-FileCopyrightText: 2026 Jiri Vyskocil
# SPDX-License-Identifier: Apache-2.0

"""Tests for gate_tokens module."""

from __future__ import annotations

import json
import tempfile
from collections.abc import Iterator
from contextlib import contextmanager
from pathlib import Path

import pytest

from terok_sandbox.gate.tokens import TokenStore
from tests.constants import (
    FAKE_TEROK_STATE_DIR,
    MISSING_TOKENS_PATH,
    NONEXISTENT_TOKENS_PATH,
)


@contextmanager
def patched_token_file(path: Path | None = None) -> Iterator[tuple[Path, TokenStore]]:
    """Create a TokenStore with a temporary JSON file and yield (path, store)."""
    if path is not None:
        store = TokenStore.__new__(TokenStore)
        store._path = path
        yield path, store
        return

    with tempfile.TemporaryDirectory() as td:
        token_path = Path(td) / "tokens.json"
        store = TokenStore.__new__(TokenStore)
        store._path = token_path
        yield token_path, store


def read_token_json(path: Path) -> dict[str, dict[str, str]]:
    """Load the persisted token data from disk."""
    return json.loads(path.read_text())


class TestTokenFilePath:
    """Tests for TokenStore.file_path."""

    def test_returns_path_under_state_root(self) -> None:
        from terok_sandbox.config import SandboxConfig

        cfg = SandboxConfig(state_dir=FAKE_TEROK_STATE_DIR)
        store = TokenStore(cfg)
        assert store.file_path == FAKE_TEROK_STATE_DIR / "gate" / "tokens.json"


class TestCreateToken:
    """Tests for TokenStore.create."""

    def test_returns_prefixed_hex(self) -> None:
        with patched_token_file() as (token_path, store):
            token = store.create("proj-a", "1")
            assert token_path.exists()
        assert token.startswith("terok-g-")
        assert len(token) == 8 + 32  # prefix + 32 hex chars
        int(token.removeprefix("terok-g-"), 16)

    def test_persists_to_file(self) -> None:
        with patched_token_file() as (token_path, store):
            token = store.create("proj-a", "1")
            data = read_token_json(token_path)
        assert data[token] == {"scope": "proj-a", "task": "1"}

    def test_multiple_tokens_coexist(self) -> None:
        with patched_token_file() as (token_path, store):
            first = store.create("proj-a", "1")
            second = store.create("proj-b", "2")
            data = read_token_json(token_path)
        assert first != second
        assert first in data
        assert second in data


class TestRevokeToken:
    """Tests for TokenStore.revoke_for_task."""

    def test_revoke_removes_entry(self) -> None:
        with patched_token_file() as (token_path, store):
            token = store.create("proj-a", "1")
            store.revoke_for_task("proj-a", "1")
            data = read_token_json(token_path)
        assert token not in data

    def test_revoke_nonexistent_is_noop(self) -> None:
        with patched_token_file() as (token_path, store):
            store.create("proj-a", "1")
            store.revoke_for_task("proj-a", "99")
            data = read_token_json(token_path)
        assert len(data) == 1

    def test_revoke_on_missing_file_is_noop(self) -> None:
        with patched_token_file(MISSING_TOKENS_PATH) as (_path, store):
            store.revoke_for_task("proj-a", "1")


class TestAtomicWrite:
    """Tests for atomic write via TokenStore._write."""

    def test_write_creates_parent_dirs(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            token_path = Path(td) / "sub" / "dir" / "tokens.json"
            store = TokenStore.__new__(TokenStore)
            store._path = token_path
            store._write({"abc": {"scope": "p", "task": "1"}})
            assert read_token_json(token_path) == {"abc": {"scope": "p", "task": "1"}}

    def test_read_missing_file_returns_empty(self) -> None:
        """Missing token file is treated as empty (first run)."""
        with tempfile.TemporaryDirectory() as td:
            store = TokenStore.__new__(TokenStore)
            store._path = Path(td) / NONEXISTENT_TOKENS_PATH.name
            assert store._read() == {}

    def test_read_corrupt_json_raises(self) -> None:
        """Corrupt JSON raises to prevent silent data loss on next write."""
        with tempfile.TemporaryDirectory() as td:
            token_path = Path(td) / "tokens.json"
            token_path.write_text("not json{{{")
            store = TokenStore.__new__(TokenStore)
            store._path = token_path
            with pytest.raises(json.JSONDecodeError):
                store._read()

    def test_read_non_dict_json_raises(self) -> None:
        """Non-dict JSON raises to prevent silent data loss on next write."""
        with tempfile.TemporaryDirectory() as td:
            token_path = Path(td) / "tokens.json"
            token_path.write_text(json.dumps(["not", "a", "dict"]))
            store = TokenStore.__new__(TokenStore)
            store._path = token_path
            with pytest.raises(ValueError, match="not a JSON object"):
                store._read()

    def test_read_filters_malformed_entries(self) -> None:
        """Malformed entries are silently filtered; valid ones survive."""
        with tempfile.TemporaryDirectory() as td:
            token_path = Path(td) / "tokens.json"
            token_path.write_text(
                json.dumps(
                    {
                        "good": {"scope": "p", "task": "1"},
                        "bad_info": "not a dict",
                        "missing_task": {"scope": "p"},
                        "int_scope": {"scope": 123, "task": "1"},
                    }
                )
            )
            store = TokenStore.__new__(TokenStore)
            store._path = token_path
            assert store._read() == {"good": {"scope": "p", "task": "1"}}

    def test_atomic_write_uses_replace(self) -> None:
        """Verify that TokenStore._write uses atomic replacement semantics."""
        with tempfile.TemporaryDirectory() as td:
            token_path = Path(td) / "tokens.json"
            store = TokenStore.__new__(TokenStore)
            store._path = token_path
            store._write({"t1": {"scope": "p", "task": "1"}})
            store._write({"t2": {"scope": "p", "task": "2"}})
            data = read_token_json(token_path)
            assert data == {"t2": {"scope": "p", "task": "2"}}
            assert list(Path(td).glob("*.tmp")) == []
