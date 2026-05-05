# SPDX-FileCopyrightText: 2026 Jiri Vyskocil
# SPDX-License-Identifier: Apache-2.0

"""Tests for the broker's append-only credential-use audit log."""

from __future__ import annotations

import asyncio
import json
import os
import stat
from pathlib import Path

import pytest

from terok_sandbox.vault.audit import (
    AuditWriter,
    credential_audit_log_path,
)


class TestCredentialAuditPath:
    """``credential_audit_log_path`` returns a stable path under the vault root."""

    def test_path_lives_directly_under_vault_root(self, tmp_path: Path) -> None:
        path = credential_audit_log_path(tmp_path)
        assert path == tmp_path / "credential_audit.jsonl"


class TestAuditWriter:
    """Verifies the writer's append, lazy-open, and soft-fail behaviours."""

    @pytest.mark.asyncio
    async def test_first_write_creates_parent_dir_and_file(self, tmp_path: Path) -> None:
        path = tmp_path / "nested" / "credential_audit.jsonl"
        writer = AuditWriter(path)
        try:
            await writer.write({"scope": "proj", "subject": "subj-1"})
        finally:
            await writer.close()
        assert path.is_file()
        line = path.read_text().strip()
        assert json.loads(line) == {"scope": "proj", "subject": "subj-1"}

    @pytest.mark.asyncio
    async def test_each_write_emits_one_jsonl_line(self, tmp_path: Path) -> None:
        path = tmp_path / "audit.jsonl"
        writer = AuditWriter(path)
        try:
            await writer.write({"a": 1})
            await writer.write({"b": 2})
            await writer.write({"c": 3})
        finally:
            await writer.close()
        lines = path.read_text().splitlines()
        assert [json.loads(line) for line in lines] == [{"a": 1}, {"b": 2}, {"c": 3}]

    @pytest.mark.asyncio
    async def test_concurrent_writes_do_not_interleave_bytes(self, tmp_path: Path) -> None:
        """Concurrent ``write`` calls under the writer's lock keep JSONL parseable."""
        path = tmp_path / "audit.jsonl"
        writer = AuditWriter(path)
        try:
            entries = [{"i": i, "blob": "x" * 200} for i in range(50)]
            await asyncio.gather(*(writer.write(e) for e in entries))
        finally:
            await writer.close()
        lines = path.read_text().splitlines()
        assert len(lines) == 50
        seen = sorted(json.loads(line)["i"] for line in lines)
        assert seen == list(range(50))

    @pytest.mark.asyncio
    async def test_file_permissions_are_owner_only(self, tmp_path: Path) -> None:
        """The audit file is created with mode 0600 — host filesystem ACL boundary."""
        path = tmp_path / "audit.jsonl"
        writer = AuditWriter(path)
        try:
            await writer.write({"x": 1})
        finally:
            await writer.close()
        mode = stat.S_IMODE(path.stat().st_mode)
        assert mode == 0o600

    @pytest.mark.asyncio
    async def test_unwritable_path_soft_fails(
        self, tmp_path: Path, caplog: pytest.LogCaptureFixture
    ) -> None:
        """A non-creatable path logs a single warning and never raises.

        Hitting ``mkdir`` on an unwritable parent is the realistic
        failure mode (e.g. read-only mount, permissions misconfig).
        """
        # Make the target's parent dir unwritable so mkdir(exist_ok=True)
        # on a deeper path fails with PermissionError.
        ro_root = tmp_path / "ro"
        ro_root.mkdir()
        os.chmod(ro_root, 0o500)
        try:
            path = ro_root / "blocked" / "audit.jsonl"
            writer = AuditWriter(path)
            with caplog.at_level("WARNING"):
                await writer.write({"x": 1})
                await writer.write({"x": 2})
            await writer.close()
            warnings = [r for r in caplog.records if r.levelname == "WARNING"]
            # Lazy-open failure logs once; subsequent writes short-circuit
            # silently so a sustained outage doesn't flood the log.
            assert len(warnings) == 1
            assert "audit log unavailable" in warnings[0].message
        finally:
            os.chmod(ro_root, 0o700)

    @pytest.mark.asyncio
    async def test_close_is_idempotent(self, tmp_path: Path) -> None:
        path = tmp_path / "audit.jsonl"
        writer = AuditWriter(path)
        await writer.write({"x": 1})
        await writer.close()
        await writer.close()  # second close: no-op, no exception
