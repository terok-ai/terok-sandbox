# SPDX-FileCopyrightText: 2026 Jiri Vyskocil
# SPDX-License-Identifier: Apache-2.0

"""Tests for backup restore/delete: a restore can always be un-restored.

Real repositories again — the properties under test are ref-level CAS
semantics.  The central scenario: an agent force-push destroyed a tip,
the hook backed it up, the operator restores it, and the *rewritten*
tip is now itself backed up — no state is ever the last copy.
"""

from __future__ import annotations

import subprocess
from pathlib import Path

import pytest

from terok_sandbox.gate.hooks import hooks_dir_for, install_hooks
from terok_sandbox.gate.mirror import _BACKUP_PREFIX, GitGate


def _git(cwd: Path, *args: str) -> str:
    """Run git in *cwd* with a fixed identity; return stripped stdout."""
    return subprocess.run(
        ["git", "-C", str(cwd), "-c", "user.name=t", "-c", "user.email=t@t", *args],
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()


def _commit(repo: Path, name: str, content: str) -> str:
    """Commit one file change in *repo*; return the new tip sha."""
    (repo / name).write_text(content)
    _git(repo, "add", ".")
    _git(repo, "commit", "-m", f"{name}={content!r}")
    return _git(repo, "rev-parse", "HEAD")


@pytest.fixture
def gate_env(tmp_path: Path) -> tuple[GitGate, Path, Path]:
    """A synced gate with hooks installed, and an agent-style clone."""
    upstream = tmp_path / "upstream"
    subprocess.run(["git", "init", "-b", "master", str(upstream)], check=True, capture_output=True)
    _commit(upstream, "file.txt", "v1\n")
    gate = GitGate(
        scope="proj",
        gate_path=tmp_path / "mirror" / "proj.git",
        upstream_url=str(upstream),
    )
    assert gate.sync()["success"] is True
    hooks_dir = hooks_dir_for(tmp_path / "mirror")
    install_hooks(hooks_dir)
    work = tmp_path / "work"
    subprocess.run(
        ["git", "clone", "-q", str(gate._gate_path), str(work)], check=True, capture_output=True
    )
    _git(work, "checkout", "-q", "master")
    return gate, hooks_dir, work


def _hooked_push(work: Path, hooks_dir: Path, *refspecs: str, force: bool = False) -> None:
    """Push through the sandbox hooks the way the server injects them."""
    receive_pack = (
        f"env GIT_CONFIG_COUNT=1 GIT_CONFIG_KEY_0=core.hooksPath"
        f" GIT_CONFIG_VALUE_0={hooks_dir} git receive-pack"
    )
    args = ["push", "-q", f"--receive-pack={receive_pack}"]
    if force:
        args.append("--force")
    subprocess.run(
        ["git", "-C", str(work), *args, "origin", *refspecs],
        check=True,
        capture_output=True,
        text=True,
    )


def _force_pushed_branch(gate: GitGate, hooks_dir: Path, work: Path) -> tuple[str, str]:
    """Create feat/x, force-push over it; return (old_tip, new_tip)."""
    _git(work, "checkout", "-q", "-b", "feat/x")
    old = _commit(work, "a.txt", "1\n")
    _hooked_push(work, hooks_dir, "HEAD:refs/heads/feat/x")
    _git(work, "reset", "-q", "--hard", "HEAD~1")
    new = _commit(work, "a.txt", "rewritten\n")
    _hooked_push(work, hooks_dir, "HEAD:refs/heads/feat/x", force=True)
    return old, new


class TestRestoreBackup:
    """Restore moves the branch back and backs up what it replaced."""

    def test_restore_round_trip_keeps_both_tips(self, gate_env) -> None:
        gate, hooks_dir, work = gate_env
        old, new = _force_pushed_branch(gate, hooks_dir, work)
        (entry,) = gate.list_backups()
        assert entry["sha"] == old

        result = gate.restore_backup(entry["ref"])

        assert result["error"] is None
        assert result["restored_sha"] == old
        assert gate.branch_heads()["feat/x"] == old
        # The rewritten tip the restore replaced is itself backed up now.
        assert result["previous_backup_ref"] is not None
        backed_up = {e["sha"] for e in gate.list_backups()}
        assert {old, new} <= backed_up

    def test_restore_recreates_deleted_branch(self, gate_env) -> None:
        gate, hooks_dir, work = gate_env
        _git(work, "checkout", "-q", "-b", "feat/gone")
        tip = _commit(work, "a.txt", "1\n")
        _hooked_push(work, hooks_dir, "HEAD:refs/heads/feat/gone")
        _hooked_push(work, hooks_dir, ":refs/heads/feat/gone")
        (entry,) = gate.list_backups()

        result = gate.restore_backup(entry["ref"])

        assert result["error"] is None
        assert result["previous_backup_ref"] is None
        assert gate.branch_heads()["feat/gone"] == tip

    def test_restore_to_current_tip_is_a_quiet_noop(self, gate_env) -> None:
        gate, hooks_dir, work = gate_env
        old, _new = _force_pushed_branch(gate, hooks_dir, work)
        (entry,) = gate.list_backups()
        gate.restore_backup(entry["ref"])

        again = gate.restore_backup(entry["ref"])

        assert again["error"] is None
        assert again["previous_backup_ref"] is None
        assert gate.branch_heads()["feat/x"] == old

    def test_restore_rejects_non_backup_ref(self, gate_env) -> None:
        gate, _hooks_dir, _work = gate_env
        result = gate.restore_backup("refs/heads/master")
        assert result["error"] is not None
        assert gate.branch_heads()["master"]

    def test_restore_of_vanished_backup_errors(self, gate_env) -> None:
        gate, hooks_dir, work = gate_env
        _force_pushed_branch(gate, hooks_dir, work)
        (entry,) = gate.list_backups()
        assert gate.delete_backup(entry["ref"]) is None

        result = gate.restore_backup(entry["ref"])

        assert result["error"] is not None


class TestDeleteBackup:
    """Deleting is scoped to the backup namespace, and only there."""

    def test_delete_removes_the_ref(self, gate_env) -> None:
        gate, hooks_dir, work = gate_env
        _force_pushed_branch(gate, hooks_dir, work)
        (entry,) = gate.list_backups()

        assert gate.delete_backup(entry["ref"]) is None
        assert gate.list_backups() == []

    def test_delete_refuses_refs_outside_the_namespace(self, gate_env) -> None:
        gate, _hooks_dir, _work = gate_env
        error = gate.delete_backup("refs/heads/master")
        assert error is not None
        assert gate.branch_heads()["master"]

    def test_delete_of_missing_backup_reports_error(self, gate_env) -> None:
        gate, _hooks_dir, _work = gate_env
        error = gate.delete_backup(f"{_BACKUP_PREFIX}feat/x/20260101T000000Z-000000000000")
        assert error is not None
