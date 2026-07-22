# SPDX-FileCopyrightText: 2026 Jiri Vyskocil
# SPDX-License-Identifier: Apache-2.0

"""Tests for the sandbox-owned gate hooks: agent-op backups + push marker.

Everything runs against real git repositories, with ``core.hooksPath``
injected on the push the same way the gate's HTTP server injects it —
``git -c`` travels to the locally spawned ``receive-pack`` via
``GIT_CONFIG_PARAMETERS``, so the receive path exercised here is the one
agent pushes take (minus HTTP).
"""

from __future__ import annotations

import stat
import subprocess
from pathlib import Path

import pytest

from terok_sandbox.gate.hooks import (
    HOOKS_DIRNAME,
    PUSH_MARKER_FILENAME,
    hooks_dir_for,
    install_hooks,
)
from terok_sandbox.gate.mirror import _BACKUP_STAMP_RE, GitGate, _read_refs

# ---------------------------------------------------------------------------
# Real-repo helpers
# ---------------------------------------------------------------------------


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
    """A synced gate, its installed hooks dir, and an agent-style clone."""
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


def _hooked_push(work: Path, hooks_dir: Path, *refspecs: str, force: bool = False) -> str:
    """Push through the sandbox hooks the way the server injects them.

    The server hands ``core.hooksPath`` to ``git http-backend`` as
    ``GIT_CONFIG_*`` environment (which receive-pack honors), but a *local*
    push scrubs repo-scoped env before spawning receive-pack — so the test
    wraps receive-pack itself to re-inject exactly the server's variables.
    """
    receive_pack = (
        f"env GIT_CONFIG_COUNT=1 GIT_CONFIG_KEY_0=core.hooksPath"
        f" GIT_CONFIG_VALUE_0={hooks_dir} git receive-pack"
    )
    args = ["push", "-q", f"--receive-pack={receive_pack}"]
    if force:
        args.append("--force")
    result = subprocess.run(
        ["git", "-C", str(work), *args, "origin", *refspecs],
        check=True,
        capture_output=True,
        text=True,
    )
    return result.stderr


def _backups(gate: GitGate) -> list[dict]:
    return list(gate.list_backups())


# ---------------------------------------------------------------------------
# install_hooks
# ---------------------------------------------------------------------------


class TestInstallHooks:
    """Rendering is executable, idempotent, and outside every gate repo."""

    def test_writes_executable_post_receive(self, tmp_path: Path) -> None:
        hooks_dir = hooks_dir_for(tmp_path)
        install_hooks(hooks_dir)
        hook = hooks_dir / "post-receive"
        assert hook.name in {p.name for p in hooks_dir.iterdir()}
        assert hook.stat().st_mode & stat.S_IXUSR
        assert hooks_dir.name == HOOKS_DIRNAME

    def test_reinstall_is_idempotent(self, tmp_path: Path) -> None:
        hooks_dir = hooks_dir_for(tmp_path)
        install_hooks(hooks_dir)
        hook = hooks_dir / "post-receive"
        before = (hook.stat().st_mtime_ns, hook.read_text())
        install_hooks(hooks_dir)
        assert (hook.stat().st_mtime_ns, hook.read_text()) == before


# ---------------------------------------------------------------------------
# Backup semantics per push kind
# ---------------------------------------------------------------------------


class TestAgentOpBackups:
    """Destructive agent updates leave a backup ref; benign ones don't."""

    def test_create_and_fast_forward_leave_no_backup(self, gate_env) -> None:
        gate, hooks_dir, work = gate_env
        _git(work, "checkout", "-q", "-b", "feat/x")
        _commit(work, "a.txt", "1\n")
        _hooked_push(work, hooks_dir, "HEAD:refs/heads/feat/x")
        _commit(work, "a.txt", "2\n")
        _hooked_push(work, hooks_dir, "HEAD:refs/heads/feat/x")
        assert _backups(gate) == []

    def test_force_push_backs_up_old_tip(self, gate_env) -> None:
        gate, hooks_dir, work = gate_env
        _git(work, "checkout", "-q", "-b", "feat/x")
        old = _commit(work, "a.txt", "1\n")
        _hooked_push(work, hooks_dir, "HEAD:refs/heads/feat/x")
        _git(work, "reset", "-q", "--hard", "HEAD~1")
        _commit(work, "a.txt", "rewritten\n")
        stderr = _hooked_push(work, hooks_dir, "HEAD:refs/heads/feat/x", force=True)

        (entry,) = _backups(gate)
        assert entry["branch"] == "feat/x"
        assert entry["sha"] == old
        leaf = entry["ref"].rsplit("/", 1)[-1]
        assert _BACKUP_STAMP_RE.match(leaf)
        assert "gate: saved" in stderr

    def test_delete_backs_up_deleted_tip(self, gate_env) -> None:
        gate, hooks_dir, work = gate_env
        _git(work, "checkout", "-q", "-b", "feat/gone")
        tip = _commit(work, "a.txt", "1\n")
        _hooked_push(work, hooks_dir, "HEAD:refs/heads/feat/gone")
        _hooked_push(work, hooks_dir, ":refs/heads/feat/gone")

        (entry,) = _backups(gate)
        assert (entry["branch"], entry["sha"]) == ("feat/gone", tip)
        assert "feat/gone" not in gate.branch_heads()

    def test_backup_failure_warns_but_push_stands(self, gate_env) -> None:
        """A backup that cannot be written must be loud, never blocking."""
        gate, hooks_dir, work = gate_env
        _git(work, "checkout", "-q", "-b", "feat/x")
        _commit(work, "a.txt", "1\n")
        _hooked_push(work, hooks_dir, "HEAD:refs/heads/feat/x")
        _git(work, "reset", "-q", "--hard", "HEAD~1")
        new = _commit(work, "a.txt", "rewritten\n")

        refs_terok = gate._gate_path / "refs" / "terok"
        refs_terok.mkdir(parents=True, exist_ok=True)
        refs_terok.chmod(0o555)
        try:
            stderr = _hooked_push(work, hooks_dir, "HEAD:refs/heads/feat/x", force=True)
        finally:
            refs_terok.chmod(0o755)

        assert "WARNING could not back up" in stderr
        assert _backups(gate) == []
        assert gate.branch_heads()["feat/x"] == new
        marker = (gate._gate_path / PUSH_MARKER_FILENAME).read_text()
        assert "backup-FAILED feat/x" in marker


# ---------------------------------------------------------------------------
# Push marker + branch_heads
# ---------------------------------------------------------------------------


class TestPushMarkerAndHeads:
    """Every push stamps the marker; branch_heads mirrors refs/heads."""

    def test_marker_records_updated_refs(self, gate_env) -> None:
        gate, hooks_dir, work = gate_env
        _git(work, "checkout", "-q", "-b", "feat/x")
        sha = _commit(work, "a.txt", "1\n")
        _hooked_push(work, hooks_dir, "HEAD:refs/heads/feat/x")

        marker = (gate._gate_path / PUSH_MARKER_FILENAME).read_text()
        assert "refs/heads/feat/x" in marker
        assert sha in marker

    def test_marker_mtime_advances_per_push(self, gate_env) -> None:
        gate, hooks_dir, work = gate_env
        _git(work, "checkout", "-q", "-b", "feat/x")
        _commit(work, "a.txt", "1\n")
        _hooked_push(work, hooks_dir, "HEAD:refs/heads/feat/x")
        marker = gate._gate_path / PUSH_MARKER_FILENAME
        first = marker.stat().st_mtime_ns
        _commit(work, "a.txt", "2\n")
        _hooked_push(work, hooks_dir, "HEAD:refs/heads/feat/x")
        assert marker.stat().st_mtime_ns > first

    def test_branch_heads_enumerates_all_pushed_branches(self, gate_env) -> None:
        gate, hooks_dir, work = gate_env
        _git(work, "checkout", "-q", "-b", "feat/x")
        sha_x = _commit(work, "a.txt", "1\n")
        _hooked_push(work, hooks_dir, "HEAD:refs/heads/feat/x")
        _git(work, "checkout", "-q", "-b", "feat/y")
        sha_y = _commit(work, "b.txt", "1\n")
        _hooked_push(work, hooks_dir, "HEAD:refs/heads/feat/y")

        heads = gate.branch_heads()
        assert heads["feat/x"] == sha_x
        assert heads["feat/y"] == sha_y
        assert "master" in heads

    def test_operator_local_push_bypasses_hooks(self, gate_env, tmp_path: Path) -> None:
        """Pushes not carrying the server-injected config see no hooks."""
        gate, _hooks_dir, work = gate_env
        _git(work, "checkout", "-q", "-b", "feat/x")
        _commit(work, "a.txt", "1\n")
        _git(work, "push", "-q", "origin", "HEAD:refs/heads/feat/x")
        _git(work, "reset", "-q", "--hard", "HEAD~1")
        _commit(work, "a.txt", "rewritten\n")
        _git(work, "push", "-q", "--force", "origin", "HEAD:refs/heads/feat/x")

        assert _backups(gate) == []
        assert not (gate._gate_path / PUSH_MARKER_FILENAME).exists()


# ---------------------------------------------------------------------------
# Hidden namespace stays agent-proof
# ---------------------------------------------------------------------------


class TestBackupsStayHidden:
    """Agents can neither see nor forge the backup namespace."""

    def test_backup_refs_not_advertised_to_clones(self, gate_env, tmp_path: Path) -> None:
        gate, hooks_dir, work = gate_env
        _git(work, "checkout", "-q", "-b", "feat/x")
        _commit(work, "a.txt", "1\n")
        _hooked_push(work, hooks_dir, "HEAD:refs/heads/feat/x")
        _git(work, "reset", "-q", "--hard", "HEAD~1")
        _commit(work, "a.txt", "rewritten\n")
        _hooked_push(work, hooks_dir, "HEAD:refs/heads/feat/x", force=True)
        assert len(_backups(gate)) == 1

        advertised = _git(work, "ls-remote", "origin")
        assert "refs/terok/" not in advertised
        assert _read_refs(gate._gate_path, "refs/terok/backup/") != {}
