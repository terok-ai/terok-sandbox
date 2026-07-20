# SPDX-FileCopyrightText: 2026 Jiri Vyskocil
# SPDX-License-Identifier: Apache-2.0

"""Tests for the gate's safe sync model: snapshot, pending ops, attic, backups.

Everything here runs against real (tiny) git repositories: the properties
under test are git-level ref semantics — what survives a sync, what gets
proposed instead of destroyed — and mocks would just restate the
implementation.  The scenario driving the whole design: an agent pushes a
branch to the gate that upstream has never seen, and no amount of syncing
may delete or overwrite it without the operator saying so.
"""

from __future__ import annotations

import subprocess
from datetime import UTC, datetime, timedelta
from pathlib import Path

import pytest

from terok_sandbox.gate.mirror import (
    _ATTIC_PREFIX,
    _BACKUP_STAMP_FORMAT,
    _FETCH_REFSPEC,
    _SNAPSHOT_PREFIX,
    GitGate,
    PendingOp,
    _read_refs,
)

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


def _make_upstream(tmp_path: Path) -> Path:
    """Create a one-commit upstream repo on ``master``."""
    upstream = tmp_path / "upstream"
    subprocess.run(["git", "init", "-b", "master", str(upstream)], check=True, capture_output=True)
    _commit(upstream, "file.txt", "v1\n")
    return upstream


def _make_gate(tmp_path: Path, upstream: Path, **kwargs: object) -> GitGate:
    """Build a GitGate for *upstream* under tmp (no cache unless asked)."""
    return GitGate(
        scope="proj",
        gate_path=tmp_path / "gate" / "proj.git",
        upstream_url=str(upstream),
        **kwargs,  # type: ignore[arg-type]
    )


def _agent_push(tmp_path: Path, gate_dir: Path, branch: str, *, start: str = "master") -> str:
    """Push one agent commit to *branch* on the gate; return its sha.

    Clones the gate the way a container would, so the push exercises the
    same receive path (minus HTTP) that real agent work uses.
    """
    name = branch.removeprefix("refs/heads/")
    work = tmp_path / f"work-{name.replace('/', '-')}-{len(list(tmp_path.iterdir()))}"
    subprocess.run(
        ["git", "clone", "-q", str(gate_dir), str(work)], check=True, capture_output=True
    )
    _git(work, "checkout", "-q", start)
    if name != start:
        _git(work, "checkout", "-q", "-b", name)
    sha = _commit(work, "agent.txt", f"work on {branch}\n")
    _git(work, "push", "-q", "origin", f"HEAD:refs/heads/{name}")
    return sha


def _heads(gate_dir: Path) -> dict[str, str]:
    return _read_refs(gate_dir, "refs/heads/")


# ---------------------------------------------------------------------------
# The regression that started all this
# ---------------------------------------------------------------------------


class TestGateOnlyWorkSurvives:
    """Branches that exist only on the gate are never deleted or moved."""

    def test_gate_only_branch_survives_syncs(self, tmp_path: Path) -> None:
        upstream = _make_upstream(tmp_path)
        gate = _make_gate(tmp_path, upstream)
        assert gate.sync()["success"] is True

        wip_sha = _agent_push(tmp_path, gate._gate_path, "feat/agent-wip")
        _commit(upstream, "file.txt", "v2\n")

        result = gate.sync()

        assert result["success"] is True
        assert "feat/agent-wip" in result["gate_only_branches"]
        assert result["pending"] == []
        assert _heads(gate._gate_path)["feat/agent-wip"] == wip_sha
        # master still followed upstream — safety is not staleness
        assert [op["kind"] for op in result["applied"]] == ["fast_forward"]

    def test_repeated_syncs_stay_quiet(self, tmp_path: Path) -> None:
        """A gate-only branch is reported, not re-proposed, sync after sync."""
        upstream = _make_upstream(tmp_path)
        gate = _make_gate(tmp_path, upstream)
        gate.sync()
        _agent_push(tmp_path, gate._gate_path, "feat/agent-wip")

        for _ in range(3):
            result = gate.sync()
            assert result["pending"] == []
            assert "feat/agent-wip" in result["gate_only_branches"]
        assert "feat/agent-wip" in _heads(gate._gate_path)


class TestSafeOpsApplied:
    """Creates and fast-forwards happen automatically, with real shas reported."""

    def test_create_and_fast_forward(self, tmp_path: Path) -> None:
        upstream = _make_upstream(tmp_path)
        gate = _make_gate(tmp_path, upstream)
        gate.sync()

        old_master = _git(upstream, "rev-parse", "master")
        new_master = _commit(upstream, "file.txt", "v2\n")
        _git(upstream, "branch", "feature")
        result = gate.sync()

        assert result["success"] is True
        by_branch = {op["branch"]: op for op in result["applied"]}
        assert by_branch["feature"]["kind"] == "create"
        assert by_branch["feature"]["old_sha"] is None
        assert by_branch["master"] == {
            "branch": "master",
            "kind": "fast_forward",
            "old_sha": old_master,
            "new_sha": new_master,
        }
        assert _heads(gate._gate_path)["master"] == new_master

    def test_up_to_date_sync_reports_nothing(self, tmp_path: Path) -> None:
        upstream = _make_upstream(tmp_path)
        gate = _make_gate(tmp_path, upstream)
        gate.sync()

        result = gate.sync()

        assert result["applied"] == [] and result["pending"] == []


class TestPendingDeletes:
    """Upstream deletions become confirmable proposals, never silent removals."""

    @pytest.fixture()
    def synced(self, tmp_path: Path) -> tuple[Path, GitGate]:
        upstream = _make_upstream(tmp_path)
        _git(upstream, "branch", "feature")
        gate = _make_gate(tmp_path, upstream)
        gate.sync()
        return upstream, gate

    def test_squash_merge_cleanup_is_lossless_pending(self, synced: tuple[Path, GitGate]) -> None:
        upstream, gate = synced
        feature_sha = _heads(gate._gate_path)["feature"]
        _git(upstream, "branch", "-D", "feature")

        result = gate.sync()

        assert result["success"] is True
        (op,) = result["pending"]
        assert op["kind"] == "delete" and op["reason"] == "upstream_delete"
        assert op["lossless"] is True and op["gate_only_commits"] == 0
        assert op["gate_sha"] == feature_sha
        assert _heads(gate._gate_path)["feature"] == feature_sha  # still there

    def test_delete_with_agent_commits_is_lossy(
        self, tmp_path: Path, synced: tuple[Path, GitGate]
    ) -> None:
        upstream, gate = synced
        _agent_push(tmp_path, gate._gate_path, "feature", start="feature")
        _git(upstream, "branch", "-D", "feature")

        (op,) = gate.sync()["pending"]

        assert op["kind"] == "delete"
        assert op["lossless"] is False
        assert op["gate_only_commits"] == 1

    def test_pending_delete_survives_further_syncs(self, synced: tuple[Path, GitGate]) -> None:
        """The attic keeps provenance across syncs — the proposal never degrades."""
        upstream, gate = synced
        _git(upstream, "branch", "-D", "feature")
        gate.sync()

        result = gate.sync()  # snapshot no longer has 'feature'; attic must

        (op,) = result["pending"]
        assert op["reason"] == "upstream_delete" and op["lossless"] is True
        assert "feature" not in result["gate_only_branches"]

    def test_branch_reappearing_upstream_clears_the_proposal(
        self, synced: tuple[Path, GitGate]
    ) -> None:
        upstream, gate = synced
        _git(upstream, "branch", "-D", "feature")
        gate.sync()
        _git(upstream, "branch", "feature", "master")

        result = gate.sync()

        assert result["pending"] == []
        assert _read_refs(gate._gate_path, _ATTIC_PREFIX) == {}


class TestPendingForces:
    """Upstream rewrites become confirmable proposals with honest loss labels."""

    @pytest.fixture()
    def rewritten(self, tmp_path: Path) -> tuple[Path, GitGate, str]:
        upstream = _make_upstream(tmp_path)
        _git(upstream, "checkout", "-q", "-b", "dev")
        _commit(upstream, "dev.txt", "d1\n")
        _git(upstream, "checkout", "-q", "master")
        gate = _make_gate(tmp_path, upstream)
        gate.sync()
        synced_dev = _heads(gate._gate_path)["dev"]
        _git(upstream, "checkout", "-q", "dev")
        _git(upstream, "commit", "--amend", "-m", "rewritten")
        _git(upstream, "checkout", "-q", "master")
        return upstream, gate, synced_dev

    def test_clean_gate_copy_is_lossless_pending_force(
        self, rewritten: tuple[Path, GitGate, str]
    ) -> None:
        upstream, gate, synced_dev = rewritten

        result = gate.sync()

        (op,) = result["pending"]
        assert op["kind"] == "force_update" and op["reason"] == "upstream_rewrite"
        assert op["lossless"] is True and op["gate_only_commits"] == 0
        assert op["upstream_sha"] == _git(upstream, "rev-parse", "dev")
        assert _heads(gate._gate_path)["dev"] == synced_dev  # untouched

    def test_lossless_survives_a_second_upstream_rewrite(
        self, rewritten: tuple[Path, GitGate, str]
    ) -> None:
        """First-writer-wins attic: still lossless after upstream rewrites again."""
        upstream, gate, _ = rewritten
        gate.sync()
        _git(upstream, "checkout", "-q", "dev")
        _git(upstream, "commit", "--amend", "-m", "rewritten again")
        _git(upstream, "checkout", "-q", "master")

        (op,) = gate.sync()["pending"]

        assert op["lossless"] is True

    def test_agent_commits_make_it_lossy(
        self, tmp_path: Path, rewritten: tuple[Path, GitGate, str]
    ) -> None:
        _, gate, _ = rewritten
        _agent_push(tmp_path, gate._gate_path, "dev", start="dev")

        (op,) = gate.sync()["pending"]

        assert op["lossless"] is False
        assert op["gate_only_commits"] == 1


class TestApplyPendingOps:
    """Confirmed ops apply exactly as proposed — CAS-guarded and backed up."""

    def _pending_after_delete(self, tmp_path: Path) -> tuple[GitGate, PendingOp]:
        upstream = _make_upstream(tmp_path)
        _git(upstream, "branch", "feature")
        gate = _make_gate(tmp_path, upstream)
        gate.sync()
        _git(upstream, "branch", "-D", "feature")
        (op,) = gate.sync()["pending"]
        return gate, op

    def test_delete_applies_with_backup(self, tmp_path: Path) -> None:
        gate, op = self._pending_after_delete(tmp_path)

        result = gate.apply_pending_ops([op])

        assert result["success"] is True
        assert "feature" not in _heads(gate._gate_path)
        backup_ref = result["backups"]["feature"]
        assert _git(gate._gate_path, "rev-parse", backup_ref) == op["gate_sha"]
        (entry,) = gate.list_backups()
        assert entry["branch"] == "feature" and entry["sha"] == op["gate_sha"]
        # applying resolved the question — nothing pending, attic clean
        assert gate.pending_ops() == []
        assert _read_refs(gate._gate_path, _ATTIC_PREFIX) == {}

    def test_force_update_applies_with_backup(self, tmp_path: Path) -> None:
        upstream = _make_upstream(tmp_path)
        _git(upstream, "checkout", "-q", "-b", "dev")
        _commit(upstream, "dev.txt", "d1\n")
        gate = _make_gate(tmp_path, upstream)
        gate.sync()
        old_dev = _heads(gate._gate_path)["dev"]
        _git(upstream, "commit", "--amend", "-m", "rewritten")
        (op,) = gate.sync()["pending"]

        result = gate.apply_pending_ops([op])

        assert result["success"] is True
        assert _heads(gate._gate_path)["dev"] == op["upstream_sha"]
        assert _git(gate._gate_path, "rev-parse", result["backups"]["dev"]) == old_dev
        assert gate.pending_ops() == []

    def test_moved_branch_refuses_that_op_only(self, tmp_path: Path) -> None:
        """An agent push between proposal and apply wins; the op fails alone."""
        gate, op = self._pending_after_delete(tmp_path)
        pushed = _agent_push(tmp_path, gate._gate_path, "feature", start="feature")

        result = gate.apply_pending_ops([op])

        assert result["success"] is False
        assert "moved since" in result["errors"][0]
        assert _heads(gate._gate_path)["feature"] == pushed
        assert result["backups"] == {} and gate.list_backups() == []

    def test_backups_can_be_opted_out(self, tmp_path: Path) -> None:
        upstream = _make_upstream(tmp_path)
        _git(upstream, "branch", "feature")
        gate = _make_gate(tmp_path, upstream, backups_enabled=False)
        gate.sync()
        _git(upstream, "branch", "-D", "feature")
        (op,) = gate.sync()["pending"]

        result = gate.apply_pending_ops([op])

        assert result["success"] is True
        assert result["backups"] == {} and gate.list_backups() == []
        assert "feature" not in _heads(gate._gate_path)


class TestApplyEdgeCases:
    """Malformed ops and post-apply plumbing failures degrade into errors."""

    def test_force_op_without_upstream_sha_is_refused(self, tmp_path: Path) -> None:
        upstream = _make_upstream(tmp_path)
        gate = _make_gate(tmp_path, upstream)
        gate.sync()
        bogus: PendingOp = {
            "branch": "master",
            "kind": "force_update",
            "reason": "upstream_rewrite",
            "gate_sha": _heads(gate._gate_path)["master"],
            "upstream_sha": None,
            "old_snapshot_sha": None,
            "lossless": False,
            "gate_only_commits": None,
        }

        result = gate.apply_pending_ops([bogus])

        assert result["success"] is False
        assert "carries no upstream sha" in result["errors"][0]
        assert gate.list_backups() == []

    def test_head_alignment_failure_is_reported(self, tmp_path: Path) -> None:
        from unittest.mock import patch

        upstream = _make_upstream(tmp_path)
        _git(upstream, "branch", "feature")
        gate = _make_gate(tmp_path, upstream)
        gate.sync()
        _git(upstream, "branch", "-D", "feature")
        (op,) = gate.sync()["pending"]

        with patch.object(gate, "_align_gate_head", return_value="HEAD broke"):
            result = gate.apply_pending_ops([op])

        assert result["applied"] and result["errors"] == ["HEAD broke"]
        assert result["success"] is False

    def test_cache_refresh_failure_after_apply_is_reported(self, tmp_path: Path) -> None:
        from unittest.mock import patch

        upstream = _make_upstream(tmp_path)
        _git(upstream, "branch", "feature")
        gate = _make_gate(tmp_path, upstream, clone_cache_base=tmp_path / "cache")
        gate.sync()
        _git(upstream, "branch", "-D", "feature")
        (op,) = gate.sync()["pending"]

        with patch.object(gate, "_refresh_clone_cache", return_value="disk full"):
            result = gate.apply_pending_ops([op])

        assert result["applied"]
        assert result["errors"] == ["clone cache refresh failed: disk full"]

    def test_empty_gate_keeps_unborn_head_for_advertised_default(self, tmp_path: Path) -> None:
        """An empty gate whose upstream already advertises a default stays unborn."""
        from unittest.mock import patch

        upstream = tmp_path / "empty-upstream"
        subprocess.run(
            ["git", "init", "--bare", "-b", "master", str(upstream)],
            check=True,
            capture_output=True,
        )
        gate = _make_gate(tmp_path, upstream)
        gate.sync()

        with patch(
            "terok_sandbox.gate.mirror._query_upstream_head_ref",
            return_value="refs/heads/master",
        ):
            assert gate._align_gate_head(env={}) is None

    def test_safe_op_cas_race_becomes_a_note(self, tmp_path: Path) -> None:
        """A stale CAS guard fails a safe op into a note, not a clobber."""
        upstream = _make_upstream(tmp_path)
        gate = _make_gate(tmp_path, upstream)
        gate.sync()
        real_master = _heads(gate._gate_path)["master"]

        error = gate._apply_ref_cas(
            {"branch": "master", "kind": "fast_forward", "old_sha": "1" * 40, "new_sha": "2" * 40}
        )

        assert error is not None and "ref moved during sync" in error
        assert _heads(gate._gate_path)["master"] == real_master

    def test_hand_deleted_branch_clears_its_attic_entry(self, tmp_path: Path) -> None:
        """Attic residue for a branch with no head left is swept by the next sync."""
        upstream = _make_upstream(tmp_path)
        _git(upstream, "branch", "feature")
        gate = _make_gate(tmp_path, upstream)
        gate.sync()
        _git(upstream, "branch", "-D", "feature")
        gate.sync()  # pending delete + attic entry
        _git(gate._gate_path, "update-ref", "-d", "refs/heads/feature")

        result = gate.sync()

        assert result["pending"] == []
        assert _read_refs(gate._gate_path, _ATTIC_PREFIX) == {}

    def test_malformed_backup_ref_names_are_ignored(self, tmp_path: Path) -> None:
        upstream = _make_upstream(tmp_path)
        gate = _make_gate(tmp_path, upstream)
        gate.sync()
        sha = _heads(gate._gate_path)["master"]
        _git(gate._gate_path, "update-ref", "refs/terok/backup/oddball", sha)

        assert gate.list_backups() == []
        assert gate.prune_backups(older_than_days=1) == []


class TestBackupRetention:
    """Backups expire on the ref-name clock, and only there."""

    def _gate_with_backup(self, tmp_path: Path, **kwargs: object) -> GitGate:
        upstream = _make_upstream(tmp_path)
        _git(upstream, "branch", "feature")
        gate = _make_gate(tmp_path, upstream, **kwargs)
        gate.sync()
        _git(upstream, "branch", "-D", "feature")
        (op,) = gate.sync()["pending"]
        gate.apply_pending_ops([op])
        return gate

    def _age_backup(self, gate: GitGate, days: int) -> None:
        """Rewrite the backup ref name as if it were taken *days* ago."""
        (entry,) = gate.list_backups()
        stamp = (datetime.now(UTC) - timedelta(days=days)).strftime(_BACKUP_STAMP_FORMAT)
        old_leaf = entry["ref"].rsplit("/", 1)[-1]
        aged = entry["ref"].replace(old_leaf, f"{stamp}-{entry['sha'][:12]}")
        _git(gate._gate_path, "update-ref", aged, entry["sha"])
        _git(gate._gate_path, "update-ref", "-d", entry["ref"])

    def test_fresh_backups_survive_prune(self, tmp_path: Path) -> None:
        gate = self._gate_with_backup(tmp_path)
        assert gate.prune_backups() == []
        assert len(gate.list_backups()) == 1

    def test_expired_backups_are_pruned_by_sync(self, tmp_path: Path) -> None:
        gate = self._gate_with_backup(tmp_path)
        self._age_backup(gate, days=45)

        result = gate.sync()

        assert result["success"] is True
        assert any("backup" in note for note in result["notes"])
        assert gate.list_backups() == []

    def test_retention_zero_keeps_forever(self, tmp_path: Path) -> None:
        gate = self._gate_with_backup(tmp_path, backup_retention_days=0)
        self._age_backup(gate, days=3650)
        assert gate.prune_backups() == []
        assert len(gate.list_backups()) == 1


class TestPendingOpsOffline:
    """pending_ops() reproduces the sync's proposals without any fetch."""

    def test_matches_sync_report(self, tmp_path: Path) -> None:
        upstream = _make_upstream(tmp_path)
        _git(upstream, "branch", "feature")
        _git(upstream, "checkout", "-q", "-b", "dev")
        _commit(upstream, "dev.txt", "d1\n")
        _git(upstream, "checkout", "-q", "master")
        gate = _make_gate(tmp_path, upstream)
        gate.sync()
        _git(upstream, "branch", "-D", "feature")
        _git(upstream, "checkout", "-q", "dev")
        _git(upstream, "commit", "--amend", "-m", "rewritten")
        _git(upstream, "checkout", "-q", "master")

        synced = gate.sync()["pending"]
        offline = gate.pending_ops()

        assert offline == synced
        assert {op["kind"] for op in offline} == {"delete", "force_update"}

    def test_missing_gate_is_empty(self, tmp_path: Path) -> None:
        gate = GitGate(scope="p", gate_path=tmp_path / "missing.git")
        assert gate.pending_ops() == []


class TestSelectiveSync:
    """A branch allowlist syncs those branches and blinds sync to the rest."""

    def test_only_requested_branch_moves(self, tmp_path: Path) -> None:
        upstream = _make_upstream(tmp_path)
        _git(upstream, "branch", "other")
        gate = _make_gate(tmp_path, upstream)
        gate.sync()
        stale_other = _heads(gate._gate_path)["other"]
        _commit(upstream, "file.txt", "v2\n")
        _git(upstream, "branch", "-f", "other", "master")

        result = gate.sync(branches=["master"])

        assert result["success"] is True
        assert [op["branch"] for op in result["applied"]] == ["master"]
        assert _heads(gate._gate_path)["other"] == stale_other

    def test_deleted_branch_in_selection_goes_pending(self, tmp_path: Path) -> None:
        upstream = _make_upstream(tmp_path)
        _git(upstream, "branch", "feature")
        gate = _make_gate(tmp_path, upstream)
        gate.sync()
        _git(upstream, "branch", "-D", "feature")

        result = gate.sync(branches=["feature", "master"])

        (op,) = result["pending"]
        assert op["branch"] == "feature" and op["kind"] == "delete"
        assert op["lossless"] is True
        assert "feature" in _heads(gate._gate_path)

    def test_selection_of_only_deleted_branches_skips_the_fetch(self, tmp_path: Path) -> None:
        upstream = _make_upstream(tmp_path)
        _git(upstream, "branch", "feature")
        gate = _make_gate(tmp_path, upstream)
        gate.sync()
        _git(upstream, "branch", "-D", "feature")

        result = gate.sync(branches=["feature"])

        assert result["success"] is True
        (op,) = result["pending"]
        assert op["branch"] == "feature" and op["kind"] == "delete"

    def test_deletion_outside_selection_is_invisible(self, tmp_path: Path) -> None:
        upstream = _make_upstream(tmp_path)
        _git(upstream, "branch", "feature")
        gate = _make_gate(tmp_path, upstream)
        gate.sync()
        _git(upstream, "branch", "-D", "feature")

        result = gate.sync(branches=["master"])

        assert result["pending"] == []
        assert "feature" in _heads(gate._gate_path)


class TestMovedTags:
    """A force-moved upstream tag must not wedge sync into permanent failure."""

    def test_moved_tag_is_a_note_not_a_failure(self, tmp_path: Path) -> None:
        upstream = _make_upstream(tmp_path)
        _git(upstream, "tag", "v1")
        gate = _make_gate(tmp_path, upstream)
        gate.sync()
        _commit(upstream, "file.txt", "v2\n")
        _git(upstream, "tag", "-f", "v1")

        for _ in range(2):  # the rejection repeats on every fetch — so must success
            result = gate.sync()
            assert result["success"] is True
            assert any("v1" in note for note in result["notes"])

    def test_deleted_upstream_tag_is_kept(self, tmp_path: Path) -> None:
        upstream = _make_upstream(tmp_path)
        _git(upstream, "tag", "v1")
        gate = _make_gate(tmp_path, upstream)
        gate.sync()
        _git(upstream, "tag", "-d", "v1")

        assert gate.sync()["success"] is True
        assert "v1" in _read_refs(gate._gate_path, "refs/tags/")


class TestMigration:
    """Old mirror-configured gates are normalised once, destroying nothing."""

    def _mirror_gate(self, tmp_path: Path) -> tuple[Path, Path]:
        upstream = _make_upstream(tmp_path)
        _git(upstream, "branch", "feature")
        gate_dir = tmp_path / "gate" / "proj.git"
        subprocess.run(
            ["git", "clone", "--mirror", str(upstream), str(gate_dir)],
            check=True,
            capture_output=True,
        )
        return upstream, gate_dir

    def test_first_sync_migrates_config_and_keeps_branches(self, tmp_path: Path) -> None:
        upstream, gate_dir = self._mirror_gate(tmp_path)
        _git(upstream, "branch", "-D", "feature")  # squash-merge style cleanup
        gate = GitGate(scope="proj", gate_path=gate_dir, upstream_url=str(upstream))

        result = gate.sync()

        assert result["success"] is True and result["migrated"] is True
        assert "feature" in _heads(gate_dir)  # the old code deleted it here
        (op,) = result["pending"]
        assert op["branch"] == "feature" and op["reason"] == "unknown_provenance"
        assert op["lossless"] is False and op["gate_only_commits"] is None
        assert _git(gate_dir, "config", "--get-all", "remote.origin.fetch") == _FETCH_REFSPEC
        assert _git(gate_dir, "config", "--get-all", "transfer.hideRefs") == "refs/terok"
        assert _git(gate_dir, "config", "core.logAllRefUpdates") == "always"
        assert (
            subprocess.run(
                ["git", "-C", str(gate_dir), "config", "--get", "remote.origin.mirror"],
                capture_output=True,
            ).returncode
            != 0
        )

    def test_second_sync_treats_unknowns_as_gate_only(self, tmp_path: Path) -> None:
        upstream, gate_dir = self._mirror_gate(tmp_path)
        _git(upstream, "branch", "-D", "feature")
        gate = GitGate(scope="proj", gate_path=gate_dir, upstream_url=str(upstream))
        gate.sync()

        result = gate.sync()

        assert result["migrated"] is False and result["pending"] == []
        assert result["gate_only_branches"] == ["feature"]

    def test_fresh_gate_seeds_snapshot_and_strips_foreign_refs(self, tmp_path: Path) -> None:
        upstream = _make_upstream(tmp_path)
        sha = _git(upstream, "rev-parse", "master")
        _git(upstream, "update-ref", "refs/pull/1/head", sha)
        gate = _make_gate(tmp_path, upstream)

        result = gate.sync()

        assert result["created"] is True and result["migrated"] is False
        gate_dir = gate._gate_path
        assert _read_refs(gate_dir, _SNAPSHOT_PREFIX) == _heads(gate_dir)
        assert _read_refs(gate_dir, "refs/pull/") == {}
        assert result["pending"] == []


class TestHiddenNamespace:
    """Containers can neither see nor write the terok-private refs."""

    def test_push_to_snapshot_namespace_is_rejected(self, tmp_path: Path) -> None:
        upstream = _make_upstream(tmp_path)
        gate = _make_gate(tmp_path, upstream)
        gate.sync()
        work = tmp_path / "work"
        subprocess.run(
            ["git", "clone", "-q", str(gate._gate_path), str(work)],
            check=True,
            capture_output=True,
        )
        push = subprocess.run(
            ["git", "-C", str(work), "push", "origin", f"master:{_SNAPSHOT_PREFIX}master"],
            capture_output=True,
            text=True,
        )
        assert push.returncode != 0
        assert "hidden ref" in push.stderr

    def test_terok_refs_are_not_advertised(self, tmp_path: Path) -> None:
        upstream = _make_upstream(tmp_path)
        gate = _make_gate(tmp_path, upstream)
        gate.sync()
        listed = subprocess.run(
            ["git", "ls-remote", str(gate._gate_path)],
            check=True,
            capture_output=True,
            text=True,
        ).stdout
        assert "refs/terok" not in listed
        assert "refs/heads/master" in listed


class TestHeadAlignment:
    """The gate's HEAD follows upstream's default branch without destruction."""

    def test_rename_repoints_head_and_keeps_old_branch(self, tmp_path: Path) -> None:
        upstream = _make_upstream(tmp_path)
        gate = _make_gate(tmp_path, upstream)
        gate.sync()
        _git(upstream, "branch", "-m", "master", "trunk")

        result = gate.sync()

        assert result["success"] is True
        assert _git(gate._gate_path, "symbolic-ref", "HEAD") == "refs/heads/trunk"
        # the old default is a pending delete like any other branch
        (op,) = result["pending"]
        assert op["branch"] == "master" and op["lossless"] is True
        assert "master" in _heads(gate._gate_path)

    def test_head_stays_healthy_after_approving_old_default_delete(self, tmp_path: Path) -> None:
        upstream = _make_upstream(tmp_path)
        gate = _make_gate(tmp_path, upstream)
        gate.sync()
        _git(upstream, "branch", "-m", "master", "trunk")
        (op,) = gate.sync()["pending"]

        assert gate.apply_pending_ops([op])["success"] is True

        assert _git(gate._gate_path, "symbolic-ref", "HEAD") == "refs/heads/trunk"
        assert "master" not in _heads(gate._gate_path)

    def test_healthy_head_kept_until_new_default_lands(self, tmp_path: Path) -> None:
        """A valid HEAD is not swapped for a default the gate doesn't have yet."""
        from unittest.mock import patch

        upstream = _make_upstream(tmp_path)
        gate = _make_gate(tmp_path, upstream)
        gate.sync()

        with patch(
            "terok_sandbox.gate.mirror._query_upstream_head_ref",
            return_value="refs/heads/ghost",
        ):
            assert gate._align_gate_head(env={}) is None

        assert _git(gate._gate_path, "symbolic-ref", "HEAD") == "refs/heads/master"

    def test_empty_upstream_syncs_cleanly(self, tmp_path: Path) -> None:
        upstream = tmp_path / "empty-upstream"
        subprocess.run(
            ["git", "init", "--bare", "-b", "master", str(upstream)],
            check=True,
            capture_output=True,
        )
        gate = _make_gate(tmp_path, upstream)

        for _ in range(2):
            result = gate.sync()
            assert result["success"] is True
            assert result["applied"] == [] and result["pending"] == []
