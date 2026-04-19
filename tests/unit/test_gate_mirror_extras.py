# SPDX-FileCopyrightText: 2026 Jiri Vyskocil
# SPDX-License-Identifier: Apache-2.0

"""Tests for GitGate sync, compare_vs_upstream, last_commit, and helpers."""

from __future__ import annotations

import subprocess
from pathlib import Path
from unittest.mock import patch

import pytest

from terok_sandbox.gate.mirror import (
    GateStalenessInfo,
    GitGate,
    _clone_gate_mirror,  # noqa: PLC2701
    _count_commits_range,  # noqa: PLC2701
    _get_gate_branch_head,  # noqa: PLC2701
    _get_upstream_head,  # noqa: PLC2701
)


def _proc(rc: int = 0, stdout: str = "", stderr: str = "") -> subprocess.CompletedProcess[str]:
    return subprocess.CompletedProcess(args=[], returncode=rc, stdout=stdout, stderr=stderr)


# ---------------------------------------------------------------------------
# GitGate.sync — guards and validation
# ---------------------------------------------------------------------------


class TestGitGateSyncGuards:
    """sync() validates inputs before touching the filesystem."""

    def test_no_upstream_url_raises_systemexit(self, tmp_path: Path) -> None:
        gate = GitGate(scope="proj", gate_path=tmp_path / "g.git", upstream_url=None)
        with pytest.raises(SystemExit, match="upstream_url"):
            gate.sync()

    def test_validate_gate_callback_invoked_before_clone(self, tmp_path: Path) -> None:
        """The injected validation callback fires before any subprocess work."""
        called: list[str] = []
        gate = GitGate(
            scope="proj",
            gate_path=tmp_path / "g.git",
            upstream_url="git@example.com:x/y.git",
            use_personal_ssh=True,  # bypass vault-socket lookup for this unit test
            validate_gate_fn=lambda scope: called.append(scope),
        )
        # Callback must run; the clone will be mocked into a no-op so the
        # subsequent sync logic doesn't matter for this test.
        with (
            patch("terok_sandbox.gate.mirror._clone_gate_mirror"),
            patch.object(
                gate,
                "sync_branches",
                return_value={"success": True, "updated_branches": [], "errors": []},
            ),
        ):
            gate.sync()
        assert called == ["proj"]


# ---------------------------------------------------------------------------
# GitGate.sync_branches — error paths
# ---------------------------------------------------------------------------


class TestSyncBranchesErrorPaths:
    """sync_branches should not crash on remote-update failures."""

    def test_missing_gate_dir_returns_error(self, tmp_path: Path) -> None:
        gate = GitGate(scope="p", gate_path=tmp_path / "missing.git")
        result = gate.sync_branches()
        assert result["success"] is False
        assert "Gate not initialized" in result["errors"][0]

    def test_remote_update_nonzero_recorded_in_errors(self, tmp_path: Path) -> None:
        gate_dir = tmp_path / "g.git"
        gate_dir.mkdir()
        gate = GitGate(scope="p", gate_path=gate_dir)
        with patch(
            "terok_sandbox.gate.mirror.subprocess.run",
            return_value=_proc(rc=1, stderr="auth failed"),
        ):
            result = gate.sync_branches()
        assert result["success"] is False
        assert "remote update failed" in result["errors"][0]
        assert "auth failed" in result["errors"][0]

    def test_timeout_recorded(self, tmp_path: Path) -> None:
        gate_dir = tmp_path / "g.git"
        gate_dir.mkdir()
        gate = GitGate(scope="p", gate_path=gate_dir)
        with patch(
            "terok_sandbox.gate.mirror.subprocess.run",
            side_effect=subprocess.TimeoutExpired("git", 1),
        ):
            result = gate.sync_branches()
        assert result["success"] is False
        assert "timed out" in result["errors"][0].lower()

    def test_unexpected_exception_recorded(self, tmp_path: Path) -> None:
        gate_dir = tmp_path / "g.git"
        gate_dir.mkdir()
        gate = GitGate(scope="p", gate_path=gate_dir)
        with patch(
            "terok_sandbox.gate.mirror.subprocess.run",
            side_effect=OSError("disk full"),
        ):
            result = gate.sync_branches()
        assert result["success"] is False
        assert "disk full" in result["errors"][0]


# ---------------------------------------------------------------------------
# GitGate.compare_vs_upstream
# ---------------------------------------------------------------------------


class TestCompareVsUpstream:
    """compare_vs_upstream returns a GateStalenessInfo for every branch."""

    def test_no_branch_configured_returns_error_info(self, tmp_path: Path) -> None:
        gate = GitGate(scope="p", gate_path=tmp_path / "g.git")
        info = gate.compare_vs_upstream()
        assert isinstance(info, GateStalenessInfo)
        assert info.branch is None
        assert info.error == "No branch configured"

    def test_uninitialised_gate_returns_error_info(self, tmp_path: Path) -> None:
        gate = GitGate(scope="p", gate_path=tmp_path / "missing.git", default_branch="main")
        with patch("terok_sandbox.gate.mirror._get_gate_branch_head", return_value=None):
            info = gate.compare_vs_upstream()
        assert info.error == "Gate not initialized"
        assert info.gate_head is None

    def test_no_upstream_url_returns_error_info(self, tmp_path: Path) -> None:
        gate = GitGate(scope="p", gate_path=tmp_path / "g.git", default_branch="main")
        with patch("terok_sandbox.gate.mirror._get_gate_branch_head", return_value="abc123"):
            info = gate.compare_vs_upstream()
        assert info.error == "No upstream URL configured"
        assert info.gate_head == "abc123"

    def test_upstream_unreachable_returns_error_info(self, tmp_path: Path) -> None:
        gate = GitGate(
            scope="p",
            gate_path=tmp_path / "g.git",
            default_branch="main",
            upstream_url="git@x.com:a/b.git",
            use_personal_ssh=True,
        )
        with (
            patch("terok_sandbox.gate.mirror._get_gate_branch_head", return_value="abc"),
            patch("terok_sandbox.gate.mirror._get_upstream_head", return_value=None),
        ):
            info = gate.compare_vs_upstream()
        assert info.error == "Could not reach upstream"

    def test_in_sync_returns_zero_counts(self, tmp_path: Path) -> None:
        gate = GitGate(
            scope="p",
            gate_path=tmp_path / "g.git",
            default_branch="main",
            upstream_url="git@x.com:a/b.git",
            use_personal_ssh=True,
        )
        with (
            patch("terok_sandbox.gate.mirror._get_gate_branch_head", return_value="same"),
            patch(
                "terok_sandbox.gate.mirror._get_upstream_head",
                return_value={
                    "commit_hash": "same",
                    "ref_name": "refs/heads/main",
                    "upstream_url": "x",
                },
            ),
        ):
            info = gate.compare_vs_upstream()
        assert info.is_stale is False
        assert info.commits_behind == 0
        assert info.commits_ahead == 0
        assert info.error is None

    def test_stale_invokes_count_commits(self, tmp_path: Path) -> None:
        gate = GitGate(
            scope="p",
            gate_path=tmp_path / "g.git",
            default_branch="main",
            upstream_url="git@x.com:a/b.git",
            use_personal_ssh=True,
        )
        with (
            patch("terok_sandbox.gate.mirror._get_gate_branch_head", return_value="old"),
            patch(
                "terok_sandbox.gate.mirror._get_upstream_head",
                return_value={"commit_hash": "new", "ref_name": "r", "upstream_url": "u"},
            ),
            patch(
                "terok_sandbox.gate.mirror._count_commits_range",
                side_effect=[3, 1],  # behind=3, ahead=1
            ) as count,
        ):
            info = gate.compare_vs_upstream(branch="feature")
        assert info.is_stale is True
        assert info.commits_behind == 3
        assert info.commits_ahead == 1
        assert info.branch == "feature"
        assert count.call_count == 2


# ---------------------------------------------------------------------------
# GitGate.last_commit
# ---------------------------------------------------------------------------


class TestLastCommit:
    """last_commit parses git-log NUL-separated output or returns None."""

    def test_returns_none_when_gate_missing(self, tmp_path: Path) -> None:
        gate = GitGate(scope="p", gate_path=tmp_path / "missing.git")
        assert gate.last_commit() is None

    def test_returns_none_on_git_failure(self, tmp_path: Path) -> None:
        gate_dir = tmp_path / "g.git"
        gate_dir.mkdir()
        gate = GitGate(scope="p", gate_path=gate_dir, default_branch="main")
        with patch(
            "terok_sandbox.gate.mirror.subprocess.run",
            return_value=_proc(rc=1, stderr="bad ref"),
        ):
            assert gate.last_commit() is None

    def test_parses_four_field_output(self, tmp_path: Path) -> None:
        gate_dir = tmp_path / "g.git"
        gate_dir.mkdir()
        gate = GitGate(scope="p", gate_path=gate_dir, default_branch="main")
        out = "abc123\x002026-04-18 10:00:00 +0000\x00Alice\x00Initial commit"
        with patch(
            "terok_sandbox.gate.mirror.subprocess.run",
            return_value=_proc(rc=0, stdout=out),
        ):
            commit = gate.last_commit()
        assert commit is not None
        assert commit["commit_hash"] == "abc123"
        assert commit["commit_author"] == "Alice"
        assert commit["commit_message"] == "Initial commit"

    def test_falls_back_to_HEAD_when_branch_ref_missing(self, tmp_path: Path) -> None:
        """If the configured branch ref is missing, retry against HEAD."""
        gate_dir = tmp_path / "g.git"
        gate_dir.mkdir()
        gate = GitGate(scope="p", gate_path=gate_dir, default_branch="main")
        out = "h\x00d\x00a\x00m"
        # call_args_list captures the cmd list by reference, so the source's
        # in-place mutation of cmd[5] would clobber the first call's snapshot.
        # Inspect cmd[5] inside the side_effect callable instead.
        observed: list[str] = []
        responses = iter([_proc(rc=128), _proc(rc=0, stdout=out)])

        def fake_run(cmd, **_kwargs):
            observed.append(cmd[5])
            return next(responses)

        with patch("terok_sandbox.gate.mirror.subprocess.run", side_effect=fake_run):
            assert gate.last_commit() is not None
        assert observed == ["refs/heads/main", "HEAD"]

    def test_malformed_stdout_returns_none(self, tmp_path: Path) -> None:
        gate_dir = tmp_path / "g.git"
        gate_dir.mkdir()
        gate = GitGate(scope="p", gate_path=gate_dir, default_branch="main")
        with patch(
            "terok_sandbox.gate.mirror.subprocess.run",
            return_value=_proc(rc=0, stdout="only-one-field"),
        ):
            assert gate.last_commit() is None

    def test_swallows_unexpected_exception(self, tmp_path: Path) -> None:
        gate_dir = tmp_path / "g.git"
        gate_dir.mkdir()
        gate = GitGate(scope="p", gate_path=gate_dir)
        with patch("terok_sandbox.gate.mirror.subprocess.run", side_effect=OSError("boom")):
            assert gate.last_commit() is None


# ---------------------------------------------------------------------------
# Module-level helpers
# ---------------------------------------------------------------------------


class TestCloneGateMirrorErrors:
    """_clone_gate_mirror translates subprocess failures into SystemExit."""

    def test_git_missing_raises_systemexit(self, tmp_path: Path) -> None:
        with patch("terok_sandbox.gate.mirror.subprocess.run", side_effect=FileNotFoundError):
            with pytest.raises(SystemExit, match="git not found"):
                _clone_gate_mirror("git@x:y/z.git", tmp_path / "g.git", env={})

    def test_clone_failure_raises_systemexit(self, tmp_path: Path) -> None:
        with patch(
            "terok_sandbox.gate.mirror.subprocess.run",
            side_effect=subprocess.CalledProcessError(128, "git", stderr=b"boom"),
        ):
            with pytest.raises(SystemExit, match="git clone --mirror failed"):
                _clone_gate_mirror("git@x:y/z.git", tmp_path / "g.git", env={})


class TestGetUpstreamHead:
    """_get_upstream_head parses the first ls-remote line or returns None."""

    def test_parses_tab_separated_response(self) -> None:
        line = "abc123def456\trefs/heads/main\n"
        with patch(
            "terok_sandbox.gate.mirror.subprocess.run",
            return_value=_proc(rc=0, stdout=line),
        ):
            info = _get_upstream_head("git@x:y/z.git", "main", env={})
        assert info == {
            "commit_hash": "abc123def456",
            "ref_name": "refs/heads/main",
            "upstream_url": "git@x:y/z.git",
        }

    def test_empty_response_returns_none(self) -> None:
        with patch(
            "terok_sandbox.gate.mirror.subprocess.run",
            return_value=_proc(rc=0, stdout=""),
        ):
            assert _get_upstream_head("u", "main", env={}) is None

    def test_nonzero_exit_returns_none(self) -> None:
        with patch(
            "terok_sandbox.gate.mirror.subprocess.run",
            return_value=_proc(rc=128, stderr="auth failed"),
        ):
            assert _get_upstream_head("u", "main", env={}) is None

    def test_malformed_line_returns_none(self) -> None:
        with patch(
            "terok_sandbox.gate.mirror.subprocess.run",
            return_value=_proc(rc=0, stdout="no-tab-here\n"),
        ):
            assert _get_upstream_head("u", "main", env={}) is None

    def test_timeout_returns_none(self) -> None:
        with patch(
            "terok_sandbox.gate.mirror.subprocess.run",
            side_effect=subprocess.TimeoutExpired("git", 30),
        ):
            assert _get_upstream_head("u", "main", env={}) is None


class TestGetGateBranchHead:
    """_get_gate_branch_head returns commit hash or None."""

    def test_returns_hash_on_success(self, tmp_path: Path) -> None:
        gate = tmp_path / "g.git"
        gate.mkdir()
        with patch(
            "terok_sandbox.gate.mirror.subprocess.run",
            return_value=_proc(rc=0, stdout="abc123\n"),
        ):
            assert _get_gate_branch_head(gate, "main", env={}) == "abc123"

    def test_missing_dir_returns_none(self, tmp_path: Path) -> None:
        assert _get_gate_branch_head(tmp_path / "missing.git", "main", env={}) is None

    def test_nonzero_returncode_returns_none(self, tmp_path: Path) -> None:
        gate = tmp_path / "g.git"
        gate.mkdir()
        with patch(
            "terok_sandbox.gate.mirror.subprocess.run",
            return_value=_proc(rc=128),
        ):
            assert _get_gate_branch_head(gate, "main", env={}) is None

    def test_filenotfound_returns_none(self, tmp_path: Path) -> None:
        gate = tmp_path / "g.git"
        gate.mkdir()
        with patch("terok_sandbox.gate.mirror.subprocess.run", side_effect=FileNotFoundError):
            assert _get_gate_branch_head(gate, "main", env={}) is None


class TestCountCommitsRange:
    """_count_commits_range parses git rev-list --count."""

    def test_returns_int_on_success(self, tmp_path: Path) -> None:
        gate = tmp_path / "g.git"
        gate.mkdir()
        with patch(
            "terok_sandbox.gate.mirror.subprocess.run",
            return_value=_proc(rc=0, stdout="42\n"),
        ):
            assert _count_commits_range(gate, "a", "b", env={}) == 42

    def test_failure_returns_none(self, tmp_path: Path) -> None:
        gate = tmp_path / "g.git"
        gate.mkdir()
        with patch(
            "terok_sandbox.gate.mirror.subprocess.run",
            return_value=_proc(rc=128),
        ):
            assert _count_commits_range(gate, "a", "b", env={}) is None
