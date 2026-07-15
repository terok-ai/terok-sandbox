# SPDX-FileCopyrightText: 2026 Jiri Vyskocil
# SPDX-License-Identifier: Apache-2.0

"""Tests for clone-cache acceleration: SandboxConfig property + GitGate refresh."""

from __future__ import annotations

from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from terok_sandbox.config import SandboxConfig
from terok_sandbox.gate.mirror import GitGate
from tests.constants import FAKE_STATE_DIR


class TestSandboxConfigCloneCachePath:
    """Verify the clone_cache_base_path property."""

    def test_derives_from_state_dir(self) -> None:
        cfg = SandboxConfig(state_dir=FAKE_STATE_DIR)
        assert cfg.clone_cache_base_path == FAKE_STATE_DIR / "clone-cache"

    def test_default_uses_default_state_dir(self) -> None:
        cfg = SandboxConfig()
        assert cfg.clone_cache_base_path == cfg.state_dir / "clone-cache"


class TestGitGateCachePath:
    """Verify the cache_path property on GitGate."""

    def test_returns_none_when_no_base(self) -> None:
        gate = GitGate(scope="proj", gate_path="/tmp/terok-testing/gate/proj.git")
        assert gate.cache_path is None

    def test_derives_from_base_and_scope(self, tmp_path: Path) -> None:
        cache_base = tmp_path / "clone-cache"
        gate = GitGate(
            scope="myproj",
            gate_path="/tmp/terok-testing/gate/myproj.git",
            clone_cache_base=cache_base,
        )
        assert gate.cache_path == cache_base / "myproj"


class TestRefreshCloneCache:
    """Verify _refresh_clone_cache creates or updates the cache."""

    def test_creates_cache_via_git_clone(self, tmp_path: Path) -> None:
        """When cache dir doesn't exist, git clone is invoked, then HEAD verified."""
        gate_dir = tmp_path / "gate" / "proj.git"
        gate_dir.mkdir(parents=True)
        cache_base = tmp_path / "clone-cache"

        gate = GitGate(
            scope="proj",
            gate_path=gate_dir,
            clone_cache_base=cache_base,
        )

        with patch("terok_sandbox.gate.mirror.subprocess.run") as mock_run:
            mock_run.return_value.returncode = 0
            result = gate._refresh_clone_cache()

        assert result is None
        # clone → rev-parse --verify HEAD (guards against empty-checkout clones)
        assert mock_run.call_count == 2
        cmd = mock_run.call_args_list[0].args[0]
        assert cmd[0:2] == ["git", "clone"]
        assert cmd[2].startswith("file:///") and str(gate_dir.name) in cmd[2]
        assert str(cache_base / "proj") in cmd
        assert "rev-parse" in mock_run.call_args_list[1].args[0]

    def test_fetches_when_cache_exists(self, tmp_path: Path) -> None:
        """When cache dir already exists, git fetch is used instead of clone."""
        gate_dir = tmp_path / "gate" / "proj.git"
        gate_dir.mkdir(parents=True)
        cache_base = tmp_path / "clone-cache"
        cache_dir = cache_base / "proj"
        cache_dir.mkdir(parents=True)

        gate = GitGate(
            scope="proj",
            gate_path=gate_dir,
            clone_cache_base=cache_base,
        )

        with patch("terok_sandbox.gate.mirror.subprocess.run") as mock_run:
            mock_run.return_value.returncode = 0
            result = gate._refresh_clone_cache()

        assert result is None
        assert mock_run.call_count == 5
        # set-url → fetch → set-head → reset --hard → clean -ffdx
        cmds = [call.args[0] for call in mock_run.call_args_list]
        assert "set-url" in cmds[0]
        assert "fetch" in cmds[1]
        assert "set-head" in cmds[2]
        assert "reset" in cmds[3] and "--hard" in cmds[3]
        assert "clean" in cmds[4] and "-ffdx" in cmds[4]

    def test_refreshes_cache_missing_origin_head(self, tmp_path: Path) -> None:
        """Regression: a cache without ``refs/remotes/origin/HEAD`` refreshes cleanly.

        ``git fetch`` never creates the ref (only ``git clone`` does), so a
        cache that lost it — or whose mirror default branch moved — used to
        fail the ``reset --hard origin/HEAD`` with exit 128.
        """
        import subprocess

        upstream = tmp_path / "upstream"
        subprocess.run(
            ["git", "init", "-b", "main", str(upstream)], check=True, capture_output=True
        )
        (upstream / "file.txt").write_text("v1\n")
        git = ["git", "-C", str(upstream)]
        subprocess.run([*git, "add", "."], check=True, capture_output=True)
        subprocess.run(
            [*git, "-c", "user.name=t", "-c", "user.email=t@t", "commit", "-m", "v1"],
            check=True,
            capture_output=True,
        )

        gate_dir = tmp_path / "gate" / "proj.git"
        subprocess.run(
            ["git", "clone", "--mirror", str(upstream), str(gate_dir)],
            check=True,
            capture_output=True,
        )
        cache_base = tmp_path / "clone-cache"
        gate = GitGate(scope="proj", gate_path=gate_dir, clone_cache_base=cache_base)

        assert gate._refresh_clone_cache() is None  # creates the cache via clone
        cache_dir = cache_base / "proj"
        subprocess.run(
            ["git", "-C", str(cache_dir), "remote", "set-head", "origin", "--delete"],
            check=True,
            capture_output=True,
        )

        assert gate._refresh_clone_cache() is None
        assert (cache_dir / "file.txt").read_text() == "v1\n"

    def test_failure_returns_description_with_stderr(self, tmp_path: Path) -> None:
        """Subprocess failures are caught and described, including git's stderr."""
        import subprocess

        gate_dir = tmp_path / "gate" / "proj.git"
        gate_dir.mkdir(parents=True)
        cache_base = tmp_path / "clone-cache"

        gate = GitGate(
            scope="proj",
            gate_path=gate_dir,
            clone_cache_base=cache_base,
        )

        with patch(
            "terok_sandbox.gate.mirror.subprocess.run",
            side_effect=subprocess.CalledProcessError(128, "git", stderr=b"fatal: bad ref\n"),
        ):
            result = gate._refresh_clone_cache()

        assert result is not None
        assert "128" in result
        assert "fatal: bad ref" in result

    def test_returns_error_when_no_cache_base(self) -> None:
        """With no clone_cache_base, _refresh_clone_cache reports it is unconfigured."""
        gate = GitGate(scope="proj", gate_path="/tmp/terok-testing/gate/proj.git")
        assert gate._refresh_clone_cache() == "no clone cache configured"


def _make_upstream_and_gate(tmp_path: Path, *, branch: str = "master") -> tuple[Path, Path]:
    """Create a real one-commit upstream repo and a ``--mirror`` gate of it."""
    import subprocess

    upstream = tmp_path / "upstream"
    subprocess.run(["git", "init", "-b", branch, str(upstream)], check=True, capture_output=True)
    (upstream / "file.txt").write_text("v1\n")
    git = ["git", "-C", str(upstream)]
    subprocess.run([*git, "add", "."], check=True, capture_output=True)
    subprocess.run(
        [*git, "-c", "user.name=t", "-c", "user.email=t@t", "commit", "-m", "v1"],
        check=True,
        capture_output=True,
    )
    gate_dir = tmp_path / "gate" / "proj.git"
    subprocess.run(
        ["git", "clone", "--mirror", str(upstream), str(gate_dir)],
        check=True,
        capture_output=True,
    )
    return upstream, gate_dir


class TestCacheAutoRebuild:
    """Any incremental-refresh failure discards the cache and rebuilds it."""

    def test_incremental_failure_falls_back_to_rebuild(self, tmp_path: Path) -> None:
        """A failing in-place update triggers a from-scratch clone, not an error."""
        import subprocess

        gate_dir = tmp_path / "gate" / "proj.git"
        gate_dir.mkdir(parents=True)
        cache_dir = tmp_path / "clone-cache" / "proj"
        cache_dir.mkdir(parents=True)

        gate = GitGate(scope="proj", gate_path=gate_dir, clone_cache_base=tmp_path / "clone-cache")

        seen: list[list[str]] = []

        def fake_git(cmd: list[str], **_kwargs: object) -> MagicMock:
            """Fail the in-place update; model clone recreating the cache dir."""
            seen.append(cmd)
            if "set-url" in cmd:
                raise subprocess.CalledProcessError(128, "git", stderr=b"fatal: bad ref\n")
            if "clone" in cmd:
                # the stale cache must be gone before the re-clone recreates it
                assert not cache_dir.exists()
                cache_dir.mkdir(parents=True)
            return MagicMock(returncode=0)

        with patch("terok_sandbox.gate.mirror.subprocess.run", side_effect=fake_git):
            result = gate._refresh_clone_cache()

        assert result is None
        assert "set-url" in seen[0]
        assert "clone" in seen[1]
        assert "rev-parse" in seen[2]
        assert cache_dir.exists()

    def test_rebuild_failure_reports_and_removes_cache(self, tmp_path: Path) -> None:
        """When the rebuild fails too, the error surfaces and no half-built cache remains."""
        import subprocess

        gate_dir = tmp_path / "gate" / "proj.git"
        gate_dir.mkdir(parents=True)
        cache_dir = tmp_path / "clone-cache" / "proj"
        cache_dir.mkdir(parents=True)

        gate = GitGate(scope="proj", gate_path=gate_dir, clone_cache_base=tmp_path / "clone-cache")

        with patch(
            "terok_sandbox.gate.mirror.subprocess.run",
            side_effect=subprocess.CalledProcessError(128, "git", stderr=b"fatal: broken\n"),
        ):
            result = gate._refresh_clone_cache()

        assert result is not None
        assert "fatal: broken" in result
        assert not cache_dir.exists()

    def test_rebuild_refuses_empty_checkout_cache(self, tmp_path: Path) -> None:
        """A gate with a dangling HEAD must not yield a 'successful' empty cache.

        ``git clone`` exits 0 against such a gate but checks out nothing;
        the post-clone HEAD verification turns that into a reported
        failure and removes the unusable cache.
        """
        import subprocess

        _, gate_dir = _make_upstream_and_gate(tmp_path)
        subprocess.run(
            ["git", "-C", str(gate_dir), "symbolic-ref", "HEAD", "refs/heads/nonexistent"],
            check=True,
            capture_output=True,
        )
        cache_base = tmp_path / "clone-cache"
        gate = GitGate(scope="proj", gate_path=gate_dir, clone_cache_base=cache_base)

        result = gate._refresh_clone_cache()

        assert result is not None
        assert not (cache_base / "proj").exists()


class TestGateHeadSelfHeal:
    """sync() re-points the gate's HEAD after an upstream default-branch rename."""

    def test_upstream_rename_heals_head_and_cache(self, tmp_path: Path) -> None:
        """End-to-end: upstream renames its default branch; plain sync recovers."""
        import subprocess

        upstream, gate_dir = _make_upstream_and_gate(tmp_path)
        cache_base = tmp_path / "clone-cache"
        gate = GitGate(
            scope="proj",
            gate_path=gate_dir,
            upstream_url=str(upstream),
            clone_cache_base=cache_base,
        )

        first = gate.sync()
        assert first["success"] is True and first["cache_refreshed"] is True

        subprocess.run(
            ["git", "-C", str(upstream), "branch", "-m", "master", "trunk"],
            check=True,
            capture_output=True,
        )

        second = gate.sync()

        assert second["success"] is True
        assert second["cache_error"] is None
        assert second["cache_refreshed"] is True
        head = subprocess.run(
            ["git", "-C", str(gate_dir), "symbolic-ref", "HEAD"],
            check=True,
            capture_output=True,
            text=True,
        ).stdout.strip()
        assert head == "refs/heads/trunk"
        assert (cache_base / "proj" / "file.txt").read_text() == "v1\n"

    def test_valid_head_is_left_alone(self, tmp_path: Path) -> None:
        """The happy path never contacts upstream or rewrites a healthy HEAD."""
        _, gate_dir = _make_upstream_and_gate(tmp_path)
        gate = GitGate(scope="proj", gate_path=gate_dir, upstream_url=str(gate_dir))

        with patch("terok_sandbox.gate.mirror._query_upstream_head_ref") as mock_query:
            gate._heal_gate_head(env={})

        mock_query.assert_not_called()


class TestForceReinitClearsCache:
    """force_reinit recreates the clone cache along with the bare mirror."""

    def test_force_reinit_removes_cache_dir(self, tmp_path: Path) -> None:
        gate_dir = tmp_path / "gate" / "proj.git"
        gate_dir.mkdir(parents=True)
        cache_dir = tmp_path / "clone-cache" / "proj"
        cache_dir.mkdir(parents=True)
        (cache_dir / "stale.txt").write_text("stale\n")

        gate = GitGate(
            scope="proj",
            gate_path=gate_dir,
            upstream_url="git@github.com:org/repo.git",
            clone_cache_base=tmp_path / "clone-cache",
        )

        with (
            patch.object(gate, "_validate_gate"),
            patch.object(gate, "_ssh_env", return_value={}),
            patch("terok_sandbox.gate.mirror._clone_gate_mirror"),
            patch.object(
                gate,
                "sync_branches",
                return_value={"success": True, "updated_branches": ["all"], "errors": []},
            ),
            patch.object(gate, "_heal_gate_head"),
            patch.object(gate, "_refresh_clone_cache", return_value=None),
        ):
            result = gate.sync(force_reinit=True)

        assert result["success"] is True
        assert not cache_dir.exists()


class TestSyncIntegration:
    """Verify sync() calls _refresh_clone_cache after successful branch sync."""

    @pytest.fixture()
    def _gate_with_cache(self, tmp_path: Path) -> GitGate:
        gate_dir = tmp_path / "gate" / "proj.git"
        gate_dir.mkdir(parents=True)
        return GitGate(
            scope="proj",
            gate_path=gate_dir,
            upstream_url="git@github.com:org/repo.git",
            clone_cache_base=tmp_path / "clone-cache",
        )

    def test_sync_refreshes_cache_on_success(self, _gate_with_cache: GitGate) -> None:
        with (
            patch.object(_gate_with_cache, "_validate_gate"),
            patch.object(_gate_with_cache, "_ssh_env", return_value={}),
            patch.object(
                _gate_with_cache,
                "sync_branches",
                return_value={"success": True, "updated_branches": ["all"], "errors": []},
            ),
            patch.object(_gate_with_cache, "_refresh_clone_cache", return_value=None) as mock_cache,
        ):
            result = _gate_with_cache.sync()

        mock_cache.assert_called_once()
        assert result["cache_refreshed"] is True
        assert result["cache_error"] is None

    def test_sync_reports_cache_refresh_failure(self, _gate_with_cache: GitGate) -> None:
        with (
            patch.object(_gate_with_cache, "_validate_gate"),
            patch.object(_gate_with_cache, "_ssh_env", return_value={}),
            patch.object(
                _gate_with_cache,
                "sync_branches",
                return_value={"success": True, "updated_branches": ["all"], "errors": []},
            ),
            patch.object(_gate_with_cache, "_refresh_clone_cache", return_value="boom"),
        ):
            result = _gate_with_cache.sync()

        assert result["success"] is True
        assert result["cache_refreshed"] is False
        assert result["cache_error"] == "boom"

    def test_sync_skips_cache_on_branch_failure(self, _gate_with_cache: GitGate) -> None:
        with (
            patch.object(_gate_with_cache, "_validate_gate"),
            patch.object(_gate_with_cache, "_ssh_env", return_value={}),
            patch.object(
                _gate_with_cache,
                "sync_branches",
                return_value={
                    "success": False,
                    "updated_branches": [],
                    "errors": ["timeout"],
                },
            ),
            patch.object(_gate_with_cache, "_refresh_clone_cache") as mock_cache,
        ):
            result = _gate_with_cache.sync()

        mock_cache.assert_not_called()
        assert result["cache_refreshed"] is False
        assert result["cache_error"] is None

    def test_sync_without_cache_base(self, tmp_path: Path) -> None:
        gate_dir = tmp_path / "gate" / "proj.git"
        gate_dir.mkdir(parents=True)
        gate = GitGate(
            scope="proj",
            gate_path=gate_dir,
            upstream_url="git@github.com:org/repo.git",
        )

        with (
            patch.object(gate, "_validate_gate"),
            patch.object(gate, "_ssh_env", return_value={}),
            patch.object(
                gate,
                "sync_branches",
                return_value={"success": True, "updated_branches": ["all"], "errors": []},
            ),
        ):
            result = gate.sync()

        assert result["cache_refreshed"] is False
