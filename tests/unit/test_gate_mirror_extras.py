# SPDX-FileCopyrightText: 2026 Jiri Vyskocil
# SPDX-License-Identifier: Apache-2.0

"""Tests for GitGate sync, compare_vs_upstream, last_commit, and helpers."""

from __future__ import annotations

import subprocess
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from terok_sandbox.gate.mirror import (
    GateStalenessInfo,
    GitGate,
    _clone_gate_mirror,  # noqa: PLC2701
    _count_commits_range,  # noqa: PLC2701
    _db_has_keys_for_scope,  # noqa: PLC2701
    _EphemeralSigner,  # noqa: PLC2701
    _get_gate_branch_head,  # noqa: PLC2701
    _get_upstream_head,  # noqa: PLC2701
    _init_remoteless_gate,  # noqa: PLC2701
)


def _proc(rc: int = 0, stdout: str = "", stderr: str = "") -> subprocess.CompletedProcess[str]:
    return subprocess.CompletedProcess(args=[], returncode=rc, stdout=stdout, stderr=stderr)


# ---------------------------------------------------------------------------
# GitGate.sync — guards and validation
# ---------------------------------------------------------------------------


class TestGitGateSyncGuards:
    """sync() validates inputs before touching the filesystem."""

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
# GitGate.sync — remoteless gate (no upstream_url configured)
# ---------------------------------------------------------------------------


class TestGitGateSyncRemoteless:
    """sync() on a remoteless gate initialises a bare repo and returns a no-op."""

    def test_fresh_gate_with_no_upstream_inits_bare_repo(self, tmp_path: Path) -> None:
        """First sync on a missing gate without upstream runs ``git init --bare``."""
        gate_path = tmp_path / "remoteless.git"
        gate = GitGate(scope="scratch", gate_path=gate_path, upstream_url=None)

        # Do not mock subprocess — we want the real ``git init --bare`` to
        # produce a real bare repo so later reads (git commands) succeed.
        result = gate.sync()

        assert result["upstream_url"] is None
        assert result["created"] is True
        assert result["success"] is True
        assert result["updated_branches"] == []
        assert result["errors"] == []
        assert result["cache_refreshed"] is False

        # Sanity: ``git init --bare`` produces at least a HEAD file.
        assert (gate_path / "HEAD").is_file()

    def test_existing_remoteless_gate_is_noop(self, tmp_path: Path) -> None:
        """Second sync on an existing remoteless gate does not re-initialise."""
        gate_path = tmp_path / "remoteless.git"
        gate = GitGate(scope="scratch", gate_path=gate_path, upstream_url=None)

        first = gate.sync()
        assert first["created"] is True

        with patch("terok_sandbox.gate.mirror._init_remoteless_gate") as mock_init:
            second = gate.sync()
        mock_init.assert_not_called()
        assert second["created"] is False
        assert second["success"] is True
        assert second["updated_branches"] == []

    def test_force_reinit_reruns_init_bare(self, tmp_path: Path) -> None:
        """``force_reinit=True`` wipes and re-initialises a remoteless gate."""
        gate_path = tmp_path / "remoteless.git"
        gate = GitGate(scope="scratch", gate_path=gate_path, upstream_url=None)
        gate.sync()

        result = gate.sync(force_reinit=True)
        assert result["created"] is True
        assert (gate_path / "HEAD").is_file()

    def test_remoteless_sync_skips_clone_cache_refresh(self, tmp_path: Path) -> None:
        """A remoteless gate has nothing to seed a clone cache from."""
        gate = GitGate(
            scope="scratch",
            gate_path=tmp_path / "remoteless.git",
            upstream_url=None,
            clone_cache_base=tmp_path / "cache",
        )
        with patch.object(gate, "_refresh_clone_cache") as mock_refresh:
            result = gate.sync()
        mock_refresh.assert_not_called()
        assert result["cache_refreshed"] is False


# ---------------------------------------------------------------------------
# _init_remoteless_gate helper
# ---------------------------------------------------------------------------


class TestInitRemotelessGate:
    """The lower-level helper that backs the remoteless-gate path."""

    def test_creates_bare_repo_at_requested_path(self, tmp_path: Path) -> None:
        gate_dir = tmp_path / "g.git"
        _init_remoteless_gate(gate_dir)
        assert (gate_dir / "HEAD").is_file()
        # Bare repos have a ``config`` file marking ``bare = true``.
        config = (gate_dir / "config").read_text()
        assert "bare = true" in config

    def test_missing_git_raises_actionable_hint(self, tmp_path: Path) -> None:
        with (
            patch(
                "terok_sandbox.gate.mirror.subprocess.run",
                side_effect=FileNotFoundError("git"),
            ),
            pytest.raises(SystemExit, match="git not found"),
        ):
            _init_remoteless_gate(tmp_path / "g.git")


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


# ---------------------------------------------------------------------------
# _db_has_keys_for_scope — the gate-auth pre-check
# ---------------------------------------------------------------------------


class TestDbHasKeysForScope:
    """The DB pre-check fails *soft* to ``False`` so the caller can surface
    the friendly ``GateAuthNotConfigured`` instead of a stack trace."""

    def test_true_when_scope_has_keys(self) -> None:
        """A scope with a key opens with the given passphrase, returns ``True``, closes."""
        db = MagicMock()
        db.list_ssh_keys_for_scope.return_value = ["key-1"]
        cfg = MagicMock()

        with patch("terok_sandbox.vault.store.db.CredentialDB", return_value=db) as ctor:
            assert _db_has_keys_for_scope(cfg, "proj-a", "pw") is True

        ctor.assert_called_once_with(cfg.db_path, passphrase="pw")
        db.list_ssh_keys_for_scope.assert_called_once_with("proj-a")
        db.close.assert_called_once()

    def test_false_when_scope_has_no_keys(self) -> None:
        """An empty key list returns ``False`` (no keys assigned to the scope)."""
        db = MagicMock()
        db.list_ssh_keys_for_scope.return_value = []
        cfg = MagicMock()

        with patch("terok_sandbox.vault.store.db.CredentialDB", return_value=db):
            assert _db_has_keys_for_scope(cfg, "proj-a", "pw") is False
        db.close.assert_called_once()

    def test_false_when_db_open_raises(self) -> None:
        """A vault that won't open (wrong passphrase / corrupt) → ``False``.

        The failure must not leak a ``sqlite3.Error``; it collapses to
        ``False`` so the caller raises the actionable
        ``GateAuthNotConfigured`` hint."""
        cfg = MagicMock()

        with patch(
            "terok_sandbox.vault.store.db.CredentialDB", side_effect=RuntimeError("vault locked")
        ):
            assert _db_has_keys_for_scope(cfg, "proj-a", "pw") is False

    def test_false_when_listing_raises_but_still_closes(self) -> None:
        """A query that raises (e.g. schema not bootstrapped) → ``False``,
        and the DB handle is still closed in the ``finally``."""
        db = MagicMock()
        db.list_ssh_keys_for_scope.side_effect = RuntimeError("no such table")
        cfg = MagicMock()

        with patch("terok_sandbox.vault.store.db.CredentialDB", return_value=db):
            assert _db_has_keys_for_scope(cfg, "proj-a", "pw") is False
        db.close.assert_called_once()


# ---------------------------------------------------------------------------
# _EphemeralSigner.stop — teardown without a live signer
# ---------------------------------------------------------------------------


class TestEphemeralSignerStart:
    """``start`` binds the signer on a background loop, or fails loudly."""

    def test_no_keys_for_scope_raises_gate_auth_not_configured(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """A scope with no assigned SSH keys is a configuration error, not a bind."""
        from terok_sandbox.gate.mirror import GateAuthNotConfigured

        monkeypatch.setattr(
            "terok_sandbox.gate.mirror._db_has_keys_for_scope", lambda _cfg, _scope, _pw: False
        )
        with patch("terok_sandbox.config.SandboxConfig"):
            with pytest.raises(GateAuthNotConfigured):
                _EphemeralSigner.start("proj-a")

    def test_bind_failure_raises_runtime_error_and_cleans_up(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """A ``start_ssh_signer_local`` that raises surfaces as a ``RuntimeError``.

        The background thread captures the bind exception, the foreground
        ``start`` joins the (now-dead) thread, cleans the temp dir, and
        re-raises with the original cause chained.  No live loop / socket
        is needed — the signer entry point is mocked to raise.
        """
        monkeypatch.setattr(
            "terok_sandbox.gate.mirror._db_has_keys_for_scope", lambda _cfg, _scope, _pw: True
        )

        async def _boom(**_kw: object) -> object:
            raise OSError("address already in use")

        with (
            patch("terok_sandbox.config.SandboxConfig"),
            patch("terok_sandbox.vault.ssh.signer.start_ssh_signer_local", side_effect=_boom),
            pytest.raises(RuntimeError, match="failed to bind"),
        ):
            _EphemeralSigner.start("proj-a")

    def test_start_binds_then_stop_terminates_cleanly(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """A successful bind serves on the background loop; ``stop`` ends it without hanging.

        Drives the real thread / loop / ``serve_forever`` machinery (the
        signer entry point is patched to return an actual bound server, so
        no vault keys are needed).  ``stop`` schedules ``server.close`` on
        the signer's loop — and on Python 3.12+ that *does* unblock
        ``serve_forever``, so ``run_until_complete`` returns and the thread
        joins instead of leaking.
        """
        import asyncio

        monkeypatch.setattr(
            "terok_sandbox.gate.mirror._db_has_keys_for_scope", lambda _cfg, _scope, _pw: True
        )

        async def _real_server(*, socket_path, **_kw: object) -> asyncio.AbstractServer:
            async def _handle(_reader: object, writer: asyncio.StreamWriter) -> None:
                writer.close()

            return await asyncio.start_unix_server(_handle, path=str(socket_path))

        with (
            patch("terok_sandbox.config.SandboxConfig"),
            patch(
                "terok_sandbox.vault.ssh.signer.start_ssh_signer_local",
                side_effect=_real_server,
            ),
        ):
            signer = _EphemeralSigner.start("proj-a")
            try:
                assert signer.socket_path.exists()
            finally:
                signer.stop()  # must terminate the serve_forever loop, not hang

        assert not signer._thread.is_alive()
        assert not Path(signer._tmpdir.name).exists()

    def test_passphrase_resolved_on_caller_and_forwarded(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """``start`` resolves the vault passphrase itself and hands it to the signer."""
        import asyncio

        monkeypatch.setattr(
            "terok_sandbox.gate.mirror._db_has_keys_for_scope", lambda _cfg, _scope, _pw: True
        )
        captured: dict[str, object] = {}

        async def _real_server(
            *, socket_path: Path, passphrase: object, **_kw: object
        ) -> asyncio.AbstractServer:
            captured["passphrase"] = passphrase

            async def _handle(_reader: object, writer: asyncio.StreamWriter) -> None:
                writer.close()

            return await asyncio.start_unix_server(_handle, path=str(socket_path))

        cfg = MagicMock()
        cfg.resolve_passphrase.return_value = "s3kret"
        with (
            patch("terok_sandbox.config.SandboxConfig", return_value=cfg),
            patch(
                "terok_sandbox.vault.ssh.signer.start_ssh_signer_local",
                side_effect=_real_server,
            ),
        ):
            signer = _EphemeralSigner.start("proj-a")
            try:
                assert captured["passphrase"] == "s3kret"
                cfg.resolve_passphrase.assert_called_once_with(prompt_on_tty=False)
            finally:
                signer.stop()


class TestEphemeralSignerStop:
    """``stop`` must clean up the temp dir on every path, even when the
    background signer thread already exited."""

    def test_stop_cleans_up_when_thread_already_dead(self, tmp_path: Path) -> None:
        """When the signer thread is no longer alive, ``stop`` just cleans
        the temp dir — it must not touch the (closed) loop / server."""
        import tempfile
        import threading
        from unittest.mock import MagicMock

        tmpdir = tempfile.TemporaryDirectory(prefix="terok-test-signer-")
        dead_thread = threading.Thread(target=lambda: None)
        dead_thread.start()
        dead_thread.join()  # ensure not alive
        loop = MagicMock()
        signer = _EphemeralSigner(
            socket_path=Path(tmpdir.name) / "agent.sock",
            _tmpdir=tmpdir,
            _thread=dead_thread,
            _loop=loop,
            _server=MagicMock(),
        )

        signer.stop()

        # The dead-thread branch never reaches into the loop.
        loop.call_soon_threadsafe.assert_not_called()
        # Temp dir cleaned up (directory removed).
        assert not Path(tmpdir.name).exists()

    def test_stop_closes_live_server_via_loop_then_joins(self, tmp_path: Path) -> None:
        """A live signer thread is shut down by scheduling ``server.close`` on its loop.

        Cross-thread reach must go through ``loop.call_soon_threadsafe`` —
        calling ``server.close()`` directly across threads is UB.  We use a
        real (alive) thread blocked on an event so the live branch is taken,
        then have the mocked ``call_soon_threadsafe`` release it so the
        ``join`` returns and the temp dir is cleaned.
        """
        import tempfile
        import threading
        from unittest.mock import MagicMock

        tmpdir = tempfile.TemporaryDirectory(prefix="terok-test-signer-")
        release = threading.Event()
        live_thread = threading.Thread(target=release.wait, daemon=True)
        live_thread.start()

        server = MagicMock()
        loop = MagicMock()
        # The production code schedules ``server.close`` on the loop; here we
        # let that scheduling also unblock the worker thread so ``join`` ends.
        loop.call_soon_threadsafe.side_effect = lambda _cb: release.set()

        signer = _EphemeralSigner(
            socket_path=Path(tmpdir.name) / "agent.sock",
            _tmpdir=tmpdir,
            _thread=live_thread,
            _loop=loop,
            _server=server,
        )

        signer.stop()

        loop.call_soon_threadsafe.assert_called_once_with(server.close)
        assert not live_thread.is_alive()
        assert not Path(tmpdir.name).exists()
