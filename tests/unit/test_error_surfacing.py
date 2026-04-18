# SPDX-FileCopyrightText: 2026 Jiri Vyskocil
# SPDX-License-Identifier: Apache-2.0

"""Tests for logging utilities and error-handling paths added for silent failure surfacing.

Covers:
- ``_util._logging``: log_debug, log_warning, warn_user
- ``git_gate``: _get_upstream_head, _get_gate_branch_head, _count_commits_range
  returning None on subprocess failures
- ``gate.server``: GateRequestHandler.log_message filtering by status code
"""

from __future__ import annotations

import subprocess
import unittest.mock
from pathlib import Path

import pytest

from tests.constants import LOCALHOST_PEER, MOCK_BASE

# ── Paths used by these tests ───────────────────────────────────────────────

MOCK_LOG_DIR = MOCK_BASE / "logging"
"""Fake state directory for logging utility tests."""

MOCK_GATE_DIR = MOCK_BASE / "gate-error-surfacing"
"""Fake gate mirror directory for git_gate error tests."""


# ═══════════════════════════════════════════════════════════════════════════
# 1. Logging utilities (_util._logging)
# ═══════════════════════════════════════════════════════════════════════════


class TestLogDebug:
    """Tests for log_debug() — writes a DEBUG line to the sandbox log file."""

    def test_writes_debug_line(self, tmp_path: Path) -> None:
        """log_debug writes a timestamped DEBUG line to the log file."""
        from terok_sandbox._util._logging import log_debug

        with unittest.mock.patch(
            "platformdirs.user_state_path",
            return_value=tmp_path,
        ):
            log_debug("test debug message")

        log_file = tmp_path / "terok-sandbox.log"
        assert log_file.exists()
        content = log_file.read_text()
        assert "DEBUG: test debug message" in content

    def test_includes_timestamp(self, tmp_path: Path) -> None:
        """Log lines contain a bracketed timestamp."""
        from terok_sandbox._util._logging import log_debug

        with unittest.mock.patch(
            "platformdirs.user_state_path",
            return_value=tmp_path,
        ):
            log_debug("timestamp check")

        content = (tmp_path / "terok-sandbox.log").read_text()
        # Expect format like [2026-04-02 12:00:00]
        assert content.startswith("[")
        assert "] DEBUG:" in content


class TestLogWarning:
    """Tests for log_warning() — writes a WARNING line to the sandbox log file."""

    def test_writes_warning_line(self, tmp_path: Path) -> None:
        """log_warning writes a WARNING-level line."""
        from terok_sandbox._util._logging import log_warning

        with unittest.mock.patch(
            "platformdirs.user_state_path",
            return_value=tmp_path,
        ):
            log_warning("something went wrong")

        content = (tmp_path / "terok-sandbox.log").read_text()
        assert "WARNING: something went wrong" in content


class TestLogSilentOnError:
    """_log never raises — any IO error is silently swallowed."""

    def test_swallows_io_error(self) -> None:
        """_log does not raise when user_state_path raises."""
        from terok_sandbox._util._logging import log_debug

        with unittest.mock.patch(
            "platformdirs.user_state_path",
            side_effect=OSError("disk full"),
        ):
            # Must not raise
            log_debug("should not crash")


class TestWarnUser:
    """Tests for warn_user() — prints to stderr AND writes a WARNING to the log."""

    def test_prints_to_stderr(self, capsys: pytest.CaptureFixture[str], tmp_path: Path) -> None:
        """warn_user prints a structured warning to stderr."""
        from terok_sandbox._util._logging import warn_user

        with unittest.mock.patch(
            "platformdirs.user_state_path",
            return_value=tmp_path,
        ):
            warn_user("gate", "upstream unreachable")

        captured = capsys.readouterr()
        assert "Warning [gate]: upstream unreachable" in captured.err

    def test_logs_warning_to_file(self, tmp_path: Path) -> None:
        """warn_user also writes a WARNING line to the log file."""
        from terok_sandbox._util._logging import warn_user

        with unittest.mock.patch(
            "platformdirs.user_state_path",
            return_value=tmp_path,
        ):
            warn_user("shield", "rule mismatch")

        content = (tmp_path / "terok-sandbox.log").read_text()
        assert "WARNING: [shield] rule mismatch" in content


# ═══════════════════════════════════════════════════════════════════════════
# 2. git_gate — narrowed exception handling returns None gracefully
# ═══════════════════════════════════════════════════════════════════════════


class TestGetUpstreamHead:
    """Tests for _get_upstream_head returning None on subprocess failures."""

    @pytest.mark.parametrize(
        "exception",
        [
            subprocess.CalledProcessError(1, "git"),
            subprocess.TimeoutExpired("git", 30),
            FileNotFoundError("git not found"),
            OSError("network error"),
        ],
        ids=["CalledProcessError", "TimeoutExpired", "FileNotFoundError", "OSError"],
    )
    def test_returns_none_on_exception(self, exception: Exception) -> None:
        """_get_upstream_head returns None when subprocess.run raises."""
        from terok_sandbox.gate.mirror import _get_upstream_head

        with unittest.mock.patch("subprocess.run", side_effect=exception):
            result = _get_upstream_head("git@example.com:repo.git", "main", {})
        assert result is None

    def test_returns_none_on_nonzero_exit(self) -> None:
        """_get_upstream_head returns None when git ls-remote exits non-zero."""
        from terok_sandbox.gate.mirror import _get_upstream_head

        mock_result = unittest.mock.Mock(returncode=1, stdout="", stderr="error")
        with unittest.mock.patch("subprocess.run", return_value=mock_result):
            result = _get_upstream_head("git@example.com:repo.git", "main", {})
        assert result is None

    def test_returns_none_on_empty_output(self) -> None:
        """_get_upstream_head returns None when git ls-remote returns no output."""
        from terok_sandbox.gate.mirror import _get_upstream_head

        mock_result = unittest.mock.Mock(returncode=0, stdout="")
        with unittest.mock.patch("subprocess.run", return_value=mock_result):
            result = _get_upstream_head("git@example.com:repo.git", "main", {})
        assert result is None

    def test_returns_dict_on_success(self) -> None:
        """_get_upstream_head returns a dict with commit_hash on success."""
        from terok_sandbox.gate.mirror import _get_upstream_head

        mock_result = unittest.mock.Mock(
            returncode=0,
            stdout="abc123def456\trefs/heads/main\n",
        )
        with unittest.mock.patch("subprocess.run", return_value=mock_result):
            result = _get_upstream_head("git@example.com:repo.git", "main", {})
        assert result is not None
        assert result["commit_hash"] == "abc123def456"
        assert result["ref_name"] == "refs/heads/main"
        assert result["upstream_url"] == "git@example.com:repo.git"


class TestGetGateBranchHead:
    """Tests for _get_gate_branch_head returning None on subprocess failures."""

    @pytest.mark.parametrize(
        "exception",
        [
            subprocess.CalledProcessError(1, "git"),
            subprocess.TimeoutExpired("git", 30),
            FileNotFoundError("git not found"),
            OSError("permission denied"),
        ],
        ids=["CalledProcessError", "TimeoutExpired", "FileNotFoundError", "OSError"],
    )
    def test_returns_none_on_exception(self, exception: Exception) -> None:
        """_get_gate_branch_head returns None when subprocess.run raises."""
        from terok_sandbox.gate.mirror import _get_gate_branch_head

        mock_path = unittest.mock.Mock()
        mock_path.exists.return_value = True
        with unittest.mock.patch("subprocess.run", side_effect=exception):
            result = _get_gate_branch_head(mock_path, "main", {})
        assert result is None

    def test_returns_none_when_gate_dir_missing(self) -> None:
        """_get_gate_branch_head returns None when gate_dir does not exist."""
        from terok_sandbox.gate.mirror import _get_gate_branch_head

        result = _get_gate_branch_head(MOCK_GATE_DIR, "main", {})
        assert result is None

    def test_returns_hash_on_success(self, tmp_path: Path) -> None:
        """_get_gate_branch_head returns the commit hash on success."""
        from terok_sandbox.gate.mirror import _get_gate_branch_head

        mock_result = unittest.mock.Mock(returncode=0, stdout="abc123\n")
        with unittest.mock.patch("subprocess.run", return_value=mock_result):
            result = _get_gate_branch_head(tmp_path, "main", {})
        assert result == "abc123"


class TestCountCommitsRange:
    """Tests for _count_commits_range returning None on subprocess failures."""

    @pytest.mark.parametrize(
        "exception",
        [
            subprocess.CalledProcessError(1, "git"),
            subprocess.TimeoutExpired("git", 30),
            FileNotFoundError("git not found"),
            OSError("io error"),
        ],
        ids=["CalledProcessError", "TimeoutExpired", "FileNotFoundError", "OSError"],
    )
    def test_returns_none_on_exception(self, exception: Exception) -> None:
        """_count_commits_range returns None when subprocess.run raises."""
        from terok_sandbox.gate.mirror import _count_commits_range

        with unittest.mock.patch("subprocess.run", side_effect=exception):
            result = _count_commits_range(MOCK_GATE_DIR, "abc", "def", {})
        assert result is None

    def test_returns_none_on_nonzero_exit(self) -> None:
        """_count_commits_range returns None when git rev-list exits non-zero."""
        from terok_sandbox.gate.mirror import _count_commits_range

        mock_result = unittest.mock.Mock(returncode=128, stdout="")
        with unittest.mock.patch("subprocess.run", return_value=mock_result):
            result = _count_commits_range(MOCK_GATE_DIR, "abc", "def", {})
        assert result is None

    def test_returns_count_on_success(self) -> None:
        """_count_commits_range returns an integer count on success."""
        from terok_sandbox.gate.mirror import _count_commits_range

        mock_result = unittest.mock.Mock(returncode=0, stdout="7\n")
        with unittest.mock.patch("subprocess.run", return_value=mock_result):
            result = _count_commits_range(MOCK_GATE_DIR, "abc", "def", {})
        assert result == 7


# ═══════════════════════════════════════════════════════════════════════════
# 3. gate.server — GateRequestHandler.log_message filtering
# ═══════════════════════════════════════════════════════════════════════════


def _make_log_handler() -> object:
    """Create a GateRequestHandler instance for log_message testing."""
    import json
    import tempfile

    from terok_sandbox.gate.server import TokenStore, _make_handler_class

    td = tempfile.mkdtemp()
    base = Path(td)
    token_file = base / "tokens.json"
    token_file.write_text(json.dumps({}))
    store = TokenStore(token_file)
    handler_class = _make_handler_class(base, store)

    handler = handler_class.__new__(handler_class)
    handler.request = None
    handler.client_address = LOCALHOST_PEER
    handler.server = type("FakeServer", (), {"server_name": "localhost", "server_port": 0})()
    return handler


# ═══════════════════════════════════════════════════════════════════════════
# 2b. git_gate — gate dir removal failure logging
# ═══════════════════════════════════════════════════════════════════════════


class TestGateDirRemovalWarning:
    """Verify logger.warning when shutil.rmtree fails during force_reinit."""

    def test_rmtree_failure_logs_warning(self, tmp_path: Path) -> None:
        """Failed gate dir removal logs a warning and continues."""
        from terok_sandbox.gate.mirror import GitGate

        gate_dir = tmp_path / "gate" / "myproject.git"
        gate_dir.mkdir(parents=True)
        gate = GitGate(
            scope="myproject",
            gate_path=gate_dir,
            upstream_url="https://example.com/org/repo.git",
        )

        with (
            unittest.mock.patch("shutil.rmtree", side_effect=PermissionError("nope")),
            unittest.mock.patch("terok_sandbox.gate.mirror._clone_gate_mirror"),
            unittest.mock.patch("terok_sandbox.gate.mirror.logger") as mock_logger,
        ):
            try:
                gate.sync(force_reinit=True)
            except Exception:
                pass  # clone may fail; we only care about the warning
            rmtree_warnings = [
                c
                for c in mock_logger.warning.call_args_list
                if "Failed to remove gate dir" in str(c)
            ]
            assert len(rmtree_warnings) == 1


# ═══════════════════════════════════════════════════════════════════════════
# 2c. runtime — stop_task_containers failure logging
# ═══════════════════════════════════════════════════════════════════════════


class TestStopTaskContainersLogging:
    """Verify log_debug when podman rm fails."""

    def test_subprocess_failure_logs_debug(self) -> None:
        """FileNotFoundError (podman missing) is caught and logged."""
        from terok_sandbox.runtime import stop_task_containers

        with (
            unittest.mock.patch(
                "terok_sandbox.runtime.subprocess.run",
                side_effect=FileNotFoundError("podman not found"),
            ),
            unittest.mock.patch("terok_sandbox.runtime.log_debug") as mock_log,
        ):
            results = stop_task_containers(["test-container"])
        # Should have logged the failure
        calls = [str(c) for c in mock_log.call_args_list]
        assert any("not found" in c for c in calls)
        # Result reports failure with reason
        assert len(results) == 1
        assert results[0].removed is False
        assert results[0].error == "podman not found"


# ═══════════════════════════════════════════════════════════════════════════
# 2d. vault_lifecycle — DB read failure logging
# ═══════════════════════════════════════════════════════════════════════════


class TestVaultStatusDbFailure:
    """Verify log_warning when credential DB read fails in get_vault_status."""

    def test_db_exception_logs_warning(self, tmp_path: Path) -> None:
        """Corrupted DB triggers log_warning and returns empty credentials."""
        from terok_sandbox import SandboxConfig
        from terok_sandbox.vault.lifecycle import VaultManager

        cfg = SandboxConfig(state_dir=tmp_path)
        # Create a corrupt DB file
        cfg.db_path.parent.mkdir(parents=True, exist_ok=True)
        cfg.db_path.write_text("not a sqlite db")

        with unittest.mock.patch("terok_sandbox.vault.lifecycle.log_warning") as mock_warn:
            VaultManager(cfg).get_status()
        mock_warn.assert_called_once()
        assert "credential db" in mock_warn.call_args[0][0].lower()


class TestGateHandlerLogMessage:
    """Tests for GateRequestHandler.log_message filtering by HTTP status code."""

    def test_log_message_silent_for_2xx(self) -> None:
        """2xx responses are silently suppressed."""
        handler = _make_log_handler()
        with unittest.mock.patch("terok_sandbox.gate.server._logger") as mock_logger:
            handler.log_message("%s %s %s", "GET /foo", "200", "-")
            mock_logger.warning.assert_not_called()

    def test_log_message_warns_for_4xx(self) -> None:
        """4xx responses are logged at WARNING level."""
        handler = _make_log_handler()
        with unittest.mock.patch("terok_sandbox.gate.server._logger") as mock_logger:
            handler.log_message("%s %s %s", "GET /foo", "404", "-")
            mock_logger.warning.assert_called_once()

    def test_log_message_warns_for_5xx(self) -> None:
        """5xx responses are logged at WARNING level."""
        handler = _make_log_handler()
        with unittest.mock.patch("terok_sandbox.gate.server._logger") as mock_logger:
            handler.log_message("%s %s %s", "POST /bar", "500", "-")
            mock_logger.warning.assert_called_once()

    def test_log_message_silent_for_3xx(self) -> None:
        """3xx responses are silently suppressed (below 400 threshold)."""
        handler = _make_log_handler()
        with unittest.mock.patch("terok_sandbox.gate.server._logger") as mock_logger:
            handler.log_message("%s %s %s", "GET /foo", "302", "-")
            mock_logger.warning.assert_not_called()

    def test_log_message_handles_malformed_args(self) -> None:
        """Malformed arguments do not raise — silently ignored."""
        handler = _make_log_handler()
        with unittest.mock.patch("terok_sandbox.gate.server._logger") as mock_logger:
            # No args at all
            handler.log_message("no args")
            mock_logger.warning.assert_not_called()
            # Integer status (below 400) — handled gracefully via str()
            handler.log_message("%s %s", "GET /foo", 200)
            mock_logger.warning.assert_not_called()

    def test_log_message_non_numeric_status_no_crash(self) -> None:
        """Non-numeric status string hits ValueError catch without crashing."""
        handler = _make_log_handler()
        with unittest.mock.patch("terok_sandbox.gate.server._logger") as mock_logger:
            handler.log_message("%s %s", "GET /foo", "not-a-number")
            mock_logger.warning.assert_not_called()
