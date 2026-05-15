# SPDX-FileCopyrightText: 2026 Jiri Vyskocil
# SPDX-License-Identifier: Apache-2.0

"""Tests for the shared ``systemctl --user`` invocation helpers."""

from __future__ import annotations

import subprocess
from unittest.mock import MagicMock, patch

import pytest

from terok_sandbox._util import _systemctl

# The helpers resolve ``systemctl`` once at module load (CWE-426 hardening).
# Tests pin the resolved path so they don't depend on whether the host running
# the suite happens to have systemctl installed.
_FAKE_SYSTEMCTL = "/usr/bin/systemctl"


@pytest.fixture
def with_systemctl_path(monkeypatch: pytest.MonkeyPatch) -> str:
    """Pin ``_SYSTEMCTL_PATH`` to a known absolute path for the duration of the test."""
    monkeypatch.setattr(_systemctl, "_SYSTEMCTL_PATH", _FAKE_SYSTEMCTL)
    return _FAKE_SYSTEMCTL


@pytest.fixture
def without_systemctl_path(monkeypatch: pytest.MonkeyPatch) -> None:
    """Pin ``_SYSTEMCTL_PATH`` to ``None`` to simulate a host with no systemctl."""
    monkeypatch.setattr(_systemctl, "_SYSTEMCTL_PATH", None)


class TestRun:
    """``run`` — every known failure mode reaches the caller as ``SystemExit``."""

    def test_happy_path_runs_systemctl(self, with_systemctl_path: str) -> None:
        with patch("terok_sandbox._util._systemctl.subprocess.run") as run:
            _systemctl.run("daemon-reload")
        assert run.call_args.args[0] == [with_systemctl_path, "--user", "daemon-reload"]

    def test_missing_systemctl_path_raises_systemexit(self, without_systemctl_path: None) -> None:
        """When the module-load resolution returned None, ``run`` fails closed."""
        with pytest.raises(SystemExit) as exc:
            _systemctl.run("daemon-reload")
        assert "systemctl" in str(exc.value)
        assert "not found" in str(exc.value)

    def test_failure_re_raises_as_systemexit_with_stderr(self, with_systemctl_path: str) -> None:
        """The re-raise flattens ``CalledProcessError`` into a readable message."""
        err = subprocess.CalledProcessError(
            returncode=1, cmd=[], stderr=b"Failed to connect to bus"
        )
        with patch("terok_sandbox._util._systemctl.subprocess.run", side_effect=err):
            with pytest.raises(SystemExit) as exc:
                _systemctl.run("enable", "--now", "terok-vault.socket")
        assert "Failed to connect to bus" in str(exc.value)

    def test_timeout_normalised_to_systemexit_with_captured_output(
        self, with_systemctl_path: str
    ) -> None:
        """A wedged systemctl call surfaces the timeout value and captured stderr."""
        err = subprocess.TimeoutExpired(cmd=[], timeout=10, stderr=b"hanging on dbus connect")
        with patch("terok_sandbox._util._systemctl.subprocess.run", side_effect=err):
            with pytest.raises(SystemExit) as exc:
                _systemctl.run("restart", "terok-gate.socket")
        msg = str(exc.value)
        assert "10" in msg
        assert "hanging on dbus connect" in msg

    def test_missing_binary_at_call_time_normalised_to_systemexit(
        self, with_systemctl_path: str
    ) -> None:
        """The resolved path vanished between module load and call (e.g. pipx upgrade)."""
        with patch(
            "terok_sandbox._util._systemctl.subprocess.run",
            side_effect=FileNotFoundError(2, "No such file or directory", _FAKE_SYSTEMCTL),
        ):
            with pytest.raises(SystemExit) as exc:
                _systemctl.run("daemon-reload")
        assert "systemctl" in str(exc.value)
        assert "not found" in str(exc.value)


class TestRunBestEffort:
    """``run_best_effort`` — swallow missing systemctl, non-zero exit, and timeouts."""

    def test_missing_systemctl_path_returns_silently(self, without_systemctl_path: None) -> None:
        with patch("terok_sandbox._util._systemctl.subprocess.run") as run:
            _systemctl.run_best_effort("daemon-reload")
        run.assert_not_called()

    def test_invokes_systemctl_with_verb_and_args(self, with_systemctl_path: str) -> None:
        with patch("terok_sandbox._util._systemctl.subprocess.run") as run:
            _systemctl.run_best_effort("disable", "--now", "terok-gate.socket")
        run.assert_called_once()
        assert run.call_args.args[0] == [
            with_systemctl_path,
            "--user",
            "disable",
            "--now",
            "terok-gate.socket",
        ]

    def test_timeout_is_swallowed(self, with_systemctl_path: str) -> None:
        """A wedged unit must not block callers that depend on the idempotent contract."""
        mock_run = MagicMock(side_effect=subprocess.TimeoutExpired(cmd=[], timeout=10))
        with patch("terok_sandbox._util._systemctl.subprocess.run", mock_run):
            _systemctl.run_best_effort("stop", "terok-vault.service")  # must not raise

    def test_toctou_missing_binary_is_swallowed(self, with_systemctl_path: str) -> None:
        """If ``systemctl`` vanishes between module-load resolve and call (pipx upgrade mid-call)."""
        mock_run = MagicMock(
            side_effect=FileNotFoundError(2, "No such file or directory", _FAKE_SYSTEMCTL)
        )
        with patch("terok_sandbox._util._systemctl.subprocess.run", mock_run):
            _systemctl.run_best_effort("disable", "terok-vault.socket")  # must not raise


class TestQuery:
    """``query`` — read-only variant that returns the captured result."""

    def test_returns_completed_process_with_stdout(self, with_systemctl_path: str) -> None:
        """Successful runs return the real ``CompletedProcess`` for caller inspection."""
        result = subprocess.CompletedProcess(args=[], returncode=0, stdout="active\n", stderr="")
        with patch("terok_sandbox._util._systemctl.subprocess.run", return_value=result) as run:
            got = _systemctl.query("is-active", "terok-vault.service")
        assert got.returncode == 0
        assert got.stdout == "active\n"
        # argv[0] is the absolute path resolved at module load (CWE-426 hardening).
        assert run.call_args.args[0] == [
            with_systemctl_path,
            "--user",
            "is-active",
            "terok-vault.service",
        ]

    def test_missing_systemctl_path_returns_synthetic_127(
        self, without_systemctl_path: None
    ) -> None:
        """Module-load resolution returned None → returncode 127, no subprocess call."""
        with patch("terok_sandbox._util._systemctl.subprocess.run") as run:
            result = _systemctl.query("is-system-running")
        run.assert_not_called()
        assert result.returncode == 127
        assert result.stdout == "" and result.stderr == ""

    def test_missing_binary_at_call_time_returns_synthetic_127(
        self, with_systemctl_path: str
    ) -> None:
        """Binary vanished between module load and call → returncode 127."""
        with patch(
            "terok_sandbox._util._systemctl.subprocess.run",
            side_effect=FileNotFoundError,
        ):
            result = _systemctl.query("is-system-running")
        assert result.returncode == 127
        assert result.stdout == "" and result.stderr == ""

    def test_timeout_returns_synthetic_124(self, with_systemctl_path: str) -> None:
        """A wedged call → returncode 124, also the shell convention."""
        with patch(
            "terok_sandbox._util._systemctl.subprocess.run",
            side_effect=subprocess.TimeoutExpired(cmd=[], timeout=5),
        ):
            result = _systemctl.query("status", "terok-vault.service")
        assert result.returncode == 124
        assert result.stdout == "" and result.stderr == ""


class TestFormatCaptured:
    """The internal output formatter copes with bytes, str, and None streams."""

    def test_str_stream_passes_through(self) -> None:
        """Already-decoded captured output is rendered verbatim."""
        rendered = _systemctl._format_captured("stdout text", "stderr text")
        assert "stderr text" in rendered
        assert "stdout text" in rendered

    def test_empty_streams_render_empty(self) -> None:
        """Both streams empty → no trailing ``"; …"`` suffix."""
        assert _systemctl._format_captured(None, None) == ""
        assert _systemctl._format_captured(b"", b"") == ""
