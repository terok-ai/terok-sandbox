# SPDX-FileCopyrightText: 2026 Jiri Vyskocil
# SPDX-License-Identifier: Apache-2.0

"""Tests for PodmanRuntime.exec and PodmanContainer lifecycle methods."""

from __future__ import annotations

import subprocess
from unittest.mock import patch

import pytest

from terok_sandbox import PodmanRuntime
from terok_sandbox.runtime.podman import (
    _DEFAULT_LOGIN_COMMAND,
    _START_TIMEOUT,
    _STOP_TIMEOUT_BUFFER,
)


class TestExec:
    """PodmanRuntime.exec delegates to ``podman exec`` with correct args."""

    @patch("terok_sandbox.runtime.podman.subprocess.run")
    def test_calls_podman_exec(self, mock_run) -> None:
        """Builds the expected argv and returns an [`ExecResult`][terok_sandbox.ExecResult]."""
        mock_run.return_value = subprocess.CompletedProcess(
            args=[], returncode=0, stdout="ok\n", stderr=""
        )
        runtime = PodmanRuntime()
        result = runtime.exec(runtime.container("mycontainer"), ["cat", "/etc/hostname"])

        mock_run.assert_called_once_with(
            ["podman", "exec", "mycontainer", "cat", "/etc/hostname"],
            capture_output=True,
            text=True,
            timeout=None,
            check=False,
        )
        assert result.exit_code == 0
        assert result.stdout == "ok\n"
        assert result.ok

    @patch("terok_sandbox.runtime.podman.subprocess.run")
    def test_custom_timeout(self, mock_run) -> None:
        """*timeout* propagates to the subprocess call."""
        mock_run.return_value = subprocess.CompletedProcess(
            args=[], returncode=0, stdout="", stderr=""
        )
        runtime = PodmanRuntime()
        runtime.exec(runtime.container("c1"), ["true"], timeout=5)

        assert mock_run.call_args[1]["timeout"] == 5

    @patch("terok_sandbox.runtime.podman.subprocess.run")
    def test_nonzero_returncode_does_not_raise(self, mock_run) -> None:
        """Non-zero exit codes surface in ``ExecResult`` rather than raising."""
        mock_run.return_value = subprocess.CompletedProcess(
            args=[], returncode=1, stdout="", stderr="fail"
        )
        runtime = PodmanRuntime()
        result = runtime.exec(runtime.container("c1"), ["false"])

        assert result.exit_code == 1
        assert result.stderr == "fail"
        assert not result.ok

    @patch("terok_sandbox.runtime.podman.subprocess.run")
    def test_filenotfounderror_propagates(self, mock_run) -> None:
        """Missing podman binary propagates unchanged."""
        mock_run.side_effect = FileNotFoundError("podman")

        runtime = PodmanRuntime()
        with pytest.raises(FileNotFoundError):
            runtime.exec(runtime.container("c1"), ["true"])

    @patch("terok_sandbox.runtime.podman.subprocess.run")
    def test_timeout_expired_propagates(self, mock_run) -> None:
        """TimeoutExpired propagates unchanged."""
        mock_run.side_effect = subprocess.TimeoutExpired(cmd="podman", timeout=30)

        runtime = PodmanRuntime()
        with pytest.raises(subprocess.TimeoutExpired):
            runtime.exec(runtime.container("c1"), ["sleep", "99"], timeout=30)


class TestLoginCommand:
    """PodmanContainer.login_command builds a podman exec -it argv."""

    def test_default_tmux_session(self) -> None:
        """Default command uses the tmux session."""
        container = PodmanRuntime().container("proj-cli-1")
        result = container.login_command()

        assert result == ["podman", "exec", "-it", "proj-cli-1", *_DEFAULT_LOGIN_COMMAND]

    def test_custom_command(self) -> None:
        """Explicit command overrides the default."""
        container = PodmanRuntime().container("proj-cli-1")
        result = container.login_command(command=("bash",))

        assert result == ["podman", "exec", "-it", "proj-cli-1", "bash"]

    def test_no_subprocess_call(self) -> None:
        """login_command is pure — it never touches subprocess."""
        with patch("terok_sandbox.runtime.podman.subprocess.run") as mock_run:
            PodmanRuntime().container("c1").login_command()
            mock_run.assert_not_called()


class TestContainerStart:
    """PodmanContainer.start delegates to ``podman start``."""

    @patch("terok_sandbox.runtime.podman.subprocess.run")
    def test_calls_podman_start(self, mock_run) -> None:
        """Builds the expected argv and returns without raising on success."""
        mock_run.return_value = subprocess.CompletedProcess(args=[], returncode=0, stderr="")
        PodmanRuntime().container("proj-cli-1").start()

        mock_run.assert_called_once_with(
            ["podman", "start", "proj-cli-1"],
            check=False,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.PIPE,
            text=True,
            timeout=_START_TIMEOUT,
        )

    @patch("terok_sandbox.runtime.podman.subprocess.run")
    def test_failure_raises_runtimeerror(self, mock_run) -> None:
        """Non-zero returncode is translated to [`RuntimeError`][RuntimeError]."""
        mock_run.return_value = subprocess.CompletedProcess(
            args=[], returncode=125, stderr="Error: no such container\n"
        )
        with pytest.raises(RuntimeError, match="no such container"):
            PodmanRuntime().container("gone").start()

    @patch("terok_sandbox.runtime.podman.subprocess.run")
    def test_missing_podman_is_runtimeerror(self, mock_run) -> None:
        """Missing podman is normalised to RuntimeError (chains the original)."""
        mock_run.side_effect = FileNotFoundError("podman")

        with pytest.raises(RuntimeError, match="podman not found") as exc_info:
            PodmanRuntime().container("c1").start()
        assert isinstance(exc_info.value.__cause__, FileNotFoundError)

    @patch("terok_sandbox.runtime.podman.subprocess.run")
    def test_timeout_is_runtimeerror(self, mock_run) -> None:
        """Timeout is normalised to RuntimeError with the elapsed seconds."""
        mock_run.side_effect = subprocess.TimeoutExpired("podman", 30)

        with pytest.raises(RuntimeError, match=r"timed out after \d+s") as exc_info:
            PodmanRuntime().container("c1").start()
        assert isinstance(exc_info.value.__cause__, subprocess.TimeoutExpired)


class TestContainerStop:
    """PodmanContainer.stop delegates to ``podman stop --time``."""

    @patch("terok_sandbox.runtime.podman.subprocess.run")
    def test_calls_podman_stop_with_default_timeout(self, mock_run) -> None:
        """Default timeout is 10 seconds."""
        mock_run.return_value = subprocess.CompletedProcess(args=[], returncode=0, stderr="")
        PodmanRuntime().container("proj-cli-1").stop()

        mock_run.assert_called_once_with(
            ["podman", "stop", "--time", "10", "proj-cli-1"],
            check=False,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.PIPE,
            text=True,
            timeout=10 + _STOP_TIMEOUT_BUFFER,
        )

    @patch("terok_sandbox.runtime.podman.subprocess.run")
    def test_custom_timeout(self, mock_run) -> None:
        """Custom *timeout* feeds into both the argv and the subprocess timeout."""
        mock_run.return_value = subprocess.CompletedProcess(args=[], returncode=0, stderr="")
        PodmanRuntime().container("c1").stop(timeout=30)

        args = mock_run.call_args[0][0]
        assert args == ["podman", "stop", "--time", "30", "c1"]
        assert mock_run.call_args[1]["timeout"] == 30 + _STOP_TIMEOUT_BUFFER

    @patch("terok_sandbox.runtime.podman.subprocess.run")
    def test_failure_raises_runtimeerror(self, mock_run) -> None:
        """Non-zero returncode is translated to [`RuntimeError`][RuntimeError]."""
        mock_run.return_value = subprocess.CompletedProcess(
            args=[], returncode=125, stderr="Error: no such container\n"
        )
        with pytest.raises(RuntimeError, match="no such container"):
            PodmanRuntime().container("gone").stop()

    @patch("terok_sandbox.runtime.podman.subprocess.run")
    def test_missing_podman_is_runtimeerror(self, mock_run) -> None:
        """Missing podman is normalised to RuntimeError (chains the original)."""
        mock_run.side_effect = FileNotFoundError("podman")

        with pytest.raises(RuntimeError, match="podman not found") as exc_info:
            PodmanRuntime().container("c1").stop()
        assert isinstance(exc_info.value.__cause__, FileNotFoundError)

    @patch("terok_sandbox.runtime.podman.subprocess.run")
    def test_timeout_is_runtimeerror(self, mock_run) -> None:
        """Timeout is normalised to RuntimeError with the elapsed seconds."""
        mock_run.side_effect = subprocess.TimeoutExpired("podman", 15)

        with pytest.raises(RuntimeError, match=r"timed out after \d+s") as exc_info:
            PodmanRuntime().container("c1").stop()
        assert isinstance(exc_info.value.__cause__, subprocess.TimeoutExpired)
