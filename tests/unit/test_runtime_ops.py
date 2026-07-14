# SPDX-FileCopyrightText: 2026 Jiri Vyskocil
# SPDX-License-Identifier: Apache-2.0

"""Tests for PodmanRuntime.exec and PodmanContainer lifecycle methods."""

from __future__ import annotations

import subprocess
from pathlib import Path
from unittest.mock import MagicMock, PropertyMock, patch

import pytest

from terok_sandbox import PodmanRuntime
from terok_sandbox.runtime.podman import (
    _DEFAULT_LOGIN_COMMAND,
    _START_TIMEOUT,
    _STOP_CLEANUP_TIMEOUT,
    _STOP_KILL_TIMEOUT,
    PodmanContainer,
    find_init_binary,
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


class TestFindInitBinary:
    """The probe mirrors podman's default helper-binaries search — nothing more."""

    def test_finds_catatonit_in_helper_dir(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """First helper dir containing the binary wins."""
        from terok_sandbox.runtime import podman as podman_mod

        (tmp_path / "catatonit").touch()
        monkeypatch.setattr(podman_mod, "_INIT_HELPER_DIRS", (tmp_path,))

        assert find_init_binary() == str(tmp_path / "catatonit")

    def test_missing_everywhere_returns_none(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """No helper dir has it → None; PATH is deliberately not consulted."""
        from terok_sandbox.runtime import podman as podman_mod

        monkeypatch.setattr(podman_mod, "_INIT_HELPER_DIRS", (tmp_path,))

        assert find_init_binary() is None


def _stop_client(*, returncode: int = 0, stderr: str = "", hanging_polls: int = 0) -> MagicMock:
    """Fake ``podman stop`` Popen: hang *hanging_polls* communicate() calls, then exit."""
    proc = MagicMock()
    proc.returncode = returncode
    proc.communicate.side_effect = [
        *([subprocess.TimeoutExpired("podman stop", 1)] * hanging_polls),
        ("", stderr),
    ]
    return proc


class TestContainerStop:
    """PodmanContainer.stop watches the container state, not a wall clock."""

    @patch("terok_sandbox.runtime.podman.subprocess.Popen")
    def test_calls_podman_stop_with_default_timeout(self, mock_popen) -> None:
        """Default grace period is 10 seconds."""
        mock_popen.return_value = _stop_client()
        PodmanRuntime().container("proj-cli-1").stop()

        mock_popen.assert_called_once_with(
            ["podman", "stop", "--time", "10", "proj-cli-1"],
            stdout=subprocess.DEVNULL,
            stderr=subprocess.PIPE,
            text=True,
        )

    @patch("terok_sandbox.runtime.podman.subprocess.Popen")
    def test_custom_timeout(self, mock_popen) -> None:
        """Custom *timeout* feeds the ``--time`` argument."""
        mock_popen.return_value = _stop_client()
        PodmanRuntime().container("c1").stop(timeout=30)

        assert mock_popen.call_args[0][0] == ["podman", "stop", "--time", "30", "c1"]

    @patch("terok_sandbox.runtime.podman.subprocess.Popen")
    def test_failure_raises_runtimeerror(self, mock_popen) -> None:
        """Non-zero returncode is translated to [`RuntimeError`][RuntimeError]."""
        mock_popen.return_value = _stop_client(returncode=125, stderr="Error: no such container\n")
        with pytest.raises(RuntimeError, match="no such container"):
            PodmanRuntime().container("gone").stop()

    @patch("terok_sandbox.runtime.podman.subprocess.Popen")
    def test_missing_podman_is_runtimeerror(self, mock_popen) -> None:
        """Missing podman is normalised to RuntimeError (chains the original)."""
        mock_popen.side_effect = FileNotFoundError("podman")

        with pytest.raises(RuntimeError, match="podman not found") as exc_info:
            PodmanRuntime().container("c1").stop()
        assert isinstance(exc_info.value.__cause__, FileNotFoundError)

    @patch.object(PodmanContainer, "state", new_callable=PropertyMock, return_value="running")
    @patch("terok_sandbox.runtime.podman.time.monotonic")
    @patch("terok_sandbox.runtime.podman.subprocess.Popen")
    def test_grace_period_is_never_cut_short(self, mock_popen, mock_clock, _state) -> None:
        """A running container gets its whole grace period plus the kill margin.

        Regression guard for the old grace+5s wall-clock guess: a stop
        that is *legitimately* still terminating well past that mark
        must be waited out, not aborted.
        """
        mock_popen.return_value = _stop_client(hanging_polls=2)
        mock_clock.side_effect = [0.0, 12.0, 20.0]

        PodmanRuntime().container("c1").stop(timeout=10)

        mock_popen.return_value.kill.assert_not_called()

    @patch.object(PodmanContainer, "state", new_callable=PropertyMock, return_value="running")
    @patch("terok_sandbox.runtime.podman.time.monotonic")
    @patch("terok_sandbox.runtime.podman.subprocess.Popen")
    def test_wedged_kill_aborts(self, mock_popen, mock_clock, _state) -> None:
        """Still ``running`` past grace + kill margin → the client is killed."""
        proc = _stop_client(hanging_polls=1000)
        mock_popen.return_value = proc
        mock_clock.side_effect = [0.0, 10.0 + _STOP_KILL_TIMEOUT + 1.0]

        with pytest.raises(RuntimeError, match="still running"):
            PodmanRuntime().container("c1").stop(timeout=10)
        proc.kill.assert_called_once()

    @patch.object(PodmanContainer, "state", new_callable=PropertyMock, return_value="exited")
    @patch("terok_sandbox.runtime.podman.time.monotonic")
    @patch("terok_sandbox.runtime.podman.subprocess.Popen")
    def test_wedged_cleanup_aborts(self, mock_popen, mock_clock, _state) -> None:
        """Container dead but the client never returns → cleanup deadline fires."""
        proc = _stop_client(hanging_polls=1000)
        mock_popen.return_value = proc
        mock_clock.side_effect = [0.0, 1.0, 1.0 + _STOP_CLEANUP_TIMEOUT + 1.0]

        with pytest.raises(RuntimeError, match="cleanup did not finish"):
            PodmanRuntime().container("c1").stop(timeout=10)
        proc.kill.assert_called_once()

    @patch.object(PodmanContainer, "state", new_callable=PropertyMock, return_value="exited")
    @patch("terok_sandbox.runtime.podman.subprocess.Popen")
    def test_early_death_enters_cleanup_phase(self, mock_popen, _state) -> None:
        """A container that dies inside the grace period finishes cleanly."""
        mock_popen.return_value = _stop_client(hanging_polls=1)

        PodmanRuntime().container("c1").stop(timeout=10)

        mock_popen.return_value.kill.assert_not_called()
