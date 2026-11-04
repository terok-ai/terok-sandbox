# SPDX-FileCopyrightText: 2026 Jiri Vyskocil
# SPDX-License-Identifier: Apache-2.0

"""Tests for PodmanRuntime.exec and PodmanContainer lifecycle methods."""

from __future__ import annotations

import subprocess
from unittest.mock import MagicMock, patch

import pytest

from terok_sandbox import PodmanRuntime
from terok_sandbox.runtime import EXEC_OUTPUT_LIMIT, ExecOutputLimitError, ExecResult
from terok_sandbox.runtime.podman import (
    _DEFAULT_LOGIN_COMMAND,
    _START_TIMEOUT,
    _STATE_ABSENT,
    _STOP_CLEANUP_TIMEOUT,
    _STOP_KILL_TIMEOUT,
    _STOP_PROBE_TIMEOUT,
    PodmanContainer,
    init_binary_unavailable,
)
from tests.constants import MISSING_PODMAN_INIT, MOCK_BASE, MOCK_HOST_BIN


class TestExec:
    """PodmanRuntime.exec runs ``podman exec`` through the shared bounded capture."""

    @patch("terok_sandbox.runtime.podman.run_captured")
    def test_calls_podman_exec(self, capture) -> None:
        """Builds the expected argv and returns the captured [`ExecResult`][terok_sandbox.ExecResult]."""
        capture.return_value = ExecResult(exit_code=0, stdout="ok\n", stderr="")
        runtime = PodmanRuntime()
        result = runtime.exec(runtime.container("mycontainer"), ["cat", "/etc/hostname"])

        capture.assert_called_once_with(
            ["podman", "exec", "mycontainer", "cat", "/etc/hostname"],
            executable=str(MOCK_HOST_BIN / "podman"),
            timeout=None,
            max_output_bytes=EXEC_OUTPUT_LIMIT,
        )
        assert result is capture.return_value

    @patch("terok_sandbox.runtime.podman.run_captured")
    def test_timeout_and_budget_propagate(self, capture) -> None:
        """*timeout* and *max_output_bytes* reach the capture unchanged."""
        runtime = PodmanRuntime()
        runtime.exec(runtime.container("c1"), ["true"], timeout=5, max_output_bytes=100)

        assert capture.call_args.kwargs["timeout"] == 5
        assert capture.call_args.kwargs["max_output_bytes"] == 100

    @pytest.mark.parametrize(
        "error",
        [
            FileNotFoundError("podman"),
            subprocess.TimeoutExpired(cmd="podman", timeout=30),
            ExecOutputLimitError("Command output exceeded 1 bytes"),
        ],
        ids=["missing-podman", "timeout", "overflow"],
    )
    @patch("terok_sandbox.runtime.podman.run_captured")
    def test_capture_failures_propagate(self, capture, error) -> None:
        """A missing binary, a deadline and an overflow surface unchanged to the caller."""
        capture.side_effect = error
        runtime = PodmanRuntime()
        with pytest.raises(type(error)):
            runtime.exec(runtime.container("c1"), ["true"])


class TestLoginCommand:
    """PodmanContainer.login_command builds a podman exec -it argv."""

    def test_default_tmux_session(self) -> None:
        """Default command uses the tmux session."""
        container = PodmanRuntime().container("proj-cli-1")
        result = container.login_command()

        assert result == [
            str(MOCK_HOST_BIN / "podman"),
            "exec",
            "-it",
            "proj-cli-1",
            *_DEFAULT_LOGIN_COMMAND,
        ]

    def test_custom_command(self) -> None:
        """Explicit command overrides the default."""
        container = PodmanRuntime().container("proj-cli-1")
        result = container.login_command(command=("bash",))

        assert result == [str(MOCK_HOST_BIN / "podman"), "exec", "-it", "proj-cli-1", "bash"]

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
            executable=str(MOCK_HOST_BIN / "podman"),
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


class TestInitBinaryUnavailable:
    """Only the early default-helper lookup failure permits degraded launch."""

    @pytest.mark.parametrize(
        "stderr",
        [
            MISSING_PODMAN_INIT,
            MISSING_PODMAN_INIT.decode(),
            b"Pulling image\n" + MISSING_PODMAN_INIT,
        ],
    )
    def test_default_helper_missing(self, stderr) -> None:
        assert init_binary_unavailable(subprocess.CalledProcessError(125, "podman", stderr=stderr))

    @pytest.mark.parametrize(
        ("code", "stderr"),
        [
            (1, MISSING_PODMAN_INIT),
            (125, None),
            (125, b""),
            (125, b"Error: image not found"),
            (125, b"Error: OCI hook failed: " + MISSING_PODMAN_INIT),
            (125, b"Error: OCI hook failed:\n" + MISSING_PODMAN_INIT),
            (125, MISSING_PODMAN_INIT + b"Error: OCI runtime failed\n"),
            (125, f"Error: container-init binary not found on the host: stat {MOCK_BASE / 'init'}"),
            (125, b"Error: lookup init binary: permission denied"),
            (
                125,
                b'Error: lookup init binary: exec: "catatonit": cannot run executable found relative to current directory',
            ),
        ],
    )
    def test_other_failures_are_not_missing_default_helper(self, code, stderr) -> None:
        assert not init_binary_unavailable(
            subprocess.CalledProcessError(code, "podman", stderr=stderr)
        )


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
            executable=str(MOCK_HOST_BIN / "podman"),
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

    @patch.object(PodmanContainer, "_probe_stop_state", return_value="running")
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

    @patch.object(PodmanContainer, "_probe_stop_state", return_value="running")
    @patch("terok_sandbox.runtime.podman.time.monotonic")
    @patch("terok_sandbox.runtime.podman.subprocess.Popen")
    def test_wedged_kill_aborts(self, mock_popen, mock_clock, _state) -> None:
        """Still ``running`` past grace + kill margin → the client is killed."""
        proc = _stop_client(hanging_polls=1000)
        mock_popen.return_value = proc
        mock_clock.side_effect = [0.0, 10.0 + _STOP_KILL_TIMEOUT + 1.0]
        container = PodmanRuntime().container("c1")

        with pytest.raises(RuntimeError, match="still running"):
            container.stop(timeout=10)
        proc.kill.assert_called_once()

    @patch.object(PodmanContainer, "_probe_stop_state", return_value="exited")
    @patch("terok_sandbox.runtime.podman.time.monotonic")
    @patch("terok_sandbox.runtime.podman.subprocess.Popen")
    def test_wedged_cleanup_aborts(self, mock_popen, mock_clock, _state) -> None:
        """Container dead but the client never returns → cleanup deadline fires."""
        proc = _stop_client(hanging_polls=1000)
        mock_popen.return_value = proc
        mock_clock.side_effect = [0.0, 1.0, 1.0 + _STOP_CLEANUP_TIMEOUT + 1.0]
        container = PodmanRuntime().container("c1")

        with pytest.raises(RuntimeError, match="cleanup did not finish"):
            container.stop(timeout=10)
        proc.kill.assert_called_once()

    @patch.object(PodmanContainer, "_probe_stop_state", return_value="exited")
    @patch("terok_sandbox.runtime.podman.subprocess.Popen")
    def test_early_death_enters_cleanup_phase(self, mock_popen, _state) -> None:
        """A container that dies inside the grace period finishes cleanly."""
        mock_popen.return_value = _stop_client(hanging_polls=1)

        PodmanRuntime().container("c1").stop(timeout=10)

        mock_popen.return_value.kill.assert_not_called()

    @patch.object(PodmanContainer, "_probe_stop_state", return_value=_STATE_ABSENT)
    @patch("terok_sandbox.runtime.podman.subprocess.Popen")
    def test_absent_container_counts_as_down(self, mock_popen, _state) -> None:
        """A container podman no longer knows is confirmed down, not unknown."""
        mock_popen.return_value = _stop_client(hanging_polls=1)

        PodmanRuntime().container("c1").stop(timeout=10)

        mock_popen.return_value.kill.assert_not_called()

    @patch.object(PodmanContainer, "_probe_stop_state", return_value=None)
    @patch("terok_sandbox.runtime.podman.time.monotonic")
    @patch("terok_sandbox.runtime.podman.subprocess.Popen")
    def test_unknown_state_keeps_phase_until_hard_ceiling(
        self, mock_popen, mock_clock, _state
    ) -> None:
        """No probe verdict → no phase flip; the combined-budget ceiling still bounds it.

        Past the kill deadline an *unknown* state must not be treated as
        running (no premature kill-abort) nor as stopped (no premature
        cleanup phase) — but a state that stays unobservable cannot stall
        the watch forever either.
        """
        proc = _stop_client(hanging_polls=1000)
        mock_popen.return_value = proc
        past_kill = 10.0 + _STOP_KILL_TIMEOUT + 1.0
        past_ceiling = 10.0 + _STOP_KILL_TIMEOUT + _STOP_CLEANUP_TIMEOUT + 1.0
        mock_clock.side_effect = [0.0, past_kill, past_ceiling]
        container = PodmanRuntime().container("c1")

        with pytest.raises(RuntimeError, match="no stop verdict"):
            container.stop(timeout=10)
        proc.kill.assert_called_once()


class TestProbeStopState:
    """The stop watch's state probe is bounded and three-way."""

    @patch("terok_sandbox.runtime.podman.subprocess.check_output")
    def test_inspectable_container_reports_its_state(self, mock_out) -> None:
        """A successful inspect yields the lowercased state string."""
        mock_out.return_value = "Running\n"

        assert PodmanRuntime().container("c1")._probe_stop_state() == "running"
        assert mock_out.call_args[1]["timeout"] == _STOP_PROBE_TIMEOUT

    @patch("terok_sandbox.runtime.podman.subprocess.check_output")
    def test_unknown_container_is_confirmed_absent(self, mock_out) -> None:
        """Inspect failing with a non-zero rc means podman no longer knows it."""
        mock_out.side_effect = subprocess.CalledProcessError(125, "podman inspect")

        assert PodmanRuntime().container("gone")._probe_stop_state() == _STATE_ABSENT

    @patch("terok_sandbox.runtime.podman.subprocess.check_output")
    def test_hung_inspect_gives_no_verdict(self, mock_out) -> None:
        """A probe timeout is *no verdict* — never mistaken for stopped."""
        mock_out.side_effect = subprocess.TimeoutExpired("podman inspect", _STOP_PROBE_TIMEOUT)

        assert PodmanRuntime().container("c1")._probe_stop_state() is None
