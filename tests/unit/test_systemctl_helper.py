# SPDX-FileCopyrightText: 2026 Jiri Vyskocil
# SPDX-License-Identifier: Apache-2.0

"""Tests for the shared ``systemctl --user`` invocation helpers."""

from __future__ import annotations

import subprocess
from unittest.mock import MagicMock, patch

import pytest

from terok_sandbox._util import _systemctl


class TestRun:
    """``run`` — raise ``SystemExit`` with captured stderr on failure."""

    def test_happy_path_runs_systemctl(self) -> None:
        with patch("terok_sandbox._util._systemctl.subprocess.run") as run:
            _systemctl.run("daemon-reload")
        assert run.call_args.args[0] == ["systemctl", "--user", "daemon-reload"]

    def test_failure_re_raises_as_systemexit_with_stderr(self) -> None:
        """The re-raise flattens ``CalledProcessError`` into a readable message."""
        err = subprocess.CalledProcessError(
            returncode=1, cmd=[], stderr=b"Failed to connect to bus"
        )
        with patch("terok_sandbox._util._systemctl.subprocess.run", side_effect=err):
            with pytest.raises(SystemExit) as exc:
                _systemctl.run("enable", "--now", "terok-vault.socket")
        assert "Failed to connect to bus" in str(exc.value)


class TestRunBestEffort:
    """``run_best_effort`` — swallow missing systemctl, non-zero exit, and timeouts."""

    def test_missing_systemctl_returns_silently(self) -> None:
        with (
            patch("terok_sandbox._util._systemctl.shutil.which", return_value=None),
            patch("terok_sandbox._util._systemctl.subprocess.run") as run,
        ):
            _systemctl.run_best_effort("daemon-reload")
        run.assert_not_called()

    def test_invokes_systemctl_with_verb_and_args(self) -> None:
        with (
            patch("terok_sandbox._util._systemctl.shutil.which", return_value="/usr/bin/systemctl"),
            patch("terok_sandbox._util._systemctl.subprocess.run") as run,
        ):
            _systemctl.run_best_effort("disable", "--now", "terok-gate.socket")
        run.assert_called_once()
        assert run.call_args.args[0] == [
            "systemctl",
            "--user",
            "disable",
            "--now",
            "terok-gate.socket",
        ]

    def test_timeout_is_swallowed(self) -> None:
        """A wedged unit must not block callers that depend on the idempotent contract."""
        mock_run = MagicMock(side_effect=subprocess.TimeoutExpired(cmd=[], timeout=10))
        with (
            patch("terok_sandbox._util._systemctl.shutil.which", return_value="/usr/bin/systemctl"),
            patch("terok_sandbox._util._systemctl.subprocess.run", mock_run),
        ):
            _systemctl.run_best_effort("stop", "terok-vault.service")  # must not raise
