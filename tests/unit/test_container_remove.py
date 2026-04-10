# SPDX-FileCopyrightText: 2026 Jiri Vyskocil
# SPDX-License-Identifier: Apache-2.0

"""Tests for stop_task_containers and ContainerRemoveResult."""

from __future__ import annotations

import subprocess
from unittest.mock import patch

import pytest

from terok_sandbox.runtime import ContainerRemoveResult, stop_task_containers


class TestContainerRemoveResult:
    """ContainerRemoveResult dataclass."""

    def test_frozen(self) -> None:
        r = ContainerRemoveResult(name="c1", removed=True)
        with pytest.raises(AttributeError):
            r.name = "c2"  # type: ignore[misc]

    def test_success_defaults_error_none(self) -> None:
        r = ContainerRemoveResult(name="c1", removed=True)
        assert r.error is None

    def test_failure_carries_reason(self) -> None:
        r = ContainerRemoveResult(name="c1", removed=False, error="busy")
        assert r.error == "busy"


class TestStopTaskContainers:
    """Per-container result reporting from stop_task_containers."""

    @patch("terok_sandbox.runtime.subprocess.run")
    def test_successful_removal(self, mock_run) -> None:
        mock_run.return_value = subprocess.CompletedProcess(args=[], returncode=0, stderr="")
        results = stop_task_containers(["c1", "c2"])

        assert len(results) == 2
        assert [r.name for r in results] == ["c1", "c2"]
        assert all(r.removed for r in results)
        assert all(r.error is None for r in results)

    @patch("terok_sandbox.runtime.subprocess.run")
    def test_no_such_container_is_success(self, mock_run) -> None:
        """'No such container' means the goal is already achieved."""
        mock_run.return_value = subprocess.CompletedProcess(
            args=[], returncode=1, stderr="Error: no such container: c1\n"
        )
        [result] = stop_task_containers(["c1"])

        assert result.removed is True
        assert result.error is None

    @patch("terok_sandbox.runtime.subprocess.run")
    def test_podman_error_reports_failure(self, mock_run) -> None:
        mock_run.return_value = subprocess.CompletedProcess(
            args=[], returncode=125, stderr="Error: container is locked\n"
        )
        [result] = stop_task_containers(["c1"])

        assert result.removed is False
        assert result.error == "Error: container is locked"

    @patch("terok_sandbox.runtime.subprocess.run")
    def test_podman_error_with_empty_stderr(self, mock_run) -> None:
        """Fall back to exit code when stderr is empty."""
        mock_run.return_value = subprocess.CompletedProcess(args=[], returncode=42, stderr="")
        [result] = stop_task_containers(["c1"])

        assert result.removed is False
        assert result.error == "exit code 42"

    @patch("terok_sandbox.runtime.subprocess.run")
    def test_timeout_reports_failure(self, mock_run) -> None:
        mock_run.side_effect = subprocess.TimeoutExpired(cmd="podman", timeout=120)
        [result] = stop_task_containers(["c1"])

        assert result.removed is False
        assert "timed out" in result.error  # type: ignore[operator]

    @patch("terok_sandbox.runtime.subprocess.run")
    def test_podman_not_found(self, mock_run) -> None:
        mock_run.side_effect = FileNotFoundError("podman")
        [result] = stop_task_containers(["c1"])

        assert result.removed is False
        assert result.error == "podman not found"

    @patch("terok_sandbox.runtime.subprocess.run")
    def test_unexpected_exception(self, mock_run) -> None:
        mock_run.side_effect = OSError("permission denied")
        [result] = stop_task_containers(["c1"])

        assert result.removed is False
        assert "permission denied" in result.error  # type: ignore[operator]

    @patch("terok_sandbox.runtime.subprocess.run")
    def test_mixed_results_all_attempted(self, mock_run) -> None:
        """All containers are attempted regardless of individual failures."""
        mock_run.side_effect = [
            subprocess.CompletedProcess(args=[], returncode=0, stderr=""),
            subprocess.CompletedProcess(args=[], returncode=125, stderr="Error: locked\n"),
            subprocess.CompletedProcess(args=[], returncode=1, stderr="no such container\n"),
        ]
        results = stop_task_containers(["c1", "c2", "c3"])

        assert len(results) == 3
        assert [r.name for r in results] == ["c1", "c2", "c3"]
        assert results[0].removed is True
        assert results[1].removed is False
        assert results[2].removed is True  # no such container = success

    def test_empty_list(self) -> None:
        assert stop_task_containers([]) == []
