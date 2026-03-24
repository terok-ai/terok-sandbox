# SPDX-FileCopyrightText: 2026 Jiri Vyskocil
# SPDX-License-Identifier: Apache-2.0

"""Tests for the Sandbox facade class."""

from __future__ import annotations

from pathlib import Path
from unittest.mock import patch

from terok_sandbox.sandbox import READY_MARKER, RunSpec, Sandbox


class TestRunSpec:
    """Verify RunSpec dataclass."""

    def test_frozen(self) -> None:
        spec = RunSpec(
            container_name="test",
            image="img:latest",
            env={"A": "1"},
            volumes=("/a:/b",),
            command=("bash",),
            task_dir=Path("/tmp/task"),
        )
        assert spec.container_name == "test"
        assert spec.gpu_enabled is False
        assert spec.extra_args == ()


class TestReadyMarker:
    """Verify READY_MARKER constant."""

    def test_matches_init_script_output(self) -> None:
        assert "init complete" in READY_MARKER


class TestSandbox:
    """Verify Sandbox facade delegates correctly."""

    def test_default_config(self) -> None:
        s = Sandbox()
        assert s.config is not None

    def test_custom_config(self) -> None:
        from terok_sandbox.config import SandboxConfig

        cfg = SandboxConfig()
        s = Sandbox(config=cfg)
        assert s.config is cfg

    def test_gate_url(self) -> None:
        s = Sandbox()
        base = s.config.gate_base_path
        repo = base / "my-project"
        url = s.gate_url(repo, "tok123")
        assert "tok123@" in url
        assert "my-project" in url
        assert url.startswith("http://")

    def test_ensure_gate_delegates(self) -> None:
        with patch("terok_sandbox.gate_server.ensure_server_reachable") as mock:
            s = Sandbox()
            s.ensure_gate()
            mock.assert_called_once_with(s.config)

    def test_gate_status_delegates(self) -> None:
        from terok_sandbox.gate_server import GateServerStatus

        mock_status = GateServerStatus(mode="none", running=False, port=9418)
        with patch("terok_sandbox.gate_server.get_server_status", return_value=mock_status) as mock:
            s = Sandbox()
            result = s.gate_status()
            assert result == mock_status
            mock.assert_called_once_with(s.config)

    def test_shield_down_delegates(self) -> None:
        with patch("terok_sandbox.shield.down") as mock:
            s = Sandbox()
            s.shield_down("ctr", Path("/tmp/task"))
            mock.assert_called_once_with("ctr", Path("/tmp/task"), cfg=s.config)

    def test_pre_start_args_delegates(self) -> None:
        with patch("terok_sandbox.shield.pre_start", return_value=["--hook"]) as mock:
            s = Sandbox()
            result = s.pre_start_args("ctr", Path("/tmp/task"))
            assert result == ["--hook"]
            mock.assert_called_once_with("ctr", Path("/tmp/task"), s.config)

    def test_stop_delegates(self) -> None:
        with patch("terok_sandbox.runtime.stop_task_containers") as mock:
            s = Sandbox()
            s.stop(["c1", "c2"])
            mock.assert_called_once_with(["c1", "c2"])

    def test_stream_logs_uses_ready_marker(self) -> None:
        with patch("terok_sandbox.runtime.stream_initial_logs", return_value=True) as mock:
            s = Sandbox()
            result = s.stream_logs("ctr", timeout=30.0)
            assert result is True
            check_fn = mock.call_args[0][2]
            assert check_fn(">> init complete")
            assert not check_fn("still waiting")
