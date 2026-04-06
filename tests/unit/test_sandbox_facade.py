# SPDX-FileCopyrightText: 2026 Jiri Vyskocil
# SPDX-License-Identifier: Apache-2.0

"""Tests for the Sandbox facade class."""

from __future__ import annotations

import subprocess
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from terok_sandbox.runtime import GpuConfigError, check_gpu_error, redact_env_args
from terok_sandbox.sandbox import READY_MARKER, LifecycleHooks, RunSpec, Sandbox
from tests.constants import MOCK_TASK_DIR


def _make_spec(**overrides) -> RunSpec:
    """Build a RunSpec with sensible defaults, overridden by **overrides."""
    defaults: dict = {
        "container_name": "test-ctr",
        "image": "alpine:latest",
        "env": {"A": "1"},
        "volumes": ("/host:/ctr",),
        "command": ("bash",),
        "task_dir": MOCK_TASK_DIR,
    }
    defaults.update(overrides)
    return RunSpec(**defaults)


class TestRunSpec:
    """Verify RunSpec dataclass."""

    def test_frozen(self) -> None:
        spec = _make_spec()
        assert spec.container_name == "test-ctr"
        assert spec.gpu_enabled is False
        assert spec.extra_args == ()

    def test_security_defaults(self) -> None:
        """New security fields default to permissive-but-safe values."""
        spec = _make_spec()
        assert spec.unrestricted is True

    def test_restricted_mode(self) -> None:
        """Restricted spec carries through the frozen dataclass."""
        spec = _make_spec(unrestricted=False)
        assert spec.unrestricted is False


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
        with patch("terok_sandbox.gate.lifecycle.ensure_server_reachable") as mock:
            s = Sandbox()
            s.ensure_gate()
            mock.assert_called_once_with(s.config)

    def test_gate_status_delegates(self) -> None:
        from terok_sandbox.gate.lifecycle import GateServerStatus

        mock_status = GateServerStatus(mode="none", running=False, port=9418)
        with patch(
            "terok_sandbox.gate.lifecycle.get_server_status", return_value=mock_status
        ) as mock:
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

    def test_run_builds_podman_command(self) -> None:
        """Sandbox.run() assembles and executes a podman run command."""
        with (
            patch("subprocess.run") as mock_run,
            patch("terok_sandbox.sandbox.shlex"),
            patch("builtins.print"),
            patch(
                "terok_sandbox.shield.pre_start",
                return_value=["--annotation", "test=1"],
            ),
        ):
            s = Sandbox()
            spec = _make_spec()
            s.run(spec)

        mock_run.assert_called_once()
        cmd = mock_run.call_args[0][0]
        assert cmd[:3] == ["podman", "run", "-d"]
        assert "--name" in cmd
        assert "test-ctr" in cmd
        assert "-v" in cmd
        assert "-e" in cmd
        assert "alpine:latest" in cmd

    def test_run_restricted_adds_no_new_privileges(self) -> None:
        """Restricted spec adds --security-opt no-new-privileges."""
        with (
            patch("subprocess.run") as mock_run,
            patch("builtins.print"),
            patch("terok_sandbox.shield.pre_start", return_value=[]),
        ):
            s = Sandbox()
            s.run(_make_spec(unrestricted=False))

        cmd = mock_run.call_args[0][0]
        assert "--security-opt" in cmd
        assert "no-new-privileges" in cmd

    def test_run_unrestricted_skips_no_new_privileges(self) -> None:
        """Unrestricted spec does not add --security-opt."""
        with (
            patch("subprocess.run") as mock_run,
            patch("builtins.print"),
            patch("terok_sandbox.shield.pre_start", return_value=[]),
        ):
            s = Sandbox()
            s.run(_make_spec(unrestricted=True))

        cmd = mock_run.call_args[0][0]
        assert "no-new-privileges" not in " ".join(cmd)

    def test_run_bypass_shield_uses_bypass_args(self) -> None:
        """Bypass mode uses bypass_network_args when cfg.shield_bypass is set."""
        from terok_sandbox.config import SandboxConfig

        cfg = SandboxConfig(shield_bypass=True)
        with (
            patch("subprocess.run"),
            patch("builtins.print"),
            patch(
                "terok_sandbox.runtime.bypass_network_args",
                return_value=["--network", "pasta:-T,9418"],
            ) as mock_bypass,
            patch("terok_sandbox.shield.pre_start") as mock_shield,
        ):
            s = Sandbox(config=cfg)
            s.run(_make_spec())

        mock_bypass.assert_called_once()
        mock_shield.assert_not_called()

    def test_run_fires_lifecycle_hooks(self) -> None:
        """Sandbox.run() calls pre_start before podman and post_start after."""
        call_order: list[str] = []

        def track_pre() -> None:
            call_order.append("pre_start")

        def track_post() -> None:
            call_order.append("post_start")

        hooks = LifecycleHooks(pre_start=track_pre, post_start=track_post)

        with (
            patch("subprocess.run") as mock_run,
            patch("builtins.print"),
            patch("terok_sandbox.shield.pre_start", return_value=[]),
        ):
            # Track podman call position
            def track_podman(*args, **kwargs) -> None:
                call_order.append("podman")

            mock_run.side_effect = track_podman
            s = Sandbox()
            s.run(_make_spec(), hooks=hooks)

        assert call_order == ["pre_start", "podman", "post_start"]

    def test_run_raises_system_exit_on_failure(self) -> None:
        """Sandbox.run() raises SystemExit for non-GPU podman failures."""
        exc = subprocess.CalledProcessError(1, ["podman", "run"])
        exc.stderr = b"image not found"
        with (
            patch("subprocess.run", side_effect=exc),
            patch("builtins.print"),
            patch("terok_sandbox.shield.pre_start", return_value=[]),
        ):
            s = Sandbox()
            with pytest.raises(SystemExit, match="image not found"):
                s.run(_make_spec())

    def test_run_raises_gpu_error_on_cdi_failure(self) -> None:
        """Sandbox.run() raises GpuConfigError for CDI-related failures."""
        exc = subprocess.CalledProcessError(1, ["podman", "run"])
        exc.stderr = b"Error: nvidia.com/gpu=all: CDI device not found"
        with (
            patch("subprocess.run", side_effect=exc),
            patch("builtins.print"),
            patch("terok_sandbox.shield.pre_start", return_value=[]),
        ):
            s = Sandbox()
            with pytest.raises(GpuConfigError):
                s.run(_make_spec())


class TestLifecycleHooks:
    """Verify LifecycleHooks dataclass."""

    def test_all_none_by_default(self) -> None:
        hooks = LifecycleHooks()
        assert hooks.pre_start is None
        assert hooks.post_start is None
        assert hooks.post_ready is None
        assert hooks.post_stop is None

    def test_partial_construction(self) -> None:
        """Hooks can be partially specified."""
        fn = MagicMock()
        hooks = LifecycleHooks(pre_start=fn)
        assert hooks.pre_start is fn
        assert hooks.post_start is None

    def test_frozen(self) -> None:
        hooks = LifecycleHooks()
        with pytest.raises(AttributeError):
            hooks.pre_start = lambda: None  # type: ignore[misc]


class TestGpuConfigError:
    """Verify CDI error detection."""

    @pytest.mark.parametrize(
        ("stderr", "expects_raise"),
        [
            (b"Error: nvidia.com/gpu=all: device not found", True),
            (b"Error: cdi.k8s.io: registry not configured", True),
            (b"Error: CDI device injection failed", True),
            (b"Error: image not found", False),
            (b"", False),
            (None, False),
        ],
        ids=["nvidia-device", "cdi-k8s", "uppercase-cdi", "unrelated", "empty", "none"],
    )
    def test_check_gpu_error(self, stderr: bytes | None, expects_raise: bool) -> None:
        """check_gpu_error raises only for CDI patterns."""
        exc = subprocess.CalledProcessError(1, ["podman", "run"])
        exc.stderr = stderr
        if expects_raise:
            with pytest.raises(GpuConfigError):
                check_gpu_error(exc)
        else:
            check_gpu_error(exc)  # should not raise


class TestRedactEnvArgs:
    """Verify sensitive environment variable redaction."""

    def test_redacts_token_keys(self) -> None:
        cmd = ["podman", "run", "-e", "API_TOKEN=secret123", "-e", "PATH=/usr/bin"]
        result = redact_env_args(cmd)
        assert "API_TOKEN=<redacted>" in result
        assert "PATH=/usr/bin" in result

    def test_redacts_code_repo(self) -> None:
        cmd = ["-e", "CODE_REPO=http://tok@host:9418/repo"]
        result = redact_env_args(cmd)
        assert "CODE_REPO=<redacted>" in result

    def test_redacts_clone_from(self) -> None:
        cmd = ["-e", "CLONE_FROM=http://tok@host:9418/repo"]
        result = redact_env_args(cmd)
        assert "CLONE_FROM=<redacted>" in result

    def test_preserves_non_sensitive(self) -> None:
        cmd = ["-e", "TASK_ID=abc123"]
        result = redact_env_args(cmd)
        assert result == ["-e", "TASK_ID=abc123"]
