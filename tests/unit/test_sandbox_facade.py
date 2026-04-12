# SPDX-FileCopyrightText: 2026 Jiri Vyskocil
# SPDX-License-Identifier: Apache-2.0

"""Tests for the Sandbox facade class."""

from __future__ import annotations

import subprocess
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from terok_sandbox.runtime import GpuConfigError, check_gpu_error, redact_env_args
from terok_sandbox.sandbox import (
    READY_MARKER,
    LifecycleHooks,
    RunSpec,
    Sandbox,
    Sharing,
    VolumeSpec,
)
from tests.constants import MOCK_BASE, MOCK_TASK_DIR

MOCK_HOST_DIR = MOCK_BASE / "host-dir"


def _make_spec(**overrides) -> RunSpec:
    """Build a RunSpec with sensible defaults, overridden by **overrides."""
    defaults: dict = {
        "container_name": "test-ctr",
        "image": "alpine:latest",
        "env": {"A": "1"},
        "volumes": (VolumeSpec(MOCK_HOST_DIR, "/ctr"),),
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


class TestVolumeSpec:
    """Verify VolumeSpec dataclass."""

    def test_to_mount_arg_shared_default(self) -> None:
        vol = VolumeSpec(Path("/host/data"), "/container/data")
        assert vol.to_mount_arg() == "/host/data:/container/data:z"

    def test_to_mount_arg_private(self) -> None:
        from terok_sandbox.sandbox import Sharing

        vol = VolumeSpec(Path("/host/ws"), "/workspace", sharing=Sharing.PRIVATE)
        assert vol.to_mount_arg() == "/host/ws:/workspace:Z"

    def test_frozen(self) -> None:
        vol = VolumeSpec(Path("/a"), "/b")
        with pytest.raises(AttributeError):
            vol.host_path = Path("/c")  # type: ignore[misc]


class TestSandboxSealed:
    """Verify sealed isolation mode (create → copy → start)."""

    def test_sealed_run_uses_create_copy_start(self, tmp_path: Path) -> None:
        """Sealed mode calls create, copy_to for each volume, then start."""
        # Create a real host dir with a marker file so copy_to triggers
        host_dir = tmp_path / "config"
        host_dir.mkdir()
        (host_dir / "marker.txt").write_text("hello")

        spec = _make_spec(
            sealed=True,
            volumes=(VolumeSpec(host_dir, "/home/dev/.terok", sharing=Sharing.PRIVATE),),
        )

        with (
            patch.object(Sandbox, "create") as mock_create,
            patch.object(Sandbox, "_ensure_parents") as mock_ensure,
            patch.object(Sandbox, "copy_to") as mock_copy,
            patch.object(Sandbox, "start") as mock_start,
        ):
            s = Sandbox()
            s.run(spec)

        mock_create.assert_called_once()
        mock_ensure.assert_called_once()
        mock_copy.assert_called_once_with("test-ctr", host_dir, "/home/dev/.terok")
        mock_start.assert_called_once()

    def test_sealed_run_skips_missing_host_dirs(self, tmp_path: Path) -> None:
        """Sealed mode skips copy_to when host_path doesn't exist."""
        missing = tmp_path / "nonexistent"
        spec = _make_spec(
            sealed=True,
            volumes=(VolumeSpec(missing, "/home/dev/.config"),),
        )

        with (
            patch.object(Sandbox, "create"),
            patch.object(Sandbox, "_ensure_parents"),
            patch.object(Sandbox, "copy_to") as mock_copy,
            patch.object(Sandbox, "start"),
        ):
            Sandbox().run(spec)

        mock_copy.assert_not_called()

    def test_ensure_parents_creates_directory_tree(self) -> None:
        """_ensure_parents injects a tar with all ancestor directories."""
        import tarfile as _tarfile

        volumes = (
            VolumeSpec(Path("/a"), "/home/dev/.config/gh"),
            VolumeSpec(Path("/b"), "/workspace"),
        )

        with patch.object(Sandbox, "_exec_podman") as mock_exec:
            Sandbox()._ensure_parents("ctr", volumes)

        mock_exec.assert_called_once()
        cmd = mock_exec.call_args[0][0]
        assert cmd == ["podman", "cp", "-", "ctr:/"]

        # Verify the tar payload contains expected directories
        tar_bytes = mock_exec.call_args[1]["input"]
        with _tarfile.open(fileobj=__import__("io").BytesIO(tar_bytes), mode="r") as tar:
            names = sorted(m.name for m in tar.getmembers())
            assert "home" in names
            assert "home/dev" in names
            assert "home/dev/.config" in names
            assert "home/dev/.config/gh" in names
            assert "workspace" in names
            # All entries are directories owned by uid 1000
            for m in tar.getmembers():
                assert m.isdir()
                assert m.uid == 1000

    def test_ensure_parents_noop_for_empty_volumes(self) -> None:
        """_ensure_parents does nothing when there are no volumes."""
        with patch.object(Sandbox, "_exec_podman") as mock_exec:
            Sandbox()._ensure_parents("ctr", ())

        mock_exec.assert_not_called()

    def test_sealed_create_omits_volume_flags(self) -> None:
        """podman create in sealed mode has no -v flags."""
        spec = _make_spec(sealed=True)

        with (
            patch("subprocess.run") as mock_run,
            patch("builtins.print"),
            patch("terok_sandbox.shield.pre_start", return_value=[]),
        ):
            Sandbox().create(spec)

        cmd = mock_run.call_args[0][0]
        assert cmd[:2] == ["podman", "create"]
        assert "-v" not in cmd

    def test_shared_run_includes_volume_flags(self) -> None:
        """podman run in shared mode includes -v flags from VolumeSpec."""
        spec = _make_spec(sealed=False)

        with (
            patch("subprocess.run") as mock_run,
            patch("builtins.print"),
            patch("terok_sandbox.shield.pre_start", return_value=[]),
        ):
            Sandbox().run(spec)

        cmd = mock_run.call_args[0][0]
        assert "-v" in cmd
        vol_idx = cmd.index("-v")
        assert cmd[vol_idx + 1] == f"{MOCK_HOST_DIR}:/ctr:z"


class TestSandboxCopyTo:
    """Verify copy_to delegates to podman cp."""

    def test_copy_to_directory(self, tmp_path: Path) -> None:
        """Directories use the src/. form to copy contents."""
        src = tmp_path / "config"
        src.mkdir()
        with patch("subprocess.run") as mock_run:
            Sandbox().copy_to("my-ctr", src, "/dest")

        cmd = mock_run.call_args[0][0]
        assert cmd == ["podman", "cp", f"{src}/.", "my-ctr:/dest"]

    def test_copy_to_file(self, tmp_path: Path) -> None:
        """Files are copied directly without the /. suffix."""
        src = tmp_path / "prompt.txt"
        src.write_text("hello")
        with patch("subprocess.run") as mock_run:
            Sandbox().copy_to("my-ctr", src, "/dest/prompt.txt")

        cmd = mock_run.call_args[0][0]
        assert cmd == ["podman", "cp", str(src), "my-ctr:/dest/prompt.txt"]


class TestSandboxStart:
    """Verify start delegates to podman start."""

    def test_start_invokes_podman_start(self) -> None:
        with patch("subprocess.run") as mock_run:
            Sandbox().start("my-ctr")

        mock_run.assert_called_once_with(
            ["podman", "start", "my-ctr"],
            check=True,
            capture_output=True,
        )

    def test_start_fires_post_start_hook(self) -> None:
        hook_called = False

        def on_post_start() -> None:
            nonlocal hook_called
            hook_called = True

        hooks = LifecycleHooks(post_start=on_post_start)
        with patch("subprocess.run"):
            Sandbox().start("my-ctr", hooks=hooks)

        assert hook_called


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
