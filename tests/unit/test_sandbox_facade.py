# SPDX-FileCopyrightText: 2026 Jiri Vyskocil
# SPDX-License-Identifier: Apache-2.0

"""Tests for the Sandbox facade class."""

from __future__ import annotations

import subprocess
from pathlib import Path
from unittest.mock import MagicMock, call, patch

import pytest

from terok_sandbox import GpuConfigError, SandboxConfig
from terok_sandbox.runtime import ContainerRemoveResult
from terok_sandbox.runtime.gpu import check_gpu_error
from terok_sandbox.runtime.podman import redact_env_args
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
        assert spec.gpus is None
        assert spec.extra_args == ()

    def test_security_defaults(self) -> None:
        """New security fields default to permissive-but-safe values."""
        spec = _make_spec()
        assert spec.unrestricted is True

    def test_deprecated_gpu_enabled_alias_maps_to_gpus(self) -> None:
        """``gpu_enabled=True`` still works but warns and folds into ``gpus``."""
        with pytest.warns(DeprecationWarning, match="gpu_enabled"):
            spec = _make_spec(gpu_enabled=True)
        assert spec.gpus == "all"
        with pytest.warns(DeprecationWarning, match="gpu_enabled"):
            spec = _make_spec(gpu_enabled=False)
        assert spec.gpus is None

    def test_gpus_and_gpu_enabled_together_rejected(self) -> None:
        """An explicit selector plus the deprecated alias is a caller bug."""
        with pytest.raises(ValueError, match="not both"):
            _make_spec(gpus=("amd",), gpu_enabled=True)

    def test_restricted_mode(self) -> None:
        """Restricted spec carries through the frozen dataclass."""
        spec = _make_spec(unrestricted=False)
        assert spec.unrestricted is False

    def test_resource_limits_default_none(self) -> None:
        """Memory and CPU limits default to None (unlimited)."""
        spec = _make_spec()
        assert spec.memory is None
        assert spec.cpus is None

    def test_resource_limits_carry_through(self) -> None:
        """Explicit limits survive the frozen dataclass round-trip."""
        spec = _make_spec(memory="4g", cpus="2.0")
        assert spec.memory == "4g"
        assert spec.cpus == "2.0"

    def test_runtime_defaults_none(self) -> None:
        """``runtime`` defaults to None (podman picks crun)."""
        assert _make_spec().runtime is None

    def test_runtime_carries_through(self) -> None:
        """Explicit runtime survives the frozen dataclass round-trip."""
        assert _make_spec(runtime="krun").runtime == "krun"

    def test_annotations_default_empty(self) -> None:
        """``annotations`` defaults to an empty mapping."""
        assert dict(_make_spec().annotations) == {}

    def test_annotations_carry_through(self) -> None:
        """Explicit annotations survive the frozen dataclass round-trip."""
        from types import MappingProxyType

        spec = _make_spec(
            annotations=MappingProxyType({"dossier.meta_path": "/var/lib/terok/tasks/t1.json"})
        )
        assert spec.annotations["dossier.meta_path"] == "/var/lib/terok/tasks/t1.json"

    def test_annotations_mutable_input_is_detached(self) -> None:
        """A caller's mutable dict can't mutate the spec after construction.

        ``MappingProxyType`` is the public type but callers may legitimately
        pass a plain dict (Pydantic / JSON-load / tests).  The post-init
        snapshot keeps the frozen guarantee intact.
        """
        from types import MappingProxyType

        live: dict[str, str] = {"k": "v1"}
        spec = _make_spec(annotations=live)  # plain dict accepted
        live["k"] = "v2"  # caller-side mutation
        live["new"] = "x"
        assert spec.annotations["k"] == "v1"
        assert "new" not in spec.annotations
        assert isinstance(spec.annotations, MappingProxyType)


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

    def test_mint_gate_token(self) -> None:
        s = Sandbox()
        token = s.mint_gate_token()
        assert token.startswith("terok-g-")

    def test_shield_down_delegates(self) -> None:
        with patch("terok_sandbox.integrations.shield.ShieldManager") as Mgr:
            s = Sandbox()
            s.shield_down("ctr", "ctr-uuid", Path("/tmp/task"))
            Mgr.assert_called_once_with(Path("/tmp/task"), s.config)
            Mgr.return_value.down.assert_called_once_with("ctr", "ctr-uuid")

    def test_pre_start_args_delegates(self) -> None:
        from terok_shield import ShieldRuntime

        with patch("terok_sandbox.integrations.shield.ShieldManager") as Mgr:
            Mgr.return_value.pre_start.return_value = ["--hook"]
            s = Sandbox()
            result = s.pre_start_args("ctr", Path("/tmp/task"))
            assert result == ["--hook"]
            Mgr.assert_called_once_with(
                Path("/tmp/task"),
                s.config,
                runtime=ShieldRuntime.DEFAULT,
                loopback_ports_override=None,
            )
            Mgr.return_value.pre_start.assert_called_once_with("ctr")

    def test_pre_start_args_maps_krun_runtime_to_shield_enum(self) -> None:
        """``runtime="krun"`` flows through as ``ShieldRuntime.KRUN``."""
        from terok_shield import ShieldRuntime

        with patch("terok_sandbox.integrations.shield.ShieldManager") as Mgr:
            Mgr.return_value.pre_start.return_value = ["--hook"]
            s = Sandbox()
            s.pre_start_args("ctr", Path("/tmp/task"), runtime="krun")
            Mgr.assert_called_once_with(
                Path("/tmp/task"),
                s.config,
                runtime=ShieldRuntime.KRUN,
                loopback_ports_override=None,
            )

    def test_stop_halts_containers_and_keeps_them(self) -> None:
        """``Sandbox.stop`` stops every named container and removes nothing."""
        s = Sandbox()
        handles = {"c1": MagicMock(), "c2": MagicMock()}
        with (
            patch.object(s.runtime, "container", side_effect=handles.get) as container,
            patch.object(s.runtime, "force_remove") as remove,
        ):
            s.stop(["c1", "c2"], timeout=5)

        assert container.call_args_list == [call("c1"), call("c2")]
        for handle in handles.values():
            handle.stop.assert_called_once_with(timeout=5)
        remove.assert_not_called()

    def test_rm_force_removes_containers(self) -> None:
        """``Sandbox.rm`` wraps names in container handles and force-removes."""
        s = Sandbox()
        results = [
            ContainerRemoveResult(name="c1", removed=True),
            ContainerRemoveResult(name="c2", removed=False, error="in use"),
        ]
        with patch.object(s.runtime, "force_remove", return_value=results) as remove:
            assert s.rm(["c1", "c2"]) == results

        handles = remove.call_args[0][0]
        assert [c.name for c in handles] == ["c1", "c2"]

    def test_start_ensures_runtime_dir_then_delegates(self) -> None:
        """``Sandbox.start`` rebuilds the /run/terok bind source before starting.

        The dir lives on the reboot-wiped runtime tmpfs and the supervisor
        removes it on every stop, so a restart must recreate it or
        ``podman start`` fails on the missing mount source.  Orchestrators
        call this one method and stay ignorant of the precondition.
        """
        s = Sandbox()
        run_dir = s.config.container_runtime_dir("ctr")
        assert not run_dir.exists()

        with patch.object(s.runtime, "container") as container:
            s.start("ctr")

        assert run_dir.is_dir()
        assert (run_dir.stat().st_mode & 0o777) == 0o700
        container.assert_called_once_with("ctr")
        container.return_value.start.assert_called_once_with()

    def test_stream_logs_uses_ready_marker(self) -> None:
        """``Sandbox.stream_logs`` delegates to the container's stream_initial_logs."""
        from terok_sandbox.runtime.podman import PodmanContainer

        with patch.object(PodmanContainer, "stream_initial_logs", return_value=True) as mock:
            result = Sandbox().stream_logs("ctr", timeout=30.0)
            assert result is True
            check_fn = mock.call_args[0][0]
            assert check_fn(">> init complete")
            assert not check_fn("still waiting")

    def test_run_builds_podman_command(self) -> None:
        """Sandbox.run() assembles and executes a podman run command."""
        with (
            patch("subprocess.run") as mock_run,
            patch("terok_sandbox.sandbox.shlex"),
            patch("builtins.print"),
            patch(
                "terok_sandbox.integrations.shield.ShieldManager.pre_start",
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

    def test_run_retains_container_by_default(self) -> None:
        """Without ephemeral, the assembled command carries no --rm."""
        with (
            patch("subprocess.run") as mock_run,
            patch("builtins.print"),
            patch("terok_sandbox.integrations.shield.ShieldManager.pre_start", return_value=[]),
        ):
            Sandbox().run(_make_spec())

        assert "--rm" not in mock_run.call_args[0][0]

    def test_run_ephemeral_adds_rm_flag(self) -> None:
        """spec.ephemeral flows through as podman --rm."""
        with (
            patch("subprocess.run") as mock_run,
            patch("builtins.print"),
            patch("terok_sandbox.integrations.shield.ShieldManager.pre_start", return_value=[]),
        ):
            Sandbox().run(_make_spec(ephemeral=True))

        assert "--rm" in mock_run.call_args[0][0]

    def test_run_injects_init_process(self) -> None:
        """A managed launch runs behind podman's ``--init`` when catatonit exists.

        The spec command must not be namespace-init itself: the kernel
        ignores default-disposition signals for init, so without a real
        pid1 a stop's SIGTERM is a no-op and every stop burns the full
        grace period before the SIGKILL.
        """
        with (
            patch("subprocess.run") as mock_run,
            patch("builtins.print"),
            patch("terok_sandbox.integrations.shield.ShieldManager.pre_start", return_value=[]),
            patch(
                "terok_sandbox.sandbox.find_init_binary",
                return_value="/usr/libexec/podman/catatonit",
            ),
        ):
            Sandbox().run(_make_spec())

        assert "--init" in mock_run.call_args[0][0]

    def test_run_without_catatonit_falls_back_to_no_init(self) -> None:
        """No catatonit → no ``--init``: degraded stops beat refused launches."""
        with (
            patch("subprocess.run") as mock_run,
            patch("builtins.print"),
            patch("terok_sandbox.integrations.shield.ShieldManager.pre_start", return_value=[]),
            patch("terok_sandbox.sandbox.find_init_binary", return_value=None),
        ):
            Sandbox().run(_make_spec())

        assert "--init" not in mock_run.call_args[0][0]

    def test_run_omits_hostname_by_default(self) -> None:
        """Without an explicit hostname, --hostname is absent (podman picks one)."""
        with (
            patch("subprocess.run") as mock_run,
            patch("builtins.print"),
            patch("terok_sandbox.integrations.shield.ShieldManager.pre_start", return_value=[]),
        ):
            Sandbox().run(_make_spec())

        cmd = mock_run.call_args[0][0]
        assert "--hostname" not in cmd

    def test_run_passes_hostname_when_set(self) -> None:
        """spec.hostname flows through as --hostname <value>."""
        with (
            patch("subprocess.run") as mock_run,
            patch("builtins.print"),
            patch("terok_sandbox.integrations.shield.ShieldManager.pre_start", return_value=[]),
        ):
            Sandbox().run(_make_spec(hostname="myproj-cli-k3v8h"))

        cmd = mock_run.call_args[0][0]
        assert "--hostname" in cmd
        assert cmd[cmd.index("--hostname") + 1] == "myproj-cli-k3v8h"

    def test_run_omits_runtime_flag_by_default(self) -> None:
        """Without spec.runtime, --runtime is absent (podman picks crun)."""
        with (
            patch("subprocess.run") as mock_run,
            patch("builtins.print"),
            patch("terok_sandbox.integrations.shield.ShieldManager.pre_start", return_value=[]),
        ):
            Sandbox().run(_make_spec())

        assert "--runtime" not in mock_run.call_args[0][0]

    def test_run_emits_runtime_flag(self) -> None:
        """spec.runtime='krun' flows through as --runtime krun."""
        with (
            patch("subprocess.run") as mock_run,
            patch("builtins.print"),
            patch("terok_sandbox.integrations.shield.ShieldManager.pre_start", return_value=[]),
        ):
            Sandbox().run(_make_spec(runtime="krun"))

        cmd = mock_run.call_args[0][0]
        assert "--runtime" in cmd
        assert cmd[cmd.index("--runtime") + 1] == "krun"

    def test_run_emits_annotations(self) -> None:
        """spec.annotations flow through as --annotation k=v entries."""
        from types import MappingProxyType

        annotations = MappingProxyType(
            {
                "dossier.meta_path": "/var/lib/terok/tasks/t1.json",
                "krun.cpus": "2",
            },
        )
        with (
            patch("subprocess.run") as mock_run,
            patch("builtins.print"),
            patch("terok_sandbox.integrations.shield.ShieldManager.pre_start", return_value=[]),
        ):
            Sandbox().run(_make_spec(annotations=annotations))

        cmd = mock_run.call_args[0][0]
        # Each annotation produces a "--annotation k=v" pair.
        emitted = [cmd[i + 1] for i, t in enumerate(cmd) if t == "--annotation"]
        assert "dossier.meta_path=/var/lib/terok/tasks/t1.json" in emitted
        assert "krun.cpus=2" in emitted

    def test_run_rejects_unknown_runtime(self) -> None:
        """A runtime outside the allowlist never reaches the podman argv.

        Podman's ``--runtime`` accepts a path to a binary — a caller
        who controls [`RunSpec.runtime`][terok_sandbox.sandbox.RunSpec]
        with no allowlist could make podman execute an arbitrary host
        binary as part of container creation.  Refused names raise
        before ``podman`` is invoked.
        """
        from unittest.mock import patch

        with (
            patch("subprocess.run") as mock_run,
            patch("builtins.print"),
            patch("terok_sandbox.integrations.shield.ShieldManager.pre_start", return_value=[]),
            pytest.raises(ValueError, match="not in allowlist"),
        ):
            Sandbox().run(_make_spec(runtime="evil"))
        mock_run.assert_not_called()

    def test_run_rejects_path_shaped_runtime(self) -> None:
        """A path-shaped ``--runtime`` value is the prime escalation vector."""
        from unittest.mock import patch

        with (
            patch("subprocess.run") as mock_run,
            patch("builtins.print"),
            patch("terok_sandbox.integrations.shield.ShieldManager.pre_start", return_value=[]),
            pytest.raises(ValueError, match="paths and whitespace"),
        ):
            Sandbox().run(_make_spec(runtime="/tmp/evil-runtime"))  # noqa: S108
        mock_run.assert_not_called()

    def test_run_rejects_unknown_annotation_key(self) -> None:
        """Annotations are runtime control plane — unrecognised keys are rejected."""
        from types import MappingProxyType
        from unittest.mock import patch

        with (
            patch("subprocess.run") as mock_run,
            patch("builtins.print"),
            patch("terok_sandbox.integrations.shield.ShieldManager.pre_start", return_value=[]),
            pytest.raises(ValueError, match="not in allowlist"),
        ):
            Sandbox().run(_make_spec(annotations=MappingProxyType({"evil.toggle": "1"})))
        mock_run.assert_not_called()

    def test_validate_runtime_rejects_non_string(self) -> None:
        """Defensive guard: the validator demands a real ``str``.

        Callers that bypass type-checked construction (Pydantic load
        with a bad schema, JSON hydration, future plugin) could pass
        ``None`` or an int.  Direct unit test on the validator so the
        defensive branch is covered without going through ``Sandbox.run``.
        """
        from terok_sandbox.sandbox import _validate_runtime

        with pytest.raises(ValueError, match="must be a string"):
            _validate_runtime(42)  # type: ignore[arg-type]

    def test_validate_runtime_rejects_whitespace_padded(self) -> None:
        """``\" krun \"`` is not a runtime name even though ``krun`` is.

        Separate from the path-shaped check so the whitespace branch is
        exercised independently of the ``/``/``\\`` rejection.
        """
        from terok_sandbox.sandbox import _validate_runtime

        with pytest.raises(ValueError, match="paths and whitespace"):
            _validate_runtime(" krun")
        with pytest.raises(ValueError, match="paths and whitespace"):
            _validate_runtime("krun\t")

    def test_validate_annotations_rejects_non_string_value(self) -> None:
        """Defensive guard: annotation values must be ``str``.

        Same rationale as ``_validate_runtime`` — non-CLI construction
        paths shouldn't be able to slip an int or ``None`` through.
        """
        from terok_sandbox.sandbox import _validate_annotations

        with pytest.raises(ValueError, match="must be a string"):
            _validate_annotations({"dossier.meta_path": 2})  # type: ignore[dict-item]

    def test_run_rejects_annotation_value_with_control_chars(self) -> None:
        """Control chars in an annotation value would split the --annotation argv."""
        from types import MappingProxyType
        from unittest.mock import patch

        with (
            patch("subprocess.run") as mock_run,
            patch("builtins.print"),
            patch("terok_sandbox.integrations.shield.ShieldManager.pre_start", return_value=[]),
            pytest.raises(ValueError, match="control character"),
        ):
            Sandbox().run(
                _make_spec(
                    annotations=MappingProxyType(
                        {"dossier.meta_path": "/var/lib/t.json\ndossier.meta_path=/etc/shadow"}
                    )
                )
            )
        mock_run.assert_not_called()

    def test_run_restricted_adds_no_new_privileges(self) -> None:
        """Restricted spec adds --security-opt no-new-privileges."""
        with (
            patch("subprocess.run") as mock_run,
            patch("builtins.print"),
            patch("terok_sandbox.integrations.shield.ShieldManager.pre_start", return_value=[]),
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
            patch("terok_sandbox.integrations.shield.ShieldManager.pre_start", return_value=[]),
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
                "terok_sandbox.sandbox.bypass_network_args",
                return_value=["--network", "pasta:-T,9418"],
            ) as mock_bypass,
            patch("terok_sandbox.integrations.shield.ShieldManager.pre_start") as mock_shield,
        ):
            s = Sandbox(config=cfg)
            s.run(_make_spec())

        mock_bypass.assert_called_once()
        mock_shield.assert_not_called()

    def test_run_refuses_when_shield_setup_fails(self) -> None:
        """A failing ``shield.pre_start`` aborts the launch with a remediation hint.

        Soft-failing past shield setup would launch the container with
        unfiltered egress under a config that explicitly asked for
        shielding — exactly the silent-fallback shape the loud-failure
        principle exists to prevent.
        """
        spec = _make_spec()
        with (
            patch("subprocess.run") as mock_run,
            patch(
                "terok_sandbox.integrations.shield.ShieldManager.pre_start",
                side_effect=FileNotFoundError("nft"),
            ),
            pytest.raises(SystemExit) as exc_info,
        ):
            Sandbox().run(spec)

        message = str(exc_info.value)
        assert "Shield setup failed" in message
        assert spec.container_name in message
        assert "shield_bypass" in message
        mock_run.assert_not_called()

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
            patch("terok_sandbox.integrations.shield.ShieldManager.pre_start", return_value=[]),
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
            patch("terok_sandbox.integrations.shield.ShieldManager.pre_start", return_value=[]),
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
            patch("terok_sandbox.integrations.shield.ShieldManager.pre_start", return_value=[]),
        ):
            s = Sandbox()
            with pytest.raises(GpuConfigError):
                s.run(_make_spec())

    def test_memory_in_podman_cmd(self) -> None:
        """--memory flag appears in the assembled podman command."""
        with (
            patch("subprocess.run") as mock_run,
            patch("builtins.print"),
            patch("terok_sandbox.integrations.shield.ShieldManager.pre_start", return_value=[]),
        ):
            Sandbox().run(_make_spec(memory="4g"))

        cmd = mock_run.call_args[0][0]
        idx = cmd.index("--memory")
        assert cmd[idx + 1] == "4g"

    def test_cpus_in_podman_cmd(self) -> None:
        """--cpus flag appears in the assembled podman command."""
        with (
            patch("subprocess.run") as mock_run,
            patch("builtins.print"),
            patch("terok_sandbox.integrations.shield.ShieldManager.pre_start", return_value=[]),
        ):
            Sandbox().run(_make_spec(cpus="2.0"))

        cmd = mock_run.call_args[0][0]
        idx = cmd.index("--cpus")
        assert cmd[idx + 1] == "2.0"

    def test_no_limits_omits_flags(self) -> None:
        """Neither --memory nor --cpus appear when limits are None."""
        with (
            patch("subprocess.run") as mock_run,
            patch("builtins.print"),
            patch("terok_sandbox.integrations.shield.ShieldManager.pre_start", return_value=[]),
        ):
            Sandbox().run(_make_spec())

        cmd = mock_run.call_args[0][0]
        assert "--memory" not in cmd
        assert "--cpus" not in cmd


class TestVolumeSpec:
    """Verify VolumeSpec dataclass."""

    def test_to_mount_arg_shared_default(self) -> None:
        vol = VolumeSpec(Path("/host/data"), "/container/data")
        assert vol.to_mount_arg() == "/host/data:/container/data:z"

    def test_to_mount_arg_private(self) -> None:
        from terok_sandbox.sandbox import Sharing

        vol = VolumeSpec(Path("/host/ws"), "/workspace", sharing=Sharing.PRIVATE)
        assert vol.to_mount_arg() == "/host/ws:/workspace:Z"

    def test_to_mount_arg_read_only(self) -> None:
        vol = VolumeSpec(Path("/host/cred"), "/home/dev/.claude/.credentials.json", read_only=True)
        assert vol.to_mount_arg() == "/host/cred:/home/dev/.claude/.credentials.json:z,ro"

    def test_to_mount_arg_read_only_private(self) -> None:
        from terok_sandbox.sandbox import Sharing

        vol = VolumeSpec(Path("/h"), "/c", sharing=Sharing.PRIVATE, read_only=True)
        assert vol.to_mount_arg() == "/h:/c:Z,ro"

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

    def test_ensure_parents_creates_directory_tree(self, tmp_path: Path) -> None:
        """_ensure_parents injects a tar with all ancestor directories.

        Uses real directories as host_path so the dir-target detection
        works (file volumes skip pre-creating the target itself — see
        the matching unit on file-volume behaviour).
        """
        import tarfile as _tarfile

        host_a = tmp_path / "a"
        host_a.mkdir()
        host_b = tmp_path / "b"
        host_b.mkdir()
        volumes = (
            VolumeSpec(host_a, "/home/dev/.config/gh"),
            VolumeSpec(host_b, "/workspace"),
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

    def test_ensure_parents_skips_file_target_itself(self, tmp_path: Path) -> None:
        """File volume targets are NOT pre-created as directories.

        For a file-shaped host path like ``/tmp/foo.sh`` landing at
        ``/run/terok/bridge.sh``, only ``run/terok`` should appear in
        the tar — pre-creating ``run/terok/bridge.sh`` as a directory
        would make the later ``copy_to`` land *inside* it.
        """
        import tarfile as _tarfile

        host_file = tmp_path / "bridge.sh"
        host_file.write_text("#!/bin/sh\n")
        volumes = (VolumeSpec(host_file, "/run/terok/bridge.sh"),)

        with patch.object(Sandbox, "_exec_podman") as mock_exec:
            Sandbox()._ensure_parents("ctr", volumes)

        tar_bytes = mock_exec.call_args[1]["input"]
        with _tarfile.open(fileobj=__import__("io").BytesIO(tar_bytes), mode="r") as tar:
            names = sorted(m.name for m in tar.getmembers())
            assert "run" in names
            assert "run/terok" in names
            assert "run/terok/bridge.sh" not in names

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
            patch("terok_sandbox.integrations.shield.ShieldManager.pre_start", return_value=[]),
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
            patch("terok_sandbox.integrations.shield.ShieldManager.pre_start", return_value=[]),
        ):
            Sandbox().run(spec)

        cmd = mock_run.call_args[0][0]
        assert "-v" in cmd
        vol_idx = cmd.index("-v")
        assert cmd[vol_idx + 1] == f"{MOCK_HOST_DIR}:/ctr:z"


class TestSandboxCopyTo:
    """Verify ``Sandbox.copy_to`` delegates through the runtime."""

    def test_copy_to_directory(self, tmp_path: Path) -> None:
        """Directories use the ``src/.`` form to copy contents into *dest*."""
        src = tmp_path / "config"
        src.mkdir()
        with patch("terok_sandbox.runtime.podman.subprocess.run") as mock_run:
            Sandbox().copy_to("my-ctr", src, "/dest")

        cmd = mock_run.call_args[0][0]
        assert cmd == ["podman", "cp", f"{src}/.", "my-ctr:/dest"]

    def test_copy_to_file(self, tmp_path: Path) -> None:
        """Files are copied directly without the ``/.`` suffix."""
        src = tmp_path / "prompt.txt"
        src.write_text("hello")
        with patch("terok_sandbox.runtime.podman.subprocess.run") as mock_run:
            Sandbox().copy_to("my-ctr", src, "/dest/prompt.txt")

        cmd = mock_run.call_args[0][0]
        assert cmd == ["podman", "cp", str(src), "my-ctr:/dest/prompt.txt"]


class TestSandboxStart:
    """Verify ``Sandbox.start`` delegates through the runtime."""

    def test_start_invokes_podman_start(self) -> None:
        """The runtime ``Container.start`` drives ``podman start``."""
        with patch("terok_sandbox.runtime.podman.subprocess.run") as mock_run:
            mock_run.return_value = subprocess.CompletedProcess(args=[], returncode=0, stderr="")
            Sandbox().start("my-ctr")

        cmd = mock_run.call_args[0][0]
        assert cmd == ["podman", "start", "my-ctr"]

    def test_start_fires_post_start_hook(self) -> None:
        """``post_start`` fires after a successful start."""
        hook_called = False

        def on_post_start() -> None:
            nonlocal hook_called
            hook_called = True

        hooks = LifecycleHooks(post_start=on_post_start)
        with patch("terok_sandbox.runtime.podman.subprocess.run") as mock_run:
            mock_run.return_value = subprocess.CompletedProcess(args=[], returncode=0, stderr="")
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


class TestTaskStateDir:
    """Verify the per-container state-dir derivation on the facade."""

    def test_task_state_dir_under_sandbox_runs(self, tmp_path: Path) -> None:
        """Resolves under ``{state_dir}/sandbox/runs/{container}`` for any container name."""
        cfg = SandboxConfig(state_dir=tmp_path / "state")
        sandbox = Sandbox(cfg)
        assert (
            sandbox.task_state_dir("my-task") == tmp_path / "state" / "sandbox" / "runs" / "my-task"
        )
