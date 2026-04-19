# SPDX-FileCopyrightText: 2026 Jiri Vyskocil
# SPDX-License-Identifier: Apache-2.0

"""Tests for runtime helpers: podman state, env redaction, GPU, network."""

from __future__ import annotations

import socket
import subprocess
from unittest.mock import patch

import pytest

from terok_sandbox.runtime import (
    ContainerRemoveResult,
    GpuConfigError,
    _detect_rootless_network_mode,  # noqa: PLC2701
    bypass_network_args,
    check_gpu_error,
    find_free_port,
    get_container_state,
    get_container_states,
    gpu_run_args,
    is_container_running,
    podman_userns_args,
    redact_env_args,
    reserve_free_port,
    stop_task_containers,
    wait_for_exit,
)

# ---------------------------------------------------------------------------
# podman_userns_args
# ---------------------------------------------------------------------------


class TestPodmanUsernsArgs:
    """podman_userns_args returns mapping args only when running rootless."""

    @patch("terok_sandbox.runtime.os.geteuid", return_value=1000)
    def test_rootless_emits_keep_id(self, _euid) -> None:
        assert podman_userns_args() == ["--userns=keep-id:uid=1000,gid=1000"]

    @patch("terok_sandbox.runtime.os.geteuid", return_value=0)
    def test_root_emits_nothing(self, _euid) -> None:
        assert podman_userns_args() == []


# ---------------------------------------------------------------------------
# check_gpu_error / GpuConfigError
# ---------------------------------------------------------------------------


class TestCheckGpuError:
    """check_gpu_error raises GpuConfigError only on CDI/NVIDIA errors."""

    def test_cdi_pattern_raises(self) -> None:
        exc = subprocess.CalledProcessError(
            returncode=125,
            cmd=["podman", "run"],
            stderr=b"Error: CDI device nvidia.com/gpu=all not registered",
        )
        with pytest.raises(GpuConfigError) as excinfo:
            check_gpu_error(exc)
        assert "GPU misconfiguration" in str(excinfo.value)
        assert excinfo.value.hint  # CDI hint attached
        assert excinfo.value.__cause__ is exc

    def test_unrelated_error_does_not_raise(self) -> None:
        exc = subprocess.CalledProcessError(
            returncode=125,
            cmd=["podman", "run"],
            stderr=b"Error: image not found",
        )
        check_gpu_error(exc)  # must not raise

    def test_no_stderr_does_not_raise(self) -> None:
        exc = subprocess.CalledProcessError(returncode=125, cmd=["podman"], stderr=None)
        check_gpu_error(exc)


# ---------------------------------------------------------------------------
# redact_env_args
# ---------------------------------------------------------------------------


class TestRedactEnvArgs:
    """Sensitive ``-e KEY=VALUE`` args are redacted; other args pass through."""

    def test_redacts_secret_keys(self) -> None:
        cmd = ["podman", "run", "-e", "ANTHROPIC_API_KEY=sk-secret", "image"]
        assert redact_env_args(cmd) == [
            "podman",
            "run",
            "-e",
            "ANTHROPIC_API_KEY=<redacted>",
            "image",
        ]

    def test_preserves_non_sensitive(self) -> None:
        cmd = ["podman", "run", "-e", "DEBUG=1", "image"]
        assert "DEBUG=1" in redact_env_args(cmd)

    @pytest.mark.parametrize(
        "var",
        ["TOKEN=t", "MY_SECRET=x", "PASSWORD=p", "PRIVATE_THING=z"],
    )
    def test_pattern_matches_sensitive_kinds(self, var: str) -> None:
        out = redact_env_args(["-e", var])
        assert "<redacted>" in out[1]

    def test_always_redact_keys(self) -> None:
        cmd = ["-e", "CODE_REPO=git@github.com:x/y.git", "-e", "CLONE_FROM=http://t@h/r.git"]
        out = redact_env_args(cmd)
        assert out[1] == "CODE_REPO=<redacted>"
        assert out[3] == "CLONE_FROM=<redacted>"

    def test_dash_e_without_kvpair_passes_through(self) -> None:
        # -e at end of args (no following pair) — function should not crash.
        assert redact_env_args(["podman", "run", "-e"]) == ["podman", "run", "-e"]


# ---------------------------------------------------------------------------
# get_container_states / get_container_state / is_container_running
# ---------------------------------------------------------------------------


class TestGetContainerStates:
    """Parses bulk podman ps output into {name: state} dict."""

    @patch("terok_sandbox.runtime.subprocess.check_output")
    def test_parses_two_columns(self, mock_co) -> None:
        mock_co.return_value = "task-a Running\ntask-b Exited\n"
        assert get_container_states("task") == {"task-a": "running", "task-b": "exited"}

    @patch("terok_sandbox.runtime.subprocess.check_output")
    def test_skips_malformed_lines(self, mock_co) -> None:
        mock_co.return_value = "good Running\nmalformed-no-state\n"
        assert get_container_states("p") == {"good": "running"}

    @patch("terok_sandbox.runtime.subprocess.check_output", side_effect=FileNotFoundError)
    def test_returns_empty_when_podman_missing(self, _co) -> None:
        assert get_container_states("p") == {}

    @patch(
        "terok_sandbox.runtime.subprocess.check_output",
        side_effect=subprocess.CalledProcessError(1, "podman"),
    )
    def test_returns_empty_on_podman_error(self, _co) -> None:
        assert get_container_states("p") == {}


class TestGetContainerState:
    """Returns single container state or None."""

    @patch("terok_sandbox.runtime.subprocess.check_output", return_value="Running\n")
    def test_returns_lowercased_state(self, _co) -> None:
        assert get_container_state("foo") == "running"

    @patch("terok_sandbox.runtime.subprocess.check_output", return_value="\n")
    def test_empty_output_is_none(self, _co) -> None:
        assert get_container_state("foo") is None

    @patch("terok_sandbox.runtime.subprocess.check_output", side_effect=FileNotFoundError)
    def test_podman_missing_is_none(self, _co) -> None:
        assert get_container_state("foo") is None

    @patch(
        "terok_sandbox.runtime.subprocess.check_output",
        side_effect=subprocess.CalledProcessError(1, "podman"),
    )
    def test_podman_error_is_none(self, _co) -> None:
        assert get_container_state("foo") is None


class TestIsContainerRunning:
    """True only when podman reports State.Running == true."""

    @patch("terok_sandbox.runtime.subprocess.check_output", return_value="true\n")
    def test_true(self, _co) -> None:
        assert is_container_running("c") is True

    @patch("terok_sandbox.runtime.subprocess.check_output", return_value="false\n")
    def test_false(self, _co) -> None:
        assert is_container_running("c") is False

    @patch("terok_sandbox.runtime.subprocess.check_output", side_effect=FileNotFoundError)
    def test_no_podman_is_false(self, _co) -> None:
        assert is_container_running("c") is False


class TestContainerImage:
    """Image-ID lookup for a container; None on any failure."""

    @patch("terok_sandbox.runtime.subprocess.check_output", return_value="sha256:abc\n")
    def test_returns_image_id(self, _co) -> None:
        from terok_sandbox.runtime import container_image

        assert container_image("c") == "sha256:abc"

    @patch("terok_sandbox.runtime.subprocess.check_output", return_value="\n")
    def test_empty_output_is_none(self, _co) -> None:
        from terok_sandbox.runtime import container_image

        assert container_image("c") is None

    @patch(
        "terok_sandbox.runtime.subprocess.check_output",
        side_effect=subprocess.CalledProcessError(125, "podman"),
    )
    def test_missing_container_is_none(self, _co) -> None:
        from terok_sandbox.runtime import container_image

        assert container_image("c") is None

    @patch("terok_sandbox.runtime.subprocess.check_output", side_effect=FileNotFoundError)
    def test_no_podman_is_none(self, _co) -> None:
        from terok_sandbox.runtime import container_image

        assert container_image("c") is None


class TestImageExists:
    """``podman image exists`` exit-code translation."""

    @patch("terok_sandbox.runtime.subprocess.run")
    def test_present(self, mock_run) -> None:
        from terok_sandbox.runtime import image_exists

        mock_run.return_value = subprocess.CompletedProcess(args=[], returncode=0)
        assert image_exists("terok-l1-cli:test") is True

    @patch("terok_sandbox.runtime.subprocess.run")
    def test_absent(self, mock_run) -> None:
        from terok_sandbox.runtime import image_exists

        mock_run.return_value = subprocess.CompletedProcess(args=[], returncode=1)
        assert image_exists("missing:tag") is False

    @patch("terok_sandbox.runtime.subprocess.run", side_effect=FileNotFoundError)
    def test_no_podman_is_false(self, _run) -> None:
        from terok_sandbox.runtime import image_exists

        assert image_exists("any:tag") is False


class TestImageLabels:
    """Parse the ``Config.Labels`` dict from ``podman inspect``."""

    @patch("terok_sandbox.runtime.subprocess.run")
    def test_returns_labels_dict(self, mock_run) -> None:
        from terok_sandbox.runtime import image_labels

        mock_run.return_value = subprocess.CompletedProcess(
            args=[],
            returncode=0,
            stdout='{"ai.terok.agents": "claude,codex", "build": "v1"}',
            stderr="",
        )
        assert image_labels("terok-l1-cli:test") == {
            "ai.terok.agents": "claude,codex",
            "build": "v1",
        }

    @patch("terok_sandbox.runtime.subprocess.run")
    def test_null_labels_returns_empty(self, mock_run) -> None:
        """Podman emits ``null`` when the image has no labels set."""
        from terok_sandbox.runtime import image_labels

        mock_run.return_value = subprocess.CompletedProcess(
            args=[], returncode=0, stdout="null", stderr=""
        )
        assert image_labels("terok-l1-cli:test") == {}

    @patch("terok_sandbox.runtime.subprocess.run", side_effect=FileNotFoundError)
    def test_no_podman_returns_empty(self, _run) -> None:
        from terok_sandbox.runtime import image_labels

        assert image_labels("any:tag") == {}

    @patch(
        "terok_sandbox.runtime.subprocess.run",
        side_effect=subprocess.CalledProcessError(1, ["podman"]),
    )
    def test_missing_image_returns_empty(self, _run) -> None:
        from terok_sandbox.runtime import image_labels

        assert image_labels("missing:tag") == {}

    @patch("terok_sandbox.runtime.subprocess.run")
    def test_unparseable_output_returns_empty(self, mock_run) -> None:
        """Garbled output from podman does not leak as a JSONDecodeError."""
        from terok_sandbox.runtime import image_labels

        mock_run.return_value = subprocess.CompletedProcess(
            args=[], returncode=0, stdout="not-json", stderr=""
        )
        assert image_labels("any:tag") == {}


class TestImagesList:
    """Parse ``podman images`` TSV output into ImageRecord rows."""

    @patch("terok_sandbox.runtime.subprocess.run")
    def test_parses_rows(self, mock_run) -> None:
        from terok_sandbox.runtime import ImageRecord, images_list

        mock_run.return_value = subprocess.CompletedProcess(
            args=[],
            returncode=0,
            stdout=(
                "docker.io/terok/l1-cli\tubuntu-24.04\tabc123\t420MB\t2 hours ago\n"
                "docker.io/terok/l0\tubuntu-24.04\tdef456\t120MB\t1 day ago\n"
            ),
            stderr="",
        )
        records = images_list()
        assert records == [
            ImageRecord("docker.io/terok/l1-cli", "ubuntu-24.04", "abc123", "420MB", "2 hours ago"),
            ImageRecord("docker.io/terok/l0", "ubuntu-24.04", "def456", "120MB", "1 day ago"),
        ]

    @patch("terok_sandbox.runtime.subprocess.run")
    def test_dangling_only_sets_filter(self, mock_run) -> None:
        from terok_sandbox.runtime import images_list

        mock_run.return_value = subprocess.CompletedProcess(
            args=[], returncode=0, stdout="", stderr=""
        )
        images_list(dangling_only=True)
        cmd = mock_run.call_args[0][0]
        assert "--filter" in cmd
        assert "dangling=true" in cmd

    @patch("terok_sandbox.runtime.subprocess.run")
    def test_skips_malformed_rows(self, mock_run) -> None:
        from terok_sandbox.runtime import images_list

        mock_run.return_value = subprocess.CompletedProcess(
            args=[],
            returncode=0,
            stdout="repo\ttag\tid\t100MB\tyesterday\nmalformed\n",
            stderr="",
        )
        assert len(images_list()) == 1

    @patch("terok_sandbox.runtime.subprocess.run")
    def test_podman_error_returns_empty(self, mock_run) -> None:
        from terok_sandbox.runtime import images_list

        mock_run.return_value = subprocess.CompletedProcess(
            args=[], returncode=1, stdout="", stderr=""
        )
        assert images_list() == []

    @patch("terok_sandbox.runtime.subprocess.run", side_effect=FileNotFoundError)
    def test_no_podman_returns_empty(self, _run) -> None:
        from terok_sandbox.runtime import images_list

        assert images_list() == []


class TestImageHistory:
    """Return the ``CreatedBy`` line for each layer."""

    @patch("terok_sandbox.runtime.subprocess.run")
    def test_returns_history_lines(self, mock_run) -> None:
        from terok_sandbox.runtime import image_history

        mock_run.return_value = subprocess.CompletedProcess(
            args=[],
            returncode=0,
            stdout="COPY . /app\nRUN apt-get install curl\nFROM ubuntu:24.04\n",
            stderr="",
        )
        assert image_history("abc123") == [
            "COPY . /app",
            "RUN apt-get install curl",
            "FROM ubuntu:24.04",
        ]

    @patch("terok_sandbox.runtime.subprocess.run")
    def test_empty_on_podman_error(self, mock_run) -> None:
        from terok_sandbox.runtime import image_history

        mock_run.return_value = subprocess.CompletedProcess(
            args=[], returncode=1, stdout="", stderr="Error: no such image\n"
        )
        assert image_history("abc123") == []

    @patch("terok_sandbox.runtime.subprocess.run", side_effect=FileNotFoundError)
    def test_no_podman_returns_empty(self, _run) -> None:
        from terok_sandbox.runtime import image_history

        assert image_history("abc123") == []


class TestImageRm:
    """Best-effort image removal; False on any failure."""

    @patch("terok_sandbox.runtime.subprocess.run")
    def test_success(self, mock_run) -> None:
        from terok_sandbox.runtime import image_rm

        mock_run.return_value = subprocess.CompletedProcess(args=[], returncode=0)
        assert image_rm("abc123") is True

    @patch("terok_sandbox.runtime.subprocess.run")
    def test_failure_returns_false(self, mock_run) -> None:
        from terok_sandbox.runtime import image_rm

        mock_run.return_value = subprocess.CompletedProcess(args=[], returncode=2)
        assert image_rm("abc123") is False

    @patch("terok_sandbox.runtime.subprocess.run", side_effect=FileNotFoundError)
    def test_no_podman_returns_false(self, _run) -> None:
        from terok_sandbox.runtime import image_rm

        assert image_rm("abc123") is False


# ---------------------------------------------------------------------------
# stop_task_containers
# ---------------------------------------------------------------------------


class TestStopTaskContainers:
    """Best-effort removal with per-container outcome tracking."""

    @staticmethod
    def _proc(rc: int, stderr: str = "") -> subprocess.CompletedProcess[str]:
        return subprocess.CompletedProcess(args=[], returncode=rc, stdout="", stderr=stderr)

    @patch("terok_sandbox.runtime.subprocess.run")
    def test_clean_remove_marks_removed(self, mock_run) -> None:
        mock_run.return_value = self._proc(0)
        results = stop_task_containers(["a"])
        assert results == [ContainerRemoveResult(name="a", removed=True)]

    @patch("terok_sandbox.runtime.subprocess.run")
    def test_already_gone_counts_as_removed(self, mock_run) -> None:
        mock_run.return_value = self._proc(1, "Error: no such container: foo")
        results = stop_task_containers(["foo"])
        assert results[0].removed is True
        assert results[0].error is None

    @patch("terok_sandbox.runtime.subprocess.run")
    def test_real_failure_kept_with_reason(self, mock_run) -> None:
        mock_run.return_value = self._proc(2, "permission denied")
        result = stop_task_containers(["x"])[0]
        assert result.removed is False
        assert "permission denied" in result.error

    @patch(
        "terok_sandbox.runtime.subprocess.run", side_effect=subprocess.TimeoutExpired("podman", 1)
    )
    def test_timeout_yields_timeout_error(self, _run) -> None:
        result = stop_task_containers(["t"])[0]
        assert result.removed is False
        assert "timed out" in result.error

    @patch("terok_sandbox.runtime.subprocess.run", side_effect=FileNotFoundError)
    def test_podman_missing_yields_clear_error(self, _run) -> None:
        result = stop_task_containers(["t"])[0]
        assert result.removed is False
        assert "podman not found" in result.error

    @patch("terok_sandbox.runtime.subprocess.run")
    def test_processes_each_container_independently(self, mock_run) -> None:
        # First fails, second succeeds — the second still runs.
        mock_run.side_effect = [self._proc(2, "boom"), self._proc(0)]
        results = stop_task_containers(["bad", "good"])
        assert results[0].removed is False
        assert results[1].removed is True


# ---------------------------------------------------------------------------
# gpu_run_args
# ---------------------------------------------------------------------------


class TestGpuRunArgs:
    """gpu_run_args returns CDI args only when explicitly enabled."""

    def test_disabled_default_is_empty(self) -> None:
        assert gpu_run_args() == []

    def test_enabled_returns_cdi_args(self) -> None:
        args = gpu_run_args(enabled=True)
        assert "--device" in args
        assert "nvidia.com/gpu=all" in args
        assert any("NVIDIA_VISIBLE_DEVICES=all" in a for a in args)


# ---------------------------------------------------------------------------
# wait_for_exit
# ---------------------------------------------------------------------------


class TestWaitForExit:
    """wait_for_exit returns the container's real exit code and signals
    timeouts, podman errors, and missing podman out of band — nothing is
    impersonated as an exit code."""

    @patch("terok_sandbox.runtime.subprocess.run")
    def test_returns_parsed_exit_code(self, mock_run) -> None:
        mock_run.return_value = subprocess.CompletedProcess(
            args=[], returncode=0, stdout="42\n", stderr=""
        )
        assert wait_for_exit("c") == 42

    @patch("terok_sandbox.runtime.subprocess.run")
    def test_returns_exit_code_124_distinctly(self, mock_run) -> None:
        """A container that legitimately exits with code 124 is returned
        as 124 — not confused with the watcher's own timeout signal."""
        mock_run.return_value = subprocess.CompletedProcess(
            args=[], returncode=0, stdout="124\n", stderr=""
        )
        assert wait_for_exit("c") == 124

    @patch("terok_sandbox.runtime.subprocess.run", side_effect=subprocess.TimeoutExpired("p", 1))
    def test_timeout_raises(self, _run) -> None:
        with pytest.raises(TimeoutError, match=r"did not exit within"):
            wait_for_exit("c", timeout_sec=0.1)

    @patch("terok_sandbox.runtime.subprocess.run", side_effect=FileNotFoundError)
    def test_missing_podman_raises(self, _run) -> None:
        with pytest.raises(FileNotFoundError):
            wait_for_exit("c")

    @patch("terok_sandbox.runtime.subprocess.run")
    def test_podman_wait_failure_raises(self, mock_run) -> None:
        """Non-zero ``podman wait`` returncode (e.g. unknown container)
        raises RuntimeError with the stderr text — never impersonated as
        the container's exit code."""
        mock_run.return_value = subprocess.CompletedProcess(
            args=[], returncode=125, stdout="", stderr="Error: no such container c\n"
        )
        with pytest.raises(RuntimeError, match=r"returncode=125.*no such container"):
            wait_for_exit("c")

    @patch("terok_sandbox.runtime.subprocess.run")
    def test_unparseable_stdout_raises(self, mock_run) -> None:
        """Non-numeric stdout raises RuntimeError with diagnostic context
        instead of leaking a bare ValueError from ``int(...)``."""
        mock_run.return_value = subprocess.CompletedProcess(
            args=[], returncode=0, stdout="oops\n", stderr=""
        )
        with pytest.raises(RuntimeError, match=r"returned unexpected output.*oops"):
            wait_for_exit("c")


# ---------------------------------------------------------------------------
# reserve_free_port / find_free_port
# ---------------------------------------------------------------------------


class TestReserveAndFindFreePort:
    """Real socket-bind smoke tests — no mocking."""

    def test_reserve_returns_open_socket_with_valid_port(self) -> None:
        sock, port = reserve_free_port()
        try:
            assert isinstance(sock, socket.socket)
            assert 1024 <= port <= 65535
            # Socket is still bound — connect to it via the OS to prove it.
            sock.listen(1)
            client = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            client.connect(("127.0.0.1", port))
            client.close()
        finally:
            sock.close()

    def test_find_free_port_returns_an_int(self) -> None:
        port = find_free_port()
        assert isinstance(port, int)
        assert 1024 <= port <= 65535


# ---------------------------------------------------------------------------
# _detect_rootless_network_mode + bypass_network_args
# ---------------------------------------------------------------------------


class TestDetectRootlessNetworkMode:
    """Detection probes ``podman info`` and falls back to slirp4netns."""

    @staticmethod
    def _proc(rc: int, stdout: str) -> subprocess.CompletedProcess[str]:
        return subprocess.CompletedProcess(args=[], returncode=rc, stdout=stdout, stderr="")

    @patch("terok_sandbox.runtime.subprocess.run")
    def test_pasta_recognised(self, mock_run) -> None:
        mock_run.return_value = self._proc(0, "pasta\n")
        assert _detect_rootless_network_mode() == "pasta"

    @patch("terok_sandbox.runtime.subprocess.run")
    def test_slirp_recognised(self, mock_run) -> None:
        mock_run.return_value = self._proc(0, "slirp4netns\n")
        assert _detect_rootless_network_mode() == "slirp4netns"

    @patch("terok_sandbox.runtime.subprocess.run")
    def test_unknown_value_falls_back_to_slirp(self, mock_run) -> None:
        mock_run.return_value = self._proc(0, "bridge\n")
        assert _detect_rootless_network_mode() == "slirp4netns"

    @patch("terok_sandbox.runtime.subprocess.run")
    def test_nonzero_exit_falls_back_to_slirp(self, mock_run) -> None:
        mock_run.return_value = self._proc(1, "")
        assert _detect_rootless_network_mode() == "slirp4netns"

    @patch("terok_sandbox.runtime.subprocess.run", side_effect=FileNotFoundError)
    def test_missing_podman_falls_back_to_slirp(self, _run) -> None:
        assert _detect_rootless_network_mode() == "slirp4netns"

    @patch(
        "terok_sandbox.runtime.subprocess.run",
        side_effect=subprocess.TimeoutExpired("podman", 1),
    )
    def test_timeout_falls_back_to_slirp(self, _run) -> None:
        assert _detect_rootless_network_mode() == "slirp4netns"


class TestBypassNetworkArgs:
    """bypass_network_args picks args based on euid + detected network mode."""

    @patch("terok_sandbox.runtime.os.geteuid", return_value=0)
    def test_root_emits_nothing(self, _euid) -> None:
        assert bypass_network_args(9418) == []

    @patch("terok_sandbox.runtime._detect_rootless_network_mode", return_value="slirp4netns")
    @patch("terok_sandbox.runtime.os.geteuid", return_value=1000)
    def test_slirp_args_include_loopback_allow(self, _euid, _net) -> None:
        args = bypass_network_args(9418)
        assert "slirp4netns:allow_host_loopback=true" in args
        assert any("host.containers.internal:" in a for a in args)

    @patch("terok_sandbox.runtime._detect_rootless_network_mode", return_value="pasta")
    @patch("terok_sandbox.runtime.os.geteuid", return_value=1000)
    def test_pasta_args_include_map_host_loopback(self, _euid, _net) -> None:
        args = bypass_network_args(9418)
        assert any("map-host-loopback" in a for a in args)
