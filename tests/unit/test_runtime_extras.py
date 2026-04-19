# SPDX-FileCopyrightText: 2026 Jiri Vyskocil
# SPDX-License-Identifier: Apache-2.0

"""Tests for podman-backend helpers and the protocol surface."""

from __future__ import annotations

import socket
import subprocess
from unittest.mock import patch

import pytest

from terok_sandbox import (
    ContainerRemoveResult,
    GpuConfigError,
    PodmanRuntime,
)
from terok_sandbox.runtime.podman import (
    _detect_rootless_network_mode,
    bypass_network_args,
    check_gpu_error,
    gpu_run_args,
    podman_userns_args,
    redact_env_args,
)

# ── Argv helpers (pure functions on the podman backend) ───────────────────


class TestPodmanUsernsArgs:
    """``podman_userns_args`` returns mapping args only rootless."""

    @patch("terok_sandbox.runtime.podman.os.geteuid", return_value=1000)
    def test_rootless_emits_keep_id(self, _euid) -> None:
        """Non-zero euid yields the keep-id user namespace flag."""
        assert podman_userns_args() == ["--userns=keep-id:uid=1000,gid=1000"]

    @patch("terok_sandbox.runtime.podman.os.geteuid", return_value=0)
    def test_root_emits_nothing(self, _euid) -> None:
        """Running as root yields no extra args."""
        assert podman_userns_args() == []


class TestCheckGpuError:
    """``check_gpu_error`` raises ``GpuConfigError`` only on CDI patterns."""

    def test_cdi_pattern_raises(self) -> None:
        """Stderr matching a CDI pattern triggers ``GpuConfigError``."""
        exc = subprocess.CalledProcessError(
            returncode=125,
            cmd=["podman", "run"],
            stderr=b"Error: CDI device nvidia.com/gpu=all not registered",
        )
        with pytest.raises(GpuConfigError) as excinfo:
            check_gpu_error(exc)
        assert "GPU misconfiguration" in str(excinfo.value)
        assert excinfo.value.hint
        assert excinfo.value.__cause__ is exc

    def test_unrelated_error_does_not_raise(self) -> None:
        """Non-CDI stderr passes through silently."""
        exc = subprocess.CalledProcessError(
            returncode=125,
            cmd=["podman", "run"],
            stderr=b"Error: image not found",
        )
        check_gpu_error(exc)

    def test_no_stderr_does_not_raise(self) -> None:
        """Missing stderr is treated as no CDI match."""
        exc = subprocess.CalledProcessError(returncode=125, cmd=["podman"], stderr=None)
        check_gpu_error(exc)

    def test_text_stderr_is_handled(self) -> None:
        """``str`` stderr (``text=True`` callers) does not trip ``.decode``."""
        exc = subprocess.CalledProcessError(
            returncode=125,
            cmd=["podman", "run"],
            stderr="Error: CDI device nvidia.com/gpu=all not registered",
        )
        with pytest.raises(GpuConfigError):
            check_gpu_error(exc)


class TestRedactEnvArgs:
    """Sensitive ``-e KEY=VALUE`` args are redacted; others pass through."""

    def test_redacts_secret_keys(self) -> None:
        """Secret-looking keys are redacted."""
        cmd = ["podman", "run", "-e", "ANTHROPIC_API_KEY=sk-secret", "image"]
        assert redact_env_args(cmd) == [
            "podman",
            "run",
            "-e",
            "ANTHROPIC_API_KEY=<redacted>",
            "image",
        ]

    def test_preserves_non_sensitive(self) -> None:
        """Innocuous KEY=VALUE pairs pass through."""
        cmd = ["podman", "run", "-e", "DEBUG=1", "image"]
        assert "DEBUG=1" in redact_env_args(cmd)

    @pytest.mark.parametrize(
        "var",
        ["TOKEN=t", "MY_SECRET=x", "PASSWORD=p", "PRIVATE_THING=z"],
    )
    def test_pattern_matches_sensitive_kinds(self, var: str) -> None:
        """Secret-like name fragments all trigger redaction."""
        out = redact_env_args(["-e", var])
        assert "<redacted>" in out[1]

    def test_always_redact_keys(self) -> None:
        """``CODE_REPO`` / ``CLONE_FROM`` are always redacted regardless of name."""
        cmd = ["-e", "CODE_REPO=git@github.com:x/y.git", "-e", "CLONE_FROM=http://t@h/r.git"]
        out = redact_env_args(cmd)
        assert out[1] == "CODE_REPO=<redacted>"
        assert out[3] == "CLONE_FROM=<redacted>"

    def test_dash_e_without_kvpair_passes_through(self) -> None:
        """A dangling ``-e`` at the end does not crash the parser."""
        assert redact_env_args(["podman", "run", "-e"]) == ["podman", "run", "-e"]


class TestGpuRunArgs:
    """``gpu_run_args`` returns CDI args only when explicitly enabled."""

    def test_disabled_default_is_empty(self) -> None:
        """Disabled by default — no CDI flags."""
        assert gpu_run_args() == []

    def test_enabled_returns_cdi_args(self) -> None:
        """Enabled emits ``--device nvidia.com/gpu=all`` plus env vars."""
        args = gpu_run_args(enabled=True)
        assert "--device" in args
        assert "nvidia.com/gpu=all" in args
        assert any("NVIDIA_VISIBLE_DEVICES=all" in a for a in args)


# ── Container observation ─────────────────────────────────────────────────


class TestContainersWithPrefix:
    """``PodmanRuntime.containers_with_prefix`` enumerates matching names."""

    @patch("terok_sandbox.runtime.podman.subprocess.check_output")
    def test_returns_handles_for_matches(self, mock_co) -> None:
        """Each line of output becomes a ``Container`` handle."""
        mock_co.return_value = "task-a\ntask-b\n"
        handles = PodmanRuntime().containers_with_prefix("task")
        assert [c.name for c in handles] == ["task-a", "task-b"]

    @patch("terok_sandbox.runtime.podman.subprocess.check_output", side_effect=FileNotFoundError)
    def test_returns_empty_when_podman_missing(self, _co) -> None:
        """Missing podman → empty list."""
        assert PodmanRuntime().containers_with_prefix("p") == []

    @patch(
        "terok_sandbox.runtime.podman.subprocess.check_output",
        side_effect=subprocess.CalledProcessError(1, "podman"),
    )
    def test_returns_empty_on_podman_error(self, _co) -> None:
        """Podman error → empty list."""
        assert PodmanRuntime().containers_with_prefix("p") == []


class TestContainerStates:
    """``PodmanRuntime.container_states`` batch-parses name+state pairs."""

    @patch("terok_sandbox.runtime.podman.subprocess.check_output")
    def test_parses_two_columns(self, mock_co) -> None:
        """Each space-separated row becomes one dict entry, state lowercased."""
        mock_co.return_value = "task-a Running\ntask-b Exited\n"
        assert PodmanRuntime().container_states("task") == {
            "task-a": "running",
            "task-b": "exited",
        }

    @patch("terok_sandbox.runtime.podman.subprocess.check_output")
    def test_skips_malformed_lines(self, mock_co) -> None:
        """Lines without a state column are skipped."""
        mock_co.return_value = "good Running\nmalformed-no-state\n"
        assert PodmanRuntime().container_states("p") == {"good": "running"}

    @patch("terok_sandbox.runtime.podman.subprocess.check_output", side_effect=FileNotFoundError)
    def test_returns_empty_when_podman_missing(self, _co) -> None:
        """Missing podman → empty dict."""
        assert PodmanRuntime().container_states("p") == {}


class TestContainerState:
    """``Container.state`` returns a single container's state or ``None``."""

    @patch("terok_sandbox.runtime.podman.subprocess.check_output", return_value="Running\n")
    def test_returns_lowercased_state(self, _co) -> None:
        """Output is lowercased."""
        assert PodmanRuntime().container("foo").state == "running"

    @patch("terok_sandbox.runtime.podman.subprocess.check_output", return_value="\n")
    def test_empty_output_is_none(self, _co) -> None:
        """Empty podman output → ``None``."""
        assert PodmanRuntime().container("foo").state is None

    @patch("terok_sandbox.runtime.podman.subprocess.check_output", side_effect=FileNotFoundError)
    def test_podman_missing_is_none(self, _co) -> None:
        """Missing podman → ``None``."""
        assert PodmanRuntime().container("foo").state is None


class TestContainerRunning:
    """``Container.running`` is a shortcut around ``State.Running``."""

    @patch("terok_sandbox.runtime.podman.subprocess.check_output", return_value="true\n")
    def test_true(self, _co) -> None:
        """Podman ``true`` → ``True``."""
        assert PodmanRuntime().container("c").running is True

    @patch("terok_sandbox.runtime.podman.subprocess.check_output", return_value="false\n")
    def test_false(self, _co) -> None:
        """Podman ``false`` → ``False``."""
        assert PodmanRuntime().container("c").running is False

    @patch("terok_sandbox.runtime.podman.subprocess.check_output", side_effect=FileNotFoundError)
    def test_no_podman_is_false(self, _co) -> None:
        """Missing podman → ``False``."""
        assert PodmanRuntime().container("c").running is False


class TestContainerImage:
    """``Container.image`` returns an ``Image`` handle or ``None``."""

    @patch("terok_sandbox.runtime.podman.subprocess.check_output", return_value="sha256:abc\n")
    def test_returns_image_handle(self, _co) -> None:
        """Non-empty image id yields a handle whose ``ref`` matches."""
        image = PodmanRuntime().container("c").image
        assert image is not None
        assert image.ref == "sha256:abc"

    @patch("terok_sandbox.runtime.podman.subprocess.check_output", return_value="\n")
    def test_empty_output_is_none(self, _co) -> None:
        """Empty podman output → ``None``."""
        assert PodmanRuntime().container("c").image is None

    @patch(
        "terok_sandbox.runtime.podman.subprocess.check_output",
        side_effect=subprocess.CalledProcessError(125, "podman"),
    )
    def test_missing_container_is_none(self, _co) -> None:
        """Podman error → ``None``."""
        assert PodmanRuntime().container("c").image is None

    @patch("terok_sandbox.runtime.podman.subprocess.check_output", side_effect=FileNotFoundError)
    def test_no_podman_is_none(self, _co) -> None:
        """Missing podman → ``None``."""
        assert PodmanRuntime().container("c").image is None


# ── Image observation ─────────────────────────────────────────────────────


class TestImageExists:
    """``Image.exists`` translates ``podman image exists`` exit code."""

    @patch("terok_sandbox.runtime.podman.subprocess.run")
    def test_present(self, mock_run) -> None:
        """Exit 0 → present."""
        mock_run.return_value = subprocess.CompletedProcess(args=[], returncode=0)
        assert PodmanRuntime().image("terok-l1-cli:test").exists() is True

    @patch("terok_sandbox.runtime.podman.subprocess.run")
    def test_absent(self, mock_run) -> None:
        """Non-zero exit → absent."""
        mock_run.return_value = subprocess.CompletedProcess(args=[], returncode=1)
        assert PodmanRuntime().image("missing:tag").exists() is False

    @patch("terok_sandbox.runtime.podman.subprocess.run", side_effect=FileNotFoundError)
    def test_no_podman_is_false(self, _run) -> None:
        """Missing podman → ``False``."""
        assert PodmanRuntime().image("any:tag").exists() is False


class TestImageLabels:
    """``Image.labels`` parses ``Config.Labels`` from ``podman inspect``."""

    @patch("terok_sandbox.runtime.podman.subprocess.run")
    def test_returns_labels_dict(self, mock_run) -> None:
        """JSON labels parse into a flat ``str→str`` dict."""
        mock_run.return_value = subprocess.CompletedProcess(
            args=[],
            returncode=0,
            stdout='{"ai.terok.agents": "claude,codex", "build": "v1"}',
            stderr="",
        )
        assert PodmanRuntime().image("terok-l1-cli:test").labels() == {
            "ai.terok.agents": "claude,codex",
            "build": "v1",
        }

    @patch("terok_sandbox.runtime.podman.subprocess.run")
    def test_null_labels_returns_empty(self, mock_run) -> None:
        """Podman emits ``null`` when the image has no labels set."""
        mock_run.return_value = subprocess.CompletedProcess(
            args=[], returncode=0, stdout="null", stderr=""
        )
        assert PodmanRuntime().image("terok-l1-cli:test").labels() == {}

    @patch("terok_sandbox.runtime.podman.subprocess.run", side_effect=FileNotFoundError)
    def test_no_podman_returns_empty(self, _run) -> None:
        """Missing podman → empty dict."""
        assert PodmanRuntime().image("any:tag").labels() == {}

    @patch(
        "terok_sandbox.runtime.podman.subprocess.run",
        side_effect=subprocess.CalledProcessError(1, ["podman"]),
    )
    def test_missing_image_returns_empty(self, _run) -> None:
        """Podman error → empty dict."""
        assert PodmanRuntime().image("missing:tag").labels() == {}

    @patch("terok_sandbox.runtime.podman.subprocess.run")
    def test_unparseable_output_returns_empty(self, mock_run) -> None:
        """Garbled output does not leak as a ``JSONDecodeError``."""
        mock_run.return_value = subprocess.CompletedProcess(
            args=[], returncode=0, stdout="not-json", stderr=""
        )
        assert PodmanRuntime().image("any:tag").labels() == {}


class TestImagesList:
    """``PodmanRuntime.images`` parses ``podman images`` TSV output."""

    @patch("terok_sandbox.runtime.podman.subprocess.run")
    def test_parses_rows(self, mock_run) -> None:
        """Each line becomes an ``Image`` with pre-populated fields."""
        mock_run.return_value = subprocess.CompletedProcess(
            args=[],
            returncode=0,
            stdout=(
                "docker.io/terok/l1-cli\tubuntu-24.04\tabc123\t420MB\t2 hours ago\n"
                "docker.io/terok/l0\tubuntu-24.04\tdef456\t120MB\t1 day ago\n"
            ),
            stderr="",
        )
        images = PodmanRuntime().images()
        assert [img.ref for img in images] == ["abc123", "def456"]
        assert images[0].repository == "docker.io/terok/l1-cli"
        assert images[0].tag == "ubuntu-24.04"
        assert images[0].size == "420MB"
        assert images[0].created == "2 hours ago"

    @patch("terok_sandbox.runtime.podman.subprocess.run")
    def test_dangling_only_sets_filter(self, mock_run) -> None:
        """``dangling_only`` injects ``--filter dangling=true`` into the argv."""
        mock_run.return_value = subprocess.CompletedProcess(
            args=[], returncode=0, stdout="", stderr=""
        )
        PodmanRuntime().images(dangling_only=True)
        cmd = mock_run.call_args[0][0]
        assert "--filter" in cmd
        assert "dangling=true" in cmd

    @patch("terok_sandbox.runtime.podman.subprocess.run")
    def test_skips_malformed_rows(self, mock_run) -> None:
        """Malformed rows are silently skipped."""
        mock_run.return_value = subprocess.CompletedProcess(
            args=[],
            returncode=0,
            stdout="repo\ttag\tid\t100MB\tyesterday\nmalformed\n",
            stderr="",
        )
        assert len(PodmanRuntime().images()) == 1

    @patch("terok_sandbox.runtime.podman.subprocess.run")
    def test_podman_error_returns_empty(self, mock_run) -> None:
        """Podman error → empty list."""
        mock_run.return_value = subprocess.CompletedProcess(
            args=[], returncode=1, stdout="", stderr=""
        )
        assert PodmanRuntime().images() == []

    @patch("terok_sandbox.runtime.podman.subprocess.run", side_effect=FileNotFoundError)
    def test_no_podman_returns_empty(self, _run) -> None:
        """Missing podman → empty list."""
        assert PodmanRuntime().images() == []


class TestImageHistory:
    """``Image.history`` returns the ``CreatedBy`` string per layer."""

    @patch("terok_sandbox.runtime.podman.subprocess.run")
    def test_returns_history_lines(self, mock_run) -> None:
        """Each non-blank line becomes a history entry."""
        mock_run.return_value = subprocess.CompletedProcess(
            args=[],
            returncode=0,
            stdout="COPY . /app\nRUN apt-get install curl\nFROM ubuntu:24.04\n",
            stderr="",
        )
        assert PodmanRuntime().image("abc123").history() == [
            "COPY . /app",
            "RUN apt-get install curl",
            "FROM ubuntu:24.04",
        ]

    @patch("terok_sandbox.runtime.podman.subprocess.run")
    def test_empty_on_podman_error(self, mock_run) -> None:
        """Podman error → empty list."""
        mock_run.return_value = subprocess.CompletedProcess(
            args=[], returncode=1, stdout="", stderr="Error: no such image\n"
        )
        assert PodmanRuntime().image("abc123").history() == []

    @patch("terok_sandbox.runtime.podman.subprocess.run", side_effect=FileNotFoundError)
    def test_no_podman_returns_empty(self, _run) -> None:
        """Missing podman → empty list."""
        assert PodmanRuntime().image("abc123").history() == []


class TestImageRemove:
    """``Image.remove`` — best-effort; ``False`` on any failure."""

    @patch("terok_sandbox.runtime.podman.subprocess.run")
    def test_success(self, mock_run) -> None:
        """Exit 0 → ``True``."""
        mock_run.return_value = subprocess.CompletedProcess(args=[], returncode=0)
        assert PodmanRuntime().image("abc123").remove() is True

    @patch("terok_sandbox.runtime.podman.subprocess.run")
    def test_failure_returns_false(self, mock_run) -> None:
        """Non-zero exit → ``False``."""
        mock_run.return_value = subprocess.CompletedProcess(args=[], returncode=2)
        assert PodmanRuntime().image("abc123").remove() is False

    @patch("terok_sandbox.runtime.podman.subprocess.run", side_effect=FileNotFoundError)
    def test_no_podman_returns_false(self, _run) -> None:
        """Missing podman → ``False``."""
        assert PodmanRuntime().image("abc123").remove() is False


# ── Force remove (shape verification redundant with test_container_remove) ─


class TestForceRemoveShape:
    """Confirms the return shape — lives here alongside image/container tests."""

    @patch("terok_sandbox.runtime.podman.subprocess.run")
    def test_clean_remove_marks_removed(self, mock_run) -> None:
        """Clean remove yields a single ``removed=True`` entry."""
        mock_run.return_value = subprocess.CompletedProcess(
            args=[], returncode=0, stdout="", stderr=""
        )
        runtime = PodmanRuntime()
        results = runtime.force_remove([runtime.container("a")])
        assert results == [ContainerRemoveResult(name="a", removed=True)]


# ── Wait for exit ─────────────────────────────────────────────────────────


class TestContainerWait:
    """``Container.wait`` returns the container's real exit code.

    Signals timeouts, podman errors, and missing podman out of band —
    nothing is impersonated as an exit code.
    """

    @patch("terok_sandbox.runtime.podman.subprocess.run")
    def test_returns_parsed_exit_code(self, mock_run) -> None:
        """Numeric stdout is returned as an int."""
        mock_run.return_value = subprocess.CompletedProcess(
            args=[], returncode=0, stdout="42\n", stderr=""
        )
        assert PodmanRuntime().container("c").wait() == 42

    @patch("terok_sandbox.runtime.podman.subprocess.run")
    def test_returns_exit_code_124_distinctly(self, mock_run) -> None:
        """A container that legitimately exits 124 is returned as 124."""
        mock_run.return_value = subprocess.CompletedProcess(
            args=[], returncode=0, stdout="124\n", stderr=""
        )
        assert PodmanRuntime().container("c").wait() == 124

    @patch(
        "terok_sandbox.runtime.podman.subprocess.run",
        side_effect=subprocess.TimeoutExpired("p", 1),
    )
    def test_timeout_raises(self, _run) -> None:
        """Timeout raises ``TimeoutError``."""
        with pytest.raises(TimeoutError, match=r"did not exit within"):
            PodmanRuntime().container("c").wait(timeout=0.1)

    @patch("terok_sandbox.runtime.podman.subprocess.run", side_effect=FileNotFoundError)
    def test_missing_podman_raises(self, _run) -> None:
        """Missing podman propagates as ``FileNotFoundError``."""
        with pytest.raises(FileNotFoundError):
            PodmanRuntime().container("c").wait()

    @patch("terok_sandbox.runtime.podman.subprocess.run")
    def test_podman_wait_failure_raises(self, mock_run) -> None:
        """Non-zero ``podman wait`` returncode raises ``RuntimeError``."""
        mock_run.return_value = subprocess.CompletedProcess(
            args=[], returncode=125, stdout="", stderr="Error: no such container c\n"
        )
        with pytest.raises(RuntimeError, match=r"rc=125.*no such container"):
            PodmanRuntime().container("c").wait()

    @patch("terok_sandbox.runtime.podman.subprocess.run")
    def test_unparseable_stdout_raises(self, mock_run) -> None:
        """Non-numeric stdout raises ``RuntimeError`` with diagnostic context."""
        mock_run.return_value = subprocess.CompletedProcess(
            args=[], returncode=0, stdout="oops\n", stderr=""
        )
        with pytest.raises(RuntimeError, match=r"returned unexpected output.*oops"):
            PodmanRuntime().container("c").wait()


# ── Port reservation ─────────────────────────────────────────────────────


class TestReservePort:
    """Real socket-bind smoke tests — no mocking."""

    def test_reserve_yields_valid_port(self) -> None:
        """Reservation exposes a port in the dynamic range."""
        with PodmanRuntime().reserve_port() as reservation:
            assert 1024 <= reservation.port <= 65535

    def test_port_is_usable_while_held(self) -> None:
        """The bound socket can be listened on and connected to."""
        reservation = PodmanRuntime().reserve_port()
        try:
            # socket is exposed internally — test via a fresh connect round-trip
            sock = reservation._socket
            assert isinstance(sock, socket.socket)
            sock.listen(1)
            client = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            client.connect(("127.0.0.1", reservation.port))
            client.close()
        finally:
            reservation.close()

    def test_close_is_idempotent(self) -> None:
        """Close can be called multiple times without error."""
        reservation = PodmanRuntime().reserve_port()
        reservation.close()
        reservation.close()  # no-op second close


# ── Rootless network mode detection ───────────────────────────────────────


class TestDetectRootlessNetworkMode:
    """Detection probes ``podman info`` and falls back to slirp4netns."""

    @staticmethod
    def _proc(rc: int, stdout: str) -> subprocess.CompletedProcess[str]:
        """Build a completed process fixture."""
        return subprocess.CompletedProcess(args=[], returncode=rc, stdout=stdout, stderr="")

    @patch("terok_sandbox.runtime.podman.subprocess.run")
    def test_pasta_recognised(self, mock_run) -> None:
        """``pasta`` output returns ``"pasta"``."""
        mock_run.return_value = self._proc(0, "pasta\n")
        assert _detect_rootless_network_mode() == "pasta"

    @patch("terok_sandbox.runtime.podman.subprocess.run")
    def test_slirp_recognised(self, mock_run) -> None:
        """``slirp4netns`` output returns ``"slirp4netns"``."""
        mock_run.return_value = self._proc(0, "slirp4netns\n")
        assert _detect_rootless_network_mode() == "slirp4netns"

    @patch("terok_sandbox.runtime.podman.subprocess.run")
    def test_unknown_value_falls_back_to_slirp(self, mock_run) -> None:
        """Unrecognised output falls back to slirp4netns."""
        mock_run.return_value = self._proc(0, "bridge\n")
        assert _detect_rootless_network_mode() == "slirp4netns"

    @patch("terok_sandbox.runtime.podman.subprocess.run")
    def test_nonzero_exit_falls_back_to_slirp(self, mock_run) -> None:
        """Non-zero podman info exit falls back to slirp4netns."""
        mock_run.return_value = self._proc(1, "")
        assert _detect_rootless_network_mode() == "slirp4netns"

    @patch("terok_sandbox.runtime.podman.subprocess.run", side_effect=FileNotFoundError)
    def test_missing_podman_falls_back_to_slirp(self, _run) -> None:
        """Missing podman falls back to slirp4netns."""
        assert _detect_rootless_network_mode() == "slirp4netns"

    @patch(
        "terok_sandbox.runtime.podman.subprocess.run",
        side_effect=subprocess.TimeoutExpired("podman", 1),
    )
    def test_timeout_falls_back_to_slirp(self, _run) -> None:
        """Timeout falls back to slirp4netns."""
        assert _detect_rootless_network_mode() == "slirp4netns"


class TestBypassNetworkArgs:
    """``bypass_network_args`` picks args based on euid + detected network mode."""

    @patch("terok_sandbox.runtime.podman.os.geteuid", return_value=0)
    def test_root_emits_nothing(self, _euid) -> None:
        """Running as root requires no bypass args."""
        assert bypass_network_args(9418) == []

    @patch(
        "terok_sandbox.runtime.podman._detect_rootless_network_mode",
        return_value="slirp4netns",
    )
    @patch("terok_sandbox.runtime.podman.os.geteuid", return_value=1000)
    def test_slirp_args_include_loopback_allow(self, _euid, _net) -> None:
        """Slirp mode emits ``allow_host_loopback=true``."""
        args = bypass_network_args(9418)
        assert "slirp4netns:allow_host_loopback=true" in args
        assert any("host.containers.internal:" in a for a in args)

    @patch(
        "terok_sandbox.runtime.podman._detect_rootless_network_mode",
        return_value="pasta",
    )
    @patch("terok_sandbox.runtime.podman.os.geteuid", return_value=1000)
    def test_pasta_args_include_map_host_loopback(self, _euid, _net) -> None:
        """Pasta mode emits ``--map-host-loopback``."""
        args = bypass_network_args(9418)
        assert any("--map-host-loopback" in a for a in args)
