# SPDX-FileCopyrightText: 2026 Jiri Vyskocil
# SPDX-License-Identifier: Apache-2.0

"""Host execution boundaries use the current absolute-PATH tool selection."""

from __future__ import annotations

import importlib
import subprocess
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from terok_util import require_host_tool

from terok_sandbox import PodmanRuntime, Sandbox
from terok_sandbox._setup_manual import _sudo_run
from terok_sandbox.gate.mirror import _git
from terok_sandbox.runtime.krun_transport import TcpEndpoint, TcpSSHTransport
from terok_sandbox.supervisor.main import _wait_for_container


@pytest.fixture
def host_profiles(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> tuple[Path, Path]:
    """Two operator tool profiles and an untrusted cwd, without real container tools."""
    monkeypatch.chdir(tmp_path)
    profiles = (tmp_path / "first", tmp_path / "second")
    for directory in (tmp_path, *profiles):
        directory.mkdir(exist_ok=True)
        for name in ("podman", "git", "ssh", "sudo", "bash"):
            binary = directory / name
            binary.write_text("#!/bin/sh\nexit 0\n")
            binary.chmod(0o755)
    for module in ("sandbox", "runtime.podman", "runtime.krun_transport", "supervisor.main"):
        monkeypatch.setattr(
            importlib.import_module(f"terok_sandbox.{module}"),
            "require_host_tool",
            require_host_tool,
        )
    return profiles


@pytest.mark.parametrize("boundary", ["runtime", "sandbox", "gate"])
def test_subprocess_uses_current_profile(
    boundary: str, host_profiles: tuple[Path, Path], monkeypatch: pytest.MonkeyPatch
) -> None:
    """The same object follows PATH changes and cannot run a cwd lookalike."""
    runtime = PodmanRuntime()
    sandbox = Sandbox()
    with (
        patch("subprocess.run", return_value=subprocess.CompletedProcess([], 0, "", "")) as run,
        patch("terok_sandbox.runtime.podman.run_captured") as capture,
    ):
        for profile in host_profiles:
            monkeypatch.setenv("PATH", f":.:relative:{profile}")
            if boundary == "runtime":
                runtime.exec(runtime.container("ctr"), ["git", "status"])
                call, tool = capture, "podman"
            elif boundary == "sandbox":
                sandbox._exec_podman(["podman", "start", "ctr"])
                call, tool = run, "podman"
            else:
                _git(profile, "status")
                call, tool = run, "git"
            assert call.call_args.kwargs["executable"] == str(profile / tool)


def test_login_argv_uses_current_profile(
    host_profiles: tuple[Path, Path], monkeypatch: pytest.MonkeyPatch
) -> None:
    """Returned Podman and SSH login argv already bind the selected host executable."""
    container = PodmanRuntime().container("ctr")
    ssh = TcpSSHTransport(
        identity_file=host_profiles[0] / "identity",
        endpoint_resolver=lambda _: TcpEndpoint(port=22),
    )
    for profile in host_profiles:
        monkeypatch.setenv("PATH", f":.:relative:{profile}")
        assert container.login_command(command=("bash",))[0] == str(profile / "podman")
        assert ssh.login_command(container, command=("bash",))[0] == str(profile / "ssh")


def test_missing_podman_preserves_optional_probe_fallback(
    host_profiles: tuple[Path, Path], monkeypatch: pytest.MonkeyPatch
) -> None:
    """Without an allowed tool directory, optional inspect still returns unknown."""
    monkeypatch.setenv("PATH", ":.:relative")
    container = PodmanRuntime().container("ctr")
    assert container.state is None
    with pytest.raises(FileNotFoundError, match="podman"):
        container.login_command()


def test_owned_installer_resolves_both_sudo_and_bash(
    host_profiles: tuple[Path, Path], monkeypatch: pytest.MonkeyPatch
) -> None:
    """The bundled installer does not lose its Bash selection inside sudo."""
    with patch("subprocess.run", return_value=subprocess.CompletedProcess([], 0)) as run:
        for profile in host_profiles:
            monkeypatch.setenv("PATH", f":.:relative:{profile}")
            assert _sudo_run((str(profile / "installer.sh"),)) == 0
            assert run.call_args.args[0][:2] == [str(profile / "sudo"), str(profile / "bash")]


@pytest.mark.parametrize("missing", ["sudo", "bash"])
def test_owned_installer_names_missing_host_tool(
    missing: str, host_profiles: tuple[Path, Path], monkeypatch: pytest.MonkeyPatch
) -> None:
    """Missing prerequisites fail before invoking the privileged installer."""
    profile = host_profiles[0]
    (profile / missing).unlink()
    monkeypatch.setenv("PATH", str(profile))
    with pytest.raises(SystemExit, match=f"{missing} not found"):
        _sudo_run((str(profile / "installer.sh"),))


@pytest.mark.asyncio
async def test_supervisor_wait_uses_current_profile(
    host_profiles: tuple[Path, Path], monkeypatch: pytest.MonkeyPatch
) -> None:
    """The asynchronous supervisor watcher obeys the same host tool selection."""
    proc = MagicMock(returncode=0, communicate=AsyncMock(return_value=(b"0", b"")))
    with patch(
        "asyncio.create_subprocess_exec", new_callable=AsyncMock, return_value=proc
    ) as spawn:
        for profile in host_profiles:
            monkeypatch.setenv("PATH", f":.:relative:{profile}")
            assert await _wait_for_container("ctr") == 0
            assert spawn.call_args.args[0] == str(profile / "podman")
