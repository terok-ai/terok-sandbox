# SPDX-FileCopyrightText: 2026 Jiri Vyskocil
# SPDX-License-Identifier: Apache-2.0

"""ASLR opt-in extends only personality grants and preserves the host policy."""

import json
import subprocess
from pathlib import Path
from unittest.mock import patch

import pytest

from terok_sandbox.runtime.podman import default_seccomp_profile
from terok_sandbox.runtime.seccomp import aslr_control_args


@pytest.fixture
def host_profile(tmp_path: Path) -> Path:
    """A default-deny policy with architecture and capability conditions."""
    path = tmp_path / "host-seccomp.json"
    path.write_text(
        json.dumps(
            {
                "defaultAction": "SCMP_ACT_ERRNO",
                "defaultErrnoRet": 38,
                "archMap": [
                    {"architecture": "SCMP_ARCH_AARCH64", "subArchitectures": ["SCMP_ARCH_ARM"]}
                ],
                "syscalls": [
                    {"names": ["read", "write"], "action": "SCMP_ACT_ALLOW"},
                    {
                        "names": ["perf_event_open"],
                        "action": "SCMP_ACT_ALLOW",
                        "includes": {"caps": ["CAP_PERFMON"]},
                    },
                    {"names": ["ptrace"], "action": "SCMP_ACT_ERRNO", "errnoRet": 1},
                    {
                        "names": ["personality"],
                        "action": "SCMP_ACT_ALLOW",
                        "args": [{"index": 0, "value": 0, "op": "SCMP_CMP_EQ"}],
                    },
                ],
            }
        )
    )
    return path


def test_aslr_profile_preserves_every_existing_rule(host_profile: Path, tmp_path: Path) -> None:
    """No host rule, capability condition, architecture, or default is replaced."""
    original = host_profile.read_text()
    with patch("terok_sandbox.runtime.seccomp.default_seccomp_profile", return_value=host_profile):
        args = aslr_control_args(tmp_path / "state")
    assert args[0] == "--security-opt"
    profile_path = Path(args[1].removeprefix("seccomp="))
    profile = json.loads(profile_path.read_text())
    added = profile["syscalls"][-5:]
    assert added == [
        {
            "names": ["personality"],
            "action": "SCMP_ACT_ALLOW",
            "args": [{"index": 0, "value": value, "op": "SCMP_CMP_EQ"}],
        }
        for value in (0xFFFFFFFF, 0x40000, 0x40008, 0x60000, 0x60008)
    ]
    del profile["syscalls"][-5:]
    assert profile == json.loads(original)
    assert host_profile.read_text() == original
    assert list(profile_path.parent.iterdir()) == [profile_path]


@pytest.mark.parametrize(
    "body",
    [
        "not JSON",
        "[]",
        "{}",
        '{"defaultAction": "SCMP_ACT_ALLOW", "syscalls": []}',
        '{"defaultAction": "SCMP_ACT_ERRNO", "syscalls": [{"names": ["personality", "ptrace"], "action": "SCMP_ACT_ERRNO"}]}',
    ],
)
def test_unusable_profile_fails_closed(tmp_path: Path, body: str) -> None:
    """Invalid or conflicting profiles never silently become an unconfined launch."""
    source = tmp_path / "bad.json"
    source.write_text(body)
    state = tmp_path / "state"
    with patch("terok_sandbox.runtime.seccomp.default_seccomp_profile", return_value=source):
        with pytest.raises(SystemExit, match="run.aslr_control"):
            aslr_control_args(state)
    assert not state.exists()


def test_profile_read_failure_fails_closed(tmp_path: Path) -> None:
    """A disappearing profile is not replaced by a less restrictive fallback."""
    with patch(
        "terok_sandbox.runtime.seccomp.default_seccomp_profile", return_value=tmp_path / "missing"
    ):
        with pytest.raises(SystemExit, match="run.aslr_control"):
            aslr_control_args(tmp_path / "state")


def test_profile_can_be_regenerated(host_profile: Path, tmp_path: Path) -> None:
    """Reusing a container name picks up the current host policy at creation."""
    state = tmp_path / "state"
    with patch("terok_sandbox.runtime.seccomp.default_seccomp_profile", return_value=host_profile):
        aslr_control_args(state)
        profile = json.loads(host_profile.read_text())
        profile["defaultErrnoRet"] = 1
        host_profile.write_text(json.dumps(profile))
        args = aslr_control_args(state)
    assert json.loads(Path(args[1].removeprefix("seccomp=")).read_text())["defaultErrnoRet"] == 1


def test_discover_configured_profile(host_profile: Path) -> None:
    """Podman's configured path, rather than a distro-specific guess, wins."""
    with patch(
        "terok_sandbox.runtime.podman.subprocess.run",
        return_value=subprocess.CompletedProcess([], 0, f"{host_profile}\n"),
    ) as run:
        assert default_seccomp_profile() == host_profile
    assert run.call_args.args[0] == [
        "podman",
        "info",
        "--format",
        "{{.Host.Security.SeccompProfilePath}}",
    ]
    assert run.call_args.kwargs["check"] is True
    assert run.call_args.kwargs["timeout"] > 0


@pytest.mark.parametrize("output", ["", "<no value>", "unconfined"])
def test_missing_profile_path_is_rejected(output: str) -> None:
    """A built-in or disabled profile cannot be safely extended from a file."""
    with patch(
        "terok_sandbox.runtime.podman.subprocess.run",
        return_value=subprocess.CompletedProcess([], 0, output),
    ):
        with pytest.raises(SystemExit, match="readable host Podman seccomp profile"):
            default_seccomp_profile()


@pytest.mark.parametrize(
    "error",
    [
        FileNotFoundError(),
        subprocess.CalledProcessError(125, "podman"),
        subprocess.TimeoutExpired("podman", 30),
    ],
)
def test_profile_discovery_failure_is_fatal(error: Exception) -> None:
    """Failed or timed-out discovery cannot skip the requested confinement."""
    with patch("terok_sandbox.runtime.podman.subprocess.run", side_effect=error):
        with pytest.raises(SystemExit, match="cannot discover"):
            default_seccomp_profile()
