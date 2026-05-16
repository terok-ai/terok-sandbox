# SPDX-FileCopyrightText: 2026 Jiri Vyskocil
# SPDX-License-Identifier: Apache-2.0

"""Tests for `VsockSSHTransport` and `podman_annotation_resolver`.

Real SSH never runs — every test mocks ``subprocess.run`` /
``subprocess.Popen`` and asserts the assembled argv.  The transport
itself is shape-only: the only "logic" it owns is the argv it emits
and where it routes streams.
"""

from __future__ import annotations

import io
import subprocess
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from terok_sandbox.runtime import ExecResult, KrunTransport
from terok_sandbox.runtime.krun_transport import (
    DEFAULT_CID_ANNOTATION,
    VsockEndpoint,
    VsockSSHTransport,
    podman_annotation_resolver,
)


class _StubContainer:
    """Minimal container handle — only ``name`` is read."""

    def __init__(self, name: str) -> None:
        self.name = name


def _make_transport(*, cid: int = 42, port: int = 22) -> VsockSSHTransport:
    """Build a transport whose resolver returns a fixed endpoint."""
    return VsockSSHTransport(
        identity_file=Path("/tmp/host.key"),  # noqa: S108 — test-only path
        endpoint_resolver=lambda _ctr: VsockEndpoint(cid=cid, port=port),
    )


class TestVsockEndpoint:
    """Trivial value type — assert defaults + frozen-dataclass semantics."""

    def test_port_defaults_to_22(self) -> None:
        """Vsock sshd is by convention on port 22 (matches L0G socket unit)."""
        assert VsockEndpoint(cid=3).port == 22

    def test_frozen(self) -> None:
        ep = VsockEndpoint(cid=3)
        with pytest.raises(AttributeError):
            ep.cid = 4  # type: ignore[misc]


class TestVsockSSHTransportArgv:
    """Argv assembly is the only side-effect-free thing in the transport."""

    def test_implements_protocol(self) -> None:
        """`VsockSSHTransport` is structurally a `KrunTransport`."""
        assert isinstance(_make_transport(), KrunTransport)

    def test_argv_contains_proxycommand_for_vsock(self) -> None:
        """The ProxyCommand encodes the CID + port for socat to bridge."""
        argv = _make_transport(cid=7, port=22)._ssh_argv(VsockEndpoint(cid=7, port=22))
        proxy_idx = argv.index("ProxyCommand=socat - VSOCK-CONNECT:7:22")
        # Confirm it's actually a value of an `-o` flag, not loose.
        assert argv[proxy_idx - 1] == "-o"

    def test_argv_uses_batch_and_disables_host_key_check(self) -> None:
        """No prompts; vsock channel obviates host-key persistence."""
        argv = _make_transport()._ssh_argv(VsockEndpoint(cid=3))
        opts = [argv[i + 1] for i, t in enumerate(argv) if t == "-o"]
        assert "BatchMode=yes" in opts
        assert "StrictHostKeyChecking=no" in opts
        assert "UserKnownHostsFile=/dev/null" in opts

    def test_argv_carries_identity_file(self) -> None:
        """``-i <key_path>`` carries the host-side private key."""
        argv = _make_transport()._ssh_argv(VsockEndpoint(cid=3))
        idx = argv.index("-i")
        assert argv[idx + 1] == "/tmp/host.key"

    def test_argv_user_at_dummy_host(self) -> None:
        """Hostname is a label only; ProxyCommand does the actual connect."""
        argv = _make_transport()._ssh_argv(VsockEndpoint(cid=3))
        assert argv[-1].startswith("dev@")


class TestVsockSSHTransportExec:
    """`exec` shells out and packages the result into an `ExecResult`."""

    def test_exec_invokes_ssh_with_remote_command(self) -> None:
        with patch("subprocess.run") as run:
            run.return_value = subprocess.CompletedProcess(
                args=[], returncode=0, stdout="hi\n", stderr=""
            )
            result = _make_transport(cid=3).exec(_StubContainer("ctr"), ["echo", "hi"])
        argv = run.call_args[0][0]
        # `--` separates ssh flags from the remote command.
        assert "--" in argv
        sep = argv.index("--")
        assert argv[sep + 1 :] == ["echo", "hi"]
        assert result == ExecResult(exit_code=0, stdout="hi\n", stderr="")

    def test_exec_propagates_nonzero_exit_and_stderr(self) -> None:
        with patch("subprocess.run") as run:
            run.return_value = subprocess.CompletedProcess(
                args=[], returncode=42, stdout="", stderr="boom\n"
            )
            result = _make_transport().exec(_StubContainer("ctr"), ["false"])
        assert result == ExecResult(exit_code=42, stdout="", stderr="boom\n")

    def test_exec_propagates_timeout(self) -> None:
        with patch("subprocess.run") as run:
            run.return_value = subprocess.CompletedProcess(
                args=[], returncode=0, stdout="", stderr=""
            )
            _make_transport().exec(_StubContainer("ctr"), ["true"], timeout=5.0)
        assert run.call_args.kwargs.get("timeout") == 5.0

    def test_exec_resolves_endpoint_per_container(self) -> None:
        """The endpoint resolver is called with the actual container handle."""
        seen: list[str] = []

        def resolver(container):  # type: ignore[no-untyped-def]
            seen.append(container.name)
            return VsockEndpoint(cid=99)

        transport = VsockSSHTransport(
            identity_file=Path("/tmp/host.key"),  # noqa: S108
            endpoint_resolver=resolver,
        )
        with patch("subprocess.run") as run:
            run.return_value = subprocess.CompletedProcess(
                args=[], returncode=0, stdout="", stderr=""
            )
            transport.exec(_StubContainer("task-abc"), ["true"])
        assert seen == ["task-abc"]


class TestVsockSSHTransportExecStdio:
    """`exec_stdio` spawns a Popen and bridges stdio via the pump helpers."""

    def test_env_is_prefixed_on_remote_side(self) -> None:
        """``env K=V`` prefix avoids dependence on guest's `AcceptEnv`."""
        with patch("subprocess.Popen") as popen:
            proc = MagicMock()
            proc.stdin = io.BytesIO()
            proc.stdout = io.BytesIO()
            proc.stderr = None
            proc.wait.return_value = 0
            popen.return_value = proc
            _make_transport().exec_stdio(
                _StubContainer("ctr"),
                ["sh"],
                stdin=io.BytesIO(),
                stdout=io.BytesIO(),
                env={"FOO": "1", "BAR": "x"},
            )
        argv = popen.call_args[0][0]
        sep = argv.index("--")
        # The env prefix lands at the start of the remote command.
        assert argv[sep + 1] == "env"
        assert "FOO=1" in argv[sep + 2 : sep + 4]
        assert "BAR=x" in argv[sep + 2 : sep + 4]
        assert argv[-1] == "sh"

    def test_no_env_skips_prefix(self) -> None:
        """Without env, the remote command is the bare argv."""
        with patch("subprocess.Popen") as popen:
            proc = MagicMock()
            proc.stdin = io.BytesIO()
            proc.stdout = io.BytesIO()
            proc.stderr = None
            proc.wait.return_value = 7
            popen.return_value = proc
            rc = _make_transport().exec_stdio(
                _StubContainer("ctr"),
                ["sh"],
                stdin=io.BytesIO(),
                stdout=io.BytesIO(),
            )
        argv = popen.call_args[0][0]
        sep = argv.index("--")
        assert argv[sep + 1 :] == ["sh"]
        assert rc == 7


class TestPodmanAnnotationResolver:
    """The default resolver shells `podman inspect` for the CID annotation."""

    def test_returns_endpoint_from_annotation(self) -> None:
        resolver = podman_annotation_resolver()
        with patch("subprocess.check_output", return_value="42\n"):
            endpoint = resolver(_StubContainer("ctr"))
        assert endpoint == VsockEndpoint(cid=42, port=22)

    def test_uses_custom_port(self) -> None:
        resolver = podman_annotation_resolver(port=2222)
        with patch("subprocess.check_output", return_value="3\n"):
            endpoint = resolver(_StubContainer("ctr"))
        assert endpoint.port == 2222

    def test_raises_when_annotation_empty(self) -> None:
        resolver = podman_annotation_resolver()
        with patch("subprocess.check_output", return_value="\n"):
            with pytest.raises(RuntimeError, match="no .* annotation"):
                resolver(_StubContainer("ctr"))

    def test_raises_when_annotation_not_integer(self) -> None:
        resolver = podman_annotation_resolver()
        with patch("subprocess.check_output", return_value="not-a-cid\n"):
            with pytest.raises(RuntimeError, match="non-integer"):
                resolver(_StubContainer("ctr"))

    def test_raises_when_podman_inspect_fails(self) -> None:
        resolver = podman_annotation_resolver()
        with patch(
            "subprocess.check_output",
            side_effect=subprocess.CalledProcessError(1, "podman"),
        ):
            with pytest.raises(RuntimeError, match="podman inspect failed"):
                resolver(_StubContainer("ctr"))

    def test_invokes_inspect_with_named_annotation(self) -> None:
        resolver = podman_annotation_resolver("custom.key")
        with patch("subprocess.check_output", return_value="9\n") as ck:
            resolver(_StubContainer("ctr"))
        argv = ck.call_args[0][0]
        fmt_idx = argv.index("--format") + 1
        assert "custom.key" in argv[fmt_idx]

    def test_default_annotation_constant_is_used(self) -> None:
        """`DEFAULT_CID_ANNOTATION` is the agreed key between orchestrator and resolver."""
        resolver = podman_annotation_resolver()
        with patch("subprocess.check_output", return_value="1\n") as ck:
            resolver(_StubContainer("ctr"))
        fmt = ck.call_args[0][0][ck.call_args[0][0].index("--format") + 1]
        assert DEFAULT_CID_ANNOTATION in fmt
