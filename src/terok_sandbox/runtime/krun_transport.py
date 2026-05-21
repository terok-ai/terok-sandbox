# SPDX-FileCopyrightText: 2026 Jiri Vyskocil
# SPDX-License-Identifier: Apache-2.0

"""TCP-bridged OpenSSH transport for `KrunRuntime`.

Implements the [`KrunTransport`][terok_sandbox.runtime.krun.KrunTransport]
protocol by shelling out to the system ``ssh`` client and reaching the
guest's sshd over a host TCP port that podman's passt has forwarded into
the guest namespace.  No custom wire protocol: sshd handles auth, PTY
allocation, signal forwarding, exit codes.

Why TCP-over-passt and not vsock: ``crun-krun`` does not configure
host-visible vsock for libkrun guests (it never calls
``krun_add_vsock``/``krun_add_vsock_port``), and libkrun's vsock
implementation is a userspace TSI bridge rather than a vhost-vsock
device the host kernel can route to.  ``socat - VSOCK-CONNECT:cid:port``
from the host therefore can't reach the guest regardless of CID.
``podman -p HOST:GUEST`` *does* compose correctly with crun-krun's
passt, so we forward a per-container host port to the guest's sshd
instead.  Costs a host-visible TCP port per task — acceptable while the
krun runtime stays behind the experimental flag.

Design choices and why:

- **stock ssh CLI** rather than a paramiko client.  The binary is
  battle-tested for the edge cases (PTY allocation, signal forwarding,
  EOF semantics) that we would otherwise reimplement.
- **Pubkey-only**, with ``IdentitiesOnly=yes`` so a stray host-side
  ssh-agent can't offer unrelated identities.  The host holds the
  private key; the guest receives the public half via a per-task
  bind-mount onto ``/etc/ssh/authorized_keys.d/terok`` (the image ships
  an empty placeholder, so it carries no per-installation secret and
  caches identically across hosts).
- **Argv-quoted remote command**: ``ssh host -- a b c`` concatenates
  the tokens and runs the result through the in-guest user's shell, so
  the transport ``shlex.quote``s each token to preserve the
  ``cmd: list[str]`` argv contract on the wire.
- **No host-key persistence**: ``StrictHostKeyChecking=no`` plus
  ``UserKnownHostsFile=/dev/null``.  The forwarded port is bound to
  ``127.0.0.1`` only (orchestrator-side reservation) and the krun
  runtime is gated on the experimental flag, so a wrong-endpoint
  connect is structurally restricted to a host with podman access
  (already root-equivalent).  Full host-key pinning would need
  orchestrator-side ``known_hosts`` plumbing and is tracked as a
  follow-up.

Endpoint discovery is pluggable via *endpoint_resolver* so unit tests
can synthesise endpoints without an actual microVM.  The default
production factory
[`podman_port_resolver`][terok_sandbox.runtime.krun_transport.podman_port_resolver]
asks podman directly for the host port forwarded to the guest's sshd —
no terok-private annotation in between.
"""

from __future__ import annotations

import re
import shlex
import subprocess  # nosec B404 — orchestrates the system ssh CLI
from collections.abc import Callable, Mapping
from dataclasses import dataclass
from pathlib import Path  # noqa: TC003 — used in dataclass field type
from typing import BinaryIO

from .podman import _start_stdio_pumps
from .protocol import Container, ExecResult

# ── Public defaults ─────────────────────────────────────────────────────────

# Guest TCP port the host-side resolver looks up.  Matches what the L0
# image's ``sshd-terok.service`` listens on.  Constant rather than
# parameter — both sides must agree, and the guest side is fixed by the
# image.
DEFAULT_GUEST_SSHD_PORT = 22

# Host address the forwarded port is bound to.  Loopback-only — the
# experimental tradeoff is that the port is visible to every local user
# on the box, but exposing it on a routable interface would broaden the
# attack surface needlessly.
DEFAULT_SSH_HOST = "127.0.0.1"

# SSH user inside the guest — the only account the hardened sshd
# config (``AllowUsers dev``) accepts.
DEFAULT_SSH_USER = "dev"

# In-guest working directory every remote payload chdirs into before
# exec'ing the user command.  Matches the L0 ``WORKDIR /workspace``
# that ``podman exec`` honours under crun, so the operator's starting
# cwd is the same across both runtimes (sshd ignores image WORKDIR).
_REMOTE_CWD = "/workspace"


# ── Endpoint ────────────────────────────────────────────────────────────────


@dataclass(frozen=True)
class TcpEndpoint:
    """A host TCP endpoint reachable via podman's passt port-forward.

    *port* is the host-side TCP port podman bound for this container's
    ``-p <port>:22`` mapping; *host* is the loopback address that port
    was bound to.

    Fields are int-coerced and range-checked in ``__post_init__`` — the
    transport interpolates *port* into the ssh argv and *host* into the
    user@host token, so a string carrying shell metacharacters or
    structural junk would otherwise reach the system ssh CLI.  Catching
    it here means a bad ``endpoint_resolver`` fails loudly at
    construction rather than silently building a hostile invocation.
    """

    port: int
    host: str = DEFAULT_SSH_HOST

    def __post_init__(self) -> None:
        """Coerce + bound-check both fields so the ssh argv stays safe."""
        try:
            port = int(self.port)
        except (TypeError, ValueError) as exc:
            raise ValueError(
                f"TcpEndpoint: port must be int-convertible, got port={self.port!r}"
            ) from exc
        if not 1 <= port <= _TCP_MAX_PORT:
            raise ValueError(f"TcpEndpoint: port {port} outside (0, 65535] range")
        if not _HOST_RE.fullmatch(self.host):
            raise ValueError(
                f"TcpEndpoint: host {self.host!r} must match {_HOST_RE.pattern} "
                "(loopback IPv4 / hostname charset)"
            )
        object.__setattr__(self, "port", port)


# ── Transport (the entry point) ─────────────────────────────────────────────


class TcpSSHTransport:
    """OpenSSH-over-loopback-TCP implementation of
    [`KrunTransport`][terok_sandbox.runtime.krun.KrunTransport].

    Holds the host-side identity (private key path) and an endpoint
    resolver that maps a [`Container`][terok_sandbox.runtime.protocol.Container]
    to a [`TcpEndpoint`][terok_sandbox.runtime.krun_transport.TcpEndpoint].
    The transport never touches the credentials vault directly — the
    orchestrator exports the ``%host`` key to a tmpfs file and passes
    that path in, keeping vault access out of the runtime layer.
    """

    def __init__(
        self,
        *,
        identity_file: Path,
        endpoint_resolver: Callable[[Container], TcpEndpoint],
        ssh_user: str = DEFAULT_SSH_USER,
        ssh_binary: str = "ssh",
    ) -> None:
        self._identity_file = identity_file
        self._resolver = endpoint_resolver
        self._user = ssh_user
        self._ssh = ssh_binary

    def exec(
        self,
        container: Container,
        cmd: list[str],
        *,
        timeout: float | None = None,
    ) -> ExecResult:
        """Run *cmd* in the guest and return its outcome.

        Each *cmd* token is ``shlex.quote``d into a single remote
        command string so the in-guest shell treats embedded
        metacharacters as literal data — argv semantics are preserved
        across the inherently-shell-parsed ssh wire format.
        """
        endpoint = self._resolver(container)
        remote_str = _remote_command(cmd)
        argv = [*self._ssh_argv(endpoint), "--", remote_str]
        proc = subprocess.run(  # nosec B603 — argv built from fixed verbs + caller-controlled scope/container names
            argv,
            capture_output=True,
            text=True,
            timeout=timeout,
            check=False,
        )
        return ExecResult(
            exit_code=proc.returncode,
            stdout=proc.stdout or "",
            stderr=proc.stderr or "",
        )

    def exec_stdio(
        self,
        container: Container,
        cmd: list[str],
        *,
        stdin: BinaryIO,
        stdout: BinaryIO,
        stderr: BinaryIO | None = None,
        env: Mapping[str, str] | None = None,
        timeout: float | None = None,
    ) -> int:
        """Bridge byte streams to *cmd* in the guest; return its exit code.

        Environment variables are propagated via a remote ``env`` prefix
        rather than ``SendEnv`` so the transport doesn't depend on the
        guest's ``AcceptEnv`` whitelist.  Env var **names** are
        validated against ``[A-Za-z_][A-Za-z0-9_]*`` because the remote
        ``env`` command expects bare identifiers; values and *cmd*
        tokens are ``shlex.quote``d so embedded shell metacharacters
        cross the wire as literal data.
        """
        endpoint = self._resolver(container)
        remote_str = _remote_command(cmd, env=env)
        argv = [*self._ssh_argv(endpoint), "--", remote_str]

        proc = subprocess.Popen(  # noqa: S603 — argv built above  # nosec B603 — argv is built from fixed verbs + caller-controlled scope/container names
            argv,
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE if stderr is not None else subprocess.DEVNULL,
        )
        _start_stdio_pumps(proc, stdin, stdout, stderr)
        try:
            return proc.wait(timeout=timeout)
        except subprocess.TimeoutExpired:
            proc.terminate()
            try:
                proc.wait(timeout=2)
            except subprocess.TimeoutExpired:
                proc.kill()
                proc.wait()
            raise

    def login_command(
        self,
        container: Container,
        *,
        command: tuple[str, ...] = (),
    ) -> list[str]:
        """Return an ``ssh`` argv that attaches a PTY to the guest's shell.

        Mirrors what [`PodmanContainer.login_command`][terok_sandbox.runtime.podman.PodmanContainer.login_command]
        does for the conventional runtime — emits the argv the operator
        (or ``terok login``) execs into.  Adds ``-tt`` so sshd allocates
        a real PTY even when stdin isn't a terminal (the caller may be
        running under tmux or an IDE proxy).

        Both the empty-*command* path (interactive login → ``bash -l``)
        and the explicit-*command* path land at ``/workspace`` via
        [`_at_workspace`][terok_sandbox.runtime.krun_transport._at_workspace],
        so the operator's starting cwd matches what ``podman exec``
        gives under crun.  Argv tokens past ``--`` are ``shlex.quote``d
        (same helper the exec paths use) so the SSH wire format
        preserves argv semantics across the login-shell parse on the
        far side.
        """
        endpoint = self._resolver(container)
        argv = self._ssh_argv(endpoint, interactive=True)
        remote = _remote_command(list(command)) if command else _at_workspace("bash -l")
        return [*argv, "--", remote]

    def _ssh_argv(self, endpoint: TcpEndpoint, *, interactive: bool = False) -> list[str]:
        """Build the ssh argv up to (but not including) the remote command.

        *interactive* swaps ``BatchMode=yes`` for ``-tt`` (force-allocate
        PTY) so the login flow gets a real terminal.  Exec paths keep
        the batch-mode default so a missing identity fails fast instead
        of prompting.
        """
        pty_flags = ["-tt"] if interactive else ["-o", "BatchMode=yes"]
        return [
            self._ssh,
            "-p",
            str(endpoint.port),
            "-i",
            str(self._identity_file),
            # Only the identity we explicitly passed is offered; a stray
            # ssh-agent running in the host environment can't slip in
            # additional keys that happen to be accepted by the guest.
            "-o",
            "IdentitiesOnly=yes",
            # The forwarded port is loopback-bound by the orchestrator and
            # the krun runtime is gated on the experimental flag, so a
            # wrong-endpoint connect is structurally restricted to a host
            # with podman access (already root-equivalent).  Full host-key
            # pinning would need orchestrator-side known_hosts plumbing
            # and is tracked as a follow-up.
            "-o",
            "StrictHostKeyChecking=no",
            "-o",
            "UserKnownHostsFile=/dev/null",
            # Quiet ssh's "Warning: Permanently added …" noise so it
            # doesn't pollute caller-visible stderr.
            "-o",
            "LogLevel=ERROR",
            # PTY/batch posture differs per call site (see *interactive*).
            *pty_flags,
            f"{self._user}@{endpoint.host}",
        ]


# ── Endpoint resolvers ──────────────────────────────────────────────────────


def podman_port_resolver(
    *,
    guest_port: int = DEFAULT_GUEST_SSHD_PORT,
    host: str = DEFAULT_SSH_HOST,
) -> Callable[[Container], TcpEndpoint]:
    """Return a resolver that reads the forwarded host port via ``podman port``.

    The orchestrator launches the container with ``-p <reserved>:22``;
    podman already records that mapping in its own metadata, so this
    resolver just asks for it back — no terok-private annotation in the
    middle.  ``podman port <name> <guest_port>/tcp`` emits a single
    ``<host_ip>:<host_port>`` line per matching mapping, which is
    exactly what we need.

    The resolved host is overridden to *host* (loopback by default) so
    the SSH connect goes through ``127.0.0.1`` even when pasta bound
    the forward to ``0.0.0.0``; trusting whatever podman reports would
    open the door to reaching the guest via a routable interface.
    """

    def _resolve(container: Container) -> TcpEndpoint:
        # ``--`` ends podman's own option parsing, so a container handle
        # carrying a leading-dash name can't be reinterpreted as a flag.
        argv = ["podman", "port", "--", container.name, f"{guest_port}/tcp"]
        # A short timeout keeps the resolver from blocking forever on a
        # wedged podman (daemon trouble, NFS-backed storage stall):
        # ``podman port`` is a metadata read, so 5 s is generous.  Raise
        # ``RuntimeError`` for every failure mode so callers see one
        # exception type across "no mapping", "unparseable output", and
        # "podman didn't answer".
        try:
            out = subprocess.check_output(  # nosec B603 B607 — argv built from fixed verbs + caller-controlled scope/container names — binary PATH lookup is the cross-distro contract
                argv,
                text=True,
                timeout=_RESOLVER_PORT_TIMEOUT_S,
            ).strip()
        except subprocess.CalledProcessError as exc:
            raise RuntimeError(
                f"podman port failed for container {container.name!r}: {exc} — "
                f"no ``-p HOST:{guest_port}`` mapping at launch?"
            ) from exc
        except subprocess.TimeoutExpired as exc:
            raise RuntimeError(
                f"podman port timed out after {_RESOLVER_PORT_TIMEOUT_S}s "
                f"resolving forwarded port for container {container.name!r} — "
                "podman daemon stuck or storage backend stalled"
            ) from exc
        if not out:
            raise RuntimeError(
                f"container {container.name!r} has no {guest_port}/tcp port mapping — "
                f"the orchestrator must launch with ``-p HOST:{guest_port}``"
            )
        # Take the first mapping line; podman emits one per binding (it
        # would only emit several if the operator added extra ``-p`` for
        # the same guest port).  ``rpartition`` lets the host-ip side
        # contain colons (IPv6 literals) without us having to special-case.
        first_line = out.splitlines()[0]
        _, sep, port_str = first_line.rpartition(":")
        if not sep:
            raise RuntimeError(
                f"container {container.name!r} podman-port output {first_line!r} "
                f"doesn't look like ``<host>:<port>``"
            )
        try:
            port = int(port_str)
        except ValueError as exc:
            raise RuntimeError(
                f"container {container.name!r} podman-port output {first_line!r} "
                f"has non-integer port: {port_str!r}"
            ) from exc
        # ``TcpEndpoint.__post_init__`` does the range check.
        try:
            return TcpEndpoint(port=port, host=host)
        except ValueError as exc:
            raise RuntimeError(
                f"container {container.name!r} has invalid forwarded port {port}: {exc}"
            ) from exc

    return _resolve


# ── Private helpers ─────────────────────────────────────────────────────────

_RESOLVER_PORT_TIMEOUT_S: float = 5.0
"""Bound on ``podman port`` in ``podman_port_resolver``.

It is a metadata read — 5 s leaves comfortable headroom over a healthy
podman + storage backend while still surfacing a wedged daemon as a
loud ``RuntimeError`` instead of a forever-hang."""

_TCP_MAX_PORT = 0xFFFF  # u16

# Loopback IPv4 literals or DNS-shaped hostnames only.  The transport
# interpolates ``host`` into the ``user@host`` token and a TCP-port
# argument; refusing anything outside this charset keeps shell
# metacharacters and structural junk out of the system ssh CLI.
_HOST_RE = re.compile(r"^[A-Za-z0-9][A-Za-z0-9.\-]*$")

# ``ssh host -- arg1 arg2`` does NOT preserve argv on the remote side —
# sshd concatenates the tokens and runs the result through the user's
# login shell.  Our public API contract is ``cmd: list[str]`` (argv
# semantics), so every remote token gets ``shlex.quote``d before going
# over the wire to keep that contract honest.  Env var *names* are
# validated against this strict pattern because there is no portable
# way to quote them — the remote ``env`` command expects bare
# identifiers.
_REMOTE_ENV_NAME_RE = re.compile(r"^[A-Za-z_][A-Za-z0-9_]*$")


def _remote_command(cmd: list[str], *, env: Mapping[str, str] | None = None) -> str:
    """Render *cmd* (and optional *env*) into a shell-safe remote string.

    OpenSSH's ``ssh host -- a b c`` concatenates the post-``--`` tokens
    and runs the result through the in-guest user's login shell, so the
    transport must do the quoting itself to honour the ``cmd: list[str]``
    argv contract on the wire.  The payload is then routed through
    [`_at_workspace`][terok_sandbox.runtime.krun_transport._at_workspace]
    so every remote command starts at the image's ``WORKDIR /workspace``
    — matching what ``podman exec`` gives under crun.

    Env var names are validated against ``[A-Za-z_][A-Za-z0-9_]*`` —
    the remote ``env`` command expects bare identifiers and there's no
    portable way to quote them.  Names that fail the pattern raise
    ``ValueError`` rather than risk silently being misinterpreted.
    """
    if not cmd:
        raise ValueError("remote cmd must not be empty")
    tokens: list[str] = []
    if env:
        for key in env:
            if not _REMOTE_ENV_NAME_RE.fullmatch(key):
                raise ValueError(f"remote env var name {key!r}: must match [A-Za-z_][A-Za-z0-9_]*")
        tokens.append("env")
        tokens.extend(f"{k}={v}" for k, v in env.items())
    tokens.extend(cmd)
    return _at_workspace(" ".join(shlex.quote(t) for t in tokens))


def _at_workspace(payload: str) -> str:
    """Wrap *payload* so it runs with cwd = ``/workspace``.

    sshd ignores image ``WORKDIR`` and lands the session in ``$HOME``;
    ``podman exec`` honours it.  Equalising the two at the transport
    boundary keeps the operator-visible behaviour identical across
    runtimes.  ``&&`` makes a missing ``/workspace`` fail loud with
    sh's own ``cd: /workspace: No such file or directory`` rather than
    silently fall back to ``$HOME``; ``exec`` keeps the process tree
    as shallow as the direct ``podman exec`` path.
    """
    return f"cd {shlex.quote(_REMOTE_CWD)} && exec {payload}"
