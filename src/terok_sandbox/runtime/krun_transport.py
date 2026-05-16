# SPDX-FileCopyrightText: 2026 Jiri Vyskocil
# SPDX-License-Identifier: Apache-2.0

"""Real OpenSSH-over-vsock transport for `KrunRuntime`.

Implements the [`KrunTransport`][terok_sandbox.runtime.krun.KrunTransport]
protocol by shelling out to the system ``ssh`` client with a ``socat``
``ProxyCommand`` that bridges to AF_VSOCK.  No custom wire protocol:
sshd handles auth, PTY allocation, signal forwarding, exit codes.

Design choices and why:

- **stock ssh CLI + socat**, not a paramiko + AF_VSOCK client.  Both
  binaries are battle-tested for the edge cases (PTY allocation, signal
  forwarding, EOF semantics) that we would otherwise reimplement.
- **Pubkey-only**, authenticating with the ``%host`` keypair from the
  credentials vault.  Host knows the private key; guest holds only the
  public half (baked into ``authorized_keys.d/terok`` at image build).
- **No host-key persistence**: ``StrictHostKeyChecking=no`` plus
  ``UserKnownHostsFile=/dev/null``.  The transport is structural — a
  guest reachable only over our own vsock channel cannot be a different
  guest pretending to be ours.

Endpoint discovery is pluggable via *endpoint_resolver* so unit tests
can synthesise endpoints without an actual microVM.  A production
factory ([`podman_annotation_resolver`][terok_sandbox.runtime.krun_transport.podman_annotation_resolver])
reads the CID from a podman annotation set at task launch.

Manual integration tests live alongside the package; podman-dependent
tests do not run in CI per project rules.
"""

from __future__ import annotations

import subprocess  # nosec B404 — orchestrates the system ssh CLI
from collections.abc import Callable, Mapping
from dataclasses import dataclass
from pathlib import Path  # noqa: TC003 — used in dataclass field type
from typing import BinaryIO

from .podman import _start_stdio_pumps
from .protocol import Container, ExecResult

# Default annotation key the orchestrator sets at ``podman run`` time so
# the host side can find the guest's vsock CID after the fact.  Read by
# [`podman_annotation_resolver`][terok_sandbox.runtime.krun_transport.podman_annotation_resolver];
# the constant is exposed so the orchestrator can name the key from one
# place rather than hard-coding the same literal twice.
DEFAULT_CID_ANNOTATION = "terok.krun.cid"

# Vsock port the guest sshd listens on, matching the systemd socket
# unit drop-in baked into the L0G guest image.  Constant rather than
# parameter — both sides must agree, and the guest side is fixed by
# the image.  Exposed so tests can reference the same value.
DEFAULT_VSOCK_SSHD_PORT = 22

# SSH user inside the guest — the only account the hardened sshd
# config (``AllowUsers dev``) accepts.
DEFAULT_SSH_USER = "dev"


@dataclass(frozen=True)
class VsockEndpoint:
    """A vsock endpoint reachable from the host.

    *cid* is a libkrun-assigned context ID (32-bit integer); *port* is
    the vsock port the in-guest service is listening on.
    """

    cid: int
    port: int = DEFAULT_VSOCK_SSHD_PORT


class VsockSSHTransport:
    """OpenSSH-over-vsock implementation of
    [`KrunTransport`][terok_sandbox.runtime.krun.KrunTransport].

    Holds the host-side identity (private key path) and an endpoint
    resolver that maps a [`Container`][terok_sandbox.runtime.protocol.Container]
    to a [`VsockEndpoint`][terok_sandbox.runtime.krun_transport.VsockEndpoint].
    The transport never touches the credentials vault directly — the
    orchestrator exports the ``%host`` key to a tmpfs file and passes
    that path in, keeping vault access out of the runtime layer.
    """

    def __init__(
        self,
        *,
        identity_file: Path,
        endpoint_resolver: Callable[[Container], VsockEndpoint],
        ssh_user: str = DEFAULT_SSH_USER,
        ssh_binary: str = "ssh",
    ) -> None:
        self._identity_file = identity_file
        self._resolver = endpoint_resolver
        self._user = ssh_user
        self._ssh = ssh_binary

    def _ssh_argv(self, endpoint: VsockEndpoint) -> list[str]:
        """Build the ssh argv up to (but not including) the remote command."""
        return [
            self._ssh,
            "-i",
            str(self._identity_file),
            "-o",
            f"ProxyCommand=socat - VSOCK-CONNECT:{endpoint.cid}:{endpoint.port}",
            # The transport is reachable only over our own vsock channel;
            # there is no opportunity for a host-key spoof to occur.
            "-o",
            "StrictHostKeyChecking=no",
            "-o",
            "UserKnownHostsFile=/dev/null",
            # Quiet ssh's "Warning: Permanently added …" noise so it
            # doesn't pollute caller-visible stderr.
            "-o",
            "LogLevel=ERROR",
            # Never prompt — the host either has the key or we fail fast.
            "-o",
            "BatchMode=yes",
            # The hostname after ``user@`` is a label only; ProxyCommand
            # does the actual connect.  Keep it short and recognisable
            # in any ssh diagnostic output.
            f"{self._user}@krun-guest",
        ]

    def exec(
        self,
        container: Container,
        cmd: list[str],
        *,
        timeout: float | None = None,
    ) -> ExecResult:
        """Run *cmd* in the guest and return its outcome."""
        endpoint = self._resolver(container)
        argv = [*self._ssh_argv(endpoint), "--", *cmd]
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
        guest's ``AcceptEnv`` whitelist.
        """
        endpoint = self._resolver(container)
        remote_cmd: list[str] = []
        if env:
            remote_cmd += ["env", *(f"{k}={v}" for k, v in env.items())]
        remote_cmd += cmd
        argv = [*self._ssh_argv(endpoint), "--", *remote_cmd]

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


# ── Endpoint resolvers ────────────────────────────────────────────────────


def podman_annotation_resolver(
    annotation_key: str = DEFAULT_CID_ANNOTATION,
    *,
    port: int = DEFAULT_VSOCK_SSHD_PORT,
) -> Callable[[Container], VsockEndpoint]:
    """Return a resolver that reads the CID from a podman annotation.

    The orchestrator sets ``--annotation <annotation_key>=<cid>`` when
    launching the task (via ``RunSpec.annotations``); this resolver
    reads it back at exec time via ``podman inspect``.  Decouples
    transport from the allocator: whatever allocates a free CID per
    task just needs to write it into the agreed annotation.
    """

    def _resolve(container: Container) -> VsockEndpoint:
        argv = [
            "podman",
            "inspect",
            "--format",
            '{{ index .Config.Annotations "' + annotation_key + '" }}',
            container.name,
        ]
        try:
            out = subprocess.check_output(  # nosec B603 B607 — argv built from fixed verbs + caller-controlled scope/container names — binary PATH lookup is the cross-distro contract
                argv,
                text=True,
            ).strip()
        except subprocess.CalledProcessError as exc:
            raise RuntimeError(
                f"podman inspect failed for container {container.name!r}: {exc}"
            ) from exc
        if not out:
            raise RuntimeError(
                f"container {container.name!r} has no {annotation_key!r} annotation — "
                "the orchestrator must allocate and set a vsock CID at launch time"
            )
        try:
            cid = int(out)
        except ValueError as exc:
            raise RuntimeError(
                f"container {container.name!r} has non-integer {annotation_key} annotation: {out!r}"
            ) from exc
        return VsockEndpoint(cid=cid, port=port)

    return _resolve
