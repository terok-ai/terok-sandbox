# SPDX-FileCopyrightText: 2026 Jiri Vyskocil
# SPDX-License-Identifier: Apache-2.0

"""Real OpenSSH-over-vsock transport for `KrunRuntime`.

Implements the [`KrunTransport`][terok_sandbox.runtime.krun.KrunTransport]
protocol by shelling out to the system ``ssh`` client with a ``socat``
``ProxyCommand`` that bridges to AF_VSOCK.  No custom wire protocol:
sshd handles auth, PTY allocation, signal forwarding, exit codes.

Design choices and why:

- **stock ssh CLI + socat** rather than a paramiko + ``AF_VSOCK``
  client.  Both binaries are battle-tested for the edge cases (PTY
  allocation, signal forwarding, EOF semantics) that we would
  otherwise reimplement.
- **Pubkey-only**, with ``IdentitiesOnly=yes`` so a stray host-side
  ssh-agent can't offer unrelated identities.  The host holds the
  private key; the guest holds only the public half (baked into
  ``authorized_keys.d/terok`` at image build).
- **Argv-quoted remote command**: ``ssh host -- a b c`` concatenates
  the tokens and runs the result through the in-guest user's shell,
  so the transport ``shlex.quote``s each token to preserve the
  ``cmd: list[str]`` argv contract on the wire.
- **No host-key persistence**: ``StrictHostKeyChecking=no`` plus
  ``UserKnownHostsFile=/dev/null``.  Vsock is host-local and the
  CID-to-guest binding is enforced by the orchestrator's allocator
  plus [`VsockEndpoint`][terok_sandbox.runtime.krun_transport.VsockEndpoint]'s
  range check, so wrong-endpoint connects are restricted to a host
  with podman access (root-equivalent).  Full per-guest host-key
  pinning would need orchestrator-side ``known_hosts`` plumbing and
  is tracked as a follow-up.

Endpoint discovery is pluggable via *endpoint_resolver* so unit tests
can synthesise endpoints without an actual microVM.  The default
production factory
[`podman_annotation_resolver`][terok_sandbox.runtime.krun_transport.podman_annotation_resolver]
reads the CID from a podman annotation set at task launch.
"""

from __future__ import annotations

import re
import shlex
import shutil
import subprocess  # nosec B404 — orchestrates the system ssh CLI
from collections.abc import Callable, Mapping
from dataclasses import dataclass
from functools import cache
from pathlib import Path  # noqa: TC003 — used in dataclass field type
from typing import BinaryIO

from .podman import _start_stdio_pumps
from .protocol import Container, ExecResult

# ── Public defaults ─────────────────────────────────────────────────────────

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


# ── Endpoint ────────────────────────────────────────────────────────────────


@dataclass(frozen=True)
class VsockEndpoint:
    """A vsock endpoint reachable from the host.

    *cid* is a libkrun-assigned context ID (32-bit integer); *port* is
    the vsock port the in-guest service is listening on.

    Both fields are int-coerced and range-checked in ``__post_init__``
    — the transport interpolates them into the ``ProxyCommand`` string
    that OpenSSH hands to a shell, so a string-shaped value carrying
    shell metacharacters would otherwise be a command-injection
    primitive.  Catching it here means a bad ``endpoint_resolver``
    fails loudly at construction rather than silently building a
    hostile ``socat`` invocation.
    """

    cid: int
    port: int = DEFAULT_VSOCK_SSHD_PORT

    def __post_init__(self) -> None:
        """Coerce + bound-check both fields so the ProxyCommand stays safe."""
        try:
            cid = int(self.cid)
            port = int(self.port)
        except (TypeError, ValueError) as exc:
            raise ValueError(
                f"VsockEndpoint: cid and port must be int-convertible, "
                f"got cid={self.cid!r}, port={self.port!r}"
            ) from exc
        if cid in _VSOCK_RESERVED_CIDS:
            raise ValueError(
                f"VsockEndpoint: cid {cid} is reserved by the vsock spec "
                "(0=ANY, 1=HYPERVISOR, 2=HOST) — refusing to connect"
            )
        if not 0 <= cid <= _VSOCK_MAX_CID:
            raise ValueError(f"VsockEndpoint: cid {cid} outside u32 range")
        if not 1 <= port <= _VSOCK_MAX_PORT:
            raise ValueError(f"VsockEndpoint: port {port} outside (0, u32] range")
        object.__setattr__(self, "cid", cid)
        object.__setattr__(self, "port", port)


# ── Transport (the entry point) ─────────────────────────────────────────────


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
        running under tmux or an IDE proxy), and falls back to a no-arg
        invocation of the in-guest user's login shell when *command* is
        empty.  Argv tokens past ``--`` are ``shlex.quote``d (same
        helper the exec paths use) so the SSH wire format preserves
        argv semantics across the login-shell parse on the far side.
        """
        endpoint = self._resolver(container)
        argv = self._ssh_argv(endpoint, interactive=True)
        if command:
            argv += ["--", _remote_command(list(command))]
        return argv

    def _ssh_argv(self, endpoint: VsockEndpoint, *, interactive: bool = False) -> list[str]:
        """Build the ssh argv up to (but not including) the remote command.

        *interactive* swaps ``BatchMode=yes`` for ``-tt`` (force-allocate
        PTY) so the login flow gets a real terminal.  Exec paths keep
        the batch-mode default so a missing identity fails fast instead
        of prompting.
        """
        pty_flags = ["-tt"] if interactive else ["-o", "BatchMode=yes"]
        return [
            self._ssh,
            "-i",
            str(self._identity_file),
            # Only the identity we explicitly passed is offered; a stray
            # ssh-agent running in the host environment can't slip in
            # additional keys that happen to be accepted by the guest.
            "-o",
            "IdentitiesOnly=yes",
            "-o",
            f"ProxyCommand={_socat_path()} - VSOCK-CONNECT:{endpoint.cid}:{endpoint.port}",
            # Vsock is host-local and the CID-to-guest binding is enforced
            # by the orchestrator's allocator + ``VsockEndpoint`` range
            # check, so a wrong-endpoint connect is structurally
            # restricted to a host with podman access (already root-
            # equivalent).  Full host-key pinning would need orchestrator-
            # side known_hosts plumbing and is tracked as a follow-up.
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
            # The hostname after ``user@`` is a label only; ProxyCommand
            # does the actual connect.  Keep it short and recognisable
            # in any ssh diagnostic output.
            f"{self._user}@krun-guest",
        ]


# ── Endpoint resolvers ──────────────────────────────────────────────────────


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

    *annotation_key* is validated against the OCI annotation charset
    ``[A-Za-z0-9][A-Za-z0-9._/-]*`` at construction time — the value
    is interpolated into a podman ``--format`` Go-template literal,
    so a ``"`` / ``}}`` in the key would break out of the string slot
    and let attacker-chosen template expressions execute against the
    container's full inspect output.
    """
    if not _ANNOTATION_KEY_RE.fullmatch(annotation_key):
        raise ValueError(
            f"podman_annotation_resolver: annotation_key {annotation_key!r} must "
            "match [A-Za-z0-9][A-Za-z0-9._/-]* (OCI annotation charset)"
        )

    def _resolve(container: Container) -> VsockEndpoint:
        # ``--`` ends podman's own option parsing, so a container handle
        # carrying a leading-dash name can't be reinterpreted as a flag.
        argv = [
            "podman",
            "inspect",
            "--format",
            '{{ index .Config.Annotations "' + annotation_key + '" }}',
            "--",
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
        # ``VsockEndpoint.__post_init__`` does the range + reserved-CID
        # check; raising ``RuntimeError`` here keeps the resolver's
        # exception type uniform across "annotation missing" and
        # "annotation invalid".
        try:
            return VsockEndpoint(cid=cid, port=port)
        except ValueError as exc:
            raise RuntimeError(
                f"container {container.name!r} has invalid {annotation_key} annotation: {exc}"
            ) from exc

    return _resolve


# ── Private helpers ─────────────────────────────────────────────────────────

_VSOCK_RESERVED_CIDS: frozenset[int] = frozenset({0, 1, 2})
"""Vsock CIDs reserved by the spec (``VMADDR_CID_ANY``, ``..._HYPERVISOR``,
``..._HOST``).  A guest is never one of these, and the transport must
not be coaxed into connecting to them."""

_VSOCK_MAX_CID = 0xFFFFFFFF  # u32
_VSOCK_MAX_PORT = 0xFFFFFFFF  # u32

# Charset shape check for ``podman_annotation_resolver``'s key
# parameter — the value is concatenated into a ``--format`` Go-template
# literal that podman parses, so any ``"`` / ``}`` / ``{`` would break
# out of the intended string slot and execute attacker-chosen template
# expressions.  Mirrors the OCI annotation charset.
#
# Note: this is the *reader-side* shape check (operator-supplied key
# name).  The *writer-side* allowlist of values terok actually emits
# lives at ``terok_sandbox.sandbox.SAFE_ANNOTATION_KEYS``.
_ANNOTATION_KEY_RE = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._/-]*$")

# ``ssh host -- arg1 arg2`` does NOT preserve argv on the remote side —
# sshd concatenates the tokens and runs the result through the user's
# login shell.  Our public API contract is ``cmd: list[str]`` (argv
# semantics), so every remote token gets ``shlex.quote``d before going
# over the wire to keep that contract honest.  Env var *names* are
# validated against this strict pattern because there is no portable
# way to quote them — the remote ``env`` command expects bare
# identifiers.
_REMOTE_ENV_NAME_RE = re.compile(r"^[A-Za-z_][A-Za-z0-9_]*$")


@cache
def _socat_path() -> str:
    """Resolve ``socat`` to an absolute path once per process.

    The ``ProxyCommand`` ssh hands to a shell would otherwise resolve
    ``socat`` via ``PATH`` at every connection — a hostile ``PATH``
    (sudo wrapper, compromised service env, caller-modified env) could
    substitute an attacker binary.  Resolution happens lazily on first
    call (when ``_ssh_argv`` runs); the cache then pins the binary
    against a later environment poke for the rest of the process.

    Falls back to ``/usr/bin/socat`` so the failure mode is "vsock
    connect fails" rather than "silent PATH search" when
    ``shutil.which`` can't see it.
    """
    return shutil.which("socat") or "/usr/bin/socat"


def _remote_command(cmd: list[str], *, env: Mapping[str, str] | None = None) -> str:
    """Render *cmd* (and optional *env*) into a shell-safe remote string.

    OpenSSH's ``ssh host -- a b c`` concatenates the post-``--`` tokens
    and runs the result through the in-guest user's login shell, so the
    transport must do the quoting itself to honour the ``cmd: list[str]``
    argv contract on the wire.

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
    return " ".join(shlex.quote(t) for t in tokens)
