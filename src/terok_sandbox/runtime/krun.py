# SPDX-FileCopyrightText: 2026 Jiri Vyskocil
# SPDX-License-Identifier: Apache-2.0

"""Krun backend for `.protocol.ContainerRuntime` — microVM isolation peer of
[`PodmanRuntime`][terok_sandbox.runtime.podman.PodmanRuntime].

Launches unmodified OCI images inside KVM microVMs via ``podman --runtime
krun``.  Container lifecycle (state, logs, force-remove, image inspection)
delegates to a held [`PodmanRuntime`][terok_sandbox.runtime.podman.PodmanRuntime]
because ``podman --runtime krun`` honours every other verb that doesn't
need to reach *into* the running guest.

``exec`` is the one verb that diverges and the reason this module exists:
``podman exec`` can't enter krun microVMs (libkrun can't inject processes
post-boot — see crun#1098), so ``exec`` routes through a pluggable
[`KrunTransport`][terok_sandbox.runtime.krun.KrunTransport] instead.  The
real transport is OpenSSH over AF_VSOCK (Phase 3 step 5); a
[`FakeKrunTransport`][terok_sandbox.runtime.krun.FakeKrunTransport] ships
here so the skeleton and its callers are unit-testable without standing
up a real guest.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, BinaryIO, Protocol, runtime_checkable

from .podman import PodmanRuntime
from .protocol import (
    Container,
    ContainerRemoveResult,
    ExecResult,
    Image,
    PortReservation,
)

if TYPE_CHECKING:
    from collections.abc import Mapping


# ── Transport seam ────────────────────────────────────────────────────────


@runtime_checkable
class KrunTransport(Protocol):
    """How [`KrunRuntime`][terok_sandbox.runtime.krun.KrunRuntime] reaches into a
    running microVM to run commands.

    The exec divergence is forced by libkrun: a microVM is sealed after
    boot and cannot accept injected processes.  The real implementation
    speaks SSH over AF_VSOCK to a socket-activated sshd inside the guest;
    that is wire-protocol shaped, not in-tree code we want to invent.

    Kept narrow on purpose — only the two operations
    [`ContainerRuntime`][terok_sandbox.runtime.protocol.ContainerRuntime]
    needs to route through it.  Lifecycle stays on the podman side.
    """

    def exec(
        self,
        container: Container,
        cmd: list[str],
        *,
        timeout: float | None = None,
    ) -> ExecResult:
        """Run *cmd* inside the guest backed by *container*; return its outcome."""
        ...

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
        """Bridge byte streams to *cmd* inside the guest; return its exit code."""
        ...


# ── Fake transport for unit tests ─────────────────────────────────────────


class FakeKrunTransport:
    """In-memory [`KrunTransport`][terok_sandbox.runtime.krun.KrunTransport] for tests.

    Mirrors [`NullRuntime`][terok_sandbox.runtime.null.NullRuntime]'s
    pre-register-then-replay shape so tests that already understand the
    null backend pick this up by analogy.  Records every call so tests
    can assert dispatch without a real vsock listener.
    """

    def __init__(self) -> None:
        self._results: dict[tuple[str, tuple[str, ...]], ExecResult] = {}
        self.exec_calls: list[tuple[str, tuple[str, ...]]] = []
        self.exec_stdio_calls: list[tuple[str, tuple[str, ...], dict[str, str]]] = []

    def set_result(
        self,
        container_name: str,
        cmd: tuple[str, ...],
        result: ExecResult,
    ) -> None:
        """Pre-register the result [`exec`][terok_sandbox.runtime.krun.FakeKrunTransport.exec]
        returns for exact *cmd* on *container_name*.
        """
        self._results[(container_name, cmd)] = result

    def exec(
        self,
        container: Container,
        cmd: list[str],
        *,
        timeout: float | None = None,
    ) -> ExecResult:
        """Return a pre-registered result, or empty success."""
        key = (container.name, tuple(cmd))
        self.exec_calls.append(key)
        return self._results.get(key, ExecResult(exit_code=0, stdout="", stderr=""))

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
        """Record the call and return exit code 0 (no I/O is moved)."""
        self.exec_stdio_calls.append((container.name, tuple(cmd), dict(env or {})))
        return 0


# ── Runtime ───────────────────────────────────────────────────────────────


class KrunRuntime:
    """Container runtime that launches tasks inside KVM microVMs.

    Composition, not inheritance: holds a
    [`PodmanRuntime`][terok_sandbox.runtime.podman.PodmanRuntime] for every
    lifecycle verb (``podman --runtime krun`` is just podman driving a
    different OCI runtime) and a
    [`KrunTransport`][terok_sandbox.runtime.krun.KrunTransport] for the
    one verb that can't go through podman — ``exec``.

    The transport is **required**: there is no sensible default beyond a
    real SSH-over-vsock implementation, and the fake exists explicitly
    for tests.  Production callers wire the real transport at the
    [`ContainerRuntime`][terok_sandbox.ContainerRuntime] selection point
    in the orchestrator.
    """

    def __init__(
        self,
        *,
        transport: KrunTransport,
        podman: PodmanRuntime | None = None,
    ) -> None:
        self._podman = podman or PodmanRuntime()
        self._transport = transport

    @property
    def transport(self) -> KrunTransport:
        """Return the transport used for [`exec`][terok_sandbox.runtime.krun.KrunRuntime.exec]."""
        return self._transport

    # -- Handle factories (delegated to podman) ----------------------------

    def container(self, name: str) -> Container:
        """Return a [`PodmanContainer`][terok_sandbox.runtime.podman.PodmanContainer] handle."""
        return self._podman.container(name)

    def containers_with_prefix(self, prefix: str) -> list[Container]:
        """Delegate prefix lookup to podman."""
        return self._podman.containers_with_prefix(prefix)

    def image(self, ref: str) -> Image:
        """Delegate image-handle construction to podman."""
        return self._podman.image(ref)

    def images(self, *, dangling_only: bool = False) -> list[Image]:
        """Delegate image enumeration to podman."""
        return self._podman.images(dangling_only=dangling_only)

    # -- Exec (transport-routed — the only divergence) ---------------------

    def exec(
        self,
        container: Container,
        cmd: list[str],
        *,
        timeout: float | None = None,
    ) -> ExecResult:
        """Route to the transport — typically SSH-over-vsock."""
        if not cmd:
            raise ValueError("exec argv must not be empty")
        return self._transport.exec(container, cmd, timeout=timeout)

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
        """Route stdio-bridged exec to the transport."""
        if not cmd:
            raise ValueError("exec_stdio argv must not be empty")
        return self._transport.exec_stdio(
            container,
            cmd,
            stdin=stdin,
            stdout=stdout,
            stderr=stderr,
            env=env,
            timeout=timeout,
        )

    # -- Operations without a single-object receiver (delegated) -----------

    def force_remove(self, containers: list[Container]) -> list[ContainerRemoveResult]:
        """Delegate forcible removal to podman."""
        return self._podman.force_remove(containers)

    def reserve_port(self, host: str = "127.0.0.1") -> PortReservation:
        """Delegate port reservation to podman."""
        return self._podman.reserve_port(host)
