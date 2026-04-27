# SPDX-FileCopyrightText: 2026 Jiri Vyskocil
# SPDX-License-Identifier: Apache-2.0

"""Container runtime protocol — the *how* behind running and observing containers.

Defines the backend-neutral surface used by higher layers (executor, terok).
Concrete implementations: `.podman.PodmanRuntime` (default),
`.null.NullRuntime` (tests / dry-run).  A future ``KrunRuntime`` for
microVM-isolated containers slots in alongside without touching callers.

The protocol deliberately covers only *runtime* concerns — state queries,
lifecycle, image inspection, exec.  Gate, shield, credentials, vault, and
SSH are orthogonal services that compose *with* the runtime at a higher
layer (see [`terok_sandbox.sandbox.Sandbox`][terok_sandbox.sandbox.Sandbox]).
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Protocol, runtime_checkable

if TYPE_CHECKING:
    from collections.abc import Callable


# ── Value types ───────────────────────────────────────────────────────────


@dataclass(frozen=True)
class ExecResult:
    """Outcome of [`ContainerRuntime.exec`][terok_sandbox.runtime.protocol.ContainerRuntime.exec].

    Backend-neutral so a future SSH-over-vsock krun backend can fill it from
    an SSH response without pretending to be a [`subprocess.CompletedProcess`][subprocess.CompletedProcess].
    """

    exit_code: int
    stdout: str
    stderr: str

    @property
    def ok(self) -> bool:
        """Convenience — ``True`` when the command exited with code 0."""
        return self.exit_code == 0


@dataclass(frozen=True)
class ContainerRemoveResult:
    """Per-container outcome from [`ContainerRuntime.force_remove`][terok_sandbox.runtime.protocol.ContainerRuntime.force_remove]."""

    name: str
    """Container name that was targeted."""

    removed: bool
    """Whether the container is confirmed absent (includes already-gone)."""

    error: str | None = None
    """Human-readable reason when *removed* is ``False``."""


# ── Handle protocols ──────────────────────────────────────────────────────


@runtime_checkable
class Container(Protocol):
    """Handle to a container managed by a [`ContainerRuntime`][terok_sandbox.runtime.protocol.ContainerRuntime].

    Handles are cheap — construction does not verify that the container
    exists.  Operations return sensible defaults (``None``, ``False``, ``[]``)
    when the underlying container is absent, matching podman's own semantics.
    """

    name: str

    @property
    def state(self) -> str | None:
        """Lifecycle state (``"running"``, ``"exited"``, ...) or ``None``."""
        ...

    @property
    def running(self) -> bool:
        """Shortcut: ``state == "running"``."""
        ...

    @property
    def image(self) -> Image | None:
        """Handle to the image this container was created from, or ``None``."""
        ...

    @property
    def rw_size(self) -> int | None:
        """Writable-layer size in bytes, or ``None`` if unavailable."""
        ...

    def start(self) -> None:
        """Start the container.  Raises [`RuntimeError`][RuntimeError] on failure."""
        ...

    def stop(self, *, timeout: int = 10) -> None:
        """Stop the container, SIGKILL after *timeout* seconds."""
        ...

    def wait(self, timeout: float | None = None) -> int:
        """Block until the container exits; return its exit code.

        Raises [`TimeoutError`][TimeoutError] when *timeout* elapses.
        """
        ...

    def copy_in(self, src: Path, dest: str) -> None:
        """Copy a host path into the (stopped) container at *dest*."""
        ...

    def login_command(self, *, command: tuple[str, ...] = ()) -> list[str]:
        """Return an argv suitable for [`os.execvp`][os.execvp] to attach interactively.

        Empty *command* uses the backend default (typically ``tmux
        new-session -A -s main``).
        """
        ...

    def logs(self, *, follow: bool = False, tail: int | None = None) -> LogStream:
        """Return a context-managed iterator over decoded log lines."""
        ...

    def stream_initial_logs(
        self,
        ready_check: Callable[[str], bool],
        timeout_sec: float | None,
    ) -> bool:
        """Stream logs until *ready_check* returns ``True`` or *timeout_sec*.

        Returns ``True`` if the ready marker was seen, ``False`` on timeout.
        Each line is printed to stdout as it is received.
        """
        ...


@runtime_checkable
class Image(Protocol):
    """Handle to a local container image.  Cheap to construct."""

    ref: str
    """Tag (``"terok-l2-cli:abcd"``) or ID (``"sha256:..."``) used on lookup."""

    @property
    def id(self) -> str | None:
        """Resolved image ID, or ``None`` if the image is not present."""
        ...

    @property
    def repository(self) -> str:
        """Repository portion of the tag (``"<none>"`` for dangling)."""
        ...

    @property
    def tag(self) -> str:
        """Tag portion (``"<none>"`` for dangling)."""
        ...

    @property
    def size(self) -> str:
        """Podman-rendered human-readable size (``"1.2GB"``)."""
        ...

    @property
    def created(self) -> str:
        """Podman-rendered creation timestamp."""
        ...

    def exists(self) -> bool:
        """Return ``True`` if the image is present locally."""
        ...

    def labels(self) -> dict[str, str]:
        """Return the OCI ``Config.Labels`` as a flat string dict."""
        ...

    def history(self) -> list[str]:
        """Return the ``CreatedBy`` string of each layer, top to bottom."""
        ...

    def remove(self) -> bool:
        """Remove the image; return ``True`` on success."""
        ...


@runtime_checkable
class LogStream(Protocol):
    """Context-managed iterator over decoded log lines.

    ``__exit__`` releases the backing process (or the krun-backend
    equivalent).  Safe to use in a ``with`` block plus ``for line in
    stream`` loop.
    """

    def __iter__(self) -> LogStream: ...

    def __next__(self) -> str: ...

    def __enter__(self) -> LogStream: ...

    def __exit__(self, *exc: object) -> None: ...


@runtime_checkable
class PortReservation(Protocol):
    """Context manager for a reserved host-side TCP port.

    The port is held open for the lifetime of the reservation; closing
    releases it.  Use to pass a port number to an external process that
    will bind it shortly.
    """

    port: int
    """The reserved port number."""

    def __enter__(self) -> PortReservation: ...

    def __exit__(self, *exc: object) -> None: ...

    def close(self) -> None:
        """Release the port explicitly (same effect as ``__exit__``)."""
        ...


# ── Runtime protocol ──────────────────────────────────────────────────────


@runtime_checkable
class ContainerRuntime(Protocol):
    """The container runtime — factory for handles, plus operations that
    have no single-object receiver.

    One instance per process, typically constructed at the top-level entry
    point and threaded down through higher layers (``Sandbox``, executor's
    ``AgentRunner``, terok's CLI/TUI).
    """

    # -- Handle factories --------------------------------------------------

    def container(self, name: str) -> Container:
        """Return a handle to the container named *name*.

        Does not verify existence; call [`Container.state`][terok_sandbox.runtime.protocol.Container.state] for that.
        """
        ...

    def containers_with_prefix(self, prefix: str) -> list[Container]:
        """Return handles for every container whose name starts with *prefix*."""
        ...

    def image(self, ref: str) -> Image:
        """Return a handle to the image identified by tag or ID *ref*.

        Does not verify existence; call [`Image.exists`][terok_sandbox.runtime.protocol.Image.exists] for that.
        """
        ...

    def images(self, *, dangling_only: bool = False) -> list[Image]:
        """Enumerate local images.

        *dangling_only* narrows to untagged images (those listed as
        ``<none>:<none>``).
        """
        ...

    # -- Operations without a single-object receiver -----------------------

    def exec(
        self,
        container: Container,
        cmd: list[str],
        *,
        timeout: float | None = None,
    ) -> ExecResult:
        """Run *cmd* inside *container* and return its completion record.

        The operation that diverges most across backends: podman uses
        ``podman exec``; a krun backend would use SSH over vsock.
        """
        ...

    def force_remove(self, containers: list[Container]) -> list[ContainerRemoveResult]:
        """Forcibly stop and remove *containers*.

        Best-effort — continues through individual failures and returns
        one [`ContainerRemoveResult`][terok_sandbox.runtime.protocol.ContainerRemoveResult] per input.  An already-absent
        container counts as *removed* (the post-condition holds).
        """
        ...

    def reserve_port(self, host: str = "127.0.0.1") -> PortReservation:
        """Reserve a free TCP port on *host*.

        The returned [`PortReservation`][terok_sandbox.runtime.protocol.PortReservation] exposes the port number via
        ``reservation.port`` and releases the socket on close.  Use to
        pass a pre-reserved port to an external process.
        """
        ...
