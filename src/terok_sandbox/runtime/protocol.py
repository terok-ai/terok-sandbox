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
from typing import TYPE_CHECKING, BinaryIO, Protocol, runtime_checkable

if TYPE_CHECKING:
    from collections.abc import Callable, Mapping


# ── Value types ───────────────────────────────────────────────────────────


class RuntimeInspectionError(RuntimeError):
    """Required runtime metadata could not be obtained reliably."""


@dataclass(frozen=True)
class ContainerInspection:
    """One successful observation of a container, including its creation-time inputs."""

    id: str
    state: str
    image: Image | None
    """Originating image pinned by immutable ID, or ``None`` for a rootfs-only container."""
    env: dict[str, str]
    mounts: tuple[tuple[str, str], ...]
    """Reported bind/volume mounts as ``(host source, container destination)`` pairs."""


@dataclass(frozen=True)
class ImageInspection:
    """One successful observation of an image's identity and OCI labels."""

    id: str
    labels: dict[str, str]


EXEC_OUTPUT_LIMIT = 16 * 1024 * 1024
"""Default byte budget for the combined stdout and stderr of
[`ContainerRuntime.exec`][terok_sandbox.runtime.protocol.ContainerRuntime.exec].
Generous for any reply a caller evaluates; small enough that a flood from
inside a container cannot grow host memory without bound."""


class ExecOutputLimitError(RuntimeError):
    """An exec reply exceeded its byte budget; no partial output is returned."""


@dataclass(frozen=True)
class ExecResult:
    """Outcome of [`ContainerRuntime.exec`][terok_sandbox.runtime.protocol.ContainerRuntime.exec].

    Backend-neutral so the SSH-over-passt krun backend can fill it from
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
class RuntimeContainer:
    """A container's immutable identity and host-recorded containment metadata."""

    id: str
    name: str
    state: str
    annotations: dict[str, str]


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

    def inspect(self) -> ContainerInspection | None:
        """Read a coherent snapshot, returning ``None`` only for confirmed absence.

        Unlike best-effort display properties, failures and malformed metadata
        raise [`RuntimeInspectionError`][terok_sandbox.RuntimeInspectionError].
        A snapshot records one observation, not a lease on the container's lifetime.
        """
        ...

    @property
    def state(self) -> str | None:
        """Lifecycle state (``"running"``, ``"exited"``, ...) or ``None``."""
        ...

    @property
    def running(self) -> bool:
        """Shortcut: ``state == "running"``."""
        ...

    @property
    def started_at(self) -> float | None:
        """Unix timestamp of the container's last start, or ``None``.

        Feeds "did this container start before event X?" health checks
        — e.g. flagging supervisors that predate a vault passphrase
        change and therefore still hold the old passphrase.
        """
        ...

    @property
    def image(self) -> Image | None:
        """Handle to the image this container was created from, or ``None``."""
        ...

    @property
    def rw_size(self) -> int | None:
        """Writable-layer size in bytes, or ``None`` if unavailable."""
        ...

    @property
    def id(self) -> str | None:
        """Full container ID (podman ``.Id``), or ``None`` when the container is absent."""
        ...

    @property
    def mounts(self) -> list[tuple[str, str]]:
        """Bind/volume mounts as ``(host_source, container_destination)`` pairs.

        Empty when the container is absent or has no mounts.
        """
        ...

    @property
    def env(self) -> dict[str, str]:
        """Environment recorded on the container when it was created.

        Frozen at creation, so it can still name a host-side contract a newer
        host no longer honours.  Empty when the container is absent.
        """
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

    def inspect(self) -> ImageInspection | None:
        """Read identity and labels; return ``None`` only for confirmed absence.

        Other failures raise [`RuntimeInspectionError`][terok_sandbox.RuntimeInspectionError],
        never an empty label map that could be mistaken for an obsolete image.
        """
        ...

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

    def inventory(self, *, timeout: float = 10) -> list[RuntimeContainer]:
        """Enumerate all containers with host annotations; raise on incomplete discovery."""
        ...

    def force_stop(self, container_id: str, *, timeout: float = 10) -> None:
        """Stop an exact container ID immediately; raise on failure or deadline expiry."""
        ...

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
        max_output_bytes: int = EXEC_OUTPUT_LIMIT,
    ) -> ExecResult:
        """Run *cmd* inside *container* and return its completion record.

        The operation that diverges most across backends: podman uses
        ``podman exec``; the krun backend uses SSH over a passt-forwarded
        TCP port.

        Combined stdout and stderr are captured within *max_output_bytes*;
        overflow raises
        [`ExecOutputLimitError`][terok_sandbox.runtime.protocol.ExecOutputLimitError]
        instead of returning partial output.  The command reads EOF on
        stdin, and its output decodes as UTF-8 with replacement for
        invalid bytes.
        """
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
        """Run *cmd* inside *container* with stdio bridged to caller-supplied streams.

        Forwards bytes bidirectionally between *stdin*/*stdout*/*stderr* and the
        spawned process — distinct from [`exec`][terok_sandbox.runtime.null.NullRuntime.exec], which captures output into
        an [`ExecResult`][terok_sandbox.runtime.protocol.ExecResult].  Used by the host-side ACP proxy to bridge a Unix
        socket to an in-container ACP-stdio agent without the runtime ever
        materialising the conversation.

        Blocks until the child exits; returns the exit code.  EOF on either
        side terminates forwarding cleanly.  Implementations are expected to
        be transport-agnostic — *stdin*/*stdout* are arbitrary byte streams
        (a socket's file-object face, a pipe end, a test buffer).
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
