# SPDX-FileCopyrightText: 2026 Jiri Vyskocil
# SPDX-License-Identifier: Apache-2.0

"""Null backend for :class:`.protocol.ContainerRuntime`.

Runs no subprocesses — useful for tests and dry-run modes.  Every handle
operation returns a safe default (``None``, ``False``, ``[]``, empty
strings).  :meth:`NullRuntime.exec` returns an empty success.

State fixtures can be injected via :meth:`NullRuntime.set_container_state`
and :meth:`NullRuntime.add_image` when tests need a specific shape.
"""

from __future__ import annotations

import socket
from collections.abc import Callable, Iterator
from pathlib import Path  # noqa: TC003 — used in a Protocol argument type

from .protocol import (
    Container,
    ContainerRemoveResult,
    ExecResult,
    Image,
    LogStream,
    PortReservation,
)


class NullLogStream:
    """Empty log stream — yields no lines and closes cleanly."""

    def __iter__(self) -> Iterator[str]:
        """Return self — iteration yields nothing."""
        return self

    def __next__(self) -> str:
        """Terminate iteration immediately."""
        raise StopIteration

    def __enter__(self) -> NullLogStream:
        """Enter no-op context."""
        return self

    def __exit__(self, *exc: object) -> None:
        """Exit no-op context."""
        return None

    def close(self) -> None:
        """Nothing to close."""
        return None


class NullPortReservation:
    """Real port reservation (binds a socket) — useful even in null mode
    because callers usually need an actually-free port to hand off."""

    def __init__(self, host: str = "127.0.0.1") -> None:
        s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        try:
            s.bind((host, 0))
        except BaseException:
            s.close()
            raise
        self._socket: socket.socket | None = s
        self.port = s.getsockname()[1]

    def __enter__(self) -> NullPortReservation:
        """Enter the context."""
        return self

    def __exit__(self, *exc: object) -> None:
        """Release the port."""
        self.close()

    def close(self) -> None:
        """Release the port (idempotent)."""
        if self._socket is not None:
            self._socket.close()
            self._socket = None


class NullContainer:
    """Container handle that reads from a :class:`NullRuntime`'s fixtures."""

    def __init__(self, name: str, *, runtime: NullRuntime) -> None:
        self.name = name
        self._runtime = runtime

    def __repr__(self) -> str:
        """Render as ``NullContainer(name='...')``."""
        return f"NullContainer(name={self.name!r})"

    def __eq__(self, other: object) -> bool:
        """Equality by name."""
        return isinstance(other, NullContainer) and self.name == other.name

    def __hash__(self) -> int:
        """Hash by name."""
        return hash(("NullContainer", self.name))

    @property
    def state(self) -> str | None:
        """Return the fixture state, or ``None`` when unset."""
        return self._runtime._container_states.get(self.name)

    @property
    def running(self) -> bool:
        """Return ``True`` when the fixture state is ``"running"``."""
        return self.state == "running"

    @property
    def image(self) -> Image | None:
        """Return the fixture image, or ``None`` when unset."""
        ref = self._runtime._container_images.get(self.name)
        return self._runtime.image(ref) if ref else None

    @property
    def rw_size(self) -> int | None:
        """Return the fixture rw_size, or ``None`` when unset."""
        return self._runtime._container_rw_sizes.get(self.name)

    def start(self) -> None:
        """Flip the fixture state to ``"running"``."""
        self._runtime._container_states[self.name] = "running"

    def stop(self, *, timeout: int = 10) -> None:
        """Flip the fixture state to ``"exited"``."""
        self._runtime._container_states[self.name] = "exited"

    def wait(self, timeout: float | None = None) -> int:
        """Return the fixture exit code (default 0)."""
        return self._runtime._container_exit_codes.get(self.name, 0)

    def copy_in(self, src: Path, dest: str) -> None:
        """Record the copy_in call for test inspection."""
        self._runtime._copy_in_calls.append((self.name, src, dest))

    def login_command(self, *, command: tuple[str, ...] = ()) -> list[str]:
        """Return a placeholder login argv."""
        return ["null-exec", self.name, *command]

    def logs(self, *, follow: bool = False, tail: int | None = None) -> LogStream:
        """Return an empty :class:`NullLogStream`."""
        return NullLogStream()

    def stream_initial_logs(
        self,
        ready_check: Callable[[str], bool],
        timeout_sec: float | None,
    ) -> bool:
        """Return the fixture readiness result (default ``True``)."""
        return self._runtime._ready_results.get(self.name, True)


class NullImage:
    """Image handle backed by :class:`NullRuntime` fixtures."""

    def __init__(self, ref: str, *, runtime: NullRuntime) -> None:
        self.ref = ref
        self._runtime = runtime

    def __repr__(self) -> str:
        """Render as ``NullImage(ref='...')``."""
        return f"NullImage(ref={self.ref!r})"

    def __eq__(self, other: object) -> bool:
        """Equality by ref."""
        return isinstance(other, NullImage) and self.ref == other.ref

    def __hash__(self) -> int:
        """Hash by ref."""
        return hash(("NullImage", self.ref))

    @property
    def id(self) -> str | None:
        """Return the ref itself when the image exists, else ``None``."""
        return self.ref if self.exists() else None

    @property
    def repository(self) -> str:
        """Return the fixture repository, or ``""``."""
        return self._runtime._image_records.get(self.ref, {}).get("repository", "")

    @property
    def tag(self) -> str:
        """Return the fixture tag, or ``""``."""
        return self._runtime._image_records.get(self.ref, {}).get("tag", "")

    @property
    def size(self) -> str:
        """Return the fixture size, or ``""``."""
        return self._runtime._image_records.get(self.ref, {}).get("size", "")

    @property
    def created(self) -> str:
        """Return the fixture created timestamp, or ``""``."""
        return self._runtime._image_records.get(self.ref, {}).get("created", "")

    def exists(self) -> bool:
        """Return ``True`` when the fixture lists this ref."""
        return self.ref in self._runtime._image_records

    def labels(self) -> dict[str, str]:
        """Return the fixture labels, or ``{}``."""
        return dict(self._runtime._image_labels.get(self.ref, {}))

    def history(self) -> list[str]:
        """Return the fixture history, or ``[]``."""
        return list(self._runtime._image_history.get(self.ref, ()))

    def remove(self) -> bool:
        """Remove from the fixture; return ``True`` when actually removed."""
        had = self.ref in self._runtime._image_records
        self._runtime._image_records.pop(self.ref, None)
        self._runtime._image_labels.pop(self.ref, None)
        self._runtime._image_history.pop(self.ref, None)
        return had


class NullRuntime:
    """Stub :class:`ContainerRuntime` for tests and dry-run modes.

    All state lives in dictionaries on the runtime instance.  Tests
    pre-populate fixtures via the :meth:`set_container_state`,
    :meth:`add_image`, etc. helpers.
    """

    def __init__(self) -> None:
        self._container_states: dict[str, str] = {}
        self._container_images: dict[str, str] = {}
        self._container_rw_sizes: dict[str, int] = {}
        self._container_exit_codes: dict[str, int] = {}
        self._ready_results: dict[str, bool] = {}
        self._image_records: dict[str, dict[str, str]] = {}
        self._image_labels: dict[str, dict[str, str]] = {}
        self._image_history: dict[str, tuple[str, ...]] = {}
        self._exec_results: dict[tuple[str, tuple[str, ...]], ExecResult] = {}
        self._copy_in_calls: list[tuple[str, Path, str]] = []
        self._force_remove_calls: list[list[str]] = []

    # -- Fixture setters ----------------------------------------------------

    def set_container_state(self, name: str, state: str) -> None:
        """Record *state* (``"running"``, ``"exited"``, ...) for container *name*."""
        self._container_states[name] = state

    def set_container_image(self, name: str, image_ref: str) -> None:
        """Record the image ref behind container *name*."""
        self._container_images[name] = image_ref

    def set_container_rw_size(self, name: str, bytes_: int) -> None:
        """Record the writable-layer size of container *name*."""
        self._container_rw_sizes[name] = bytes_

    def set_exit_code(self, name: str, code: int) -> None:
        """Record the exit code :meth:`Container.wait` will return for *name*."""
        self._container_exit_codes[name] = code

    def set_ready_result(self, name: str, ready: bool) -> None:
        """Record the outcome :meth:`Container.stream_initial_logs` returns."""
        self._ready_results[name] = ready

    def add_image(
        self,
        ref: str,
        *,
        repository: str = "",
        tag: str = "",
        size: str = "",
        created: str = "",
        labels: dict[str, str] | None = None,
        history: tuple[str, ...] = (),
    ) -> None:
        """Register an image fixture."""
        self._image_records[ref] = {
            "repository": repository,
            "tag": tag,
            "size": size,
            "created": created,
        }
        if labels:
            self._image_labels[ref] = dict(labels)
        if history:
            self._image_history[ref] = tuple(history)

    def set_exec_result(
        self,
        container_name: str,
        cmd: tuple[str, ...],
        result: ExecResult,
    ) -> None:
        """Pre-register the result :meth:`exec` returns for exact *cmd*."""
        self._exec_results[(container_name, cmd)] = result

    # -- Protocol surface ---------------------------------------------------

    def container(self, name: str) -> Container:
        """Return a :class:`NullContainer` handle."""
        return NullContainer(name, runtime=self)

    def containers_with_prefix(self, prefix: str) -> list[Container]:
        """Return fixtures whose name starts with ``prefix-``."""
        return [
            NullContainer(name, runtime=self)
            for name in self._container_states
            if name.startswith(f"{prefix}-")
        ]

    def image(self, ref: str) -> Image:
        """Return a :class:`NullImage` handle."""
        return NullImage(ref, runtime=self)

    def images(self, *, dangling_only: bool = False) -> list[Image]:
        """Return fixture images; *dangling_only* filters by ``tag == "<none>"``."""
        images: list[Image] = []
        for ref, rec in self._image_records.items():
            if dangling_only and rec.get("tag") != "<none>":
                continue
            images.append(NullImage(ref, runtime=self))
        return images

    def exec(
        self,
        container: Container,
        cmd: list[str],
        *,
        timeout: float | None = None,
    ) -> ExecResult:
        """Return a pre-registered result, or a default empty success."""
        key = (container.name, tuple(cmd))
        return self._exec_results.get(key, ExecResult(exit_code=0, stdout="", stderr=""))

    def force_remove(self, containers: list[Container]) -> list[ContainerRemoveResult]:
        """Record the call and clear every fixture for each container."""
        names = [c.name for c in containers]
        self._force_remove_calls.append(names)
        for name in names:
            self._container_states.pop(name, None)
            self._container_images.pop(name, None)
            self._container_rw_sizes.pop(name, None)
            self._container_exit_codes.pop(name, None)
            self._ready_results.pop(name, None)
            # Drop any pre-registered exec results keyed by this container name
            self._exec_results = {
                key: result for key, result in self._exec_results.items() if key[0] != name
            }
        return [ContainerRemoveResult(name=n, removed=True) for n in names]

    def reserve_port(self, host: str = "127.0.0.1") -> PortReservation:
        """Reserve a real host port (even null backend callers want a live port)."""
        return NullPortReservation(host)
