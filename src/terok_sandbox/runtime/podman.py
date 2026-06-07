# SPDX-FileCopyrightText: 2026 Jiri Vyskocil
# SPDX-License-Identifier: Apache-2.0

"""Podman backend for the `.protocol` container runtime.

This is the concrete default runtime.  Every ``subprocess`` call that
ends in a ``podman`` invocation lives in this module — other layers
speak only through the protocol.

The public export is [`PodmanRuntime`][terok_sandbox.runtime.podman.PodmanRuntime] plus the argv helpers that
[`terok_sandbox.sandbox.Sandbox`][terok_sandbox.sandbox.Sandbox] uses to assemble ``podman run``
commands (which remain podman-specific for now; a krun backend will
replace ``Sandbox.run`` when Phase 3 lands).
"""

from __future__ import annotations

import contextlib
import json
import os
import re
import select
import socket
import subprocess  # nosec B404 — podman CLI orchestration
import sys
import threading
import time
from collections.abc import Callable, Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import BinaryIO

from .._util import log_debug, log_warning
from .protocol import (
    Container,
    ContainerRemoveResult,
    ExecResult,
    Image,
    LogStream,
    PortReservation,
)

_CDI_HINT = (
    "Hint: NVIDIA CDI configuration appears to be missing or broken.\n"
    "Ensure the NVIDIA Container Toolkit is installed and CDI is configured.\n"
    "See: https://podman-desktop.io/docs/podman/gpu"
)

_NVIDIA_GPU_KIND = "nvidia.com/gpu"
_CDI_ERROR_PATTERNS = ("cdi.k8s.io", _NVIDIA_GPU_KIND, "CDI")


class GpuConfigError(RuntimeError):
    """CDI/NVIDIA misconfiguration detected during container launch."""

    def __init__(self, message: str, *, hint: str = _CDI_HINT) -> None:
        """Store the CDI *hint* alongside the standard error *message*."""
        self.hint = hint
        super().__init__(message)


def check_gpu_error(exc: subprocess.CalledProcessError) -> None:
    """Raise [`GpuConfigError`][terok_sandbox.runtime.podman.GpuConfigError] if *exc* looks like a CDI/NVIDIA issue.

    Does nothing if the error does not match any known CDI patterns.
    Defensively handles both ``bytes`` and ``str`` stderr so callers
    that ran subprocess with ``text=True`` are not punished with an
    ``AttributeError`` on ``.decode``.
    """
    stderr_raw = exc.stderr or b""
    if isinstance(stderr_raw, bytes):
        stderr = stderr_raw.decode(errors="replace")
    else:
        stderr = str(stderr_raw)
    if any(pat in stderr for pat in _CDI_ERROR_PATTERNS):
        msg = f"Container launch failed (GPU misconfiguration):\n{stderr.strip()}\n\n{_CDI_HINT}"
        raise GpuConfigError(msg) from exc


_CDI_DEFAULT_DIRS: tuple[Path, ...] = (Path("/etc/cdi"), Path("/var/run/cdi"))


def _cdi_spec_paths() -> list[Path]:
    """Return CDI spec file paths podman is configured to scan.

    Asks podman first (so a custom ``cdi_spec_dirs`` in
    ``containers.conf`` is honoured) and falls back to the documented
    default directories when podman is absent or its info template
    doesn't expose ``Host.CDISpecs`` (older versions).
    """
    try:
        proc = subprocess.run(  # nosec B603 B607 — podman CLI invocation matches existing pattern in this module
            ["podman", "info", "--format", "{{json .Host.CDISpecs}}"],
            capture_output=True,
            text=True,
            timeout=5,
            check=True,
        )
        listed = json.loads(proc.stdout or "null") or []
        if listed:
            return [Path(p) for p in listed]
    except (
        FileNotFoundError,
        subprocess.CalledProcessError,
        subprocess.TimeoutExpired,
        json.JSONDecodeError,
    ):
        pass
    return [f for d in _CDI_DEFAULT_DIRS if d.is_dir() for f in d.iterdir() if f.is_file()]


def check_gpu_available() -> bool:
    """Return ``True`` when a CDI spec declares the ``nvidia.com/gpu`` kind.

    Wizards call this to decide whether to offer the NVIDIA base image;
    the on-launch [`check_gpu_error`][terok_sandbox.runtime.podman.check_gpu_error]
    path is the authoritative one and stays in place.  Any failure
    (missing podman, missing CDI dirs, unreadable spec) collapses to
    ``False`` so callers can treat this as a pure yes/no signal.
    """
    return any(_NVIDIA_GPU_KIND in _safe_read(p) for p in _cdi_spec_paths())


def _safe_read(path: Path) -> str:
    try:
        return path.read_text(errors="replace")
    except OSError:
        return ""


_SENSITIVE_KEY_RE = re.compile(r"(?i)(KEY|TOKEN|SECRET|API|PASSWORD|PRIVATE)")
_ALWAYS_REDACT_KEYS = frozenset({"CODE_REPO", "CLONE_FROM"})


def redact_env_args(cmd: list[str]) -> list[str]:
    """Return a copy of *cmd* with sensitive ``-e KEY=VALUE`` args redacted.

    Handles the two-arg form only (``-e KEY=VALUE``).  Callers passing
    sensitive values via single-arg forms (``--env=...``) must pre-redact.
    """
    out: list[str] = []
    redact_next = False
    for arg in cmd:
        if redact_next:
            key, _, _val = arg.partition("=")
            if _SENSITIVE_KEY_RE.search(key) or key in _ALWAYS_REDACT_KEYS:
                out.append(f"{key}=<redacted>")
            else:
                out.append(arg)
            redact_next = False
        elif arg == "-e":
            out.append(arg)
            redact_next = True
        else:
            out.append(arg)
    return out


def gpu_run_args(*, enabled: bool = False) -> list[str]:
    """Return ``podman run`` args for NVIDIA GPU passthrough."""
    if not enabled:
        return []
    return [
        "--device",
        f"{_NVIDIA_GPU_KIND}=all",
        "-e",
        "NVIDIA_VISIBLE_DEVICES=all",
        "-e",
        "NVIDIA_DRIVER_CAPABILITIES=all",
    ]


_SLIRP_GATEWAY = "10.0.2.2"
_PASTA_HOST_LOOPBACK_MAP = "169.254.1.2"


def _detect_rootless_network_mode() -> str:
    """Return ``"pasta"`` or ``"slirp4netns"``; falls back to the latter."""
    try:
        out = subprocess.run(  # nosec B603 B607 — argv built from fixed verbs + caller-controlled scope/container names — binary PATH lookup is the cross-distro contract
            ["podman", "info", "-f", "{{.Host.RootlessNetworkCmd}}"],
            capture_output=True,
            text=True,
            timeout=10,
        )
        if out.returncode != 0:
            log_debug(f"podman info failed (rc={out.returncode}), defaulting to slirp4netns")
            return "slirp4netns"
        cmd = out.stdout.strip()
        return cmd if cmd in ("pasta", "slirp4netns") else "slirp4netns"
    except (FileNotFoundError, subprocess.TimeoutExpired):
        return "slirp4netns"


def bypass_network_args(gate_port: int | None) -> list[str]:
    """Return podman network args for running without shield.

    Replicates shield's normal networking (reachable ``host.containers.internal``)
    without nftables rules.  **Dangerous fallback** — all egress is unfiltered.
    """
    if os.geteuid() == 0:
        return []
    if _detect_rootless_network_mode() == "slirp4netns":
        return [
            "--network",
            "slirp4netns:allow_host_loopback=true",
            "--add-host",
            f"host.containers.internal:{_SLIRP_GATEWAY}",
        ]
    return [
        "--network",
        f"pasta:--map-host-loopback,{_PASTA_HOST_LOOPBACK_MAP}",
        "--add-host",
        f"host.containers.internal:{_PASTA_HOST_LOOPBACK_MAP}",
    ]


# ── Timeouts / constants ──────────────────────────────────────────────────

_DEFAULT_LOGIN_COMMAND: tuple[str, ...] = ("tmux", "new-session", "-A", "-s", "main")
_START_TIMEOUT = 30
_STOP_TIMEOUT_BUFFER = 5
_CONTAINER_REMOVE_TIMEOUT = 120
_IMAGES_FORMAT = "{{.Repository}}\t{{.Tag}}\t{{.ID}}\t{{.Size}}\t{{.Created}}"


# ── Size parsing (used by PodmanContainer.rw_size batch path) ─────────────

_SIZE_RE = re.compile(r"([\d.]+)\s*([a-zA-Z]+)")

_SIZE_UNITS: dict[str, int] = {
    "B": 1,
    "KB": 1_000,
    "MB": 1_000_000,
    "GB": 1_000_000_000,
    "TB": 1_000_000_000_000,
    "KIB": 1 << 10,
    "MIB": 1 << 20,
    "GIB": 1 << 30,
    "TIB": 1 << 40,
}


def _parse_human_size(text: str) -> int | None:
    """Best-effort parse of podman's ``12.5MB (virtual 1.23GB)`` size strings."""
    m = _SIZE_RE.search(text)
    if not m:
        return None
    try:
        value = float(m.group(1))
    except ValueError:
        return None
    unit = m.group(2).upper()
    multiplier = _SIZE_UNITS.get(unit)
    if multiplier is None:
        return None
    return int(value * multiplier)


# ── Container handle ──────────────────────────────────────────────────────


class PodmanContainer:
    """Podman implementation of [`Container`][terok_sandbox.runtime.podman.Container].

    Cheap to construct — does not verify existence.  Each property /
    method does a fresh ``podman inspect`` or equivalent.
    """

    def __init__(self, name: str, *, runtime: PodmanRuntime) -> None:
        self.name = name
        self._runtime = runtime

    def __repr__(self) -> str:
        """Render as ``PodmanContainer(name='...')``."""
        return f"PodmanContainer(name={self.name!r})"

    def __eq__(self, other: object) -> bool:
        """Equality by name (handles are stateless identifiers)."""
        return isinstance(other, PodmanContainer) and self.name == other.name

    def __hash__(self) -> int:
        """Hash by name."""
        return hash(("PodmanContainer", self.name))

    @property
    def state(self) -> str | None:
        """Lifecycle state (``"running"``, ``"exited"``, ...) or ``None``."""
        try:
            out = subprocess.check_output(  # nosec B603 B607 — argv built from fixed verbs + caller-controlled scope/container names — binary PATH lookup is the cross-distro contract
                ["podman", "inspect", "-f", "{{.State.Status}}", self.name],
                stderr=subprocess.DEVNULL,
                text=True,
            ).strip()
            return out.lower() if out else None
        except (subprocess.CalledProcessError, FileNotFoundError):
            return None

    @property
    def running(self) -> bool:
        """Shortcut: ``state == "running"``."""
        try:
            out = subprocess.check_output(  # nosec B603 B607 — argv built from fixed verbs + caller-controlled scope/container names — binary PATH lookup is the cross-distro contract
                ["podman", "inspect", "-f", "{{.State.Running}}", self.name],
                stderr=subprocess.DEVNULL,
                text=True,
            ).strip()
        except (subprocess.CalledProcessError, FileNotFoundError):
            return False
        return out.lower() == "true"

    @property
    def image(self) -> Image | None:
        """Handle to the image this container was created from, or ``None``."""
        try:
            out = subprocess.check_output(  # nosec B603 B607 — argv built from fixed verbs + caller-controlled scope/container names — binary PATH lookup is the cross-distro contract
                ["podman", "inspect", "-f", "{{.Image}}", self.name],
                stderr=subprocess.DEVNULL,
                text=True,
            ).strip()
        except (subprocess.CalledProcessError, FileNotFoundError):
            return None
        return self._runtime.image(out) if out else None

    @property
    def rw_size(self) -> int | None:
        """Writable-layer size in bytes, or ``None`` if unavailable.

        Uses ``podman container inspect --size`` — expect a brief pause
        for large containers while overlay diffs are computed.
        """
        try:
            out = subprocess.check_output(  # nosec B603 B607 — argv built from fixed verbs + caller-controlled scope/container names — binary PATH lookup is the cross-distro contract
                [
                    "podman",
                    "container",
                    "inspect",
                    "--size",
                    "-f",
                    "{{.SizeRw}}",
                    self.name,
                ],
                stderr=subprocess.DEVNULL,
                text=True,
                timeout=60,
            ).strip()
            return int(out) if out else None
        except (
            subprocess.CalledProcessError,
            FileNotFoundError,
            subprocess.TimeoutExpired,
            ValueError,
        ):
            return None

    def start(self) -> None:
        """Start the container.

        Every lifecycle failure — missing podman, timeout, or non-zero
        exit — surfaces as [`RuntimeError`][RuntimeError] so callers have a
        single exception type to catch.  The original exception is
        preserved via ``__cause__`` when applicable.
        """
        log_debug(f"PodmanContainer.start({self.name})")
        try:
            proc = subprocess.run(  # nosec B603 B607 — argv built from fixed verbs + caller-controlled scope/container names — binary PATH lookup is the cross-distro contract
                ["podman", "start", self.name],
                check=False,
                stdout=subprocess.DEVNULL,
                stderr=subprocess.PIPE,
                text=True,
                timeout=_START_TIMEOUT,
            )
        except FileNotFoundError as exc:
            raise RuntimeError(f"podman start {self.name!r} failed: podman not found") from exc
        except subprocess.TimeoutExpired as exc:
            raise RuntimeError(
                f"podman start {self.name!r} timed out after {_START_TIMEOUT}s"
            ) from exc
        if proc.returncode != 0:
            raise RuntimeError(
                f"podman start {self.name!r} failed "
                f"(rc={proc.returncode}): {(proc.stderr or '').strip() or '<no output>'}"
            )

    def stop(self, *, timeout: int = 10) -> None:
        """Stop the container, SIGKILL after *timeout* seconds.

        Every lifecycle failure surfaces as [`RuntimeError`][RuntimeError]; see
        [`start`][terok_sandbox.runtime.podman.PodmanContainer.start] for the rationale.
        """
        log_debug(f"PodmanContainer.stop({self.name}, timeout={timeout})")
        try:
            proc = subprocess.run(  # nosec B603 B607 — argv built from fixed verbs + caller-controlled scope/container names — binary PATH lookup is the cross-distro contract
                ["podman", "stop", "--time", str(timeout), self.name],
                check=False,
                stdout=subprocess.DEVNULL,
                stderr=subprocess.PIPE,
                text=True,
                timeout=timeout + _STOP_TIMEOUT_BUFFER,
            )
        except FileNotFoundError as exc:
            raise RuntimeError(f"podman stop {self.name!r} failed: podman not found") from exc
        except subprocess.TimeoutExpired as exc:
            raise RuntimeError(
                f"podman stop {self.name!r} timed out after {timeout + _STOP_TIMEOUT_BUFFER}s"
            ) from exc
        if proc.returncode != 0:
            raise RuntimeError(
                f"podman stop {self.name!r} failed "
                f"(rc={proc.returncode}): {(proc.stderr or '').strip() or '<no output>'}"
            )

    def wait(self, timeout: float | None = None) -> int:
        """Block until the container exits; return its exit code.

        Raises [`TimeoutError`][TimeoutError] on timeout, [`RuntimeError`][RuntimeError] on
        ``podman wait`` failures or non-numeric output.
        """
        try:
            proc = subprocess.run(  # nosec B603 B607 — argv built from fixed verbs + caller-controlled scope/container names — binary PATH lookup is the cross-distro contract
                ["podman", "wait", self.name],
                check=False,
                capture_output=True,
                text=True,
                timeout=timeout,
            )
        except subprocess.TimeoutExpired as exc:
            raise TimeoutError(f"container {self.name!r} did not exit within {timeout}s") from exc

        if proc.returncode != 0:
            detail = (proc.stderr or proc.stdout or "").strip() or "<no output>"
            raise RuntimeError(f"podman wait {self.name!r} failed (rc={proc.returncode}): {detail}")

        stdout = (proc.stdout or "").strip()
        try:
            return int(stdout)
        except ValueError as exc:
            raise RuntimeError(
                f"podman wait {self.name!r} returned unexpected output: "
                f"stdout={proc.stdout!r}, stderr={proc.stderr!r}"
            ) from exc

    def copy_in(self, src: Path, dest: str) -> None:
        """Copy a host path into the (stopped) container at *dest*.

        Directories are copied *contents-first* (``src/.``) so existing
        container contents at *dest* are preserved and augmented.
        """
        src_arg = f"{src}/." if src.is_dir() else str(src)
        subprocess.run(  # nosec B603 B607 — argv built from fixed verbs + caller-controlled scope/container names — binary PATH lookup is the cross-distro contract
            ["podman", "cp", src_arg, f"{self.name}:{dest}"],
            check=True,
            capture_output=True,
        )

    def login_command(
        self,
        *,
        command: tuple[str, ...] = _DEFAULT_LOGIN_COMMAND,
    ) -> list[str]:
        """Return an argv for [`os.execvp`][os.execvp] to attach interactively.

        Empty *command* uses the default tmux session.
        """
        return ["podman", "exec", "-it", self.name, *(command or _DEFAULT_LOGIN_COMMAND)]

    def logs(self, *, follow: bool = False, tail: int | None = None) -> LogStream:
        """Return a context-managed iterator over decoded log lines."""
        return PodmanLogStream(self.name, follow=follow, tail=tail)

    def stream_initial_logs(
        self,
        ready_check: Callable[[str], bool],
        timeout_sec: float | None,
    ) -> bool:
        """Stream logs until *ready_check* matches or *timeout_sec* elapses.

        Prints each line to stdout as it arrives.  Returns ``True`` when
        the ready marker is observed, ``False`` on timeout.
        """
        return _stream_initial_logs(self.name, timeout_sec, ready_check)


# ── Image handle ──────────────────────────────────────────────────────────


class PodmanImage:
    """Podman implementation of [`Image`][terok_sandbox.runtime.podman.Image].

    Values listed by ``podman images`` (``repository``, ``tag``, ``size``,
    ``created``) may be pre-populated at construction to avoid an extra
    inspect; when absent they fall back to empty strings.
    """

    def __init__(
        self,
        ref: str,
        *,
        repository: str = "",
        tag: str = "",
        size: str = "",
        created: str = "",
    ) -> None:
        self.ref = ref
        self._repository = repository
        self._tag = tag
        self._size = size
        self._created = created

    def __repr__(self) -> str:
        """Render as ``PodmanImage(ref='...')``."""
        return f"PodmanImage(ref={self.ref!r})"

    def __eq__(self, other: object) -> bool:
        """Equality by ref."""
        return isinstance(other, PodmanImage) and self.ref == other.ref

    def __hash__(self) -> int:
        """Hash by ref."""
        return hash(("PodmanImage", self.ref))

    @property
    def id(self) -> str | None:
        """Resolved image ID, or ``None`` when the image is absent."""
        try:
            out = subprocess.check_output(  # nosec B603 B607 — argv built from fixed verbs + caller-controlled scope/container names — binary PATH lookup is the cross-distro contract
                ["podman", "inspect", "-f", "{{.Id}}", self.ref],
                stderr=subprocess.DEVNULL,
                text=True,
            ).strip()
        except (subprocess.CalledProcessError, FileNotFoundError):
            return None
        return out or None

    @property
    def repository(self) -> str:
        """Repository portion (pre-populated or ``""``)."""
        return self._repository

    @property
    def tag(self) -> str:
        """Tag portion (pre-populated or ``""``)."""
        return self._tag

    @property
    def size(self) -> str:
        """Podman-rendered size string (pre-populated or ``""``)."""
        return self._size

    @property
    def created(self) -> str:
        """Podman-rendered creation timestamp (pre-populated or ``""``)."""
        return self._created

    def exists(self) -> bool:
        """Return ``True`` if the image is present locally."""
        try:
            result = subprocess.run(  # nosec B603 B607 — argv built from fixed verbs + caller-controlled scope/container names — binary PATH lookup is the cross-distro contract
                ["podman", "image", "exists", self.ref],
                capture_output=True,
                timeout=10,
            )
        except (FileNotFoundError, subprocess.TimeoutExpired):
            return False
        return result.returncode == 0

    def labels(self) -> dict[str, str]:
        """Return the OCI ``Config.Labels`` as a flat string dict."""
        try:
            result = subprocess.run(  # nosec B603 B607 — argv built from fixed verbs + caller-controlled scope/container names — binary PATH lookup is the cross-distro contract
                ["podman", "inspect", "--format", "{{json .Config.Labels}}", self.ref],
                capture_output=True,
                text=True,
                check=True,
                timeout=10,
            )
        except (FileNotFoundError, subprocess.CalledProcessError, subprocess.TimeoutExpired):
            return {}

        import json as _json

        try:
            parsed = _json.loads(result.stdout) or {}
        except _json.JSONDecodeError:
            return {}
        if not isinstance(parsed, dict):
            return {}
        return {str(k): str(v) for k, v in parsed.items()}

    def history(self) -> list[str]:
        """Return the ``CreatedBy`` string of each layer, top to bottom."""
        try:
            result = subprocess.run(  # nosec B603 B607 — argv built from fixed verbs + caller-controlled scope/container names — binary PATH lookup is the cross-distro contract
                [
                    "podman",
                    "image",
                    "history",
                    "--format",
                    "{{.CreatedBy}}",
                    self.ref,
                ],
                capture_output=True,
                text=True,
                timeout=30,
                check=False,
            )
        except (FileNotFoundError, subprocess.TimeoutExpired):
            return []
        if result.returncode != 0:
            return []
        return [line for line in result.stdout.splitlines() if line]

    def remove(self) -> bool:
        """Remove the image; return ``True`` on success.

        No force flag — an image referenced by a running container stays,
        which matches cleanup semantics (sweeping safe garbage, not
        reaping live state).
        """
        try:
            result = subprocess.run(  # nosec B603 B607 — argv built from fixed verbs + caller-controlled scope/container names — binary PATH lookup is the cross-distro contract
                ["podman", "image", "rm", self.ref],
                capture_output=True,
                timeout=30,
                check=False,
            )
        except (FileNotFoundError, subprocess.TimeoutExpired):
            return False
        return result.returncode == 0


# ── LogStream ─────────────────────────────────────────────────────────────


class PodmanLogStream:
    """Iterator over podman log lines.

    Wraps ``podman logs [-f] [--tail N]`` in a ``subprocess.Popen`` and
    yields decoded lines.  ``__exit__`` terminates the child; calling
    ``close()`` mid-iteration has the same effect.
    """

    def __init__(self, container_name: str, *, follow: bool, tail: int | None) -> None:
        cmd = ["podman", "logs"]
        if follow:
            cmd.append("-f")
        if tail is not None:
            cmd.extend(["--tail", str(tail)])
        cmd.append(container_name)
        self._proc = subprocess.Popen(  # noqa: S603 — cmd built above  # nosec B603 — argv is a fixed list controlled by this module
            cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
        )

    @property
    def process(self) -> subprocess.Popen:
        """Underlying ``Popen`` handle — exposed for callers needing low-level access."""
        return self._proc

    def __iter__(self) -> PodmanLogStream:
        """Return ``self`` — iteration reads from the child's stdout."""
        return self

    def __next__(self) -> str:
        """Read the next decoded log line; raise ``StopIteration`` at EOF."""
        if self._proc.stdout is None:
            raise StopIteration
        line = self._proc.stdout.readline()
        if not line:
            raise StopIteration
        return line.decode("utf-8", errors="replace").rstrip("\n")

    def __enter__(self) -> PodmanLogStream:
        """Enter the context — the stream is already live."""
        return self

    def __exit__(self, *exc: object) -> None:
        """Close the stream; terminate the child if still running."""
        self.close()

    def close(self) -> None:
        """Terminate the underlying ``podman logs`` process and release its pipes.

        Reaps the child (terminate → wait → kill fallback) and then
        closes both parent-side file descriptors so repeated
        ``container.logs()`` calls do not leak FDs.  Safe to call
        multiple times; second call is a no-op.
        """
        if self._proc.poll() is None:
            self._proc.terminate()
            try:
                self._proc.wait(timeout=2)
            except subprocess.TimeoutExpired:
                self._proc.kill()
                self._proc.wait()
        for stream in (self._proc.stdout, self._proc.stderr):
            if stream is not None:
                try:
                    stream.close()
                except OSError:
                    pass
        self._proc.stdout = None
        self._proc.stderr = None


# ── EventStream ───────────────────────────────────────────────────────────

# Container lifecycle transitions worth waking a watcher for.  Deliberately
# excludes the noisy ``exec``/``attach`` pair that fires every time terok runs a
# command inside a container — those don't change what a task list renders.
_LIFECYCLE_STATUSES = frozenset(
    {"create", "init", "start", "died", "stop", "kill", "remove", "pause", "unpause"}
)


@dataclass(frozen=True)
class ContainerEvent:
    """A single podman container lifecycle event: a name and what happened."""

    name: str
    status: str


def _parse_event(line: str, prefix: str) -> ContainerEvent | None:
    """Decode one ``podman events --format '{{json .}}'`` line.

    Returns a [`ContainerEvent`][terok_sandbox.runtime.podman.ContainerEvent]
    for a lifecycle change on a container named ``<prefix>-…``, or ``None`` for
    everything else — other containers, ``exec``/``attach`` noise, or malformed
    JSON.  Field names are read case-insensitively because podman has shipped
    both ``Status`` and ``status`` spellings across versions.
    """
    try:
        raw = json.loads(line)
    except ValueError:  # JSONDecodeError is a ValueError subclass
        return None
    if (raw.get("Type") or raw.get("type")) != "container":
        return None
    name = raw.get("Name") or raw.get("name") or ""
    if not name.startswith(f"{prefix}-"):
        return None
    status = (raw.get("Status") or raw.get("status") or "").lower()
    if status not in _LIFECYCLE_STATUSES:
        return None
    return ContainerEvent(name=name, status=status)


class PodmanEventStream:
    """Iterator over podman container lifecycle events for a name prefix.

    The push-based companion to
    [`container_states`][terok_sandbox.runtime.podman.PodmanRuntime.container_states]:
    wraps ``podman events --filter type=container --format '{{json .}}'`` in a
    ``subprocess.Popen`` and yields
    [`ContainerEvent`][terok_sandbox.runtime.podman.ContainerEvent] records whose
    container name starts with ``<prefix>-``.  Iteration blocks between events;
    ``close()`` (or ``__exit__``) terminates the child, which unblocks a consumer
    thread parked in ``__next__``.
    """

    def __init__(self, prefix: str) -> None:
        self._prefix = prefix
        self._proc = subprocess.Popen(  # noqa: S603  # nosec B603 B607 — fixed argv; podman PATH lookup is the cross-distro contract
            ["podman", "events", "--filter", "type=container", "--format", "{{json .}}"],
            stdout=subprocess.PIPE,
            stderr=subprocess.DEVNULL,
        )

    @property
    def process(self) -> subprocess.Popen:
        """Underlying ``Popen`` handle — exposed for callers needing the fd."""
        return self._proc

    def __iter__(self) -> PodmanEventStream:
        """Return ``self`` — iteration reads from the child's stdout."""
        return self

    def __next__(self) -> ContainerEvent:
        """Block until the next matching lifecycle event; stop at EOF."""
        if self._proc.stdout is None:
            raise StopIteration
        while True:
            line = self._proc.stdout.readline()
            if not line:
                raise StopIteration
            event = _parse_event(line.decode("utf-8", errors="replace"), self._prefix)
            if event is not None:
                return event

    def __enter__(self) -> PodmanEventStream:
        """Enter the context — the stream is already live."""
        return self

    def __exit__(self, *exc: object) -> None:
        """Close the stream; terminate the child if still running."""
        self.close()

    def close(self) -> None:
        """Terminate the ``podman events`` child and release its pipe.

        Reaps the child (terminate → wait → kill fallback) and closes the
        parent-side fd so a long-running TUI doesn't leak one per project
        switch.  Safe to call multiple times; the second call is a no-op.
        """
        if self._proc.poll() is None:
            self._proc.terminate()
            try:
                self._proc.wait(timeout=2)
            except subprocess.TimeoutExpired:
                self._proc.kill()
                self._proc.wait()
        if self._proc.stdout is not None:
            try:
                self._proc.stdout.close()
            except OSError:
                pass
            self._proc.stdout = None


# ── PortReservation ───────────────────────────────────────────────────────


class PodmanPortReservation:
    """Holds a TCP port open until released.

    Bind on construction; the reserved port number is exposed via the
    ``port`` attribute.  Caller is responsible for closing (directly via
    [`close`][terok_sandbox.runtime.podman.PodmanPortReservation.close] or via ``with``).
    """

    def __init__(self, host: str = "127.0.0.1") -> None:
        s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        try:
            s.bind((host, 0))
        except BaseException:
            s.close()
            raise
        self._socket: socket.socket | None = s
        self.port = s.getsockname()[1]

    def __enter__(self) -> PodmanPortReservation:
        """Enter the context — reservation is already held."""
        return self

    def __exit__(self, *exc: object) -> None:
        """Release the port."""
        self.close()

    def close(self) -> None:
        """Release the port explicitly (idempotent)."""
        if self._socket is not None:
            self._socket.close()
            self._socket = None


# ── Runtime ───────────────────────────────────────────────────────────────


class PodmanRuntime:
    """The default [`ContainerRuntime`][terok_sandbox.ContainerRuntime] — talks to the podman CLI."""

    def container(self, name: str) -> Container:
        """Return a handle to the container named *name*."""
        return PodmanContainer(name, runtime=self)

    def containers_with_prefix(self, prefix: str) -> list[Container]:
        """Return handles for every container whose name starts with *prefix-*.

        Single ``podman ps -a`` call under the hood; the returned handles
        are lazy (fresh inspect on property access).
        """
        try:
            out = subprocess.check_output(  # nosec B603 B607 — argv built from fixed verbs + caller-controlled scope/container names — binary PATH lookup is the cross-distro contract
                [
                    "podman",
                    "ps",
                    "-a",
                    "--filter",
                    f"name=^{prefix}-",
                    "--format",
                    "{{.Names}}",
                    "--no-trunc",
                ],
                stderr=subprocess.DEVNULL,
                text=True,
            )
        except (subprocess.CalledProcessError, FileNotFoundError):
            return []
        return [PodmanContainer(name, runtime=self) for name in out.strip().splitlines() if name]

    def image(self, ref: str) -> Image:
        """Return a handle to the image identified by tag or ID *ref*."""
        return PodmanImage(ref)

    def images(self, *, dangling_only: bool = False) -> list[Image]:
        """Enumerate local images.

        *dangling_only* narrows to untagged ``<none>:<none>`` entries.
        """
        cmd = ["podman", "images", "--format", _IMAGES_FORMAT, "--no-trunc"]
        if dangling_only:
            cmd[2:2] = ["--filter", "dangling=true"]
        try:
            result = subprocess.run(  # nosec B603 — argv is a fixed list controlled by this module
                cmd,
                capture_output=True,
                text=True,
                timeout=30,
                check=False,
            )
        except (FileNotFoundError, subprocess.TimeoutExpired):
            return []
        if result.returncode != 0:
            return []

        images: list[Image] = []
        for line in result.stdout.strip().splitlines():
            parts = line.split("\t")
            if len(parts) == 5:
                repo, tag, image_id, size, created = parts
                images.append(
                    PodmanImage(
                        ref=image_id,
                        repository=repo,
                        tag=tag,
                        size=size,
                        created=created,
                    )
                )
        return images

    def exec(
        self,
        container: Container,
        cmd: list[str],
        *,
        timeout: float | None = None,
    ) -> ExecResult:
        """Run *cmd* inside *container* via ``podman exec``.

        Lets [`FileNotFoundError`][FileNotFoundError] (podman missing) and
        [`subprocess.TimeoutExpired`][subprocess.TimeoutExpired] propagate unchanged.

        Raises [`ValueError`][ValueError] if *cmd* is empty — podman exec with
        no argv is never a valid request and catching it here avoids a
        later ``IndexError`` in the debug log.
        """
        if not cmd:
            raise ValueError("exec argv must not be empty")
        log_debug(
            f"PodmanRuntime.exec({container.name}, cmd[0]={cmd[0]!r}, "
            f"argc={len(cmd)}, timeout={timeout})"
        )
        proc = subprocess.run(  # nosec B603 B607 — argv built from fixed verbs + caller-controlled scope/container names — binary PATH lookup is the cross-distro contract
            ["podman", "exec", container.name, *cmd],
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
        """Bridge byte streams to ``podman exec -i`` for *cmd* inside *container*.

        Synchronous: spawns the child, runs three daemon pump threads
        (one per direction) copying bytes until either side reaches
        EOF or the child exits, joins the pumps, returns the exit code.
        Async callers drive this via
        [`run_in_executor`][asyncio.loop.run_in_executor].

        Lets [`FileNotFoundError`][] (podman missing) propagate.  On
        timeout, terminates the child (terminate → 2 s wait → kill) and
        re-raises [`TimeoutExpired`][subprocess.TimeoutExpired].
        """
        if not cmd:
            raise ValueError("exec_stdio argv must not be empty")
        log_debug(
            f"PodmanRuntime.exec_stdio({container.name}, cmd[0]={cmd[0]!r}, "
            f"argc={len(cmd)}, timeout={timeout})"
        )
        argv = ["podman", "exec", "-i"]
        for k, v in (env or {}).items():
            argv += ["-e", f"{k}={v}"]
        argv += [container.name, *cmd]

        proc = subprocess.Popen(  # noqa: S603 — argv built above  # nosec B603 — argv is a fixed list controlled by this module
            argv,
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE if stderr is not None else subprocess.DEVNULL,
        )

        pumps = _start_stdio_pumps(proc, stdin, stdout, stderr)
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
        finally:
            # Join pumps before closing parent-side pipes — the stdout
            # and stderr pumps drain whatever the child wrote before
            # exiting, and closing the read end first would chop off
            # tail bytes that are still in the kernel buffer.
            for t in pumps:
                t.join(timeout=1)
            _close_proc_streams(proc)

    def force_remove(self, containers: list[Container]) -> list[ContainerRemoveResult]:
        """Best-effort ``podman rm -f`` of each container.

        Continues through individual failures.  An already-absent
        container counts as *removed* — the post-condition holds.
        """
        results: list[ContainerRemoveResult] = []
        for container in containers:
            name = container.name
            try:
                log_debug(f"force_remove: podman rm -f {name} (start)")
                proc = subprocess.run(  # nosec B603 B607 — argv built from fixed verbs + caller-controlled scope/container names — binary PATH lookup is the cross-distro contract
                    ["podman", "rm", "-f", name],
                    check=False,
                    stdout=subprocess.DEVNULL,
                    stderr=subprocess.PIPE,
                    text=True,
                    timeout=_CONTAINER_REMOVE_TIMEOUT,
                )
                if proc.returncode == 0:
                    log_debug(f"force_remove: {name} (done)")
                    results.append(ContainerRemoveResult(name=name, removed=True))
                elif "no such container" in (proc.stderr or "").lower():
                    log_debug(f"force_remove: {name} already absent")
                    results.append(ContainerRemoveResult(name=name, removed=True))
                else:
                    reason = (proc.stderr or "").strip() or f"exit code {proc.returncode}"
                    log_debug(f"force_remove: {name} failed: {reason}")
                    results.append(ContainerRemoveResult(name=name, removed=False, error=reason))
            except subprocess.TimeoutExpired:
                log_debug(f"force_remove: {name} timed out")
                results.append(
                    ContainerRemoveResult(
                        name=name,
                        removed=False,
                        error=f"timed out after {_CONTAINER_REMOVE_TIMEOUT}s",
                    )
                )
            except FileNotFoundError:
                log_debug(f"force_remove: podman not found for {name}")
                results.append(
                    ContainerRemoveResult(name=name, removed=False, error="podman not found")
                )
            except Exception as exc:  # noqa: BLE001
                log_debug(f"force_remove: {name} failed: {exc}")
                results.append(ContainerRemoveResult(name=name, removed=False, error=str(exc)))
        return results

    def reserve_port(self, host: str = "127.0.0.1") -> PortReservation:
        """Reserve a free TCP port; release on close."""
        return PodmanPortReservation(host)

    # -- Batch helpers (optimisations over repeated handle ops) -------------

    def container_states(self, prefix: str) -> dict[str, str]:
        """Return ``{container_name: state}`` for matching containers.

        Optimisation over ``[c.state for c in containers_with_prefix(prefix)]``
        — single ``podman ps -a`` instead of N inspects.  Backend-specific;
        not part of the [`ContainerRuntime`][terok_sandbox.ContainerRuntime] protocol.
        """
        try:
            out = subprocess.check_output(  # nosec B603 B607 — argv built from fixed verbs + caller-controlled scope/container names — binary PATH lookup is the cross-distro contract
                [
                    "podman",
                    "ps",
                    "-a",
                    "--filter",
                    f"name=^{prefix}-",
                    "--format",
                    "{{.Names}} {{.State}}",
                    "--no-trunc",
                ],
                stderr=subprocess.DEVNULL,
                text=True,
            )
        except (subprocess.CalledProcessError, FileNotFoundError):
            return {}

        result: dict[str, str] = {}
        for line in out.strip().splitlines():
            parts = line.split(None, 1)
            if len(parts) == 2:
                result[parts[0]] = parts[1].lower()
        return result

    def events(self, prefix: str) -> PodmanEventStream:
        """Stream container lifecycle events for containers named ``<prefix>-…``.

        The push-based companion to
        [`container_states`][terok_sandbox.runtime.podman.PodmanRuntime.container_states]:
        a long-lived ``podman events`` subscription so a watcher reacts to a
        container starting, dying or being removed instead of polling
        ``podman ps`` on a clock.  Returns a
        [`PodmanEventStream`][terok_sandbox.runtime.podman.PodmanEventStream] the
        caller iterates (typically on a worker thread) and ``close()``s when
        done.  Backend-specific; not part of the
        [`ContainerRuntime`][terok_sandbox.ContainerRuntime] protocol.
        """
        return PodmanEventStream(prefix)

    def container_rw_sizes(self, prefix: str) -> dict[str, int]:
        """Return ``{container_name: rw_bytes}`` for matching containers.

        Single ``podman ps --size`` call — ``--size`` is expensive (overlay
        diffs) but one bulk call beats N inspects.  Backend-specific; not
        part of the [`ContainerRuntime`][terok_sandbox.ContainerRuntime] protocol.
        """
        try:
            out = subprocess.check_output(  # nosec B603 B607 — argv built from fixed verbs + caller-controlled scope/container names — binary PATH lookup is the cross-distro contract
                [
                    "podman",
                    "ps",
                    "-a",
                    "--size",
                    "--filter",
                    f"name=^{prefix}-",
                    "--format",
                    "{{.Names}}\t{{.Size}}",
                    "--no-trunc",
                ],
                stderr=subprocess.DEVNULL,
                text=True,
                timeout=120,
            )
        except (subprocess.CalledProcessError, FileNotFoundError, subprocess.TimeoutExpired):
            return {}

        result: dict[str, int] = {}
        for line in out.strip().splitlines():
            parts = line.split("\t", 1)
            if len(parts) == 2:
                parsed = _parse_human_size(parts[1])
                if parsed is not None:
                    result[parts[0]] = parsed
        return result


# ── exec_stdio internals ──────────────────────────────────────────────────


_STDIO_PUMP_CHUNK = 64 * 1024


_STDIN_PUMP_LABEL = "stdin→child"


def _pump_stream(src: BinaryIO, dst: BinaryIO, *, label: str) -> None:
    """Copy *src* → *dst* in chunks until EOF or write error.

    Both ends are byte-oriented file objects.  Errors are logged and
    silently end the pump — the partner pump (or the child's exit) is
    expected to terminate the overall call.

    The read uses ``read1`` when available so a small chunk arrives
    promptly: ``BufferedReader.read(n)`` blocks until ``n`` bytes
    *or* EOF and would stall this pump on JSON-RPC traffic where
    each frame is well under the 64 KB chunk size.  Raw / unbuffered
    streams (``os.fdopen(fd, "rb", buffering=0)``) lack ``read1``
    but their plain ``read`` already has "return whatever's
    available" semantics, so the fallback is correct.

    For the *stdin→child* pump specifically, ``dst`` is the host-side
    write-end of the child's stdin pipe.  When the source reaches EOF
    we close ``dst`` so the child sees EOF on its stdin promptly,
    rather than waiting for ``_close_proc_streams`` to run after
    ``proc.wait()`` (which would deadlock any child that exits on
    stdin EOF — most ACP wrappers do).
    """
    read = getattr(src, "read1", src.read)
    try:
        while True:
            chunk = read(_STDIO_PUMP_CHUNK)
            if not chunk:
                break
            dst.write(chunk)
            dst.flush()
    except (OSError, ValueError) as exc:  # closed pipe / detached fd
        log_debug(f"exec_stdio pump {label} ended: {exc}")
    except Exception as exc:  # noqa: BLE001
        log_warning(f"exec_stdio pump {label} unexpected error: {exc}")
    finally:
        with contextlib.suppress(OSError, ValueError):
            dst.flush()
        if label == _STDIN_PUMP_LABEL:
            with contextlib.suppress(OSError, ValueError):
                dst.close()


def _start_stdio_pumps(
    proc: subprocess.Popen,
    stdin: BinaryIO,
    stdout: BinaryIO,
    stderr: BinaryIO | None,
) -> list[threading.Thread]:
    """Spawn daemon pump threads bridging caller streams ↔ child stdio.

    Returns the spawned threads so the caller can ``join`` them after the
    child exits.  Threads are daemons so a wedged pump cannot keep the
    process alive past interpreter shutdown.
    """
    threads: list[threading.Thread] = []
    if proc.stdin is not None:
        threads.append(
            threading.Thread(
                target=_pump_stream,
                args=(stdin, proc.stdin),
                kwargs={"label": _STDIN_PUMP_LABEL},
                daemon=True,
            )
        )
    if proc.stdout is not None:
        threads.append(
            threading.Thread(
                target=_pump_stream,
                args=(proc.stdout, stdout),
                kwargs={"label": "child→stdout"},
                daemon=True,
            )
        )
    if proc.stderr is not None and stderr is not None:
        threads.append(
            threading.Thread(
                target=_pump_stream,
                args=(proc.stderr, stderr),
                kwargs={"label": "child→stderr"},
                daemon=True,
            )
        )
    for t in threads:
        t.start()
    return threads


def _close_proc_streams(proc: subprocess.Popen) -> None:
    """Close the child's stdio pipes once the call is done."""
    for stream in (proc.stdin, proc.stdout, proc.stderr):
        if stream is not None:
            with contextlib.suppress(OSError):
                stream.close()


# ── stream_initial_logs internals ─────────────────────────────────────────


class _Reaper:
    """One-shot reap guard attached to a ``Popen`` via ``__dict__.setdefault``."""

    __slots__ = ("lock", "done")

    def __init__(self) -> None:
        self.lock = threading.Lock()
        self.done = False


_REAPER_KEY = "_terok_reaper"


def _reap_logs_proc(proc: subprocess.Popen | None) -> None:
    """Terminate, wait, and close the ``podman logs`` child if still alive.

    Shared by every exit path of `_stream_initial_logs` so the
    podman child never leaks as a zombie and its stdout pipe never
    leaks as an open file descriptor.  Safe to call with ``None`` or
    with a child that has already exited.

    Idempotent and thread-safe under two callers racing through (reader
    thread's ``finally`` and the main thread's fallback): a per-process
    `_Reaper` guard claims ownership under a short lock, then
    releases it so the winner can run the terminate/wait/close sequence
    unsynchronised — a wedged reader mustn't stall the main thread's
    fallback.  Writing via ``__dict__`` (not ``setattr``) keeps
    MagicMock-typed procs in tests from auto-creating a truthy flag on
    the very first call.
    """
    if proc is None:
        return
    reaper: _Reaper = proc.__dict__.setdefault(_REAPER_KEY, _Reaper())
    with reaper.lock:
        if reaper.done:
            return
        reaper.done = True
    try:
        if proc.poll() is None:
            proc.terminate()
            try:
                proc.wait(timeout=2)
            except subprocess.TimeoutExpired:
                proc.kill()
                proc.wait()
        else:
            # Already exited — still reap to release zombie slot.
            proc.wait()
    finally:
        if proc.stdout is not None:
            try:
                proc.stdout.close()
            except OSError:
                pass


def _stream_initial_logs(
    container_name: str,
    timeout_sec: float | None,
    ready_check: Callable[[str], bool],
) -> bool:
    """Follow ``podman logs -f`` in a thread until the ready marker fires."""
    holder: list[bool] = [False]
    stop_event = threading.Event()
    proc_holder: list[subprocess.Popen | None] = [None]

    def _read_loop() -> None:
        proc: subprocess.Popen | None = None
        try:
            proc = subprocess.Popen(  # nosec B603 B607 — argv built from fixed verbs + caller-controlled scope/container names — binary PATH lookup is the cross-distro contract
                ["podman", "logs", "-f", container_name],
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
            )
            if proc.stdout is None:  # stdout=PIPE above guarantees a pipe
                return
            proc_holder[0] = proc
            start_time = time.time()
            buf = b""

            while not stop_event.is_set():
                if timeout_sec is not None and time.time() - start_time >= timeout_sec:
                    break
                if proc.poll() is not None:
                    remaining = proc.stdout.read()
                    if remaining:
                        buf += remaining
                    break
                try:
                    ready, _, _ = select.select([proc.stdout], [], [], 0.2)
                    if not ready:
                        continue
                    chunk = proc.stdout.read1(4096) if hasattr(proc.stdout, "read1") else b""
                    if not chunk:
                        continue
                    buf += chunk
                except Exception as exc:  # noqa: BLE001
                    log_warning(f"stream_initial_logs read error: {exc}")
                    break

                while b"\n" in buf:
                    raw_line, buf = buf.split(b"\n", 1)
                    line = raw_line.decode("utf-8", errors="replace").strip()
                    if line:
                        print(line, file=sys.stdout, flush=True)
                        if ready_check(line):
                            holder[0] = True
                            return

            if buf:
                line = buf.decode("utf-8", errors="replace").strip()
                if line:
                    print(line, file=sys.stdout, flush=True)
                    if ready_check(line):
                        holder[0] = True
        except Exception as exc:  # noqa: BLE001
            log_warning(f"stream_initial_logs error: {exc}")
        finally:
            _reap_logs_proc(proc)

    stream_thread = threading.Thread(target=_read_loop)
    stream_thread.start()
    stream_thread.join(timeout_sec)

    if stream_thread.is_alive():
        stop_event.set()
        # Reap from the main thread too in case the reader is wedged in
        # select.select and hasn't reached its finally block yet — the
        # podman child still needs its fds closed before we return.
        _reap_logs_proc(proc_holder[0])
        stream_thread.join(timeout=5)

    return holder[0]
