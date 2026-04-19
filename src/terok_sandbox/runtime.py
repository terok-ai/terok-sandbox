# SPDX-FileCopyrightText: 2026 Jiri Vyskocil
# SPDX-License-Identifier: Apache-2.0

"""Container runtime gateway wrapping the Podman CLI.

Provides module-level functions for container lifecycle operations
(state queries, GPU args, log streaming, port allocation, etc.).

All functions accept plain parameters (strings, paths) — no terok-specific
types like ``ProjectConfig``.  Container naming is orchestration policy and
lives in the caller.
"""

import os
import re
import socket
import subprocess
from collections.abc import Callable
from dataclasses import dataclass

from ._util import log_debug, log_warning

# ---------- Container removal result ----------


@dataclass(frozen=True)
class ContainerRemoveResult:
    """Per-container outcome from :func:`stop_task_containers`."""

    name: str
    """Container name that was targeted."""

    removed: bool
    """Whether the container is confirmed absent (includes already-gone)."""

    error: str | None = None
    """Human-readable reason when *removed* is ``False``."""


_CONTAINER_REMOVE_TIMEOUT = 120
"""Per-container timeout (seconds) for ``podman rm -f``."""


# ---------- User namespace ----------


def podman_userns_args() -> list[str]:
    """Return user namespace args for rootless podman so UID 1000 maps correctly.

    Maps the host user to container UID/GID 1000, the conventional non-root
    ``dev`` user in terok container images.
    """
    if os.geteuid() == 0:
        return []
    return ["--userns=keep-id:uid=1000,gid=1000"]


# ---------- GPU error handling ----------

_CDI_HINT = (
    "Hint: NVIDIA CDI configuration appears to be missing or broken.\n"
    "Ensure the NVIDIA Container Toolkit is installed and CDI is configured.\n"
    "See: https://podman-desktop.io/docs/podman/gpu"
)

_CDI_ERROR_PATTERNS = ("cdi.k8s.io", "nvidia.com/gpu", "CDI")


class GpuConfigError(RuntimeError):
    """CDI/NVIDIA misconfiguration detected during container launch."""

    def __init__(self, message: str, *, hint: str = _CDI_HINT) -> None:
        """Store the CDI *hint* alongside the standard error *message*."""
        self.hint = hint
        super().__init__(message)


def check_gpu_error(exc: subprocess.CalledProcessError) -> None:
    """Raise :class:`GpuConfigError` if *exc* looks like a CDI/NVIDIA issue.

    Does nothing if the error does not match any known CDI patterns.
    """
    stderr = (exc.stderr or b"").decode(errors="replace")
    if any(pat in stderr for pat in _CDI_ERROR_PATTERNS):
        msg = f"Container launch failed (GPU misconfiguration):\n{stderr.strip()}\n\n{_CDI_HINT}"
        raise GpuConfigError(msg) from exc


# ---------- Env redaction ----------

_SENSITIVE_KEY_RE = re.compile(r"(?i)(KEY|TOKEN|SECRET|API|PASSWORD|PRIVATE)")
_ALWAYS_REDACT_KEYS = frozenset({"CODE_REPO", "CLONE_FROM"})


def redact_env_args(cmd: list[str]) -> list[str]:
    """Return a copy of *cmd* with sensitive ``-e KEY=VALUE`` args redacted.

    Handles the two-arg form (``-e KEY=VALUE``) produced by
    :meth:`~.sandbox.Sandbox.run`.  Does not handle ``--env``,
    ``-e=KEY=VALUE``, or ``--env=KEY=VALUE`` — callers passing sensitive
    values via ``extra_args`` must pre-redact them.
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


# ---------- Public functions ----------


def get_container_states(name_prefix: str) -> dict[str, str]:
    """Return ``{container_name: state}`` for all containers matching *name_prefix*.

    Uses a single ``podman ps -a`` call with a name filter instead of
    per-container ``podman inspect`` calls.  Returns an empty dict when
    podman is unavailable.
    """
    try:
        out = subprocess.check_output(
            [
                "podman",
                "ps",
                "-a",
                "--filter",
                f"name=^{name_prefix}-",
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


def get_container_state(cname: str) -> str | None:
    """Return container state ('running', 'exited', ...) or ``None`` if not found."""
    try:
        out = subprocess.check_output(
            ["podman", "inspect", "-f", "{{.State.Status}}", cname],
            stderr=subprocess.DEVNULL,
            text=True,
        ).strip()
        return out.lower() if out else None
    except (subprocess.CalledProcessError, FileNotFoundError):
        return None


def is_container_running(cname: str) -> bool:
    """Return ``True`` if the named container is currently running."""
    try:
        out = subprocess.check_output(
            ["podman", "inspect", "-f", "{{.State.Running}}", cname],
            stderr=subprocess.DEVNULL,
            text=True,
        ).strip()
    except (subprocess.CalledProcessError, FileNotFoundError):
        return False
    return out.lower() == "true"


def container_image(cname: str) -> str | None:
    """Return the image ID the container *cname* is built on, or ``None``.

    Used by callers that need to reason about the running container's
    image (e.g. is-its-build-hash-still-current checks).  ``None`` on
    missing container, absent podman, or any inspect failure — the three
    signals collapse because the caller's reaction is identical ("can't
    determine, skip").
    """
    try:
        out = subprocess.check_output(
            ["podman", "inspect", "-f", "{{.Image}}", cname],
            stderr=subprocess.DEVNULL,
            text=True,
        ).strip()
    except (subprocess.CalledProcessError, FileNotFoundError):
        return None
    return out or None


def image_exists(tag: str) -> bool:
    """Return ``True`` when a container image with *tag* is present locally.

    Uses ``podman image exists`` (exit 0 = present).  Returns ``False`` when
    podman is not on PATH — callers that need to distinguish "image missing"
    from "podman missing" should check executable availability themselves.
    """
    try:
        result = subprocess.run(
            ["podman", "image", "exists", tag],
            capture_output=True,
            timeout=10,
        )
    except (FileNotFoundError, subprocess.TimeoutExpired):
        return False
    return result.returncode == 0


def image_labels(tag: str) -> dict[str, str]:
    """Return the OCI ``Config.Labels`` of *tag* as a string dict.

    Empty dict when the image is absent, has no labels, or podman is
    unavailable.  Labels are the natural vocabulary for cross-cutting
    image metadata (agent taxonomy, build-context hash, provenance) —
    :func:`get_container_state` and friends cover container state, this
    covers image state.
    """
    try:
        result = subprocess.run(
            ["podman", "inspect", "--format", "{{json .Config.Labels}}", tag],
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
    # podman sometimes emits ``null`` when Labels is unset
    if not isinstance(parsed, dict):
        return {}
    return {str(k): str(v) for k, v in parsed.items()}


@dataclass(frozen=True)
class ImageRecord:
    """One row from ``podman images`` — name, tag, id, size, created.

    All fields are strings exactly as podman renders them (sizes are
    human-readable like ``"1.2GB"``; created is the podman timestamp
    string).  Callers that need structured sizes should post-process with
    :func:`_parse_human_size`.
    """

    repository: str
    tag: str
    image_id: str
    size: str
    created: str


_IMAGES_FORMAT = "{{.Repository}}\t{{.Tag}}\t{{.ID}}\t{{.Size}}\t{{.Created}}"


def images_list(*, dangling_only: bool = False) -> list[ImageRecord]:
    """List local container images as :class:`ImageRecord` rows.

    When *dangling_only*, applies ``--filter dangling=true`` — callers
    doing cleanup use this to find removable images without matching
    ``<none>:<none>`` tag strings by hand.  Empty list on podman error
    or absence.
    """
    cmd = ["podman", "images", "--format", _IMAGES_FORMAT, "--no-trunc"]
    if dangling_only:
        cmd[2:2] = ["--filter", "dangling=true"]
    try:
        result = subprocess.run(
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

    records: list[ImageRecord] = []
    for line in result.stdout.strip().splitlines():
        parts = line.split("\t")
        if len(parts) == 5:
            records.append(ImageRecord(*parts))
    return records


def image_history(image_id: str) -> list[str]:
    """Return the ``CreatedBy`` string of each layer of *image_id*.

    Empty list when the image is missing or podman is unavailable —
    callers use the strings to match build-time layer provenance
    (e.g. ``terok.`` layer prefixes) without parsing JSON.
    """
    try:
        result = subprocess.run(
            ["podman", "image", "history", "--format", "{{.CreatedBy}}", image_id],
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


def image_rm(image_id: str) -> bool:
    """Remove *image_id*; return ``True`` on success.

    No force flag — an image referenced by a running container stays,
    which matches cleanup semantics (sweeping safe garbage, not reaping
    live state).  Returns ``False`` when podman is absent or the image
    is not removable.
    """
    try:
        result = subprocess.run(
            ["podman", "image", "rm", image_id],
            capture_output=True,
            timeout=30,
            check=False,
        )
    except (FileNotFoundError, subprocess.TimeoutExpired):
        return False
    return result.returncode == 0


def get_container_rw_size(cname: str) -> int | None:
    """Return the writable-layer size in bytes for *cname*, or ``None``.

    Uses ``podman container inspect --size`` which computes the overlay
    diff on the fly — expect a brief pause for large containers.
    """
    try:
        out = subprocess.check_output(
            ["podman", "container", "inspect", "--size", "-f", "{{.SizeRw}}", cname],
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
    """Best-effort parse of podman's human-readable size strings.

    Handles forms like ``"12.5MB (virtual 1.23GB)"`` — extracts only
    the first number (the writable layer, before "virtual").
    """
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


def get_container_rw_sizes(name_prefix: str) -> dict[str, int]:
    """Return ``{container_name: rw_bytes}`` for all containers matching *name_prefix*.

    Uses a single ``podman ps --size`` call — the ``--size`` flag is
    expensive (podman computes overlay diffs), but one bulk call beats
    N individual inspects.
    """
    try:
        out = subprocess.check_output(
            [
                "podman",
                "ps",
                "-a",
                "--size",
                "--filter",
                f"name=^{name_prefix}-",
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


def stop_task_containers(container_names: list[str]) -> list[ContainerRemoveResult]:
    """Best-effort ``podman rm -f`` of the given containers.

    Always attempts every container regardless of individual failures.
    Returns a per-container :class:`ContainerRemoveResult` so the caller
    can decide how to present successes and failures.

    A container that is already absent (``"no such container"``) counts as
    *removed* — the goal is achieved.
    """
    results: list[ContainerRemoveResult] = []
    for name in container_names:
        try:
            log_debug(f"stop_containers: podman rm -f {name} (start)")
            proc = subprocess.run(
                ["podman", "rm", "-f", name],
                check=False,
                stdout=subprocess.DEVNULL,
                stderr=subprocess.PIPE,
                text=True,
                timeout=_CONTAINER_REMOVE_TIMEOUT,
            )
            if proc.returncode == 0:
                log_debug(f"stop_containers: podman rm -f {name} (done)")
                results.append(ContainerRemoveResult(name=name, removed=True))
            elif "no such container" in (proc.stderr or "").lower():
                log_debug(f"stop_containers: {name} already absent")
                results.append(ContainerRemoveResult(name=name, removed=True))
            else:
                reason = (proc.stderr or "").strip() or f"exit code {proc.returncode}"
                log_debug(f"stop_containers: {name} failed: {reason}")
                results.append(ContainerRemoveResult(name=name, removed=False, error=reason))
        except subprocess.TimeoutExpired:
            log_debug(f"stop_containers: {name} timed out")
            results.append(
                ContainerRemoveResult(
                    name=name,
                    removed=False,
                    error=f"timed out after {_CONTAINER_REMOVE_TIMEOUT}s",
                )
            )
        except FileNotFoundError:
            log_debug(f"stop_containers: podman not found for {name}")
            results.append(
                ContainerRemoveResult(name=name, removed=False, error="podman not found")
            )
        except Exception as exc:
            log_debug(f"stop_containers: {name} failed: {exc}")
            results.append(ContainerRemoveResult(name=name, removed=False, error=str(exc)))
    return results


# ---------- Container exec / start / stop ----------

_DEFAULT_LOGIN_COMMAND: tuple[str, ...] = ("tmux", "new-session", "-A", "-s", "main")


def sandbox_exec(
    cname: str,
    cmd: list[str],
    *,
    timeout: int = 30,
) -> subprocess.CompletedProcess[str]:
    """Run *cmd* inside container *cname* via ``podman exec``.

    Returns the completed process so the caller can inspect
    *returncode*, *stdout*, and *stderr*.  Lets ``FileNotFoundError``
    (podman missing) and ``subprocess.TimeoutExpired`` propagate.
    """
    log_debug(f"sandbox_exec({cname}, cmd[0]={cmd[0]!r}, argc={len(cmd)}, timeout={timeout})")
    return subprocess.run(
        ["podman", "exec", cname, *cmd],
        capture_output=True,
        text=True,
        timeout=timeout,
        check=False,
    )


def login_command(
    cname: str,
    *,
    command: tuple[str, ...] = _DEFAULT_LOGIN_COMMAND,
) -> list[str]:
    """Return the full command to interactively enter *cname*.

    The returned list is suitable for :func:`os.execvp` or
    :func:`shlex.join` (display).  No subprocess is spawned.
    """
    return ["podman", "exec", "-it", cname, *command]


_START_TIMEOUT = 30
"""Subprocess timeout (seconds) for ``podman start``."""


def container_start(cname: str) -> subprocess.CompletedProcess[str]:
    """Start a stopped container, returning the process result.

    Captures *stderr* for error reporting; *stdout* is discarded.
    Lets ``FileNotFoundError`` and ``TimeoutExpired`` propagate.
    """
    log_debug(f"container_start({cname})")
    return subprocess.run(
        ["podman", "start", cname],
        check=False,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.PIPE,
        text=True,
        timeout=_START_TIMEOUT,
    )


_STOP_TIMEOUT_BUFFER = 5
"""Extra seconds beyond the podman ``--time`` grace period."""


def container_stop(cname: str, *, timeout: int = 10) -> subprocess.CompletedProcess[str]:
    """Stop a running container, returning the process result.

    Sends ``podman stop --time <timeout>`` which gives the container
    *timeout* seconds to exit before SIGKILL.  The subprocess timeout
    adds a buffer on top so the CLI itself can finish.  Captures
    *stderr* for error reporting; *stdout* is discarded.
    Lets ``FileNotFoundError`` and ``TimeoutExpired`` propagate.
    """
    log_debug(f"container_stop({cname}, timeout={timeout})")
    return subprocess.run(
        ["podman", "stop", "--time", str(timeout), cname],
        check=False,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.PIPE,
        text=True,
        timeout=timeout + _STOP_TIMEOUT_BUFFER,
    )


def gpu_run_args(*, enabled: bool = False) -> list[str]:
    """Return additional ``podman run`` args to enable NVIDIA GPU passthrough.

    The caller is responsible for determining whether GPUs are enabled
    (e.g. by reading project configuration).  This function only maps
    the boolean flag to the appropriate podman CLI arguments.
    """
    if not enabled:
        return []

    return [
        "--device",
        "nvidia.com/gpu=all",
        "-e",
        "NVIDIA_VISIBLE_DEVICES=all",
        "-e",
        "NVIDIA_DRIVER_CAPABILITIES=all",
    ]


def stream_initial_logs(
    container_name: str,
    timeout_sec: float | None,
    ready_check: Callable[[str], bool],
) -> bool:
    """Stream logs until ready marker is seen or timeout.

    Returns ``True`` if the ready marker was found, ``False`` on timeout.
    """
    import select
    import sys
    import threading
    import time

    holder: list[bool] = [False]
    stop_event = threading.Event()
    proc_holder: list[subprocess.Popen | None] = [None]

    def _stream_logs() -> None:
        """Follow container logs in a thread, setting *holder[0]* on ready."""
        try:
            proc = subprocess.Popen(
                ["podman", "logs", "-f", container_name],
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
            )
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
                except Exception as exc:
                    log_warning(f"_stream_initial_logs read error: {exc}")
                    break

                while b"\n" in buf:
                    raw_line, buf = buf.split(b"\n", 1)
                    line = raw_line.decode("utf-8", errors="replace").strip()
                    if line:
                        print(line, file=sys.stdout, flush=True)
                        if ready_check(line):
                            holder[0] = True
                            proc.terminate()
                            return

            if buf:
                line = buf.decode("utf-8", errors="replace").strip()
                if line:
                    print(line, file=sys.stdout, flush=True)
                    if ready_check(line):
                        holder[0] = True

            proc.terminate()
        except Exception as exc:
            log_warning(f"_stream_initial_logs error: {exc}")

    stream_thread = threading.Thread(target=_stream_logs)
    stream_thread.start()
    stream_thread.join(timeout_sec)

    if stream_thread.is_alive():
        stop_event.set()
        proc = proc_holder[0]
        if proc is not None:
            proc.terminate()
        stream_thread.join(timeout=5)

    return holder[0]


def wait_for_exit(cname: str, timeout_sec: float | None = None) -> int:
    """Wait for a container to exit and return its exit code.

    Raises :class:`TimeoutError` when *timeout_sec* elapses before the
    container exits — signalled out of band so a container that
    legitimately exits with code 124 (the ``timeout(1)`` convention)
    is returned as its real exit code rather than conflated with the
    wait timing out.

    Raises :class:`RuntimeError` when ``podman wait`` itself fails
    (non-zero returncode, e.g. unknown container) or returns output
    that is not a parseable integer — the podman diagnostic is never
    impersonated as the container's exit code.

    Raises :class:`FileNotFoundError` when ``podman`` is not on PATH;
    previously swallowed as the sentinel ``1``, which was
    indistinguishable from a container that really exited with 1.
    """
    try:
        proc = subprocess.run(
            ["podman", "wait", cname],
            check=False,
            capture_output=True,
            text=True,
            timeout=timeout_sec,
        )
    except subprocess.TimeoutExpired as exc:
        raise TimeoutError(f"container {cname!r} did not exit within {timeout_sec}s") from exc

    if proc.returncode != 0:
        detail = (proc.stderr or proc.stdout or "").strip() or "<no output>"
        raise RuntimeError(f"podman wait {cname!r} failed (returncode={proc.returncode}): {detail}")

    stdout = (proc.stdout or "").strip()
    try:
        return int(stdout)
    except ValueError as exc:
        raise RuntimeError(
            f"podman wait {cname!r} returned unexpected output: "
            f"stdout={proc.stdout!r}, stderr={proc.stderr!r}"
        ) from exc


def reserve_free_port(host: str = "127.0.0.1") -> tuple[socket.socket, int]:
    """Reserve a TCP port on *host* and return ``(socket, port)``.

    The socket stays open — the caller holds the reservation until they
    close it (typically right before binding the actual service).  Useful
    for Python-native servers that can accept a pre-bound socket.
    """
    s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    try:
        s.bind((host, 0))
        return s, s.getsockname()[1]
    except BaseException:
        s.close()
        raise


def find_free_port(host: str = "127.0.0.1") -> int:
    """Find and return a free TCP port on *host*.

    Releases the socket immediately — there is a small race window before
    the caller binds the port.  This is the standard approach when passing
    a port number to an external process (e.g. ``podman run -p``).
    """
    s, port = reserve_free_port(host)
    s.close()
    return port


# ---------------------------------------------------------------------------
# Bypass network args (when shield is completely skipped)
# ---------------------------------------------------------------------------

_SLIRP_GATEWAY = "10.0.2.2"
_PASTA_HOST_LOOPBACK_MAP = "169.254.1.2"


def _detect_rootless_network_mode() -> str:
    """Detect whether podman uses pasta or slirp4netns for rootless networking.

    Falls back to slirp4netns when the field is absent (podman < 4.4
    doesn't have ``RootlessNetworkCmd``) or when detection fails.
    slirp4netns is the safe default — it works on all podman versions.
    """
    try:
        out = subprocess.run(
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


def bypass_network_args(gate_port: int) -> list[str]:
    """Return podman network args for running without shield.

    Replicates the networking that terok-shield's OCI hook normally provides
    (allowing the container to reach ``host.containers.internal`` for the gate
    server) but without nftables rules, annotations, or cap-drops.

    This is a **dangerous fallback** for environments where shield can't run.
    All egress is unfiltered.
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
