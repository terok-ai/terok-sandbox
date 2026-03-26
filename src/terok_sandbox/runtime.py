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

from ._util import log_debug

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


def get_project_container_states(name_prefix: str) -> dict[str, str]:
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


def stop_task_containers(container_names: list[str]) -> None:
    """Best-effort ``podman rm -f`` of the given containers.

    Ignores all errors so that task deletion succeeds even when podman is
    absent or the containers are already gone.
    """
    for name in container_names:
        try:
            log_debug(f"stop_containers: podman rm -f {name} (start)")
            subprocess.run(
                ["podman", "rm", "-f", name],
                check=False,
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
                timeout=120,
            )
            log_debug(f"stop_containers: podman rm -f {name} (done)")
        except Exception:
            pass


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
                    log_debug(f"_stream_initial_logs read error: {exc}")
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
            log_debug(f"_stream_initial_logs error: {exc}")

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

    Returns 124 on timeout, 1 if podman is not found.
    """
    try:
        proc = subprocess.run(
            ["podman", "wait", cname],
            check=False,
            capture_output=True,
            timeout=timeout_sec,
        )
        stdout = proc.stdout.decode().strip() if isinstance(proc.stdout, bytes) else proc.stdout
        if stdout:
            return int(stdout)
        return proc.returncode
    except subprocess.TimeoutExpired:
        return 124
    except (FileNotFoundError, ValueError):
        return 1


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

_LOCALHOST = "127.0.0.1"
_SLIRP_GATEWAY = "10.0.2.2"


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
        f"pasta:-T,{gate_port}",
        "--add-host",
        f"host.containers.internal:{_LOCALHOST}",
    ]
