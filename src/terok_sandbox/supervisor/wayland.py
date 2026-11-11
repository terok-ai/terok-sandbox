# SPDX-FileCopyrightText: 2026 Jiri Vyskocil
# SPDX-License-Identifier: Apache-2.0

"""Supervise stock wl-mitm with an immutable policy and no credential access.

The Python service keeps the supervisor's process identity while the proxy
executes separately. Parent-death signalling covers abrupt wrapper death;
normal shutdown terminates and reaps the proxy. Filesystem confinement does
not restrict pathname Unix sockets or network access on every host kernel.
The optional SELinux domain supplies additional IPC restrictions.
"""

from __future__ import annotations

import asyncio
import contextlib
import ctypes
import fcntl
import os
import resource
import signal
from collections.abc import Iterator
from functools import partial
from pathlib import Path

from terok_util import confine_filesystem

from .._util._selinux import require_wayland_exec_context
from ..wayland import proxy_config
from .launcher import finish_output, relay_output
from .sidecar import SidecarConfig, SupervisorPaths

_LIBC = ctypes.CDLL(None, use_errno=True)
_PR_SET_PDEATHSIG = 1
_PROXY_TERM_GRACE = 2.0
_PROXY_FD_LIMIT = 1024
_PROXY_MEMORY_LIMIT = 1024 * 1024 * 1024
_PROC_FDS = Path("/proc/self/fd")
# Linux UAPI values: some supported CPython builds omit the fcntl seal names.
_F_ADD_SEALS = 1033
_POLICY_SEALS = 0x01 | 0x02 | 0x04 | 0x08
_PROXY_READ_ROOTS = tuple(
    Path(path) for path in ("/usr", "/lib", "/lib64", "/bin", "/nix/store", "/etc/ld.so.cache")
)


async def run_proxy(cfg: SidecarConfig, paths: SupervisorPaths, stop: asyncio.Event) -> None:
    """Run the task's optional proxy until it exits or the supervisor stops it."""
    session = cfg.wayland
    if not session.proxied:
        raise RuntimeError("Wayland child launched without proxy wiring")
    assert session.binary is not None and session.upstream is not None  # nosec B101 - narrowed by proxied
    with _policy_fd(proxy_config(session.upstream, paths.wayland_socket)) as config_fd:
        process = await asyncio.create_subprocess_exec(
            str(session.binary),
            str(_PROC_FDS / str(config_fd)),
            pass_fds=(config_fd,),
            close_fds=True,
            stdin=asyncio.subprocess.DEVNULL,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.STDOUT,
            env={"LANG": "C.UTF-8", "TOKIO_WORKER_THREADS": "2"},
            cwd=paths.wayland_socket.parent,
            preexec_fn=partial(
                _confine_proxy, os.getpid(), session.binary, paths.wayland_socket.parent
            ),
        )
        exited = asyncio.create_task(process.wait())
        stopping = asyncio.create_task(stop.wait())
        output = asyncio.create_task(relay_output(process, "wl-mitm"))
        try:
            done, _ = await asyncio.wait((exited, stopping), return_when=asyncio.FIRST_COMPLETED)
            if exited in done:
                raise RuntimeError(f"wl-mitm exited with status {exited.result()}")
        finally:
            stopping.cancel()
            if process.returncode is None:
                with contextlib.suppress(ProcessLookupError):
                    process.terminate()
                try:
                    await asyncio.wait_for(asyncio.shield(exited), _PROXY_TERM_GRACE)
                except TimeoutError:
                    with contextlib.suppress(ProcessLookupError):
                        process.kill()
            await exited
            await finish_output(output)
            with contextlib.suppress(asyncio.CancelledError):
                await stopping


@contextlib.contextmanager
def _policy_fd(policy: bytes) -> Iterator[int]:
    """Seal the policy in an anonymous file that no container can replace."""
    fd = os.memfd_create("terok-wayland-policy", os.MFD_CLOEXEC | os.MFD_ALLOW_SEALING)
    try:
        os.write(fd, policy)
        fcntl.fcntl(fd, _F_ADD_SEALS, _POLICY_SEALS)
        yield fd
    finally:
        os.close(fd)


def _confine_proxy(parent_pid: int, binary: Path, runtime_dir: Path) -> None:
    """Constrain the sole exec child, while the wrapper is still single-threaded.

    The wrapper already entered its SELinux domain. A same-domain exec keeps
    the parent-death signal; a late policy installation requires a restart.
    Neither the domain check nor confinement is retried unconfined.
    The signal is armed before comparing parent identity, including subreapers.
    """
    if _LIBC.prctl(_PR_SET_PDEATHSIG, signal.SIGKILL, 0, 0, 0) != 0:
        raise OSError(ctypes.get_errno(), "cannot arm proxy parent-death signal")
    if os.getppid() != parent_pid:
        raise RuntimeError("Wayland wrapper died before proxy startup")
    require_wayland_exec_context()
    resource.setrlimit(resource.RLIMIT_NOFILE, (_PROXY_FD_LIMIT, _PROXY_FD_LIMIT))
    resource.setrlimit(resource.RLIMIT_AS, (_PROXY_MEMORY_LIMIT, _PROXY_MEMORY_LIMIT))
    report = confine_filesystem((*_PROXY_READ_ROOTS, binary), (runtime_dir,))
    if not report.confined or report.partially_confined:
        raise RuntimeError(f"Wayland filesystem confinement unavailable: {report.reason}")
