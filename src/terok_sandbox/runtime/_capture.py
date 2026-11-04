# SPDX-FileCopyrightText: 2026 Jiri Vyskocil
# SPDX-License-Identifier: Apache-2.0

"""Capture a command's reply within a byte budget and one deadline."""

from __future__ import annotations

import os
import selectors
import signal
import subprocess  # nosec B404 - runtime transport subprocesses
import time
from contextlib import suppress
from typing import BinaryIO, cast

from .protocol import ExecOutputLimitError, ExecResult

_READ_SIZE = 64 * 1024


def run_captured(
    argv: list[str],
    *,
    max_output_bytes: int,
    timeout: float | None,
    executable: str | None = None,
) -> ExecResult:
    """Run *argv* and return its reply, holding at most *max_output_bytes* of it.

    Overflow raises
    [`ExecOutputLimitError`][terok_sandbox.runtime.protocol.ExecOutputLimitError]
    rather than returning a partial reply.  One deadline covers the pipe
    reads and the process exit.  Failure closes the pipes and kills the
    local transport process group; it does not promise to stop processes
    inside a container or remote guest.
    """
    if max_output_bytes <= 0:
        raise ValueError("max_output_bytes must be positive")
    deadline = None if timeout is None else time.monotonic() + timeout

    def remaining() -> float | None:
        if deadline is None:
            return None
        left = deadline - time.monotonic()
        if left <= 0:
            raise subprocess.TimeoutExpired(argv, cast(float, timeout))
        return left

    outputs = (bytearray(), bytearray())
    with subprocess.Popen(  # nosec B603 - argv supplied by the runtime transport
        argv,
        executable=executable,
        stdin=subprocess.DEVNULL,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        bufsize=0,
        start_new_session=True,
    ) as proc:
        streams = (cast(BinaryIO, proc.stdout), cast(BinaryIO, proc.stderr))
        try:
            with selectors.DefaultSelector() as selector:
                for stream, output in zip(streams, outputs, strict=True):
                    selector.register(stream, selectors.EVENT_READ, output)
                size = 0
                while selector.get_map():
                    for key, _ in selector.select(remaining()):
                        chunk = os.read(key.fd, min(_READ_SIZE, max_output_bytes - size + 1))
                        if not chunk:
                            selector.unregister(key.fileobj)
                            continue
                        size += len(chunk)
                        if size > max_output_bytes:
                            raise ExecOutputLimitError(
                                f"Command output exceeded {max_output_bytes} bytes"
                            )
                        key.data.extend(chunk)
            exit_code = proc.wait(timeout=remaining())
        except BaseException:
            # No reader threads hold buffered locks; inherited writers cannot
            # delay pipe closure or keep the selector waiting beyond its deadline.
            for stream in streams:
                stream.close()
            with suppress(ProcessLookupError):
                os.killpg(proc.pid, signal.SIGKILL)
            proc.wait()
            raise
    return ExecResult(exit_code, *(output.decode("utf-8", errors="replace") for output in outputs))
