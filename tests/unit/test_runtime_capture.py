# SPDX-FileCopyrightText: 2026 Jiri Vyskocil
# SPDX-License-Identifier: Apache-2.0

"""Bound hostile transport replies using local subprocesses, without Podman."""

from __future__ import annotations

import subprocess
import sys
import time
from pathlib import Path

import pytest

from terok_sandbox.runtime import ExecOutputLimitError, ExecResult
from terok_sandbox.runtime._capture import run_captured

_BUDGET = 1024
_TIMEOUT = 5.0
_TWO_STREAMS = [sys.executable, "-c", "import os; os.write(1, b'abc'); os.write(2, b'def')"]


@pytest.fixture
def children(monkeypatch):
    """Record real local children so every outcome can assert cleanup."""
    popen = subprocess.Popen
    children = []

    def spawn(*args, **kwargs):
        child = popen(*args, **kwargs)
        children.append(child)
        return child

    monkeypatch.setattr("terok_sandbox.runtime._capture.subprocess.Popen", spawn)
    yield children
    for child in children:
        assert child.returncode is not None
        assert child.stdout.closed
        assert child.stderr.closed


def _alive(pid: int) -> bool:
    """Whether *pid* still runs; a zombie waiting for init counts as exited."""
    try:
        stat = Path(f"/proc/{pid}/stat").read_text()
    except FileNotFoundError:
        return False
    return stat.rsplit(")", 1)[1].split()[0] != "Z"


@pytest.mark.parametrize("timeout", [None, _TIMEOUT])
def test_small_reply_preserves_streams_and_exit_status(timeout, children) -> None:
    """Both streams and a nonzero exit remain available to the evaluator."""
    result = run_captured(
        [sys.executable, "-c", "import os; os.write(1, b'hello'); os.write(2, b'error'); exit(7)"],
        max_output_bytes=_BUDGET,
        timeout=timeout,
    )
    assert result == ExecResult(7, "hello", "error")


@pytest.mark.parametrize("fd", [1, 2])
def test_flood_is_stopped_at_the_byte_budget(fd, children) -> None:
    """A constant-memory producer on either stream cannot grow host capture."""
    with pytest.raises(ExecOutputLimitError, match=f"output exceeded {_BUDGET} bytes"):
        run_captured(
            [sys.executable, "-c", f"import os\nwhile True: os.write({fd}, b'x' * 8192)"],
            max_output_bytes=_BUDGET,
            timeout=_TIMEOUT,
        )


def test_budget_counts_both_streams_together(children) -> None:
    """Neither stream gets a separate full budget."""
    with pytest.raises(ExecOutputLimitError, match="output exceeded"):
        run_captured(_TWO_STREAMS, max_output_bytes=5, timeout=_TIMEOUT)


def test_exactly_the_budget_is_accepted(children) -> None:
    """A reply of precisely the budget is complete, not an overflow."""
    assert run_captured(_TWO_STREAMS, max_output_bytes=6, timeout=_TIMEOUT) == ExecResult(
        0, "abc", "def"
    )


def test_invalid_utf8_cannot_crash_diagnostics(children) -> None:
    """Decode only bounded output and replace invalid bytes in both streams."""
    result = run_captured(
        [sys.executable, "-c", "import os; os.write(1, b'\\xff'); os.write(2, b'\\xfe')"],
        max_output_bytes=2,
        timeout=_TIMEOUT,
    )
    assert result == ExecResult(0, "\ufffd", "\ufffd")


@pytest.mark.parametrize("close_pipes", [False, True])
def test_deadline_covers_pipes_and_process_wait(close_pipes, children) -> None:
    """Silence and premature pipe EOF cannot bypass the overall deadline."""
    close = "os.close(1); os.close(2); " if close_pipes else ""
    with pytest.raises(subprocess.TimeoutExpired):
        run_captured(
            [sys.executable, "-c", f"import os, time; {close}time.sleep(30)"],
            max_output_bytes=_BUDGET,
            timeout=0.2,
        )


def test_inherited_writer_does_not_outlive_the_capture(children, tmp_path) -> None:
    """A grandchild keeping its exited parent's pipes open is gone once capture fails."""
    pid_file = tmp_path / "grandchild.pid"
    script = (
        "import os, time\n"
        "if os.fork() == 0:\n"
        f"    open({str(pid_file)!r}, 'w').write(str(os.getpid()))\n"
        "    time.sleep(30)\n"
    )
    with pytest.raises(subprocess.TimeoutExpired):
        run_captured([sys.executable, "-c", script], max_output_bytes=_BUDGET, timeout=0.5)
    grandchild = int(pid_file.read_text())
    deadline = time.monotonic() + 2
    while _alive(grandchild) and time.monotonic() < deadline:
        time.sleep(0.01)
    assert not _alive(grandchild)


@pytest.mark.parametrize("budget", [0, -1])
def test_invalid_limit_does_not_spawn(budget, children) -> None:
    """A byte budget must be positive before a process is launched."""
    with pytest.raises(ValueError, match="must be positive"):
        run_captured([sys.executable, "-c", "pass"], max_output_bytes=budget, timeout=_TIMEOUT)
    assert not children
