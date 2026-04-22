# SPDX-FileCopyrightText: 2026 Jiri Vyskocil
# SPDX-License-Identifier: Apache-2.0

"""Tests for the ``_stream_initial_logs`` threading helper and its reap path.

The function spawns a ``podman logs -f`` child and drives it from a
reader thread; every exit path must reap the child (no zombies) and
close its stdout pipe (no leaked FDs).
"""

from __future__ import annotations

import subprocess
from unittest.mock import MagicMock, patch

from terok_sandbox.runtime.podman import _reap_logs_proc, _stream_initial_logs


def _fake_popen_with_lines(
    lines: list[bytes],
    *,
    alive_at_end: bool = True,
) -> MagicMock:
    """Build a ``Popen`` mock that yields *lines* and has configurable liveness.

    ``stdout.read1`` returns each line once, then ``b""`` forever; that
    plus a patched ``select.select`` makes the reader loop act as if the
    child is dribbling output to a real pipe.  ``read`` (used on the
    drain path) returns all of the accumulated content in one shot.
    """
    proc = MagicMock()
    proc.stdout = MagicMock()
    proc.stdout.read1.side_effect = [*lines, b"", b"", b""]
    proc.stdout.read.return_value = b""
    proc.poll.return_value = None if alive_at_end else 0
    return proc


# ── _reap_logs_proc ──────────────────────────────────────────────────────


class TestReapLogsProc:
    """The cleanup helper — called from every exit path."""

    def test_none_is_noop(self) -> None:
        """Passing ``None`` is a no-op."""
        _reap_logs_proc(None)  # does not raise

    def test_alive_is_terminated_waited_closed(self) -> None:
        """Live child gets terminate → wait → stdout.close."""
        proc = MagicMock()
        proc.poll.return_value = None
        proc.stdout = MagicMock()
        _reap_logs_proc(proc)
        proc.terminate.assert_called_once()
        proc.wait.assert_called_once_with(timeout=2)
        proc.stdout.close.assert_called_once()

    def test_timeout_triggers_kill(self) -> None:
        """When wait times out, the child is killed and waited again."""
        proc = MagicMock()
        proc.poll.return_value = None
        proc.wait.side_effect = [subprocess.TimeoutExpired("podman", 2), None]
        proc.stdout = MagicMock()
        _reap_logs_proc(proc)
        proc.terminate.assert_called_once()
        proc.kill.assert_called_once()
        assert proc.wait.call_count == 2

    def test_exited_child_still_reaped(self) -> None:
        """A child that already exited is still ``wait``-ed to release zombie slot."""
        proc = MagicMock()
        proc.poll.return_value = 0  # already exited
        proc.stdout = MagicMock()
        _reap_logs_proc(proc)
        proc.terminate.assert_not_called()
        proc.kill.assert_not_called()
        proc.wait.assert_called_once_with()
        proc.stdout.close.assert_called_once()

    def test_stdout_close_failure_is_swallowed(self) -> None:
        """A broken ``stdout.close`` does not propagate."""
        proc = MagicMock()
        proc.poll.return_value = 0
        proc.stdout = MagicMock()
        proc.stdout.close.side_effect = OSError("fd already closed")
        _reap_logs_proc(proc)  # no raise

    def test_no_stdout_is_safe(self) -> None:
        """A child with no stdout pipe skips the close step cleanly."""
        proc = MagicMock()
        proc.poll.return_value = 0
        proc.stdout = None
        _reap_logs_proc(proc)
        proc.wait.assert_called_once_with()

    def test_second_call_is_a_noop(self) -> None:
        """Reap is idempotent — thread + main can both fire on slow CI hosts."""
        proc = MagicMock()
        proc.poll.return_value = 0
        proc.stdout = MagicMock()
        _reap_logs_proc(proc)
        _reap_logs_proc(proc)
        # ``wait`` + ``close`` ran exactly once across both calls.
        proc.wait.assert_called_once_with()
        proc.stdout.close.assert_called_once()

    def test_second_call_is_a_noop_concurrent(self) -> None:
        """Two threads racing through the guard must not double-reap."""
        import threading

        proc = MagicMock()
        proc.poll.return_value = 0
        proc.stdout = MagicMock()
        barrier = threading.Barrier(2)

        def _racer() -> None:
            barrier.wait()
            _reap_logs_proc(proc)

        threads = [threading.Thread(target=_racer) for _ in range(2)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()
        proc.wait.assert_called_once_with()
        proc.stdout.close.assert_called_once()


# ── _stream_initial_logs ─────────────────────────────────────────────────


class TestStreamInitialLogs:
    """End-to-end behaviour of the threaded log streamer."""

    @patch("terok_sandbox.runtime.podman.select.select", return_value=([object()], [], []))
    @patch("terok_sandbox.runtime.podman.subprocess.Popen")
    def test_ready_marker_returns_true_and_reaps(self, mock_popen_cls, _sel) -> None:
        """Seeing the marker yields ``True`` and reaps the child."""
        proc = _fake_popen_with_lines([b">> init complete\n"], alive_at_end=True)
        mock_popen_cls.return_value = proc

        assert _stream_initial_logs("ctr", 5.0, lambda line: "init complete" in line) is True

        # Reaped: terminate called once from _reap_logs_proc, stdout closed.
        proc.terminate.assert_called_once()
        proc.wait.assert_called()
        proc.stdout.close.assert_called_once()

    @patch("terok_sandbox.runtime.podman.select.select", return_value=([], [], []))
    @patch("terok_sandbox.runtime.podman.subprocess.Popen")
    def test_no_marker_ever_returns_false_and_reaps(self, mock_popen_cls, _sel) -> None:
        """A timeout with no marker yields ``False`` and still reaps the child."""
        proc = _fake_popen_with_lines([], alive_at_end=True)
        mock_popen_cls.return_value = proc

        assert _stream_initial_logs("ctr", 0.05, lambda _line: False) is False
        proc.terminate.assert_called()
        proc.stdout.close.assert_called_once()

    @patch("terok_sandbox.runtime.podman.select.select", return_value=([], [], []))
    @patch("terok_sandbox.runtime.podman.subprocess.Popen")
    def test_exited_child_is_still_reaped(self, mock_popen_cls, _sel) -> None:
        """If podman exits before the marker, the loop drains + reaps."""
        proc = _fake_popen_with_lines([], alive_at_end=False)
        mock_popen_cls.return_value = proc

        assert _stream_initial_logs("ctr", 1.0, lambda _line: False) is False
        # Already-exited: terminate skipped, but ``wait`` still called to
        # release the zombie slot and ``stdout.close`` is called.
        proc.terminate.assert_not_called()
        proc.wait.assert_called_once_with()
        proc.stdout.close.assert_called_once()

    @patch("terok_sandbox.runtime.podman.subprocess.Popen")
    def test_popen_raises_is_logged(self, mock_popen_cls) -> None:
        """If ``Popen`` raises, the helper logs and returns ``False``."""
        mock_popen_cls.side_effect = OSError("podman exploded")

        assert _stream_initial_logs("ctr", 1.0, lambda _line: True) is False
