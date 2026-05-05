# SPDX-FileCopyrightText: 2026 Jiri Vyskocil
# SPDX-License-Identifier: Apache-2.0

"""Tests for :meth:`ContainerRuntime.exec_stdio` — bidirectional stdio bridge.

Covers the in-memory :class:`NullRuntime` script replay and the podman pump
helpers (``_pump_stream`` / ``_start_stdio_pumps``) without spinning up a
real container — those are exercised by the manual integration walk-through
documented in the ACP host-proxy plan.
"""

from __future__ import annotations

import io
import os
import subprocess
import threading
import time
from unittest.mock import MagicMock, patch

import pytest

from terok_sandbox import NullRuntime
from terok_sandbox.runtime.podman import (
    PodmanContainer,
    PodmanRuntime,
    _pump_stream,
    _start_stdio_pumps,
)


class TestNullRuntimeExecStdio:
    """Script-driven fake for ``exec_stdio``."""

    def test_no_script_returns_zero_and_records_call(self) -> None:
        """Without a registered script, the call records and exits 0."""
        rt = NullRuntime()
        rc = rt.exec_stdio(
            rt.container("c"),
            ["true"],
            stdin=io.BytesIO(b""),
            stdout=io.BytesIO(),
        )
        assert rc == 0
        assert rt._exec_stdio_calls == [("c", ("true",), {})]

    def test_write_step_emits_to_stdout(self) -> None:
        """A ``write`` step pushes bytes to the caller-supplied *stdout*."""
        rt = NullRuntime()
        rt.set_exec_stdio_script(
            "c",
            ("agent",),
            (("write", b'{"jsonrpc":"2.0"}\n'),),
        )
        out = io.BytesIO()
        rc = rt.exec_stdio(
            rt.container("c"),
            ["agent"],
            stdin=io.BytesIO(b""),
            stdout=out,
        )
        assert rc == 0
        assert out.getvalue() == b'{"jsonrpc":"2.0"}\n'

    def test_read_step_consumes_stdin_in_order(self) -> None:
        """``read`` steps assert what the caller pushes through *stdin*."""
        rt = NullRuntime()
        rt.set_exec_stdio_script(
            "c",
            ("agent",),
            (
                ("read", b"hello\n"),
                ("write", b"world\n"),
                ("read", b"bye\n"),
            ),
        )
        out = io.BytesIO()
        rc = rt.exec_stdio(
            rt.container("c"),
            ["agent"],
            stdin=io.BytesIO(b"hello\nbye\n"),
            stdout=out,
        )
        assert rc == 0
        assert out.getvalue() == b"world\n"

    def test_read_mismatch_raises(self) -> None:
        """Mismatched stdin bytes surface as ``AssertionError``."""
        rt = NullRuntime()
        rt.set_exec_stdio_script("c", ("agent",), (("read", b"expected"),))
        with pytest.raises(AssertionError):
            rt.exec_stdio(
                rt.container("c"),
                ["agent"],
                stdin=io.BytesIO(b"actual!!"),
                stdout=io.BytesIO(),
            )

    def test_exit_code_is_returned(self) -> None:
        """Custom ``exit_code`` propagates back from the script."""
        rt = NullRuntime()
        rt.set_exec_stdio_script("c", ("agent",), (), exit_code=42)
        rc = rt.exec_stdio(
            rt.container("c"),
            ["agent"],
            stdin=io.BytesIO(),
            stdout=io.BytesIO(),
        )
        assert rc == 42

    def test_env_is_recorded(self) -> None:
        """Per-call env reaches the call log so tests can assert on it."""
        rt = NullRuntime()
        rt.exec_stdio(
            rt.container("c"),
            ["agent"],
            stdin=io.BytesIO(),
            stdout=io.BytesIO(),
            env={"TEROK_PROBE": "1"},
        )
        assert rt._exec_stdio_calls[0][2] == {"TEROK_PROBE": "1"}


class TestPumpStreamHelper:
    """``_pump_stream`` is the byte-copy primitive behind the podman backend."""

    def test_copies_until_eof(self) -> None:
        """Copies every byte from *src* to *dst* and stops on EOF."""
        src = io.BytesIO(b"abcdef")
        dst = io.BytesIO()
        _pump_stream(src, dst, label="t")
        assert dst.getvalue() == b"abcdef"

    def test_handles_closed_destination(self) -> None:
        """Closed *dst* ends the pump cleanly without raising upward."""
        src = io.BytesIO(b"abc")
        dst = io.BytesIO()
        dst.close()
        # Should not raise — the pump logs and returns.
        _pump_stream(src, dst, label="t-closed")

    def test_stdin_pump_closes_dst_on_eof(self) -> None:
        """The stdin→child pump closes ``dst`` on EOF so the child sees EOF.

        Without this, an ACP wrapper that exits on stdin EOF would
        deadlock until ``_close_proc_streams`` runs after
        ``proc.wait``, defeating the whole point of bidirectional
        streaming.  The pump distinguishes itself by the
        ``stdin→child`` label.
        """
        from terok_sandbox.runtime.podman import _STDIN_PUMP_LABEL

        src = io.BytesIO(b"abc")
        dst = io.BytesIO()
        _pump_stream(src, dst, label=_STDIN_PUMP_LABEL)
        assert dst.closed, "stdin pump must close dst on EOF"

    def test_other_pumps_leave_dst_open(self) -> None:
        """Non-stdin pumps leave ``dst`` open — the caller may still want to write."""
        src = io.BytesIO(b"abc")
        dst = io.BytesIO()
        _pump_stream(src, dst, label="child→stdout")
        assert not dst.closed

    def test_buffered_reader_short_payload_does_not_stall(self) -> None:
        """A short payload on a ``BufferedReader`` arrives without waiting for
        the full chunk size.

        Regression for the ACP probe stalling at 8 s on every wrapper:
        ``proc.stdout`` from :class:`subprocess.Popen` is a
        :class:`io.BufferedReader`, and its ``read(n)`` blocks until
        ``n`` bytes *or* EOF — not "whatever's available".  A 200-byte
        JSON-RPC reply on a 64 KB chunk would never get pumped until the
        pipe closed.  The fix is to use ``read1`` on streams that have
        it; this test pins that behaviour with a real OS pipe wrapped
        as a ``BufferedReader``.
        """
        r_fd, w_fd = os.pipe()
        try:
            # Produce a small payload and *leave the writer open* — that's
            # the production shape (the agent is still alive after sending
            # its first frame).  A non-buffered ``os.fdopen`` here would
            # hide the bug; we want the buffered wrapper that Popen uses.
            os.write(w_fd, b"hello-world")
            src = open(r_fd, "rb", closefd=True)  # noqa: SIM115 — owned by the pump thread until EOF
            r_fd = -1  # ownership transferred
            dst = io.BytesIO()
            t = threading.Thread(target=_pump_stream, args=(src, dst), kwargs={"label": "buf"})
            t.start()
            # Without read1, the pump would block forever waiting for
            # 64 KB or EOF; with read1 the 11 bytes propagate immediately.
            for _ in range(50):
                if dst.getvalue() == b"hello-world":
                    break
                time.sleep(0.01)
            else:  # pragma: no cover — only fires when the pump regresses
                raise AssertionError(f"pump didn't deliver short payload; dst={dst.getvalue()!r}")
            os.close(w_fd)  # signal EOF so the pump can exit
            w_fd = -1
            t.join(timeout=2)
            assert not t.is_alive(), "pump didn't exit on EOF"
        finally:
            if r_fd != -1:
                os.close(r_fd)
            if w_fd != -1:
                os.close(w_fd)

    def test_concurrent_pumps_finish(self) -> None:
        """Two pumps in threads each finish independently.

        Asserting ``is_alive() is False`` after the join makes a stuck
        pump fail the test deterministically — bare ``join(timeout=...)``
        otherwise silently swallows hangs.
        """
        src1 = io.BytesIO(b"x" * 100)
        dst1 = io.BytesIO()
        src2 = io.BytesIO(b"y" * 100)
        dst2 = io.BytesIO()
        t1 = threading.Thread(target=_pump_stream, args=(src1, dst1), kwargs={"label": "1"})
        t2 = threading.Thread(target=_pump_stream, args=(src2, dst2), kwargs={"label": "2"})
        t1.start()
        t2.start()
        t1.join(timeout=5)
        t2.join(timeout=5)
        assert not t1.is_alive(), "pump 1 stuck — join timed out"
        assert not t2.is_alive(), "pump 2 stuck — join timed out"
        assert dst1.getvalue() == b"x" * 100
        assert dst2.getvalue() == b"y" * 100


class TestStartStdioPumpsHelper:
    """``_start_stdio_pumps`` wires up to the child's stdio attributes."""

    class _FakeProc:
        """Minimal stand-in for a ``Popen`` exposing only stdio attrs."""

        def __init__(
            self,
            *,
            stdin: io.BytesIO | None,
            stdout: io.BytesIO | None,
            stderr: io.BytesIO | None,
        ) -> None:
            self.stdin = stdin
            self.stdout = stdout
            self.stderr = stderr

    def test_skips_streams_that_are_none(self) -> None:
        """Spawns a thread only when both child and caller streams exist."""
        proc = self._FakeProc(
            stdin=io.BytesIO(),
            stdout=io.BytesIO(b"out"),
            stderr=None,
        )
        threads = _start_stdio_pumps(
            proc,  # type: ignore[arg-type]
            stdin=io.BytesIO(),
            stdout=io.BytesIO(),
            stderr=None,
        )
        for t in threads:
            t.join(timeout=2)
            assert not t.is_alive(), f"pump {t.name} stuck — join timed out"
        # stdin pump exits immediately on empty input; stdout pump consumes
        # and exits.  No stderr pump is created.
        assert len(threads) == 2

    def test_all_streams_create_three_pumps(self) -> None:
        """All three streams present → three pump threads running in parallel.

        Complements the skip-on-None test above so the full wiring path
        is covered: stderr is plumbed end-to-end the same way stdout is.
        """
        proc = self._FakeProc(
            stdin=io.BytesIO(),
            stdout=io.BytesIO(b"out"),
            stderr=io.BytesIO(b"err"),
        )
        out = io.BytesIO()
        err = io.BytesIO()
        threads = _start_stdio_pumps(
            proc,  # type: ignore[arg-type]
            stdin=io.BytesIO(),
            stdout=out,
            stderr=err,
        )
        for t in threads:
            t.join(timeout=2)
            assert not t.is_alive(), f"pump {t.name} stuck — join timed out"
        assert len(threads) == 3
        assert out.getvalue() == b"out"
        assert err.getvalue() == b"err"


class TestPodmanRuntimeExecStdio:
    """``PodmanRuntime.exec_stdio`` orchestrates Popen + pumps + cleanup.

    Mocks :class:`subprocess.Popen` so the test never touches ``podman`` —
    the goal is to cover the orchestration paths (argv build, timeout
    fallback, empty-cmd guard) without spinning up a real container.
    """

    @pytest.fixture
    def runtime(self) -> PodmanRuntime:
        return PodmanRuntime()

    @pytest.fixture
    def container(self, runtime: PodmanRuntime) -> PodmanContainer:
        return PodmanContainer("c-test", runtime=runtime)

    @pytest.fixture
    def proc(self) -> MagicMock:
        """Popen mock with stdio attrs the pump-spawn code reads.

        Tests that need a specific ``wait`` shape override
        ``proc.wait.return_value`` or ``.side_effect`` after the fact.
        """
        proc = MagicMock()
        proc.stdin = MagicMock()
        proc.stdout = MagicMock()
        proc.stderr = None
        return proc

    def test_empty_cmd_raises_value_error(
        self, runtime: PodmanRuntime, container: PodmanContainer
    ) -> None:
        """An empty argv is a programming error, surfaced before Popen."""
        with pytest.raises(ValueError, match="argv must not be empty"):
            runtime.exec_stdio(container, [], stdin=io.BytesIO(), stdout=io.BytesIO())

    def test_argv_includes_env_flags_and_container_name(
        self, runtime: PodmanRuntime, container: PodmanContainer, proc: MagicMock
    ) -> None:
        """``env`` becomes ``-e KEY=VAL`` pairs preceding the container name.

        Captures the argv handed to ``subprocess.Popen`` so future
        regressions on argv shape (e.g. flag ordering) get caught at
        unit level instead of in the manual integration walk-through.
        """
        proc.wait.return_value = 0
        with patch(
            "terok_sandbox.runtime.podman.subprocess.Popen", return_value=proc
        ) as popen_mock:
            rc = runtime.exec_stdio(
                container,
                ["agent", "--flag"],
                stdin=io.BytesIO(),
                stdout=io.BytesIO(),
                env={"X": "1"},
            )
        assert rc == 0
        argv = popen_mock.call_args.args[0]
        assert argv == ["podman", "exec", "-i", "-e", "X=1", "c-test", "agent", "--flag"]

    def test_timeout_terminates_then_kills_child(
        self, runtime: PodmanRuntime, container: PodmanContainer, proc: MagicMock
    ) -> None:
        """Timeout escalation fires terminate → wait → kill in order."""
        # Two ``TimeoutExpired``s in a row simulate a child that ignores
        # SIGTERM, forcing the kill fallback.
        proc.wait.side_effect = [
            subprocess.TimeoutExpired("podman", 1.0),
            subprocess.TimeoutExpired("podman", 2.0),
            0,
        ]
        with patch("terok_sandbox.runtime.podman.subprocess.Popen", return_value=proc):
            with pytest.raises(subprocess.TimeoutExpired):
                runtime.exec_stdio(
                    container,
                    ["agent"],
                    stdin=io.BytesIO(),
                    stdout=io.BytesIO(),
                    timeout=1.0,
                )
        proc.terminate.assert_called_once()
        proc.kill.assert_called_once()

    def test_returns_child_exit_code(
        self, runtime: PodmanRuntime, container: PodmanContainer, proc: MagicMock
    ) -> None:
        """The child's exit code propagates back as the return value."""
        proc.wait.return_value = 42
        with patch("terok_sandbox.runtime.podman.subprocess.Popen", return_value=proc):
            rc = runtime.exec_stdio(container, ["agent"], stdin=io.BytesIO(), stdout=io.BytesIO())
        assert rc == 42
