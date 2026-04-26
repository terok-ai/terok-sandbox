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
import threading

import pytest

from terok_sandbox import NullRuntime
from terok_sandbox.runtime.podman import _pump_stream, _start_stdio_pumps


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
