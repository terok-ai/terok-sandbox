# SPDX-FileCopyrightText: 2026 Jiri Vyskocil
# SPDX-License-Identifier: Apache-2.0

"""Coverage for the PodmanContainer / PodmanImage / PodmanLogStream /
PodmanPortReservation handle types.

These are the low-ish level pieces that the higher-level runtime and
executor call into; individual verb coverage lives here rather than in
the broader ``test_runtime_extras.py``.
"""

from __future__ import annotations

import subprocess
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from terok_sandbox import PodmanRuntime
from terok_sandbox.runtime.podman import (
    PodmanLogStream,
    PodmanPortReservation,
)

# ── Identity / equality ───────────────────────────────────────────────────


class TestContainerIdentity:
    """PodmanContainer eq / hash / repr."""

    def test_repr(self) -> None:
        """``repr`` shows the container name."""
        runtime = PodmanRuntime()
        assert repr(runtime.container("abc")) == "PodmanContainer(name='abc')"

    def test_eq_by_name(self) -> None:
        """Handles with the same name compare equal and share a hash."""
        runtime = PodmanRuntime()
        assert runtime.container("c") == runtime.container("c")
        assert hash(runtime.container("c")) == hash(runtime.container("c"))

    def test_ne_different_name(self) -> None:
        """Handles with different names compare unequal."""
        runtime = PodmanRuntime()
        assert runtime.container("a") != runtime.container("b")

    def test_ne_non_container(self) -> None:
        """Handles are not equal to arbitrary objects."""
        runtime = PodmanRuntime()
        assert runtime.container("a") != "a"


class TestImageIdentity:
    """PodmanImage eq / hash / repr."""

    def test_repr(self) -> None:
        """``repr`` shows the image ref."""
        runtime = PodmanRuntime()
        assert repr(runtime.image("sha256:abc")) == "PodmanImage(ref='sha256:abc')"

    def test_eq_by_ref(self) -> None:
        """Images with the same ref compare equal."""
        runtime = PodmanRuntime()
        assert runtime.image("sha256:abc") == runtime.image("sha256:abc")
        assert hash(runtime.image("sha256:abc")) == hash(runtime.image("sha256:abc"))

    def test_ne_non_image(self) -> None:
        """Images are not equal to arbitrary objects."""
        runtime = PodmanRuntime()
        assert runtime.image("ref") != "ref"


# ── Image.id ─────────────────────────────────────────────────────────────


class TestImageId:
    """``Image.id`` resolves through ``podman inspect``."""

    @patch("terok_sandbox.runtime.podman.subprocess.check_output", return_value="sha256:aaa\n")
    def test_resolved_id(self, _co) -> None:
        """A non-empty stripped id is returned."""
        assert PodmanRuntime().image("terok-l1-cli:test").id == "sha256:aaa"

    @patch("terok_sandbox.runtime.podman.subprocess.check_output", return_value="\n")
    def test_empty_id_is_none(self, _co) -> None:
        """Empty podman output → ``None``."""
        assert PodmanRuntime().image("gone").id is None

    @patch(
        "terok_sandbox.runtime.podman.subprocess.check_output",
        side_effect=subprocess.CalledProcessError(125, "podman"),
    )
    def test_missing_image_is_none(self, _co) -> None:
        """Podman error → ``None``."""
        assert PodmanRuntime().image("missing").id is None

    @patch("terok_sandbox.runtime.podman.subprocess.check_output", side_effect=FileNotFoundError)
    def test_no_podman_is_none(self, _co) -> None:
        """Missing podman → ``None``."""
        assert PodmanRuntime().image("ref").id is None


# ── Copy-in ──────────────────────────────────────────────────────────────


class TestContainerCopyIn:
    """``Container.copy_in`` wraps ``podman cp`` with directory awareness."""

    @patch("terok_sandbox.runtime.podman.subprocess.run")
    def test_directory_uses_dot_suffix(self, mock_run, tmp_path: Path) -> None:
        """Directory sources use the ``src/.`` form for content copy."""
        src = tmp_path / "cfg"
        src.mkdir()
        PodmanRuntime().container("ctr").copy_in(src, "/dest")
        cmd = mock_run.call_args[0][0]
        assert cmd == ["podman", "cp", f"{src}/.", "ctr:/dest"]

    @patch("terok_sandbox.runtime.podman.subprocess.run")
    def test_file_is_copied_directly(self, mock_run, tmp_path: Path) -> None:
        """File sources are copied directly — no ``/.`` suffix."""
        src = tmp_path / "prompt.txt"
        src.write_text("hi")
        PodmanRuntime().container("ctr").copy_in(src, "/dest/prompt.txt")
        cmd = mock_run.call_args[0][0]
        assert cmd == ["podman", "cp", str(src), "ctr:/dest/prompt.txt"]


# ── PodmanLogStream ───────────────────────────────────────────────────────


def _fake_popen(lines: list[bytes], *, returncode: int | None = 0) -> MagicMock:
    """Build a ``Popen``-compatible mock yielding *lines* from stdout."""
    proc = MagicMock()
    # readline returns each line once, then b"" to signal EOF
    proc.stdout = MagicMock()
    proc.stdout.readline.side_effect = [*lines, b""]
    proc.poll.return_value = returncode
    return proc


class TestPodmanLogStream:
    """Iterator + context manager + close behaviour."""

    @patch("terok_sandbox.runtime.podman.subprocess.Popen")
    def test_iterates_decoded_lines(self, mock_popen_cls) -> None:
        """Iteration yields decoded log lines, stripping trailing newline."""
        mock_popen_cls.return_value = _fake_popen([b"hello\n", b"world\n"])
        stream = PodmanLogStream("ctr", follow=False, tail=None)
        assert list(stream) == ["hello", "world"]

    @patch("terok_sandbox.runtime.podman.subprocess.Popen")
    def test_follow_and_tail_in_argv(self, mock_popen_cls) -> None:
        """Constructor wires ``-f`` and ``--tail`` into the podman argv."""
        mock_popen_cls.return_value = _fake_popen([])
        PodmanLogStream("ctr", follow=True, tail=50)
        cmd = mock_popen_cls.call_args[0][0]
        assert cmd == ["podman", "logs", "-f", "--tail", "50", "ctr"]

    @patch("terok_sandbox.runtime.podman.subprocess.Popen")
    def test_close_terminates_live_process(self, mock_popen_cls) -> None:
        """``close`` terminates a still-running child."""
        proc = _fake_popen([], returncode=None)
        mock_popen_cls.return_value = proc
        stream = PodmanLogStream("ctr", follow=True, tail=None)
        stream.close()
        proc.terminate.assert_called_once()
        proc.wait.assert_called()

    @patch("terok_sandbox.runtime.podman.subprocess.Popen")
    def test_close_noop_when_exited(self, mock_popen_cls) -> None:
        """``close`` is a no-op when the child has already exited."""
        proc = _fake_popen([], returncode=0)
        mock_popen_cls.return_value = proc
        stream = PodmanLogStream("ctr", follow=False, tail=None)
        stream.close()
        proc.terminate.assert_not_called()

    @patch("terok_sandbox.runtime.podman.subprocess.Popen")
    def test_context_manager(self, mock_popen_cls) -> None:
        """``with`` closes the stream on exit."""
        proc = _fake_popen([b"a\n"], returncode=None)
        mock_popen_cls.return_value = proc
        with PodmanLogStream("ctr", follow=True, tail=None) as stream:
            assert next(stream) == "a"
        proc.terminate.assert_called_once()

    @patch("terok_sandbox.runtime.podman.subprocess.Popen")
    def test_kill_when_terminate_times_out(self, mock_popen_cls) -> None:
        """``close`` kills the child if it doesn't terminate cleanly."""
        proc = _fake_popen([], returncode=None)
        proc.wait.side_effect = [subprocess.TimeoutExpired("podman", 2), None]
        mock_popen_cls.return_value = proc
        stream = PodmanLogStream("ctr", follow=True, tail=None)
        stream.close()
        proc.terminate.assert_called_once()
        proc.kill.assert_called_once()

    @patch("terok_sandbox.runtime.podman.subprocess.Popen")
    def test_close_releases_parent_side_fds(self, mock_popen_cls) -> None:
        """``close`` closes both stdout and stderr parent-side pipes."""
        proc = _fake_popen([], returncode=0)
        stderr = MagicMock()
        proc.stderr = stderr
        mock_popen_cls.return_value = proc
        stream = PodmanLogStream("ctr", follow=False, tail=None)
        stdout = proc.stdout
        stream.close()
        stdout.close.assert_called_once()
        stderr.close.assert_called_once()
        assert proc.stdout is None
        assert proc.stderr is None

    @patch("terok_sandbox.runtime.podman.subprocess.Popen")
    def test_close_swallows_stream_close_errors(self, mock_popen_cls) -> None:
        """``close`` tolerates already-closed or broken parent pipes."""
        proc = _fake_popen([], returncode=0)
        proc.stdout.close.side_effect = OSError("fd closed")
        proc.stderr = MagicMock()
        proc.stderr.close.side_effect = OSError("fd closed")
        mock_popen_cls.return_value = proc
        stream = PodmanLogStream("ctr", follow=False, tail=None)
        stream.close()  # must not raise
        assert proc.stdout is None
        assert proc.stderr is None

    @patch("terok_sandbox.runtime.podman.subprocess.Popen")
    def test_next_returns_stopiteration_when_no_stdout(self, mock_popen_cls) -> None:
        """Missing stdout terminates iteration immediately."""
        proc = MagicMock()
        proc.stdout = None
        mock_popen_cls.return_value = proc
        stream = PodmanLogStream("ctr", follow=False, tail=None)
        with pytest.raises(StopIteration):
            next(stream)


# ── PodmanPortReservation ────────────────────────────────────────────────


class TestPodmanPortReservation:
    """Bind / close / context-manager semantics."""

    def test_enter_returns_self(self) -> None:
        """Entering the context returns the same reservation."""
        reservation = PodmanPortReservation()
        try:
            with reservation as entered:
                assert entered is reservation
        finally:
            reservation.close()

    def test_exit_releases_port(self) -> None:
        """Exiting the context closes the backing socket."""
        reservation = PodmanPortReservation()
        original = reservation._socket
        assert original is not None
        reservation.close()
        assert reservation._socket is None


# ── Exec: empty argv guard ───────────────────────────────────────────────


class TestExecEmptyArgvGuard:
    """Empty exec argv raises ``ValueError`` early (no podman call)."""

    @patch("terok_sandbox.runtime.podman.subprocess.run")
    def test_empty_cmd_raises_value_error(self, mock_run) -> None:
        """Calling exec with ``[]`` raises and never touches subprocess."""
        runtime = PodmanRuntime()
        with pytest.raises(ValueError, match="exec argv must not be empty"):
            runtime.exec(runtime.container("ctr"), [])
        mock_run.assert_not_called()
