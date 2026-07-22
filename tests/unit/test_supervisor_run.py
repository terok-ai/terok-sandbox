# SPDX-FileCopyrightText: 2026 Jiri Vyskocil
# SPDX-License-Identifier: Apache-2.0

"""Tests for the parent supervisor — [`run_supervisor`][terok_sandbox.supervisor.main.run_supervisor].

The parent now supervises *processes*: it launches one child per
service via [`launch_child`][terok_sandbox.supervisor.launcher.launch_child],
monitors them racing ``podman wait``, and terminates them in reverse
launch order.  These tests stub the spawn with fake child processes —
the contract is "launch the right service set, run until the container
exits, then tear the children down."  The per-service construction lives
in ``test_supervisor_children.py``; the spawn mechanics in
``test_supervisor_launcher.py``.
"""

from __future__ import annotations

import asyncio
import contextlib
import json
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from terok_sandbox.supervisor.launcher import ChildHandle
from terok_sandbox.supervisor.main import (
    _select_services,
    _terminate_children,
    _wait_for_container,
    run_supervisor,
)


@pytest.fixture
def sidecar(tmp_path: Path) -> Path:
    """Write a minimal (socket-mode, gate-less) sidecar and return its path."""
    sidecar_dir = tmp_path / "sidecar"
    sidecar_dir.mkdir(parents=True)
    payload = {
        "container_name": "demo",
        "ipc_mode": "socket",
        "db_path": str(tmp_path / "vault.db"),
        "runtime_dir": str(tmp_path / "run"),
        "scope_id": "default",
    }
    sidecar_path = sidecar_dir / "demo.json"
    sidecar_path.write_text(json.dumps(payload))
    return sidecar_path


class _FakeProc:
    """A stand-in for ``asyncio.subprocess.Process`` a child runs in.

    Stays alive (``wait`` blocks) until ``terminate``/``kill`` fires, so a
    test can model both the run-forever and the died-early cases.
    """

    def __init__(self, *, exit_code: int | None = None) -> None:
        self.pid = 4321
        self.returncode = exit_code
        self._exited = asyncio.Event()
        if exit_code is not None:
            self._exited.set()
        self.terminate = MagicMock(side_effect=lambda: self._finish(-15))
        self.kill = MagicMock(side_effect=lambda: self._finish(-9))

    def _finish(self, code: int) -> None:
        if self.returncode is None:
            self.returncode = code
        self._exited.set()

    async def wait(self) -> int:
        await self._exited.wait()
        return self.returncode or 0


class _FakeSpawner:
    """A ``launch_child`` stand-in: records launches, fails the services in *fail*."""

    def __init__(self, *, fail: frozenset[str] = frozenset()) -> None:
        self.launched: list[str] = []
        self.handles: list[ChildHandle] = []
        self._fail = fail

    async def __call__(self, service: str, container_id: str, sidecar_path: Path) -> ChildHandle:
        if service in self._fail:
            raise OSError(f"spawn failed for {service}")
        handle = ChildHandle(service=service, process=_FakeProc())
        self.launched.append(service)
        self.handles.append(handle)
        return handle


@pytest.mark.asyncio
async def test_missing_sidecar_bails_with_rc_2(tmp_path: Path) -> None:
    """A nonexistent sidecar path ⇒ rc 2 (nothing to launch)."""
    assert await run_supervisor("abc123", tmp_path / "missing.json") == 2


class TestSelectServices:
    """``_select_services`` launches the gate only when the sidecar wired it."""

    def test_gate_dropped_when_unwired(self) -> None:
        from terok_sandbox.supervisor.main import SidecarConfig

        cfg = SidecarConfig(
            container_name="demo",
            ipc_mode="socket",
            db_path=Path("/x/vault.db"),
            runtime_dir=Path("/run"),
        )
        selected = _select_services(cfg, ("verdict", "clearance", "gate", "vault", "signer"))
        assert selected == ("verdict", "clearance", "vault", "signer")

    def test_gate_kept_when_wired(self) -> None:
        from terok_sandbox.supervisor.main import SidecarConfig

        cfg = SidecarConfig(
            container_name="demo",
            ipc_mode="socket",
            db_path=Path("/x/vault.db"),
            runtime_dir=Path("/run"),
            gate_base_path=Path("/mirrors"),
            gate_token="terok-g-abc",
        )
        selected = _select_services(cfg, ("verdict", "clearance", "gate", "vault", "signer"))
        assert "gate" in selected


@pytest.mark.asyncio
async def test_runs_until_container_exits_and_terminates_children(sidecar: Path) -> None:
    """Happy path: launch the gate-less set, await ``podman wait``, SIGTERM all."""
    spawn = _FakeSpawner()
    with (
        patch("terok_sandbox.supervisor.main.launch_child", spawn),
        patch("terok_sandbox.supervisor.main._wait_for_container", AsyncMock(return_value=0)),
        patch("terok_sandbox.supervisor.main._install_signal_handlers"),
    ):
        rc = await run_supervisor("abc123def456", sidecar)
    assert rc == 0
    assert spawn.launched == ["verdict", "clearance", "vault", "signer"]
    for handle in spawn.handles:
        handle.process.terminate.assert_called_once()


@pytest.mark.asyncio
async def test_one_failed_launch_costs_only_that_child(sidecar: Path) -> None:
    """A single service that won't spawn degrades to the rest still running."""
    spawn = _FakeSpawner(fail=frozenset({"vault"}))
    with (
        patch("terok_sandbox.supervisor.main.launch_child", spawn),
        patch("terok_sandbox.supervisor.main._wait_for_container", AsyncMock(return_value=0)),
        patch("terok_sandbox.supervisor.main._install_signal_handlers"),
    ):
        rc = await run_supervisor("abc123def456", sidecar)
    assert rc == 0
    assert "vault" not in spawn.launched
    assert spawn.launched == ["verdict", "clearance", "signer"]


@pytest.mark.asyncio
async def test_no_child_launched_returns_rc_3(sidecar: Path) -> None:
    """Every launch failing ⇒ rc 3 so the wrapper's retry loop gets a go."""
    spawn = _FakeSpawner(fail=frozenset({"verdict", "clearance", "vault", "signer"}))
    with (
        patch("terok_sandbox.supervisor.main.launch_child", spawn),
        patch("terok_sandbox.supervisor.main._install_signal_handlers"),
    ):
        rc = await run_supervisor("abc123def456", sidecar)
    assert rc == 3


@pytest.mark.asyncio
async def test_all_children_dying_unblocks_the_supervisor(sidecar: Path) -> None:
    """When every child exits on its own, the parent stops instead of hanging.

    ``podman wait`` never returns here (patched to block forever); the
    ``all children exited`` arm of ``_supervise`` is what unblocks
    teardown.
    """
    spawn = _FakeSpawner()

    async def _never_returns(_cid: str) -> int:
        await asyncio.Event().wait()
        return 0

    async def _kill_children_soon() -> None:
        # Let ``_launch_children`` populate ``spawn.handles`` first,
        # then make every child exit so ``_await_all_children`` fires.
        await asyncio.sleep(0.05)
        for handle in spawn.handles:
            handle.process._finish(4)

    with (
        patch("terok_sandbox.supervisor.main.launch_child", spawn),
        patch("terok_sandbox.supervisor.main._wait_for_container", _never_returns),
        patch("terok_sandbox.supervisor.main._install_signal_handlers"),
    ):
        killer = asyncio.create_task(_kill_children_soon())
        rc = await asyncio.wait_for(run_supervisor("abc123def456", sidecar), timeout=5)
        await killer
    assert rc == 0


class TestTerminateChildren:
    """``_terminate_children`` SIGTERMs in reverse order, escalating to SIGKILL."""

    @pytest.mark.asyncio
    async def test_terminates_live_children_in_reverse(self) -> None:
        handles = [ChildHandle(service=f"s{i}", process=_FakeProc()) for i in range(3)]
        await _terminate_children(handles)
        for handle in handles:
            handle.process.terminate.assert_called_once()
            assert handle.process.returncode is not None

    @pytest.mark.asyncio
    async def test_already_exited_child_is_left_alone(self) -> None:
        dead = ChildHandle(service="s", process=_FakeProc(exit_code=0))
        await _terminate_children([dead])
        dead.process.terminate.assert_not_called()

    @pytest.mark.asyncio
    async def test_unresponsive_child_is_killed(self) -> None:
        """A child that ignores SIGTERM past the grace window is SIGKILLed.

        ``terminate`` is a no-op so the child never exits on its own; the
        real grace timeout (squeezed to near-zero) elapses and the kill
        path fires — where the default ``_FakeProc.kill`` finally lets
        ``wait`` return.
        """
        proc = _FakeProc()
        proc.terminate = MagicMock()  # swallow SIGTERM — never exits on its own
        handle = ChildHandle(service="s", process=proc)
        with patch("terok_sandbox.supervisor.main._CHILD_TERM_GRACE_S", 0.01):
            await _terminate_children([handle])
        proc.kill.assert_called_once()

    @pytest.mark.asyncio
    async def test_external_cancellation_kills_children_and_reraises(self) -> None:
        """Cancelling the teardown kills the children (no orphans) and re-raises.

        Distinct from the grace timeout: an outer cancellation must
        propagate, not be swallowed as if it were an internal timeout.
        """
        proc = _FakeProc()
        proc.terminate = MagicMock()  # never exits on its own → the wait blocks
        handle = ChildHandle(service="s", process=proc)
        task = asyncio.create_task(_terminate_children([handle]))
        await asyncio.sleep(0)  # let it SIGTERM and reach the shared wait
        task.cancel()
        with pytest.raises(asyncio.CancelledError):
            await task
        proc.kill.assert_called_once()

    @pytest.mark.asyncio
    async def test_shared_grace_window_not_per_child(self) -> None:
        """All children wait within one grace window, not len(handles) × grace."""
        procs = [_FakeProc() for _ in range(4)]
        for p in procs:
            p.terminate = MagicMock()  # none exit on their own
        handles = [ChildHandle(service=f"s{i}", process=p) for i, p in enumerate(procs)]
        with patch("terok_sandbox.supervisor.main._CHILD_TERM_GRACE_S", 0.05):
            # 4 children × 0.05s sequential would be 0.2s; one shared window is ~0.05s.
            await asyncio.wait_for(_terminate_children(handles), timeout=0.15)
        for p in procs:
            p.kill.assert_called_once()


class TestSupervise:
    """``_supervise`` races the monitors and logs (not raises) a monitor failure."""

    @pytest.mark.asyncio
    async def test_monitor_task_exception_is_logged_not_raised(self) -> None:
        from terok_sandbox.supervisor.main import _supervise

        handle = ChildHandle(service="s", process=_FakeProc())  # never exits on its own
        stop = asyncio.Event()
        # podman-wait arm raises a non-cancellation error → the await loop
        # must swallow+log it, and the pending monitors get cancelled.
        with patch(
            "terok_sandbox.supervisor.main._wait_for_container",
            AsyncMock(side_effect=ValueError("boom")),
        ):
            await _supervise("cid", [handle], stop)  # must return, not raise


class TestContainerPidWatch:
    """The direct container-init-PID watch — the podman-free death signal."""

    @pytest.mark.asyncio
    async def test_pid_watch_returns_when_the_process_exits(self) -> None:
        """A real short-lived process: the watch resolves once it's killed.

        Exercises the true ``pidfd_open`` path (or the poll fallback on an
        old kernel) against an actual PID, not a mock.
        """
        import sys

        from terok_sandbox.supervisor.main import _wait_for_container_pid

        proc = await asyncio.create_subprocess_exec(
            sys.executable, "-c", "import time; time.sleep(30)"
        )
        watch = asyncio.create_task(_wait_for_container_pid(proc.pid))
        await asyncio.sleep(0.1)
        assert not watch.done()  # still alive → still watching
        proc.terminate()
        await asyncio.wait_for(watch, timeout=5)  # exit unblocks the watch
        await proc.wait()

    @pytest.mark.asyncio
    async def test_pid_watch_returns_immediately_for_a_dead_pid(self) -> None:
        """A PID already gone at open time resolves at once (tear down now)."""
        from terok_sandbox.supervisor.main import _wait_for_container_pid

        # create=True: on kernels/libc without pidfd (musl, older glibc) the
        # attribute is absent, and a plain patch would fail to find it — the
        # production path already treats an AttributeError as "no pidfd".
        with patch(
            "terok_sandbox.supervisor.main.os.pidfd_open",
            side_effect=ProcessLookupError,
            create=True,
        ):
            await asyncio.wait_for(_wait_for_container_pid(999999), timeout=2)

    @pytest.mark.asyncio
    async def test_poll_stays_pending_on_permission_error(self) -> None:
        """EPERM means the process exists but is momentarily unsignalable — keep watching.

        The watch must NOT resolve (which would falsely tear the container
        down); only an actual exit (ProcessLookupError) ends it.
        """
        from terok_sandbox.supervisor.main import _poll_pid_exit

        with (
            patch(
                "terok_sandbox.supervisor.main.os.kill",
                side_effect=PermissionError("operation not permitted"),
            ),
            patch("terok_sandbox.supervisor.main._PID_POLL_INTERVAL_S", 0.01),
        ):
            watch = asyncio.create_task(_poll_pid_exit(4242))
            await asyncio.sleep(0.1)  # several poll cycles
            assert not watch.done()  # EPERM never resolves the watch
            watch.cancel()
            with contextlib.suppress(asyncio.CancelledError):
                await watch

    @pytest.mark.asyncio
    async def test_poll_fallback_used_when_pidfd_unavailable(self) -> None:
        """Without ``pidfd_open`` the watch degrades to a ``kill(pid, 0)`` poll."""
        import sys

        from terok_sandbox.supervisor.main import _wait_for_container_pid

        proc = await asyncio.create_subprocess_exec(
            sys.executable, "-c", "import time; time.sleep(30)"
        )
        with (
            patch(
                "terok_sandbox.supervisor.main.os.pidfd_open",
                side_effect=OSError("no pidfd"),
                create=True,  # absent on musl/older libc — see the dead-pid test
            ),
            patch("terok_sandbox.supervisor.main._PID_POLL_INTERVAL_S", 0.05),
        ):
            watch = asyncio.create_task(_wait_for_container_pid(proc.pid))
            await asyncio.sleep(0.1)
            assert not watch.done()
            proc.terminate()
            await asyncio.wait_for(watch, timeout=5)
        await proc.wait()

    @pytest.mark.asyncio
    async def test_supervise_tears_down_on_container_pid_exit(self) -> None:
        """The PID arm unblocks ``_supervise`` even when podman-wait never fires."""
        from terok_sandbox.supervisor.main import _supervise

        handle = ChildHandle(service="s", process=_FakeProc())  # never exits
        stop = asyncio.Event()
        hang = asyncio.Event()  # podman-wait + children arms both hang on this

        async def _hang(*_a: object) -> int:
            await hang.wait()
            return 0

        with (
            patch("terok_sandbox.supervisor.main._wait_for_container", _hang),
            patch("terok_sandbox.supervisor.main._await_all_children", _hang),
            patch(
                "terok_sandbox.supervisor.main._wait_for_container_pid",
                AsyncMock(return_value=None),
            ) as pid_watch,
        ):
            await asyncio.wait_for(_supervise("cid", [handle], stop, 4242), timeout=5)
        pid_watch.assert_awaited_once_with(4242)

    @pytest.mark.asyncio
    async def test_supervise_skips_pid_arm_when_absent(self) -> None:
        """No container_pid → no PID-watch task is created."""
        from terok_sandbox.supervisor.main import _supervise

        handle = ChildHandle(service="s", process=_FakeProc())
        stop = asyncio.Event()
        with (
            patch(
                "terok_sandbox.supervisor.main._wait_for_container",
                AsyncMock(return_value=0),
            ),
            patch(
                "terok_sandbox.supervisor.main._wait_for_container_pid",
                AsyncMock(return_value=None),
            ) as pid_watch,
        ):
            await _supervise("cid", [handle], stop, None)
        pid_watch.assert_not_awaited()


class TestKillChildren:
    """``_kill_children`` SIGKILLs live children and reaps them; skips the dead."""

    @pytest.mark.asyncio
    async def test_kills_and_reaps_live_skips_dead(self) -> None:
        from terok_sandbox.supervisor.main import _kill_children

        live = _FakeProc()
        dead = _FakeProc(exit_code=0)
        await _kill_children(
            [ChildHandle(service="live", process=live), ChildHandle(service="dead", process=dead)]
        )
        live.kill.assert_called_once()
        assert live.returncode is not None
        dead.kill.assert_not_called()


class TestWaitForContainer:
    """``_wait_for_container`` surfaces ``podman wait``'s exit code."""

    @pytest.mark.asyncio
    async def test_returns_parsed_exit_code(self) -> None:
        proc = MagicMock()
        proc.communicate = AsyncMock(return_value=(b"137\n", b""))
        proc.returncode = 0
        with patch(
            "terok_sandbox.supervisor.main.asyncio.create_subprocess_exec",
            AsyncMock(return_value=proc),
        ):
            assert await _wait_for_container("abc123") == 137

    @pytest.mark.asyncio
    async def test_non_integer_stdout_falls_back_to_zero(self) -> None:
        proc = MagicMock()
        proc.communicate = AsyncMock(return_value=(b"not-a-number\n", b""))
        proc.returncode = 0
        with patch(
            "terok_sandbox.supervisor.main.asyncio.create_subprocess_exec",
            AsyncMock(return_value=proc),
        ):
            assert await _wait_for_container("abc123") == 0

    @pytest.mark.asyncio
    async def test_failed_invocation_retries_instead_of_unblocking(self) -> None:
        """A failed ``podman wait`` *invocation* is a watcher failure, not a container exit.

        Regression: returning on it unblocked ``_supervise``'s teardown race,
        so a supervisor whose nested podman was broken (crun 0.17 hook env)
        self-terminated seconds after start and the wrapper — seeing a clean
        exit — never restarted it.
        """
        broken = MagicMock()
        broken.communicate = AsyncMock(return_value=(b"", b"mkdir /var/lib/containers: denied"))
        broken.returncode = 125
        healthy = MagicMock()
        healthy.communicate = AsyncMock(return_value=(b"0\n", b""))
        healthy.returncode = 0
        with (
            patch(
                "terok_sandbox.supervisor.main.asyncio.create_subprocess_exec",
                AsyncMock(side_effect=[broken, healthy]),
            ),
            patch("terok_sandbox.supervisor.main._PODMAN_WAIT_RETRY_S", 0),
        ):
            assert await _wait_for_container("abc123") == 0

    @pytest.mark.asyncio
    async def test_persistently_failing_invocation_never_unblocks(self) -> None:
        """The watcher parks (degraded) while ``podman wait`` keeps failing."""

        def _broken_proc(*_a, **_k):
            proc = MagicMock()
            proc.communicate = AsyncMock(return_value=(b"", b"permission denied"))
            proc.returncode = 125
            proc.terminate = MagicMock()
            proc.wait = AsyncMock(return_value=0)
            return proc

        with (
            patch(
                "terok_sandbox.supervisor.main.asyncio.create_subprocess_exec",
                AsyncMock(side_effect=_broken_proc),
            ),
            patch("terok_sandbox.supervisor.main._PODMAN_WAIT_RETRY_S", 0),
        ):
            task = asyncio.ensure_future(_wait_for_container("abc123"))
            await asyncio.sleep(0.05)
            assert not task.done()
            task.cancel()
            with pytest.raises(asyncio.CancelledError):
                await task

    @pytest.mark.asyncio
    async def test_cancellation_terminates_podman_wait(self) -> None:
        proc = MagicMock()
        proc.communicate = AsyncMock(side_effect=asyncio.CancelledError)
        proc.wait = AsyncMock(return_value=0)
        proc.terminate = MagicMock()
        with (
            patch(
                "terok_sandbox.supervisor.main.asyncio.create_subprocess_exec",
                AsyncMock(return_value=proc),
            ),
            pytest.raises(asyncio.CancelledError),
        ):
            await _wait_for_container("abc123")
        proc.terminate.assert_called_once()

    @pytest.mark.asyncio
    async def test_cancellation_kills_when_terminate_times_out(self) -> None:
        proc = MagicMock()
        proc.communicate = AsyncMock(side_effect=asyncio.CancelledError)
        proc.wait = AsyncMock(side_effect=[TimeoutError, 0])
        proc.terminate = MagicMock()
        proc.kill = MagicMock()
        with (
            patch(
                "terok_sandbox.supervisor.main.asyncio.create_subprocess_exec",
                AsyncMock(return_value=proc),
            ),
            pytest.raises(asyncio.CancelledError),
        ):
            await _wait_for_container("abc123")
        proc.terminate.assert_called_once()
        proc.kill.assert_called_once()
