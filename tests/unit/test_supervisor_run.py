# SPDX-FileCopyrightText: 2026 Jiri Vyskocil
# SPDX-License-Identifier: Apache-2.0

"""Tests for the parent supervisor — [`run_supervisor`][terok_sandbox.supervisor.main.run_supervisor].

The parent now supervises *processes*: it launches one child per
service through a [`ProcessLauncher`][terok_sandbox.supervisor.launcher.ProcessLauncher],
monitors them racing ``podman wait``, and terminates them in reverse
launch order.  These tests stub the launcher with fake child processes —
the contract is "launch the right service set, run until the container
exits, then tear the children down."  The per-service construction lives
in ``test_supervisor_children.py``; the spawn mechanics in
``test_supervisor_launcher.py``.
"""

from __future__ import annotations

import asyncio
import json
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from terok_sandbox.supervisor.launcher import ChildHandle
from terok_sandbox.supervisor.main import (
    _install_signal_handlers,
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


class _FakeLauncher:
    """Records launches; fails the services named in *fail*."""

    def __init__(self, *, fail: frozenset[str] = frozenset()) -> None:
        self.launched: list[str] = []
        self.handles: list[ChildHandle] = []
        self._fail = fail

    async def launch(self, service: str, container_id: str, sidecar_path: Path) -> ChildHandle:
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
    launcher = _FakeLauncher()
    with (
        patch("terok_sandbox.supervisor.launcher.default_launcher", return_value=launcher),
        patch("terok_sandbox.supervisor.main._wait_for_container", AsyncMock(return_value=0)),
        patch("terok_sandbox.supervisor.main._install_signal_handlers"),
    ):
        rc = await run_supervisor("abc123def456", sidecar)
    assert rc == 0
    assert launcher.launched == ["verdict", "clearance", "vault", "signer"]
    for handle in launcher.handles:
        handle.process.terminate.assert_called_once()


@pytest.mark.asyncio
async def test_one_failed_launch_costs_only_that_child(sidecar: Path) -> None:
    """A single service that won't spawn degrades to the rest still running."""
    launcher = _FakeLauncher(fail=frozenset({"vault"}))
    with (
        patch("terok_sandbox.supervisor.launcher.default_launcher", return_value=launcher),
        patch("terok_sandbox.supervisor.main._wait_for_container", AsyncMock(return_value=0)),
        patch("terok_sandbox.supervisor.main._install_signal_handlers"),
    ):
        rc = await run_supervisor("abc123def456", sidecar)
    assert rc == 0
    assert "vault" not in launcher.launched
    assert launcher.launched == ["verdict", "clearance", "signer"]


@pytest.mark.asyncio
async def test_no_child_launched_returns_rc_3(sidecar: Path) -> None:
    """Every launch failing ⇒ rc 3 so the wrapper's retry loop gets a go."""
    launcher = _FakeLauncher(fail=frozenset({"verdict", "clearance", "vault", "signer"}))
    with (
        patch("terok_sandbox.supervisor.launcher.default_launcher", return_value=launcher),
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
    launcher = _FakeLauncher()

    async def _never_returns(_cid: str) -> int:
        await asyncio.Event().wait()
        return 0

    async def _kill_children_soon() -> None:
        # Let ``_launch_children`` populate ``launcher.handles`` first,
        # then make every child exit so ``_await_all_children`` fires.
        await asyncio.sleep(0.05)
        for handle in launcher.handles:
            handle.process._finish(4)

    with (
        patch("terok_sandbox.supervisor.launcher.default_launcher", return_value=launcher),
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


class TestInstallSignalHandlers:
    """``_install_signal_handlers`` wires SIGTERM/SIGINT onto the running loop."""

    def test_no_running_loop_is_a_soft_noop(self) -> None:
        """Called outside a loop it returns without raising."""
        event = asyncio.Event()
        _install_signal_handlers(event)
        assert not event.is_set()

    @pytest.mark.asyncio
    async def test_registers_handlers_on_the_running_loop(self) -> None:
        """With a running loop, SIGTERM/SIGINT are registered as handlers."""
        import signal

        event = asyncio.Event()
        loop = asyncio.get_running_loop()
        registered: list[int] = []
        with patch.object(
            loop, "add_signal_handler", side_effect=lambda s, _cb: registered.append(s)
        ):
            _install_signal_handlers(event)
        assert signal.SIGTERM in registered
        assert signal.SIGINT in registered

    @pytest.mark.asyncio
    async def test_falls_back_to_signal_signal_when_unsupported(self) -> None:
        """Where the loop can't register handlers, it falls back to signal.signal."""
        import signal

        event = asyncio.Event()
        loop = asyncio.get_running_loop()
        fell_back: list[int] = []
        with (
            patch.object(loop, "add_signal_handler", side_effect=NotImplementedError),
            patch(
                "terok_sandbox.supervisor.main.signal.signal",
                side_effect=lambda s, _h: fell_back.append(s),
            ),
        ):
            _install_signal_handlers(event)
        assert signal.SIGTERM in fell_back and signal.SIGINT in fell_back


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
    async def test_nonzero_invocation_returns_its_returncode(self) -> None:
        proc = MagicMock()
        proc.communicate = AsyncMock(return_value=(b"", b"no such container"))
        proc.returncode = 125
        with patch(
            "terok_sandbox.supervisor.main.asyncio.create_subprocess_exec",
            AsyncMock(return_value=proc),
        ):
            assert await _wait_for_container("abc123") == 125

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
            patch(
                "terok_sandbox.supervisor.main.asyncio.wait_for",
                AsyncMock(side_effect=TimeoutError),
            ),
            pytest.raises(asyncio.CancelledError),
        ):
            await _wait_for_container("abc123")
        proc.terminate.assert_called_once()
        proc.kill.assert_called_once()
