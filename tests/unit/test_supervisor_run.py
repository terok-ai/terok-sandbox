# SPDX-FileCopyrightText: 2026 Jiri Vyskocil
# SPDX-License-Identifier: Apache-2.0

"""Tests for [`run_supervisor`][terok_sandbox.supervisor.main.run_supervisor].

All subservices (VerdictServer, ClearanceHub, VaultProxy, notifier)
are mocked — exercising the real start/stop chain would require D-Bus
+ varlink + aiohttp listeners and is integration-test territory.
The contract under test here is "compose the right service set in
the right order, then tear it down in reverse."
"""

from __future__ import annotations

import asyncio
import contextlib
import json
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from terok_sandbox.supervisor.main import (
    _install_signal_handlers,
    _Services,
    _wait_for_container,
    run_supervisor,
)


@pytest.fixture
def sidecar(tmp_path: Path) -> Path:
    """Write a minimal sidecar JSON and return its absolute path."""
    sidecar_dir = tmp_path / "sidecar"
    sidecar_dir.mkdir(parents=True)
    payload = {
        "container_name": "demo",
        "ipc_mode": "socket",
        "db_path": str(tmp_path / "vault.db"),
        "runtime_dir": str(tmp_path / "run"),
        "scope_id": "default",
        "socket_path": str(tmp_path / "vault.sock"),
    }
    sidecar_path = sidecar_dir / "demo.json"
    sidecar_path.write_text(json.dumps(payload))
    return sidecar_path


@pytest.mark.asyncio
async def test_missing_sidecar_bails_with_rc_2(tmp_path: Path) -> None:
    """A nonexistent sidecar path ⇒ rc 2 (no services started)."""
    rc = await run_supervisor("abc123", tmp_path / "missing.json")
    assert rc == 2


@pytest.mark.asyncio
async def test_runs_until_container_exits_and_unwinds(sidecar: Path) -> None:
    """Happy path: start all services, await podman wait, stop in reverse."""
    services = _Services()
    services.start = AsyncMock()  # type: ignore[method-assign]
    services.stop = AsyncMock()  # type: ignore[method-assign]

    async def _fake_wait(_container_id: str) -> int:
        return 0

    with (
        patch("terok_sandbox.supervisor.main._Services", return_value=services),
        patch("terok_sandbox.supervisor.main._wait_for_container", side_effect=_fake_wait),
        patch("terok_sandbox.supervisor.main._install_signal_handlers"),
    ):
        rc = await run_supervisor("abc123", sidecar)
    assert rc == 0
    services.start.assert_awaited_once()
    services.stop.assert_awaited_once()


@pytest.mark.asyncio
async def test_service_start_failure_returns_rc_3(sidecar: Path) -> None:
    """A failed ``start()`` short-circuits to rc 3 and runs ``stop()`` once."""
    services = _Services()
    services.start = AsyncMock(side_effect=RuntimeError("vault bind failed"))  # type: ignore[method-assign]
    services.stop = AsyncMock()  # type: ignore[method-assign]

    with (
        patch("terok_sandbox.supervisor.main._Services", return_value=services),
        patch("terok_sandbox.supervisor.main._install_signal_handlers"),
    ):
        rc = await run_supervisor("abc123", sidecar)
    assert rc == 3
    services.stop.assert_awaited_once()


class TestServicesStop:
    """Teardown swallows exceptions so a flaky service can't block others."""

    @pytest.mark.asyncio
    async def test_stop_swallows_subservice_failures(self) -> None:
        """An exception inside one service's ``stop()`` must not stop the chain."""
        services = _Services()
        # Capture references before ``stop()`` zeroes the attribute slots.
        subscriber = MagicMock(stop=AsyncMock(side_effect=RuntimeError("flaky")))
        hub = MagicMock(stop=AsyncMock())
        verdict = MagicMock(stop=AsyncMock())
        vault = MagicMock(stop=AsyncMock())
        notifier = MagicMock(disconnect=AsyncMock())
        services.subscriber = subscriber
        services.hub = hub
        services.verdict = verdict
        services.vault = vault
        services.notifier = notifier

        await services.stop()
        # Every service's stop must run despite the subscriber raising —
        # asserting the awaits *and* the slot nulls catches a regression
        # where ``stop()`` short-circuits or skips one branch.
        subscriber.stop.assert_awaited_once()
        hub.stop.assert_awaited_once()
        verdict.stop.assert_awaited_once()
        vault.stop.assert_awaited_once()
        notifier.disconnect.assert_awaited_once()
        assert services.subscriber is None
        assert services.hub is None
        assert services.verdict is None
        assert services.vault is None
        assert services.notifier is None


@pytest.mark.asyncio
async def test_hub_and_verdict_receive_selinux_socket_context(sidecar: Path) -> None:
    """Both clearance binds are handed ``socket_selinux_context`` so the
    sockets carry ``terok_socket_t`` for ``container_t connectto``.

    Without this the per-container sockets bind under the operator's
    domain (``unconfined_t``) and confined Podman containers are
    denied with the well-known ``avc: denied { connectto }`` AVC.
    """
    from terok_sandbox._util._selinux import socket_selinux_context

    captured: dict[str, object] = {}

    class _StubVerdictServer:
        def __init__(self, *, socket_path: Path, socket_context=None):  # noqa: ANN001
            captured["verdict_ctx"] = socket_context

        async def start(self) -> None:
            pass

        async def stop(self) -> None:
            pass

    class _StubClearanceHub:
        def __init__(
            self,
            *,
            clearance_socket: Path,
            reader_socket: Path,
            verdict_client,  # noqa: ANN001
            socket_context=None,  # noqa: ANN001
        ):
            captured["hub_ctx"] = socket_context
            captured["hub_clearance_socket"] = clearance_socket
            captured["hub_reader_socket"] = reader_socket

        async def start(self) -> None:
            pass

        async def stop(self) -> None:
            pass

    with (
        patch("terok_sandbox.integrations.clearance.VerdictServer", _StubVerdictServer),
        patch("terok_sandbox.integrations.clearance.ClearanceHub", _StubClearanceHub),
        patch(
            "terok_sandbox.integrations.clearance.VerdictClient",
            return_value=MagicMock(),
        ),
        patch(
            "terok_sandbox.integrations.clearance.create_notifier",
            return_value=MagicMock(connect=AsyncMock(), disconnect=AsyncMock()),
        ),
        patch("terok_sandbox.integrations.clearance.EventSubscriber"),
        patch("terok_sandbox.vault.daemon.token_broker.VaultProxy") as vault_cls,
        patch("terok_sandbox.vault.ssh.signer.start_ssh_signer", AsyncMock()),
        patch("terok_sandbox.supervisor.main._wait_for_container", AsyncMock(return_value=0)),
        patch("terok_sandbox.supervisor.main._install_signal_handlers"),
    ):
        vault_cls.return_value = MagicMock(start=AsyncMock(), stop=AsyncMock())
        await run_supervisor("abc123def456", sidecar)

    assert captured["verdict_ctx"] is socket_selinux_context
    assert captured["hub_ctx"] is socket_selinux_context
    # The reader/ingester socket must be distinct from the varlink
    # subscriber socket (operator UIs glob the latter) and live under
    # the dedicated ``events/`` dir the shield reader pushes to.
    assert captured["hub_reader_socket"] != captured["hub_clearance_socket"]
    assert captured["hub_reader_socket"].parent.name == "events"
    assert captured["hub_clearance_socket"].parent.name == "clearance"


class TestInstallSignalHandlers:
    """``_install_signal_handlers`` wires SIGTERM/SIGINT onto the running loop."""

    def test_no_running_loop_is_a_soft_noop(self) -> None:
        """Called outside a loop (sync context) it returns without raising.

        The function is import-safe for testing; with no running loop it
        leaves the caller to manage signals itself rather than crashing.
        """
        event = asyncio.Event()
        _install_signal_handlers(event)  # must not raise
        assert not event.is_set()

    @pytest.mark.asyncio
    async def test_registers_handlers_on_the_running_loop(self) -> None:
        """With a running loop, SIGTERM/SIGINT are registered as handlers.

        Patches ``loop.add_signal_handler`` to capture the registered
        signals rather than mutate the test process's real disposition.
        """
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


@pytest.fixture
def supervisor_paths(tmp_path: Path):
    """A ``SupervisorPaths`` bundle rooted under *tmp_path* for ``_Services.start``."""
    from terok_sandbox.supervisor.main import SupervisorPaths

    return SupervisorPaths.for_container(
        container_id="abc123def456789",
        container_name="demo",
        sidecar_path=tmp_path / "state" / "sidecar" / "demo.json",
        runtime_dir=tmp_path / "rt" / "sandbox",
    )


class _StubGate:
    """Records the kwargs ``_Services.start`` constructs ``GateServer`` with."""

    captured: dict[str, object]

    def __init__(self, **kw: object) -> None:
        type(self).captured = dict(kw)

    async def start(self) -> None:
        return None

    async def stop(self) -> None:
        return None


@contextlib.contextmanager
def _services_stubs():
    """Stub every non-gate service ``_Services.start`` brings up.

    The clearance trio, vault proxy, SSH signer, and notifier are
    replaced with trivial awaitables so ``start`` reaches the gate +
    bind-mode branches without binding a real listener.  ``GateServer``
    is swapped for [`_StubGate`][_StubGate] which records its kwargs.
    """
    with (
        patch(
            "terok_sandbox.integrations.clearance.VerdictServer",
            return_value=MagicMock(start=AsyncMock(), stop=AsyncMock()),
        ),
        patch(
            "terok_sandbox.integrations.clearance.ClearanceHub",
            return_value=MagicMock(start=AsyncMock(), stop=AsyncMock()),
        ),
        patch("terok_sandbox.integrations.clearance.VerdictClient", return_value=MagicMock()),
        patch(
            "terok_sandbox.integrations.clearance.create_notifier",
            new=AsyncMock(return_value=MagicMock(disconnect=AsyncMock())),
        ),
        patch(
            "terok_sandbox.integrations.clearance.EventSubscriber",
            return_value=MagicMock(start=AsyncMock(), stop=AsyncMock()),
        ),
        patch(
            "terok_sandbox.vault.daemon.token_broker.VaultProxy",
            return_value=MagicMock(start=AsyncMock(), stop=AsyncMock()),
        ),
        patch("terok_sandbox.vault.ssh.signer.start_ssh_signer", new=AsyncMock()),
        patch("terok_sandbox.gate.server.GateServer", _StubGate),
    ):
        yield


class TestServicesStartGateComposition:
    """``_Services.start`` composes the git gate only when both base path + token are set.

    The clearance trio, vault proxy, SSH signer, and notifier are stubbed
    to trivial awaitables — the contract under test is *which* gate
    constructor (socket vs TCP) the start path picks, and the TCP-mode
    guard rails that raise on a missing port.
    """

    @pytest.mark.asyncio
    async def test_socket_mode_gate_uses_socket_path(
        self, tmp_path: Path, supervisor_paths
    ) -> None:
        """Socket mode + wired gate → ``GateServer(socket_path=…)``, no host/port."""
        from terok_sandbox.supervisor.main import SidecarConfig, _Services

        cfg = SidecarConfig(
            container_name="demo",
            ipc_mode="socket",
            db_path=tmp_path / "vault.db",
            runtime_dir=tmp_path / "rt" / "sandbox",
            scope_id="proj",
            project_id="proj",
            gate_base_path=tmp_path / "mirrors",
            gate_token="terok-g-abc",
        )
        services = _Services()
        with _services_stubs():
            await services.start(cfg, supervisor_paths)

        captured = _StubGate.captured
        assert captured["mirror_root"] == cfg.gate_base_path
        assert captured["token"] == "terok-g-abc"
        assert captured["scope"] == "proj"
        assert captured["socket_path"] == supervisor_paths.gate_socket
        assert "port" not in captured

    @pytest.mark.asyncio
    async def test_tcp_mode_gate_uses_loopback_port(self, tmp_path: Path, supervisor_paths) -> None:
        """TCP mode + wired gate → ``GateServer(host=127.0.0.1, port=…)``."""
        from terok_sandbox.supervisor.main import SidecarConfig, _Services

        cfg = SidecarConfig(
            container_name="demo",
            ipc_mode="tcp",
            db_path=tmp_path / "vault.db",
            runtime_dir=tmp_path / "rt" / "sandbox",
            scope_id="proj",
            project_id="proj",
            tcp_port=22001,
            ssh_signer_port=22002,
            gate_port=22003,
            gate_base_path=tmp_path / "mirrors",
            gate_token="terok-g-abc",
        )
        services = _Services()
        with _services_stubs():
            await services.start(cfg, supervisor_paths)

        captured = _StubGate.captured
        assert captured["host"] == "127.0.0.1"
        assert captured["port"] == 22003
        assert "socket_path" not in captured

    @pytest.mark.asyncio
    async def test_tcp_mode_missing_gate_port_raises(
        self, tmp_path: Path, supervisor_paths
    ) -> None:
        """TCP mode with a wired gate but no allocated gate port is a hard error."""
        from terok_sandbox.supervisor.main import SidecarConfig, _Services

        cfg = SidecarConfig(
            container_name="demo",
            ipc_mode="tcp",
            db_path=tmp_path / "vault.db",
            runtime_dir=tmp_path / "rt" / "sandbox",
            project_id="proj",
            tcp_port=22001,
            ssh_signer_port=22002,
            gate_port=None,  # the bug this guards against
            gate_base_path=tmp_path / "mirrors",
            gate_token="terok-g-abc",
        )
        services = _Services()
        with _services_stubs(), pytest.raises(RuntimeError, match="gate_port"):
            await services.start(cfg, supervisor_paths)

    @pytest.mark.asyncio
    async def test_tcp_mode_missing_vault_port_raises(
        self, tmp_path: Path, supervisor_paths
    ) -> None:
        """TCP mode with no vault ``tcp_port`` raises before binding the proxy."""
        from terok_sandbox.supervisor.main import SidecarConfig, _Services

        cfg = SidecarConfig(
            container_name="demo",
            ipc_mode="tcp",
            db_path=tmp_path / "vault.db",
            runtime_dir=tmp_path / "rt" / "sandbox",
            tcp_port=None,  # the bug this guards against
        )
        services = _Services()
        with _services_stubs(), pytest.raises(RuntimeError, match="tcp_port"):
            await services.start(cfg, supervisor_paths)

    @pytest.mark.asyncio
    async def test_tcp_mode_missing_ssh_signer_port_raises(
        self, tmp_path: Path, supervisor_paths
    ) -> None:
        """TCP mode with no ``ssh_signer_port`` raises after the vault binds."""
        from terok_sandbox.supervisor.main import SidecarConfig, _Services

        cfg = SidecarConfig(
            container_name="demo",
            ipc_mode="tcp",
            db_path=tmp_path / "vault.db",
            runtime_dir=tmp_path / "rt" / "sandbox",
            tcp_port=22001,
            ssh_signer_port=None,  # the bug this guards against
        )
        services = _Services()
        with _services_stubs(), pytest.raises(RuntimeError, match="ssh_signer_port"):
            await services.start(cfg, supervisor_paths)


class TestWaitForContainer:
    """``_wait_for_container`` surfaces ``podman wait``'s exit code."""

    @pytest.mark.asyncio
    async def test_returns_parsed_exit_code(self) -> None:
        """A clean ``podman wait`` returns the integer printed on stdout."""
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
        """Garbage on stdout (no exit code) collapses to 0, not a crash."""
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
        """A failed ``podman wait`` invocation (container typo / podman crash)
        surfaces its own returncode as a soft failure, never blocking shutdown."""
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
        """A stop-signal cancellation terminates the lingering ``podman wait``.

        The supervisor cancels the wait task on shutdown; the handler must
        ``terminate()`` the subprocess (and re-raise ``CancelledError``) so a
        hung ``podman wait`` can't outlive the supervisor and pin the
        container ID.
        """
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
        """If the graceful terminate window elapses, the subprocess is killed."""
        proc = MagicMock()
        proc.communicate = AsyncMock(side_effect=asyncio.CancelledError)
        # First wait (graceful) times out; the kill-path wait then resolves.
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
