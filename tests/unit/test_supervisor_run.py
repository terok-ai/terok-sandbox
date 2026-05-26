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

import json
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from terok_sandbox.supervisor.main import _Services, run_supervisor


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
        patch("terok_clearance.VerdictServer", _StubVerdictServer),
        patch("terok_clearance.ClearanceHub", _StubClearanceHub),
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
