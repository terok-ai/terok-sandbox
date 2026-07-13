# SPDX-FileCopyrightText: 2026 Jiri Vyskocil
# SPDX-License-Identifier: Apache-2.0

"""Tests for the per-service child runners in
[`terok_sandbox.supervisor.children`][terok_sandbox.supervisor.children].

Each runner builds one service, runs it until a stop event, then tears it
down.  The service classes are stubbed — the contract under test is
"construct the right service with the right per-container arguments, and
tear it down."  A pre-set stop event lets a runner reach teardown without
a real listener blocking.
"""

from __future__ import annotations

import asyncio
import json
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from terok_sandbox.supervisor.children import (
    SERVICE_NAMES,
    _ensure_socket_dirs,
    _run_clearance,
    _run_gate,
    _run_verdict,
    run_child,
)
from terok_sandbox.supervisor.main import SidecarConfig, SupervisorPaths


@pytest.fixture
def paths(tmp_path: Path) -> SupervisorPaths:
    """A per-container path bundle rooted under *tmp_path*."""
    return SupervisorPaths.for_container(
        container_id="abc123def456789",
        container_name="demo",
        sidecar_path=tmp_path / "state" / "sidecar" / "demo.json",
        runtime_dir=tmp_path / "rt" / "sandbox",
    )


def _socket_cfg(tmp_path: Path, **extra: object) -> SidecarConfig:
    return SidecarConfig(
        container_name="demo",
        ipc_mode="socket",
        db_path=tmp_path / "vault.db",
        runtime_dir=tmp_path / "rt" / "sandbox",
        scope_id="proj",
        project_id="proj",
        **extra,
    )


def _preset_stop() -> asyncio.Event:
    """A stop event already set, so a runner reaches teardown at once."""
    event = asyncio.Event()
    event.set()
    return event


def test_service_names_are_the_five_children() -> None:
    """The launch-ordered set is exactly the five services, secret-holders last."""
    assert SERVICE_NAMES == ("verdict", "clearance", "gate", "vault", "signer")


class TestSelinuxSocketContext:
    """Verdict + clearance binds carry ``socket_selinux_context`` (``terok_socket_t``).

    Without it the per-container sockets bind under the operator's domain
    and confined ``container_t`` Podman is denied with ``avc: denied {
    connectto }``.
    """

    @pytest.mark.asyncio
    async def test_verdict_bind_gets_selinux_context(
        self, tmp_path: Path, paths: SupervisorPaths
    ) -> None:
        from terok_sandbox._util._selinux import socket_selinux_context

        captured: dict[str, object] = {}

        class _StubVerdict:
            def __init__(self, *, socket_path: Path, socket_context: object = None) -> None:
                captured["ctx"] = socket_context

            async def start(self) -> None: ...
            async def stop(self) -> None: ...

        with patch("terok_sandbox.integrations.clearance.VerdictServer", _StubVerdict):
            await _run_verdict(_socket_cfg(tmp_path), paths, _preset_stop())
        assert captured["ctx"] is socket_selinux_context

    @pytest.mark.asyncio
    async def test_clearance_hub_context_and_distinct_sockets(
        self, tmp_path: Path, paths: SupervisorPaths
    ) -> None:
        from terok_sandbox._util._selinux import socket_selinux_context

        captured: dict[str, object] = {}

        class _StubHub:
            def __init__(
                self,
                *,
                clearance_socket: Path,
                reader_socket: Path,
                verdict_client: object,
                socket_context: object = None,
            ) -> None:
                captured.update(
                    ctx=socket_context,
                    clearance_socket=clearance_socket,
                    reader_socket=reader_socket,
                )

            async def start(self) -> None: ...
            async def stop(self) -> None: ...

        with (
            patch("terok_sandbox.integrations.clearance.ClearanceHub", _StubHub),
            patch("terok_sandbox.integrations.clearance.VerdictClient", return_value=MagicMock()),
            patch(
                "terok_sandbox.integrations.clearance.create_notifier",
                new=AsyncMock(return_value=MagicMock(disconnect=AsyncMock())),
            ),
            patch(
                "terok_sandbox.integrations.clearance.EventSubscriber",
                return_value=MagicMock(start=AsyncMock(), stop=AsyncMock()),
            ),
        ):
            await _run_clearance(_socket_cfg(tmp_path), paths, _preset_stop())
        assert captured["ctx"] is socket_selinux_context
        # The ingester socket must be distinct from the varlink subscriber
        # socket and live under the dedicated ``events/`` dir.
        assert captured["reader_socket"] != captured["clearance_socket"]
        assert captured["reader_socket"].parent.name == "events"  # type: ignore[union-attr]
        assert captured["clearance_socket"].parent.name == "clearance"  # type: ignore[union-attr]


class TestGateRunner:
    """``_run_gate`` picks the socket vs TCP constructor from the sidecar mode."""

    @pytest.mark.asyncio
    async def test_socket_mode_uses_socket_path(
        self, tmp_path: Path, paths: SupervisorPaths
    ) -> None:
        captured: dict[str, object] = {}

        class _StubGate:
            def __init__(self, **kw: object) -> None:
                captured.update(kw)

            async def start(self) -> None: ...
            async def stop(self) -> None: ...

        cfg = _socket_cfg(tmp_path, gate_base_path=tmp_path / "mirrors", gate_token="terok-g-abc")
        with patch("terok_sandbox.gate.server.GateServer", _StubGate):
            await _run_gate(cfg, paths, _preset_stop())
        assert captured["socket_path"] == paths.gate_socket
        assert captured["token"] == "terok-g-abc"
        assert "port" not in captured

    @pytest.mark.asyncio
    async def test_tcp_mode_uses_loopback_port(
        self, tmp_path: Path, paths: SupervisorPaths
    ) -> None:
        captured: dict[str, object] = {}

        class _StubGate:
            def __init__(self, **kw: object) -> None:
                captured.update(kw)

            async def start(self) -> None: ...
            async def stop(self) -> None: ...

        cfg = SidecarConfig(
            container_name="demo",
            ipc_mode="tcp",
            db_path=tmp_path / "vault.db",
            runtime_dir=tmp_path / "rt" / "sandbox",
            project_id="proj",
            gate_port=22003,
            gate_base_path=tmp_path / "mirrors",
            gate_token="terok-g-abc",
        )
        with patch("terok_sandbox.gate.server.GateServer", _StubGate):
            await _run_gate(cfg, paths, _preset_stop())
        assert captured["host"] == "127.0.0.1"
        assert captured["port"] == 22003
        assert "socket_path" not in captured

    @pytest.mark.asyncio
    async def test_tcp_mode_without_port_raises(
        self, tmp_path: Path, paths: SupervisorPaths
    ) -> None:
        """A wired gate in TCP mode with no allocated port fails its own start."""
        cfg = SidecarConfig(
            container_name="demo",
            ipc_mode="tcp",
            db_path=tmp_path / "vault.db",
            runtime_dir=tmp_path / "rt" / "sandbox",
            gate_port=None,
            gate_base_path=tmp_path / "mirrors",
            gate_token="terok-g-abc",
        )
        with pytest.raises(RuntimeError, match="gate_port"):
            await _run_gate(cfg, paths, _preset_stop())


class TestEnsureSocketDirs:
    """``_ensure_socket_dirs`` creates each service's socket parent at 0o700."""

    def test_creates_and_tightens_dirs(self, paths: SupervisorPaths) -> None:
        _ensure_socket_dirs("vault", paths)
        parent = paths.vault_socket.parent
        assert parent.is_dir()
        assert (parent.stat().st_mode & 0o777) == 0o700


class TestRunChildGuards:
    """``run_child`` bails cleanly on an unknown service or an unusable sidecar."""

    def test_unknown_service_returns_bad_sidecar_code(self, tmp_path: Path) -> None:
        assert run_child("bogus", "cid", tmp_path / "missing.json") == 2

    def test_missing_sidecar_returns_bad_sidecar_code(self, tmp_path: Path) -> None:
        assert run_child("vault", "cid", tmp_path / "missing.json") == 2

    def test_start_failure_returns_start_failed_code(self, tmp_path: Path) -> None:
        """A runner that raises during bring-up gives the parent exit code 4."""
        sidecar = tmp_path / "demo.json"
        sidecar.write_text(
            json.dumps(
                {
                    "container_name": "demo",
                    "ipc_mode": "socket",
                    "db_path": str(tmp_path / "vault.db"),
                    "runtime_dir": str(tmp_path / "rt"),
                }
            )
        )

        async def _boom(*_a: object, **_k: object) -> None:
            raise RuntimeError("bring-up failed")

        with patch.dict(
            "terok_sandbox.supervisor.children._RUNNERS", {"vault": _boom}, clear=False
        ):
            assert run_child("vault", "abc123def456", sidecar) == 4
