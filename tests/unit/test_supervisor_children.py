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
import os
import shutil
import signal
import socket
import subprocess
import sys
import textwrap
import time
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from terok_sandbox.supervisor.children import (
    SERVICE_NAMES,
    _arm_parent_death_signal,
    _ensure_socket_dirs,
    _install_signal_handlers,
    _resolve_service_passphrase,
    _run_clearance,
    _run_gate,
    _run_signer,
    _run_vault,
    _run_verdict,
    _writable_paths,
    run_child,
)
from terok_sandbox.supervisor.main import SidecarConfig, SupervisorPaths
from tests.constants import LOCALHOST, SYSTEM_RUNTIME_ROOT


@pytest.fixture(autouse=True)
def _no_irreversible_self_restriction(monkeypatch: pytest.MonkeyPatch) -> None:
    """Keep ``run_child`` tests from irreversibly restricting the pytest process.

    Two of ``run_child``'s startup steps are process-wide and permanent:
    ``_arm_parent_death_signal`` would tie the pytest process's life to its
    parent, and ``confine_filesystem`` would Landlock-restrict every later
    test's filesystem access.  Both are stubbed to no-ops here; the dedicated
    ``TestArmParentDeathSignal`` cases bypass the first explicitly, and
    ``TestWritablePaths`` exercises the policy without applying it.
    """
    monkeypatch.setattr("terok_sandbox.supervisor.children._arm_parent_death_signal", lambda: True)
    from terok_util import LandlockReport

    monkeypatch.setattr(
        "terok_sandbox.supervisor.children.confine_filesystem",
        lambda _re, _rw: LandlockReport(confined=True, reason="stubbed in tests"),
    )
    monkeypatch.setattr(
        "terok_sandbox.supervisor.children._resolve_service_passphrase",
        lambda service, _cfg: "test-passphrase" if service in {"vault", "signer"} else None,
    )


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
        routes_path=tmp_path / "routes.json",
        scope_id="proj",
        project_id="proj",
        _resolved_passphrase="test-passphrase",
        **extra,
    )


def _tcp_cfg(tmp_path: Path, **extra: object) -> SidecarConfig:
    kw: dict[str, object] = {"tcp_port": 22001, "ssh_signer_port": 22002}
    kw.update(extra)  # let callers override the ports (e.g. set one to None)
    return SidecarConfig(
        container_name="demo",
        ipc_mode="tcp",
        db_path=tmp_path / "vault.db",
        runtime_dir=tmp_path / "rt" / "sandbox",
        routes_path=tmp_path / "routes.json",
        scope_id="proj",
        project_id="proj",
        _resolved_passphrase="test-passphrase",
        **kw,  # type: ignore[arg-type]
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
        assert captured["reader_socket"].name == "ingester.sock"  # type: ignore[union-attr]
        assert captured["reader_socket"].parent.parent.name == "events"  # type: ignore[union-attr]
        assert captured["clearance_socket"].name == "hub.sock"  # type: ignore[union-attr]
        assert captured["clearance_socket"].parent.parent.name == "clearance"  # type: ignore[union-attr]


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
        assert captured["home_path"] == paths.gate_socket.parent / "home"
        assert captured["hooks_path"] == paths.gate_socket.parent / "hooks"
        assert not (tmp_path / "mirrors" / ".terok-hooks").exists()
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
        stop = _preset_stop()
        with pytest.raises(RuntimeError, match="gate_port"):
            await _run_gate(cfg, paths, stop)

    @pytest.mark.asyncio
    async def test_unwired_gate_raises(self, tmp_path: Path, paths: SupervisorPaths) -> None:
        """The gate runner refuses to start without gate_base_path + gate_token."""
        cfg = _socket_cfg(tmp_path)
        stop = _preset_stop()
        with pytest.raises(RuntimeError, match="gate wiring"):
            await _run_gate(cfg, paths, stop)


class TestEnsureSocketDirs:
    """``_ensure_socket_dirs`` creates each service's socket parent at 0o700."""

    def test_creates_and_tightens_dirs(self, paths: SupervisorPaths) -> None:
        _ensure_socket_dirs("vault", paths)
        parent = paths.vault_socket.parent
        assert parent.is_dir()
        assert (parent.stat().st_mode & 0o777) == 0o700

    def test_secret_services_have_distinct_socket_parents(self, paths: SupervisorPaths) -> None:
        """A write grant for one listener cannot replace a sibling listener."""
        assert (
            len(
                {
                    paths.vault_socket.parent,
                    paths.ssh_signer_socket.parent,
                    paths.gate_socket.parent,
                }
            )
            == 3
        )

    def test_cross_package_sockets_have_per_container_parents(self, tmp_path: Path) -> None:
        """A clearance child cannot replace another container's event sockets."""
        first = SupervisorPaths.for_container(
            "abc123def456",
            "first",
            tmp_path / "state" / "sidecar" / "first.json",
            tmp_path / "runtime" / "sandbox",
        )
        second = SupervisorPaths.for_container(
            "def456abc123",
            "second",
            tmp_path / "state" / "sidecar" / "second.json",
            tmp_path / "runtime" / "sandbox",
        )
        assert first.clearance_socket.parent != second.clearance_socket.parent
        assert first.events_socket.parent != second.events_socket.parent


class TestArmParentDeathSignal:
    """The kernel dead-man's switch — armed best-effort, orphan check binding."""

    def test_arms_pdeathsig_and_reports_live_parent(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """prctl is invoked with PR_SET_PDEATHSIG/SIGTERM; a live parent → True."""
        calls: list[tuple] = []
        libc = MagicMock()
        libc.prctl = MagicMock(side_effect=lambda *a: calls.append(a) or 0)
        monkeypatch.setattr("terok_sandbox.supervisor.children.ctypes.CDLL", lambda *a, **k: libc)
        # Pin a live parent: in a container the pytest process can itself be
        # a direct child of pid 1, which the orphan check would (correctly)
        # read as "parent already gone".
        monkeypatch.setattr("terok_sandbox.supervisor.children.os.getppid", lambda: 1000)
        assert _arm_parent_death_signal() is True
        assert calls and calls[0][:2] == (1, signal.SIGTERM)

    def test_orphaned_at_startup_is_refused(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """ppid == 1 means the supervisor died before the switch was armed."""
        monkeypatch.setattr("terok_sandbox.supervisor.children.ctypes.CDLL", MagicMock())
        monkeypatch.setattr("terok_sandbox.supervisor.children.os.getppid", lambda: 1)
        assert _arm_parent_death_signal() is False

    def test_missing_prctl_is_best_effort(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """A libc without prctl (non-Linux) degrades silently — reaps still cover."""
        monkeypatch.setattr(
            "terok_sandbox.supervisor.children.ctypes.CDLL",
            MagicMock(side_effect=OSError("no libc")),
        )
        # Pin a live parent — see test_arms_pdeathsig_and_reports_live_parent.
        monkeypatch.setattr("terok_sandbox.supervisor.children.os.getppid", lambda: 1000)
        assert _arm_parent_death_signal() is True

    def test_run_child_exits_when_orphaned(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """An already-orphaned child never starts its service."""
        monkeypatch.setattr(
            "terok_sandbox.supervisor.children._arm_parent_death_signal", lambda: False
        )
        assert run_child("vault", "cid", tmp_path / "missing.json") == 4


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

    def test_passphrase_resolution_failure_returns_start_failed_code(self, tmp_path: Path) -> None:
        """A secret-holder that cannot unlock its DB fails before confinement."""
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

        with patch(
            "terok_sandbox.supervisor.children._resolve_service_passphrase",
            side_effect=RuntimeError("unavailable"),
        ):
            assert run_child("vault", "abc123def456", sidecar) == 4

    def test_happy_path_runs_the_service_and_returns_ok(self, tmp_path: Path) -> None:
        """A clean run hardens, binds the socket dir, runs the service, exits 0.

        Drives ``run_child`` end-to-end with a no-op runner, so the real
        ``harden_self`` (partial in this rootless container — exercising the
        debug branch), ``load_sidecar``, socket-dir setup, and signal wiring
        all run.  ``runtime_dir`` stays under ``tmp_path`` for isolation.
        """
        sidecar = tmp_path / "demo.json"
        sidecar.write_text(
            json.dumps(
                {
                    "container_name": "demo",
                    "ipc_mode": "socket",
                    "db_path": str(tmp_path / "vault.db"),
                    "runtime_dir": str(tmp_path / "rt" / "sandbox"),
                }
            )
        )
        ran: list[bool] = []

        async def _noop(cfg: object, paths: object, stop: object) -> None:
            ran.append(True)

        with patch.dict(
            "terok_sandbox.supervisor.children._RUNNERS", {"vault": _noop}, clear=False
        ):
            assert run_child("vault", "abc123def456", sidecar) == 0
        assert ran == [True]
        # the vault socket dir was created + tightened
        vault_dir = tmp_path / "rt" / "sandbox" / "run" / "demo" / "vault"
        assert vault_dir.is_dir()
        assert (vault_dir.stat().st_mode & 0o777) == 0o700

    def test_partial_hardening_is_logged_not_fatal(self, tmp_path: Path) -> None:
        """A partial harden (e.g. mlockall denied) logs at debug and still runs."""
        from terok_util import HardeningReport

        sidecar = tmp_path / "demo.json"
        sidecar.write_text(
            json.dumps(
                {
                    "container_name": "demo",
                    "ipc_mode": "socket",
                    "db_path": str(tmp_path / "vault.db"),
                    "runtime_dir": str(tmp_path / "rt" / "sandbox"),
                }
            )
        )

        async def _noop(cfg: object, paths: object, stop: object) -> None: ...

        partial = HardeningReport(
            no_dump=True, no_core=True, memory_locked=False, no_new_privs=True
        )
        with (
            patch("terok_sandbox.supervisor.children.harden_self", return_value=partial),
            patch.dict("terok_sandbox.supervisor.children._RUNNERS", {"vault": _noop}, clear=False),
        ):
            assert run_child("vault", "abc123def456", sidecar) == 0

    def test_debug_mode_passes_allow_debugger_to_harden(self, tmp_path: Path) -> None:
        """A debug-mode sidecar makes the child harden with allow_debugger=True."""
        from terok_util import HardeningReport

        sidecar = tmp_path / "demo.json"
        sidecar.write_text(
            json.dumps(
                {
                    "container_name": "demo",
                    "ipc_mode": "socket",
                    "db_path": str(tmp_path / "vault.db"),
                    "runtime_dir": str(tmp_path / "rt" / "sandbox"),
                    "allow_debugger": True,
                }
            )
        )
        captured: dict[str, bool] = {}

        def _fake_harden(*, allow_debugger: bool = False) -> HardeningReport:
            captured["allow_debugger"] = allow_debugger
            return HardeningReport(
                no_dump=not allow_debugger,
                no_core=True,
                memory_locked=True,
                no_new_privs=not allow_debugger,
            )

        async def _noop(cfg: object, paths: object, stop: object) -> None: ...

        with (
            patch("terok_sandbox.supervisor.children.harden_self", side_effect=_fake_harden),
            patch.dict("terok_sandbox.supervisor.children._RUNNERS", {"vault": _noop}, clear=False),
        ):
            assert run_child("vault", "abc123def456", sidecar) == 0
        assert captured["allow_debugger"] is True


class TestServicePassphraseResolution:
    """Secret holders resolve the configured tier before installing Landlock."""

    def test_non_secret_service_needs_no_passphrase(self, tmp_path: Path) -> None:
        assert _resolve_service_passphrase("gate", _socket_cfg(tmp_path)) is None

    def test_vault_resolves_captured_policy(self, tmp_path: Path) -> None:
        cfg = _socket_cfg(
            tmp_path,
            credentials_use_keyring=True,
            credentials_passphrase_command="secret-helper",
        )
        with patch(
            "terok_sandbox.vault.store.encryption.resolve_passphrase_with_source",
            return_value=("resolved-secret", "command"),
        ) as resolve:
            assert _resolve_service_passphrase("vault", cfg) == "resolved-secret"

        resolve.assert_called_once_with(
            credentials_db=cfg.db_path,
            systemd_creds_file=tmp_path / "vault.passphrase.cred",
            use_keyring=True,
            passphrase_command="secret-helper",
        )

    def test_missing_passphrase_raises(self, tmp_path: Path) -> None:
        from terok_sandbox.vault.store.encryption import NoPassphraseError

        cfg = _socket_cfg(tmp_path)
        with patch(
            "terok_sandbox.vault.store.encryption.resolve_passphrase_with_source",
            return_value=(None, None),
        ):
            with pytest.raises(NoPassphraseError, match="no SQLCipher passphrase"):
                _resolve_service_passphrase("signer", cfg)


class TestVaultRunner:
    """``_run_vault`` builds the proxy with the sidecar transport, then tears it down."""

    @pytest.mark.asyncio
    async def test_socket_mode_builds_and_stops_proxy(
        self, tmp_path: Path, paths: SupervisorPaths
    ) -> None:
        captured: dict[str, object] = {}

        class _StubVault:
            def __init__(self, **kw: object) -> None:
                captured.update(kw)

            async def start(self) -> None: ...
            async def stop(self) -> None:
                captured["stopped"] = True

        with patch("terok_sandbox.vault.daemon.token_broker.VaultProxy", _StubVault):
            await _run_vault(_socket_cfg(tmp_path), paths, _preset_stop())
        assert captured["db_path"] == (tmp_path / "vault.db")
        assert captured["scope_id"] == "proj"
        assert captured["routes_path"] == tmp_path / "routes.json"
        assert captured["passphrase"] == "test-passphrase"
        assert captured["stopped"] is True

    @pytest.mark.asyncio
    async def test_tcp_mode_binds_loopback_port(
        self, tmp_path: Path, paths: SupervisorPaths
    ) -> None:
        from terok_sandbox.vault.daemon.token_broker import TcpBind

        captured: dict[str, object] = {}

        class _StubVault:
            def __init__(self, **kw: object) -> None:
                captured.update(kw)

            async def start(self) -> None: ...
            async def stop(self) -> None: ...

        with patch("terok_sandbox.vault.daemon.token_broker.VaultProxy", _StubVault):
            await _run_vault(_tcp_cfg(tmp_path), paths, _preset_stop())
        assert isinstance(captured["bind"], TcpBind)
        assert captured["bind"].port == 22001  # type: ignore[attr-defined]

    @pytest.mark.asyncio
    async def test_tcp_mode_without_port_raises(
        self, tmp_path: Path, paths: SupervisorPaths
    ) -> None:
        cfg = _tcp_cfg(tmp_path, tcp_port=None)
        stop = _preset_stop()
        with pytest.raises(RuntimeError, match="tcp_port"):
            await _run_vault(cfg, paths, stop)


class TestSignerRunner:
    """``_run_signer`` starts the SSH-agent server and closes it on stop."""

    @pytest.mark.asyncio
    async def test_socket_mode_starts_and_closes_server(
        self, tmp_path: Path, paths: SupervisorPaths
    ) -> None:
        server = MagicMock(close=MagicMock(), wait_closed=AsyncMock())
        with patch(
            "terok_sandbox.vault.ssh.signer.start_ssh_signer",
            new=AsyncMock(return_value=server),
        ) as start:
            await _run_signer(_socket_cfg(tmp_path), paths, _preset_stop())
        start.assert_awaited_once()
        assert start.await_args.kwargs["passphrase"] == "test-passphrase"
        server.close.assert_called_once()
        server.wait_closed.assert_awaited_once()

    @pytest.mark.asyncio
    async def test_tcp_mode_passes_loopback_port(
        self, tmp_path: Path, paths: SupervisorPaths
    ) -> None:
        server = MagicMock(close=MagicMock(), wait_closed=AsyncMock())
        with patch(
            "terok_sandbox.vault.ssh.signer.start_ssh_signer",
            new=AsyncMock(return_value=server),
        ) as start:
            await _run_signer(_tcp_cfg(tmp_path), paths, _preset_stop())
        assert start.await_args.kwargs["host"] == "127.0.0.1"
        assert start.await_args.kwargs["port"] == 22002

    @pytest.mark.asyncio
    async def test_tcp_mode_without_port_raises(
        self, tmp_path: Path, paths: SupervisorPaths
    ) -> None:
        cfg = _tcp_cfg(tmp_path, ssh_signer_port=None)
        stop = _preset_stop()
        with pytest.raises(RuntimeError, match="ssh_signer_port"):
            await _run_signer(cfg, paths, stop)


class TestChildSignalHandlers:
    """``_install_signal_handlers`` wires SIGTERM/SIGINT onto the running loop."""

    def test_no_running_loop_is_a_soft_noop(self) -> None:
        stop = asyncio.Event()
        _install_signal_handlers(stop)
        assert not stop.is_set()

    @pytest.mark.asyncio
    async def test_registers_handlers_on_the_running_loop(self) -> None:
        import signal

        stop = asyncio.Event()
        loop = asyncio.get_running_loop()
        registered: list[int] = []
        with patch.object(
            loop, "add_signal_handler", side_effect=lambda s, _cb: registered.append(s)
        ):
            _install_signal_handlers(stop)
        assert signal.SIGTERM in registered
        assert signal.SIGINT in registered


class TestWritablePaths:
    """The per-service write policy Landlock enforces — everything else is denied."""

    def test_socket_services_get_only_their_own_socket_directories(
        self, tmp_path: Path, paths: SupervisorPaths
    ) -> None:
        cfg = _socket_cfg(tmp_path)
        assert _writable_paths("verdict", cfg, paths) == [paths.verdict_socket.parent]
        assert _writable_paths("clearance", cfg, paths) == [
            paths.clearance_socket.parent,
            paths.events_socket.parent,
        ]

    def test_secret_holders_add_db_and_vault_adds_its_lock_lane(
        self, tmp_path: Path, paths: SupervisorPaths
    ) -> None:
        cfg = _socket_cfg(tmp_path)  # db_path = tmp_path / "vault.db"
        assert _writable_paths("signer", cfg, paths) == [
            paths.ssh_signer_socket.parent,
            tmp_path,
        ]
        assert _writable_paths("vault", cfg, paths) == [
            paths.vault_socket.parent,
            tmp_path,
            cfg.runtime_dir / "terok" / "vault" / "locks",
        ]

    def test_gate_adds_only_its_scoped_repo_and_dev_null(
        self, tmp_path: Path, paths: SupervisorPaths
    ) -> None:
        mirror = tmp_path / "gate"
        cfg = _socket_cfg(tmp_path, gate_base_path=mirror, gate_token="tok")  # nosec B106
        assert _writable_paths("gate", cfg, paths) == [
            paths.gate_socket.parent,
            mirror / "proj.git",
            Path(os.devnull),
        ]

    def test_unwired_gate_still_gets_private_runtime_and_dev_null(
        self, tmp_path: Path, paths: SupervisorPaths
    ) -> None:
        assert _writable_paths("gate", _socket_cfg(tmp_path), paths) == [
            paths.gate_socket.parent,
            Path(os.devnull),
        ]


class TestConfinementWiring:
    """``run_child`` confines the filesystem on a normal start, opens it in debug mode."""

    def _run(self, tmp_path: Path, *, allow_debugger: bool) -> list[tuple[object, object]]:
        sidecar = tmp_path / "demo.json"
        payload: dict[str, object] = {
            "container_name": "demo",
            "ipc_mode": "socket",
            "db_path": str(tmp_path / "vault.db"),
            "runtime_dir": str(tmp_path / "rt" / "sandbox"),
        }
        if allow_debugger:
            payload["allow_debugger"] = True
        sidecar.write_text(json.dumps(payload))

        calls: list[tuple[object, object]] = []

        def _spy(read_exec: object, read_write: object) -> object:
            calls.append((read_exec, read_write))
            from terok_util import LandlockReport

            return LandlockReport(confined=True, reason="spy")

        async def _noop(cfg: object, paths: object, stop: object) -> None:
            return None

        with (
            patch("terok_sandbox.supervisor.children.confine_filesystem", _spy),
            patch.dict("terok_sandbox.supervisor.children._RUNNERS", {"vault": _noop}, clear=False),
        ):
            assert run_child("vault", "abc123def456", sidecar) == 0
        return calls

    def test_normal_start_confines_to_the_service_policy(self, tmp_path: Path) -> None:
        from terok_sandbox.supervisor.children import _SYSTEM_READABLE_ROOTS

        ((read_exec, read_write),) = self._run(tmp_path, allow_debugger=False)
        assert read_exec == (*_SYSTEM_READABLE_ROOTS, tmp_path / "routes.json")
        assert read_write == [
            tmp_path / "rt" / "sandbox" / "run" / "demo" / "vault",
            tmp_path,
            tmp_path / "rt" / "sandbox" / "terok" / "vault" / "locks",
        ]

    def test_debug_mode_leaves_the_filesystem_open(self, tmp_path: Path) -> None:
        assert self._run(tmp_path, allow_debugger=True) == []

    def test_verdict_broker_leaves_the_filesystem_open(self, tmp_path: Path) -> None:
        """The Podman/nsenter broker cannot run inside the service path policy."""
        sidecar = tmp_path / "demo.json"
        sidecar.write_text(
            json.dumps(
                {
                    "container_name": "demo",
                    "ipc_mode": "socket",
                    "db_path": str(tmp_path / "vault.db"),
                    "runtime_dir": str(tmp_path / "rt" / "sandbox"),
                }
            )
        )

        async def _noop(cfg: object, paths: object, stop: object) -> None: ...

        with (
            patch("terok_sandbox.supervisor.children.confine_filesystem") as confine,
            patch.dict(
                "terok_sandbox.supervisor.children._RUNNERS",
                {"verdict": _noop},
                clear=False,
            ),
        ):
            assert run_child("verdict", "abc123def456", sidecar) == 0
        confine.assert_not_called()

    def test_shared_roots_exclude_the_runtime_tree(self) -> None:
        from terok_sandbox.supervisor.children import _SYSTEM_READABLE_ROOTS

        assert SYSTEM_RUNTIME_ROOT not in _SYSTEM_READABLE_ROOTS

    def test_unavailable_policy_warns(
        self, tmp_path: Path, caplog: pytest.LogCaptureFixture
    ) -> None:
        """A fail-open kernel is visible at the default log level."""
        from terok_util import LandlockReport

        sidecar = tmp_path / "demo.json"
        sidecar.write_text(
            json.dumps(
                {
                    "container_name": "demo",
                    "ipc_mode": "socket",
                    "db_path": str(tmp_path / "vault.db"),
                    "runtime_dir": str(tmp_path / "rt" / "sandbox"),
                }
            )
        )

        async def _noop(cfg: object, paths: object, stop: object) -> None: ...

        with (
            patch(
                "terok_sandbox.supervisor.children.confine_filesystem",
                return_value=LandlockReport(confined=False, reason="unsupported"),
            ),
            patch.dict(
                "terok_sandbox.supervisor.children._RUNNERS",
                {"clearance": _noop},
                clear=False,
            ),
            caplog.at_level("WARNING"),
        ):
            assert run_child("clearance", "abc123def456", sidecar) == 0
        assert "filesystem-confinement not applied: unsupported" in caplog.text

    def test_partial_policy_warns(self, tmp_path: Path, caplog: pytest.LogCaptureFixture) -> None:
        """An ABI-limited policy is distinguished from no confinement."""
        from terok_util import LandlockReport

        sidecar = tmp_path / "demo.json"
        sidecar.write_text(
            json.dumps(
                {
                    "container_name": "demo",
                    "ipc_mode": "socket",
                    "db_path": str(tmp_path / "vault.db"),
                    "runtime_dir": str(tmp_path / "rt" / "sandbox"),
                }
            )
        )

        async def _noop(cfg: object, paths: object, stop: object) -> None: ...

        with (
            patch(
                "terok_sandbox.supervisor.children.confine_filesystem",
                return_value=LandlockReport(
                    confined=False,
                    reason="ABI 2 cannot deny truncation",
                    partially_confined=True,
                ),
            ),
            patch.dict(
                "terok_sandbox.supervisor.children._RUNNERS",
                {"clearance": _noop},
                clear=False,
            ),
            caplog.at_level("WARNING"),
        ):
            assert run_child("clearance", "abc123def456", sidecar) == 0
        assert "filesystem-confinement partial: ABI 2 cannot deny truncation" in caplog.text


class TestPolicyConfinesOnTheLiveKernel:
    """The whole ``run_child`` bring-up isolates filesystem paths on the live kernel.

    The single high-surface proof: a fresh process loads a real sidecar and
    calls [`run_child`][terok_sandbox.supervisor.children.run_child], which
    hardens, applies ``_SYSTEM_READABLE_ROOTS`` + ``_writable_paths`` through
    real Landlock, then drives a stub ``vault`` runner that probes its lane.
    The service writes its own data but cannot read the runtime passphrase
    escrow or replace a sibling service's socket.  This deliberately proves
    path isolation only: Linux keyring permissions remain a separate same-UID
    boundary.  Runs on every matrix slot, so each distro's kernel exercises the
    confinement; a kernel without Landlock skips.
    """

    def test_vault_cannot_read_escrow_or_replace_sibling_socket(self, tmp_path: Path) -> None:
        lane = tmp_path / "lane"
        db_path = lane / "vault" / "vault.db"
        runtime_dir = lane / "rt" / "sandbox"
        db_path.parent.mkdir(parents=True)
        runtime_dir.mkdir(parents=True)
        routes_path = db_path.parent / "routes.json"
        routes_path.write_text("{}")
        pending_passphrase = runtime_dir / "vault.passphrase.pending"
        pending_passphrase.write_text("rekey escrow")
        signer_socket = runtime_dir / "run" / "demo" / "signer" / "ssh-agent.sock"
        signer_socket.parent.mkdir(parents=True)
        signer_socket.write_text("stand-in sibling socket inode")

        sidecar = tmp_path / "demo.json"
        sidecar.write_text(
            json.dumps(
                {
                    "container_name": "demo",
                    "ipc_mode": "socket",
                    "db_path": str(db_path),
                    "runtime_dir": str(runtime_dir),
                    "routes_path": str(routes_path),
                    "credentials_passphrase_command": "printf test-passphrase",
                }
            )
        )

        # A stub vault runner: replaces the real proxy so run_child drives the
        # full harden→confine→_drive path, then the runner probes the live lane.
        probe = textwrap.dedent(
            f"""
            from pathlib import Path
            from terok_util import hardening
            from terok_sandbox.supervisor import children

            libc = hardening._libc()
            if libc is None or hardening._landlock_abi(libc) < 1:
                print("unsupported:no-landlock")
                raise SystemExit(0)

            async def _probe(cfg, paths, stop):
                out = []
                Path(cfg.db_path.parent, "ok").write_text("x")
                out.append("lane-write-ok")
                try:
                    Path({str(pending_passphrase)!r}).read_text()
                    out.append("escrow-read-LEAK")
                except (PermissionError, OSError):
                    out.append("escrow-read-denied")
                try:
                    Path({str(signer_socket)!r}).unlink()
                    out.append("sibling-unlink-LEAK")
                except (PermissionError, OSError):
                    out.append("sibling-unlink-denied")
                print(";".join(out))

            children._RUNNERS["vault"] = _probe
            raise SystemExit(children.run_child("vault", "abc123def456", Path({str(sidecar)!r})))
            """
        )
        result = subprocess.run(
            [sys.executable, "-c", probe], capture_output=True, text=True, check=True
        )
        out = result.stdout.strip().splitlines()[-1] if result.stdout.strip() else ""
        if out.startswith("unsupported:"):
            pytest.skip(f"kernel without Landlock: {out}")
        assert out == "lane-write-ok;escrow-read-denied;sibling-unlink-denied", (
            f"vault policy leaked: {out!r}"
        )

    def test_passphrase_helper_runs_before_config_becomes_unreadable(self, tmp_path: Path) -> None:
        """A config-selected helper resolves once; its files stay outside the long-lived lane."""
        operator_dir = tmp_path / "operator-config"
        operator_dir.mkdir()
        helper_secret = operator_dir / "vault-passphrase"
        helper_secret.write_text("headless-passphrase\n")
        config_file = operator_dir / "config.yml"
        config_file.write_text(
            f"credentials:\n  use_keyring: false\n  passphrase_command: cat {helper_secret}\n"
        )

        state_dir = tmp_path / "state"
        runtime_dir = tmp_path / "runtime" / "sandbox"
        vault_dir = tmp_path / "vault"
        vault_dir.mkdir()
        (vault_dir / "routes.json").write_text("{}")
        worker = textwrap.dedent(
            f"""
            from pathlib import Path
            from terok_util import hardening
            from terok_sandbox.config import SandboxConfig
            from terok_sandbox.launch import PerContainerResources, write_sidecar
            from terok_sandbox.supervisor import children

            libc = hardening._libc()
            if libc is None or hardening._landlock_abi(libc) < 1:
                print("unsupported:no-landlock")
                raise SystemExit(0)

            cfg = SandboxConfig(
                state_dir=Path({str(state_dir)!r}),
                runtime_dir=Path({str(runtime_dir)!r}),
                vault_dir=Path({str(vault_dir)!r}),
                services_mode="socket",
            )
            sidecar = write_sidecar(
                "demo",
                cfg=cfg,
                per_container=PerContainerResources(
                    container_runtime_dir=cfg.container_runtime_dir("demo"),
                    token_broker_port=None,
                    ssh_signer_port=None,
                    gate_port=None,
                ),
            )
            assert sidecar is not None

            confine = children.confine_filesystem
            def reporting_confine(read_exec, read_write):
                report = confine(read_exec, read_write)
                print(f"landlock:{{int(report.confined)}}:{{report.reason}}", flush=True)
                return report

            async def probe(resolved, paths, stop):
                checks = [
                    f"resolved={{int(resolved._resolved_passphrase == 'headless-passphrase')}}",
                    f"keyring-off={{int(resolved.credentials_use_keyring is False)}}",
                ]
                for label, path in (
                    ("config", Path({str(config_file)!r})),
                    ("helper", Path({str(helper_secret)!r})),
                ):
                    try:
                        path.read_text()
                        checks.append(f"{{label}}-LEAK")
                    except (PermissionError, OSError):
                        checks.append(f"{{label}}-denied")
                print(";".join(checks), flush=True)

            children.confine_filesystem = reporting_confine
            children._RUNNERS["vault"] = probe
            raise SystemExit(children.run_child("vault", "abc123def456", sidecar))
            """
        )
        env = {**os.environ, "TEROK_CONFIG_FILE": str(config_file)}
        result = subprocess.run(
            [sys.executable, "-c", worker],
            capture_output=True,
            text=True,
            env=env,
            check=True,
        )
        lines = result.stdout.strip().splitlines()
        if lines and lines[0].startswith("unsupported:"):
            pytest.skip(f"kernel without Landlock: {lines[0]}")
        assert lines[0].startswith("landlock:1:"), result.stderr
        assert lines[-1] == "resolved=1;keyring-off=1;config-denied;helper-denied"

    def test_gate_accepts_real_git_push_inside_scoped_policy(self, tmp_path: Path) -> None:
        """Git can migrate quarantine objects without opening the other mirrors."""
        if shutil.which("git") is None:
            pytest.skip("needs git")
        mirror_root = tmp_path / "mirrors"
        scoped_repo = mirror_root / "proj.git"
        other_repo = mirror_root / "other.git"
        source = tmp_path / "source"
        for repo in (scoped_repo, other_repo):
            subprocess.run(  # nosec B603 B607
                ["git", "init", "--bare", str(repo)],
                capture_output=True,
                check=True,
            )
        subprocess.run(  # nosec B603 B607
            ["git", "init", str(source)],
            capture_output=True,
            check=True,
        )
        (source / "README").write_text("landlocked push\n")
        subprocess.run(  # nosec B603 B607
            ["git", "-C", str(source), "add", "README"],
            capture_output=True,
            check=True,
        )
        subprocess.run(  # nosec B603 B607
            [
                "git",
                "-C",
                str(source),
                "-c",
                "user.name=Terok Test",
                "-c",
                "user.email=terok@example.invalid",
                "commit",
                "-m",
                "test",
            ],
            capture_output=True,
            check=True,
        )

        with socket.socket() as listener:
            listener.bind((LOCALHOST, 0))
            port = listener.getsockname()[1]

        runtime_dir = tmp_path / "runtime" / "sandbox"
        runtime_dir.mkdir(parents=True)
        sidecar = tmp_path / "gate.json"
        token = "terok-g-landlock"
        sidecar.write_text(
            json.dumps(
                {
                    "container_name": "demo",
                    "ipc_mode": "tcp",
                    "db_path": str(tmp_path / "vault.db"),
                    "runtime_dir": str(runtime_dir),
                    "project_id": "proj",
                    "gate_port": port,
                    "gate_base_path": str(mirror_root),
                    "gate_token": token,
                }
            )
        )
        worker = textwrap.dedent(
            f"""
            from pathlib import Path
            from terok_sandbox.supervisor import children

            confine = children.confine_filesystem
            def reporting_confine(read_exec, read_write):
                report = confine(read_exec, read_write)
                print(f"landlock:{{int(report.confined)}}:{{report.reason}}", flush=True)
                return report

            run_gate = children._RUNNERS["gate"]
            async def probing_gate(cfg, paths, stop):
                probes = []
                try:
                    Path({str(other_repo / "HEAD")!r}).read_text()
                    probes.append("read-LEAK")
                except (PermissionError, OSError):
                    probes.append("read-denied")
                try:
                    Path({str(other_repo / "landlock-leak")!r}).write_text("leak")
                    probes.append("write-LEAK")
                except (PermissionError, OSError):
                    probes.append("write-denied")
                print("other-mirror:" + ",".join(probes), flush=True)
                await run_gate(cfg, paths, stop)

            children.confine_filesystem = reporting_confine
            children._RUNNERS["gate"] = probing_gate
            raise SystemExit(
                children.run_child("gate", "abc123def456", Path({str(sidecar)!r}))
            )
            """
        )
        process = subprocess.Popen(  # nosec B603
            [sys.executable, "-c", worker],
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
        )
        try:
            assert process.stdout is not None
            report = process.stdout.readline().strip()
            if report.startswith("landlock:0:"):
                pytest.skip(f"kernel cannot install the complete gate policy: {report}")
            assert report.startswith("landlock:1:"), (
                f"gate child failed before reporting confinement: {report!r}"
            )
            assert process.stdout.readline().strip() == "other-mirror:read-denied,write-denied"

            deadline = time.monotonic() + 5
            while True:
                try:
                    with socket.create_connection((LOCALHOST, port), timeout=0.1):
                        break
                except OSError:
                    if process.poll() is not None or time.monotonic() >= deadline:
                        assert process.stderr is not None
                        pytest.fail(f"gate did not start: {process.stderr.read()}")
                    time.sleep(0.02)

            gate_url = f"http://{token}:x@{LOCALHOST}:{port}/proj.git"
            pushed = subprocess.run(  # nosec B603 B607
                [
                    "git",
                    "-C",
                    str(source),
                    "-c",
                    "credential.helper=",
                    "push",
                    gate_url,
                    "HEAD:refs/heads/main",
                ],
                capture_output=True,
                text=True,
                check=False,
                env={**os.environ, "GIT_TERMINAL_PROMPT": "0"},
            )
            assert pushed.returncode == 0, pushed.stderr
            assert (
                subprocess.run(  # nosec B603 B607
                    ["git", "--git-dir", str(scoped_repo), "rev-parse", "refs/heads/main"],
                    capture_output=True,
                    check=True,
                    text=True,
                ).stdout.strip()
                == subprocess.run(  # nosec B603 B607
                    ["git", "-C", str(source), "rev-parse", "HEAD"],
                    capture_output=True,
                    check=True,
                    text=True,
                ).stdout.strip()
            )
            assert not any(other_repo.glob("objects/??/*"))
        finally:
            process.terminate()
            try:
                process.communicate(timeout=5)
            except subprocess.TimeoutExpired:
                process.kill()
                process.communicate()
