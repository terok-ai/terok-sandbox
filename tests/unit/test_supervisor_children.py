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
import signal
import subprocess
import sys
import textwrap
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from terok_sandbox.supervisor.children import (
    SERVICE_NAMES,
    _arm_parent_death_signal,
    _ensure_socket_dirs,
    _install_signal_handlers,
    _run_clearance,
    _run_gate,
    _run_signer,
    _run_vault,
    _run_verdict,
    _writable_paths,
    run_child,
)
from terok_sandbox.supervisor.main import SidecarConfig, SupervisorPaths


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
    from terok_sandbox._util._landlock import LandlockReport

    monkeypatch.setattr(
        "terok_sandbox.supervisor.children.confine_filesystem",
        lambda _re, _rw: LandlockReport(confined=True, reason="stubbed in tests"),
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
        scope_id="proj",
        project_id="proj",
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
        scope_id="proj",
        project_id="proj",
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
        vault_dir = tmp_path / "rt" / "sandbox" / "run" / "demo"
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

    def test_secretless_services_get_only_the_runtime_root(self, tmp_path: Path) -> None:
        cfg = _socket_cfg(tmp_path)
        for service in ("verdict", "clearance", "signer"):
            assert _writable_paths(service, cfg) == [tmp_path / "rt"]

    def test_vault_adds_its_db_directory(self, tmp_path: Path) -> None:
        cfg = _socket_cfg(tmp_path)  # db_path = tmp_path / "vault.db"
        assert _writable_paths("vault", cfg) == [tmp_path / "rt", tmp_path]

    def test_gate_adds_the_mirror_tree_when_wired(self, tmp_path: Path) -> None:
        mirror = tmp_path / "gate"
        cfg = _socket_cfg(tmp_path, gate_base_path=mirror, gate_token="tok")  # nosec B106
        assert _writable_paths("gate", cfg) == [tmp_path / "rt", mirror]

    def test_unwired_gate_gets_only_the_runtime_root(self, tmp_path: Path) -> None:
        assert _writable_paths("gate", _socket_cfg(tmp_path)) == [tmp_path / "rt"]


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
            from terok_sandbox._util._landlock import LandlockReport

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
        assert read_exec is _SYSTEM_READABLE_ROOTS  # shared read+exec roots
        assert read_write == [tmp_path / "rt", tmp_path]  # vault write lane: runtime root + db dir

    def test_debug_mode_leaves_the_filesystem_open(self, tmp_path: Path) -> None:
        assert self._run(tmp_path, allow_debugger=True) == []


class TestPolicyConfinesOnTheLiveKernel:
    """The per-service policy, applied through real Landlock, isolates the service.

    Unlike ``TestWritablePaths`` (which asserts *which* directories the policy
    names) and ``test_landlock`` (which asserts the primitive works), this ties
    them together on the running kernel: the actual ``_SYSTEM_READABLE_ROOTS`` +
    ``_writable_paths`` for a service are applied via ``confine_filesystem`` in a
    fresh process, and the service can write its own lane but **cannot read a
    sibling service's data**.  It is the end-to-end proof of cross-service
    isolation, not just of the primitive.
    """

    def test_vault_writes_its_lane_and_cannot_read_a_sibling(self, tmp_path: Path) -> None:
        from terok_sandbox.supervisor.children import _SYSTEM_READABLE_ROOTS

        # Nest runtime_dir + db_path under ``lane/`` so the granted parents stay
        # there and a sibling service's dir under the same tree is outside them.
        lane = tmp_path / "lane"
        cfg = SidecarConfig(
            container_name="demo",
            ipc_mode="socket",
            db_path=lane / "vault" / "vault.db",
            runtime_dir=lane / "rt" / "sandbox",
        )
        read_write = _writable_paths("vault", cfg)  # [lane/rt, lane/vault]
        for path in read_write:
            path.mkdir(parents=True, exist_ok=True)
        sibling = lane / "gate-mirror"  # another service's data, not in vault's lane
        sibling.mkdir(parents=True)
        (sibling / "secret").write_text("a gate secret vault must never read")

        read_exec = [str(p) for p in _SYSTEM_READABLE_ROOTS]
        writable = [str(p) for p in read_write]
        probe = textwrap.dedent(
            f"""
            import ctypes
            from pathlib import Path
            from terok_sandbox._util._landlock import confine_filesystem

            ctypes.CDLL(None, use_errno=True).prctl(38, 1, 0, 0, 0)  # no_new_privs
            report = confine_filesystem(
                [Path(p) for p in {read_exec!r}], [Path(p) for p in {writable!r}]
            )
            if not report.confined:
                print(f"unsupported:{{report.reason}}")
                raise SystemExit(0)
            out = []
            Path({writable[0]!r}, "ok").write_text("x")
            out.append("lane-write-ok")
            try:
                Path({str(sibling)!r}, "secret").read_text()
                out.append("sibling-read-LEAK")
            except (PermissionError, OSError):
                out.append("sibling-read-denied")
            print(";".join(out))
            """
        )
        result = subprocess.run(
            [sys.executable, "-c", probe], capture_output=True, text=True, check=True
        )
        out = result.stdout.strip()
        if out.startswith("unsupported:"):
            pytest.skip(f"kernel without Landlock: {out}")
        assert out == "lane-write-ok;sibling-read-denied", f"vault policy leaked: {out!r}"
