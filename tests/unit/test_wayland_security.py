# SPDX-FileCopyrightText: 2026 Jiri Vyskocil
# SPDX-License-Identifier: Apache-2.0

"""Keep desktop access opt-in and the generated proxy policy restrictive."""

from __future__ import annotations

import asyncio
import contextlib
import errno
import fcntl
import json
import logging
import os
import select
import shutil
import signal
import socket
import struct
import sys
import textwrap
import tomllib
from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import AsyncMock, Mock, call, patch

import pytest
from terok_util import harden_self

from terok_sandbox.supervisor import wayland as proxy_supervision
from terok_sandbox.supervisor.launcher import relay_output
from terok_sandbox.supervisor.sidecar import SidecarConfig, SupervisorPaths
from terok_sandbox.supervisor.wayland import _confine_proxy, _policy_fd, run_proxy
from terok_sandbox.vault.context import BrokerContext
from terok_sandbox.wayland import (
    WAYLAND_PROBE,
    WaylandMode,
    WaylandSession,
    proxy_config,
    resolve_session,
)
from tests.constants import FAKE_CONTAINER_ID, FAKE_CONTAINER_NAME

_PROCESS_TIMEOUT = 5
_FD_SENTINEL_MINIMUM = 100
_REQUIRED_POLICY_SEALS = 0x0F
_PROXY_SCRIPT = """\
import fcntl, json, os, signal, sys, tomllib
from pathlib import Path
with open(sys.argv[1], 'rb') as policy_file:
    policy = tomllib.load(policy_file)
    seals = fcntl.fcntl(policy_file.fileno(), fcntl.F_GET_SEALS)
try:
    os.fstat({sentinel_fd})
    unrelated_fd_open = True
except OSError:
    unrelated_fd_open = False
try:
    Path(policy['socket']['upstream']).read_bytes()
    host_path_denied = False
except PermissionError:
    host_path_denied = True
{prepare}
report = {{'pid': os.getpid(), 'policy': policy, 'seals': seals,
           'environment': dict(os.environ), 'unrelated_fd_open': unrelated_fd_open,
           'host_path_denied': host_path_denied}}
report_file = Path(policy['socket']['listen']).with_suffix('.json')
temporary = report_file.with_suffix('.tmp')
temporary.write_text(json.dumps(report))
temporary.replace(report_file)
{ending}
"""


@pytest.fixture(autouse=True)
def _harden_only_exec_children(monkeypatch: pytest.MonkeyPatch) -> None:
    """Supply run_child's prerequisite without restricting the pytest process."""
    monkeypatch.setattr("terok_sandbox.supervisor.wayland._confine_proxy", _harden_exec_child)


def _harden_exec_child(parent_pid: int, binary: Path, runtime_dir: Path) -> None:
    """Apply irreversible process restrictions only after subprocess fork."""
    harden_self()
    _confine_proxy(parent_pid, binary, runtime_dir)


def test_proxy_policy_grants_only_app_surfaces_input_and_rendering(tmp_path: Path) -> None:
    """A new global is an explicit access-policy change, not a compatibility default."""
    policy = tomllib.loads(proxy_config(tmp_path / "upstream", tmp_path / "listener").decode())

    assert set(policy["filter"]["allowed_globals"]) == {
        "wl_compositor",
        "wl_subcompositor",
        "wl_shm",
        "wl_output",
        "wl_seat",
        "xdg_wm_base",
        "zxdg_decoration_manager_v1",
        "wp_viewporter",
        "wp_fractional_scale_manager_v1",
        "wp_presentation",
        "zwp_linux_dmabuf_v1",
        "zwp_linux_explicit_synchronization_v1",
        "wp_linux_drm_syncobj_manager_v1",
        "wp_single_pixel_buffer_manager_v1",
        "zwp_relative_pointer_manager_v1",
        "zwp_pointer_constraints_v1",
        "wp_cursor_shape_manager_v1",
    }
    assert policy["filter"]["requests"] == []
    assert policy["filter"]["dry_run"] is False
    assert "exec" not in policy


@pytest.mark.parametrize(
    "filename", ['socket"with-quotes', "socket\\with-backslash", "socket\nnext"]
)
def test_session_paths_cannot_inject_proxy_policy(tmp_path: Path, filename: str) -> None:
    """Socket paths remain TOML string values even with configuration metacharacters."""
    upstream = tmp_path / filename
    listener = tmp_path / "listener"
    policy = tomllib.loads(proxy_config(upstream, listener).decode())

    assert policy["socket"] == {"listen": str(listener), "upstream": str(upstream)}
    assert set(policy) == {"socket", "filter"}


def test_disabled_wayland_never_resolves_desktop_or_binary() -> None:
    """The default mode does not inspect host desktop state or optional tools."""
    with (
        patch("terok_sandbox.wayland._session_socket") as desktop,
        patch("terok_sandbox.wayland.find_host_tool") as binary,
    ):
        session = resolve_session(WaylandMode.NONE)

    assert session.mode is WaylandMode.NONE
    assert not session.proxied
    desktop.assert_not_called()
    binary.assert_not_called()


def test_socket_below_replaceable_ancestor_is_rejected(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A private socket parent does not protect it from a writable ancestor rename."""
    ancestor = tmp_path / "shared"
    parent = ancestor / "private"
    parent.mkdir(parents=True, mode=0o700)
    ancestor.chmod(0o777)
    endpoint = parent / "socket"
    with socket.socket(socket.AF_UNIX) as listener:
        listener.bind(str(endpoint))
        monkeypatch.setenv("WAYLAND_DISPLAY", str(endpoint))
        session = resolve_session(WaylandMode.DIRECT)
    assert session.upstream is None
    assert "ancestor" in session.error


@pytest.mark.parametrize("reply", ["fragmented", "closed", "wrong-object", "timeout"])
@pytest.mark.asyncio
async def test_probe_requires_compositor_roundtrip(
    tmp_path_factory: pytest.TempPathFactory, reply: str
) -> None:
    """Accepting a connection is insufficient; only a matching bounded sync succeeds."""
    endpoint = tmp_path_factory.mktemp("wl-probe") / "socket"
    request = asyncio.get_running_loop().create_future()
    finished = asyncio.Event()

    async def compositor(reader: asyncio.StreamReader, writer: asyncio.StreamWriter) -> None:
        try:
            request.set_result(await reader.readexactly(12))
            if reply == "fragmented":
                response = struct.pack("=III", 2, 12 << 16, 42)
                for fragment in (response[:4], response[4:]):
                    writer.write(fragment)
                    await writer.drain()
                    await asyncio.sleep(0)
            elif reply == "wrong-object":
                writer.write(struct.pack("=III", 3, 12 << 16, 42))
                await writer.drain()
            elif reply == "timeout":
                await reader.read()
        finally:
            writer.close()
            await writer.wait_closed()
            finished.set()

    server = await asyncio.start_unix_server(compositor, path=endpoint)
    async with server:
        process = await asyncio.create_subprocess_exec(
            sys.executable,
            *WAYLAND_PROBE[1:-1],
            str(endpoint),
            stderr=asyncio.subprocess.PIPE,
        )
        try:
            await asyncio.wait_for(process.communicate(), _PROCESS_TIMEOUT)
        finally:
            if process.returncode is None:
                process.kill()
                await process.wait()
        assert await request == struct.pack("=III", 1, 12 << 16, 2)
        assert (process.returncode == 0) == (reply == "fragmented")
        await asyncio.wait_for(finished.wait(), _PROCESS_TIMEOUT)


def test_proxy_policy_is_immutable_and_closed_after_use() -> None:
    """Seals prevent policy replacement even by the process holding its descriptor."""
    with _policy_fd(b"reviewed policy") as descriptor:
        assert os.pread(descriptor, 100, 0) == b"reviewed policy"
        with pytest.raises(PermissionError):
            os.pwrite(descriptor, b"changed", 0)
        with pytest.raises(PermissionError):
            os.ftruncate(descriptor, 0)
    with pytest.raises(OSError):
        os.fstat(descriptor)


@pytest.mark.parametrize(
    "failure,reached,message",
    [
        (None, 6, None),
        ("parent-death", 1, "cannot arm proxy parent-death signal"),
        ("parent-race", 2, "wrapper died before proxy startup"),
        ("selinux", 3, "wrong proxy domain"),
        ("fd-limit", 4, "resource limit denied"),
        ("memory-limit", 5, "resource limit denied"),
        ("unavailable", 6, "filesystem confinement unavailable"),
        ("partial", 6, "filesystem confinement unavailable"),
    ],
)
def test_proxy_confinement_requires_every_gate_in_order(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    failure: str | None,
    reached: int,
    message: str | None,
) -> None:
    """Any failed gate aborts startup without retries or later, weaker confinement."""
    binary, runtime_dir = tmp_path / "wl-mitm", tmp_path / "runtime"
    parent_pid = os.getpid()
    gates = Mock()
    gates.prctl.return_value = -1 if failure == "parent-death" else 0
    gates.parent.return_value = parent_pid + (failure == "parent-race")
    if failure == "selinux":
        gates.selinux.side_effect = RuntimeError("wrong proxy domain")
    if failure in {"fd-limit", "memory-limit"}:
        gates.limit.side_effect = [None] * (reached - 4) + [
            OSError(errno.EPERM, "resource limit denied")
        ]
    gates.filesystem.return_value = SimpleNamespace(
        confined=failure != "unavailable",
        partially_confined=failure == "partial",
        reason="test confinement report",
    )
    monkeypatch.setattr(proxy_supervision, "_LIBC", SimpleNamespace(prctl=gates.prctl))
    monkeypatch.setattr(proxy_supervision.ctypes, "get_errno", lambda: errno.EPERM)
    monkeypatch.setattr(proxy_supervision.os, "getppid", gates.parent)
    monkeypatch.setattr(proxy_supervision, "require_wayland_exec_context", gates.selinux)
    monkeypatch.setattr(proxy_supervision.resource, "setrlimit", gates.limit)
    monkeypatch.setattr(proxy_supervision, "confine_filesystem", gates.filesystem)

    if failure is None:
        _confine_proxy(parent_pid, binary, runtime_dir)
    else:
        exception = (
            OSError if failure in {"parent-death", "fd-limit", "memory-limit"} else RuntimeError
        )
        with pytest.raises(exception, match=message):
            _confine_proxy(parent_pid, binary, runtime_dir)

    assert (
        gates.mock_calls
        == [
            call.prctl(proxy_supervision._PR_SET_PDEATHSIG, signal.SIGKILL, 0, 0, 0),
            call.parent(),
            call.selinux(),
            call.limit(
                proxy_supervision.resource.RLIMIT_NOFILE,
                (proxy_supervision._PROXY_FD_LIMIT,) * 2,
            ),
            call.limit(
                proxy_supervision.resource.RLIMIT_AS,
                (proxy_supervision._PROXY_MEMORY_LIMIT,) * 2,
            ),
            call.filesystem((*proxy_supervision._PROXY_READ_ROOTS, binary), (runtime_dir,)),
        ][:reached]
    )


@pytest.mark.parametrize("mode", [WaylandMode.NONE, WaylandMode.DIRECT, WaylandMode.MITM])
async def test_unwired_proxy_child_refuses_before_creating_policy_or_process(
    tmp_path: Path, mode: WaylandMode
) -> None:
    """A wrong mode or incomplete filtered wiring cannot accidentally launch a proxy."""
    cfg, paths = _proxy_fixture(tmp_path)
    cfg = replace(cfg, wayland=WaylandSession(mode, upstream=cfg.wayland.upstream))
    with (
        patch.object(proxy_supervision, "_policy_fd") as policy,
        patch.object(
            proxy_supervision.asyncio, "create_subprocess_exec", new_callable=AsyncMock
        ) as spawn,
        pytest.raises(RuntimeError, match="without proxy wiring"),
    ):
        await run_proxy(cfg, paths, asyncio.Event())
    policy.assert_not_called()
    spawn.assert_not_awaited()


async def test_proxy_diagnostics_are_bounded_and_tolerate_non_utf8(
    caplog: pytest.LogCaptureFixture,
) -> None:
    """The host logger receives bounded diagnostics even from a malformed proxy output stream."""
    output = asyncio.StreamReader()
    output.feed_data(b"x" * 4096 + b"invalid: \xff\n")
    output.feed_eof()
    with caplog.at_level(logging.INFO, logger="terok-supervisor.wl-mitm"):
        await relay_output(SimpleNamespace(stdout=output), "wl-mitm")
    assert caplog.messages == ["x" * 4096, "invalid: \ufffd"]
    assert all(record.name == "terok-supervisor.wl-mitm" for record in caplog.records)


@pytest.mark.asyncio
async def test_proxy_exec_keeps_only_policy_fd_and_minimal_environment(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Real confined exec can reopen its sealed policy without inheriting host secrets."""
    monkeypatch.setenv("TEROK_TEST_SECRET", "not-for-the-proxy")
    with (tmp_path / "unrelated-host-file").open("w") as secret:
        sentinel = fcntl.fcntl(secret.fileno(), fcntl.F_DUPFD, _FD_SENTINEL_MINIMUM)
        os.set_inheritable(sentinel, True)
        cfg, paths = _proxy_fixture(tmp_path, sentinel_fd=sentinel)
        stop = asyncio.Event()
        running = asyncio.create_task(run_proxy(cfg, paths, stop))
        try:
            report = await _wait_for_report(paths, running)
            assert report["policy"]["socket"]["upstream"] == str(cfg.wayland.upstream)
            assert report["unrelated_fd_open"] is False
            assert report["host_path_denied"] is True
            assert report["seals"] == _REQUIRED_POLICY_SEALS
            assert "TEROK_TEST_SECRET" not in report["environment"]
            assert report["environment"]["TOKIO_WORKER_THREADS"] == "2"
        finally:
            stop.set()
            try:
                await asyncio.wait_for(running, _PROCESS_TIMEOUT)
            finally:
                os.close(sentinel)
        with pytest.raises(ProcessLookupError):
            os.kill(report["pid"], 0)


@pytest.mark.asyncio
async def test_proxy_unexpected_exit_is_not_reported_as_healthy(tmp_path: Path) -> None:
    """The service wrapper exits when its external proxy dies."""
    cfg, paths = _proxy_fixture(tmp_path, ending="sys.exit(7)")
    with pytest.raises(RuntimeError, match="status 7"):
        await asyncio.wait_for(run_proxy(cfg, paths, asyncio.Event()), _PROCESS_TIMEOUT)


@pytest.mark.asyncio
async def test_proxy_cancellation_kills_and_reaps_uncooperative_process(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Cancellation escalates past ignored SIGTERM without leaving a GUI proxy alive."""
    monkeypatch.setattr("terok_sandbox.supervisor.wayland._PROXY_TERM_GRACE", 0.05)
    cfg, paths = _proxy_fixture(tmp_path, prepare="signal.signal(signal.SIGTERM, signal.SIG_IGN)")
    running = asyncio.create_task(run_proxy(cfg, paths, asyncio.Event()))
    report = await _wait_for_report(paths, running)
    running.cancel()
    with pytest.raises(asyncio.CancelledError):
        await asyncio.wait_for(running, _PROCESS_TIMEOUT)
    with pytest.raises(ProcessLookupError):
        os.kill(report["pid"], 0)


@pytest.mark.asyncio
async def test_proxy_dies_when_its_service_wrapper_is_killed(tmp_path: Path) -> None:
    """The kernel kills the external proxy even when Python cannot run teardown."""
    cfg, paths = _proxy_fixture(tmp_path)
    wrapper_code = textwrap.dedent("""\
        import asyncio, sys
        from pathlib import Path
        from terok_sandbox.wayland import WaylandMode, WaylandSession
        from terok_sandbox.supervisor.sidecar import SidecarConfig, SupervisorPaths
        from terok_sandbox.supervisor.wayland import run_proxy
        from terok_sandbox.vault.context import BrokerContext
        from terok_util import harden_self
        root, binary = map(Path, sys.argv[1:3])
        paths = SupervisorPaths.for_container(sys.argv[3], sys.argv[4],
                                             root / 'sidecar.json', root / 'runtime')
        cfg = SidecarConfig(broker_context=BrokerContext('project', 'task', 'agent'),
                            container_name=sys.argv[4], ipc_mode='socket',
                            db_path=root / 'vault.db', runtime_dir=root / 'runtime',
                            wayland=WaylandSession(WaylandMode.MITM, root / 'upstream', binary))
        harden_self()
        asyncio.run(run_proxy(cfg, paths, asyncio.Event()))
        """)
    wrapper = await asyncio.create_subprocess_exec(
        sys.executable,
        "-c",
        wrapper_code,
        str(tmp_path),
        str(cfg.wayland.binary),
        FAKE_CONTAINER_ID,
        FAKE_CONTAINER_NAME,
        stderr=asyncio.subprocess.PIPE,
    )
    waiting = asyncio.create_task(wrapper.wait())
    proxy_pid = None
    try:
        report = await _wait_for_report(paths, waiting)
        proxy_pid = report["pid"]
        descriptor = os.pidfd_open(proxy_pid)
        try:
            wrapper.kill()
            await asyncio.wait_for(waiting, _PROCESS_TIMEOUT)
            assert select.select([descriptor], [], [], _PROCESS_TIMEOUT)[0]
        finally:
            os.close(descriptor)
    finally:
        if wrapper.returncode is None:
            wrapper.kill()
            await wrapper.wait()
        if proxy_pid is not None:
            with contextlib.suppress(ProcessLookupError):
                os.kill(proxy_pid, signal.SIGKILL)


def _proxy_fixture(
    root: Path,
    *,
    sentinel_fd: int = _FD_SENTINEL_MINIMUM,
    prepare: str = "",
    ending: str = "signal.pause()",
) -> tuple[SidecarConfig, SupervisorPaths]:
    """Prepare a stock-CLI-compatible test process under disposable host paths."""
    interpreter = shutil.which("python3", path=os.defpath)
    if interpreter is None:
        pytest.skip("a system Python interpreter is required for the confined executable fixture")
    binary = root / "fake-wl-mitm"
    binary.write_text(
        f"#!{interpreter}\n"
        + _PROXY_SCRIPT.format(sentinel_fd=sentinel_fd, prepare=prepare, ending=ending)
    )
    binary.chmod(0o700)
    (root / "upstream").write_text("private host data outside the proxy runtime")
    paths = SupervisorPaths.for_container(
        FAKE_CONTAINER_ID, FAKE_CONTAINER_NAME, root / "sidecar.json", root / "runtime"
    )
    paths.wayland_socket.parent.mkdir(parents=True, mode=0o700)
    cfg = SidecarConfig(
        broker_context=BrokerContext("project", "task", "agent"),
        container_name=FAKE_CONTAINER_NAME,
        ipc_mode="socket",
        db_path=root / "vault.db",
        runtime_dir=root / "runtime",
        wayland=WaylandSession(WaylandMode.MITM, root / "upstream", binary),
    )
    return cfg, paths


async def _wait_for_report(paths: SupervisorPaths, running: asyncio.Task) -> dict:
    """Wait briefly for the confined fixture to report, surfacing an early process exit."""
    async with asyncio.timeout(_PROCESS_TIMEOUT):
        while not paths.wayland_socket.with_suffix(".json").exists():
            if running.done():
                await running
                raise AssertionError("proxy exited before reporting readiness")
            await asyncio.sleep(0.01)
    return json.loads(paths.wayland_socket.with_suffix(".json").read_text())
