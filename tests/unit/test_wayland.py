# SPDX-FileCopyrightText: 2026 Jiri Vyskocil
# SPDX-License-Identifier: Apache-2.0

"""Wayland remains optional, label-preserving, and bound to the selected desktop."""

from __future__ import annotations

import asyncio
import json
import logging
import os
import socket
import stat
import subprocess
from dataclasses import replace
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from terok_sandbox import (
    BindMountSpec,
    LifecycleHooks,
    RunSpec,
    Sandbox,
    SandboxConfig,
    SocketSpec,
    VolumeSpec,
    WaylandMode,
    WaylandRecreationRequired,
)
from terok_sandbox._util import _selinux
from terok_sandbox.podman_args import (
    CONTAINER_RUNTIME_DIR,
    CONTAINER_USER_RUNTIME_DIR,
    CONTAINER_WAYLAND_SOCKET,
)
from terok_sandbox.runtime.protocol import ExecResult
from terok_sandbox.supervision import _expected_endpoints
from terok_sandbox.supervisor.launcher import EXPECTED_PARENT_ENV, finish_output, launch_child
from terok_sandbox.supervisor.sidecar import (
    SupervisorPaths,
    load_sidecar,
    record_wayland,
    store_sidecar,
    wired_services,
)
from terok_sandbox.wayland import WaylandSession, emit_warning, resolve_session, warn_desktop_access
from tests.constants import FAKE_SERVICE_PORTS

_NAME = "wayland-test"
_RESOLVER = "terok_sandbox.wayland.resolve_session"


@pytest.fixture
def sandbox(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Sandbox:
    """A task with canonical private sidecar state and no real container runtime."""
    monkeypatch.setattr(_selinux, "is_selinux_enabled", lambda: False)
    cfg = SandboxConfig(state_dir=tmp_path / "state", runtime_dir=tmp_path / "runtime")
    path = cfg.sidecar_path(_NAME)
    path.parent.mkdir(parents=True)
    store_sidecar(
        path,
        {
            "container_name": _NAME,
            "broker_context": {"scope": "project", "subject": "task", "credential_set": "default"},
            "ipc_mode": "socket",
            "db_path": str(tmp_path / "vault.db"),
            "runtime_dir": str(cfg.runtime_dir),
        },
    )
    return Sandbox(cfg, runtime=MagicMock())


@pytest.fixture
def desktop_socket(tmp_path: Path):
    """An actual Unix socket to detect accidental directory/file bind acceptance."""
    path = tmp_path / "wayland-0"
    with socket.socket(socket.AF_UNIX) as listener:
        listener.bind(str(path))
        yield path


@pytest.fixture
def session_socket(desktop_socket: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    """Select a real socket belonging to the test runner's desktop user."""
    monkeypatch.setenv("WAYLAND_DISPLAY", desktop_socket.name)
    monkeypatch.setenv("XDG_RUNTIME_DIR", str(desktop_socket.parent))
    for name in ("CONTAINER_HOST", "CONTAINER_CONNECTION"):
        monkeypatch.delenv(name, raising=False)
    return desktop_socket


def _spec(tmp_path: Path, mode: WaylandMode = WaylandMode.MITM, **kwargs) -> RunSpec:
    """A task specification whose desktop request is the only optional grant."""
    return RunSpec(
        container_name=_NAME,
        image="test-image",
        env={},
        volumes=(),
        command=(),
        task_dir=tmp_path,
        wayland=mode,
        **kwargs,
    )


def test_socket_mount_preserves_labels(desktop_socket: Path) -> None:
    """Socket and volume specs share coordinates, but never label/copy semantics."""
    socket_spec = SocketSpec(desktop_socket, CONTAINER_WAYLAND_SOCKET)
    assert isinstance(socket_spec, BindMountSpec)
    assert isinstance(VolumeSpec(desktop_socket.parent, CONTAINER_RUNTIME_DIR), BindMountSpec)
    assert socket_spec.to_mount_arg() == (
        f"type=bind,source={desktop_socket},destination={CONTAINER_WAYLAND_SOCKET},readonly"
    )


@pytest.mark.parametrize("kind", ["file", "directory", "symlink", "missing"])
def test_socket_mount_rejects_non_socket(tmp_path: Path, desktop_socket: Path, kind: str) -> None:
    """Never let Podman create a missing socket source as a directory."""
    source = tmp_path / kind
    if kind == "file":
        source.touch()
    elif kind == "directory":
        source.mkdir()
    elif kind == "symlink":
        source.symlink_to(desktop_socket)
    with pytest.raises((ValueError, FileNotFoundError)):
        SocketSpec(source, CONTAINER_WAYLAND_SOCKET).to_mount_arg()


def test_none_does_not_discover_desktop_or_binary(sandbox: Sandbox, tmp_path: Path, capsys) -> None:
    """Default tasks do not acquire any new host dependency or filesystem access."""
    with (
        patch(_RESOLVER) as resolver,
        patch("terok_sandbox.wayland.find_host_tool") as lookup,
    ):
        spec = _spec(tmp_path, WaylandMode.NONE)
        assert sandbox._prepare_wayland(spec) == (spec, False)
        assert resolve_session(WaylandMode.NONE) == WaylandSession()
        observer = MagicMock()
        warn_desktop_access(WaylandMode.NONE, observer=observer)
    resolver.assert_not_called()
    lookup.assert_not_called()
    observer.assert_not_called()
    assert not capsys.readouterr().err


@pytest.mark.parametrize("mode", [WaylandMode.MITM, WaylandMode.DIRECT])
@pytest.mark.parametrize(("tty", "no_color"), [(True, False), (False, False), (True, True)])
def test_only_direct_desktop_warning_prefix_is_red_on_a_terminal(
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture,
    mode: WaylandMode,
    tty: bool,
    no_color: bool,
) -> None:
    """Only trusted prefix styling reaches the terminal; observers always get plain text."""
    monkeypatch.setattr("sys.stderr.isatty", lambda: tty)
    monkeypatch.setenv("FORCE_COLOR", "1")
    if no_color:
        monkeypatch.setenv("NO_COLOR", "1")
    else:
        monkeypatch.delenv("NO_COLOR", raising=False)
    observed: list[str] = []
    warn_desktop_access(mode, observer=observed.append)
    plain = observed[0]
    assert "\x1b" not in plain
    expected = (
        plain.replace("WARNING:", "\x1b[31mWARNING:\x1b[0m", 1)
        if mode == WaylandMode.DIRECT and tty and not no_color
        else plain
    )
    assert capsys.readouterr().err == expected + "\n"


@pytest.mark.parametrize(
    ("message", "highlight"),
    [
        ("WARNING: Wayland (wl-direct) unavailable: denied.", True),
        ("WARNING: Wayland (wl-direct): policy missing.", True),
        ("WARNING: unable to verify Wayland access.", False),
    ],
)
def test_direct_failure_warnings_use_the_same_plain_observer_contract(
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture,
    message: str,
    highlight: bool,
) -> None:
    """Initial direct failures match replay styling without coloring generic diagnostics."""
    monkeypatch.setattr("sys.stderr.isatty", lambda: True)
    monkeypatch.delenv("NO_COLOR", raising=False)
    observed: list[str] = []
    emit_warning(message, observer=observed.append)
    assert observed == [message]
    expected = message.replace("WARNING:", "\x1b[31mWARNING:\x1b[0m", 1) if highlight else message
    assert capsys.readouterr().err == expected + "\n"


@pytest.mark.parametrize("mode", [WaylandMode.MITM, WaylandMode.DIRECT])
@pytest.mark.parametrize("absolute", [False, True])
def test_desktop_session_resolves_owned_socket_and_selected_binary(
    session_socket: Path, monkeypatch: pytest.MonkeyPatch, mode: WaylandMode, absolute: bool
) -> None:
    """Both display spellings identify one socket; only filtered access discovers a proxy."""
    if absolute:
        monkeypatch.setenv("WAYLAND_DISPLAY", str(session_socket))
        monkeypatch.delenv("XDG_RUNTIME_DIR")
    binary = session_socket.parent / "proxy-binary"
    binary.touch(mode=0o700)
    alias = session_socket.parent / "wl-mitm"
    alias.symlink_to(binary)
    with patch("terok_sandbox.wayland.find_host_tool", return_value=str(alias)) as lookup:
        session = resolve_session(mode)
    assert session.upstream == session_socket and session.error is None
    assert session.binary == (binary if mode == WaylandMode.MITM else None)
    assert session.proxied is (mode == WaylandMode.MITM)
    if mode == WaylandMode.MITM:
        lookup.assert_called_once_with("wl-mitm")
    else:
        lookup.assert_not_called()


def test_filtered_session_never_falls_back_when_proxy_missing(session_socket: Path) -> None:
    """A valid desktop socket does not authorize direct access when filtering cannot start."""
    with patch("terok_sandbox.wayland.find_host_tool", return_value=None) as lookup:
        session = resolve_session(WaylandMode.MITM)
    lookup.assert_called_once_with("wl-mitm")
    assert session.upstream is None and session.binary is None and not session.proxied
    assert session.error and "install it on the host" in session.error


def test_root_invocation_cannot_select_a_user_desktop(session_socket: Path) -> None:
    """Even a valid selected desktop must be invoked as the unprivileged desktop user."""
    with (
        patch("terok_sandbox.wayland.os.geteuid", return_value=0),
        patch("terok_sandbox.wayland._session_socket") as discover,
    ):
        session = resolve_session(WaylandMode.DIRECT)
    discover.assert_not_called()
    assert session.upstream is None and session.error and "rootless" in session.error


@pytest.mark.parametrize(
    ("display", "runtime", "reason"),
    [
        (None, None, "WAYLAND_DISPLAY is unset"),
        ("../wayland-0", None, "socket name or an absolute path"),
        (".", None, "socket name or an absolute path"),
        ("..", None, "socket name or an absolute path"),
        ("wayland-0", "relative-runtime", "XDG_RUNTIME_DIR must be absolute"),
        ("wayland-0", "", "XDG_RUNTIME_DIR must be absolute"),
    ],
)
def test_session_requires_an_unambiguous_desktop_endpoint(
    session_socket: Path,
    monkeypatch: pytest.MonkeyPatch,
    display: str | None,
    runtime: str | None,
    reason: str,
) -> None:
    """Missing or relative session coordinates never trigger socket guessing."""
    if display is None:
        monkeypatch.delenv("WAYLAND_DISPLAY")
    else:
        monkeypatch.setenv("WAYLAND_DISPLAY", display)
    if runtime is not None:
        monkeypatch.setenv("XDG_RUNTIME_DIR", runtime)
    session = resolve_session(WaylandMode.DIRECT)
    assert session.upstream is None and session.error and reason in session.error


@pytest.mark.parametrize("permissions", [0o770, 0o707])
def test_other_users_cannot_replace_socket_in_its_parent(
    session_socket: Path, permissions: int
) -> None:
    """A user-owned socket is insufficient when its directory permits replacement."""
    session_socket.parent.chmod(permissions)
    session = resolve_session(WaylandMode.DIRECT)
    assert session.upstream is None and session.error and "socket directory" in session.error


@pytest.mark.parametrize(
    ("owned_path", "reason"),
    [("parent", "socket directory"), ("ancestor", "ancestor"), ("socket", "not a socket owned")],
)
def test_foreign_ownership_cannot_become_desktop_access(
    session_socket: Path, owned_path: str, reason: str
) -> None:
    """Reject another user's socket or replaceable directories without privileged chown."""
    target = {
        "parent": session_socket.parent,
        "ancestor": session_socket.parent.parent,
        "socket": session_socket,
    }[owned_path]
    method = "lstat" if owned_path == "socket" else "stat"
    original = getattr(Path, method)
    foreign_uid = os.geteuid() + 1

    def foreign_owner(path: Path, *args, **kwargs):
        """Change only the selected inode's ownership, not the calling user's identity."""
        info = original(path, *args, **kwargs)
        if path == target:
            fields = list(info)
            fields[stat.ST_UID] = foreign_uid
            return os.stat_result(fields)
        return info

    with patch.object(Path, method, foreign_owner):
        session = resolve_session(WaylandMode.DIRECT)
    assert session.upstream is None and session.error and reason in session.error


@pytest.mark.parametrize("kind", ["file", "directory", "symlink"])
def test_session_rejects_non_socket_endpoint(
    session_socket: Path, monkeypatch: pytest.MonkeyPatch, kind: str
) -> None:
    """Only the socket inode itself may be selected, never a file or a redirecting symlink."""
    candidate = session_socket.parent / kind
    if kind == "file":
        candidate.touch()
    elif kind == "directory":
        candidate.mkdir()
    else:
        candidate.symlink_to(session_socket)
    monkeypatch.setenv("WAYLAND_DISPLAY", candidate.name)
    session = resolve_session(WaylandMode.DIRECT)
    assert session.upstream is None and session.error and "not a socket owned" in session.error


def test_session_requires_socket_connect_permission(session_socket: Path) -> None:
    """Ownership alone does not grant the write access required for a Unix connection."""
    with patch("terok_sandbox.wayland.os.access", return_value=False) as access:
        session = resolve_session(WaylandMode.DIRECT)
    access.assert_called_once_with(session_socket, os.W_OK)
    assert session.upstream is None and session.error and "cannot access" in session.error


@pytest.mark.parametrize("field", ["upstream", "binary"])
def test_saved_desktop_paths_cannot_depend_on_working_directory(field: str) -> None:
    """A relative sidecar path must not select a different endpoint or executable on restart."""
    session = WaylandSession.from_payload({"mode": "wl-mitm", field: "relative-path"})
    assert session.mode == WaylandMode.NONE and not session.proxied
    assert session.error and "paths must be absolute" in session.error


@pytest.mark.parametrize("mode", [WaylandMode.MITM, WaylandMode.DIRECT])
def test_selected_mode_reaches_mount_and_environment(
    sandbox: Sandbox,
    desktop_socket: Path,
    tmp_path: Path,
    mode: WaylandMode,
) -> None:
    """Only the selected endpoint enters the task; the desktop directory never does."""
    session = WaylandSession(mode, desktop_socket, tmp_path / "wl-mitm")
    with patch(_RESOLVER, return_value=session):
        spec, _ = sandbox._prepare_wayland(_spec(tmp_path, mode))
    assert spec.env["WAYLAND_DISPLAY"] == CONTAINER_WAYLAND_SOCKET
    if mode == WaylandMode.DIRECT:
        assert spec.sockets == (SocketSpec(desktop_socket, CONTAINER_WAYLAND_SOCKET),)
        assert not spec.volumes
    else:
        assert not spec.sockets
        assert len(spec.volumes) == 1
        assert spec.volumes[0].host_path != desktop_socket.parent
        assert spec.volumes[0].read_only and spec.volumes[0].live
    saved = load_sidecar(sandbox.config.sidecar_path(_NAME))
    assert saved is not None and saved.wayland == session


@pytest.mark.parametrize("mode", [WaylandMode.MITM, WaylandMode.DIRECT])
def test_desktop_applications_get_a_private_user_runtime(
    sandbox: Sandbox, desktop_socket: Path, tmp_path: Path, mode: WaylandMode
) -> None:
    """Runtime state is empty, user-owned tmpfs, not the host desktop's runtime directory."""
    original = replace(
        _spec(tmp_path, mode),
        env={"XDG_RUNTIME_DIR": str(desktop_socket.parent), "KEEP": "value"},
        extra_args=("--user", "1234:5678"),
    )
    session = WaylandSession(mode, desktop_socket, tmp_path / "wl-mitm")
    with patch(_RESOLVER, return_value=session):
        prepared, _ = sandbox._prepare_wayland(original)
    assert prepared.env == {
        "KEEP": "value",
        "WAYLAND_DISPLAY": CONTAINER_WAYLAND_SOCKET,
        "XDG_RUNTIME_DIR": CONTAINER_USER_RUNTIME_DIR,
    }
    assert prepared.extra_args[:2] == original.extra_args
    assert prepared.extra_args[2] == "--mount"
    assert set(prepared.extra_args[3].split(",")) == {
        "type=tmpfs",
        f"destination={CONTAINER_USER_RUNTIME_DIR}",
        "tmpfs-mode=0700",
        "U",
        "nosuid",
        "nodev",
        "noexec",
        "notmpcopyup",
    }
    with patch.object(sandbox, "pre_start_args", return_value=[]):
        command = sandbox._build_cmd(prepared)
    assert command[command.index("--mount") + 1] == prepared.extra_args[3]
    assert f"XDG_RUNTIME_DIR={CONTAINER_USER_RUNTIME_DIR}" in command
    assert original.env["XDG_RUNTIME_DIR"] == str(desktop_socket.parent)


@pytest.mark.parametrize("mode", [WaylandMode.MITM, WaylandMode.DIRECT])
def test_runtime_mount_rolls_back_when_desktop_wiring_cannot_be_recorded(
    sandbox: Sandbox, desktop_socket: Path, tmp_path: Path, mode: WaylandMode
) -> None:
    """Failed desktop preparation leaves no application runtime mount or environment behind."""
    original = _spec(tmp_path, mode)
    session = WaylandSession(mode, desktop_socket, tmp_path / "wl-mitm")
    with (
        patch(_RESOLVER, return_value=session),
        patch("terok_sandbox.supervisor.sidecar.record_wayland", side_effect=OSError("unwritable")),
    ):
        prepared, _ = sandbox._prepare_wayland(original)
    assert prepared is original


def test_proxy_directory_is_stable_even_when_desktop_unavailable(
    sandbox: Sandbox,
    tmp_path: Path,
    capsys: pytest.CaptureFixture,
) -> None:
    """A missing initial dependency may recover on restart without changing mounts."""
    with patch(_RESOLVER, return_value=WaylandSession(WaylandMode.MITM, error="no desktop")):
        spec, _ = sandbox._prepare_wayland(_spec(tmp_path))
    assert spec.volumes and not spec.sockets
    assert spec.env["XDG_RUNTIME_DIR"] == CONTAINER_USER_RUNTIME_DIR
    assert "--mount" in spec.extra_args
    assert "no desktop" in capsys.readouterr().err
    saved = load_sidecar(sandbox.config.sidecar_path(_NAME))
    assert saved is not None and "wayland" not in wired_services(saved)


@pytest.mark.parametrize("kwargs", [{"sealed": True}, {"runtime": "krun"}])
@pytest.mark.parametrize("mode", [WaylandMode.MITM, WaylandMode.DIRECT])
def test_unsupported_isolation_cannot_enable_gui_on_restart(
    sandbox: Sandbox,
    tmp_path: Path,
    kwargs: dict,
    mode: WaylandMode,
) -> None:
    """Session refresh cannot add a grant that container creation refused."""
    spec, _ = sandbox._prepare_wayland(_spec(tmp_path, mode, **kwargs))
    assert not spec.sockets and not spec.volumes
    assert not spec.extra_args and "XDG_RUNTIME_DIR" not in spec.env
    sandbox._refresh_wayland(_NAME)
    saved = load_sidecar(sandbox.config.sidecar_path(_NAME))
    assert saved is not None and saved.wayland.disabled_reason
    assert "wayland" not in wired_services(saved)


def test_proxy_restart_reresolves_session(sandbox: Sandbox, tmp_path: Path) -> None:
    """A host reboot or a new desktop session changes upstream, not the proxy mount."""
    original = WaylandSession(WaylandMode.MITM, tmp_path / "old", tmp_path / "wl-mitm")
    current = replace(original, upstream=tmp_path / "new")
    with patch(_RESOLVER, side_effect=[original, current]) as resolve:
        spec, _ = sandbox._prepare_wayland(_spec(tmp_path))
        sandbox._refresh_wayland(_NAME)
    assert resolve.call_count == 2
    saved = load_sidecar(sandbox.config.sidecar_path(_NAME))
    assert saved is not None and saved.wayland.upstream == current.upstream
    assert spec.volumes[0].host_path.is_dir()


@pytest.mark.parametrize("change", ["missing", "different", "initially-missing"])
def test_raw_restart_requires_recreation_for_changed_source(
    sandbox: Sandbox,
    desktop_socket: Path,
    tmp_path: Path,
    change: str,
) -> None:
    """Raw mounts never silently retarget or trigger destructive recreation."""
    old = WaylandSession(WaylandMode.DIRECT, desktop_socket)
    new = replace(old, upstream=None if change == "missing" else tmp_path / "new")
    if change == "initially-missing":
        old = replace(old, upstream=None, error="no desktop")
    with patch(_RESOLVER, side_effect=[old, new]):
        sandbox._prepare_wayland(_spec(tmp_path, WaylandMode.DIRECT))
        with pytest.raises(WaylandRecreationRequired, match="recreation required"):
            sandbox.start(_NAME)
    sandbox.runtime.container.assert_not_called()


def test_same_raw_path_restarts(sandbox: Sandbox, desktop_socket: Path, tmp_path: Path) -> None:
    """A reboot-created socket at the same source needs no mount retargeting."""
    session = WaylandSession(WaylandMode.DIRECT, desktop_socket)
    with patch(_RESOLVER, return_value=session), patch.object(sandbox, "_verify_supervision"):
        sandbox._prepare_wayland(_spec(tmp_path, WaylandMode.DIRECT))
        sandbox.start(_NAME)
    sandbox.runtime.container.return_value.start.assert_called_once()


def test_tcp_services_keep_wayland_socket_endpoint(sandbox: Sandbox, tmp_path: Path) -> None:
    """Wayland is always local socket IPC, independent of the other service transport."""
    path = sandbox.config.sidecar_path(_NAME)
    session = WaylandSession(WaylandMode.MITM, tmp_path / "upstream", tmp_path / "wl-mitm")
    record_wayland(path, session)
    saved = load_sidecar(path)
    assert saved is not None
    saved = replace(
        saved, ipc_mode="tcp", tcp_port=FAKE_SERVICE_PORTS[0], ssh_signer_port=FAKE_SERVICE_PORTS[1]
    )
    paths = SupervisorPaths.for_container("abcdef", _NAME, path, saved.runtime_dir)
    endpoints = _expected_endpoints(saved, paths)
    assert next(e for e in endpoints if e.service == "wayland").socket == paths.wayland_socket
    assert all(e.port for e in endpoints if e.service != "wayland")


def test_gui_probe_failure_is_nonfatal(sandbox: Sandbox, tmp_path: Path, capsys) -> None:
    """A container-domain denial is loud without stopping the task or claiming SELinux."""
    session = WaylandSession(WaylandMode.MITM, tmp_path / "upstream", tmp_path / "wl-mitm")
    with patch(_RESOLVER, return_value=session):
        sandbox._prepare_wayland(_spec(tmp_path))
    sandbox.runtime.exec.return_value = ExecResult(1, "", "Permission denied")
    sandbox._verify_wayland(_NAME)
    warning = capsys.readouterr().err
    assert "Permission denied" in warning and "Task continues" in warning
    assert "SELinux" not in warning


def test_malformed_optional_state_keeps_other_services(sandbox: Sandbox) -> None:
    """A corrupt optional field never costs the whole supervisor its sidecar."""
    path = sandbox.config.sidecar_path(_NAME)
    raw = json.loads(path.read_text())
    raw["wayland"] = {"mode": "invalid"}
    store_sidecar(path, raw)
    saved = load_sidecar(path)
    assert saved is not None
    assert "vault" in wired_services(saved) and "wayland" not in wired_services(saved)


def test_raw_failed_mount_does_not_record_a_grant(
    sandbox: Sandbox, desktop_socket: Path, tmp_path: Path
) -> None:
    """A socket disappearing during preparation cannot become a saved bind source."""
    session = WaylandSession(WaylandMode.DIRECT, desktop_socket)
    with (
        patch(_RESOLVER, return_value=session),
        patch.object(SocketSpec, "to_mount_arg", side_effect=FileNotFoundError("socket vanished")),
    ):
        prepared, _ = sandbox._prepare_wayland(_spec(tmp_path, WaylandMode.DIRECT))
    assert not prepared.sockets
    assert not prepared.extra_args and "XDG_RUNTIME_DIR" not in prepared.env
    saved = load_sidecar(sandbox.config.sidecar_path(_NAME))
    assert saved is not None and saved.wayland.upstream is None
    with patch(_RESOLVER, return_value=session), pytest.raises(WaylandRecreationRequired):
        sandbox._refresh_wayland(_NAME)


@pytest.mark.parametrize("variable", ["CONTAINER_HOST", "CONTAINER_CONNECTION"])
def test_remote_podman_never_mounts_a_local_desktop(
    sandbox: Sandbox, tmp_path: Path, monkeypatch, variable: str
) -> None:
    """A remote host's coincidentally identical pathname is not the invoking desktop."""
    monkeypatch.setenv(variable, "remote-selection")
    prepared, _ = sandbox._prepare_wayland(_spec(tmp_path))
    assert not prepared.volumes and not prepared.sockets
    saved = load_sidecar(sandbox.config.sidecar_path(_NAME))
    assert saved is not None and saved.wayland.disabled_reason


async def test_proxy_wrapper_transitions_before_arming_its_death_signal(tmp_path: Path) -> None:
    """SELinux exec transitions precede the wrapper, not the external proxy's signal."""
    process = MagicMock(stdout=MagicMock(read=AsyncMock(return_value=b"")))
    with patch(
        "terok_sandbox.supervisor.launcher.asyncio.create_subprocess_exec", return_value=process
    ) as spawn:
        handle = await launch_child("wayland", "container-id", tmp_path / "sidecar.json")
        await finish_output(handle.output)
    args = spawn.call_args.kwargs
    assert args["preexec_fn"].__name__ == "prepare_wayland_exec_context"
    assert args["env"][EXPECTED_PARENT_ENV] == str(os.getpid())
    assert args["stdout"] == asyncio.subprocess.PIPE
    assert args["stderr"] == asyncio.subprocess.STDOUT


def test_wayland_wrapper_logging_never_contacts_the_journal(tmp_path: Path) -> None:
    """The confined wrapper sends only its pipe; its supervisor owns host logging."""
    from terok_sandbox.cli import main

    with (
        patch("terok_util.configure") as configure,
        patch("logging.basicConfig") as basic,
        patch("terok_sandbox.supervisor.children.run_child", return_value=0),
        pytest.raises(SystemExit) as exit_status,
    ):
        main(["supervise-child", "wayland", "container-id", str(tmp_path / "sidecar.json")])
    assert exit_status.value.code == 0
    configure.assert_not_called()
    basic.assert_called_once()


def test_missing_probe_interpreter_does_not_claim_gui_is_broken(
    sandbox: Sandbox, tmp_path: Path, capsys
) -> None:
    """A bare custom image may lack Python while its GUI endpoint still works."""
    session = WaylandSession(WaylandMode.MITM, tmp_path / "upstream", tmp_path / "wl-mitm")
    record_wayland(sandbox.config.sidecar_path(_NAME), session)
    sandbox.runtime.exec.return_value = ExecResult(127, "", "python3 not found")
    sandbox._verify_wayland(_NAME)
    warning = capsys.readouterr().err
    assert "unable to verify" in warning
    assert "without GUI access" not in warning


@pytest.mark.parametrize("mode", [WaylandMode.MITM, WaylandMode.DIRECT])
def test_preflight_failure_warns_once_without_probing(
    sandbox: Sandbox, tmp_path: Path, capsys, mode: WaylandMode
) -> None:
    """An unavailable desktop does not trigger a second, misleading endpoint failure."""
    observed: list[str] = []
    session = WaylandSession(mode, error="WAYLAND_DISPLAY is unset")
    with (
        patch(_RESOLVER, return_value=session),
        patch("terok_sandbox.sandbox.podman_hosts_file_args", return_value=[]),
        patch.object(sandbox, "pre_start_args", return_value=[]),
        patch.object(sandbox, "_exec_podman") as launch,
        patch.object(sandbox, "_verify_supervision"),
    ):
        sandbox.run(_spec(tmp_path, mode), hooks=LifecycleHooks(warning=observed.append))
    launch.assert_called_once()
    sandbox.runtime.exec.assert_not_called()
    assert observed == capsys.readouterr().err.splitlines()
    assert len(observed) == 2
    assert "significantly increases host attack surface" in observed[0]
    assert "WAYLAND_DISPLAY is unset" in observed[1]
    if mode == WaylandMode.DIRECT:
        assert "without protocol filtering" in observed[0]


@pytest.mark.parametrize(
    "session",
    [
        WaylandSession(WaylandMode.NONE),
        WaylandSession(WaylandMode.DIRECT),
        WaylandSession(WaylandMode.MITM, error="no binary"),
        WaylandSession(WaylandMode.MITM, disabled_reason="sealed"),
    ],
)
def test_unwired_desktop_never_probes(sandbox: Sandbox, session: WaylandSession) -> None:
    """The probe can verify only an endpoint actually prepared for this start."""
    record_wayland(sandbox.config.sidecar_path(_NAME), session)
    sandbox._verify_wayland(_NAME)
    sandbox.runtime.exec.assert_not_called()


@pytest.mark.parametrize("exit_code", [0, 1, 127])
def test_probe_warning_observer_receives_compact_original_message(
    sandbox: Sandbox, tmp_path: Path, capsys, exit_code: int
) -> None:
    """Successful probes stay quiet; failures retain the useful cause, not a traceback."""
    session = WaylandSession(WaylandMode.MITM, tmp_path / "upstream", tmp_path / "wl-mitm")
    record_wayland(sandbox.config.sidecar_path(_NAME), session)
    sandbox.runtime.exec.return_value = ExecResult(
        exit_code,
        "",
        'Traceback (most recent call last):\n  File "<string>"\nPermissionError: denied',
    )
    observed: list[str] = []
    sandbox._verify_wayland(_NAME, hooks=LifecycleHooks(warning=observed.append))
    sandbox.runtime.exec.assert_called_once()
    assert observed == capsys.readouterr().err.splitlines()
    assert len(observed) == int(exit_code != 0)
    if observed:
        assert "PermissionError: denied" in observed[0]
        assert "Traceback" not in observed[0]


@pytest.mark.parametrize("outcome", [1, 127, "exception"])
def test_hostile_probe_diagnostics_are_safe_to_print_and_repeat(
    sandbox: Sandbox, tmp_path: Path, capsys, caplog, outcome: int | str
) -> None:
    """Task-controlled diagnostics cannot execute terminal commands through either output sink."""
    record_wayland(
        sandbox.config.sidecar_path(_NAME),
        WaylandSession(WaylandMode.MITM, tmp_path / "upstream", tmp_path / "wl-mitm"),
    )
    detail = (
        "Traceback (most recent call last):\n"
        "PermissionError: denied \x1b]52;c;Y2xpcA==\x07\x1b[2J\x1b[31mred\x1b[0m \u202espoof"
    )
    if outcome == "exception":
        sandbox.runtime.exec.side_effect = RuntimeError(detail)
    else:
        sandbox.runtime.exec.return_value = ExecResult(outcome, "", detail)
    observed: list[str] = []
    with caplog.at_level(logging.DEBUG, logger="terok_sandbox.sandbox"):
        sandbox._verify_wayland(_NAME, hooks=LifecycleHooks(warning=observed.append))

    assert observed == capsys.readouterr().err.splitlines()
    assert len(observed) == 1 and "PermissionError: denied" in observed[0]
    assert "Traceback" not in observed[0] and r"\x1b" in observed[0]
    messages = [*observed, *caplog.messages]
    assert all(
        control not in message for message in messages for control in ("\x1b", "\x07", "\u202e")
    )
    if outcome != "exception":
        assert "Traceback" in caplog.text and "\nPermissionError: denied" in caplog.text


def test_probe_timeout_does_not_dump_embedded_script(sandbox: Sandbox, tmp_path: Path) -> None:
    """A runtime timeout reports the problem, not the entire Python probe command."""
    record_wayland(
        sandbox.config.sidecar_path(_NAME),
        WaylandSession(WaylandMode.MITM, tmp_path / "upstream", tmp_path / "wl-mitm"),
    )
    sandbox.runtime.exec.side_effect = subprocess.TimeoutExpired("embedded-probe-script", 3)
    observed: list[str] = []
    sandbox._verify_wayland(_NAME, hooks=LifecycleHooks(warning=observed.append))
    assert len(observed) == 1 and "timed out" in observed[0]
    assert "embedded-probe-script" not in observed[0]


def test_restart_preflight_warning_reaches_observer(sandbox: Sandbox, tmp_path: Path) -> None:
    """The common restart path forwards the refreshed failure without probing again."""
    record_wayland(
        sandbox.config.sidecar_path(_NAME),
        WaylandSession(WaylandMode.MITM, tmp_path / "upstream", tmp_path / "wl-mitm"),
    )
    observed: list[str] = []
    with (
        patch(_RESOLVER, return_value=WaylandSession(WaylandMode.MITM, error="no desktop")),
        patch.object(sandbox, "_verify_supervision"),
        patch.object(sandbox, "_warn_if_outdated"),
    ):
        sandbox.start(_NAME, hooks=LifecycleHooks(warning=observed.append))
    sandbox.runtime.container.return_value.start.assert_called_once()
    sandbox.runtime.exec.assert_not_called()
    assert "no desktop" in observed[-1]


def test_warning_observer_cannot_stop_startup(sandbox: Sandbox, tmp_path: Path) -> None:
    """An application reporting bug never promotes optional GUI failure to task failure."""
    observer = MagicMock(side_effect=RuntimeError("observer failed"))
    with (
        patch(_RESOLVER, return_value=WaylandSession(WaylandMode.MITM, error="no desktop")),
        patch.object(sandbox, "pre_start_args", return_value=[]),
        patch.object(sandbox, "_exec_podman") as launch,
        patch.object(sandbox, "_verify_supervision"),
    ):
        sandbox.run(_spec(tmp_path), hooks=LifecycleHooks(warning=observer))
    launch.assert_called_once()
    assert observer.call_count == 2


@pytest.mark.parametrize("verb", ["run", "create"])
@pytest.mark.parametrize(
    ("mode", "installed", "endpoint", "selected"),
    [
        (WaylandMode.DIRECT, True, True, True),
        (WaylandMode.DIRECT, False, True, False),
        (WaylandMode.DIRECT, True, False, False),
        (WaylandMode.MITM, True, True, False),
        (WaylandMode.NONE, True, True, False),
    ],
)
def test_direct_domain_is_selected_only_for_valid_opt_in_mount(
    sandbox: Sandbox,
    desktop_socket: Path,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    verb: str,
    mode: WaylandMode,
    installed: bool,
    endpoint: bool,
    selected: bool,
) -> None:
    """Only the direct process type changes; Podman still chooses the MCS range."""
    monkeypatch.setattr(_selinux, "is_selinux_enabled", lambda: True)
    monkeypatch.setattr(_selinux, "is_wayland_direct_policy_installed", lambda: installed)
    session = WaylandSession(
        mode, desktop_socket if endpoint else None, error=None if endpoint else "no desktop"
    )
    observed: list[str] = []
    with (
        patch(_RESOLVER, return_value=session),
        patch.object(sandbox, "pre_start_args", return_value=[]),
        patch.object(sandbox, "_exec_podman") as launch,
        patch.object(sandbox, "_verify_supervision"),
        patch.object(sandbox, "_verify_wayland"),
    ):
        getattr(sandbox, verb)(_spec(tmp_path, mode), hooks=LifecycleHooks(warning=observed.append))
    command = launch.call_args.args[0]
    labels = [arg for arg in command if arg.startswith("label=")]
    assert labels == ([f"label=type:{_selinux.SELINUX_WAYLAND_DIRECT_TYPE}"] if selected else [])
    assert "--privileged" not in command
    saved = load_sidecar(sandbox.config.sidecar_path(_NAME))
    assert saved is not None and saved.wayland.direct_selinux is selected
    if mode == WaylandMode.DIRECT and endpoint and not installed:
        assert "--with-wayland-direct" in observed[-1]
        assert "normal SELinux domain" in observed[-1]
    if mode == WaylandMode.NONE:
        assert not observed


@pytest.mark.parametrize(("selected", "installed"), [(True, True), (True, False), (False, True)])
def test_restart_keeps_creation_time_direct_domain(
    sandbox: Sandbox,
    desktop_socket: Path,
    monkeypatch: pytest.MonkeyPatch,
    selected: bool,
    installed: bool,
) -> None:
    """Policy removal requires restoration or recreation; late installation cannot relabel."""
    monkeypatch.setattr(_selinux, "is_selinux_enabled", lambda: True)
    monkeypatch.setattr(_selinux, "is_wayland_direct_policy_installed", lambda: installed)
    session = WaylandSession(WaylandMode.DIRECT, desktop_socket, direct_selinux=selected)
    record_wayland(sandbox.config.sidecar_path(_NAME), session)
    observed: list[str] = []
    with (
        patch(_RESOLVER, return_value=replace(session, direct_selinux=False)),
        patch.object(sandbox, "_verify_supervision"),
        patch.object(sandbox, "_verify_wayland"),
        patch.object(sandbox, "_warn_if_outdated"),
    ):
        if selected and not installed:
            with pytest.raises(WaylandRecreationRequired, match="restore.*recreate"):
                sandbox.start(_NAME)
            sandbox.runtime.container.assert_not_called()
        else:
            sandbox.start(_NAME, hooks=LifecycleHooks(warning=observed.append))
            sandbox.runtime.container.return_value.start.assert_called_once()
            if not selected:
                assert "created without the direct-access SELinux domain" in observed[-1]
    saved = load_sidecar(sandbox.config.sidecar_path(_NAME))
    assert saved is not None and saved.wayland.direct_selinux is selected


def test_invalid_direct_domain_metadata_records_an_error() -> None:
    """A string cannot turn an optional policy decision into a saved boolean grant."""
    session = WaylandSession.from_payload({"mode": "wl-direct", "direct_selinux": "true"})
    assert session.mode == WaylandMode.NONE and session.error


@pytest.mark.parametrize(
    "wayland",
    [
        {"mode": "wl-unsafe"},
        {"mode": "wl-filtered"},
        "malformed",
        {"mode": "wl-direct", "direct_selinux": "true"},
    ],
)
def test_unknown_saved_wayland_grant_requires_recreation(sandbox: Sandbox, wayland: object) -> None:
    """Unreadable metadata cannot revoke an existing mount; never silently resume it."""
    path = sandbox.config.sidecar_path(_NAME)
    payload = json.loads(path.read_text())
    payload["wayland"] = wayland
    store_sidecar(path, payload)
    original = path.read_bytes()
    with (
        patch(_RESOLVER) as resolve,
        pytest.raises(WaylandRecreationRequired, match="cannot verify saved desktop access"),
    ):
        sandbox.start(_NAME)
    resolve.assert_not_called()
    sandbox.runtime.container.assert_not_called()
    assert path.read_bytes() == original
