# SPDX-FileCopyrightText: 2026 Jiri Vyskocil
# SPDX-License-Identifier: Apache-2.0

"""Invalid desktop mounts and optional state failures remain visible without adding grants."""

from dataclasses import replace
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from terok_sandbox import LifecycleHooks, RunSpec, Sandbox, SocketSpec, WaylandMode
from terok_sandbox.podman_args import CONTAINER_WAYLAND_SOCKET
from terok_sandbox.wayland import WaylandSession


@pytest.fixture
def spec(tmp_path: Path) -> RunSpec:
    """A desktop request with no pre-existing mount grants."""
    return RunSpec(
        container_name="desktop-diagnostics",
        image="test-image",
        env={},
        volumes=(),
        command=(),
        task_dir=tmp_path,
        wayland=WaylandMode.MITM,
    )


@pytest.mark.parametrize("host_coordinate", [False, True])
@pytest.mark.parametrize("invalid", ["relative", *':,"\r\n\0'])
def test_socket_coordinates_cannot_change_mount_options(
    tmp_path: Path, host_coordinate: bool, invalid: str
) -> None:
    """Both coordinates reject relative paths and every mount-option delimiter before stat."""
    source, destination = tmp_path / "socket", CONTAINER_WAYLAND_SOCKET
    coordinate = Path("relative") if invalid == "relative" else tmp_path / f"socket{invalid}"
    if host_coordinate:
        source = coordinate
    else:
        destination = str(coordinate)
    reason = "must be absolute" if invalid == "relative" else "mount-option delimiters"
    with pytest.raises(ValueError, match=reason):
        SocketSpec(source, destination).to_mount_arg()


def test_sealed_creation_rejects_explicit_host_socket(spec: RunSpec) -> None:
    """A caller cannot bypass sealed isolation by supplying its own SocketSpec."""
    socket = SocketSpec(spec.task_dir / "socket", CONTAINER_WAYLAND_SOCKET)
    sandbox = Sandbox(runtime=MagicMock())
    with (
        patch.object(sandbox, "pre_start_args", return_value=[]),
        pytest.raises(ValueError, match="existing host sockets are not supported in sealed"),
    ):
        sandbox._build_cmd(replace(spec, sealed=True, sockets=(socket,)))


@pytest.mark.parametrize("failure", [OSError, ValueError])
def test_unrecorded_session_drops_prepared_mounts(
    spec: RunSpec, failure: type[Exception], capsys: pytest.CaptureFixture[str]
) -> None:
    """A failed state write cannot leave an untracked desktop grant on the container."""
    sandbox = Sandbox(runtime=MagicMock())
    warnings: list[str] = []
    session = WaylandSession(spec.wayland, spec.task_dir / "upstream", spec.task_dir / "wl-mitm")
    with (
        patch("terok_sandbox.wayland.resolve_session", return_value=session),
        patch(
            "terok_sandbox.supervisor.sidecar.record_wayland", side_effect=failure("state denied")
        ),
    ):
        prepared, direct_selinux = sandbox._prepare_wayland(
            spec, hooks=LifecycleHooks(warning=warnings.append)
        )
    assert prepared is spec and not direct_selinux
    assert "cannot record desktop wiring: state denied" in warnings[-1]
    assert "Task continues" in warnings[-1]
    assert warnings[-1] in capsys.readouterr().err


@pytest.mark.parametrize("failure", [OSError, ValueError])
def test_session_refresh_failure_warns_without_stopping_restart(
    spec: RunSpec, failure: type[Exception], capsys: pytest.CaptureFixture[str]
) -> None:
    """Unreadable optional state remains a warning, including the deferred runner summary."""
    sandbox = Sandbox(runtime=MagicMock())
    path = sandbox.config.sidecar_path(spec.container_name)
    path.parent.mkdir(parents=True)
    path.touch()
    warnings: list[str] = []
    with patch(
        "terok_sandbox.supervisor.sidecar.refresh_wayland", side_effect=failure("state denied")
    ):
        sandbox._refresh_wayland(spec.container_name, hooks=LifecycleHooks(warning=warnings.append))
    assert len(warnings) == 1
    assert "could not refresh desktop session: state denied" in warnings[0]
    assert "Task continues" in warnings[0]
    assert warnings[0] in capsys.readouterr().err
