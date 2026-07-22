# SPDX-FileCopyrightText: 2026 Jiri Vyskocil
# SPDX-License-Identifier: Apache-2.0

"""Real supervisor re-fire under podman — the engine behind sickbay ``--fix``.

[`respawn_supervisor`][terok_sandbox.diagnostics.respawn_supervisor] re-invokes
the installed OCI hook to bring a dead per-container supervisor back; terok's
``sickbay --fix`` (terok-ai/terok#1189) drives it.  Unit tests mock the hook
call — this exercises the whole loop against real processes: install a rendered
hook + wrapper, boot a container for the supervisor to watch, respawn it, kill
the whole group, and respawn again, asserting
[`supervisor_liveness`][terok_sandbox.diagnostics.supervisor_liveness] throughout.

Why the shape:

* ``respawn_supervisor`` invokes the hook **directly**, so the container's
  podman ``hooks_dir`` registration is irrelevant — only the on-disk hook +
  rendered wrapper matter, and the operator's ``containers.conf`` is left
  untouched.
* The wrapper only stays up while its container is alive, so a real (``sleep``)
  container is required — hence ``needs_podman``.
* The install renders the wrapper with the resolved terok-sandbox argv, so the
  supervisor it launches can import the package.  The service children may
  degrade (there is no real vault DB behind the minimal sidecar), but the
  wrapper — what liveness tracks — stays up.
"""

from __future__ import annotations

import contextlib
import json
import os
import shutil
import signal
import subprocess
import time
import uuid
from collections.abc import Callable, Iterator
from pathlib import Path

import pytest

from terok_sandbox import respawn_supervisor, supervisor_liveness
from terok_sandbox.supervisor.install import install_supervisor_hooks
from tests.constants import PODMAN_BASE_IMAGE, PODMAN_PULL_TIMEOUT

pytestmark = [
    pytest.mark.needs_podman,
    pytest.mark.skipif(shutil.which("podman") is None, reason="podman not on PATH"),
]

_LAUNCH_TIMEOUT = 60
"""Seconds allowed for a ``podman run -d`` of the (pre-pulled) base image."""

_DOWN_TIMEOUT = 5.0
"""How long to wait for a SIGKILL'd supervisor group to read as dead."""


@pytest.fixture(scope="session")
def podman_image() -> str:
    """Ensure the base image is in the local store; return its reference."""
    if (
        subprocess.run(
            ["podman", "image", "exists", PODMAN_BASE_IMAGE], capture_output=True, check=False
        ).returncode
        == 0
    ):
        return PODMAN_BASE_IMAGE
    proc = subprocess.run(
        ["podman", "pull", PODMAN_BASE_IMAGE],
        capture_output=True,
        text=True,
        timeout=PODMAN_PULL_TIMEOUT,
        check=False,
    )
    if proc.returncode != 0:
        pytest.skip(f"cannot pull {PODMAN_BASE_IMAGE}: {proc.stderr.strip()}")
    return PODMAN_BASE_IMAGE


@pytest.fixture()
def installed_state(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    """A tmp state root holding a real, rendered hook + wrapper.

    ``state_root`` is redirected here and the ``containers.conf`` writer is
    stubbed, so the install is hermetic — nothing on the operator's host is
    touched, and the podman hook registration is not needed because the
    respawn re-invokes the hook directly.
    """
    state = tmp_path / "state"
    monkeypatch.setattr("terok_sandbox.supervisor.install.state_root", lambda: state)
    monkeypatch.setattr(
        "terok_sandbox.supervisor.install.ensure_user_hooks_dir_configured", lambda _dir: None
    )
    install_supervisor_hooks()
    return state


@pytest.fixture()
def sleeper(podman_image: str) -> Iterator[Callable[[], tuple[str, str, int]]]:
    """Yield a launcher for detached ``sleep`` containers; force-remove them after.

    Each launch returns ``(name, container_id, init_pid)`` — the id keys the
    supervisor PID file, and the init pid is what the respawned supervisor
    watches so it stays up.
    """
    names: list[str] = []

    def _inspect(name: str, fmt: str) -> str:
        return subprocess.run(
            ["podman", "inspect", "-f", fmt, name],
            capture_output=True,
            text=True,
            timeout=10,
            check=True,
        ).stdout.strip()

    def _launch() -> tuple[str, str, int]:
        name = f"terok-respawn-{uuid.uuid4().hex[:8]}"
        names.append(name)  # register before run: a failed launch may still exist
        subprocess.run(
            ["podman", "run", "-d", "--name", name, podman_image, "sleep", "3600"],
            check=True,
            capture_output=True,
            timeout=_LAUNCH_TIMEOUT,
        )
        return name, _inspect(name, "{{.Id}}"), int(_inspect(name, "{{.State.Pid}}"))

    yield _launch
    for name in names:
        subprocess.run(["podman", "rm", "-f", "-t", "0", name], capture_output=True, check=False)


def _write_sidecar(state: Path, name: str, tmp_path: Path) -> None:
    """Drop the minimal sidecar the hook validates (socket mode, no gate)."""
    sidecar = state / "sidecar" / f"{name}.json"
    sidecar.parent.mkdir(parents=True, exist_ok=True)
    sidecar.write_text(
        json.dumps(
            {
                "container_name": name,
                "ipc_mode": "socket",
                "db_path": str(tmp_path / "vault.db"),
                "runtime_dir": str(tmp_path / "rt"),
            }
        )
    )


def _kill_group(pgid: int | None) -> None:
    """SIGKILL a supervisor process group (the wrapper is its session leader)."""
    if pgid:
        with contextlib.suppress(ProcessLookupError, OSError):
            os.killpg(pgid, signal.SIGKILL)


def _wait_until_dead(cid: str, state: Path) -> None:
    """Poll until the supervisor reads as not-alive, or fail after the timeout."""
    deadline = time.monotonic() + _DOWN_TIMEOUT
    while supervisor_liveness(cid, state_dir=state).alive:
        if time.monotonic() >= deadline:
            pytest.fail("supervisor group survived SIGKILL past the timeout")
        time.sleep(0.1)


def test_respawn_recovers_a_killed_supervisor(
    installed_state: Path,
    sleeper: Callable[[], tuple[str, str, int]],
    tmp_path: Path,
) -> None:
    """Bring a supervisor up, kill it, and prove the re-fire recovers it."""
    name, cid, init_pid = sleeper()
    _write_sidecar(installed_state, name, tmp_path)

    # Nothing spawned yet — the container is up but unsupervised.
    assert supervisor_liveness(cid, state_dir=installed_state).alive is False

    spawned: list[int | None] = []
    try:
        # Re-fire the hook → spawns the real wrapper (respawn polls until it shows).
        first = respawn_supervisor(cid, name, state_dir=installed_state, container_pid=init_pid)
        spawned.append(first.pid)
        assert first.alive, f"supervisor did not come up: {first.detail}"

        # Kill the whole group → back to unsupervised, the #458 incident state.
        _kill_group(first.pid)
        _wait_until_dead(cid, installed_state)

        # The idempotent re-fire sickbay --fix drives recovers it.
        second = respawn_supervisor(cid, name, state_dir=installed_state, container_pid=init_pid)
        spawned.append(second.pid)
        assert second.alive, f"respawn did not recover the supervisor: {second.detail}"
        assert second.pid != first.pid, "expected a fresh supervisor, not the dead pid"
    finally:
        for pgid in spawned:
            _kill_group(pgid)
