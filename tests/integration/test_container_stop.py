# SPDX-FileCopyrightText: 2026 Jiri Vyskocil
# SPDX-License-Identifier: Apache-2.0

"""Stop semantics under real podman — with and without an init pid1.

The matrix replays this across the distro / podman-version spread,
covering both regimes managed launches can land in:

* ``--init``: catatonit is pid1 and forwards SIGTERM, so a stop returns
  long before the grace period is up.
* no init: the payload command is namespace-init and immune to SIGTERM
  — the degraded service a catatonit-less host gets.  The stop must
  still succeed, by burning the grace period and force-killing.

Slots without catatonit skip the ``--init`` case and prove exactly the
fallback their users would live with; slots with it prove both.
"""

from __future__ import annotations

import shutil
import subprocess
import time
import uuid
from collections.abc import Callable, Iterator

import pytest

from terok_sandbox import PodmanRuntime
from terok_sandbox.runtime.podman import find_init_binary
from tests.constants import PODMAN_BASE_IMAGE, PODMAN_PULL_TIMEOUT

pytestmark = [
    pytest.mark.needs_podman,
    pytest.mark.skipif(shutil.which("podman") is None, reason="podman not on PATH"),
]

GRACE = 2
"""Grace period (s) for the degraded case — short, since it is burned in full."""

INIT_GRACE = 30
"""Grace period (s) for the ``--init`` case — generous, since none of it
should be consumed; a stop that eats into it means SIGTERM went unheard."""

LAUNCH_TIMEOUT = 60
"""Seconds allowed for a ``podman run -d`` of the (pre-pulled) base image."""


@pytest.fixture(scope="session")
def podman_image() -> str:
    """Ensure the base image is in the local store; return its reference."""
    if (
        subprocess.run(
            ["podman", "image", "exists", PODMAN_BASE_IMAGE],
            capture_output=True,
            check=False,
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
def launch_sleeper(podman_image: str) -> Iterator[Callable[..., str]]:
    """Yield a launcher for detached ``sleep`` containers; force-remove them after.

    ``sleep`` is the ideal payload: it installs no signal handlers, so
    whether it dies on SIGTERM is decided purely by whether it runs as
    namespace-init — exactly the variable under test.
    """
    names: list[str] = []

    def _launch(*, init: bool) -> str:
        name = f"terok-sandbox-stoptest-{uuid.uuid4().hex[:8]}"
        # Registered before the run: a failed or timed-out launch can
        # still have created the container, and teardown must reap it.
        names.append(name)
        argv = ["podman", "run", "-d", "--name", name]
        if init:
            argv.append("--init")
        argv += [podman_image, "sleep", "3600"]
        subprocess.run(argv, check=True, capture_output=True, timeout=LAUNCH_TIMEOUT)
        return name

    yield _launch
    for name in names:
        subprocess.run(["podman", "rm", "-f", "-t", "0", name], capture_output=True, check=False)


def test_stop_with_init_returns_before_grace(launch_sleeper: Callable[..., str]) -> None:
    """catatonit forwards SIGTERM: the stop ends well inside the grace period."""
    if find_init_binary() is None:
        pytest.skip("catatonit not on this host — the no-init test covers it")
    name = launch_sleeper(init=True)
    container = PodmanRuntime().container(name)

    start = time.monotonic()
    container.stop(timeout=INIT_GRACE)
    elapsed = time.monotonic() - start

    assert container.state == "exited"
    assert elapsed < INIT_GRACE / 2, f"SIGTERM went unheard: stop took {elapsed:.1f}s"


def test_stop_without_init_burns_grace_then_kills(launch_sleeper: Callable[..., str]) -> None:
    """The degraded regime: pid1 ignores SIGTERM, yet the stop still succeeds."""
    name = launch_sleeper(init=False)
    container = PodmanRuntime().container(name)

    start = time.monotonic()
    container.stop(timeout=GRACE)
    elapsed = time.monotonic() - start

    assert container.state == "exited"
    assert elapsed >= GRACE, "namespace-init should have ignored SIGTERM"
