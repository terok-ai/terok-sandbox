# SPDX-FileCopyrightText: 2026 Jiri Vyskocil
# SPDX-License-Identifier: Apache-2.0

"""Tests for the supervisor hook installer.

Verifies the single-root layout
[`install_supervisor_hooks`][terok_sandbox.supervisor.install.install_supervisor_hooks]
writes — every artefact under ``state_root()`` except the OS-fixed
podman hook descriptor — and that the symmetric
[`uninstall_supervisor_hooks`][terok_sandbox.supervisor.install.uninstall_supervisor_hooks]
removes them.
"""

from __future__ import annotations

import json
from pathlib import Path
from unittest.mock import patch

import pytest

from terok_sandbox.supervisor.install import (
    install_supervisor_hooks,
    kill_all_supervisors,
    uninstall_supervisor_hooks,
)


@pytest.fixture
def install_env(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> dict[str, Path]:
    """Redirect ``state_root()`` to ``tmp_path``.

    Single layout: descriptors land next to scripts under
    ``state_root() / "hooks"``.  Overriding ``state_root`` keeps the
    test hermetic — install lands entirely under ``tmp_path``.
    """
    state = tmp_path / "state"
    monkeypatch.setattr("terok_sandbox.supervisor.install.state_root", lambda: state)
    monkeypatch.setattr(
        "terok_sandbox.supervisor.install.ensure_user_hooks_dir_configured",
        lambda _dir: None,
    )
    return {"state": state, "hooks_dir": state / "hooks"}


def test_install_lays_down_full_artefact_set(install_env: dict[str, Path]) -> None:
    """Every file the install promises shows up under ``state_root()``."""
    with patch(
        "terok_sandbox.supervisor.install._resolve_sandbox_argv",
        return_value=["/usr/local/bin/terok-sandbox"],
    ):
        install_supervisor_hooks()

    root = install_env["state"]
    assert (root / "hooks" / "supervisor_hook.py").is_file()
    assert (root / "hooks" / "_supervisor_state.py").is_file()
    wrapper = root / "supervisor_wrapper.py"
    assert wrapper.is_file()
    # Sandbox argv is baked into the wrapper at install time.
    assert '["/usr/local/bin/terok-sandbox"]' in wrapper.read_text()


def test_install_descriptor_targets_installed_entrypoint(install_env: dict[str, Path]) -> None:
    """The OCI hook descriptor JSON points at the installed entrypoint.

    One descriptor per stage — podman/crun reuse the same ``hook.args``
    for every stage listed in a single descriptor, so each stage gets
    its own JSON with the matching ``args[1]``.
    """
    with patch(
        "terok_sandbox.supervisor.install._resolve_sandbox_argv",
        return_value=["/usr/bin/terok-sandbox"],
    ):
        install_supervisor_hooks()

    expected_entrypoint = install_env["state"] / "hooks" / "supervisor_hook.py"
    for stage in ("createRuntime", "poststop"):
        descriptor = install_env["hooks_dir"] / f"terok-sandbox-supervisor-{stage}.json"
        payload = json.loads(descriptor.read_text())
        assert payload["hook"]["path"] == str(expected_entrypoint)
        assert payload["stages"] == [stage]
        assert payload["hook"]["args"] == ["supervisor_hook", stage]
        # The trigger annotation gates the hook fire-list and also
        # carries the sidecar path the hook reads.
        assert payload["when"]["annotations"] == {"terok.sandbox.sidecar": ".+"}


def test_install_raises_when_binary_missing(
    install_env: dict[str, Path], monkeypatch: pytest.MonkeyPatch
) -> None:
    """Missing terok-sandbox entry point is a hard error (operator must reinstall)."""
    monkeypatch.setattr("terok_sandbox.supervisor.install.shutil.which", lambda _name: None)
    monkeypatch.setattr("terok_sandbox.supervisor.install.sys.executable", "")
    with pytest.raises(RuntimeError, match="terok-sandbox entry point"):
        install_supervisor_hooks()


def test_uninstall_removes_every_install_artefact(install_env: dict[str, Path]) -> None:
    """The symmetric uninstall sweeps the install-side file set."""
    with patch(
        "terok_sandbox.supervisor.install._resolve_sandbox_argv",
        return_value=["/usr/bin/terok-sandbox"],
    ):
        install_supervisor_hooks()
    uninstall_supervisor_hooks()

    root = install_env["state"]
    for relative in (
        "hooks/supervisor_hook.py",
        "hooks/_supervisor_state.py",
        "supervisor_wrapper.py",
    ):
        assert not (root / relative).exists()
    for stage in ("createRuntime", "poststop"):
        assert not (install_env["hooks_dir"] / f"terok-sandbox-supervisor-{stage}.json").exists()


def test_uninstall_idempotent_on_empty_layout(install_env: dict[str, Path]) -> None:
    """Calling uninstall without a prior install is a no-op, not a crash."""
    uninstall_supervisor_hooks()  # must not raise


# ── kill_all_supervisors ────────────────────────────────────────────────


def test_kill_all_supervisors_empty_when_no_pids_dir(install_env: dict[str, Path]) -> None:
    """Returns an empty list before the OCI hook has written any PID file."""
    assert kill_all_supervisors() == []


def test_kill_all_supervisors_skips_stale_pids(install_env: dict[str, Path]) -> None:
    """A PID file whose process isn't our wrapper is unlinked without signalling.

    PID-recycle guard: the file name carries the container ID, but the
    PID inside may have been recycled into an unrelated process — the
    ``/proc/<pid>/cmdline`` check rejects it.
    """
    root = install_env["state"]
    pids_dir = root / "pids"
    pids_dir.mkdir(parents=True)
    stale = pids_dir / "supervisor-deadbeef.pid"
    # PID 1 is init — never our wrapper, always alive on a Linux host.
    stale.write_text("1\n")

    result = kill_all_supervisors()

    assert result == [("deadbeef", None)]
    assert not stale.exists()


def test_kill_all_supervisors_reports_unreadable_pid_file(install_env: dict[str, Path]) -> None:
    """Garbage PID file content is surfaced as an error, not a crash."""
    root = install_env["state"]
    pids_dir = root / "pids"
    pids_dir.mkdir(parents=True)
    garbage = pids_dir / "supervisor-cafe.pid"
    garbage.write_text("not-a-number\n")

    result = kill_all_supervisors()

    assert len(result) == 1
    container_id, err = result[0]
    assert container_id == "cafe"
    assert err is not None and "unreadable pid file" in err
    assert not garbage.exists()
