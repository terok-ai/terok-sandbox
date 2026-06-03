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
    _is_our_wrapper,
    _resolve_sandbox_argv,
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


def test_kill_all_supervisors_skips_stale_pids(
    install_env: dict[str, Path], monkeypatch: pytest.MonkeyPatch
) -> None:
    """A PID file whose process isn't our wrapper is unlinked without signalling.

    PID-recycle guard: the file name carries the container ID, but the
    PID inside may have been recycled into an unrelated process — the
    ``/proc/<pid>/cmdline`` check rejects it.  Stubbing ``_is_our_wrapper``
    keeps the test off the host's real ``/proc`` state.
    """
    root = install_env["state"]
    pids_dir = root / "pids"
    pids_dir.mkdir(parents=True)
    stale = pids_dir / "supervisor-deadbeef.pid"
    stale.write_text("424242\n")
    monkeypatch.setattr("terok_sandbox.supervisor.install._is_our_wrapper", lambda *_a: False)

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


def test_kill_all_supervisors_sigkills_matching_wrapper(
    install_env: dict[str, Path], monkeypatch: pytest.MonkeyPatch
) -> None:
    """A PID file whose process IS our wrapper gets ``SIGKILL``ed then unlinked.

    Stubs ``_is_our_wrapper`` True and captures ``os.kill`` so the sweep
    never signals a real process; the PID file is always removed after.
    """
    root = install_env["state"]
    pids_dir = root / "pids"
    pids_dir.mkdir(parents=True)
    live = pids_dir / "supervisor-beef.pid"
    live.write_text("99999\n")
    monkeypatch.setattr("terok_sandbox.supervisor.install._is_our_wrapper", lambda *_a: True)
    killed: list[tuple[int, int]] = []
    monkeypatch.setattr(
        "terok_sandbox.supervisor.install.os.kill",
        lambda pid, sig: killed.append((pid, sig)),
    )

    result = kill_all_supervisors()

    assert result == [("beef", None)]
    assert killed and killed[0][0] == 99999
    assert not live.exists()


def test_kill_all_supervisors_tolerates_already_exited(
    install_env: dict[str, Path], monkeypatch: pytest.MonkeyPatch
) -> None:
    """A wrapper that has already exited (``ProcessLookupError``) is no error."""
    root = install_env["state"]
    pids_dir = root / "pids"
    pids_dir.mkdir(parents=True)
    gone = pids_dir / "supervisor-d00d.pid"
    gone.write_text("99998\n")
    monkeypatch.setattr("terok_sandbox.supervisor.install._is_our_wrapper", lambda *_a: True)

    def _raise_lookup(_pid: int, _sig: int) -> None:
        raise ProcessLookupError

    monkeypatch.setattr("terok_sandbox.supervisor.install.os.kill", _raise_lookup)

    result = kill_all_supervisors()

    assert result == [("d00d", None)]
    assert not gone.exists()


def test_kill_all_supervisors_surfaces_oserror(
    install_env: dict[str, Path], monkeypatch: pytest.MonkeyPatch
) -> None:
    """A ``SIGKILL`` that fails with ``OSError`` (e.g. EPERM) is reported."""
    root = install_env["state"]
    pids_dir = root / "pids"
    pids_dir.mkdir(parents=True)
    locked = pids_dir / "supervisor-face.pid"
    locked.write_text("99997\n")
    monkeypatch.setattr("terok_sandbox.supervisor.install._is_our_wrapper", lambda *_a: True)

    def _raise_eperm(_pid: int, _sig: int) -> None:
        raise OSError("operation not permitted")

    monkeypatch.setattr("terok_sandbox.supervisor.install.os.kill", _raise_eperm)

    result = kill_all_supervisors()

    container_id, err = result[0]
    assert container_id == "face"
    assert err is not None and "SIGKILL failed" in err
    assert not locked.exists()


# ── _is_our_wrapper ─────────────────────────────────────────────────────


class TestIsOurWrapper:
    """The PID-recycle guard requires BOTH the wrapper path and container ID."""

    def test_returns_true_when_both_present_in_cmdline(self, tmp_path: Path) -> None:
        """A cmdline carrying the wrapper path and the container ID matches.

        Writes a fake null-separated ``/proc/<pid>/cmdline`` and points the
        check at it via a patched ``Path("/proc")/...`` read — no real
        process is involved.
        """
        wrapper = "/state/supervisor_wrapper.py"
        container_id = "abc123"
        cmdline = b"\x00".join([b"python3", wrapper.encode(), container_id.encode()]) + b"\x00"
        with patch("terok_sandbox.supervisor.install.Path.read_bytes", return_value=cmdline):
            assert _is_our_wrapper(4242, wrapper, container_id) is True

    def test_returns_false_when_container_id_absent(self) -> None:
        """The wrapper path alone is not enough — a recycled PID running an
        unrelated container's wrapper must be rejected."""
        wrapper = "/state/supervisor_wrapper.py"
        cmdline = b"\x00".join([b"python3", wrapper.encode(), b"other-id"]) + b"\x00"
        with patch("terok_sandbox.supervisor.install.Path.read_bytes", return_value=cmdline):
            assert _is_our_wrapper(4242, wrapper, "abc123") is False

    def test_returns_false_when_cmdline_unreadable(self) -> None:
        """A missing/unreadable ``/proc/<pid>/cmdline`` (OSError) → False."""
        with patch("terok_sandbox.supervisor.install.Path.read_bytes", side_effect=OSError("gone")):
            assert _is_our_wrapper(4242, "/w.py", "abc123") is False


# ── uninstall with explicit hooks_dir ───────────────────────────────────


def test_uninstall_removes_descriptors_from_explicit_hooks_dir(
    install_env: dict[str, Path], tmp_path: Path
) -> None:
    """When ``hooks_dir`` is passed, per-stage descriptors there are removed too.

    This covers the legacy split-dir layout where the OCI descriptors
    were installed somewhere other than ``state_root()/hooks``.
    """
    external = tmp_path / "external-hooks"
    external.mkdir()
    for stage in ("createRuntime", "poststop"):
        (external / f"terok-sandbox-supervisor-{stage}.json").write_text("{}")

    uninstall_supervisor_hooks(hooks_dir=external)

    for stage in ("createRuntime", "poststop"):
        assert not (external / f"terok-sandbox-supervisor-{stage}.json").exists()


# ── _resolve_sandbox_argv ────────────────────────────────────────────────


class TestResolveSandboxArgv:
    """Resolution order: ``$PATH`` first, then the venv sibling bin."""

    def test_prefers_path_resolution(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """A ``terok-sandbox`` on ``$PATH`` wins outright."""
        monkeypatch.setattr(
            "terok_sandbox.supervisor.install.shutil.which",
            lambda _name: "/usr/local/bin/terok-sandbox",
        )
        assert _resolve_sandbox_argv() == ["/usr/local/bin/terok-sandbox"]

    def test_falls_back_to_venv_sibling(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """When not on ``$PATH``, the executable sibling of ``sys.executable`` is used.

        Covers the pipx / venv shape where the console script lives next
        to Python but the venv ``bin/`` isn't on ``$PATH``.
        """
        bindir = tmp_path / "venv" / "bin"
        bindir.mkdir(parents=True)
        sibling = bindir / "terok-sandbox"
        sibling.write_text("#!/bin/sh\n")
        sibling.chmod(0o755)
        monkeypatch.setattr("terok_sandbox.supervisor.install.shutil.which", lambda _name: None)
        monkeypatch.setattr(
            "terok_sandbox.supervisor.install.sys.executable", str(bindir / "python")
        )
        assert _resolve_sandbox_argv() == [str(sibling)]
