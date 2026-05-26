# SPDX-FileCopyrightText: 2026 Jiri Vyskocil
# SPDX-License-Identifier: Apache-2.0

"""Tests for the supervisor OCI hook script.

The hook is stdlib-only and lives outside the package import graph
(OCI runtimes execute it under ``/usr/bin/python3``); these tests
load it via ``importlib.util`` so we can patch ``subprocess.Popen``
and the stdin payload without spawning real containers.

Annotation-driven contract: the hook does no XDG resolution.  It
reads ``terok.sandbox.sidecar`` from the OCI annotations, opens
the sidecar at that exact path, and derives ``logs/`` and ``pids/``
from ``sidecar_path.parent.parent``.
"""

from __future__ import annotations

import importlib.util
import io
import json
import os
import sys
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest


def _load_hook_module() -> object:
    """Import the hook + its sibling ballast as standalone modules."""
    hooks_dir = (
        Path(__file__).resolve().parents[2] / "src" / "terok_sandbox" / "resources" / "hooks"
    )
    for name in ("supervisor_hook", "_supervisor_state"):
        sys.modules.pop(name, None)
    sys.path.insert(0, str(hooks_dir))
    try:
        spec = importlib.util.spec_from_file_location(
            "supervisor_hook", hooks_dir / "supervisor_hook.py"
        )
        assert spec is not None and spec.loader is not None
        mod = importlib.util.module_from_spec(spec)
        sys.modules["supervisor_hook"] = mod
        spec.loader.exec_module(mod)
        return mod
    finally:
        sys.path.remove(str(hooks_dir))


@pytest.fixture
def hook_root(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    """Tmp ``<root>`` for the supervisor — wrapper + sidecar dir + pid dir."""
    runtime = tmp_path / "run"
    monkeypatch.setenv("XDG_RUNTIME_DIR", str(runtime))
    monkeypatch.setenv("HOME", str(tmp_path / "home"))
    return tmp_path


def _write_sidecar(root: Path, name: str, payload: dict[str, object]) -> Path:
    """Drop a sidecar JSON under ``<root>/sidecar/<name>.json``."""
    target = root / "sidecar" / f"{name}.json"
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(json.dumps(payload))
    return target


def _install_wrapper_alongside_hook(hook_module: object, wrapper_path: Path) -> None:
    """Drop a fake wrapper next to the hook module so ``is_file()`` succeeds."""
    wrapper_path.parent.mkdir(parents=True, exist_ok=True)
    wrapper_path.write_text("#!/usr/bin/env python3\n")


def _feed_stdin(monkeypatch: pytest.MonkeyPatch, payload: dict[str, object]) -> None:
    """Replace ``sys.stdin`` with a string-buffer containing *payload*."""
    monkeypatch.setattr("sys.stdin", io.StringIO(json.dumps(payload)))


class TestHookSoftFail:
    """Every error path must return ``None`` (rc 0) — container start never blocked."""

    def test_missing_annotation_is_silent_noop(
        self, hook_root: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """No ``terok.sandbox.sidecar`` annotation ⇒ not terok-managed; no spawn."""
        mod = _load_hook_module()
        _feed_stdin(monkeypatch, {"id": "abc123", "annotations": {}})
        monkeypatch.setattr(mod.sys, "argv", ["supervisor_hook", "createRuntime"])
        with patch.object(mod.subprocess, "Popen") as popen:
            mod.main()  # must not raise
        popen.assert_not_called()

    def test_bad_oci_state_returns_none(
        self, hook_root: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Malformed JSON on stdin ⇒ logged + early return."""
        mod = _load_hook_module()
        monkeypatch.setattr("sys.stdin", io.StringIO("{ not json"))
        monkeypatch.setattr(mod.sys, "argv", ["supervisor_hook", "createRuntime"])
        mod.main()  # must not raise

    def test_missing_container_id_returns_none(
        self, hook_root: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Empty ``id`` in OCI state ⇒ log + return; no spawn."""
        mod = _load_hook_module()
        _feed_stdin(
            monkeypatch,
            {"id": "", "annotations": {"terok.sandbox.sidecar": str(hook_root / "x.json")}},
        )
        monkeypatch.setattr(mod.sys, "argv", ["supervisor_hook", "createRuntime"])
        with patch.object(mod.subprocess, "Popen") as popen:
            mod.main()
        popen.assert_not_called()

    def test_traversal_annotation_rejected(
        self, hook_root: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Annotation values with ``..`` segments must be refused before any I/O."""
        mod = _load_hook_module()
        _feed_stdin(
            monkeypatch,
            {
                "id": "abc",
                "annotations": {"terok.sandbox.sidecar": "/var/lib/foo/../../etc/passwd"},
            },
        )
        monkeypatch.setattr(mod.sys, "argv", ["supervisor_hook", "createRuntime"])
        monkeypatch.setattr(mod._supervisor_state, "outer_host_uid", lambda: os.getuid())
        with patch.object(mod.subprocess, "Popen") as popen:
            mod.main()
        popen.assert_not_called()

    def test_wrong_parent_dir_rejected(
        self, hook_root: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """A sidecar JSON whose parent dir isn't named ``sidecar`` must be refused."""
        mod = _load_hook_module()
        bogus = hook_root / "not_sidecar" / "demo.json"
        bogus.parent.mkdir(parents=True)
        bogus.write_text("{}")
        _feed_stdin(
            monkeypatch,
            {"id": "abc", "annotations": {"terok.sandbox.sidecar": str(bogus)}},
        )
        monkeypatch.setattr(mod.sys, "argv", ["supervisor_hook", "createRuntime"])
        monkeypatch.setattr(mod._supervisor_state, "outer_host_uid", lambda: os.getuid())
        with patch.object(mod.subprocess, "Popen") as popen:
            mod.main()
        popen.assert_not_called()


class TestHookSpawn:
    """Happy-path spawn: write PID file, call Popen with the right command."""

    def test_createRuntime_spawns_wrapper_and_writes_pid(
        self, hook_root: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """createRuntime spawns the wrapper with [container_id, sidecar_path] and records PID."""
        mod = _load_hook_module()
        container_id = "abc123def456"
        sidecar_path = _write_sidecar(
            hook_root,
            "demo",
            {"container_name": "demo", "db_path": str(hook_root / "v.db"), "ipc_mode": "socket"},
        )
        # The hook computes wrapper_path = Path(__file__).parent.parent / "supervisor_wrapper.py";
        # patch __file__ so it points at a tmp tree that has the wrapper.
        fake_hooks_dir = hook_root / "hooks"
        fake_hooks_dir.mkdir()
        fake_hook_file = fake_hooks_dir / "supervisor_hook.py"
        fake_hook_file.write_text("# fake")
        wrapper = hook_root / "supervisor_wrapper.py"
        _install_wrapper_alongside_hook(mod, wrapper)
        monkeypatch.setattr(mod, "__file__", str(fake_hook_file))

        _feed_stdin(
            monkeypatch,
            {"id": container_id, "annotations": {"terok.sandbox.sidecar": str(sidecar_path)}},
        )
        monkeypatch.setattr(mod.sys, "argv", ["supervisor_hook", "createRuntime"])
        # The hook validates the sidecar owner against ``outer_host_uid``
        # (rootless namespace remaps host uid to 0 inside the runtime);
        # under tests the sidecar file is owned by the real test uid, so
        # stub the helper to return that.
        monkeypatch.setattr(mod._supervisor_state, "outer_host_uid", lambda: os.getuid())

        fake_proc = MagicMock(pid=12345)
        with patch.object(mod.subprocess, "Popen", return_value=fake_proc) as popen:
            mod.main()

        # Argv: /usr/bin/python3 <wrapper> <container_id> <sidecar_path>
        popen.assert_called_once()
        (argv,), _kwargs = popen.call_args
        assert argv[1] == str(wrapper)
        assert argv[2] == container_id
        assert argv[3] == str(sidecar_path)

        # PID file under <root>/pids
        pid_file = hook_root / "pids" / f"supervisor-{container_id}.pid"
        assert pid_file.read_text().strip() == "12345"

    def test_poststop_reaps_recorded_pid(
        self, hook_root: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """poststop sends SIGTERM to the PID the createRuntime hook recorded."""
        mod = _load_hook_module()
        container_id = "abc123def456"
        sidecar_path = _write_sidecar(
            hook_root,
            "demo",
            {"container_name": "demo", "db_path": str(hook_root / "v.db"), "ipc_mode": "socket"},
        )
        # Same wrapper-path setup as the createRuntime test.
        fake_hooks_dir = hook_root / "hooks"
        fake_hooks_dir.mkdir()
        fake_hook_file = fake_hooks_dir / "supervisor_hook.py"
        fake_hook_file.write_text("# fake")
        wrapper = hook_root / "supervisor_wrapper.py"
        _install_wrapper_alongside_hook(mod, wrapper)
        monkeypatch.setattr(mod, "__file__", str(fake_hook_file))

        pid_file = hook_root / "pids" / f"supervisor-{container_id}.pid"
        pid_file.parent.mkdir(parents=True)
        pid_file.write_text("12345\n")

        _feed_stdin(
            monkeypatch,
            {"id": container_id, "annotations": {"terok.sandbox.sidecar": str(sidecar_path)}},
        )
        monkeypatch.setattr(mod.sys, "argv", ["supervisor_hook", "poststop"])
        monkeypatch.setattr(mod._supervisor_state, "outer_host_uid", lambda: os.getuid())

        with (
            patch.object(mod, "_is_our_wrapper", return_value=True),
            patch.object(mod._supervisor_state, "pid_exists", return_value=False),
            patch.object(os, "kill") as kill_mock,
        ):
            mod.main()

        kill_mock.assert_called_once()
        sent_pid, sent_signal = kill_mock.call_args.args
        assert sent_pid == 12345
        assert sent_signal == mod.signal.SIGTERM
        assert not pid_file.exists()
        # poststop also unlinks the sidecar so a long-running host
        # doesn't accumulate stale JSONs.
        assert not sidecar_path.exists()
