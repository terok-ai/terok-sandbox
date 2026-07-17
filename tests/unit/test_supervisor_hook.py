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


def _fake_proc(hook_module: object, root: Path) -> Path:
    """Point the module's stray-children sweep at an empty fake ``/proc``.

    Mandatory for every poststop-path test: against the real ``/proc``
    the sweep could find — and signal — actual processes on the host
    running the suite.
    """
    proc = root / "proc"
    proc.mkdir(exist_ok=True)
    hook_module._PROC_DIR = proc
    return proc


def _add_process(proc: Path, pid: int, argv: list[str]) -> None:
    """Materialise one fake ``/proc/<pid>`` with a null-separated cmdline."""
    pid_dir = proc / str(pid)
    pid_dir.mkdir()
    (pid_dir / "cmdline").write_bytes(b"\x00".join(a.encode() for a in argv) + b"\x00")


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
        # ``outer_host_uid`` only feeds the spawn env (XDG_RUNTIME_DIR);
        # the ownership check uses ``geteuid()`` directly.  Stub it to the
        # test uid so the spawn env stays sane.
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
        assert len(argv) == 4  # no OCI pid in this state → no 5th positional

        # PID file under <root>/pids
        pid_file = hook_root / "pids" / f"supervisor-{container_id}.pid"
        assert pid_file.read_text().strip() == "12345"

    def test_createRuntime_forwards_container_init_pid(
        self, hook_root: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """A ``pid`` in the OCI state is appended to the wrapper argv for the PID watch."""
        mod = _load_hook_module()
        container_id = "abc123def456"
        sidecar_path = _write_sidecar(
            hook_root,
            "demo",
            {"container_name": "demo", "db_path": str(hook_root / "v.db"), "ipc_mode": "socket"},
        )
        fake_hooks_dir = hook_root / "hooks"
        fake_hooks_dir.mkdir()
        fake_hook_file = fake_hooks_dir / "supervisor_hook.py"
        fake_hook_file.write_text("# fake")
        wrapper = hook_root / "supervisor_wrapper.py"
        _install_wrapper_alongside_hook(mod, wrapper)
        monkeypatch.setattr(mod, "__file__", str(fake_hook_file))
        _feed_stdin(
            monkeypatch,
            {
                "id": container_id,
                "pid": 1504136,
                "annotations": {"terok.sandbox.sidecar": str(sidecar_path)},
            },
        )
        monkeypatch.setattr(mod.sys, "argv", ["supervisor_hook", "createRuntime"])
        monkeypatch.setattr(mod._supervisor_state, "outer_host_uid", lambda: os.getuid())

        with patch.object(mod.subprocess, "Popen", return_value=MagicMock(pid=12345)) as popen:
            mod.main()

        (argv,), _kwargs = popen.call_args
        assert argv[3] == str(sidecar_path)
        assert argv[4] == "1504136"  # container init host-PID, for the direct watch

    def test_ownership_check_uses_in_namespace_uid_not_outer(
        self, hook_root: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Sidecar ownership is judged in-namespace (``geteuid``), not by ``outer_host_uid``.

        Under crun the createRuntime hook runs in the container's rootless
        user namespace: the operator's host UID (e.g. 1000) maps to
        in-namespace 0, so an operator-owned sidecar stats as ``st_uid == 0``
        while ``outer_host_uid()`` recovers 1000.  Comparing the on-disk
        owner against the *outer* uid would never match there and silently
        skip the supervisor — the regression this test pins.  We simulate
        the mismatch by stubbing ``outer_host_uid`` to a value that is *not*
        the file's owner and assert the hook still spawns.
        """
        mod = _load_hook_module()
        container_id = "abc123def456"
        sidecar_path = _write_sidecar(
            hook_root,
            "demo",
            {"container_name": "demo", "db_path": str(hook_root / "v.db"), "ipc_mode": "socket"},
        )
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
        # The sidecar is owned by the real test uid, but the *outer* uid is
        # reported as something else — exactly the crun userns shape.
        monkeypatch.setattr(mod._supervisor_state, "outer_host_uid", lambda: os.getuid() + 12345)

        fake_proc = MagicMock(pid=12345)
        with patch.object(mod.subprocess, "Popen", return_value=fake_proc) as popen:
            mod.main()

        popen.assert_called_once()

    def test_poststop_reaps_recorded_pid(
        self, hook_root: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """poststop group-SIGTERMs the process group the createRuntime hook recorded.

        The recorded PID doubles as the group ID (the wrapper is
        spawned with ``start_new_session=True``), so the reap signals
        the whole tree — this is what actually delivers SIGTERM to the
        supervisor, since the wrapper's restart loop forwards nothing.
        """
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

        _fake_proc(mod, hook_root)
        _feed_stdin(
            monkeypatch,
            {"id": container_id, "annotations": {"terok.sandbox.sidecar": str(sidecar_path)}},
        )
        monkeypatch.setattr(mod.sys, "argv", ["supervisor_hook", "poststop"])
        monkeypatch.setattr(mod._supervisor_state, "outer_host_uid", lambda: os.getuid())

        with (
            patch.object(mod, "_is_group_ours", return_value=True),
            patch.object(mod, "_group_exists", return_value=False),
            patch.object(os, "killpg") as killpg_mock,
        ):
            mod.main()

        killpg_mock.assert_called_once()
        sent_pgid, sent_signal = killpg_mock.call_args.args
        assert sent_pgid == 12345
        assert sent_signal == mod.signal.SIGTERM
        assert not pid_file.exists()
        # The sidecar must survive the reap: stop/start cycles re-fire
        # createRuntime, and the preserved file is the only wiring that
        # still matches the container's immutable env.  Removal belongs
        # to real teardown (cleanup / task delete / doctor stray sweep).
        assert sidecar_path.exists()

    def test_poststop_without_pid_file_preserves_sidecar(
        self, hook_root: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """The no-pid-file reap path must not delete the sidecar either.

        This is the restart contract's regression guard: every poststop
        exit path used to unlink the sidecar, which made any later
        ``podman start`` come up unsupervised (the createRuntime hook
        soft-fails on the missing file).
        """
        mod = _load_hook_module()
        container_id = "abc123def456"
        sidecar_path = _write_sidecar(
            hook_root,
            "demo",
            {"container_name": "demo", "db_path": str(hook_root / "v.db"), "ipc_mode": "socket"},
        )
        _fake_proc(mod, hook_root)
        _feed_stdin(
            monkeypatch,
            {"id": container_id, "annotations": {"terok.sandbox.sidecar": str(sidecar_path)}},
        )
        monkeypatch.setattr(mod.sys, "argv", ["supervisor_hook", "poststop"])
        monkeypatch.setattr(mod._supervisor_state, "outer_host_uid", lambda: os.getuid())

        mod.main()

        assert sidecar_path.exists()

    def test_poststop_group_kill_reaches_orphaned_members(
        self, hook_root: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """A dead wrapper does not shield its group — killpg fires anyway.

        The leader can die without teardown (crash, OOM) while its
        service children keep the group alive; a group ID stays pinned
        while any member lives, so signalling it is always precise.
        No ``/proc`` entry is materialised for the leader here.
        """
        mod = _load_hook_module()
        container_id = "abc123def456"
        sidecar_path = _write_sidecar(
            hook_root,
            "demo",
            {"container_name": "demo", "db_path": str(hook_root / "v.db"), "ipc_mode": "socket"},
        )
        _fake_proc(mod, hook_root)
        pid_file = hook_root / "pids" / f"supervisor-{container_id}.pid"
        pid_file.parent.mkdir(parents=True)
        pid_file.write_text("54321\n")
        _feed_stdin(
            monkeypatch,
            {"id": container_id, "annotations": {"terok.sandbox.sidecar": str(sidecar_path)}},
        )
        monkeypatch.setattr(mod.sys, "argv", ["supervisor_hook", "poststop"])
        monkeypatch.setattr(mod._supervisor_state, "outer_host_uid", lambda: os.getuid())

        with (
            patch.object(mod, "_group_exists", return_value=False),
            patch.object(os, "killpg") as killpg_mock,
        ):
            mod.main()

        assert killpg_mock.call_args.args == (54321, mod.signal.SIGTERM)
        assert not pid_file.exists()

    def test_poststop_escalates_to_group_sigkill(
        self, hook_root: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """A group that outlives the SIGTERM grace window gets group-SIGKILLed."""
        mod = _load_hook_module()
        container_id = "abc123def456"
        sidecar_path = _write_sidecar(
            hook_root,
            "demo",
            {"container_name": "demo", "db_path": str(hook_root / "v.db"), "ipc_mode": "socket"},
        )
        _fake_proc(mod, hook_root)
        pid_file = hook_root / "pids" / f"supervisor-{container_id}.pid"
        pid_file.parent.mkdir(parents=True)
        pid_file.write_text("54321\n")
        _feed_stdin(
            monkeypatch,
            {"id": container_id, "annotations": {"terok.sandbox.sidecar": str(sidecar_path)}},
        )
        monkeypatch.setattr(mod.sys, "argv", ["supervisor_hook", "poststop"])
        monkeypatch.setattr(mod._supervisor_state, "outer_host_uid", lambda: os.getuid())
        monkeypatch.setattr(mod, "_REAP_POLL_INTERVAL_S", 0.0)

        with (
            patch.object(mod, "_group_exists", return_value=True),
            patch.object(os, "killpg") as killpg_mock,
        ):
            mod.main()

        signals = [call.args for call in killpg_mock.call_args_list]
        assert signals == [(54321, mod.signal.SIGTERM), (54321, mod.signal.SIGKILL)]
        assert not pid_file.exists()

    def test_poststop_reaps_stray_children(
        self, hook_root: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Service children the wrapper's teardown missed are swept — this container's only.

        The post-split regression guard: a wrapper SIGKILLed past the
        grace window (or dead before poststop, as observed on the
        podman 3.4 host) leaves its ``supervise-child`` processes
        orphaned, and the vault-daemon child keeps the credentials DB
        open forever.  No PID file exists here at all — the sweep must
        find the children by argv.
        """
        mod = _load_hook_module()
        container_id = "abc123def456"
        sidecar_path = _write_sidecar(
            hook_root,
            "demo",
            {"container_name": "demo", "db_path": str(hook_root / "v.db"), "ipc_mode": "socket"},
        )
        proc = _fake_proc(mod, hook_root)
        child_argv = ["/usr/bin/python3", "-P", "-m", "terok_sandbox", "supervise-child"]
        _add_process(proc, 5551, [*child_argv, "vault", container_id, "/x/sidecar.json"])
        _add_process(proc, 5552, [*child_argv, "vault", "othercontainer", "/y/sidecar.json"])
        _feed_stdin(
            monkeypatch,
            {"id": container_id, "annotations": {"terok.sandbox.sidecar": str(sidecar_path)}},
        )
        monkeypatch.setattr(mod.sys, "argv", ["supervisor_hook", "poststop"])
        monkeypatch.setattr(mod._supervisor_state, "outer_host_uid", lambda: os.getuid())

        killed: list[tuple[int, int]] = []
        with (
            patch.object(mod._supervisor_state, "pid_exists", return_value=False),
            patch.object(os, "kill", lambda pid, sig: killed.append((pid, sig))),
        ):
            mod.main()

        assert killed == [(5551, mod.signal.SIGTERM)]


class TestSpawnEnv:
    """``_spawn_env`` composes a trustworthy env for the wrapper subprocess."""

    def test_home_pinned_from_host_passwd_entry(
        self, hook_root: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """``HOME`` comes from the host passwd entry, not the inherited env.

        crun 0.17 (Ubuntu 22.04) hands hooks the *container's* process env
        with ``HOME=/root``; inheriting it sent the supervisor's vault and
        SSH-signer paths into the real root's home (EPERM from the rootless
        namespace) — the regression this test pins.
        """
        mod = _load_hook_module()
        monkeypatch.setenv("HOME", "/root")
        operator_home = str(hook_root / "home")
        fake_pwent = MagicMock(pw_dir=operator_home)
        monkeypatch.setattr(mod.pwd, "getpwuid", lambda uid: fake_pwent)

        env = mod._spawn_env(1017)

        assert env["HOME"] == operator_home
        assert env["XDG_RUNTIME_DIR"] == "/run/user/1017"

    def test_missing_passwd_entry_keeps_inherited_home(
        self, hook_root: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """No passwd entry for the host uid → the inherited ``HOME`` survives."""
        mod = _load_hook_module()
        monkeypatch.setenv("HOME", "/somewhere/else")

        def _no_entry(uid: int) -> object:
            raise KeyError(uid)

        monkeypatch.setattr(mod.pwd, "getpwuid", _no_entry)

        env = mod._spawn_env(1017)

        assert env["HOME"] == "/somewhere/else"
