# SPDX-FileCopyrightText: 2026 Jiri Vyskocil
# SPDX-License-Identifier: Apache-2.0

"""Unit tests for the container-diagnostics path resolver."""

from __future__ import annotations

import json
import os
import sys
from pathlib import Path

from terok_sandbox import (
    ContainerDiagnostics,
    SupervisorLiveness,
    container_diagnostics,
    diagnostics as diag,
    respawn_supervisor,
    supervisor_liveness,
)

_CID = "abc123def456"
_CNAME = "demo-cli-w9xk3"


def _plant_supervisor(tmp_path: Path, cid: str, *, pid: int, cmdline: list[str]) -> Path:
    """Write a PID file + a fake ``/proc/<pid>/cmdline``; return the fake /proc dir."""
    (tmp_path / "pids").mkdir(parents=True, exist_ok=True)
    (tmp_path / "pids" / f"supervisor-{cid}.pid").write_text(f"{pid}\n")
    proc = tmp_path / "proc"
    (proc / str(pid)).mkdir(parents=True, exist_ok=True)
    (proc / str(pid) / "cmdline").write_bytes(b"\x00".join(a.encode() for a in cmdline) + b"\x00")
    return proc


def test_paths_key_on_id_and_name(tmp_path: Path) -> None:
    """Log + PID key on the container ID; sidecar keys on the name; wrapper is global."""
    d = container_diagnostics(_CID, _CNAME, state_dir=tmp_path)

    assert isinstance(d, ContainerDiagnostics)
    assert d.container_id == _CID
    assert d.log == tmp_path / "logs" / f"{_CID}.log"
    assert d.pid == tmp_path / "pids" / f"supervisor-{_CID}.pid"
    assert d.wrapper == tmp_path / "supervisor_wrapper.py"
    assert d.sidecar == tmp_path / "sidecar" / f"{_CNAME}.json"
    assert d.hook_log == tmp_path / "logs" / "hook.log"


def test_hook_log_is_container_independent(tmp_path: Path) -> None:
    """The hook diary is install-global — same path for any container."""
    a = container_diagnostics(_CID, _CNAME, state_dir=tmp_path)
    b = container_diagnostics("ffffffffffff", "other-task-abc12", state_dir=tmp_path)
    assert a.hook_log == b.hook_log == tmp_path / "logs" / "hook.log"


def test_paths_are_computed_not_probed(tmp_path: Path) -> None:
    """Resolution never touches disk — every path comes back absent-but-named."""
    d = container_diagnostics(_CID, _CNAME, state_dir=tmp_path)
    assert not d.log.exists()
    assert not d.sidecar.exists()


def test_default_state_dir_uses_state_root(monkeypatch, tmp_path: Path) -> None:
    """Omitting *state_dir* falls back to the resolved ``state_root()``."""
    monkeypatch.setattr(diag, "state_root", lambda: tmp_path / "rooted")
    d = container_diagnostics(_CID, _CNAME)
    assert d.log == tmp_path / "rooted" / "logs" / f"{_CID}.log"


def test_frozen(tmp_path: Path) -> None:
    """The bundle is immutable — paths are a snapshot, not a mutable handle."""
    import dataclasses

    import pytest

    d = container_diagnostics(_CID, _CNAME, state_dir=tmp_path)
    with pytest.raises(dataclasses.FrozenInstanceError):
        d.log = tmp_path  # type: ignore[misc]


class TestSupervisorLiveness:
    """Probe of ``<state>/pids/supervisor-<id>.pid`` + ``/proc`` argv match."""

    def test_no_pid_file_is_not_alive(self, tmp_path: Path) -> None:
        r = supervisor_liveness(_CID, state_dir=tmp_path)
        assert r.alive is False
        assert r.pid is None
        assert "no PID file" in r.detail

    def test_alive_when_pid_live_and_argv_matches(self, tmp_path: Path, monkeypatch) -> None:
        pid = os.getpid()  # this test process is unquestionably alive
        wrapper = str(tmp_path / "supervisor_wrapper.py")
        proc = _plant_supervisor(
            tmp_path, _CID, pid=pid, cmdline=["/usr/bin/python3", wrapper, _CID, "/s.json"]
        )
        monkeypatch.setattr(diag, "_PROC_DIR", proc)
        r = supervisor_liveness(_CID, state_dir=tmp_path)
        assert r.alive is True
        assert r.pid == pid
        assert f"pid {pid}" in r.detail

    def test_stale_when_pid_dead(self, tmp_path: Path, monkeypatch) -> None:
        wrapper = str(tmp_path / "supervisor_wrapper.py")
        proc = _plant_supervisor(
            tmp_path, _CID, pid=4242, cmdline=["/usr/bin/python3", wrapper, _CID]
        )
        monkeypatch.setattr(diag, "_PROC_DIR", proc)
        monkeypatch.setattr(diag, "_pid_alive", lambda _pid: False)
        r = supervisor_liveness(_CID, state_dir=tmp_path)
        assert r.alive is False
        assert r.pid == 4242
        assert "stale" in r.detail

    def test_recycled_pid_without_container_id_is_not_ours(
        self, tmp_path: Path, monkeypatch
    ) -> None:
        """A live PID whose argv lacks *this* container's id is a recycled process."""
        pid = os.getpid()
        wrapper = str(tmp_path / "supervisor_wrapper.py")
        proc = _plant_supervisor(
            tmp_path, _CID, pid=pid, cmdline=["/usr/bin/python3", wrapper, "SOME-OTHER-ID"]
        )
        monkeypatch.setattr(diag, "_PROC_DIR", proc)
        r = supervisor_liveness(_CID, state_dir=tmp_path)
        assert r.alive is False
        assert r.pid == pid

    def test_oversized_pid_is_not_alive(self) -> None:
        """A corrupt PID file value too large for ``pid_t`` (OverflowError) is dead, not a crash."""
        assert diag._pid_alive(2**63) is False


class TestRespawnSupervisor:
    """Re-firing the installed OCI hook to respawn a container's supervisor."""

    @staticmethod
    def _install_hook_and_sidecar(tmp_path: Path) -> None:
        (tmp_path / "hooks").mkdir(parents=True, exist_ok=True)
        (tmp_path / "hooks" / "supervisor_hook.py").write_text("# hook")
        (tmp_path / "sidecar").mkdir(parents=True, exist_ok=True)
        (tmp_path / "sidecar" / f"{_CNAME}.json").write_text("{}")

    def test_reinvokes_hook_with_synthesized_oci_state(self, tmp_path: Path, monkeypatch) -> None:
        self._install_hook_and_sidecar(tmp_path)
        monkeypatch.setattr(diag, "_RESPAWN_SETTLE_S", 0.0)  # no PID appears → don't poll
        captured: dict[str, object] = {}

        def _fake_run(argv, **kwargs):  # noqa: ANN001, ANN202
            captured["argv"] = argv
            captured["input"] = kwargs.get("input")
            return None

        monkeypatch.setattr(diag.subprocess, "run", _fake_run)
        result = respawn_supervisor(_CID, _CNAME, state_dir=tmp_path)

        argv = captured["argv"]
        assert argv[0] == sys.executable
        assert argv[1] == str(tmp_path / "hooks" / "supervisor_hook.py")
        assert argv[2] == "createRuntime"
        state = json.loads(captured["input"])
        assert state["id"] == _CID
        assert state["annotations"]["terok.sandbox.sidecar"] == str(
            tmp_path / "sidecar" / f"{_CNAME}.json"
        )
        assert "pid" not in state
        assert result.alive is False  # no PID file planted → not up afterwards

    def test_container_pid_is_forwarded_when_given(self, tmp_path: Path, monkeypatch) -> None:
        self._install_hook_and_sidecar(tmp_path)
        monkeypatch.setattr(diag, "_RESPAWN_SETTLE_S", 0.0)
        captured: dict[str, object] = {}
        monkeypatch.setattr(
            diag.subprocess, "run", lambda argv, **kw: captured.update(input=kw.get("input"))
        )
        respawn_supervisor(_CID, _CNAME, state_dir=tmp_path, container_pid=4321)
        assert json.loads(captured["input"])["pid"] == 4321

    def test_polls_until_the_wrapper_shows_alive(self, tmp_path: Path, monkeypatch) -> None:
        """The detached wrapper may exec after the hook returns — the poll waits for it."""
        self._install_hook_and_sidecar(tmp_path)
        monkeypatch.setattr(diag.subprocess, "run", lambda *_a, **_k: None)
        monkeypatch.setattr(diag.time, "sleep", lambda _s: None)  # keep the poll instant
        probes = iter(
            [
                SupervisorLiveness(alive=False, pid=None, detail="not up yet"),
                SupervisorLiveness(alive=True, pid=555, detail="supervisor pid 555 alive"),
            ]
        )
        monkeypatch.setattr(diag, "supervisor_liveness", lambda *_a, **_k: next(probes))

        result = respawn_supervisor(_CID, _CNAME, state_dir=tmp_path)
        assert result.alive is True
        assert result.pid == 555

    def test_missing_hook_skips_spawn(self, tmp_path: Path, monkeypatch) -> None:
        # sidecar present, hook absent → nothing to re-fire.
        (tmp_path / "sidecar").mkdir(parents=True)
        (tmp_path / "sidecar" / f"{_CNAME}.json").write_text("{}")
        ran = False

        def _fake_run(*_a, **_k):  # noqa: ANN202
            nonlocal ran
            ran = True

        monkeypatch.setattr(diag.subprocess, "run", _fake_run)
        result = respawn_supervisor(_CID, _CNAME, state_dir=tmp_path)
        assert ran is False
        assert result.alive is False

    def test_reports_alive_after_a_successful_respawn(self, tmp_path: Path, monkeypatch) -> None:
        self._install_hook_and_sidecar(tmp_path)
        pid = os.getpid()
        wrapper = str(tmp_path / "supervisor_wrapper.py")
        proc = _plant_supervisor(
            tmp_path, _CID, pid=pid, cmdline=["/usr/bin/python3", wrapper, _CID]
        )
        monkeypatch.setattr(diag, "_PROC_DIR", proc)
        monkeypatch.setattr(diag.subprocess, "run", lambda *_a, **_k: None)

        result = respawn_supervisor(_CID, _CNAME, state_dir=tmp_path)
        assert result.alive is True
        assert result.pid == pid
