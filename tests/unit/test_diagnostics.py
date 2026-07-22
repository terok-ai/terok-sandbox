# SPDX-FileCopyrightText: 2026 Jiri Vyskocil
# SPDX-License-Identifier: Apache-2.0

"""Unit tests for the container-diagnostics path resolver."""

from __future__ import annotations

import os
from pathlib import Path

from terok_sandbox import (
    ContainerDiagnostics,
    container_diagnostics,
    diagnostics as diag,
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
