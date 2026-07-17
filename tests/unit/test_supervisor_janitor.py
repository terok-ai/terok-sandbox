# SPDX-FileCopyrightText: 2026 Jiri Vyskocil
# SPDX-License-Identifier: Apache-2.0

"""Tests for the orphaned-supervisor reconciliation sweep.

The sweep runs against a fake ``/proc`` (pid dirs with ``cmdline`` and
``stat``) and a stubbed container lister, so it never scans or signals
real host processes — ``os.killpg`` is captured, never delivered.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from terok_sandbox.supervisor import janitor
from terok_sandbox.supervisor.janitor import reap_orphaned_supervisors

_WRAPPER = "/home/op/.local/share/terok/sandbox/supervisor_wrapper.py"
_CHILD = ["/usr/bin/python3", "-P", "-m", "terok_sandbox", "supervise-child"]
_SIDE = "/home/op/.local/share/terok/sandbox/sidecar/proj-cli-abc.json"

_RUNNING = "a" * 64
_STOPPED = "b" * 64


@pytest.fixture
def fake_proc(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    """An empty fake ``/proc`` plus a fixed clock tick for age math."""
    proc = tmp_path / "proc"
    proc.mkdir()
    (proc / "uptime").write_text("100000.0 0.0\n")
    monkeypatch.setattr(janitor, "_PROC_DIR", proc)
    return proc


def _add(proc: Path, pid: int, argv: list[str], *, age_s: float = 3600.0) -> None:
    """Materialise a fake process with a null-separated cmdline and an age.

    ``stat`` field 22 (starttime, in clock ticks since boot) is derived
    from ``uptime - age_s`` so ``_process_age_s`` reports *age_s*.
    """
    pid_dir = proc / str(pid)
    pid_dir.mkdir()
    (pid_dir / "cmdline").write_bytes(b"\x00".join(a.encode() for a in argv) + b"\x00")
    uptime = float((proc / "uptime").read_text().split()[0])
    starttime_ticks = int((uptime - age_s) * janitor._CLOCK_TICKS)
    # /proc/<pid>/stat: "pid (comm) <field3> <field4> …".  The parser splits
    # after ")", so token[i] is stat field i+3.  starttime is field 22 →
    # token index 19.  comm carries a paren to exercise the rindex(')') parse.
    tokens = ["0"] * 40
    tokens[0] = "S"  # field 3 (state)
    tokens[19] = str(starttime_ticks)  # field 22 (starttime)
    (pid_dir / "stat").write_text(f"{pid} (py(thon3) " + " ".join(tokens) + "\n")


def _child(cid: str, service: str) -> list[str]:
    return [*_CHILD, service, cid, _SIDE]


@pytest.fixture
def one_pgid(monkeypatch: pytest.MonkeyPatch) -> None:
    """Make every fake pid its own process group (pgid == pid)."""
    monkeypatch.setattr(janitor.os, "getpgid", lambda pid: pid)


class TestReapOrphanedSupervisors:
    """Container liveness is the ground truth; only stray trees are killed."""

    def test_reaps_tree_whose_container_is_gone(
        self, fake_proc: Path, one_pgid: None, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """A supervise-child whose container isn't running gets its group killed."""
        _add(fake_proc, 5001, _child(_STOPPED, "vault"))
        _add(fake_proc, 5002, _child(_STOPPED, "signer"))
        monkeypatch.setattr(janitor, "_live_container_ids", lambda: frozenset())
        monkeypatch.setattr(janitor, "_group_alive", lambda pgid: False)
        killed: list[tuple[int, int]] = []
        monkeypatch.setattr(janitor.os, "killpg", lambda pgid, sig: killed.append((pgid, sig)))

        result = reap_orphaned_supervisors()

        assert result == [(_STOPPED, None)]
        # Both children share the container's group; each pid is its own
        # pgid here, so both get the SIGTERM.
        assert {pgid for pgid, sig in killed} == {5001, 5002}
        assert all(sig == janitor.signal.SIGTERM for _, sig in killed)

    def test_leaves_running_container_supervisor_alone(
        self, fake_proc: Path, one_pgid: None, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """A supervisor whose container is live is never touched."""
        _add(fake_proc, 6001, _child(_RUNNING, "vault"))
        monkeypatch.setattr(janitor, "_live_container_ids", lambda: frozenset({_RUNNING}))
        monkeypatch.setattr(
            janitor.os, "killpg", lambda *a: pytest.fail("must not signal a live tree")
        )

        assert reap_orphaned_supervisors() == []

    def test_wrapper_argv_is_also_recognised(
        self, fake_proc: Path, one_pgid: None, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """The restart-loop wrapper carries the container id too."""
        _add(fake_proc, 7001, ["/usr/bin/python3", _WRAPPER, _STOPPED, _SIDE])
        monkeypatch.setattr(janitor, "_live_container_ids", lambda: frozenset())
        monkeypatch.setattr(janitor, "_group_alive", lambda pgid: False)
        killed: list[int] = []
        monkeypatch.setattr(janitor.os, "killpg", lambda pgid, sig: killed.append(pgid))

        assert reap_orphaned_supervisors() == [(_STOPPED, None)]
        assert killed == [7001]

    def test_young_process_is_spared_create_race(
        self, fake_proc: Path, one_pgid: None, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """A just-spawned supervisor whose container isn't listed yet is left alone."""
        _add(fake_proc, 8001, _child(_STOPPED, "vault"), age_s=5.0)
        monkeypatch.setattr(janitor, "_live_container_ids", lambda: frozenset())
        monkeypatch.setattr(
            janitor.os, "killpg", lambda *a: pytest.fail("must not reap within the grace window")
        )

        assert reap_orphaned_supervisors() == []

    def test_foreign_processes_are_ignored(
        self, fake_proc: Path, one_pgid: None, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Neither mark → not a supervisor process → never scanned for a group."""
        _add(fake_proc, 9001, ["/usr/bin/nano", "supervise-child-notes.txt"])
        _add(fake_proc, 9002, ["/usr/bin/python3", "-m", "terok_sandbox", "doctor"])
        monkeypatch.setattr(janitor, "_live_container_ids", lambda: frozenset())
        monkeypatch.setattr(janitor.os, "killpg", lambda *a: pytest.fail("no supervisor here"))

        assert reap_orphaned_supervisors() == []

    def test_unreachable_podman_reaps_nothing(
        self, fake_proc: Path, one_pgid: None, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """If liveness can't be determined, the sweep declines to guess."""
        _add(fake_proc, 1001, _child(_STOPPED, "vault"))
        monkeypatch.setattr(janitor, "_live_container_ids", lambda: None)
        monkeypatch.setattr(
            janitor.os, "killpg", lambda *a: pytest.fail("must not kill without ground truth")
        )

        assert reap_orphaned_supervisors() == []

    def test_sigkill_escalation_reported(
        self, fake_proc: Path, one_pgid: None, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """A group that survives SIGTERM is SIGKILLed; an EPERM there is surfaced."""
        _add(fake_proc, 2001, _child(_STOPPED, "vault"))
        monkeypatch.setattr(janitor, "_live_container_ids", lambda: frozenset())
        monkeypatch.setattr(janitor, "_KILL_GRACE_S", 0.0)
        monkeypatch.setattr(janitor, "_group_alive", lambda pgid: True)

        def _killpg(pgid: int, sig: int) -> None:
            if sig == janitor.signal.SIGKILL:
                raise OSError("operation not permitted")

        monkeypatch.setattr(janitor.os, "killpg", _killpg)

        (cid, err) = reap_orphaned_supervisors()[0]
        assert cid == _STOPPED
        assert err is not None and "SIGKILL failed" in err


class TestLiveContainerIds:
    """The podman-backed ground truth parses ids by state."""

    def test_only_alive_states_count(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """Running/paused/created count as alive; exited/dead do not."""
        rows = [
            {"Id": _RUNNING, "State": "running"},
            {"Id": "c" * 64, "State": "created"},
            {"Id": _STOPPED, "State": "exited"},
            {"Id": "d" * 64, "State": "dead"},
        ]
        monkeypatch.setattr(janitor.shutil, "which", lambda _n: "/usr/bin/podman")

        class _Res:
            stdout = __import__("json").dumps(rows)

        monkeypatch.setattr(janitor.subprocess, "run", lambda *a, **k: _Res())

        alive = janitor._live_container_ids()
        assert alive == frozenset({_RUNNING, "c" * 64})

    def test_missing_podman_is_none(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """No podman on PATH → ``None`` (unknown), never an empty set."""
        monkeypatch.setattr(janitor.shutil, "which", lambda _n: None)
        assert janitor._live_container_ids() is None

    def test_subprocess_error_is_none(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """A podman invocation that errors/times out → ``None`` (unknown)."""
        monkeypatch.setattr(janitor.shutil, "which", lambda _n: "/usr/bin/podman")

        def _boom(*_a: object, **_k: object) -> None:
            raise janitor.subprocess.TimeoutExpired(cmd="podman", timeout=10)

        monkeypatch.setattr(janitor.subprocess, "run", _boom)
        assert janitor._live_container_ids() is None

    def test_unparsable_json_is_none(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """Non-JSON stdout → ``None`` rather than a bogus empty set."""
        monkeypatch.setattr(janitor.shutil, "which", lambda _n: "/usr/bin/podman")

        class _Res:
            stdout = "not json"

        monkeypatch.setattr(janitor.subprocess, "run", lambda *a, **k: _Res())
        assert janitor._live_container_ids() is None


class TestProcessHelpers:
    """The /proc-derived age and group-liveness helpers degrade gracefully."""

    def test_process_age_none_when_stat_missing(self, fake_proc: Path) -> None:
        """A pid with no ``stat`` file yields ``None`` age (never crashes the scan)."""
        assert janitor._process_age_s(424242) is None

    def test_process_age_none_when_starttime_malformed(self, fake_proc: Path) -> None:
        """A stat line with a non-numeric starttime field yields ``None``, not a crash."""
        pid_dir = fake_proc / "999"
        pid_dir.mkdir()
        (pid_dir / "stat").write_text("999 (py) S 1 not-a-number\n")
        assert janitor._process_age_s(999) is None

    def test_group_alive_reflects_signal0(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """``_group_alive`` is True when killpg(0) succeeds, False on OSError."""
        monkeypatch.setattr(janitor.os, "killpg", lambda pgid, sig: None)
        assert janitor._group_alive(1) is True
        monkeypatch.setattr(
            janitor.os, "killpg", lambda pgid, sig: (_ for _ in ()).throw(ProcessLookupError)
        )
        assert janitor._group_alive(1) is False

    def test_scan_skips_process_that_vanished(
        self, fake_proc: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """A pid whose group can't be read (raced exit) is skipped, not fatal."""
        _add(fake_proc, 4321, _child(_STOPPED, "vault"))
        monkeypatch.setattr(
            janitor.os, "getpgid", lambda pid: (_ for _ in ()).throw(ProcessLookupError)
        )
        assert janitor._scan_supervisor_groups() == {}


class TestDoctorCheck:
    """The doctor-check wrapper renders the sweep result."""

    def test_ok_when_nothing_stray(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setattr(janitor, "reap_orphaned_supervisors", lambda: [])
        verdict = janitor.make_orphan_supervisor_check().evaluate(0, "", "")
        assert verdict.severity == "ok"
        assert "no orphaned" in verdict.detail

    def test_reports_reaped_trees(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setattr(
            janitor, "reap_orphaned_supervisors", lambda: [(_STOPPED, None), (_RUNNING, None)]
        )
        verdict = janitor.make_orphan_supervisor_check().evaluate(0, "", "")
        assert verdict.severity == "ok"
        assert "reaped 2" in verdict.detail

    def test_warns_when_a_tree_would_not_die(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setattr(
            janitor, "reap_orphaned_supervisors", lambda: [(_STOPPED, "SIGKILL failed: EPERM")]
        )
        verdict = janitor.make_orphan_supervisor_check().evaluate(0, "", "")
        assert verdict.severity == "warn"
        assert "would not die" in verdict.detail
