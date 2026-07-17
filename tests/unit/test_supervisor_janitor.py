# SPDX-FileCopyrightText: 2026 Jiri Vyskocil
# SPDX-License-Identifier: Apache-2.0

"""Tests for the orphaned-supervisor reconciliation sweep.

The sweep runs against a fake ``/proc`` (pid dirs with ``cmdline`` and
``stat``) and a stubbed container lister, so it never scans or signals
real host processes — ``os.killpg`` is captured, never delivered.  A
"kill" is simulated by removing the target's fake ``/proc`` entry, which
is what the identity revalidation reads to decide the group is gone.
"""

from __future__ import annotations

import signal
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
    """An empty fake ``/proc`` plus a fixed uptime for age math."""
    proc = tmp_path / "proc"
    proc.mkdir()
    (proc / "uptime").write_text("100000.0 0.0\n")
    monkeypatch.setattr(janitor, "_PROC_DIR", proc)
    return proc


def _add(
    proc: Path, pid: int, argv: list[str], *, age_s: float = 3600.0, stat: bool = True
) -> None:
    """Materialise a fake process with a null-separated cmdline and (optional) stat.

    ``stat`` field 22 (starttime, in clock ticks since boot) is derived
    from ``uptime - age_s`` so the derived age is *age_s*.  ``stat=False``
    omits the stat file, modelling a process whose age/identity can't be
    read.
    """
    pid_dir = proc / str(pid)
    pid_dir.mkdir()
    (pid_dir / "cmdline").write_bytes(b"\x00".join(a.encode() for a in argv) + b"\x00")
    if not stat:
        return
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


def _killpg_removing(proc: Path, record: list) -> object:
    """A ``killpg`` stub that records the call and removes the fake process.

    Removing the ``/proc`` entry is how the test models the process dying,
    so the identity revalidation sees the group gone on the next scan.
    """

    def _killpg(pgid: int, sig: int) -> None:
        record.append((pgid, sig))
        import shutil

        shutil.rmtree(proc / str(pgid), ignore_errors=True)

    return _killpg


class TestReapOrphanedSupervisors:
    """Container liveness is the ground truth; only stray trees are killed."""

    def test_reaps_tree_whose_container_is_gone(
        self, fake_proc: Path, one_pgid: None, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """A supervise-child whose container isn't running gets its group SIGTERMed."""
        _add(fake_proc, 5001, _child(_STOPPED, "vault"))
        _add(fake_proc, 5002, _child(_STOPPED, "signer"))
        monkeypatch.setattr(janitor, "_live_container_ids", lambda: frozenset())
        killed: list[tuple[int, int]] = []
        monkeypatch.setattr(janitor.os, "killpg", _killpg_removing(fake_proc, killed))

        result = reap_orphaned_supervisors()

        assert result == [(_STOPPED, None)]
        # Each child is its own pgid here; both get SIGTERM and die (their
        # /proc entry removed), so the SIGKILL escalation never fires.
        assert {pgid for pgid, sig in killed} == {5001, 5002}
        assert all(sig == signal.SIGTERM for _, sig in killed)

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
        killed: list[tuple[int, int]] = []
        monkeypatch.setattr(janitor.os, "killpg", _killpg_removing(fake_proc, killed))

        assert reap_orphaned_supervisors() == [(_STOPPED, None)]
        assert killed == [(7001, signal.SIGTERM)]

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

    def test_unknown_age_tree_is_excluded(
        self, fake_proc: Path, one_pgid: None, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """A member whose age can't be read marks the group unknown → not reaped."""
        _add(fake_proc, 8100, _child(_STOPPED, "vault"), stat=False)  # no stat → age unknown
        monkeypatch.setattr(janitor, "_live_container_ids", lambda: frozenset())
        monkeypatch.setattr(
            janitor.os, "killpg", lambda *a: pytest.fail("must not reap an unknown-age tree")
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

    def test_unreachable_podman_returns_none(
        self, fake_proc: Path, one_pgid: None, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Unknown liveness is ``None`` — distinct from an empty (clean) sweep."""
        _add(fake_proc, 1001, _child(_STOPPED, "vault"))
        monkeypatch.setattr(janitor, "_live_container_ids", lambda: None)
        monkeypatch.setattr(
            janitor.os, "killpg", lambda *a: pytest.fail("must not kill without ground truth")
        )

        assert reap_orphaned_supervisors() is None

    def test_sigterm_eperm_is_recorded_not_raised(
        self, fake_proc: Path, one_pgid: None, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """A SIGTERM that fails with EPERM is folded into the result, not raised."""
        _add(fake_proc, 1201, _child(_STOPPED, "vault"))
        monkeypatch.setattr(janitor, "_live_container_ids", lambda: frozenset())
        monkeypatch.setattr(janitor, "_KILL_GRACE_S", 0.0)

        def _killpg(pgid: int, sig: int) -> None:
            raise PermissionError("operation not permitted")

        monkeypatch.setattr(janitor.os, "killpg", _killpg)

        (cid, err) = reap_orphaned_supervisors()[0]
        assert cid == _STOPPED
        assert err is not None and "SIGTERM failed" in err

    def test_sigkill_escalation_reported(
        self, fake_proc: Path, one_pgid: None, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """A group that survives SIGTERM is SIGKILLed; an EPERM there is surfaced."""
        _add(fake_proc, 2001, _child(_STOPPED, "vault"))
        monkeypatch.setattr(janitor, "_live_container_ids", lambda: frozenset())
        monkeypatch.setattr(janitor, "_KILL_GRACE_S", 0.0)

        def _killpg(pgid: int, sig: int) -> None:
            if sig == signal.SIGKILL:
                raise PermissionError("operation not permitted")
            # SIGTERM is a no-op → the process lingers (fake /proc untouched).

        monkeypatch.setattr(janitor.os, "killpg", _killpg)

        (cid, err) = reap_orphaned_supervisors()[0]
        assert cid == _STOPPED
        assert err is not None and "SIGKILL failed" in err

    def test_group_racing_to_death_is_clean(
        self, fake_proc: Path, one_pgid: None, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """killpg racing ProcessLookupError (the group just exited) is not an error."""
        _add(fake_proc, 2100, _child(_STOPPED, "vault"))
        monkeypatch.setattr(janitor, "_live_container_ids", lambda: frozenset())

        def _killpg(pgid: int, sig: int) -> None:
            raise ProcessLookupError  # gone between the liveness check and the signal

        monkeypatch.setattr(janitor.os, "killpg", _killpg)

        assert reap_orphaned_supervisors() == [(_STOPPED, None)]


class TestPgidRecycleGuard:
    """A PGID recycled between the scan and the kill must never be signalled."""

    def test_recycled_pgid_is_not_signalled(
        self, fake_proc: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """A member whose PGID changed after the scan is dropped from the kill set."""
        _add(fake_proc, 3001, _child(_STOPPED, "vault"))
        # Scan records pgid=3001; by kill time getpgid reports a different
        # pgid (the PID was recycled into an unrelated session).
        calls = {"n": 0}

        def _getpgid(pid: int) -> int:
            calls["n"] += 1
            return 3001 if calls["n"] == 1 else 9999  # first (scan) matches, later differs

        monkeypatch.setattr(janitor.os, "getpgid", _getpgid)
        monkeypatch.setattr(janitor, "_live_container_ids", lambda: frozenset())
        monkeypatch.setattr(
            janitor.os, "killpg", lambda *a: pytest.fail("must not signal a recycled PGID")
        )

        assert reap_orphaned_supervisors() == [(_STOPPED, None)]

    def test_changed_starttime_is_not_signalled(
        self, fake_proc: Path, one_pgid: None, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """A PID whose start time changed (recycled) is not a member anymore."""
        _add(fake_proc, 3101, _child(_STOPPED, "vault"))
        member = janitor._Member(pid=3101, starttime_ticks=1, pgid=3101)  # wrong starttime
        assert janitor._member_present(member) is False


class TestLiveContainerIds:
    """The podman-backed ground truth parses ids by state."""

    def test_only_alive_states_count(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """Running/paused/created count as alive; exited/dead do not."""
        import json

        rows = [
            {"Id": _RUNNING, "State": "running"},
            {"Id": "c" * 64, "State": "created"},
            {"Id": _STOPPED, "State": "exited"},
            {"Id": "d" * 64, "State": "dead"},
        ]
        monkeypatch.setattr(janitor.shutil, "which", lambda _n: "/usr/bin/podman")

        class _Res:
            stdout = json.dumps(rows)

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
    """The /proc-derived identity/age helpers degrade gracefully."""

    def test_identity_none_when_stat_missing(self, fake_proc: Path) -> None:
        """A pid with no ``stat`` file yields ``None`` (never crashes the scan)."""
        assert janitor._process_identity(424242) is None

    def test_identity_none_when_starttime_malformed(self, fake_proc: Path) -> None:
        """A stat line with a non-numeric starttime field yields ``None``, not a crash."""
        pid_dir = fake_proc / "999"
        pid_dir.mkdir()
        (pid_dir / "stat").write_text("999 (py) S 1 not-a-number\n")
        assert janitor._process_identity(999) is None

    def test_member_present_true_on_full_match(self, fake_proc: Path, one_pgid: None) -> None:
        """A member whose PID, PGID, and start time all match is present."""
        _add(fake_proc, 4100, _child(_STOPPED, "vault"))
        starttime = janitor._process_identity(4100)[0]
        member = janitor._Member(pid=4100, starttime_ticks=starttime, pgid=4100)
        assert janitor._member_present(member) is True

    def test_member_present_false_when_pid_gone(
        self, fake_proc: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """A PID whose getpgid raises (gone) is not present."""
        monkeypatch.setattr(
            janitor.os, "getpgid", lambda pid: (_ for _ in ()).throw(ProcessLookupError)
        )
        assert janitor._member_present(janitor._Member(1, 1, 1)) is False

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

    def test_warns_when_podman_unreachable(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """Unknown liveness (reap → ``None``) is a warning, not a silent ok."""
        monkeypatch.setattr(janitor, "reap_orphaned_supervisors", lambda: None)
        verdict = janitor.make_orphan_supervisor_check().evaluate(0, "", "")
        assert verdict.severity == "warn"
        assert "unreachable" in verdict.detail

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
