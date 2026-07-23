# SPDX-FileCopyrightText: 2026 Jiri Vyskocil
# SPDX-License-Identifier: Apache-2.0

"""Tests for [`confine_filesystem`][terok_sandbox._util._landlock.confine_filesystem].

The real restriction is irreversible and process-wide, so its effect is
exercised in a fresh interpreter (``subprocess``); the degradation logic is
checked in-process with the Landlock ABI probe stubbed out.
"""

from __future__ import annotations

import subprocess
import sys
import textwrap

import pytest

from terok_sandbox._util import _landlock
from terok_sandbox._util._landlock import LandlockReport, confine_filesystem

pytestmark = pytest.mark.skipif(sys.platform != "linux", reason="Landlock is Linux-only")


def test_confines_reads_and_writes_to_the_lane(tmp_path) -> None:
    """A fresh process reads+writes its lane, but a sibling is unreadable and unwritable."""
    ro = tmp_path / "ro"
    rw = tmp_path / "rw"
    outside = tmp_path / "outside"
    for d in (ro, rw, outside):
        d.mkdir()
    (outside / "secret").write_text("classified")  # a sibling's file it must not read

    probe = textwrap.dedent(
        f"""
        import ctypes
        from pathlib import Path
        from terok_sandbox._util._landlock import confine_filesystem

        ctypes.CDLL(None, use_errno=True).prctl(38, 1, 0, 0, 0)  # no_new_privs
        report = confine_filesystem([Path({str(ro)!r})], [Path({str(rw)!r})])
        if not report.confined:
            print(f"unsupported:{{report.reason}}")
            raise SystemExit(0)

        out = []
        Path({str(rw)!r}, "ok").write_text("x")            # read-write lane → write OK
        list(Path({str(ro)!r}).iterdir())                  # read-exec lane → read OK
        try:
            Path({str(ro)!r}, "no").write_text("x")
            out.append("ro-write-LEAK")
        except PermissionError:
            out.append("ro-write-denied")           # read-exec lane is not writable
        try:
            Path({str(outside)!r}, "secret").read_text()
            out.append("sibling-read-LEAK")
        except (PermissionError, OSError):
            out.append("sibling-read-denied")        # outside the lane → not even readable
        print(";".join(out))
        """
    )
    result = subprocess.run(
        [sys.executable, "-c", probe], capture_output=True, text=True, check=True
    )
    line = result.stdout.strip()
    if line.startswith("unsupported:"):
        pytest.skip(f"kernel without Landlock: {line}")
    assert line == "ro-write-denied;sibling-read-denied", f"confinement leaked: {line!r}"


def test_unsupported_kernel_is_a_noop(monkeypatch) -> None:
    """A kernel without Landlock reports ``confined=False`` and restricts nothing."""
    monkeypatch.setattr(_landlock, "_landlock_abi", lambda _libc: -1)
    report = confine_filesystem([], [])
    assert isinstance(report, LandlockReport)
    assert report.confined is False
    assert "unavailable" in report.reason


def test_missing_lane_path_is_skipped(monkeypatch, tmp_path) -> None:
    """A non-existent grant path is skipped rather than aborting the restriction.

    Stubs ``restrict_self`` so the test process itself stays unconfined while
    still exercising the add-rule loop over a path that isn't there.
    """
    calls: list[int] = []
    real_syscall = _landlock.ctypes.CDLL(None, use_errno=True).syscall

    def _syscall(nr, *args):  # noqa: ANN001, ANN202
        if nr == _landlock._NR_RESTRICT_SELF:
            calls.append(nr)
            return 0
        return real_syscall(nr, *args)

    fake_libc = type("Libc", (), {"syscall": staticmethod(_syscall)})()
    monkeypatch.setattr(_landlock.ctypes, "CDLL", lambda *_a, **_k: fake_libc)

    report = confine_filesystem([tmp_path / "nope-r"], [tmp_path / "nope-w"])
    assert report.confined is True  # restrict_self stubbed to succeed
    assert calls == [_landlock._NR_RESTRICT_SELF]
