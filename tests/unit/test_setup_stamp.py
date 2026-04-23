# SPDX-FileCopyrightText: 2026 Jiri Vyskocil
# SPDX-License-Identifier: Apache-2.0

"""Tests for the setup-stamp primitive (epic #685 phase 1).

Pins the five :class:`SetupVerdict` outcomes against a real on-disk
stamp file, plus the round-trip through :func:`write_stamp` /
:func:`needs_setup` and the atomic-write contract.

Each test relocates the stamp to a tmp dir via ``TEROK_ROOT`` so a
real installed stamp on the dev machine can't pollute the assertions.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from terok_sandbox import setup_stamp
from terok_sandbox.setup_stamp import (
    SetupVerdict,
    clear_stamp,
    needs_setup,
    stamp_path,
    write_stamp,
)


@pytest.fixture(autouse=True)
def _isolated_root(tmp_path, monkeypatch):
    """Re-route the stamp to a per-test tmp dir via ``TEROK_ROOT``."""
    monkeypatch.setenv("TEROK_ROOT", str(tmp_path))


# ── Verdicts ──────────────────────────────────────────────────────────


def test_first_run_when_stamp_absent() -> None:
    """A fresh install has no stamp → ``FIRST_RUN``."""
    assert needs_setup() is SetupVerdict.FIRST_RUN


def test_ok_after_write_with_unchanged_packages(monkeypatch) -> None:
    """Write a stamp, immediately read — installed versions match exactly → ``OK``."""
    monkeypatch.setattr(
        setup_stamp,
        "_installed_versions",
        lambda: {"terok-sandbox": "0.0.95"},
    )
    write_stamp()
    assert needs_setup() is SetupVerdict.OK


def test_stale_after_update_when_installed_is_newer(monkeypatch) -> None:
    """Bumping any installed package > stamp → ``STALE_AFTER_UPDATE``."""
    monkeypatch.setattr(
        setup_stamp,
        "_installed_versions",
        lambda: {"terok-sandbox": "0.0.95"},
    )
    write_stamp()
    monkeypatch.setattr(
        setup_stamp,
        "_installed_versions",
        lambda: {"terok-sandbox": "0.0.96"},
    )
    assert needs_setup() is SetupVerdict.STALE_AFTER_UPDATE


def test_stale_after_downgrade_when_installed_is_older(monkeypatch) -> None:
    """Any installed package < stamp → ``STALE_AFTER_DOWNGRADE`` — refuses by default."""
    monkeypatch.setattr(
        setup_stamp,
        "_installed_versions",
        lambda: {"terok-sandbox": "0.0.95"},
    )
    write_stamp()
    monkeypatch.setattr(
        setup_stamp,
        "_installed_versions",
        lambda: {"terok-sandbox": "0.0.90"},
    )
    assert needs_setup() is SetupVerdict.STALE_AFTER_DOWNGRADE


def test_stale_after_downgrade_when_installed_lost_a_package(monkeypatch) -> None:
    """A stamped package missing from the install also counts as a downgrade.

    Means setup state may reference packages that aren't there anymore —
    the install is functionally older than what setup expected to find.
    """
    monkeypatch.setattr(
        setup_stamp,
        "_installed_versions",
        lambda: {"terok-sandbox": "0.0.95", "terok-shield": "0.6.30"},
    )
    write_stamp()
    monkeypatch.setattr(
        setup_stamp,
        "_installed_versions",
        lambda: {"terok-sandbox": "0.0.95"},  # shield gone
    )
    assert needs_setup() is SetupVerdict.STALE_AFTER_DOWNGRADE


def test_stamp_corrupt_when_unparseable() -> None:
    """A garbage stamp doesn't trip on JSON errors — surfaces ``STAMP_CORRUPT``."""
    path = stamp_path()
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("not json", encoding="utf-8")
    assert needs_setup() is SetupVerdict.STAMP_CORRUPT


def test_stamp_corrupt_when_schema_version_mismatch() -> None:
    """A future schema version is treated as corrupt — explicit migration only."""
    path = stamp_path()
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps({"version": 999, "completed_at": "x", "packages": {}}),
        encoding="utf-8",
    )
    assert needs_setup() is SetupVerdict.STAMP_CORRUPT


@pytest.mark.parametrize(
    "content",
    [
        pytest.param(json.dumps([1, 2, 3]), id="root-is-array"),
        pytest.param(
            json.dumps({"version": 1, "completed_at": "x", "packages": "oops"}),
            id="packages-is-string",
        ),
    ],
)
def test_stamp_corrupt_when_shape_is_wrong(content: str) -> None:
    """A wrong-shape stamp surfaces as ``STAMP_CORRUPT`` instead of crashing the read."""
    path = stamp_path()
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(content, encoding="utf-8")
    assert needs_setup() is SetupVerdict.STAMP_CORRUPT


def test_stamp_corrupt_when_path_is_a_directory() -> None:
    """A directory at the stamp location is corrupt state, not a fresh install.

    Distinct from FIRST_RUN: setup hasn't been bypassed, the slot is
    occupied by something that ``write_stamp`` would also fail to
    overwrite.  Surfacing STAMP_CORRUPT keeps the diagnostic accurate.
    """
    stamp_path().mkdir(parents=True, exist_ok=True)
    assert needs_setup() is SetupVerdict.STAMP_CORRUPT


# ── write_stamp / clear_stamp ─────────────────────────────────────────


def test_write_stamp_atomic_via_tmp_then_rename(monkeypatch, tmp_path) -> None:
    """``write_stamp`` writes to ``setup.stamp.tmp`` first, then renames atomically.

    Guards against the half-stamp scenario: if the process is killed
    mid-write, the next launch sees no stamp (FIRST_RUN), not a half
    one (STAMP_CORRUPT).
    """
    monkeypatch.setattr(setup_stamp, "_installed_versions", lambda: {"x": "1.0.0"})
    path = write_stamp()
    assert path.is_file()
    # The temp sibling shouldn't survive a successful write.
    assert not path.with_suffix(path.suffix + ".tmp").exists()
    # Full schema contract — every documented field round-trips.
    payload = json.loads(path.read_text(encoding="utf-8"))
    assert payload["version"] == 1
    assert payload["packages"] == {"x": "1.0.0"}
    # ``completed_at`` is the audit trail downstream tooling reads;
    # dropping it would silently break consumers.
    from datetime import datetime

    assert datetime.fromisoformat(payload["completed_at"])


def test_clear_stamp_removes_file_returns_true(monkeypatch) -> None:
    """``clear_stamp`` removes a present file and reports True."""
    monkeypatch.setattr(setup_stamp, "_installed_versions", lambda: {"x": "1.0.0"})
    write_stamp()
    assert clear_stamp() is True
    assert not stamp_path().exists()


def test_clear_stamp_no_op_when_absent_returns_false() -> None:
    """``clear_stamp`` is idempotent — no file means no-op + False."""
    assert clear_stamp() is False


# ── _compare_versions fallback ────────────────────────────────────────


def test_installed_versions_includes_at_least_terok_sandbox() -> None:
    """The default reader walks the tracked-package list via ``importlib.metadata``.

    Run against the real install — the test harness has terok-sandbox
    installed, so its version must show up.  Other tracked packages
    (terok, terok-executor, etc.) may or may not be present in a
    sandbox-only environment; we don't assert on them.
    """
    from terok_sandbox.setup_stamp import _installed_versions

    versions = _installed_versions()
    assert "terok-sandbox" in versions
    # Any present version must be a non-empty string.
    assert all(isinstance(v, str) and v for v in versions.values())


def test_compare_versions_falls_back_to_string_compare_for_invalid_version() -> None:
    """Non-PEP-440 stamp strings fall back to lexicographic compare instead of crashing."""
    from terok_sandbox.setup_stamp import _compare_versions

    assert _compare_versions("not-a-version", "not-a-version") == 0
    assert _compare_versions("zzz", "aaa") == 1
    # Mixed: "0" < "n" — lexicographic, since the valid side can't be compared to an invalid one.
    assert _compare_versions("0.0.10", "not-a-version") == -1


# ── stamp location respects umbrella root ─────────────────────────────


def test_stamp_path_honours_terok_root(tmp_path) -> None:
    """``TEROK_ROOT`` (the umbrella state root) moves the stamp with it."""
    expected = Path(tmp_path) / "setup.stamp"
    assert stamp_path() == expected
