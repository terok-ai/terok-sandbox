# SPDX-FileCopyrightText: 2026 Jiri Vyskocil
# SPDX-License-Identifier: Apache-2.0

"""Unit tests for the container-diagnostics path resolver."""

from __future__ import annotations

from pathlib import Path

from terok_sandbox import ContainerDiagnostics, container_diagnostics, diagnostics as diag

_CID = "abc123def456"
_CNAME = "demo-cli-w9xk3"


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
