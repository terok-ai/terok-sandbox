# SPDX-FileCopyrightText: 2026 Jiri Vyskocil
# SPDX-License-Identifier: Apache-2.0

"""Tests for the post-start supervision check (issue #458, fix 2)."""

from __future__ import annotations

import json
import socket
from pathlib import Path

import pytest

from terok_sandbox.config import SandboxConfig
from terok_sandbox.supervision import (
    SupervisionStatus,
    verify_supervision,
    warn_unsupervised,
)

_NAME = "demo-cli-w9xk3"
_FAST = 0.2  # timeout for the intentionally-missing cases — keep the suite snappy


def _cfg(state_dir: Path) -> SandboxConfig:
    return SandboxConfig(state_dir=state_dir)


def _write_sidecar(
    state_dir: Path, runtime_dir: Path, *, ipc_mode: str = "socket", gate: bool = False
) -> None:
    """Drop a valid sidecar under ``<state>/sidecar/<name>.json``."""
    sidecar_dir = state_dir / "sidecar"
    sidecar_dir.mkdir(parents=True, exist_ok=True)
    payload: dict[str, object] = {
        "container_name": _NAME,
        "ipc_mode": ipc_mode,
        "db_path": str(state_dir / "v.db"),
        "runtime_dir": str(runtime_dir),
    }
    if gate:
        payload["gate_base_path"] = str(state_dir / "gate")
        payload["gate_token"] = "tok123"
    (sidecar_dir / f"{_NAME}.json").write_text(json.dumps(payload))


def _socket_paths(runtime_dir: Path) -> tuple[Path, Path]:
    """The (vault, gate) sockets the supervisor binds for ``_NAME`` in socket mode."""
    per_container = runtime_dir / "run" / _NAME
    return per_container / "vault.sock", per_container / "gate-server.sock"


def _bind(path: Path) -> socket.socket:
    """Bind a real AF_UNIX socket at *path* (kept alive by the returned handle)."""
    path.parent.mkdir(parents=True, exist_ok=True)
    sock = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
    sock.bind(str(path))
    return sock


class TestVerifySupervision:
    def test_no_sidecar_is_skipped(self, tmp_path: Path) -> None:
        """No sidecar on disk ⇒ nothing to verify, no polling, healthy-ok."""
        status = verify_supervision(_cfg(tmp_path), _NAME, timeout=_FAST)
        assert status.skipped and status.ok
        assert status.checked == () and status.missing == ()

    def test_tcp_mode_is_skipped(self, tmp_path: Path) -> None:
        """TCP wiring binds loopback ports, not sockets — nothing on disk to poll."""
        _write_sidecar(tmp_path, tmp_path / "rt", ipc_mode="tcp")
        status = verify_supervision(_cfg(tmp_path), _NAME, timeout=_FAST)
        assert status.skipped and status.ok

    def test_vault_socket_present_is_ok(self, tmp_path: Path) -> None:
        rt = tmp_path / "rt"
        _write_sidecar(tmp_path, rt)
        vault, _gate = _socket_paths(rt)
        keepalive = _bind(vault)
        try:
            status = verify_supervision(_cfg(tmp_path), _NAME)
        finally:
            keepalive.close()
        assert status.ok and not status.skipped
        assert status.checked == (vault,) and status.missing == ()

    def test_missing_vault_socket_is_flagged(self, tmp_path: Path) -> None:
        _write_sidecar(tmp_path, tmp_path / "rt")
        vault, _gate = _socket_paths(tmp_path / "rt")
        status = verify_supervision(_cfg(tmp_path), _NAME, timeout=_FAST)
        assert not status.ok
        assert status.missing == (vault,)

    def test_gate_socket_polled_only_when_wired(self, tmp_path: Path) -> None:
        rt = tmp_path / "rt"
        _write_sidecar(tmp_path, rt, gate=True)
        vault, gate = _socket_paths(rt)
        va, ga = _bind(vault), _bind(gate)
        try:
            status = verify_supervision(_cfg(tmp_path), _NAME)
        finally:
            va.close()
            ga.close()
        assert status.ok
        assert set(status.checked) == {vault, gate}

    def test_missing_gate_socket_is_flagged(self, tmp_path: Path) -> None:
        """Vault up but the wired gate never bound ⇒ only the gate is reported."""
        rt = tmp_path / "rt"
        _write_sidecar(tmp_path, rt, gate=True)
        vault, gate = _socket_paths(rt)
        keepalive = _bind(vault)
        try:
            status = verify_supervision(_cfg(tmp_path), _NAME, timeout=_FAST)
        finally:
            keepalive.close()
        assert status.missing == (gate,)


class TestSupervisionStatus:
    def test_warning_names_container_socket_and_diary(self, tmp_path: Path) -> None:
        missing = tmp_path / "run" / _NAME / "vault.sock"
        hook_log = tmp_path / "logs" / "hook.log"
        status = SupervisionStatus(_NAME, (missing,), (missing,), hook_log)
        text = status.warning()
        assert _NAME in text
        assert str(missing) in text
        assert str(hook_log) in text

    def test_warn_unsupervised_is_silent_when_healthy(
        self, tmp_path: Path, capsys: pytest.CaptureFixture[str]
    ) -> None:
        healthy = SupervisionStatus(_NAME, (), (), tmp_path / "logs" / "hook.log")
        warn_unsupervised(healthy)
        assert capsys.readouterr().err == ""

    def test_warn_unsupervised_shouts_on_failure(
        self, tmp_path: Path, capsys: pytest.CaptureFixture[str]
    ) -> None:
        missing = tmp_path / "run" / _NAME / "vault.sock"
        warn_unsupervised(SupervisionStatus(_NAME, (missing,), (missing,), tmp_path / "hook.log"))
        assert "not responding" in capsys.readouterr().err
