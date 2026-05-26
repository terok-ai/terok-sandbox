# SPDX-FileCopyrightText: 2026 Jiri Vyskocil
# SPDX-License-Identifier: Apache-2.0

"""Tests for [`load_sidecar`][terok_sandbox.supervisor.main.load_sidecar].

The supervisor's first action is to parse the per-container sidecar
JSON at the path the OCI hook pinned via the ``terok.sandbox.sidecar``
annotation.  No XDG guessing — the function opens the named file
directly and either returns a [`SidecarConfig`][terok_sandbox.supervisor.main.SidecarConfig]
or soft-fails to ``None`` on every error path.
"""

from __future__ import annotations

import json
from pathlib import Path

from terok_sandbox.supervisor.main import SidecarConfig, load_sidecar


def _write_sidecar(tmp_path: Path, payload: dict[str, object]) -> Path:
    """Drop a sidecar JSON under a sidecar-shaped layout for the test."""
    target = tmp_path / "sidecar" / "demo.json"
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(json.dumps(payload))
    return target


class TestLoadSidecar:
    """Schema + I/O contract for the supervisor's sidecar reader."""

    def test_missing_file_returns_none(self, tmp_path: Path) -> None:
        """A path that doesn't exist soft-fails to None."""
        assert load_sidecar(tmp_path / "missing.json") is None

    def test_unix_sidecar_is_loaded(self, tmp_path: Path) -> None:
        """A socket-mode sidecar parses; vault/ssh socket paths are NOT
        carried (the supervisor derives them from container_name +
        runtime_dir via ``SupervisorPaths.for_container``)."""
        payload = {
            "container_name": "demo",
            "ipc_mode": "socket",
            "db_path": "/home/dev/.terok/vault.db",
            "runtime_dir": "/run/user/1000/terok/sandbox",
            "scope_id": "default",
        }
        path = _write_sidecar(tmp_path, payload)

        cfg = load_sidecar(path)
        assert isinstance(cfg, SidecarConfig)
        assert cfg.container_name == "demo"
        assert cfg.ipc_mode == "socket"
        assert cfg.scope_id == "default"
        assert cfg.tcp_port is None
        assert cfg.ssh_signer_port is None

    def test_tcp_mode_parses_port(self, tmp_path: Path) -> None:
        """TCP-mode sidecar carries per-container ``tcp_port`` +
        ``ssh_signer_port`` (allocated by the launch path via bind(0))."""
        payload = {
            "container_name": "demo-tcp",
            "ipc_mode": "tcp",
            "db_path": "/home/dev/.terok/vault.db",
            "runtime_dir": "/run/user/1000/terok/sandbox",
            "tcp_port": 54321,
            "ssh_signer_port": 54322,
        }
        path = _write_sidecar(tmp_path, payload)

        cfg = load_sidecar(path)
        assert cfg is not None
        assert cfg.ipc_mode == "tcp"
        assert cfg.tcp_port == 54321
        assert cfg.ssh_signer_port == 54322

    def test_malformed_json_returns_none(self, tmp_path: Path) -> None:
        """A non-JSON file degrades to ``None`` rather than raising."""
        bad = tmp_path / "sidecar" / "demo.json"
        bad.parent.mkdir(parents=True, exist_ok=True)
        bad.write_text("{ this is not json")
        assert load_sidecar(bad) is None

    def test_non_object_root_returns_none(self, tmp_path: Path) -> None:
        """JSON arrays / scalars at root degrade to ``None``."""
        bad = tmp_path / "sidecar" / "demo.json"
        bad.parent.mkdir(parents=True, exist_ok=True)
        bad.write_text("[]")
        assert load_sidecar(bad) is None

    def test_missing_required_field_returns_none(self, tmp_path: Path) -> None:
        """Missing ``db_path`` triggers the soft-fail KeyError path."""
        path = _write_sidecar(tmp_path, {"ipc_mode": "socket"})
        assert load_sidecar(path) is None
