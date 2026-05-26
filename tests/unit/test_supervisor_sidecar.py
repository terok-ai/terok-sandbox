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

    def test_gate_fields_parsed(self, tmp_path: Path) -> None:
        """``gate_base_path`` / ``gate_token`` / ``gate_port`` round-trip."""
        payload = {
            "container_name": "demo-gate",
            "ipc_mode": "tcp",
            "db_path": "/home/dev/.terok/vault.db",
            "runtime_dir": "/run/user/1000/terok/sandbox",
            "project_id": "myproj",
            "gate_base_path": "/home/dev/.terok/gate",
            "gate_token": "terok-g-abc",
            "gate_port": 54323,
        }
        path = _write_sidecar(tmp_path, payload)

        cfg = load_sidecar(path)
        assert cfg is not None
        assert cfg.gate_base_path == Path("/home/dev/.terok/gate")
        assert cfg.gate_token == "terok-g-abc"
        assert cfg.gate_port == 54323
        assert cfg.project_id == "myproj"

    def test_gate_fields_absent_default_to_none(self, tmp_path: Path) -> None:
        """A sidecar without gate fields parses with gate disabled."""
        payload = {
            "container_name": "demo",
            "ipc_mode": "socket",
            "db_path": "/home/dev/.terok/vault.db",
            "runtime_dir": "/run/user/1000/terok/sandbox",
        }
        cfg = load_sidecar(_write_sidecar(tmp_path, payload))
        assert cfg is not None
        assert cfg.gate_base_path is None
        assert cfg.gate_token is None
        assert cfg.gate_port is None

    def test_relative_gate_base_path_returns_none(self, tmp_path: Path) -> None:
        """A relative ``gate_base_path`` soft-fails like the other paths."""
        payload = {
            "container_name": "demo",
            "ipc_mode": "socket",
            "db_path": "/home/dev/.terok/vault.db",
            "runtime_dir": "/run/user/1000/terok/sandbox",
            "gate_base_path": "relative/gate",
            "gate_token": "terok-g-abc",
        }
        assert load_sidecar(_write_sidecar(tmp_path, payload)) is None

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

    def test_unsafe_container_name_rejected(self, tmp_path: Path) -> None:
        """A ``container_name`` that escapes its runtime subdir is refused.

        The name is interpolated into ``runtime_dir/run/<name>`` which the
        supervisor mkdir's / chmod's / rmtree's, so a value with a path
        separator or ``..`` must soft-fail rather than redirect those
        filesystem operations.
        """
        for bad_name in ("../evil", "a/b", "/abs", "..", "."):
            payload = {
                "container_name": bad_name,
                "ipc_mode": "socket",
                "db_path": "/home/dev/.terok/vault.db",
                "runtime_dir": "/run/user/1000/terok/sandbox",
            }
            path = _write_sidecar(tmp_path, payload)
            assert load_sidecar(path) is None, bad_name

    def test_invalid_ipc_mode_rejected(self, tmp_path: Path) -> None:
        """``ipc_mode`` outside {socket, tcp} soft-fails to ``None``.

        The supervisor branches on this value to pick the bind transport;
        an unknown mode would fall through to the socket path silently, so
        the parser refuses it up front.
        """
        payload = {
            "container_name": "demo",
            "ipc_mode": "carrier-pigeon",
            "db_path": "/home/dev/.terok/vault.db",
            "runtime_dir": "/run/user/1000/terok/sandbox",
        }
        assert load_sidecar(_write_sidecar(tmp_path, payload)) is None

    def test_missing_ipc_mode_defaults_to_socket(self, tmp_path: Path) -> None:
        """An absent ``ipc_mode`` defaults to socket (the common case)."""
        payload = {
            "container_name": "demo",
            "db_path": "/home/dev/.terok/vault.db",
            "runtime_dir": "/run/user/1000/terok/sandbox",
        }
        cfg = load_sidecar(_write_sidecar(tmp_path, payload))
        assert cfg is not None
        assert cfg.ipc_mode == "socket"

    def test_relative_db_path_returns_none(self, tmp_path: Path) -> None:
        """A relative ``db_path`` soft-fails — the supervisor binds/rmtrees
        against it and must not resolve against the hook's cwd."""
        payload = {
            "container_name": "demo",
            "ipc_mode": "socket",
            "db_path": "relative/vault.db",
            "runtime_dir": "/run/user/1000/terok/sandbox",
        }
        assert load_sidecar(_write_sidecar(tmp_path, payload)) is None

    def test_relative_runtime_dir_returns_none(self, tmp_path: Path) -> None:
        """A relative ``runtime_dir`` soft-fails for the same reason."""
        payload = {
            "container_name": "demo",
            "ipc_mode": "socket",
            "db_path": "/home/dev/.terok/vault.db",
            "runtime_dir": "run/sandbox",
        }
        assert load_sidecar(_write_sidecar(tmp_path, payload)) is None

    def test_absolute_dossier_path_round_trips(self, tmp_path: Path) -> None:
        """An absolute ``dossier_path`` parses through to the config."""
        payload = {
            "container_name": "demo",
            "ipc_mode": "socket",
            "db_path": "/home/dev/.terok/vault.db",
            "runtime_dir": "/run/user/1000/terok/sandbox",
            "dossier_path": "/home/dev/.terok/dossier.json",
        }
        cfg = load_sidecar(_write_sidecar(tmp_path, payload))
        assert cfg is not None
        assert cfg.dossier_path == Path("/home/dev/.terok/dossier.json")

    def test_relative_dossier_path_returns_none(self, tmp_path: Path) -> None:
        """A relative ``dossier_path`` soft-fails like the other paths."""
        payload = {
            "container_name": "demo",
            "ipc_mode": "socket",
            "db_path": "/home/dev/.terok/vault.db",
            "runtime_dir": "/run/user/1000/terok/sandbox",
            "dossier_path": "relative/dossier.json",
        }
        assert load_sidecar(_write_sidecar(tmp_path, payload)) is None

    def test_non_integer_port_returns_none(self, tmp_path: Path) -> None:
        """A non-integer ``tcp_port`` trips the schema-error soft-fail path.

        ``int(raw["tcp_port"])`` raises ``ValueError`` for a non-numeric
        string; ``load_sidecar`` must collapse that to ``None`` rather than
        propagate (the supervisor surfaces ``None`` as exit-code 2)."""
        payload = {
            "container_name": "demo",
            "ipc_mode": "tcp",
            "db_path": "/home/dev/.terok/vault.db",
            "runtime_dir": "/run/user/1000/terok/sandbox",
            "tcp_port": "not-a-port",
        }
        assert load_sidecar(_write_sidecar(tmp_path, payload)) is None
