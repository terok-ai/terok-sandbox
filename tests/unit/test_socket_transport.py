# SPDX-FileCopyrightText: 2026 Jiri Vyskocil
# SPDX-License-Identifier: Apache-2.0

"""Tests for Unix socket transport support across services.

Covers: probe_unix_socket utility, SandboxConfig socket path properties,
gate server Unix socket factory, gate lifecycle socket reachability,
and SSH signer Unix socket mode.
"""

from __future__ import annotations

import asyncio
import socket
import struct
import unittest.mock
from pathlib import Path

import pytest

from terok_sandbox._util._net import harden_socket, prepare_socket_path, probe_unix_socket
from terok_sandbox.config import SandboxConfig
from terok_sandbox.credentials.db import CredentialDB
from terok_sandbox.gate.lifecycle import GateServerManager, GateServerStatus
from terok_sandbox.vault.ssh_signer import (
    SSH_AGENT_IDENTITIES_ANSWER,
    SSH_AGENTC_REQUEST_IDENTITIES,
    _unpack_string,
    start_ssh_signer,
)
from tests.constants import MOCK_BASE

MOCK_RUNTIME_DIR = MOCK_BASE / "runtime"


# ── probe_unix_socket ───────────────────────────────────────────────────


class TestProbeUnixSocket:
    """Verify the shared Unix socket probe helper."""

    def test_returns_true_for_listening_socket(self, tmp_path: Path) -> None:
        """Probe succeeds when a real listener is bound to the path."""
        sock_path = tmp_path / "test.sock"
        srv = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
        srv.bind(str(sock_path))
        srv.listen(1)
        try:
            assert probe_unix_socket(sock_path) is True
        finally:
            srv.close()

    def test_returns_false_for_nonexistent_path(self, tmp_path: Path) -> None:
        """Probe returns False when the socket file doesn't exist."""
        assert probe_unix_socket(tmp_path / "missing.sock") is False

    def test_returns_false_for_dead_socket(self, tmp_path: Path) -> None:
        """Probe returns False when the socket file exists but nobody is listening."""
        sock_path = tmp_path / "dead.sock"
        srv = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
        srv.bind(str(sock_path))
        srv.listen(1)
        srv.close()
        # Socket file still exists, but no listener
        assert probe_unix_socket(sock_path) is False


# ── prepare_socket_path / harden_socket ─────────────────────────────────


class TestPrepareSocketPath:
    """Verify shared socket path preparation utility."""

    def test_removes_stale_socket(self, tmp_path: Path) -> None:
        """Stale socket file is unlinked."""
        sock_path = tmp_path / "s.sock"
        srv = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
        srv.bind(str(sock_path))
        srv.close()

        prepare_socket_path(sock_path)
        assert not sock_path.exists()

    def test_rejects_non_socket(self, tmp_path: Path) -> None:
        """RuntimeError when a regular file occupies the path."""
        path = tmp_path / "s.sock"
        path.write_text("x")
        with pytest.raises(RuntimeError, match="non-socket"):
            prepare_socket_path(path)

    def test_creates_parent_dirs(self, tmp_path: Path) -> None:
        """Parent directories are created when missing."""
        path = tmp_path / "a" / "b" / "s.sock"
        prepare_socket_path(path)
        assert path.parent.is_dir()

    def test_noop_when_absent(self, tmp_path: Path) -> None:
        """No error when socket path does not exist yet."""
        prepare_socket_path(tmp_path / "new.sock")


class TestHardenSocket:
    """Verify socket permission hardening."""

    def test_sets_owner_only(self, tmp_path: Path) -> None:
        """Socket file is restricted to owner-only access."""
        import stat

        sock_path = tmp_path / "s.sock"
        srv = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
        srv.bind(str(sock_path))
        srv.close()

        harden_socket(sock_path)
        mode = stat.S_IMODE(sock_path.stat().st_mode)
        assert mode == 0o600


# ── SandboxConfig socket paths ──────────────────────────────────────────


class TestConfigSocketPaths:
    """Verify derived socket path properties on SandboxConfig."""

    def test_gate_socket_path(self) -> None:
        """gate_socket_path returns runtime_dir / 'gate-server.sock'."""
        cfg = SandboxConfig(runtime_dir=MOCK_RUNTIME_DIR)
        assert cfg.gate_socket_path == MOCK_RUNTIME_DIR / "gate-server.sock"

    def test_ssh_signer_socket_path(self) -> None:
        """ssh_signer_socket_path returns runtime_dir / 'ssh-agent.sock'."""
        cfg = SandboxConfig(runtime_dir=MOCK_RUNTIME_DIR)
        assert cfg.ssh_signer_socket_path == MOCK_RUNTIME_DIR / "ssh-agent.sock"


# ── Gate server: _create_unix_server ────────────────────────────────────


class TestCreateUnixServer:
    """Verify the gate HTTP server Unix socket factory."""

    def test_creates_socket_at_path(self, tmp_path: Path) -> None:
        """Server binds to the given socket path."""
        from terok_sandbox.gate.server import _create_unix_server, _make_handler_class

        sock_path = tmp_path / "gate.sock"
        handler = _make_handler_class(tmp_path, unittest.mock.Mock())
        server = _create_unix_server(handler, sock_path)
        try:
            assert sock_path.exists()
            # Verify we can connect to it
            client = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
            client.connect(str(sock_path))
            client.close()
        finally:
            server.socket.close()

    def test_removes_stale_socket(self, tmp_path: Path) -> None:
        """Stale socket file is removed before binding."""
        from terok_sandbox.gate.server import _create_unix_server, _make_handler_class

        sock_path = tmp_path / "gate.sock"
        # Create a stale socket
        stale = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
        stale.bind(str(sock_path))
        stale.close()
        assert sock_path.exists()

        handler = _make_handler_class(tmp_path, unittest.mock.Mock())
        server = _create_unix_server(handler, sock_path)
        try:
            assert sock_path.exists()
        finally:
            server.socket.close()

    def test_rejects_non_socket_file(self, tmp_path: Path) -> None:
        """RuntimeError raised when path exists but is a regular file."""
        from terok_sandbox.gate.server import _create_unix_server, _make_handler_class

        sock_path = tmp_path / "gate.sock"
        sock_path.write_text("not a socket")

        handler = _make_handler_class(tmp_path, unittest.mock.Mock())
        with pytest.raises(RuntimeError, match="Refusing to remove non-socket"):
            _create_unix_server(handler, sock_path)

    def test_creates_parent_directories(self, tmp_path: Path) -> None:
        """Parent directories are created if they don't exist."""
        from terok_sandbox.gate.server import _create_unix_server, _make_handler_class

        sock_path = tmp_path / "sub" / "dir" / "gate.sock"
        handler = _make_handler_class(tmp_path, unittest.mock.Mock())
        server = _create_unix_server(handler, sock_path)
        try:
            assert sock_path.exists()
        finally:
            server.socket.close()


# ── Gate lifecycle: socket reachability ──────────────────────────────────


class TestWaitForUnixSocket:
    """Verify the _wait_for_unix_socket retry loop."""

    def test_returns_true_for_immediate_listener(self, tmp_path: Path) -> None:
        """Succeeds immediately when the socket is already listening."""
        from terok_sandbox.vault.lifecycle import VaultManager

        sock_path = tmp_path / "test.sock"
        srv = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
        srv.bind(str(sock_path))
        srv.listen(1)
        try:
            assert VaultManager._wait_for_unix_socket(sock_path, timeout=1.0) is True
        finally:
            srv.close()

    def test_returns_false_on_timeout(self, tmp_path: Path) -> None:
        """Returns False when socket never appears within timeout."""
        from terok_sandbox.vault.lifecycle import VaultManager

        missing = tmp_path / "missing.sock"
        assert VaultManager._wait_for_unix_socket(missing, timeout=0.3) is False


class TestGateSocketReachability:
    """Verify gate lifecycle socket-mode detection."""

    def test_socket_reachable_returns_daemon_running(self) -> None:
        """get_status reports daemon/running when Unix socket is reachable."""
        mock_cfg = unittest.mock.MagicMock(spec=SandboxConfig)
        mock_cfg.gate_port = 9418
        with unittest.mock.patch.object(
            GateServerManager, "__init__", lambda self, cfg=None: setattr(self, "_cfg", mock_cfg)
        ):
            mgr = GateServerManager()
            with (
                unittest.mock.patch.object(mgr, "is_socket_installed", return_value=False),
                unittest.mock.patch.object(mgr, "is_socket_reachable", return_value=True),
            ):
                status = mgr.get_status()
        assert status == GateServerStatus(
            mode="daemon", running=True, port=9418, transport="socket"
        )

    def test_systemd_installed_inactive_socket_reachable(self) -> None:
        """Foreground socket server detected even when systemd units are installed but inactive."""
        mock_cfg = unittest.mock.MagicMock(spec=SandboxConfig)
        mock_cfg.gate_port = 9418
        with unittest.mock.patch.object(
            GateServerManager, "__init__", lambda self, cfg=None: setattr(self, "_cfg", mock_cfg)
        ):
            mgr = GateServerManager()
            with (
                unittest.mock.patch.object(mgr, "is_socket_installed", return_value=True),
                unittest.mock.patch.object(mgr, "is_socket_active", return_value=False),
                unittest.mock.patch.object(mgr, "is_daemon_running", return_value=False),
                unittest.mock.patch.object(mgr, "is_socket_reachable", return_value=True),
            ):
                status = mgr.get_status()
        assert status == GateServerStatus(
            mode="daemon", running=True, port=9418, transport="socket"
        )

    def test_socket_not_reachable_falls_through(self) -> None:
        """get_status falls through to daemon PID check when socket is not reachable."""
        mock_cfg = unittest.mock.MagicMock(spec=SandboxConfig)
        mock_cfg.gate_port = 9418
        with unittest.mock.patch.object(
            GateServerManager, "__init__", lambda self, cfg=None: setattr(self, "_cfg", mock_cfg)
        ):
            mgr = GateServerManager()
            with (
                unittest.mock.patch.object(mgr, "is_socket_installed", return_value=False),
                unittest.mock.patch.object(mgr, "is_socket_reachable", return_value=False),
                unittest.mock.patch.object(mgr, "is_daemon_running", return_value=False),
            ):
                status = mgr.get_status()
        assert status == GateServerStatus(mode="none", running=False, port=9418)


# ── SSH agent: Unix socket mode ─────────────────────────────────────────


def _build_handshake(token: str) -> bytes:
    """Build the phantom-token handshake prefix."""
    encoded = token.encode("utf-8")
    return struct.pack(">I", len(encoded)) + encoded


def _build_msg(msg_type: int, payload: bytes = b"") -> bytes:
    """Build one SSH agent wire-format message."""
    body = bytes([msg_type]) + payload
    return struct.pack(">I", len(body)) + body


async def _read_response(reader: asyncio.StreamReader) -> tuple[int, bytes]:
    """Read one SSH agent response message."""
    raw_len = await reader.readexactly(4)
    (msg_len,) = struct.unpack(">I", raw_len)
    body = await reader.readexactly(msg_len)
    return body[0], body[1:]


@pytest.mark.asyncio()
class TestSSHSignerUnixSocket:
    """Verify the SSH agent server in Unix socket mode."""

    async def test_roundtrip_via_unix_socket(self, tmp_path: Path) -> None:
        """Full handshake + identity listing via a Unix domain socket."""
        from terok_sandbox.credentials.ssh_keypair import generate_keypair

        kp = generate_keypair("ed25519", comment="test-socket")
        db = CredentialDB(tmp_path / "test.db")
        key_id = db.store_ssh_key(
            key_type=kp.key_type,
            private_der=kp.private_der,
            public_blob=kp.public_blob,
            comment=kp.comment,
            fingerprint=kp.fingerprint,
        )
        db.assign_ssh_key("proj", key_id)
        token = db.create_token("proj", "task-1", "proj", "ssh")
        db.close()
        pub_blob = kp.public_blob

        sock_path = tmp_path / "ssh-agent.sock"
        server = await start_ssh_signer(str(tmp_path / "test.db"), socket_path=str(sock_path))
        try:
            assert sock_path.exists()

            reader, writer = await asyncio.open_unix_connection(str(sock_path))
            writer.write(_build_handshake(token))
            writer.write(_build_msg(SSH_AGENTC_REQUEST_IDENTITIES))
            await writer.drain()

            msg_type, payload = await _read_response(reader)
            assert msg_type == SSH_AGENT_IDENTITIES_ANSWER
            (nkeys,) = struct.unpack_from(">I", payload, 0)
            assert nkeys == 1

            mv = memoryview(payload)
            returned_blob, _ = _unpack_string(mv, 4)
            assert returned_blob == pub_blob

            writer.close()
            await writer.wait_closed()
        finally:
            server.close()
            await server.wait_closed()

    async def test_socket_rejects_non_socket_file(self, tmp_path: Path) -> None:
        """RuntimeError when a regular file exists at the socket path."""
        sock_path = tmp_path / "ssh-agent.sock"
        sock_path.write_text("not a socket")

        db = CredentialDB(tmp_path / "test.db")
        db.close()

        with pytest.raises(RuntimeError, match="Refusing to remove non-socket"):
            await start_ssh_signer(str(tmp_path / "test.db"), socket_path=str(sock_path))

    async def test_socket_removes_stale_socket(self, tmp_path: Path) -> None:
        """Stale socket file is cleaned up before binding."""
        sock_path = tmp_path / "ssh-agent.sock"
        stale = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
        stale.bind(str(sock_path))
        stale.close()
        assert sock_path.exists()

        db = CredentialDB(tmp_path / "test.db")
        db.close()

        server = await start_ssh_signer(str(tmp_path / "test.db"), socket_path=str(sock_path))
        try:
            assert sock_path.exists()
        finally:
            server.close()
            await server.wait_closed()

    async def test_raises_without_transport(self, tmp_path: Path) -> None:
        """ValueError when neither socket_path nor host+port is given."""
        db = CredentialDB(tmp_path / "test.db")
        db.close()

        with pytest.raises(ValueError, match="Either socket_path or host\\+port"):
            await start_ssh_signer(str(tmp_path / "test.db"))


# ── Gate server: _serve_foreground validation ────────────────────────────


class TestServeForeground:
    """Verify _serve_foreground argument validation."""

    def test_raises_without_socket_or_port(self, tmp_path: Path) -> None:
        """RuntimeError when neither socket_path nor port is given."""
        from terok_sandbox.gate.server import _serve_foreground

        with pytest.raises(RuntimeError, match="--socket-path or --port"):
            _serve_foreground(tmp_path, unittest.mock.Mock())

    def test_tcp_server_created(self, tmp_path: Path) -> None:
        """TCP-only mode creates a listening server on the given port."""
        import threading

        from terok_sandbox.gate.server import _serve_foreground

        def _run():
            _serve_foreground(tmp_path, unittest.mock.Mock(), port=0)

        # signal.signal() only works in the main thread — mock it out
        with unittest.mock.patch("terok_sandbox.gate.server.signal.signal"):
            t = threading.Thread(target=_run, daemon=True)
            t.start()
            import time

            time.sleep(0.3)
            assert t.is_alive()

    def test_pid_file_written(self, tmp_path: Path) -> None:
        """PID file is created when pid_file is specified."""
        import os
        import threading

        from terok_sandbox.gate.server import _serve_foreground

        pid_file = tmp_path / "gate.pid"
        sock_path = tmp_path / "gate.sock"

        def _run():
            _serve_foreground(
                tmp_path,
                unittest.mock.Mock(),
                socket_path=sock_path,
                pid_file=pid_file,
            )

        with unittest.mock.patch("terok_sandbox.gate.server.signal.signal"):
            t = threading.Thread(target=_run, daemon=True)
            t.start()
            import time

            time.sleep(0.2)
            assert pid_file.exists()
            assert pid_file.read_text().strip() == str(os.getpid())


# ── Gate server: CLI --port sentinel ─────────────────────────────────────


class TestGateMainPortDefault:
    """Verify --port default/sentinel behavior in the gate CLI."""

    def test_port_default_is_none(self) -> None:
        """Argparse default for --port is None (sentinel), not 9418."""

        from terok_sandbox.gate.server import main

        # Extract the parser by intercepting parse_args
        with (
            unittest.mock.patch(
                "sys.argv",
                [
                    "terok-gate",
                    "--base-path=/tmp/terok-testing/b",
                    "--token-file=/tmp/terok-testing/t",
                    "--socket-path=/tmp/terok-testing/s",
                ],
            ),
            unittest.mock.patch("terok_sandbox.gate.server._serve_foreground") as mock_fg,
        ):
            main()

        # port should be None (not provided), not 9418
        _, kwargs = mock_fg.call_args
        assert kwargs["port"] is None

    def test_explicit_port_passed_through(self) -> None:
        """Explicit --port 9418 is forwarded, not swallowed."""
        with (
            unittest.mock.patch(
                "sys.argv",
                [
                    "terok-gate",
                    "--base-path=/tmp/terok-testing/b",
                    "--token-file=/tmp/terok-testing/t",
                    "--socket-path=/tmp/terok-testing/s",
                    "--port=9418",
                ],
            ),
            unittest.mock.patch("terok_sandbox.gate.server._serve_foreground") as mock_fg,
        ):
            from terok_sandbox.gate.server import main

            main()

        _, kwargs = mock_fg.call_args
        assert kwargs["port"] == 9418


# ── Vault: SSH signer mutual exclusion ───────────────────────────────────


class TestSSHSignerMutualExclusion:
    """Verify --ssh-signer-port and --ssh-signer-socket-path are mutually exclusive."""

    def test_rejects_both_port_and_socket(self) -> None:
        """Passing both --ssh-signer-port and --ssh-signer-socket-path is an error."""
        with unittest.mock.patch(
            "sys.argv",
            [
                "terok-vault",
                "--socket-path=/tmp/terok-testing/proxy.sock",
                "--db-path=/tmp/terok-testing/db",
                "--routes-file=/tmp/terok-testing/routes.json",
                "--ssh-signer-port=18732",
                "--ssh-signer-socket-path=/tmp/terok-testing/ssh.sock",
                "--ssh-keys-file=/tmp/terok-testing/keys.json",
            ],
        ):
            from terok_sandbox.vault.token_broker import main as vault_main

            with pytest.raises(SystemExit):
                vault_main()


class TestInstallSystemdPortGuards:
    """Verify tcp-mode installers refuse to render when the port is unset.

    Prevents the class of bug where ``services.mode: socket`` (which
    skips port allocation) reaches the tcp install path by mistake and
    emits ``ListenStream=127.0.0.1:None`` — systemd rejects that.
    """

    def test_gate_tcp_install_without_port_raises(self) -> None:
        """GateServerManager.install_systemd_units(transport='tcp') needs gate_port."""
        mock_cfg = unittest.mock.MagicMock(spec=SandboxConfig)
        mock_cfg.gate_port = None
        with unittest.mock.patch.object(
            GateServerManager, "__init__", lambda self, cfg=None: setattr(self, "_cfg", mock_cfg)
        ):
            mgr = GateServerManager()
            with pytest.raises(SystemExit, match="no gate port is set"):
                mgr.install_systemd_units(transport="tcp")

    def test_gate_socket_install_without_port_is_fine(self) -> None:
        """Socket transport never reads the port — ``None`` must pass the guard."""
        from terok_sandbox.vault.token_broker import main as _unused_main  # noqa: F401

        mock_cfg = unittest.mock.MagicMock(spec=SandboxConfig)
        mock_cfg.gate_port = None
        with unittest.mock.patch.object(
            GateServerManager, "__init__", lambda self, cfg=None: setattr(self, "_cfg", mock_cfg)
        ):
            mgr = GateServerManager()
            # The full install body would touch systemd; we only care
            # that the pre-flight guard in the new fail-loud branch does
            # not fire for socket-transport installs.  Run just enough
            # of the method to exercise the guard, then short-circuit.
            with unittest.mock.patch("shutil.which", return_value=None):
                with pytest.raises(SystemExit, match="terok-gate"):
                    # shutil.which returning None triggers the *next*
                    # SystemExit (missing binary) — reaching that proves
                    # the port guard did not fire.
                    mgr.install_systemd_units(transport="socket")

    def test_vault_tcp_install_without_port_raises(self) -> None:
        """VaultManager rejects tcp install with no token_broker_port."""
        from terok_sandbox.vault.lifecycle import VaultManager

        mock_cfg = unittest.mock.MagicMock(spec=SandboxConfig)
        mock_cfg.token_broker_port = None
        mock_cfg.ssh_signer_port = 18732
        with unittest.mock.patch.object(
            VaultManager,
            "__init__",
            lambda self, cfg=None: setattr(self, "_cfg", mock_cfg),
        ):
            mgr = VaultManager()
            with pytest.raises(SystemExit, match="no port is set"):
                mgr.install_systemd_units(transport="tcp")

    def test_vault_tcp_install_without_ssh_signer_port_raises(self) -> None:
        """Same guard fires when only ssh_signer_port is unset."""
        from terok_sandbox.vault.lifecycle import VaultManager

        mock_cfg = unittest.mock.MagicMock(spec=SandboxConfig)
        mock_cfg.token_broker_port = 18731
        mock_cfg.ssh_signer_port = None
        with unittest.mock.patch.object(
            VaultManager,
            "__init__",
            lambda self, cfg=None: setattr(self, "_cfg", mock_cfg),
        ):
            mgr = VaultManager()
            with pytest.raises(SystemExit, match="no port is set"):
                mgr.install_systemd_units(transport="tcp")


class TestInstallSystemdTransportResolution:
    """Verify the top-level wrappers default transport from ``services.mode``."""

    def test_gate_wrapper_resolves_transport_from_config(self) -> None:
        """``install_systemd_units(transport=None)`` reads services.mode."""
        from terok_sandbox import install_systemd_units

        with (
            unittest.mock.patch("terok_sandbox.config.services_mode", return_value="socket"),
            unittest.mock.patch.object(GateServerManager, "install_systemd_units") as mock_install,
            unittest.mock.patch.object(GateServerManager, "__init__", lambda self, cfg=None: None),
        ):
            install_systemd_units()
        mock_install.assert_called_once_with(transport="socket")

    def test_gate_wrapper_honours_explicit_transport(self) -> None:
        """An explicit ``transport=`` argument bypasses the config read."""
        from terok_sandbox import install_systemd_units

        with (
            unittest.mock.patch(
                "terok_sandbox.config.services_mode", return_value="socket"
            ) as mock_mode,
            unittest.mock.patch.object(GateServerManager, "install_systemd_units") as mock_install,
            unittest.mock.patch.object(GateServerManager, "__init__", lambda self, cfg=None: None),
        ):
            install_systemd_units(transport="tcp")
        mock_mode.assert_not_called()
        mock_install.assert_called_once_with(transport="tcp")

    def test_vault_wrapper_resolves_transport_from_config(self) -> None:
        """``install_vault_systemd(transport=None)`` reads services.mode."""
        from terok_sandbox import install_vault_systemd
        from terok_sandbox.vault.lifecycle import VaultManager

        with (
            unittest.mock.patch("terok_sandbox.config.services_mode", return_value="socket"),
            unittest.mock.patch.object(VaultManager, "install_systemd_units") as mock_install,
            unittest.mock.patch.object(VaultManager, "__init__", lambda self, cfg=None: None),
        ):
            install_vault_systemd()
        mock_install.assert_called_once_with(transport="socket")

    def test_vault_wrapper_honours_explicit_transport(self) -> None:
        """An explicit ``transport=`` argument bypasses the config read."""
        from terok_sandbox import install_vault_systemd
        from terok_sandbox.vault.lifecycle import VaultManager

        with (
            unittest.mock.patch(
                "terok_sandbox.config.services_mode", return_value="socket"
            ) as mock_mode,
            unittest.mock.patch.object(VaultManager, "install_systemd_units") as mock_install,
            unittest.mock.patch.object(VaultManager, "__init__", lambda self, cfg=None: None),
        ):
            install_vault_systemd(transport="tcp")
        mock_mode.assert_not_called()
        mock_install.assert_called_once_with(transport="tcp")


class TestGateOrphanUnitSweep:
    """Verify ``GateServerManager._sweep_orphan_units`` — marker-based cleanup."""

    @staticmethod
    def _with_unit_dir(unit_dir):
        return unittest.mock.patch.object(
            GateServerManager, "_systemd_unit_dir", return_value=unit_dir
        )

    def test_legacy_marked_file_removed(self, tmp_path):
        """A terok-gate-* file with our marker but not current name is swept."""
        unit_dir = tmp_path / "systemd-units"
        unit_dir.mkdir()
        legacy = unit_dir / "terok-gate-legacy.service"
        legacy.write_text("# terok-gate-version: 3\n[Service]\n")
        with self._with_unit_dir(unit_dir), unittest.mock.patch("subprocess.run"):
            GateServerManager()._sweep_orphan_units()
        assert not legacy.exists()

    def test_current_name_skipped(self, tmp_path):
        """Current-version filenames stay for the main removal pass to handle."""
        unit_dir = tmp_path / "systemd-units"
        unit_dir.mkdir()
        current = unit_dir / "terok-gate.socket"
        current.write_text("# terok-gate-version: 7\n[Socket]\n")
        with self._with_unit_dir(unit_dir), unittest.mock.patch("subprocess.run"):
            GateServerManager()._sweep_orphan_units()
        assert current.exists()

    def test_foreign_file_preserved(self, tmp_path):
        """A user-authored file matching the glob but lacking the marker is not touched."""
        unit_dir = tmp_path / "systemd-units"
        unit_dir.mkdir()
        foreign = unit_dir / "terok-gate-custom.service"
        foreign.write_text("[Service]\nExecStart=/bin/true\n")
        with self._with_unit_dir(unit_dir), unittest.mock.patch("subprocess.run"):
            GateServerManager()._sweep_orphan_units()
        assert foreign.exists()

    def test_non_matching_glob_preserved(self, tmp_path):
        """Files outside our glob are never read, regardless of content."""
        unit_dir = tmp_path / "systemd-units"
        unit_dir.mkdir()
        other = unit_dir / "unrelated.service"
        other.write_text("# terok-gate-version: 99\n")
        with self._with_unit_dir(unit_dir), unittest.mock.patch("subprocess.run"):
            GateServerManager()._sweep_orphan_units()
        assert other.exists()

    def test_disable_invoked_before_unlink(self, tmp_path):
        """Each removed legacy unit is systemctl-disabled first (best-effort)."""
        unit_dir = tmp_path / "systemd-units"
        unit_dir.mkdir()
        legacy = unit_dir / "terok-gate-legacy.service"
        legacy.write_text("# terok-gate-version: 3\n[Service]\n")
        with (
            self._with_unit_dir(unit_dir),
            unittest.mock.patch("subprocess.run") as mock_run,
        ):
            GateServerManager()._sweep_orphan_units()
        disable_calls = [c.args[0] for c in mock_run.call_args_list if "disable" in c.args[0]]
        assert any("terok-gate-legacy.service" in cmd for cmd in disable_calls)

    def test_missing_unit_dir_is_a_noop(self, tmp_path):
        """Running on a host with no user systemd dir must not error."""
        unit_dir = tmp_path / "does-not-exist"
        with self._with_unit_dir(unit_dir), unittest.mock.patch("subprocess.run"):
            GateServerManager()._sweep_orphan_units()  # must not raise
