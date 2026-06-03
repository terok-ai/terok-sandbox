# SPDX-FileCopyrightText: 2026 Jiri Vyskocil
# SPDX-License-Identifier: Apache-2.0

"""Tests for Unix socket transport support across services.

Covers: probe_unix_socket utility, SandboxConfig socket path properties,
the gate server Unix socket factory, and SSH signer Unix socket mode.
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
from terok_sandbox.vault.ssh.signer import (
    SSH_AGENT_IDENTITIES_ANSWER,
    SSH_AGENTC_REQUEST_IDENTITIES,
    _unpack_string,
    start_ssh_signer,
)
from terok_sandbox.vault.store.db import CredentialDB
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
        from terok_sandbox.vault.ssh.keypair import generate_keypair

        kp = generate_keypair("ed25519", comment="test-socket")
        db = CredentialDB(tmp_path / "test.db", passphrase="test")
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

        db = CredentialDB(tmp_path / "test.db", passphrase="test")
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

        db = CredentialDB(tmp_path / "test.db", passphrase="test")
        db.close()

        server = await start_ssh_signer(str(tmp_path / "test.db"), socket_path=str(sock_path))
        try:
            assert sock_path.exists()
        finally:
            server.close()
            await server.wait_closed()

    async def test_raises_without_transport(self, tmp_path: Path) -> None:
        """ValueError when neither socket_path nor host+port is given."""
        db = CredentialDB(tmp_path / "test.db", passphrase="test")
        db.close()

        with pytest.raises(ValueError, match="Either socket_path or host\\+port"):
            await start_ssh_signer(str(tmp_path / "test.db"))


# The gate's foreground / CLI / systemd-install tests that lived here
# covered the retired host gate daemon.  The gate now lives inside the
# per-container supervisor (see ``test_gate_server.py`` for its
# start→serve→stop coverage), so those host-daemon tests have no live
# code path to exercise.


class TestServicesModeSSOT:
    """Verify ``services.mode`` flows through ``SandboxConfig`` as the only path.

    Replaces the old wrapper-resolution tests.  The refactor removed the
    ``install_systemd_units`` / ``install_vault_systemd`` module-level
    wrappers and their ``transport=`` kwarg; transport is now a
    ``SandboxConfig`` field resolved once at construction.  These tests
    verify that the control flow passes through the config layer and
    nowhere else.
    """

    def test_services_mode_default_factory_reads_config(self) -> None:
        """``SandboxConfig()`` resolves ``services_mode`` through the pydantic schema."""
        from terok_sandbox.config import SandboxConfig as _SandboxConfig

        with unittest.mock.patch(
            "terok_sandbox.config.services_mode", return_value="tcp"
        ) as mock_mode:
            cfg = _SandboxConfig()
        # The factory is called during construction; the ported value
        # lives on the instance from then on.
        mock_mode.assert_called_once()
        assert cfg.services_mode == "tcp"

    def test_explicit_services_mode_overrides_factory(self) -> None:
        """Passing ``services_mode=`` to the constructor skips the factory."""
        from terok_sandbox.config import SandboxConfig as _SandboxConfig

        with unittest.mock.patch(
            "terok_sandbox.config.services_mode", return_value="tcp"
        ) as mock_mode:
            cfg = _SandboxConfig(services_mode="socket")
        mock_mode.assert_not_called()
        assert cfg.services_mode == "socket"
