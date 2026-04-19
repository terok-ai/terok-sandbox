# SPDX-FileCopyrightText: 2026 Jiri Vyskocil
# SPDX-License-Identifier: Apache-2.0

"""Tests for the host-local SSH signer — no handshake, scope fixed at bind."""

from __future__ import annotations

import asyncio
import os
import stat
import struct
from pathlib import Path

import pytest

from terok_sandbox.credentials.db import CredentialDB
from terok_sandbox.credentials.ssh_keypair import generate_keypair
from terok_sandbox.vault.ssh_signer import (
    SSH_AGENT_IDENTITIES_ANSWER,
    SSH_AGENTC_REQUEST_IDENTITIES,
    _unpack_string,
    start_ssh_signer_local,
)

_READ_TIMEOUT = 5.0


async def _read_response(reader: asyncio.StreamReader) -> tuple[int, bytes]:
    """Read one SSH signer response message."""
    raw_len = await asyncio.wait_for(reader.readexactly(4), timeout=_READ_TIMEOUT)
    (msg_len,) = struct.unpack(">I", raw_len)
    body = await asyncio.wait_for(reader.readexactly(msg_len), timeout=_READ_TIMEOUT)
    return body[0], body[1:]


def _build_msg(msg_type: int, payload: bytes = b"") -> bytes:
    """Build an SSH signer protocol message."""
    body = bytes([msg_type]) + payload
    return struct.pack(">I", len(body)) + body


@pytest.mark.asyncio
class TestLocalSigner:
    """Verify the scope-bound local listener serves identities without a handshake."""

    async def test_returns_scope_identities_without_handshake(self, tmp_path: Path) -> None:
        """A direct connection lists the scope's keys — no token required."""
        db_path = tmp_path / "vault.db"
        db = CredentialDB(db_path)
        kp = generate_keypair("ed25519", comment="tk-main:proj")
        key_id = db.store_ssh_key(
            key_type=kp.key_type,
            private_pem=kp.private_pem,
            public_blob=kp.public_blob,
            comment=kp.comment,
            fingerprint=kp.fingerprint,
        )
        db.assign_ssh_key("proj", key_id)
        db.close()

        sock_path = tmp_path / "ssh-agent-local-proj.sock"
        server = await start_ssh_signer_local(
            scope="proj",
            socket_path=sock_path,
            db_path=str(db_path),
        )
        try:
            mode = stat.S_IMODE(os.lstat(sock_path).st_mode)
            assert mode == 0o600

            reader, writer = await asyncio.open_unix_connection(str(sock_path))
            writer.write(_build_msg(SSH_AGENTC_REQUEST_IDENTITIES))
            await writer.drain()

            msg_type, payload = await _read_response(reader)
            assert msg_type == SSH_AGENT_IDENTITIES_ANSWER
            (nkeys,) = struct.unpack_from(">I", payload, 0)
            assert nkeys == 1
            blob, off = _unpack_string(memoryview(payload), 4)
            assert blob == kp.public_blob

            writer.close()
            await writer.wait_closed()
        finally:
            server.close()
            await server.wait_closed()

    async def test_scope_without_keys_closes_connection(self, tmp_path: Path) -> None:
        """Binding to a scope with no keys is tolerated; connections return empty."""
        db_path = tmp_path / "vault.db"
        CredentialDB(db_path).close()

        sock_path = tmp_path / "ssh-agent-local-ghost.sock"
        server = await start_ssh_signer_local(
            scope="ghost",
            socket_path=sock_path,
            db_path=str(db_path),
        )
        try:
            reader, writer = await asyncio.open_unix_connection(str(sock_path))
            writer.write(_build_msg(SSH_AGENTC_REQUEST_IDENTITIES))
            await writer.drain()

            # Server terminates the connection — either clean EOF or ECONNRESET
            # depending on timing.  Both are acceptable for the empty-scope case.
            try:
                data = await reader.read(1024)
            except ConnectionResetError:
                data = b""
            assert data == b""

            writer.close()
            try:
                await writer.wait_closed()
            except ConnectionResetError:
                pass
        finally:
            server.close()
            await server.wait_closed()
