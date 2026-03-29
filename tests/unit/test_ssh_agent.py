# SPDX-FileCopyrightText: 2026 Jiri Vyskocil
# SPDX-License-Identifier: Apache-2.0

"""Tests for the SSH agent proxy — wire format, handshake, identity, signing."""

from __future__ import annotations

import asyncio
import base64
import json
import struct
from pathlib import Path

import pytest
from cryptography.hazmat.primitives.asymmetric.ed25519 import Ed25519PrivateKey
from cryptography.hazmat.primitives.serialization import (
    Encoding,
    NoEncryption,
    PrivateFormat,
    PublicFormat,
)

from terok_sandbox.credential_db import CredentialDB
from terok_sandbox.credential_proxy.ssh_agent import (
    SSH_AGENT_FAILURE,
    SSH_AGENT_IDENTITIES_ANSWER,
    SSH_AGENT_SIGN_RESPONSE,
    SSH_AGENTC_REQUEST_IDENTITIES,
    SSH_AGENTC_SIGN_REQUEST,
    _KeyTable,
    _pack_string,
    _sign,
    _unpack_string,
    start_ssh_agent_server,
)

# ── Fixtures ────────────────────────────────────────────────────────────


@pytest.fixture()
def ed25519_keypair(tmp_path: Path) -> tuple[Path, Path, bytes]:
    """Generate an ed25519 keypair and return (priv_path, pub_path, pub_blob)."""
    key = Ed25519PrivateKey.generate()
    priv_pem = key.private_bytes(Encoding.PEM, PrivateFormat.OpenSSH, NoEncryption())
    pub_raw = key.public_key().public_bytes(Encoding.OpenSSH, PublicFormat.OpenSSH)

    priv_path = tmp_path / "id_ed25519_test"
    pub_path = tmp_path / "id_ed25519_test.pub"
    priv_path.write_bytes(priv_pem)
    pub_path.write_text(f"{pub_raw.decode()} test-comment\n")

    # The wire-format blob is the base64 middle field of the .pub line
    pub_blob = base64.b64decode(pub_raw.decode().split()[1])
    return priv_path, pub_path, pub_blob


@pytest.fixture()
def ssh_agent_env(tmp_path: Path, ed25519_keypair: tuple[Path, Path, bytes]):
    """Set up DB + keys JSON + keypair.  Returns (db_path, keys_file, token, pub_blob)."""
    priv_path, pub_path, pub_blob = ed25519_keypair

    # Create DB with a phantom token
    db = CredentialDB(tmp_path / "test.db")
    token = db.create_proxy_token("test-project", "task-1", "test-project", "ssh")
    db.close()

    # Write ssh-keys.json
    keys_file = tmp_path / "ssh-keys.json"
    keys_file.write_text(
        json.dumps(
            {
                "test-project": {
                    "private_key": str(priv_path),
                    "public_key": str(pub_path),
                }
            }
        )
    )

    return str(tmp_path / "test.db"), str(keys_file), token, pub_blob


# ── Wire format tests ───────────────────────────────────────────────────


class TestWireFormat:
    """Verify SSH wire-format encoding/decoding helpers."""

    def test_pack_string(self) -> None:
        """pack_string produces [4-byte len][data]."""
        result = _pack_string(b"hello")
        assert result == b"\x00\x00\x00\x05hello"

    def test_pack_empty_string(self) -> None:
        """pack_string handles empty data."""
        assert _pack_string(b"") == b"\x00\x00\x00\x00"

    def test_unpack_string_roundtrip(self) -> None:
        """unpack_string reads back what pack_string wrote."""
        packed = _pack_string(b"test-data")
        data, offset = _unpack_string(memoryview(packed), 0)
        assert data == b"test-data"
        assert offset == len(packed)

    def test_unpack_multiple_strings(self) -> None:
        """unpack_string handles sequential strings."""
        buf = _pack_string(b"first") + _pack_string(b"second")
        mv = memoryview(buf)
        s1, off = _unpack_string(mv, 0)
        s2, off2 = _unpack_string(mv, off)
        assert s1 == b"first"
        assert s2 == b"second"
        assert off2 == len(buf)


# ── Key table ───────────────────────────────────────────────────────────


class TestKeyTable:
    """Verify project-keyed SSH key path lookup."""

    def test_get_known_project(self, tmp_path: Path) -> None:
        """Known project returns key paths."""
        kf = tmp_path / "keys.json"
        kf.write_text(json.dumps({"proj": {"private_key": "/a", "public_key": "/b"}}))
        kt = _KeyTable(str(kf))
        assert kt.get("proj") == {"private_key": "/a", "public_key": "/b"}

    def test_get_unknown_project(self, tmp_path: Path) -> None:
        """Unknown project returns None."""
        kf = tmp_path / "keys.json"
        kf.write_text("{}")
        assert _KeyTable(str(kf)).get("nope") is None


# ── Signing ─────────────────────────────────────────────────────────────


class TestSign:
    """Verify SSH signature format."""

    def test_ed25519_signature_format(self, ed25519_keypair: tuple[Path, Path, bytes]) -> None:
        """Ed25519 signature blob contains correct algorithm name."""
        priv_path, _, _ = ed25519_keypair
        key = Ed25519PrivateKey.generate()
        sig_blob = _sign(key, b"test-data", 0)
        algo, off = _unpack_string(memoryview(sig_blob), 0)
        assert algo == b"ssh-ed25519"
        raw_sig, _ = _unpack_string(memoryview(sig_blob), off)
        assert len(raw_sig) == 64  # ed25519 signatures are 64 bytes

    def test_ed25519_signature_verifies(self) -> None:
        """Ed25519 signature is valid when verified with the public key."""
        key = Ed25519PrivateKey.generate()
        data = b"sign-this-data"
        sig_blob = _sign(key, data, 0)
        # Extract raw signature from blob
        _, off = _unpack_string(memoryview(sig_blob), 0)
        raw_sig, _ = _unpack_string(memoryview(sig_blob), off)
        # Verify — raises InvalidSignature on failure
        key.public_key().verify(raw_sig, data)


# ── Full round-trip tests ───────────────────────────────────────────────


def _build_handshake(token: str) -> bytes:
    """Build a phantom-token handshake prefix."""
    token_bytes = token.encode("utf-8")
    return struct.pack(">I", len(token_bytes)) + token_bytes


def _build_msg(msg_type: int, payload: bytes = b"") -> bytes:
    """Build an SSH agent protocol message."""
    body = bytes([msg_type]) + payload
    return struct.pack(">I", len(body)) + body


async def _read_response(reader: asyncio.StreamReader) -> tuple[int, bytes]:
    """Read one SSH agent response message."""
    raw_len = await reader.readexactly(4)
    (msg_len,) = struct.unpack(">I", raw_len)
    body = await reader.readexactly(msg_len)
    return body[0], body[1:]


@pytest.mark.asyncio
class TestSSHAgentRoundTrip:
    """Full TCP round-trip tests using a real asyncio server."""

    async def test_identity_listing(self, ssh_agent_env) -> None:
        """REQUEST_IDENTITIES returns the project's public key."""
        db_path, keys_file, token, pub_blob = ssh_agent_env

        server = await start_ssh_agent_server(db_path, keys_file, "127.0.0.1", 0)
        port = server.sockets[0].getsockname()[1]
        try:
            reader, writer = await asyncio.open_connection("127.0.0.1", port)
            writer.write(_build_handshake(token))
            writer.write(_build_msg(SSH_AGENTC_REQUEST_IDENTITIES))
            await writer.drain()

            msg_type, payload = await _read_response(reader)
            assert msg_type == SSH_AGENT_IDENTITIES_ANSWER
            (nkeys,) = struct.unpack_from(">I", payload, 0)
            assert nkeys == 1
            key_blob, off = _unpack_string(memoryview(payload), 4)
            assert key_blob == pub_blob
            comment, _ = _unpack_string(memoryview(payload), off)
            assert comment == b"test-comment"

            writer.close()
            await writer.wait_closed()
        finally:
            server.close()
            await server.wait_closed()

    async def test_sign_request(self, ssh_agent_env) -> None:
        """SIGN_REQUEST returns a valid signature."""
        db_path, keys_file, token, pub_blob = ssh_agent_env

        server = await start_ssh_agent_server(db_path, keys_file, "127.0.0.1", 0)
        port = server.sockets[0].getsockname()[1]
        try:
            reader, writer = await asyncio.open_connection("127.0.0.1", port)
            writer.write(_build_handshake(token))

            # Build sign request: [string: key_blob][string: data][uint32: flags]
            sign_data = b"data-to-sign"
            sign_payload = _pack_string(pub_blob) + _pack_string(sign_data) + struct.pack(">I", 0)
            writer.write(_build_msg(SSH_AGENTC_SIGN_REQUEST, sign_payload))
            await writer.drain()

            msg_type, payload = await _read_response(reader)
            assert msg_type == SSH_AGENT_SIGN_RESPONSE

            # Unwrap: response is [string: signature_blob]
            sig_blob, _ = _unpack_string(memoryview(payload), 0)
            algo, off = _unpack_string(memoryview(sig_blob), 0)
            assert algo == b"ssh-ed25519"
            raw_sig, _ = _unpack_string(memoryview(sig_blob), off)
            assert len(raw_sig) == 64

            writer.close()
            await writer.wait_closed()
        finally:
            server.close()
            await server.wait_closed()

    async def test_invalid_token_closes_connection(self, ssh_agent_env) -> None:
        """Invalid phantom token closes the connection without response."""
        db_path, keys_file, _token, _pub_blob = ssh_agent_env

        server = await start_ssh_agent_server(db_path, keys_file, "127.0.0.1", 0)
        port = server.sockets[0].getsockname()[1]
        try:
            reader, writer = await asyncio.open_connection("127.0.0.1", port)
            writer.write(_build_handshake("invalid-token"))
            writer.write(_build_msg(SSH_AGENTC_REQUEST_IDENTITIES))
            await writer.drain()

            # Server should close the connection
            data = await reader.read(1024)
            assert data == b""

            writer.close()
            await writer.wait_closed()
        finally:
            server.close()
            await server.wait_closed()

    async def test_unknown_message_returns_failure(self, ssh_agent_env) -> None:
        """Unknown message type returns SSH_AGENT_FAILURE."""
        db_path, keys_file, token, _pub_blob = ssh_agent_env

        server = await start_ssh_agent_server(db_path, keys_file, "127.0.0.1", 0)
        port = server.sockets[0].getsockname()[1]
        try:
            reader, writer = await asyncio.open_connection("127.0.0.1", port)
            writer.write(_build_handshake(token))
            writer.write(_build_msg(99))  # unknown type
            await writer.drain()

            msg_type, _payload = await _read_response(reader)
            assert msg_type == SSH_AGENT_FAILURE

            writer.close()
            await writer.wait_closed()
        finally:
            server.close()
            await server.wait_closed()

    async def test_sign_wrong_key_returns_failure(self, ssh_agent_env) -> None:
        """Sign request for an unknown key blob returns SSH_AGENT_FAILURE."""
        db_path, keys_file, token, _pub_blob = ssh_agent_env

        server = await start_ssh_agent_server(db_path, keys_file, "127.0.0.1", 0)
        port = server.sockets[0].getsockname()[1]
        try:
            reader, writer = await asyncio.open_connection("127.0.0.1", port)
            writer.write(_build_handshake(token))

            fake_blob = b"\x00" * 32
            sign_payload = _pack_string(fake_blob) + _pack_string(b"data") + struct.pack(">I", 0)
            writer.write(_build_msg(SSH_AGENTC_SIGN_REQUEST, sign_payload))
            await writer.drain()

            msg_type, _payload = await _read_response(reader)
            assert msg_type == SSH_AGENT_FAILURE

            writer.close()
            await writer.wait_closed()
        finally:
            server.close()
            await server.wait_closed()
