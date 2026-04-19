# SPDX-FileCopyrightText: 2026 Jiri Vyskocil
# SPDX-License-Identifier: Apache-2.0

"""Tests for the SSH signer — wire format, token handshake, identity, signing.

Exercises the container-facing listener (``start_ssh_signer``) that gates
connections behind a phantom token.  Key material is seeded straight into
the vault DB via :class:`CredentialDB` — no sidecar JSON.
"""

from __future__ import annotations

import asyncio
import struct
from pathlib import Path

import pytest
from cryptography.hazmat.primitives.asymmetric.ed25519 import Ed25519PrivateKey

from terok_sandbox.credentials.db import CredentialDB
from terok_sandbox.credentials.ssh_keypair import generate_keypair
from terok_sandbox.vault.ssh_signer import (
    SSH_AGENT_FAILURE,
    SSH_AGENT_IDENTITIES_ANSWER,
    SSH_AGENT_SIGN_RESPONSE,
    SSH_AGENTC_REQUEST_IDENTITIES,
    SSH_AGENTC_SIGN_REQUEST,
    _pack_string,
    _sign,
    _unpack_string,
    start_ssh_signer,
)

# ── Fixtures ────────────────────────────────────────────────────────────


def _seed_key(db: CredentialDB, scope: str, comment: str = "test-comment") -> bytes:
    """Generate an ed25519 keypair, store it in *db*, and assign it to *scope*."""
    kp = generate_keypair("ed25519", comment=comment)
    key_id = db.store_ssh_key(
        key_type=kp.key_type,
        private_pem=kp.private_pem,
        public_blob=kp.public_blob,
        comment=kp.comment,
        fingerprint=kp.fingerprint,
    )
    db.assign_ssh_key(scope, key_id)
    return kp.public_blob


@pytest.fixture()
def signer_env(tmp_path: Path):
    """Seed one scope+key and return (db_path, token, pub_blob)."""
    db_path = tmp_path / "vault.db"
    db = CredentialDB(db_path)
    pub_blob = _seed_key(db, "proj")
    token = db.create_token("proj", "task-1", "proj", "ssh")
    db.close()
    return str(db_path), token, pub_blob


# ── Wire format tests ───────────────────────────────────────────────────


class TestWireFormat:
    """Verify SSH wire-format encoding/decoding helpers."""

    def test_pack_string(self) -> None:
        """pack_string produces [4-byte len][data]."""
        assert _pack_string(b"hello") == b"\x00\x00\x00\x05hello"

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


class TestSign:
    """Verify SSH signature blob format."""

    def test_ed25519_signature_format(self) -> None:
        """Ed25519 sig blob carries the algorithm name + 64-byte raw signature."""
        key = Ed25519PrivateKey.generate()
        sig_blob = _sign(key, b"test-data", 0)
        algo, off = _unpack_string(memoryview(sig_blob), 0)
        assert algo == b"ssh-ed25519"
        raw_sig, _ = _unpack_string(memoryview(sig_blob), off)
        assert len(raw_sig) == 64

    def test_ed25519_signature_verifies(self) -> None:
        """Signature validates against the public key."""
        key = Ed25519PrivateKey.generate()
        data = b"sign-this-data"
        sig_blob = _sign(key, data, 0)
        _, off = _unpack_string(memoryview(sig_blob), 0)
        raw_sig, _ = _unpack_string(memoryview(sig_blob), off)
        key.public_key().verify(raw_sig, data)

    def test_rsa_default_uses_sha256(self) -> None:
        """RSA sign with flags=0 emits rsa-sha2-256 (widest compatible)."""
        from cryptography.hazmat.primitives.asymmetric.rsa import generate_private_key

        key = generate_private_key(public_exponent=65537, key_size=2048)
        sig_blob = _sign(key, b"payload", 0)
        algo, _ = _unpack_string(memoryview(sig_blob), 0)
        assert algo == b"rsa-sha2-256"

    def test_rsa_flag_selects_sha512(self) -> None:
        """RSA sign with SSH_AGENT_RSA_SHA2_512 flag emits rsa-sha2-512."""
        from cryptography.hazmat.primitives.asymmetric.rsa import generate_private_key

        from terok_sandbox.vault.ssh_signer import SSH_AGENT_RSA_SHA2_512

        key = generate_private_key(public_exponent=65537, key_size=2048)
        sig_blob = _sign(key, b"payload", SSH_AGENT_RSA_SHA2_512)
        algo, _ = _unpack_string(memoryview(sig_blob), 0)
        assert algo == b"rsa-sha2-512"


class TestUnpackStringGuards:
    """Verify ``_unpack_string`` rejects under-sized buffers with a clear message."""

    def test_header_shorter_than_four_bytes(self) -> None:
        """A 3-byte buffer can't even hold a length prefix."""
        with pytest.raises(ValueError, match="Buffer too short"):
            _unpack_string(memoryview(b"abc"), 0)

    def test_declared_length_exceeds_buffer(self) -> None:
        """Length prefix larger than remaining bytes is a protocol error."""
        # Declares 1000-byte string, only 4 bytes follow.
        bad = struct.pack(">I", 1000) + b"\x00" * 4
        with pytest.raises(ValueError, match="exceeds buffer"):
            _unpack_string(memoryview(bad), 0)


class TestDecodeRecord:
    """Verify ``_decode_record`` rejects non-RSA/ed25519 keys."""

    def test_unsupported_key_type_is_rejected(self) -> None:
        """A DSA key round-trips through decode as an explicit ValueError."""
        from cryptography.hazmat.primitives.asymmetric.ec import (
            SECP256R1,
            generate_private_key,
        )
        from cryptography.hazmat.primitives.serialization import (
            Encoding,
            NoEncryption,
            PrivateFormat,
        )

        from terok_sandbox.credentials.db import SSHKeyRecord
        from terok_sandbox.vault.ssh_signer import _decode_record

        ec_key = generate_private_key(SECP256R1())
        pem = ec_key.private_bytes(
            encoding=Encoding.PEM,
            format=PrivateFormat.PKCS8,
            encryption_algorithm=NoEncryption(),
        )
        rec = SSHKeyRecord(
            id=1,
            key_type="ecdsa",  # not what the cache accepts
            private_pem=pem,
            public_blob=b"x",
            comment="c",
            fingerprint="00" * 32,
        )
        with pytest.raises(ValueError, match="Unsupported key type"):
            _decode_record(rec)


class TestDBKeyCache:
    """Verify the version-counter cache behaviour."""

    def test_second_call_same_version_hits_cache(self, tmp_path: Path) -> None:
        """When ``ssh_keys_version`` is unchanged, the cache returns the prior slot."""
        from terok_sandbox.vault.ssh_signer import _DBKeyCache

        db_path = tmp_path / "vault.db"
        db = CredentialDB(db_path)
        _seed_key(db, "proj")

        cache = _DBKeyCache(db)
        first = cache.get("proj")
        second = cache.get("proj")
        assert second is first  # same list object — cache hit
        db.close()

    def test_version_bump_reloads(self, tmp_path: Path) -> None:
        """A new ``assign_ssh_key`` bumps the version and invalidates the cached slot."""
        from terok_sandbox.vault.ssh_signer import _DBKeyCache

        db_path = tmp_path / "vault.db"
        db = CredentialDB(db_path)
        _seed_key(db, "proj", comment="first")

        cache = _DBKeyCache(db)
        before = cache.get("proj")
        _seed_key(db, "proj", comment="second")  # bumps ssh_keys_version
        after = cache.get("proj")
        assert after is not before
        assert len(after) == 2
        db.close()


# ── Round-trip (TCP + token handshake) ──────────────────────────────────


def _build_handshake(token: str) -> bytes:
    """Build a token handshake prefix."""
    token_bytes = token.encode("utf-8")
    return struct.pack(">I", len(token_bytes)) + token_bytes


def _build_msg(msg_type: int, payload: bytes = b"") -> bytes:
    """Build an SSH signer protocol message."""
    body = bytes([msg_type]) + payload
    return struct.pack(">I", len(body)) + body


_READ_TIMEOUT = 5.0


async def _read_response(reader: asyncio.StreamReader) -> tuple[int, bytes]:
    """Read one SSH signer response message."""
    raw_len = await asyncio.wait_for(reader.readexactly(4), timeout=_READ_TIMEOUT)
    (msg_len,) = struct.unpack(">I", raw_len)
    body = await asyncio.wait_for(reader.readexactly(msg_len), timeout=_READ_TIMEOUT)
    return body[0], body[1:]


@pytest.mark.asyncio
class TestSSHSignerRoundTrip:
    """Full TCP round-trip tests using a real asyncio server."""

    async def test_identity_listing(self, signer_env) -> None:
        """REQUEST_IDENTITIES returns the scope's public key."""
        db_path, token, pub_blob = signer_env
        server = await start_ssh_signer(db_path, host="127.0.0.1", port=0)
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

    async def test_sign_request(self, signer_env) -> None:
        """SIGN_REQUEST returns a valid ed25519 signature blob."""
        db_path, token, pub_blob = signer_env
        server = await start_ssh_signer(db_path, host="127.0.0.1", port=0)
        port = server.sockets[0].getsockname()[1]
        try:
            reader, writer = await asyncio.open_connection("127.0.0.1", port)
            writer.write(_build_handshake(token))

            sign_payload = _pack_string(pub_blob) + _pack_string(b"data") + struct.pack(">I", 0)
            writer.write(_build_msg(SSH_AGENTC_SIGN_REQUEST, sign_payload))
            await writer.drain()

            msg_type, payload = await _read_response(reader)
            assert msg_type == SSH_AGENT_SIGN_RESPONSE
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

    async def test_multi_key_listing(self, tmp_path: Path) -> None:
        """A scope with two assigned keys exposes both on REQUEST_IDENTITIES."""
        db_path = tmp_path / "vault.db"
        db = CredentialDB(db_path)
        blob1 = _seed_key(db, "proj", comment="first")
        blob2 = _seed_key(db, "proj", comment="second")
        token = db.create_token("proj", "task-1", "proj", "ssh")
        db.close()

        server = await start_ssh_signer(str(db_path), host="127.0.0.1", port=0)
        port = server.sockets[0].getsockname()[1]
        try:
            reader, writer = await asyncio.open_connection("127.0.0.1", port)
            writer.write(_build_handshake(token))
            writer.write(_build_msg(SSH_AGENTC_REQUEST_IDENTITIES))
            await writer.drain()

            msg_type, payload = await _read_response(reader)
            assert msg_type == SSH_AGENT_IDENTITIES_ANSWER
            (nkeys,) = struct.unpack_from(">I", payload, 0)
            assert nkeys == 2

            mv = memoryview(payload)
            off = 4
            returned = []
            for _ in range(nkeys):
                blob, off = _unpack_string(mv, off)
                _comment, off = _unpack_string(mv, off)
                returned.append(blob)
            assert set(returned) == {blob1, blob2}

            writer.close()
            await writer.wait_closed()
        finally:
            server.close()
            await server.wait_closed()

    async def test_invalid_token_closes_connection(self, signer_env) -> None:
        """Invalid token closes the connection without response."""
        db_path, _token, _pub_blob = signer_env
        server = await start_ssh_signer(db_path, host="127.0.0.1", port=0)
        port = server.sockets[0].getsockname()[1]
        try:
            reader, writer = await asyncio.open_connection("127.0.0.1", port)
            writer.write(_build_handshake("invalid-token"))
            writer.write(_build_msg(SSH_AGENTC_REQUEST_IDENTITIES))
            await writer.drain()

            data = await reader.read(1024)
            assert data == b""

            writer.close()
            await writer.wait_closed()
        finally:
            server.close()
            await server.wait_closed()

    async def test_non_ssh_provider_token_rejected(self, tmp_path: Path) -> None:
        """A token whose provider != 'ssh' is rejected."""
        db_path = tmp_path / "vault.db"
        db = CredentialDB(db_path)
        _seed_key(db, "proj")
        api_token = db.create_token("proj", "task-1", "default", "claude")
        db.close()

        server = await start_ssh_signer(str(db_path), host="127.0.0.1", port=0)
        port = server.sockets[0].getsockname()[1]
        try:
            reader, writer = await asyncio.open_connection("127.0.0.1", port)
            writer.write(_build_handshake(api_token))
            writer.write(_build_msg(SSH_AGENTC_REQUEST_IDENTITIES))
            await writer.drain()

            data = await reader.read(1024)
            assert data == b""

            writer.close()
            await writer.wait_closed()
        finally:
            server.close()
            await server.wait_closed()

    async def test_unknown_message_returns_failure(self, signer_env) -> None:
        """Unknown message type returns SSH_AGENT_FAILURE."""
        db_path, token, _pub_blob = signer_env
        server = await start_ssh_signer(db_path, host="127.0.0.1", port=0)
        port = server.sockets[0].getsockname()[1]
        try:
            reader, writer = await asyncio.open_connection("127.0.0.1", port)
            writer.write(_build_handshake(token))
            writer.write(_build_msg(99))
            await writer.drain()

            msg_type, _payload = await _read_response(reader)
            assert msg_type == SSH_AGENT_FAILURE

            writer.close()
            await writer.wait_closed()
        finally:
            server.close()
            await server.wait_closed()

    async def test_sign_wrong_key_returns_failure(self, signer_env) -> None:
        """Sign request for an unknown key blob returns SSH_AGENT_FAILURE."""
        db_path, token, _pub_blob = signer_env
        server = await start_ssh_signer(db_path, host="127.0.0.1", port=0)
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

    async def test_missing_scope_closes_connection(self, tmp_path: Path) -> None:
        """Token for a scope with no assigned keys closes the connection."""
        db_path = tmp_path / "vault.db"
        db = CredentialDB(db_path)
        token = db.create_token("ghost", "task-1", "ghost", "ssh")
        db.close()

        server = await start_ssh_signer(str(db_path), host="127.0.0.1", port=0)
        port = server.sockets[0].getsockname()[1]
        try:
            reader, writer = await asyncio.open_connection("127.0.0.1", port)
            writer.write(_build_handshake(token))
            writer.write(_build_msg(SSH_AGENTC_REQUEST_IDENTITIES))
            await writer.drain()

            data = await reader.read(1024)
            assert data == b""

            writer.close()
            await writer.wait_closed()
        finally:
            server.close()
            await server.wait_closed()
