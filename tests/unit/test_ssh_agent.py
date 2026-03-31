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
    SSH_AGENT_RSA_SHA2_256,
    SSH_AGENT_RSA_SHA2_512,
    SSH_AGENT_SIGN_RESPONSE,
    SSH_AGENTC_REQUEST_IDENTITIES,
    SSH_AGENTC_SIGN_REQUEST,
    _KeyCache,
    _load_private_key,
    _load_public_key_blob,
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
        json.dumps({"test-project": [{"private_key": str(priv_path), "public_key": str(pub_path)}]})
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


class TestKeyCache:
    """Verify project-keyed SSH key cache."""

    def test_get_known_project(
        self, tmp_path: Path, ed25519_keypair: tuple[Path, Path, bytes]
    ) -> None:
        """Known project with valid keys returns list of resolved key material."""
        priv_path, pub_path, expected_blob = ed25519_keypair
        kf = tmp_path / "keys.json"
        kf.write_text(
            json.dumps({"proj": [{"private_key": str(priv_path), "public_key": str(pub_path)}]})
        )
        keys = _KeyCache(str(kf)).get("proj")
        assert keys is not None and len(keys) == 1
        private_key, pub_blob, comment = keys[0]
        assert isinstance(private_key, Ed25519PrivateKey)
        assert pub_blob == expected_blob
        assert comment == "test-comment"

    def test_get_multiple_keys(
        self, tmp_path: Path, ed25519_keypair: tuple[Path, Path, bytes]
    ) -> None:
        """Project with a list of key entries returns all resolved keys."""
        priv1, pub1, blob1 = ed25519_keypair
        # Generate second keypair
        key2 = Ed25519PrivateKey.generate()
        priv2 = tmp_path / "id2"
        pub2 = tmp_path / "id2.pub"
        priv2.write_bytes(key2.private_bytes(Encoding.PEM, PrivateFormat.OpenSSH, NoEncryption()))
        pub_raw2 = key2.public_key().public_bytes(Encoding.OpenSSH, PublicFormat.OpenSSH)
        pub2.write_text(f"{pub_raw2.decode()} key-two\n")

        kf = tmp_path / "keys.json"
        kf.write_text(
            json.dumps(
                {
                    "proj": [
                        {"private_key": str(priv1), "public_key": str(pub1)},
                        {"private_key": str(priv2), "public_key": str(pub2)},
                    ]
                }
            )
        )
        keys = _KeyCache(str(kf)).get("proj")
        assert keys is not None and len(keys) == 2
        assert keys[0][1] == blob1
        assert keys[1][2] == "key-two"

    def test_get_unknown_project(self, tmp_path: Path) -> None:
        """Unknown project returns None."""
        kf = tmp_path / "keys.json"
        kf.write_text("{}")
        assert _KeyCache(str(kf)).get("nope") is None

    def test_caches_across_calls(
        self, tmp_path: Path, ed25519_keypair: tuple[Path, Path, bytes]
    ) -> None:
        """Second get() for the same project returns cached objects (same identity)."""
        priv_path, pub_path, _ = ed25519_keypair
        kf = tmp_path / "keys.json"
        kf.write_text(
            json.dumps({"proj": [{"private_key": str(priv_path), "public_key": str(pub_path)}]})
        )
        cache = _KeyCache(str(kf))
        r1 = cache.get("proj")
        r2 = cache.get("proj")
        assert r1 is not None and r2 is not None
        assert r1[0][0] is r2[0][0]  # same private key object (cached, not re-loaded)


# ── Signing ─────────────────────────────────────────────────────────────


class TestSign:
    """Verify SSH signature format."""

    def test_ed25519_signature_format(self) -> None:
        """Ed25519 signature blob contains correct algorithm name."""
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


_READ_TIMEOUT = 5.0  # seconds — prevent tests from hanging on CI


async def _read_response(reader: asyncio.StreamReader) -> tuple[int, bytes]:
    """Read one SSH agent response message."""
    raw_len = await asyncio.wait_for(reader.readexactly(4), timeout=_READ_TIMEOUT)
    (msg_len,) = struct.unpack(">I", raw_len)
    body = await asyncio.wait_for(reader.readexactly(msg_len), timeout=_READ_TIMEOUT)
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

    async def test_multi_key_identity_listing_and_signing(self, tmp_path: Path) -> None:
        """IDENTITIES_ANSWER returns all keys for a list-format project; each key signs."""

        # Generate two independent ed25519 keypairs
        def _make_pair(name: str, comment: str | None = None) -> tuple[Path, Path, bytes]:
            k = Ed25519PrivateKey.generate()
            priv = tmp_path / name
            pub = tmp_path / f"{name}.pub"
            priv.write_bytes(k.private_bytes(Encoding.PEM, PrivateFormat.OpenSSH, NoEncryption()))
            pub_raw = k.public_key().public_bytes(Encoding.OpenSSH, PublicFormat.OpenSSH)
            pub.write_text(f"{pub_raw.decode()} {comment or name}\n")
            blob = base64.b64decode(pub_raw.decode().split()[1])
            return priv, pub, blob

        priv1, pub1, blob1 = _make_pair("id_github")
        priv2, pub2, blob2 = _make_pair("id_gitlab")

        keys_file = tmp_path / "ssh-keys.json"
        keys_file.write_text(
            json.dumps(
                {
                    "proj": [
                        {"private_key": str(priv1), "public_key": str(pub1)},
                        {"private_key": str(priv2), "public_key": str(pub2)},
                    ]
                }
            )
        )

        db = CredentialDB(tmp_path / "test.db")
        token = db.create_proxy_token("proj", "task-1", "proj", "ssh")
        db.close()

        server = await start_ssh_agent_server(
            str(tmp_path / "test.db"), str(keys_file), "127.0.0.1", 0
        )
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

            # Parse both blobs from the identities response
            mv = memoryview(payload)
            off = 4
            returned_blobs = []
            for _ in range(nkeys):
                blob, off = _unpack_string(mv, off)
                _comment, off = _unpack_string(mv, off)
                returned_blobs.append(blob)
            assert set(returned_blobs) == {blob1, blob2}

            # Sign with each key individually
            for blob in (blob1, blob2):
                sign_payload = _pack_string(blob) + _pack_string(b"sign-me") + struct.pack(">I", 0)
                writer.write(_build_msg(SSH_AGENTC_SIGN_REQUEST, sign_payload))
                await writer.drain()
                msg_type, resp = await _read_response(reader)
                assert msg_type == SSH_AGENT_SIGN_RESPONSE

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

    async def test_non_ssh_provider_token_rejected(self, tmp_path: Path) -> None:
        """A phantom token with provider != 'ssh' is rejected."""
        from cryptography.hazmat.primitives.asymmetric.ed25519 import Ed25519PrivateKey as EK
        from cryptography.hazmat.primitives.serialization import (
            Encoding as E,
            NoEncryption as NE,
            PrivateFormat as PF,
            PublicFormat as PuF,
        )

        ssh_dir = tmp_path / "keys"
        ssh_dir.mkdir()
        key = EK.generate()
        (ssh_dir / "id").write_bytes(key.private_bytes(E.PEM, PF.OpenSSH, NE()))
        pub_raw = key.public_key().public_bytes(E.OpenSSH, PuF.OpenSSH)
        (ssh_dir / "id.pub").write_text(f"{pub_raw.decode()} c\n")

        keys_file = tmp_path / "keys.json"
        keys_file.write_text(
            json.dumps(
                {
                    "proj": [
                        {"private_key": str(ssh_dir / "id"), "public_key": str(ssh_dir / "id.pub")}
                    ]
                }
            )
        )

        db = CredentialDB(tmp_path / "test.db")
        # Create a token with provider="claude" (not "ssh")
        api_token = db.create_proxy_token("proj", "task-1", "default", "claude")
        db.close()

        server = await start_ssh_agent_server(
            str(tmp_path / "test.db"), str(keys_file), "127.0.0.1", 0
        )
        port = server.sockets[0].getsockname()[1]
        try:
            reader, writer = await asyncio.open_connection("127.0.0.1", port)
            writer.write(_build_handshake(api_token))
            writer.write(_build_msg(SSH_AGENTC_REQUEST_IDENTITIES))
            await writer.drain()

            # Server rejects non-SSH tokens and closes connection
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

    async def test_malformed_sign_payload_returns_failure(self, ssh_agent_env) -> None:
        """Truncated sign request payload returns SSH_AGENT_FAILURE."""
        db_path, keys_file, token, _pub_blob = ssh_agent_env

        server = await start_ssh_agent_server(db_path, keys_file, "127.0.0.1", 0)
        port = server.sockets[0].getsockname()[1]
        try:
            reader, writer = await asyncio.open_connection("127.0.0.1", port)
            writer.write(_build_handshake(token))
            writer.write(_build_msg(SSH_AGENTC_SIGN_REQUEST, b"\x00\x00"))
            await writer.drain()

            msg_type, _payload = await _read_response(reader)
            assert msg_type == SSH_AGENT_FAILURE

            writer.close()
            await writer.wait_closed()
        finally:
            server.close()
            await server.wait_closed()

    async def test_missing_keys_file_closes_connection(self, tmp_path: Path) -> None:
        """When ssh-keys.json doesn't exist, connections fail gracefully."""
        db = CredentialDB(tmp_path / "test.db")
        token = db.create_proxy_token("ghost", "task-1", "ghost", "ssh")
        db.close()

        server = await start_ssh_agent_server(
            str(tmp_path / "test.db"), str(tmp_path / "no-such.json"), "127.0.0.1", 0
        )
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


# ── _unpack_string bounds checks ────────────────────────────────────────


class TestUnpackStringBounds:
    """Verify _unpack_string raises on invalid buffers."""

    def test_buffer_too_short_for_header(self) -> None:
        """Buffer shorter than 4 bytes raises ValueError."""
        with pytest.raises(ValueError, match="too short for string header"):
            _unpack_string(memoryview(b"\x00\x00"), 0)

    def test_length_exceeds_buffer(self) -> None:
        """Encoded length larger than remaining buffer raises ValueError."""
        buf = struct.pack(">I", 100) + b"ab"
        with pytest.raises(ValueError, match="exceeds buffer"):
            _unpack_string(memoryview(buf), 0)

    def test_offset_past_end(self) -> None:
        """Offset past buffer end raises ValueError."""
        buf = _pack_string(b"hi")
        with pytest.raises(ValueError, match="too short"):
            _unpack_string(memoryview(buf), len(buf))


# ── Key loading ─────────────────────────────────────────────────────────


class TestLoadPrivateKey:
    """Verify _load_private_key handles various key formats."""

    def test_loads_openssh_ed25519(self, ed25519_keypair: tuple[Path, Path, bytes]) -> None:
        """Loads an OpenSSH-format ed25519 key (default ssh-keygen output)."""
        priv_path, _, _ = ed25519_keypair
        key = _load_private_key(str(priv_path))
        assert isinstance(key, Ed25519PrivateKey)

    def test_loads_pem_rsa(self, tmp_path: Path) -> None:
        """Loads a traditional PEM-format RSA key."""
        from cryptography.hazmat.primitives.asymmetric.rsa import (
            RSAPrivateKey as RSAKeyType,
            generate_private_key,
        )
        from cryptography.hazmat.primitives.serialization import (
            Encoding as Enc,
            NoEncryption as NoEnc,
            PrivateFormat as PF,
        )

        rsa_key = generate_private_key(65537, 2048)
        pem = rsa_key.private_bytes(Enc.PEM, PF.TraditionalOpenSSL, NoEnc())
        key_path = tmp_path / "id_rsa"
        key_path.write_bytes(pem)
        loaded = _load_private_key(str(key_path))
        assert isinstance(loaded, RSAKeyType)

    def test_missing_file_raises(self, tmp_path: Path) -> None:
        """Missing key file raises FileNotFoundError."""
        with pytest.raises(FileNotFoundError):
            _load_private_key(str(tmp_path / "nonexistent"))


class TestLoadPublicKeyBlob:
    """Verify _load_public_key_blob handles various .pub formats."""

    def test_loads_standard_pub(self, ed25519_keypair: tuple[Path, Path, bytes]) -> None:
        """Loads a standard .pub file with type, blob, and comment."""
        _, pub_path, expected_blob = ed25519_keypair
        blob, comment = _load_public_key_blob(str(pub_path))
        assert blob == expected_blob
        assert comment == "test-comment"

    def test_pub_without_comment(self, tmp_path: Path) -> None:
        """Loads a .pub file with no comment field."""
        key = Ed25519PrivateKey.generate()
        pub_raw = key.public_key().public_bytes(Encoding.OpenSSH, PublicFormat.OpenSSH)
        pub_path = tmp_path / "id.pub"
        pub_path.write_text(pub_raw.decode() + "\n")
        blob, comment = _load_public_key_blob(str(pub_path))
        assert len(blob) > 0
        assert comment == ""

    def test_malformed_pub_raises(self, tmp_path: Path) -> None:
        """Malformed .pub file with only one field raises ValueError."""
        pub_path = tmp_path / "bad.pub"
        pub_path.write_text("just-one-field\n")
        with pytest.raises(ValueError, match="Malformed"):
            _load_public_key_blob(str(pub_path))


# ── RSA signing ─────────────────────────────────────────────────────────


class TestRSASign:
    """Verify RSA signature algorithm selection based on flags."""

    @pytest.fixture()
    def rsa_key(self):
        """Generate a test RSA key."""
        from cryptography.hazmat.primitives.asymmetric.rsa import generate_private_key

        return generate_private_key(65537, 2048)

    def test_rsa_sha2_256_flag(self, rsa_key) -> None:
        """Flag SSH_AGENT_RSA_SHA2_256 selects rsa-sha2-256 algorithm."""
        sig_blob = _sign(rsa_key, b"test", SSH_AGENT_RSA_SHA2_256)
        algo, _ = _unpack_string(memoryview(sig_blob), 0)
        assert algo == b"rsa-sha2-256"

    def test_rsa_sha2_512_flag(self, rsa_key) -> None:
        """Flag SSH_AGENT_RSA_SHA2_512 selects rsa-sha2-512 and signs with SHA-512."""
        from cryptography.hazmat.primitives.asymmetric.padding import PKCS1v15
        from cryptography.hazmat.primitives.hashes import SHA512

        data = b"test"
        sig_blob = _sign(rsa_key, data, SSH_AGENT_RSA_SHA2_512)
        algo, off = _unpack_string(memoryview(sig_blob), 0)
        assert algo == b"rsa-sha2-512"
        raw_sig, _ = _unpack_string(memoryview(sig_blob), off)
        rsa_key.public_key().verify(raw_sig, data, PKCS1v15(), SHA512())

    def test_rsa_no_flags_uses_ssh_rsa_sha1(self, rsa_key) -> None:
        """No flags defaults to ssh-rsa with SHA-1 per RFC 4253 §6.6."""
        from cryptography.hazmat.primitives.asymmetric.padding import PKCS1v15
        from cryptography.hazmat.primitives.hashes import SHA1

        data = b"test"
        sig_blob = _sign(rsa_key, data, 0)
        algo, off = _unpack_string(memoryview(sig_blob), 0)
        assert algo == b"ssh-rsa"
        raw_sig, _ = _unpack_string(memoryview(sig_blob), off)
        rsa_key.public_key().verify(raw_sig, data, PKCS1v15(), SHA1())  # noqa: S303

    def test_rsa_sha256_signature_verifies(self, rsa_key) -> None:
        """RSA-SHA2-256 signature verifies against the public key."""
        from cryptography.hazmat.primitives.asymmetric.padding import PKCS1v15
        from cryptography.hazmat.primitives.hashes import SHA256

        data = b"verify-me"
        sig_blob = _sign(rsa_key, data, SSH_AGENT_RSA_SHA2_256)
        _, off = _unpack_string(memoryview(sig_blob), 0)
        raw_sig, _ = _unpack_string(memoryview(sig_blob), off)
        rsa_key.public_key().verify(raw_sig, data, PKCS1v15(), SHA256())


# ── _KeyCache edge cases ─────────────────────────────────────────────────


class TestKeyCacheEdgeCases:
    """Verify _KeyCache re-reads the file, caches, and handles malformed data."""

    def test_missing_file_returns_none(self, tmp_path: Path) -> None:
        """Non-existent keys file returns None for any project."""
        assert _KeyCache(str(tmp_path / "no-such.json")).get("any") is None

    def test_corrupt_json_returns_none(self, tmp_path: Path) -> None:
        """Corrupt JSON file returns None gracefully."""
        kf = tmp_path / "bad.json"
        kf.write_text("{invalid json")
        assert _KeyCache(str(kf)).get("proj") is None

    def test_reflects_file_updates(
        self, tmp_path: Path, ed25519_keypair: tuple[Path, Path, bytes]
    ) -> None:
        """New entries in ssh-keys.json are visible on next get() call."""
        priv_path, pub_path, _ = ed25519_keypair
        kf = tmp_path / "keys.json"
        kf.write_text("{}")
        cache = _KeyCache(str(kf))
        assert cache.get("proj") is None

        kf.write_text(
            json.dumps({"proj": [{"private_key": str(priv_path), "public_key": str(pub_path)}]})
        )
        assert cache.get("proj") is not None

    def test_json_list_returns_none(self, tmp_path: Path) -> None:
        """JSON root is a list (not a dict) — returns None gracefully."""
        kf = tmp_path / "keys.json"
        kf.write_text('[{"private_key": "/a"}]')
        assert _KeyCache(str(kf)).get("proj") is None

    def test_entry_missing_keys_returns_none(self, tmp_path: Path) -> None:
        """List entry without required 'private_key'/'public_key' strings returns None."""
        kf = tmp_path / "keys.json"
        kf.write_text(json.dumps({"proj": [{"only_one": "/a"}]}))
        assert _KeyCache(str(kf)).get("proj") is None

    def test_entry_non_string_values_returns_none(self, tmp_path: Path) -> None:
        """List entry with non-string key path values returns None."""
        kf = tmp_path / "keys.json"
        kf.write_text(json.dumps({"proj": [{"private_key": 42, "public_key": "/b"}]}))
        assert _KeyCache(str(kf)).get("proj") is None

    def test_project_not_a_list_returns_none(self, tmp_path: Path) -> None:
        """Project entry that is not a list returns None."""
        kf = tmp_path / "keys.json"
        kf.write_text(json.dumps({"proj": {"private_key": "/a", "public_key": "/a.pub"}}))
        assert _KeyCache(str(kf)).get("proj") is None

    def test_invalidates_cache_on_path_change(
        self, tmp_path: Path, ed25519_keypair: tuple[Path, Path, bytes]
    ) -> None:
        """Cache is invalidated when key paths change in ssh-keys.json."""
        priv_path, pub_path, _ = ed25519_keypair
        kf = tmp_path / "keys.json"
        kf.write_text(
            json.dumps({"proj": [{"private_key": str(priv_path), "public_key": str(pub_path)}]})
        )
        cache = _KeyCache(str(kf))
        r1 = cache.get("proj")

        # Generate a second keypair with different paths
        key2 = Ed25519PrivateKey.generate()
        priv2 = tmp_path / "id2"
        pub2 = tmp_path / "id2.pub"
        priv2.write_bytes(key2.private_bytes(Encoding.PEM, PrivateFormat.OpenSSH, NoEncryption()))
        pub_raw2 = key2.public_key().public_bytes(Encoding.OpenSSH, PublicFormat.OpenSSH)
        pub2.write_text(f"{pub_raw2.decode()} comment2\n")

        kf.write_text(json.dumps({"proj": [{"private_key": str(priv2), "public_key": str(pub2)}]}))
        r2 = cache.get("proj")
        assert r1 is not None and r2 is not None
        assert r1[0][0] is not r2[0][0]  # different private key object (re-loaded)

    def test_reloads_on_same_path_rotation(
        self, tmp_path: Path, ed25519_keypair: tuple[Path, Path, bytes]
    ) -> None:
        """In-place key rotation (same paths, new content) reloads the cache."""
        import os

        priv_path, pub_path, _ = ed25519_keypair
        kf = tmp_path / "keys.json"
        kf.write_text(
            json.dumps({"proj": [{"private_key": str(priv_path), "public_key": str(pub_path)}]})
        )
        cache = _KeyCache(str(kf))
        r1 = cache.get("proj")

        # Overwrite the same files with a freshly generated key, then force a
        # deterministic mtime advance to avoid filesystem-granularity flakiness.
        key2 = Ed25519PrivateKey.generate()
        priv_path.write_bytes(
            key2.private_bytes(Encoding.PEM, PrivateFormat.OpenSSH, NoEncryption())
        )
        pub_raw2 = key2.public_key().public_bytes(Encoding.OpenSSH, PublicFormat.OpenSSH)
        pub_path.write_text(f"{pub_raw2.decode()} rotated\n")
        for p in (priv_path, pub_path):
            mt = p.stat().st_mtime_ns
            os.utime(p, ns=(mt, mt + 1_000_000))

        r2 = cache.get("proj")
        assert r1 is not None and r2 is not None
        assert r1[0][0] is not r2[0][0]  # different key object (reloaded from same path)


@pytest.mark.asyncio
class TestTkMainOrdering:
    """The tk-main tagged key is always returned first in IDENTITIES_ANSWER."""

    async def test_tk_main_key_is_first_regardless_of_json_order(self, tmp_path: Path) -> None:
        """tk-main key sorts first even when it is second in ssh-keys.json."""

        def _make_pair(name: str, comment: str) -> tuple[Path, Path, bytes]:
            k = Ed25519PrivateKey.generate()
            priv = tmp_path / name
            pub = tmp_path / f"{name}.pub"
            priv.write_bytes(k.private_bytes(Encoding.PEM, PrivateFormat.OpenSSH, NoEncryption()))
            pub_raw = k.public_key().public_bytes(Encoding.OpenSSH, PublicFormat.OpenSSH)
            pub.write_text(f"{pub_raw.decode()} {comment}\n")
            blob = base64.b64decode(pub_raw.decode().split()[1])
            return priv, pub, blob

        # extra key is FIRST in JSON; tk-main key is second
        priv_extra, pub_extra, blob_extra = _make_pair("id_extra", "extra-key")
        priv_main, pub_main, blob_main = _make_pair("id_main", "tk-main:myproject")

        keys_file = tmp_path / "ssh-keys.json"
        keys_file.write_text(
            json.dumps(
                {
                    "myproject": [
                        {"private_key": str(priv_extra), "public_key": str(pub_extra)},
                        {"private_key": str(priv_main), "public_key": str(pub_main)},
                    ]
                }
            )
        )

        db = CredentialDB(tmp_path / "test.db")
        token = db.create_proxy_token("myproject", "task-1", "myproject", "ssh")
        db.close()

        server = await start_ssh_agent_server(
            str(tmp_path / "test.db"), str(keys_file), "127.0.0.1", 0
        )
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

            # First returned key must be the tk-main key
            mv = memoryview(payload)
            first_blob, off = _unpack_string(mv, 4)
            first_comment, _ = _unpack_string(mv, off)
            assert first_blob == blob_main, "tk-main key must be returned first"
            assert first_comment == b"tk-main:myproject"

            writer.close()
            await writer.wait_closed()
        finally:
            server.close()
            await server.wait_closed()

    async def test_non_tagged_keys_preserve_relative_order(self, tmp_path: Path) -> None:
        """Keys without tk-main tag keep their original JSON order."""

        def _make_pair(name: str, comment: str) -> tuple[Path, Path, bytes]:
            k = Ed25519PrivateKey.generate()
            priv = tmp_path / name
            pub = tmp_path / f"{name}.pub"
            priv.write_bytes(k.private_bytes(Encoding.PEM, PrivateFormat.OpenSSH, NoEncryption()))
            pub_raw = k.public_key().public_bytes(Encoding.OpenSSH, PublicFormat.OpenSSH)
            pub.write_text(f"{pub_raw.decode()} {comment}\n")
            blob = base64.b64decode(pub_raw.decode().split()[1])
            return priv, pub, blob

        priv_a, pub_a, blob_a = _make_pair("id_a", "key-a")
        priv_b, pub_b, blob_b = _make_pair("id_b", "key-b")

        keys_file = tmp_path / "ssh-keys.json"
        keys_file.write_text(
            json.dumps(
                {
                    "proj": [
                        {"private_key": str(priv_a), "public_key": str(pub_a)},
                        {"private_key": str(priv_b), "public_key": str(pub_b)},
                    ]
                }
            )
        )

        db = CredentialDB(tmp_path / "test.db")
        token = db.create_proxy_token("proj", "task-1", "proj", "ssh")
        db.close()

        server = await start_ssh_agent_server(
            str(tmp_path / "test.db"), str(keys_file), "127.0.0.1", 0
        )
        port = server.sockets[0].getsockname()[1]
        try:
            reader, writer = await asyncio.open_connection("127.0.0.1", port)
            writer.write(_build_handshake(token))
            writer.write(_build_msg(SSH_AGENTC_REQUEST_IDENTITIES))
            await writer.drain()

            msg_type, payload = await _read_response(reader)
            assert msg_type == SSH_AGENT_IDENTITIES_ANSWER
            mv = memoryview(payload)
            first_blob, _ = _unpack_string(mv, 4)
            assert first_blob == blob_a, "JSON order preserved when no tk-main key present"

            writer.close()
            await writer.wait_closed()
        finally:
            server.close()
            await server.wait_closed()
