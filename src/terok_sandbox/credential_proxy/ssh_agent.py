# SPDX-FileCopyrightText: 2026 Jiri Vyskocil
# SPDX-License-Identifier: Apache-2.0

"""SSH agent proxy — signs with host-side private keys on behalf of containers.

Implements the `SSH agent protocol`_ over TCP with a phantom-token handshake.
Containers connect via a socat bridge (``UNIX-LISTEN → TCP``) and set
``SSH_AUTH_SOCK`` to the local Unix socket.  Private keys never enter the
container — the proxy reads them from the host filesystem.

Like :mod:`server`, this module has **zero terok imports**.  It is a
self-contained security component that reads phantom tokens from the same
sqlite3 database and key paths from a JSON sidecar file.

Wire format (per `draft-miller-ssh-agent`_):

    [4-byte big-endian length][1-byte message type][payload]

Custom handshake (first bytes on each TCP connection):

    [4-byte big-endian length][phantom-token UTF-8 bytes]

.. _SSH agent protocol: https://datatracker.ietf.org/doc/html/draft-miller-ssh-agent
"""

from __future__ import annotations

import asyncio
import base64
import json
import logging
import struct
from pathlib import Path

from cryptography.hazmat.primitives.asymmetric.ed25519 import Ed25519PrivateKey
from cryptography.hazmat.primitives.asymmetric.rsa import RSAPrivateKey
from cryptography.hazmat.primitives.hashes import SHA256, SHA512
from cryptography.hazmat.primitives.serialization import (
    load_pem_private_key,
    load_ssh_private_key,
)

if RSAPrivateKey:  # keep ruff happy about the conditional import
    from cryptography.hazmat.primitives.asymmetric.padding import PKCS1v15

_logger = logging.getLogger("terok-ssh-agent")

# ---------------------------------------------------------------------------
# SSH agent protocol constants
# ---------------------------------------------------------------------------

SSH_AGENT_FAILURE = 5
SSH_AGENTC_REQUEST_IDENTITIES = 11
SSH_AGENT_IDENTITIES_ANSWER = 12
SSH_AGENTC_SIGN_REQUEST = 13
SSH_AGENT_SIGN_RESPONSE = 14

# Sign request flag bits
SSH_AGENT_RSA_SHA2_256 = 0x02
SSH_AGENT_RSA_SHA2_512 = 0x04

_HANDSHAKE_TIMEOUT = 5.0  # seconds
_MSG_MAX_LEN = 256 * 1024  # 256 KiB — generous for sign requests

# ---------------------------------------------------------------------------
# Wire format helpers
# ---------------------------------------------------------------------------


def _pack_string(data: bytes) -> bytes:
    """Pack *data* as an SSH wire-format string: ``[4-byte len][data]``."""
    return struct.pack(">I", len(data)) + data


def _unpack_string(buf: memoryview, offset: int) -> tuple[bytes, int]:
    """Unpack an SSH string from *buf* at *offset*.  Returns ``(data, new_offset)``."""
    (slen,) = struct.unpack_from(">I", buf, offset)
    start = offset + 4
    return bytes(buf[start : start + slen]), start + slen


async def _read_msg(reader: asyncio.StreamReader) -> tuple[int, bytes]:
    """Read one SSH agent message.  Returns ``(msg_type, payload)``."""
    raw_len = await reader.readexactly(4)
    (msg_len,) = struct.unpack(">I", raw_len)
    if msg_len < 1 or msg_len > _MSG_MAX_LEN:
        raise ValueError(f"Invalid message length: {msg_len}")
    body = await reader.readexactly(msg_len)
    return body[0], body[1:]


def _write_msg(writer: asyncio.StreamWriter, msg_type: int, payload: bytes = b"") -> None:
    """Write one SSH agent message."""
    body = bytes([msg_type]) + payload
    writer.write(struct.pack(">I", len(body)) + body)


# ---------------------------------------------------------------------------
# Key table (loads ssh-keys.json)
# ---------------------------------------------------------------------------


class _KeyTable:
    """Project-keyed SSH key path table loaded from a JSON file."""

    def __init__(self, keys_path: str) -> None:
        """Load the key mapping from *keys_path* (JSON file)."""
        with open(keys_path) as f:
            self._keys: dict[str, dict[str, str]] = json.load(f)

    def get(self, project: str) -> dict[str, str] | None:
        """Return ``{"private_key": path, "public_key": path}`` or ``None``."""
        return self._keys.get(project)


# ---------------------------------------------------------------------------
# Key loading and signing
# ---------------------------------------------------------------------------


def _load_private_key(key_path: str) -> Ed25519PrivateKey | RSAPrivateKey:
    """Load an SSH private key from a file on the host filesystem.

    Supports both OpenSSH format (``BEGIN OPENSSH PRIVATE KEY``, the default
    for ``ssh-keygen`` since OpenSSH 7.8) and traditional PEM format.
    """
    raw = Path(key_path).read_bytes()
    if b"OPENSSH PRIVATE KEY" in raw:
        key = load_ssh_private_key(raw, password=None)
    else:
        key = load_pem_private_key(raw, password=None)
    if not isinstance(key, (Ed25519PrivateKey, RSAPrivateKey)):
        raise ValueError(f"Unsupported key type: {type(key).__name__}")
    return key


def _load_public_key_blob(pub_key_path: str) -> tuple[bytes, str]:
    """Load the SSH wire-format public key blob and comment from a ``.pub`` file.

    Returns ``(key_blob, comment)``.  The blob is the base64-decoded middle
    field of the ``<type> <base64> <comment>`` format.
    """
    text = Path(pub_key_path).read_text(encoding="utf-8").strip()
    parts = text.split(None, 2)
    blob = base64.b64decode(parts[1])
    comment = parts[2] if len(parts) > 2 else ""
    return blob, comment


def _sign(key: Ed25519PrivateKey | RSAPrivateKey, data: bytes, flags: int) -> bytes:
    """Sign *data* and return the SSH wire-format signature blob.

    The signature blob is ``[string: algorithm][string: raw signature]``.
    """
    if isinstance(key, Ed25519PrivateKey):
        raw_sig = key.sign(data)
        return _pack_string(b"ssh-ed25519") + _pack_string(raw_sig)

    # RSA: choose algorithm based on flags
    if flags & SSH_AGENT_RSA_SHA2_512:
        algo, hash_cls = b"rsa-sha2-512", SHA512()
    elif flags & SSH_AGENT_RSA_SHA2_256:
        algo, hash_cls = b"rsa-sha2-256", SHA256()
    else:
        algo, hash_cls = b"ssh-rsa", SHA256()
    raw_sig = key.sign(data, PKCS1v15(), hash_cls)
    return _pack_string(algo) + _pack_string(raw_sig)


# ---------------------------------------------------------------------------
# Connection handler
# ---------------------------------------------------------------------------


async def _read_handshake(reader: asyncio.StreamReader) -> str | None:
    """Read the phantom-token handshake prefix.

    Returns the token string, or ``None`` on invalid input.
    """
    try:
        raw_len = await asyncio.wait_for(reader.readexactly(4), timeout=_HANDSHAKE_TIMEOUT)
    except (TimeoutError, asyncio.IncompleteReadError):
        return None
    (token_len,) = struct.unpack(">I", raw_len)
    if token_len < 1 or token_len > 1024:
        return None
    try:
        raw_token = await asyncio.wait_for(
            reader.readexactly(token_len), timeout=_HANDSHAKE_TIMEOUT
        )
    except (TimeoutError, asyncio.IncompleteReadError):
        return None
    return raw_token.decode("utf-8", errors="replace")


async def _handle_connection(
    reader: asyncio.StreamReader,
    writer: asyncio.StreamWriter,
    token_db: object,
    key_table: _KeyTable,
) -> None:
    """Handle one SSH agent TCP connection.

    1. Read phantom-token handshake → validate via DB
    2. Load SSH key from host filesystem
    3. Serve agent protocol messages until EOF
    """
    peer = writer.get_extra_info("peername")
    try:
        # --- Handshake ---
        token = await _read_handshake(reader)
        if not token:
            _logger.debug("Handshake failed from %s", peer)
            return

        token_info = token_db.lookup_token(token)  # type: ignore[attr-defined]
        if token_info is None:
            _logger.warning("Invalid SSH agent token from %s", peer)
            return

        project = token_info["project"]
        key_entry = key_table.get(project)
        if key_entry is None:
            _logger.warning("No SSH key configured for project %r", project)
            return

        try:
            private_key = _load_private_key(key_entry["private_key"])
            pub_blob, comment = _load_public_key_blob(key_entry["public_key"])
        except (FileNotFoundError, ValueError) as exc:
            _logger.error("Failed to load SSH key for project %r: %s", project, exc)
            return

        _logger.debug("SSH agent session for project %r from %s", project, peer)

        # --- Agent message loop ---
        while True:
            try:
                msg_type, payload = await _read_msg(reader)
            except (asyncio.IncompleteReadError, ValueError):
                break

            if msg_type == SSH_AGENTC_REQUEST_IDENTITIES:
                # Respond with one identity: the project's public key
                body = struct.pack(">I", 1)  # nkeys = 1
                body += _pack_string(pub_blob)
                body += _pack_string(comment.encode("utf-8"))
                _write_msg(writer, SSH_AGENT_IDENTITIES_ANSWER, body)

            elif msg_type == SSH_AGENTC_SIGN_REQUEST:
                mv = memoryview(payload)
                req_blob, off = _unpack_string(mv, 0)
                sign_data, off = _unpack_string(mv, off)
                (flags,) = struct.unpack_from(">I", mv, off)

                if req_blob != pub_blob:
                    _logger.debug("Sign request for unknown key, returning failure")
                    _write_msg(writer, SSH_AGENT_FAILURE)
                else:
                    sig_blob = _sign(private_key, sign_data, flags)
                    _write_msg(writer, SSH_AGENT_SIGN_RESPONSE, _pack_string(sig_blob))

            else:
                _write_msg(writer, SSH_AGENT_FAILURE)

            await writer.drain()

    except Exception:
        _logger.exception("SSH agent connection error from %s", peer)
    finally:
        writer.close()
        try:
            await writer.wait_closed()
        except Exception:  # noqa: BLE001
            pass


# ---------------------------------------------------------------------------
# Server factory
# ---------------------------------------------------------------------------


async def start_ssh_agent_server(
    db_path: str, keys_file: str, host: str, port: int
) -> asyncio.Server:
    """Start the SSH agent TCP server.

    Args:
        db_path: Path to the credential proxy sqlite3 database (for phantom token lookups).
        keys_file: Path to ``ssh-keys.json`` mapping project IDs to key file paths.
        host: Bind address (typically ``"127.0.0.1"``).
        port: TCP port to listen on.

    Returns:
        The running :class:`asyncio.Server` — caller is responsible for closing it.
    """
    from .server import _TokenDB

    token_db = _TokenDB(db_path)
    key_table = _KeyTable(keys_file)

    async def _on_connect(reader: asyncio.StreamReader, writer: asyncio.StreamWriter) -> None:
        """Handle an incoming SSH agent connection."""
        await _handle_connection(reader, writer, token_db, key_table)

    server = await asyncio.start_server(_on_connect, host, port)
    _logger.info("SSH agent proxy listening on %s:%d", host, port)
    return server
