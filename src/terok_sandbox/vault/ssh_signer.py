# SPDX-FileCopyrightText: 2026 Jiri Vyskocil
# SPDX-License-Identifier: Apache-2.0

"""SSH signer — signs with vault-stored private keys on behalf of clients.

Implements the `SSH agent protocol`_ in two deployment flavours that share
one connection handler:

- **Container-facing** (``start_ssh_signer``) — TCP or Unix, guarded by a
  phantom-token handshake so a compromised container can't impersonate
  another.  Scope is resolved from the token.
- **Host-local, per-scope** (``start_ssh_signer_local``) — one Unix socket
  per scope at mode 0600.  Scope is fixed at bind time and the UID-gated
  filesystem permissions are the whole access control — host processes
  don't cross a trust boundary.

Private keys live in ``credentials.db`` (``ssh_keys`` table) and never touch
the filesystem; the ``_DBKeyCache`` reloads them only when the DB version
counter advances.

.. _SSH agent protocol: https://datatracker.ietf.org/doc/html/draft-miller-ssh-agent
"""

from __future__ import annotations

import asyncio
import logging
import struct
from pathlib import Path

from cryptography.hazmat.primitives.asymmetric.ed25519 import Ed25519PrivateKey
from cryptography.hazmat.primitives.asymmetric.padding import PKCS1v15
from cryptography.hazmat.primitives.asymmetric.rsa import RSAPrivateKey
from cryptography.hazmat.primitives.hashes import SHA256, SHA512
from cryptography.hazmat.primitives.serialization import (
    load_pem_private_key,
    load_ssh_private_key,
)

from ..credentials.db import SSHKeyRecord

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

_LOCAL_SOCKET_MODE = 0o600


# ---------------------------------------------------------------------------
# Wire format helpers
# ---------------------------------------------------------------------------


def _pack_string(data: bytes) -> bytes:
    """Pack *data* as an SSH wire-format string: ``[4-byte len][data]``."""
    return struct.pack(">I", len(data)) + data


def _unpack_string(buf: memoryview, offset: int) -> tuple[bytes, int]:
    """Unpack an SSH string from *buf* at *offset*.  Returns ``(data, new_offset)``.

    Raises ``ValueError`` if the encoded length exceeds the remaining buffer.
    """
    if offset + 4 > len(buf):
        raise ValueError(f"Buffer too short for string header at offset {offset}")
    (slen,) = struct.unpack_from(">I", buf, offset)
    start = offset + 4
    end = start + slen
    if end > len(buf):
        raise ValueError(f"String length {slen} exceeds buffer (remaining {len(buf) - start})")
    return bytes(buf[start:end]), end


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
# Key cache — DB-backed, version-counter invalidation
# ---------------------------------------------------------------------------


_ResolvedKey = tuple[Ed25519PrivateKey | RSAPrivateKey, bytes, str]
"""(private_key, pub_blob, comment) — one loaded keypair."""


class _DBKeyCache:
    """Caches decoded SSH keys per scope, reloading when the DB version bumps.

    The DB's ``ssh_keys_version`` counter increments on every insert,
    assignment, or unassignment.  A cached slot is valid only as long as
    the version it was loaded at matches the current version — so
    ``ssh-init``, ``ssh-import``, and ``ssh-remove`` all invalidate
    transparently without a proxy restart.
    """

    def __init__(self, token_db) -> None:
        self._token_db = token_db
        self._cache: dict[str, tuple[int, list[_ResolvedKey]]] = {}

    def get(self, scope: str) -> list[_ResolvedKey]:
        """Return the scope's resolved keys, or ``[]`` if none are assigned."""
        version = self._token_db.ssh_keys_version()
        cached = self._cache.get(scope)
        if cached and cached[0] == version:
            return cached[1]

        resolved: list[_ResolvedKey] = []
        for record in self._token_db.load_ssh_keys_for_scope(scope):
            try:
                resolved.append(_decode_record(record))
            except ValueError as exc:
                _logger.error(
                    "Failed to decode SSH key id=%d for scope %r: %s",
                    record.id,
                    scope,
                    exc,
                )

        self._cache[scope] = (version, resolved)
        return resolved


def _decode_record(record: SSHKeyRecord) -> _ResolvedKey:
    """Decode ``record.private_pem`` into a cryptography private key object."""
    if b"OPENSSH PRIVATE KEY" in record.private_pem:
        key = load_ssh_private_key(record.private_pem, password=None)
    else:
        key = load_pem_private_key(record.private_pem, password=None)
    if not isinstance(key, (Ed25519PrivateKey, RSAPrivateKey)):
        raise ValueError(f"Unsupported key type: {type(key).__name__}")
    return key, record.public_blob, record.comment


# ---------------------------------------------------------------------------
# Signing
# ---------------------------------------------------------------------------


def _sign(key: Ed25519PrivateKey | RSAPrivateKey, data: bytes, flags: int) -> bytes:
    """Sign *data* and return the SSH wire-format signature blob.

    The signature blob is ``[string: algorithm][string: raw signature]``.
    """
    if isinstance(key, Ed25519PrivateKey):
        raw_sig = key.sign(data)
        return _pack_string(b"ssh-ed25519") + _pack_string(raw_sig)

    # RSA: prefer RFC 8332 RSA-SHA2 algorithms; fall back to SHA-256 when the
    # client requests no specific hash.  Legacy ssh-rsa (SHA-1) is not offered:
    # OpenSSH 8.7+ rejects SHA-1 signatures, and SHA-1 is no longer collision
    # resistant.  Clients still asking for ssh-rsa are served with SHA-256
    # signatures labelled ``rsa-sha2-256`` (the widest compatible choice).
    if flags & SSH_AGENT_RSA_SHA2_512:
        algo, hash_cls = b"rsa-sha2-512", SHA512()
    else:
        algo, hash_cls = b"rsa-sha2-256", SHA256()
    raw_sig = key.sign(data, PKCS1v15(), hash_cls)
    return _pack_string(algo) + _pack_string(raw_sig)


# ---------------------------------------------------------------------------
# Connection handlers
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


async def _resolve_scope_from_token(
    reader: asyncio.StreamReader, writer: asyncio.StreamWriter, token_db
) -> str | None:
    """Container-facing: read and validate the phantom token, return the scope."""
    peer = writer.get_extra_info("peername")
    token = await _read_handshake(reader)
    if not token:
        _logger.info("Handshake failed (no token) from %s", peer)
        return None
    token_info = token_db.lookup_token(token)
    if token_info is None:
        _logger.warning("Invalid SSH agent token from %s", peer)
        return None
    if token_info.get("provider") != "ssh":
        _logger.warning("Token provider %r is not 'ssh', rejecting", token_info.get("provider"))
        return None
    return token_info["scope"]


async def _serve_agent_session(
    reader: asyncio.StreamReader,
    writer: asyncio.StreamWriter,
    scope: str,
    key_cache: _DBKeyCache,
) -> None:
    """Run the agent message loop for a connection bound to *scope*."""
    peer = writer.get_extra_info("peername")
    keys = key_cache.get(scope)
    if not keys:
        _logger.warning("No SSH keys loaded for scope %r", scope)
        return

    # Promote the tk-main key to the front so SSH offers it first to GitHub,
    # ensuring the primary workspace key is used for the main repo without
    # requiring IdentityFile configuration in the container.
    keys = sorted(keys, key=lambda k: 0 if k[2].startswith("tk-main:") else 1)
    key_by_blob = {pub_blob: (priv, comment) for priv, pub_blob, comment in keys}
    _logger.info(
        "SSH agent session ready for scope %r from %s — %d key(s) available",
        scope,
        peer,
        len(keys),
    )

    while True:
        try:
            msg_type, payload = await _read_msg(reader)
        except (asyncio.IncompleteReadError, ValueError):
            break

        if msg_type == SSH_AGENTC_REQUEST_IDENTITIES:
            _logger.info("Identity request for scope %r — returning %d key(s)", scope, len(keys))
            body = struct.pack(">I", len(keys))
            for _priv, pub_blob, comment in keys:
                body += _pack_string(pub_blob)
                body += _pack_string(comment.encode("utf-8"))
            _write_msg(writer, SSH_AGENT_IDENTITIES_ANSWER, body)

        elif msg_type == SSH_AGENTC_SIGN_REQUEST:
            try:
                mv = memoryview(payload)
                req_blob, off = _unpack_string(mv, 0)
                sign_data, off = _unpack_string(mv, off)
                if off + 4 > len(mv):
                    raise ValueError("Payload too short for flags field")
                (flags,) = struct.unpack_from(">I", mv, off)
            except (ValueError, struct.error) as exc:
                _logger.warning("Malformed sign request from %s: %s", peer, exc)
                _write_msg(writer, SSH_AGENT_FAILURE)
            else:
                match = key_by_blob.get(req_blob)
                if match is None:
                    _logger.warning(
                        "Sign request for unknown key from %s (scope %r) — no matching key blob",
                        peer,
                        scope,
                    )
                    _write_msg(writer, SSH_AGENT_FAILURE)
                else:
                    _logger.info("Sign request fulfilled for scope %r key %r", scope, match[1])
                    sig_blob = _sign(match[0], sign_data, flags)
                    _write_msg(writer, SSH_AGENT_SIGN_RESPONSE, _pack_string(sig_blob))

        else:
            _write_msg(writer, SSH_AGENT_FAILURE)

        await writer.drain()


async def _handle_container_connection(
    reader: asyncio.StreamReader,
    writer: asyncio.StreamWriter,
    token_db,
    key_cache: _DBKeyCache,
) -> None:
    """Container-facing connection: handshake → scope → agent session."""
    try:
        scope = await _resolve_scope_from_token(reader, writer, token_db)
        if scope is None:
            return
        await _serve_agent_session(reader, writer, scope, key_cache)
    except Exception:
        _logger.exception("SSH agent connection error from %s", writer.get_extra_info("peername"))
    finally:
        writer.close()
        try:
            await writer.wait_closed()
        except Exception:  # noqa: BLE001
            pass


async def _handle_local_connection(
    reader: asyncio.StreamReader,
    writer: asyncio.StreamWriter,
    scope: str,
    key_cache: _DBKeyCache,
) -> None:
    """Host-local connection: scope is fixed at bind time, no handshake."""
    try:
        await _serve_agent_session(reader, writer, scope, key_cache)
    except Exception:
        _logger.exception("SSH agent connection error on local socket for scope %r", scope)
    finally:
        writer.close()
        try:
            await writer.wait_closed()
        except Exception:  # noqa: BLE001
            pass


# ---------------------------------------------------------------------------
# Server factories
# ---------------------------------------------------------------------------


async def start_ssh_signer(
    db_path: str,
    host: str | None = None,
    port: int | None = None,
    socket_path: str | None = None,
) -> asyncio.Server:
    """Start the container-facing SSH signer (token-gated).

    Args:
        db_path: Path to the vault sqlite3 database (phantom tokens + SSH keys).
        host: Bind address for TCP (typically ``"127.0.0.1"``).
        port: TCP port to listen on.
        socket_path: Unix socket path to listen on.

    Returns:
        The running :class:`asyncio.Server` — caller is responsible for closing it.

    Raises:
        ValueError: If neither TCP (host+port) nor socket_path is provided.
    """
    from .token_broker import _TokenDB

    token_db = _TokenDB(db_path)
    key_cache = _DBKeyCache(token_db)

    async def _on_connect(reader: asyncio.StreamReader, writer: asyncio.StreamWriter) -> None:
        """Handle an incoming SSH agent connection (token-gated)."""
        await _handle_container_connection(reader, writer, token_db, key_cache)

    if socket_path:
        server = await _bind_hardened_unix(socket_path, _on_connect)
        _logger.info("SSH signer listening on %s", socket_path)
    elif host is not None and port is not None:
        server = await asyncio.start_server(_on_connect, host, port)
        _logger.info("SSH signer listening on %s:%d", host, port)
    else:
        raise ValueError("Either socket_path or host+port must be provided")
    return server


async def start_ssh_signer_local(*, scope: str, socket_path: Path, db_path: str) -> asyncio.Server:
    """Start a host-local SSH signer bound to a single scope.

    The returned server listens on a mode-0600 Unix socket; same-UID
    filesystem permissions are the whole access control.  No token
    handshake — every accepted connection immediately enters the agent
    message loop with *scope* as its fixed identity source.
    """
    from .token_broker import _TokenDB

    token_db = _TokenDB(db_path)
    key_cache = _DBKeyCache(token_db)

    async def _on_connect(reader: asyncio.StreamReader, writer: asyncio.StreamWriter) -> None:
        """Handle an incoming host-local SSH agent connection."""
        await _handle_local_connection(reader, writer, scope, key_cache)

    server = await _bind_hardened_unix(str(socket_path), _on_connect, mode=_LOCAL_SOCKET_MODE)
    _logger.info("SSH signer listening on %s for scope %r", socket_path, scope)
    return server


async def _bind_hardened_unix(
    path_str: str,
    on_connect,
    *,
    mode: int | None = None,
) -> asyncio.Server:
    """Bind a Unix-domain socket with SELinux labelling and optional 0600 hardening."""
    import socket as _socket

    from terok_sandbox._util._net import harden_socket, prepare_socket_path
    from terok_sandbox._util._selinux import socket_selinux_context

    path = Path(path_str)
    prepare_socket_path(path)
    # Bind+listen synchronously inside the SELinux context — the
    # socket-creation-context is process-scoped, so awaiting inside the
    # context manager could leak ``terok_socket_t`` onto sockets created
    # by unrelated coroutines running during the await.
    with socket_selinux_context():
        sock = _socket.socket(_socket.AF_UNIX, _socket.SOCK_STREAM)
        sock.bind(str(path))
        sock.listen(128)
    server = await asyncio.start_unix_server(on_connect, sock=sock)
    harden_socket(path)
    if mode is not None:
        import os as _os

        _os.chmod(path, mode)
    return server
