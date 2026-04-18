# SPDX-FileCopyrightText: 2026 Jiri Vyskocil
# SPDX-License-Identifier: Apache-2.0

"""SSH signer — signs with host-side private keys on behalf of containers.

Implements the `SSH agent protocol`_ over TCP with a phantom-token handshake.
Containers connect via a socat bridge (``UNIX-LISTEN → TCP``) and set
``SSH_AUTH_SOCK`` to the local Unix socket.  Private keys never enter the
container — the signer reads them from the host filesystem.

Like :mod:`token_broker`, this module has **zero terok imports**.  It is a
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
from cryptography.hazmat.primitives.asymmetric.padding import PKCS1v15
from cryptography.hazmat.primitives.asymmetric.rsa import RSAPrivateKey
from cryptography.hazmat.primitives.hashes import SHA256, SHA512
from cryptography.hazmat.primitives.serialization import (
    load_pem_private_key,
    load_ssh_private_key,
)

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
# Key table (loads ssh-keys.json)
# ---------------------------------------------------------------------------


_ResolvedKey = tuple[Ed25519PrivateKey | RSAPrivateKey, bytes, str]
"""(private_key, pub_blob, comment) — one loaded keypair."""

# JSON entry: {"private_key": str, "public_key": str}
_KeyEntry = dict[str, str]

# Cache: fingerprint string → list of resolved keys
_CacheSlot = tuple[str, list[_ResolvedKey]]


class _KeyCache:
    """Caches resolved SSH key material per credential scope.

    Each scope may have one key (dict entry) or multiple keys (list of
    dict entries) in ``ssh-keys.json``.  On each :meth:`get` call the
    sidecar JSON is re-read (so ``ssh-init`` changes are visible without
    a proxy restart).  When the key paths and mtimes haven't changed
    since the last load, the cached material is returned directly.

    The file may not exist on a fresh install (returns ``None``).
    """

    def __init__(self, keys_path: str) -> None:
        """Store the *keys_path* for on-demand reads."""
        self._path = Path(keys_path)
        self._cache: dict[str, _CacheSlot] = {}

    def get(self, scope: str) -> list[_ResolvedKey] | None:
        """Return a list of ``(private_key, pub_blob, comment)`` or ``None``."""
        entries = self._lookup_entries(scope)
        if not entries:
            self._cache.pop(scope, None)
            return None

        fingerprint = self._fingerprint(entries)
        cached = self._cache.get(scope)
        if cached and cached[0] == fingerprint:
            return cached[1]

        # Cache miss or stale: load all keys from disk
        resolved: list[_ResolvedKey] = []
        for entry in entries:
            try:
                private_key = _load_private_key(entry["private_key"])
                pub_blob, comment = _load_public_key_blob(entry["public_key"])
                resolved.append((private_key, pub_blob, comment))
            except (OSError, ValueError) as exc:
                _logger.error(
                    "Failed to load SSH key for scope %r (%s): %s",
                    scope,
                    entry.get("private_key", "?"),
                    exc,
                )

        if not resolved:
            self._cache.pop(scope, None)
            return None

        self._cache[scope] = (fingerprint, resolved)
        return resolved

    @staticmethod
    def _fingerprint(entries: list[_KeyEntry]) -> str:
        """Build a cache-invalidation fingerprint from paths + mtimes."""
        parts: list[str] = []
        for e in entries:
            for key in ("private_key", "public_key"):
                path = e[key]
                try:
                    mt = Path(path).stat().st_mtime_ns
                except OSError:
                    mt = 0
                parts.append(f"{path}:{mt}")
        return "|".join(parts)

    def _lookup_entries(self, scope: str) -> list[_KeyEntry] | None:
        """Read ssh-keys.json and return the key entries for *scope*.

        The JSON maps credential scopes to lists of ``{"private_key", "public_key"}``
        dicts.  Uses ``LOCK_SH`` to coordinate with the ``LOCK_EX`` writer in
        :func:`update_ssh_keys_json`.
        """
        import fcntl

        if not self._path.is_file():
            return None
        try:
            fd = self._path.open()
            try:
                fcntl.flock(fd, fcntl.LOCK_SH)
                mapping = json.loads(fd.read())
            finally:
                fcntl.flock(fd, fcntl.LOCK_UN)
                fd.close()
        except (json.JSONDecodeError, OSError):
            return None
        if not isinstance(mapping, dict):
            return None
        entries = mapping.get(scope)
        if not isinstance(entries, list):
            return None
        return [
            e
            for e in entries
            if isinstance(e, dict)
            and isinstance(e.get("private_key"), str)
            and isinstance(e.get("public_key"), str)
        ] or None


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

    Raises ``ValueError`` if the file format is invalid.
    """
    text = Path(pub_key_path).read_text(encoding="utf-8").strip()
    parts = text.split(None, 2)
    if len(parts) < 2:
        raise ValueError("Malformed public key file: expected '<type> <base64> [comment]'")
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
    key_cache: _KeyCache,
) -> None:
    """Handle one SSH agent TCP connection.

    1. Read phantom-token handshake → validate via DB
    2. Resolve SSH key from cache (or load from filesystem on miss)
    3. Serve agent protocol messages until EOF
    """
    peer = writer.get_extra_info("peername")
    try:
        # --- Handshake ---
        token = await _read_handshake(reader)
        if not token:
            _logger.info("Handshake failed (no token) from %s", peer)
            return

        token_info = token_db.lookup_token(token)  # type: ignore[attr-defined]
        if token_info is None:
            _logger.warning("Invalid SSH agent token from %s", peer)
            return

        if token_info.get("provider") != "ssh":
            _logger.warning("Token provider %r is not 'ssh', rejecting", token_info.get("provider"))
            return

        scope = token_info["scope"]
        _logger.info("SSH agent connection from %s for scope %r", peer, scope)
        keys = key_cache.get(scope)
        if not keys:
            _logger.warning("No SSH keys loaded for scope %r — check ssh-keys.json paths", scope)
            return

        # Promote the tk-main key to the front so SSH offers it first to GitHub,
        # ensuring the primary workspace key is used for the main repo without
        # requiring IdentityFile configuration in the container.
        keys = sorted(keys, key=lambda k: 0 if k[2].startswith("tk-main:") else 1)

        # Build a lookup: pub_blob → (private_key, comment) for sign requests
        key_by_blob = {pub_blob: (priv, comment) for priv, pub_blob, comment in keys}
        _logger.info(
            "SSH agent session ready for scope %r from %s — %d key(s) available",
            scope,
            peer,
            len(keys),
        )

        # --- Agent message loop ---
        while True:
            try:
                msg_type, payload = await _read_msg(reader)
            except (asyncio.IncompleteReadError, ValueError):
                break

            if msg_type == SSH_AGENTC_REQUEST_IDENTITIES:
                _logger.info(
                    "Identity request for scope %r — returning %d key(s)", scope, len(keys)
                )
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


async def start_ssh_signer(
    db_path: str,
    keys_file: str,
    host: str | None = None,
    port: int | None = None,
    socket_path: str | None = None,
) -> asyncio.Server:
    """Start the SSH signer on TCP, a Unix socket, or both.

    Args:
        db_path: Path to the vault sqlite3 database (for phantom token lookups).
        keys_file: Path to ``ssh-keys.json`` mapping credential scopes to key file paths.
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
    key_cache = _KeyCache(keys_file)

    async def _on_connect(reader: asyncio.StreamReader, writer: asyncio.StreamWriter) -> None:
        """Handle an incoming SSH agent connection."""
        await _handle_connection(reader, writer, token_db, key_cache)

    if socket_path:
        import socket as _socket

        from terok_sandbox._util._net import harden_socket, prepare_socket_path
        from terok_sandbox._util._selinux import socket_selinux_context

        path = Path(socket_path)
        prepare_socket_path(path)
        # Bind+listen synchronously inside the SELinux context — the
        # socket-creation-context is process-scoped, so awaiting inside
        # the context manager could leak ``terok_socket_t`` onto sockets
        # created by unrelated coroutines running during the await.
        with socket_selinux_context():
            sock = _socket.socket(_socket.AF_UNIX, _socket.SOCK_STREAM)
            sock.bind(str(path))
            sock.listen(128)
        server = await asyncio.start_unix_server(_on_connect, sock=sock)
        harden_socket(path)
        _logger.info("SSH signer listening on %s", path)
    elif host is not None and port is not None:
        server = await asyncio.start_server(_on_connect, host, port)
        _logger.info("SSH signer listening on %s:%d", host, port)
    else:
        raise ValueError("Either socket_path or host+port must be provided")
    return server
