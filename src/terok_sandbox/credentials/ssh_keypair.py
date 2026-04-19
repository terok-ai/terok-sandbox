# SPDX-FileCopyrightText: 2026 Jiri Vyskocil
# SPDX-License-Identifier: Apache-2.0

"""SSH keypair generation, import, and export against the credential DB.

The DB is the canonical home for SSH keys.  This module moves material in
and out of that form:

- :func:`generate_keypair` creates a fresh keypair in memory.
- :func:`import_ssh_keypair` reads an existing OpenSSH keypair from files
  and registers it against a scope.
- :func:`export_ssh_keypair` writes a scope's key to a standard
  ``id_<type>_<fingerprint>`` / ``.pub`` file pair for handing to tools
  that cannot use the SSH agent.

All flows share one vocabulary: the :class:`GeneratedKeypair` dataclass is
the portable in-memory form, and :func:`fingerprint_of` defines the
cross-call dedup key (SHA-256 hex of the SSH wire-format public blob).
"""

from __future__ import annotations

import base64
import dataclasses
import hashlib
import os
import re
from pathlib import Path

from cryptography.hazmat.primitives.asymmetric import ed25519, rsa
from cryptography.hazmat.primitives.serialization import (
    Encoding,
    NoEncryption,
    PrivateFormat,
    PublicFormat,
    load_ssh_private_key,
    load_ssh_public_key,
)

from .db import CredentialDB, SSHKeyRecord

# ── Constants ───────────────────────────────────────────────────────────────

DEFAULT_RSA_BITS = 3072
"""Matches the ``ssh-keygen`` default as of OpenSSH 9.x."""

PRIVATE_KEY_MODE = 0o600
PUBLIC_KEY_MODE = 0o644

_PASSPHRASE_HINT = re.compile(r"(encrypted|password|passphrase)", re.IGNORECASE)
"""Message substrings cryptography uses when a private key is passphrase-protected.

Detection drives :class:`PasswordProtectedKeyError` translation so malformed
non-protected PEMs keep bubbling up as plain ``ValueError``.
"""


# ── Domain types ────────────────────────────────────────────────────────────


@dataclasses.dataclass(frozen=True, slots=True)
class GeneratedKeypair:
    """A keypair in the portable bytes form the vault stores."""

    key_type: str
    private_pem: bytes
    public_blob: bytes
    public_line: str
    comment: str
    fingerprint: str


@dataclasses.dataclass(frozen=True, slots=True)
class ImportResult:
    """Outcome of importing an OpenSSH keypair into the DB."""

    key_id: int
    fingerprint: str
    comment: str
    already_present: bool


@dataclasses.dataclass(frozen=True, slots=True)
class ExportResult:
    """Paths written by :func:`export_ssh_keypair`."""

    key_id: int
    fingerprint: str
    private_path: Path
    public_path: Path


class PasswordProtectedKeyError(ValueError):
    """Raised when an imported private key is encrypted with a passphrase."""


class KeypairMismatchError(ValueError):
    """Raised when provided public and private keys disagree."""


# ── Generation ──────────────────────────────────────────────────────────────


def generate_keypair(key_type: str, *, comment: str) -> GeneratedKeypair:
    """Generate a fresh keypair entirely in memory.

    Args:
        key_type: ``"ed25519"`` or ``"rsa"``.
        comment: Comment to embed in the public line.  Surfaces in
            ``ssh-add -L`` output and drives the signer's ``tk-main:``
            promotion heuristic.
    """
    if key_type == "ed25519":
        private_key = ed25519.Ed25519PrivateKey.generate()
    elif key_type == "rsa":
        private_key = rsa.generate_private_key(public_exponent=65537, key_size=DEFAULT_RSA_BITS)
    else:
        raise ValueError(f"Unsupported key type: {key_type!r} (expected ed25519 or rsa)")

    private_pem = private_key.private_bytes(
        encoding=Encoding.PEM,
        format=PrivateFormat.OpenSSH,
        encryption_algorithm=NoEncryption(),
    )
    public_key = private_key.public_key()
    public_blob, public_line = _serialize_public(public_key, comment=comment)
    return GeneratedKeypair(
        key_type=key_type,
        private_pem=private_pem,
        public_blob=public_blob,
        public_line=public_line,
        comment=comment,
        fingerprint=fingerprint_of(public_blob),
    )


# ── Import ──────────────────────────────────────────────────────────────────


def import_ssh_keypair(
    db: CredentialDB,
    scope: str,
    priv_path: Path,
    pub_path: Path | None = None,
    comment: str | None = None,
) -> ImportResult:
    """Read a keypair from OpenSSH files and assign it to *scope*.

    The public key is optional; when omitted it is derived from the private
    key.  When both are given they must match — fingerprint mismatch raises
    :class:`KeypairMismatchError`.  Password-protected private keys raise
    :class:`PasswordProtectedKeyError`; strip the passphrase with
    ``ssh-keygen -p`` and retry.
    """
    priv_bytes = priv_path.read_bytes()
    pub_bytes = pub_path.read_bytes() if pub_path else None
    parsed = parse_openssh_keypair(priv_bytes, pub_bytes, comment_override=comment)

    already = db.get_ssh_key_by_fingerprint(parsed.fingerprint) is not None
    key_id = db.store_ssh_key(
        key_type=parsed.key_type,
        private_pem=parsed.private_pem,
        public_blob=parsed.public_blob,
        comment=parsed.comment,
        fingerprint=parsed.fingerprint,
    )
    db.assign_ssh_key(scope, key_id)
    return ImportResult(
        key_id=key_id,
        fingerprint=parsed.fingerprint,
        comment=parsed.comment,
        already_present=already,
    )


def parse_openssh_keypair(
    priv_bytes: bytes,
    pub_bytes: bytes | None = None,
    *,
    comment_override: str | None = None,
) -> GeneratedKeypair:
    """Parse raw OpenSSH bytes into the canonical :class:`GeneratedKeypair` form.

    Passphrase-protected keys raise :class:`PasswordProtectedKeyError`.
    Cryptography signals that condition with either ``TypeError`` ("Password
    was not given but private key is encrypted" on older releases) or
    ``ValueError`` (newer releases), depending on version — we catch both
    and only translate when the message mentions encryption/password.
    Malformed non-protected PEMs keep bubbling up as plain ``ValueError``.
    """
    try:
        private_key = load_ssh_private_key(priv_bytes, password=None)
    except (TypeError, ValueError) as exc:
        if isinstance(exc, TypeError) or _PASSPHRASE_HINT.search(str(exc)):
            raise PasswordProtectedKeyError(
                "private key is passphrase-protected; run `ssh-keygen -p -f <file>` to strip it"
            ) from exc
        raise

    key_type = _classify_key(private_key)
    private_pem = private_key.private_bytes(
        encoding=Encoding.PEM,
        format=PrivateFormat.OpenSSH,
        encryption_algorithm=NoEncryption(),
    )
    derived_blob, _derived_line = _serialize_public(private_key.public_key(), comment="")

    if pub_bytes is None:
        public_blob = derived_blob
        pub_comment = ""
    else:
        public_blob, pub_comment = _parse_public_line(pub_bytes)
        if public_blob != derived_blob:
            raise KeypairMismatchError("public key does not match private key")

    comment = comment_override if comment_override is not None else pub_comment
    algo = _algo_name(key_type)
    public_line = f"{algo} {base64.b64encode(public_blob).decode('ascii')} {comment}".rstrip()
    return GeneratedKeypair(
        key_type=key_type,
        private_pem=private_pem,
        public_blob=public_blob,
        public_line=public_line,
        comment=comment,
        fingerprint=fingerprint_of(public_blob),
    )


# ── Export ──────────────────────────────────────────────────────────────────


def export_ssh_keypair(
    db: CredentialDB,
    scope: str,
    out_dir: Path,
    key_id: int | None = None,
    out_name: str | None = None,
) -> ExportResult:
    """Write a scope's key back out as a standard OpenSSH file pair.

    The directory is allowed to contain unrelated files; only the *output*
    files are protected with ``O_EXCL`` so nothing gets silently clobbered.
    Default filename stem: ``id_<keytype>_<fp8>`` where ``fp8`` is the
    first eight hex chars of the fingerprint — stable, collision-safe
    across scopes sharing an ``out_dir``.
    """
    record = _pick_key_for_export(db, scope, key_id)
    stem = out_name or f"id_{record.key_type}_{record.fingerprint[:8]}"
    out_dir = out_dir.expanduser().resolve()
    out_dir.mkdir(parents=True, exist_ok=True)
    priv_path = out_dir / stem
    pub_path = out_dir / f"{stem}.pub"

    _write_exclusive(priv_path, record.private_pem, PRIVATE_KEY_MODE)
    _write_exclusive(pub_path, (public_line_of(record) + "\n").encode("utf-8"), PUBLIC_KEY_MODE)

    return ExportResult(
        key_id=record.id,
        fingerprint=record.fingerprint,
        private_path=priv_path,
        public_path=pub_path,
    )


# ── Projections on an SSH key ───────────────────────────────────────────────


def fingerprint_of(public_blob: bytes) -> str:
    """Return the canonical fingerprint — hex SHA-256 of the SSH wire blob."""
    return hashlib.sha256(public_blob).hexdigest()


def public_line_of(record: SSHKeyRecord) -> str:
    """Render *record* as the one-line OpenSSH public key form.

    Format: ``<algo> <base64-blob> <comment>`` — matches what
    ``ssh-keygen`` writes to ``.pub`` files and what a remote's deploy-key
    field expects.  Callers that rendered this inline now go through this
    single helper so the algo-name mapping lives in one place.
    """
    algo = _algo_name(record.key_type)
    b64 = base64.b64encode(record.public_blob).decode("ascii")
    return f"{algo} {b64} {record.comment}".rstrip()


# ── Private helpers ─────────────────────────────────────────────────────────


def _serialize_public(public_key, *, comment: str) -> tuple[bytes, str]:
    """Render *public_key* as ``(wire_blob, one_line_text)`` with *comment* appended."""
    one_line = public_key.public_bytes(encoding=Encoding.OpenSSH, format=PublicFormat.OpenSSH)
    parts = one_line.decode("ascii").split(None, 2)
    if len(parts) < 2:
        raise ValueError("cryptography returned malformed OpenSSH public key")
    algo, b64 = parts[0], parts[1]
    blob = base64.b64decode(b64)
    public_line = f"{algo} {b64} {comment}".rstrip()
    return blob, public_line


def _parse_public_line(pub_bytes: bytes) -> tuple[bytes, str]:
    """Parse a ``.pub`` file's ``<algo> <base64> [comment]`` form."""
    text = pub_bytes.decode("utf-8", errors="replace").strip()
    parts = text.split(None, 2)
    if len(parts) < 2:
        raise ValueError("malformed public key file: expected '<type> <base64> [comment]'")
    blob = base64.b64decode(parts[1])
    comment = parts[2] if len(parts) > 2 else ""
    # Sanity: cryptography accepts the blob as a public key (catches corruption).
    load_ssh_public_key(f"{parts[0]} {parts[1]}".encode("ascii"))
    return blob, comment


def _classify_key(private_key) -> str:
    """Return ``"ed25519"`` or ``"rsa"`` for a decoded private key object."""
    if isinstance(private_key, ed25519.Ed25519PrivateKey):
        return "ed25519"
    if isinstance(private_key, rsa.RSAPrivateKey):
        return "rsa"
    raise ValueError(f"unsupported key type: {type(private_key).__name__}")


def _algo_name(key_type: str) -> str:
    """Return the SSH protocol algorithm name for a key type."""
    return "ssh-ed25519" if key_type == "ed25519" else "ssh-rsa"


def _pick_key_for_export(db: CredentialDB, scope: str, key_id: int | None) -> SSHKeyRecord:
    """Resolve which of the scope's keys to export."""
    records = db.load_ssh_keys_for_scope(scope)
    if not records:
        raise ValueError(f"scope {scope!r} has no SSH keys assigned")
    if key_id is None:
        return records[-1]  # most recently assigned
    for r in records:
        if r.id == key_id:
            return r
    raise ValueError(f"key_id {key_id} is not assigned to scope {scope!r}")


def _write_exclusive(path: Path, data: bytes, mode: int) -> None:
    """Create *path* with ``O_EXCL`` and *mode*; refuses to overwrite."""
    fd = os.open(str(path), os.O_WRONLY | os.O_CREAT | os.O_EXCL, mode)
    try:
        os.write(fd, data)
    finally:
        os.close(fd)
    os.chmod(path, mode)  # honor explicit mode even under a restrictive umask
