# SPDX-FileCopyrightText: 2026 Jiri Vyskocil
# SPDX-License-Identifier: Apache-2.0

"""SSH keypair generation, import, and export against the credential DB.

The DB is the canonical home for SSH keys.  This module moves material in
and out of that form:

- :func:`generate_keypair` creates a fresh keypair in memory.
- :func:`import_ssh_keypair` reads an existing OpenSSH keypair from files
  and registers it against a scope.
- :func:`export_ssh_keypair` writes a scope's key back to an OpenSSH file
  pair for handing to tools that cannot use the SSH agent.

Internally, private keys live in the DB as unencrypted PKCS#8 DER — the
single, opaque binary form the signer loads directly via
``load_der_private_key``.  Import converts any supported inbound PEM to
that form at the boundary, and export re-armors it as OpenSSH PEM.

All flows share one vocabulary: the :class:`GeneratedKeypair` dataclass is
the portable in-memory form, and :func:`fingerprint_of` defines the
cross-call dedup key — the standard OpenSSH ``SHA256:<base64>`` fingerprint
of the SSH wire-format public blob.
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
    load_der_private_key,
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

_UNSAFE_COMMENT_CHARS = re.compile(r"[\x00-\x1F\x7F]")
"""Any C0 control character or DEL.  Newlines break the one-line public-key
contract; ESC (``\\x1B``) enables terminal-escape output spoofing (CWE-150);
the rest have no legitimate place in an SSH key comment."""

_MAX_COMMENT_LEN = 200
"""Bound embedded comments so a pathological input can't bloat every
listing/export/stream indefinitely."""


# ── Domain types ────────────────────────────────────────────────────────────


@dataclasses.dataclass(frozen=True, slots=True)
class GeneratedKeypair:
    """A keypair in the portable bytes form the vault stores.

    ``private_der`` is unencrypted PKCS#8 DER — the raw-binary form we
    persist and feed straight to the signer.  The public half stays in its
    usual SSH wire-format blob plus a pre-rendered ``public_line``.
    """

    key_type: str
    private_der: bytes
    public_blob: bytes
    public_line: str
    comment: str
    fingerprint: str


@dataclasses.dataclass(frozen=True, slots=True)
class ImportResult:
    """Outcome of importing an OpenSSH keypair into the DB.

    ``already_present`` reflects whether the *key* (by fingerprint) was
    already in the ``ssh_keys`` table.  ``scope_was_assigned`` reflects
    whether the *scope* already owned a link to that key before this
    call.  The two combine into four honest call outcomes: minted +
    linked, minted + re-linked (can't happen), re-used + linked (the
    common "multi-scope share" path), and re-used + no-op.
    """

    key_id: int
    fingerprint: str
    comment: str
    already_present: bool
    scope_was_assigned: bool


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


class UnsafeCommentError(ValueError):
    """Raised when a comment contains control characters or is too long.

    Comments flow into SSH ``authorized_keys`` lines, public-line rendering,
    ``ssh-add -L`` output, and terminal summaries — so embedded newlines or
    escape sequences could break the wire format or spoof terminal output.
    Rejection happens at the storage entry points; every display site then
    trusts the DB to hold only safe strings.
    """


def _require_safe_comment(comment: str) -> str:
    """Validate *comment* and return it unchanged; raise on unsafe input."""
    if not isinstance(comment, str):
        raise UnsafeCommentError(f"comment must be a string, got {type(comment).__name__}")
    if len(comment) > _MAX_COMMENT_LEN:
        raise UnsafeCommentError(
            f"comment exceeds {_MAX_COMMENT_LEN}-character limit ({len(comment)} chars)"
        )
    match = _UNSAFE_COMMENT_CHARS.search(comment)
    if match:
        raise UnsafeCommentError(
            f"comment contains disallowed control character "
            f"\\x{ord(match.group(0)):02x} at position {match.start()}"
        )
    return comment


# ── Generation ──────────────────────────────────────────────────────────────


def generate_keypair(key_type: str, *, comment: str) -> GeneratedKeypair:
    """Generate a fresh keypair entirely in memory.

    Args:
        key_type: ``"ed25519"`` or ``"rsa"``.
        comment: Comment to embed in the public line.  Surfaces in
            ``ssh-add -L`` output and drives the signer's ``tk-main:``
            promotion heuristic.  Rejected with :class:`UnsafeCommentError`
            if it contains control characters or exceeds the length limit.
    """
    _require_safe_comment(comment)
    if key_type == "ed25519":
        private_key = ed25519.Ed25519PrivateKey.generate()
    elif key_type == "rsa":
        private_key = rsa.generate_private_key(public_exponent=65537, key_size=DEFAULT_RSA_BITS)
    else:
        raise ValueError(f"Unsupported key type: {key_type!r} (expected ed25519 or rsa)")

    private_der = _serialize_private_der(private_key)
    public_key = private_key.public_key()
    public_blob, public_line = _serialize_public(public_key, comment=comment)
    return GeneratedKeypair(
        key_type=key_type,
        private_der=private_der,
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
    :class:`PasswordProtectedKeyError`; the library stays diagnostic-only,
    so callers own the remediation hint they render to the user.
    """
    priv_bytes = priv_path.read_bytes()
    pub_bytes = pub_path.read_bytes() if pub_path else None
    parsed = parse_openssh_keypair(priv_bytes, pub_bytes, comment_override=comment)

    existing_row = db.get_ssh_key_by_fingerprint(parsed.fingerprint)
    already = existing_row is not None
    scope_was_assigned = already and any(
        r.id == existing_row.id for r in db.list_ssh_keys_for_scope(scope)
    )

    key_id = db.store_ssh_key(
        key_type=parsed.key_type,
        private_der=parsed.private_der,
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
        scope_was_assigned=scope_was_assigned,
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
            raise PasswordProtectedKeyError("private key is passphrase-protected") from exc
        raise

    key_type = _classify_key(private_key)
    private_der = _serialize_private_der(private_key)
    derived_blob, _derived_line = _serialize_public(private_key.public_key(), comment="")

    if pub_bytes is None:
        public_blob = derived_blob
        pub_comment = ""
    else:
        public_blob, pub_comment = _parse_public_line(pub_bytes)
        if public_blob != derived_blob:
            raise KeypairMismatchError("public key does not match private key")

    comment = comment_override if comment_override is not None else pub_comment
    _require_safe_comment(comment)
    algo = _algo_name(key_type)
    public_line = f"{algo} {base64.b64encode(public_blob).decode('ascii')} {comment}".rstrip()
    return GeneratedKeypair(
        key_type=key_type,
        private_der=private_der,
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

    The private bytes come out of the DB as PKCS#8 DER; this function
    re-armors them as OpenSSH PEM — the same format ``ssh-keygen`` writes
    and that ``ssh -i`` consumes.  The directory is allowed to contain
    unrelated files; only the *output* files are protected with ``O_EXCL``
    so nothing gets silently clobbered.  Default filename stem:
    ``id_<keytype>_<fp8>`` where ``fp8`` is the first eight hex chars of
    the raw SHA-256 digest of the public blob — stable and format-agnostic
    for the user-facing fingerprint string.
    """
    record = _pick_key_for_export(db, scope, key_id)
    stem = _sanitize_out_name(out_name) or f"id_{record.key_type}_{_short_id(record.public_blob)}"
    out_dir = out_dir.expanduser().resolve()
    out_dir.mkdir(parents=True, exist_ok=True)
    priv_path = out_dir / stem
    pub_path = out_dir / f"{stem}.pub"

    _write_exclusive(priv_path, openssh_pem_of(record.private_der), PRIVATE_KEY_MODE)
    try:
        _write_exclusive(
            pub_path,
            (public_line_of(record) + "\n").encode("utf-8"),
            PUBLIC_KEY_MODE,
        )
    except BaseException:
        # Don't leave a lone private key on disk if the matching public
        # write failed; the user would have no way to identify which
        # scope it belongs to without the companion ``.pub`` file.
        priv_path.unlink(missing_ok=True)
        raise

    return ExportResult(
        key_id=record.id,
        fingerprint=record.fingerprint,
        private_path=priv_path,
        public_path=pub_path,
    )


# ── Projections on an SSH key ───────────────────────────────────────────────


def fingerprint_of(public_blob: bytes) -> str:
    """Return the canonical OpenSSH fingerprint of *public_blob*.

    Format matches what ``ssh-keygen -lf``, ``ssh-add -l``, GitHub's UI,
    and ``gh ssh-key list`` all print: ``SHA256:<base64-unpadded>`` over
    the raw SHA-256 digest of the SSH wire-format public blob.
    """
    digest = hashlib.sha256(public_blob).digest()
    return f"SHA256:{base64.b64encode(digest).decode('ascii').rstrip('=')}"


def _short_id(public_blob: bytes) -> str:
    """8-char hex stem for file naming, independent of fingerprint display format."""
    return hashlib.sha256(public_blob).hexdigest()[:8]


def openssh_pem_of(private_der: bytes) -> bytes:
    """Re-armor a stored PKCS#8 DER blob as OpenSSH PEM — the on-disk wire format.

    This is what ``ssh-keygen`` writes and what ``ssh -i`` reads.  Exposed for
    the CLI export path and for test fixtures that need to round-trip a key
    through the OpenSSH-file form.
    """
    key = load_der_private_key(private_der, password=None)
    return key.private_bytes(
        encoding=Encoding.PEM,
        format=PrivateFormat.OpenSSH,
        encryption_algorithm=NoEncryption(),
    )


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


def _serialize_private_der(private_key) -> bytes:
    """Serialize *private_key* as unencrypted PKCS#8 DER — the on-disk form."""
    return private_key.private_bytes(
        encoding=Encoding.DER,
        format=PrivateFormat.PKCS8,
        encryption_algorithm=NoEncryption(),
    )


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
    """Return the SSH protocol algorithm name for a stored key type."""
    if key_type == "ed25519":
        return "ssh-ed25519"
    if key_type == "rsa":
        return "ssh-rsa"
    raise ValueError(f"unsupported key type: {key_type!r}")


def _sanitize_out_name(out_name: str | None) -> str | None:
    """Reject out-of-directory stems; ``None`` means "use the default"."""
    if not out_name:
        return None
    if out_name in {".", ".."} or Path(out_name).name != out_name:
        raise ValueError(f"out_name must be a bare filename stem, not a path: {out_name!r}")
    return out_name


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
    """Create *path* with ``O_EXCL`` and *mode*, writing *data* atomically-ish.

    POSIX lets ``os.write`` short-write even for regular files, so loop
    until every byte lands.  If any write or the final ``chmod`` fails,
    the partially-written file is unlinked so no truncated key material
    survives.
    """
    fd = os.open(str(path), os.O_WRONLY | os.O_CREAT | os.O_EXCL, mode)
    try:
        view = memoryview(data)
        offset = 0
        while offset < len(view):
            written = os.write(fd, view[offset:])
            if written <= 0:
                raise OSError(f"os.write made no progress at offset {offset}")
            offset += written
        os.close(fd)
        fd = -1
        # Honor explicit mode even under a restrictive umask (0022 etc.).
        os.chmod(path, mode)
    except BaseException:
        if fd >= 0:
            os.close(fd)
        path.unlink(missing_ok=True)
        raise
