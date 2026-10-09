# SPDX-FileCopyrightText: 2026 Jiri Vyskocil
# SPDX-License-Identifier: Apache-2.0

"""Linux kernel-keyring binding for the volatile passphrase cache tier.

A thin, dependency-free ``ctypes`` wrapper over ``libkeyutils.so.1`` —
the userspace shim for the kernel key-retention service
(``add_key(2)`` / ``keyctl(2)``).  It exposes exactly the four
operations the vault's volatile passphrase cache needs, and nothing
else:

- [`store`][terok_sandbox.vault.store.kernel_keyring.store] — cache the
  SQLCipher passphrase for this uid;
- [`load`][terok_sandbox.vault.store.kernel_keyring.load] — read it back
  from any same-uid process;
- [`forget`][terok_sandbox.vault.store.kernel_keyring.forget] — clear it;
- [`is_cached`][terok_sandbox.vault.store.kernel_keyring.is_cached] —
  answer the status surfaces' presence question without materialising
  the secret;
- [`unavailable_reason`][terok_sandbox.vault.store.kernel_keyring.unavailable_reason] —
  the setup/probe gate, mirroring
  [`terok_sandbox.vault.store.systemd_creds.unavailable_reason`][terok_sandbox.vault.store.systemd_creds.unavailable_reason].

**Why these exact choices** (see the prior-art survey on the tier — the
kernel keyring is how systemd-ask-password and MIT Kerberos cache
passphrases):

- *Key type ``user``, not ``logon``.*  The passphrase must be read back
  to open SQLCipher; ``logon`` payloads are unreadable from userspace by
  anyone, for any permission mask.  ``user`` is the only workable type
  (the same choice cryptsetup's readback path and eCryptfs are forced
  into).
- *Anchor ``@u`` (the kernel user keyring), not the kernel persistent keyring.*
  The cache is shared across same-uid processes in the operator's user
  namespace. Its lifetime depends on processes and references, not
  logout alone; remaining references can keep it alive after logout.
  It never survives a reboot. No kernel persistent-keyring retention is needed.
- *Read it from the operator's own user namespace, nowhere else.*  A
  kernel user keyring is per user *namespace*: a process inside podman's
  rootless namespace resolves ``@u`` to its own empty kernel keyring, and the
  cache is invisible there however the permissions read.  So this tier
  serves a supervisor that runs as a user unit of the operator's
  systemd manager, in the operator's namespaces; a supervisor inside the
  container namespace gets the session-file backing instead
  ([`session_cache`][terok_sandbox.vault.store.session_cache] chooses,
  by the same fact the OCI hook reads). No bridge through the kernel
  session keyring: that reader would depend on which login cached the passphrase.
- *Explicit ``keyctl_setperm``.*  A fresh ``user`` key defaults to
  ``possessor=all, uid=view`` — the uid can *see* the key but not read
  or search it.  systemd gets away without a setperm because its readers
  possess ``@u`` through a shared kernel session keyring; our CLI in a
  *different* terminal does not possess the supervisor's key, so it
  would fall to the uid class and be unable to find or read it.  We
  therefore open uid ``view|read|write|search|setattr`` and zero the
  group/other classes — no other user can read it, and any same-uid
  terminal can read, revoke, or update it.  Applying that mask needs
  the writer to *possess* the key, so ``store`` first links ``@u`` into
  the kernel session keyring (a headless supervisor / cron / CI has no
  pam_keyinit possession otherwise, and the setperm would fail EACCES).
- *No auto-expiry.*  The cache does not time out mid-session.
  ``vault lock`` or a move to a durable tier explicitly clears it;
  reboot or loss of references also removes it. The payload lives in
  unswappable kernel memory, not a desktop keyring or a disk file.

Linux-only: on any host without the kernel key facility
(``CONFIG_KEYS`` off, no ``libkeyutils``, WSL1, non-Linux) every entry
point degrades to "unavailable" and the tier simply drops out of the
resolution chain, exactly like systemd-creds on a systemd < 257 box.
"""

from __future__ import annotations

import ctypes
import ctypes.util
import errno
import functools
import hashlib
import logging
import os
import socket
from typing import Final

_logger = logging.getLogger(__name__)

#: The ``user`` key type: readable back from userspace (``logon`` is not),
#: so it can hand the SQLCipher passphrase to a later reader.
KEY_TYPE: Final = b"user"

#: Prefix of the per-vault ``@u`` key description; the full description
#: appends a digest of ``(hostname, credentials-DB path)`` (see
#: [`key_description`][terok_sandbox.vault.store.kernel_keyring.key_description]).
#: Scoping by DB path keeps every vault a uid can reach — a side-by-side
#: install, a test's throwaway tmp DB, the operator's real vault — in its
#: own key, so one process never reads or clears another's passphrase even
#: though ``@u`` is shared across the whole uid.  Folding in the hostname
#: separates environments that share one ``@u`` yet differ by UTS namespace
#: — concurrent rootless containers with identical in-container paths — so
#: a cached passphrase never leaks between them.  Keep the format stable
#: across releases: the writer and every later reader must agree.
KEY_DESCRIPTION_PREFIX: Final = b"terok-sandbox:vault-passphrase:"

#: ``KEY_SPEC_USER_KEYRING`` from ``linux/keyctl.h`` — the special id
#: that resolves to the caller's per-uid kernel user keyring (``@u``).
_KEY_SPEC_USER_KEYRING: Final = -4

#: ``KEY_SPEC_SESSION_KEYRING`` (``@s``).  ``store`` links ``@u`` into it
#: before writing so the process *possesses* the key it is about to set
#: permissions on; nothing reads through it.
_KEY_SPEC_SESSION_KEYRING: Final = -3

#: Permission mask applied right after the key is created
#: (``keyctl_setperm``).  Nibbles, high→low: possessor · user(uid) ·
#: group · other.  ``0x3f`` = all six bits (view·read·write·search·
#: link·setattr); ``0x2f`` = all except ``link`` (``0x10``).
#:
#: - possessor ``0x3f`` — the creating process keeps full control;
#: - uid ``0x2f`` — any same-uid terminal may view/read (open the DB),
#:   write/setattr (``vault lock`` revoke, re-``store`` update), and
#:   search (locate it), but **not** link it elsewhere;
#: - group ``0x00`` / other ``0x00`` — no other user can even see it.
_KEY_PERM: Final = 0x3F2F0000

#: Payloads over this are refused before hitting the kernel — a vault
#: passphrase is tens of bytes; anything near the 32 KiB ``user``-key
#: ceiling means a caller bug, not a real secret.
_MAX_PAYLOAD_BYTES: Final = 4096


def key_description(db_path: str | os.PathLike[str]) -> bytes:
    """Return the ``@u`` key description scoping the cache to one vault on this host.

    Anchored on ``(hostname, absolute credentials-DB path)`` and hashed to
    a fixed-width, ASCII-safe token appended to
    [`KEY_DESCRIPTION_PREFIX`][terok_sandbox.vault.store.kernel_keyring.KEY_DESCRIPTION_PREFIX]:
    two vaults on one uid never collide, and a path carrying spaces or
    non-UTF-8 bytes can't corrupt the description.  The hostname component
    separates environments that share one ``@u`` but differ by UTS
    namespace (concurrent rootless containers with identical in-container
    paths); the path component keeps a test's throwaway DB off the
    operator's real key even on the same host.  ``abspath`` (not
    ``realpath``) and ``gethostname`` keep this pure and stable — the
    writer and every same-host reader derive both from the same config and
    the same UTS namespace, so they always agree.
    """
    return KEY_DESCRIPTION_PREFIX + cache_digest(db_path).encode("ascii")


def cache_digest(db_path: str | os.PathLike[str]) -> str:
    """Return the per-vault cache token every backing scopes its entry with.

    [`key_description`][terok_sandbox.vault.store.kernel_keyring.key_description]
    composes it into the ``@u`` description;
    [`session_file`][terok_sandbox.vault.store.session_file] uses it as
    the cache file name.  One derivation, so the backings always agree.
    """
    ident = f"{socket.gethostname()}\0{os.path.abspath(db_path)}"
    return hashlib.sha256(ident.encode("utf-8")).hexdigest()[:32]


def store(passphrase: str, db_path: str | os.PathLike[str]) -> bool:
    """Cache *passphrase* for *db_path* so later processes can unlock that vault.

    The cache has no timeout, but reboot or loss of references removes
    it. An explicit ``vault lock`` or a move to a durable tier clears it
    sooner. An unreachable facility, an exhausted key quota or a refused
    permission change is logged and reported as a failed write.

    Returns:
        True when the passphrase is cached and readable by this uid.

    Raises:
        ValueError: The passphrase is empty — SQLCipher reads that back
            as "no encryption" — or implausibly large for a passphrase.
    """
    if not passphrase:
        raise ValueError("refusing to cache an empty passphrase in the kernel keyring")
    payload = passphrase.encode("utf-8")
    if len(payload) > _MAX_PAYLOAD_BYTES:
        raise ValueError(f"passphrase exceeds {_MAX_PAYLOAD_BYTES} bytes — refusing to cache")
    try:
        lib = _load_library()
    except _KeyutilsUnavailable as exc:
        _logger.warning("kernel keyring unavailable, not caching passphrase: %s", exc)
        return False

    # Possession first: a fresh key grants the possessor everything but
    # the uid only ``view`` (0x3f010000), and on a host without a
    # pam_keyinit-linked kernel session keyring — a headless supervisor, cron,
    # CI — this process does not possess ``@u``, so the keyctl_setperm
    # below (which needs ``setattr``) would fail EACCES.  Idempotent
    # where a login session already linked it.
    ctypes.set_errno(0)
    if lib.keyctl_link(_KEY_SPEC_USER_KEYRING, _KEY_SPEC_SESSION_KEYRING) == -1:
        _logger.warning("kernel keyring @u -> @s link failed: %s", os.strerror(ctypes.get_errno()))

    ctypes.set_errno(0)
    serial = lib.add_key(
        KEY_TYPE, key_description(db_path), payload, len(payload), _KEY_SPEC_USER_KEYRING
    )
    if serial == -1:
        _logger.warning("kernel keyring add_key failed: %s", os.strerror(ctypes.get_errno()))
        return False
    # Lock the mask down before anything can race a read on the default
    # (uid-view-only) permissions.
    if lib.keyctl_setperm(serial, _KEY_PERM) == -1:
        _logger.warning("kernel keyring keyctl_setperm failed: %s", os.strerror(ctypes.get_errno()))
        lib.keyctl_unlink(serial, _KEY_SPEC_USER_KEYRING)
        return False
    return True


def load(db_path: str | os.PathLike[str]) -> str | None:
    """Return the passphrase cached for *db_path*.

    Silent on every miss: an absent key and an unusable facility are
    both the ordinary "locked" outcome, which the next tier of the
    resolver chain handles.  Reach for
    [`is_cached`][terok_sandbox.vault.store.kernel_keyring.is_cached]
    when only presence matters — this materialises the secret.

    Returns:
        The cached passphrase, or None when nothing is cached for this vault.
    """
    try:
        lib = _load_library()
    except _KeyutilsUnavailable:
        return None

    try:
        serial = _find_cached_key(lib, key_description(db_path))
    except OSError as exc:
        _logger.warning("kernel keyring search failed: %s", exc)
        return None
    if serial is None:
        return None
    # Sized by a first pass so the buffer is never a guess.
    length = lib.keyctl_read(serial, None, 0)
    if length <= 0:
        return None
    buf = ctypes.create_string_buffer(length)
    got = lib.keyctl_read(serial, buf, length)
    if got <= 0:
        return None
    try:
        return buf.raw[:got].decode("utf-8") or None
    finally:
        # The decoded str is out of our hands; this buffer is not.
        ctypes.memset(buf, 0, length)


def forget(db_path: str | os.PathLike[str]) -> bool:
    """Clear the passphrase cached for *db_path*.

    Backs ``vault lock``.  An already-absent key counts as success: the
    contract is the end state — nothing cached for this vault — not the
    act of removing something.  A lookup that *fails* is not that end
    state, so it reports failure rather than claim the passphrase is
    gone.  Any same-uid terminal may call it, not only the one that
    cached the passphrase.

    The key is anchored in ``@u`` and this unlinks it from there, so
    clearing the cache is an operator-context operation.

    Returns:
        True when no passphrase remains cached for this vault.
    """
    try:
        lib = _load_library()
    except _KeyutilsUnavailable:
        return True

    try:
        serial = _find_cached_key(lib, key_description(db_path))
    except OSError as exc:
        _logger.warning("kernel keyring search failed, cannot confirm removal: %s", exc)
        return False
    if serial is None:
        return True
    if lib.keyctl_unlink(serial, _KEY_SPEC_USER_KEYRING) == -1:
        _logger.warning("kernel keyring keyctl_unlink failed: %s", os.strerror(ctypes.get_errno()))
        return False
    return True


def is_cached(db_path: str | os.PathLike[str]) -> bool:
    """Whether a passphrase is currently cached for *db_path*.

    The presence question every status surface asks — ``vault status``,
    the doctor checks, the TUI pill's poll — answered without reading
    the payload, so reporting *on* the secret never materialises it.

    Returns:
        True when this vault's key exists in the kernel user keyring.
    """
    try:
        lib = _load_library()
    except _KeyutilsUnavailable:
        return False
    try:
        return _find_cached_key(lib, key_description(db_path)) is not None
    except OSError as exc:
        _logger.warning("kernel keyring search failed: %s", exc)
        return False


def unavailable_reason() -> str | None:
    """Explain why this host cannot hold the cache, or ``None`` if it can.

    The gate the setup chooser and the status surfaces consult before
    *offering* the tier, mirroring
    [`systemd_creds.unavailable_reason`][terok_sandbox.vault.store.systemd_creds.unavailable_reason]
    so both tiers are gated alike.  A probe, not a guarantee —
    [`store`][terok_sandbox.vault.store.kernel_keyring.store]'s return
    value is the definitive answer — and it neither creates nor reads a
    key.

    Returns:
        A human-readable reason the tier is unusable here, or None when
        it is usable.
    """
    try:
        lib = _load_library()
    except _KeyutilsUnavailable as exc:
        return str(exc)
    # keyctl_get_keyring_ID(@u, create=0): resolves the kernel user keyring's
    # real serial without creating anything.  ENOSYS ⇒ kernel built
    # without CONFIG_KEYS (or a syscall-translation layer like WSL1);
    # any other failure ⇒ the tier can't run here.
    ctypes.set_errno(0)
    if lib.keyctl_get_keyring_ID(_KEY_SPEC_USER_KEYRING, 0) != -1:
        return None
    err = ctypes.get_errno()
    if err == errno.ENOSYS:
        return "kernel keyring support unavailable (CONFIG_KEYS)"
    return f"kernel keyring (@u) unreachable ({os.strerror(err)})"


# ── Key lookup and library binding (private) ────────────────────────

#: Search results that answer "no usable key here", as opposed to "the
#: lookup did not complete".  A revoked or expired key can never yield a
#: passphrase again, so every caller wants exactly the answer it would
#: get for a key that was never stored: ``forget`` reports success,
#: because its contract is the end state, and nobody warns, because a
#: retry cannot change the verdict.
_MISS_ERRNOS: Final = frozenset({errno.ENOKEY, errno.EKEYEXPIRED, errno.EKEYREVOKED})


def _find_cached_key(lib: ctypes.CDLL, description: bytes) -> int | None:
    """Serial of the key under *description* in ``@u``, or ``None`` when genuinely absent.

    A miss is any answer that means *no usable key here*: absent
    (``ENOKEY``), revoked, or expired (see ``_MISS_ERRNOS``).  Every
    other failure — a permission fault, most of all — is a lookup that
    did not complete, and the caller must not read it as "absent": a
    ``forget`` that did so would report the passphrase cleared while it
    may still be cached.

    Returns:
        The key's serial number, or None when no usable key exists.

    Raises:
        OSError: The search failed for a reason other than a missing key.
    """
    ctypes.set_errno(0)
    serial = lib.keyctl_search(_KEY_SPEC_USER_KEYRING, KEY_TYPE, description, 0)
    if serial != -1:
        return serial
    err = ctypes.get_errno()
    if err in _MISS_ERRNOS:
        return None
    raise OSError(err, os.strerror(err))


class _KeyutilsUnavailable(Exception):
    """``libkeyutils`` could not be loaded or the facility is absent."""


@functools.cache
def _load_library() -> ctypes.CDLL:
    """Return a configured ``libkeyutils`` handle, or raise ``_KeyutilsUnavailable``.

    Prefers ``ctypes.util.find_library`` (walks the ``ld.so`` cache) and
    falls back to the ``.so.1`` soname directly — the runtime library is
    present wherever the containers stack is, even when the ``-dev``
    package (and the bare ``libkeyutils.so`` symlink ``find_library``
    needs) is not installed.

    Cached: the handle and its ``argtypes``/``restype`` registrations are
    process-stable, and ``vault status`` probes the tier a few times per
    render.  ``functools.cache`` does not memoise the exception, so a
    failed load is simply retried on the next call.  Both a load failure
    (``OSError``) and a missing/ABI-mismatched symbol (``AttributeError``
    from binding a function this ``libkeyutils`` doesn't export) degrade
    to ``_KeyutilsUnavailable`` — the module's "drops out of the chain"
    contract must hold even against a wrong library on the ``ld.so`` path.
    """
    soname = ctypes.util.find_library("keyutils") or "libkeyutils.so.1"
    try:
        lib = ctypes.CDLL(soname, use_errno=True)
        # key_serial_t is a signed 32-bit int; key_perm_t an unsigned 32-bit.
        lib.add_key.restype = ctypes.c_int32
        lib.add_key.argtypes = [
            ctypes.c_char_p,
            ctypes.c_char_p,
            ctypes.c_void_p,
            ctypes.c_size_t,
            ctypes.c_int32,
        ]
        lib.keyctl_search.restype = ctypes.c_int32
        lib.keyctl_search.argtypes = [
            ctypes.c_int32,
            ctypes.c_char_p,
            ctypes.c_char_p,
            ctypes.c_int32,
        ]
        lib.keyctl_read.restype = ctypes.c_long
        lib.keyctl_read.argtypes = [ctypes.c_int32, ctypes.c_char_p, ctypes.c_size_t]
        lib.keyctl_setperm.restype = ctypes.c_long
        lib.keyctl_setperm.argtypes = [ctypes.c_int32, ctypes.c_uint32]
        lib.keyctl_link.restype = ctypes.c_long
        lib.keyctl_link.argtypes = [ctypes.c_int32, ctypes.c_int32]
        lib.keyctl_unlink.restype = ctypes.c_long
        lib.keyctl_unlink.argtypes = [ctypes.c_int32, ctypes.c_int32]
        lib.keyctl_get_keyring_ID.restype = ctypes.c_int32
        lib.keyctl_get_keyring_ID.argtypes = [ctypes.c_int32, ctypes.c_int32]
    except OSError as exc:
        raise _KeyutilsUnavailable(f"libkeyutils not loadable ({exc})") from exc
    except AttributeError as exc:
        raise _KeyutilsUnavailable(f"libkeyutils missing expected symbol ({exc})") from exc
    return lib


__all__ = [
    "KEY_DESCRIPTION_PREFIX",
    "KEY_TYPE",
    "forget",
    "is_cached",
    "key_description",
    "load",
    "store",
    "unavailable_reason",
]
