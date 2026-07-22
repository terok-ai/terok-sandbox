# SPDX-FileCopyrightText: 2026 Jiri Vyskocil
# SPDX-License-Identifier: Apache-2.0

"""Linux kernel-keyring binding for the volatile passphrase cache tier.

A thin, dependency-free ``ctypes`` wrapper over ``libkeyutils.so.1`` —
the userspace shim for the kernel key-retention service
(``add_key(2)`` / ``keyctl(2)``).  It exposes exactly the four
operations the vault's volatile passphrase cache needs, and nothing
else:

- [`store`][terok_sandbox.vault.store.kernel_keyring.store] — stash the
  SQLCipher passphrase as a ``user``-type key in the caller's **user
  keyring** (``@u``) and lock its permission mask to the owning uid;
- [`load`][terok_sandbox.vault.store.kernel_keyring.load] — read it
  back from any same-uid process;
- [`forget`][terok_sandbox.vault.store.kernel_keyring.forget] — unlink
  it early;
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
- *Anchor ``@u`` (the user keyring), not the persistent keyring.*  The
  file this tier replaces lived under ``$XDG_RUNTIME_DIR``, which logind
  wipes on final logout — so its effective lifetime already *was* the
  user keyring's lifetime (per-uid, shared across every same-uid
  terminal, torn down at logout).  ``@u`` is the semantic drop-in; the
  persistent keyring would over-deliver (survive logout) and needs
  ``keyctl_get_persistent`` machinery we deliberately avoid.
- *Explicit ``keyctl_setperm``.*  A fresh ``user`` key defaults to
  ``possessor=all, uid=view`` — the uid can *see* the key but not read
  or search it.  systemd gets away without a setperm because its readers
  possess ``@u`` through a shared session keyring; our CLI in a
  *different* terminal does not possess the supervisor's key, so it
  would fall to the uid class and be unable to find or read it.  We
  therefore open uid ``view|read|write|search|setattr`` and zero the
  group/other classes — no other user can read it, and any same-uid
  terminal can read, refresh, or revoke.
- *No auto-expiry.*  The cache persists for the whole login session —
  until an explicit ``vault lock`` (or a move to a durable tier), just
  like the tmpfs file it replaces — rather than timing out mid-session.
  The payload lives in unswappable kernel memory, so it never reaches
  disk or swap regardless.

Linux-only: on any host without the kernel key facility
(``CONFIG_KEYS`` off, no ``libkeyutils``, WSL1, non-Linux) every entry
point degrades to "unavailable" and the tier simply drops out of the
resolution chain, exactly like systemd-creds on a systemd < 257 box.
"""

from __future__ import annotations

import ctypes
import ctypes.util
import logging
import os
from typing import Final

_logger = logging.getLogger(__name__)

#: Well-known ``(type, description)`` the key is stored under.  Every
#: same-uid process finds the cached passphrase by searching ``@u`` for
#: this exact description — keep it stable across releases.
KEY_TYPE: Final = b"user"
KEY_DESCRIPTION: Final = b"terok-sandbox:vault-passphrase"

#: ``KEY_SPEC_USER_KEYRING`` from ``linux/keyctl.h`` — the special id
#: that resolves to the caller's per-uid user keyring (``@u``).
_KEY_SPEC_USER_KEYRING: Final = -4

#: Permission mask applied right after the key is created
#: (``keyctl_setperm``).  Nibbles, high→low: possessor · user(uid) ·
#: group · other.  ``0x3f`` = all six bits (view·read·write·search·
#: link·setattr); ``0x2f`` = all except ``link`` (``0x10``).
#:
#: - possessor ``0x3f`` — the creating process keeps full control;
#: - uid ``0x2f`` — any same-uid terminal may view/read (open the DB),
#:   write/setattr (``vault lock`` revoke + timeout refresh), and search
#:   (locate it), but **not** link it elsewhere;
#: - group ``0x00`` / other ``0x00`` — no other user can even see it.
_KEY_PERM: Final = 0x3F2F0000

#: Payloads over this are refused before hitting the kernel — a vault
#: passphrase is tens of bytes; anything near the 32 KiB ``user``-key
#: ceiling means a caller bug, not a real secret.
_MAX_PAYLOAD_BYTES: Final = 4096


class _KeyutilsUnavailable(Exception):
    """``libkeyutils`` could not be loaded or the facility is absent."""


def _load_library() -> ctypes.CDLL:
    """Return a configured ``libkeyutils`` handle, or raise ``_KeyutilsUnavailable``.

    Prefers ``ctypes.util.find_library`` (walks the ``ld.so`` cache) and
    falls back to the ``.so.1`` soname directly — the runtime library is
    present wherever the containers stack is, even when the ``-dev``
    package (and the bare ``libkeyutils.so`` symlink ``find_library``
    needs) is not installed.
    """
    soname = ctypes.util.find_library("keyutils") or "libkeyutils.so.1"
    try:
        lib = ctypes.CDLL(soname, use_errno=True)
    except OSError as exc:
        raise _KeyutilsUnavailable(f"libkeyutils not loadable ({exc})") from exc

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
    lib.keyctl_unlink.restype = ctypes.c_long
    lib.keyctl_unlink.argtypes = [ctypes.c_int32, ctypes.c_int32]
    lib.keyctl_get_keyring_ID.restype = ctypes.c_int32
    lib.keyctl_get_keyring_ID.argtypes = [ctypes.c_int32, ctypes.c_int32]
    return lib


def unavailable_reason() -> str | None:
    """Return why the kernel keyring can't be used here, or ``None`` if it can.

    The setup/probe gate — mirrors
    [`systemd_creds.unavailable_reason`][terok_sandbox.vault.store.systemd_creds.unavailable_reason]
    so the frontends decide whether to *offer* the tier the same way for
    both.  Side-effect free: it only resolves the id of the (already
    existing) user keyring, creating and reading nothing.  A ``None``
    return means "usable"; any string is a human reason a setup chooser
    can surface.
    """
    try:
        lib = _load_library()
    except _KeyutilsUnavailable as exc:
        return str(exc)
    # keyctl_get_keyring_ID(@u, create=0): resolves the user keyring's
    # real serial without creating anything.  ENOSYS ⇒ kernel built
    # without CONFIG_KEYS (or a syscall-translation layer like WSL1);
    # any other failure ⇒ the tier can't run here.
    ctypes.set_errno(0)
    if lib.keyctl_get_keyring_ID(_KEY_SPEC_USER_KEYRING, 0) != -1:
        return None
    errno = ctypes.get_errno()
    if errno == 38:  # ENOSYS
        return "kernel built without keyring support (CONFIG_KEYS)"
    return f"user keyring unreachable ({os.strerror(errno)})"


def store(passphrase: str) -> bool:
    """Cache *passphrase* in the user keyring; return success.

    Adds (or, if it already exists, updates in place — so the footprint
    stays a single long-lived key rather than one per unlock) a
    ``user``-type key under [`KEY_DESCRIPTION`][terok_sandbox.vault.store.kernel_keyring.KEY_DESCRIPTION]
    in ``@u`` and tightens its permission mask to
    [`_KEY_PERM`][terok_sandbox.vault.store.kernel_keyring._KEY_PERM].
    No timeout is armed: the cache persists for the whole login session
    until an explicit ``vault lock`` (or a move to a durable tier),
    matching the tmpfs file it replaces.  Any failure (facility
    unavailable, quota ``EDQUOT``, permission fault) is logged at WARNING
    and returns ``False`` so the caller falls through — this is a cache,
    never the sole home of the secret.

    Refuses an empty passphrase: an empty key would read back as
    SQLCipher's no-encryption sentinel.
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

    ctypes.set_errno(0)
    serial = lib.add_key(KEY_TYPE, KEY_DESCRIPTION, payload, len(payload), _KEY_SPEC_USER_KEYRING)
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


def load() -> str | None:
    """Return the cached passphrase, or ``None`` if it's absent/unreachable.

    Searches ``@u`` for the well-known key and reads its payload.  A
    missing key (``ENOKEY``) or an unavailable facility yields ``None`` —
    the normal "locked" outcome, kept silent so the resolver can fall
    through to the next tier.
    """
    try:
        lib = _load_library()
    except _KeyutilsUnavailable:
        return None

    ctypes.set_errno(0)
    serial = lib.keyctl_search(_KEY_SPEC_USER_KEYRING, KEY_TYPE, KEY_DESCRIPTION, 0)
    if serial == -1:
        return None
    # Two-pass read: NULL/0 returns the payload length, then fill a
    # right-sized buffer.  Passphrases are tiny, but this stays correct
    # without guessing a size.
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
        # Scrub the ctypes buffer; the decoded str is out of our hands.
        ctypes.memset(buf, 0, length)


def forget() -> bool:
    """Unlink the cached passphrase from the user keyring; return success.

    Used by ``vault lock`` / relock.  A key that isn't there (already
    expired or never stored) counts as success — the desired end state
    is "no cached passphrase".  Works from any same-uid terminal because
    [`_KEY_PERM`][terok_sandbox.vault.store.kernel_keyring._KEY_PERM]
    grants the uid class ``write``.
    """
    try:
        lib = _load_library()
    except _KeyutilsUnavailable:
        return True

    ctypes.set_errno(0)
    serial = lib.keyctl_search(_KEY_SPEC_USER_KEYRING, KEY_TYPE, KEY_DESCRIPTION, 0)
    if serial == -1:
        return True  # nothing to forget
    if lib.keyctl_unlink(serial, _KEY_SPEC_USER_KEYRING) == -1:
        _logger.warning("kernel keyring keyctl_unlink failed: %s", os.strerror(ctypes.get_errno()))
        return False
    return True


__all__ = [
    "KEY_DESCRIPTION",
    "KEY_TYPE",
    "forget",
    "load",
    "store",
    "unavailable_reason",
]
