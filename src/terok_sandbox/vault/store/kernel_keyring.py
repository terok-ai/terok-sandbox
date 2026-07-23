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
  terminal can read, revoke, or update it.  Applying that mask needs
  the writer to *possess* the key, so ``store`` first links ``@u`` into
  the session keyring (a headless supervisor / cron / CI has no
  pam_keyinit possession otherwise, and the setperm would fail EACCES).
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
import functools
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

#: ``KEY_SPEC_SESSION_KEYRING`` (``@s``).  We link ``@u`` into it before
#: writing so the process *possesses* the new key (see ``store``).
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


def store(passphrase: str) -> bool:
    """Cache *passphrase* so later processes of this user can unlock the vault.

    The cache is deliberately untimed: it lives for the login session and
    is cleared only by an explicit ``vault lock`` or a move to a durable
    tier, matching the tmpfs file this tier replaces.  Failure is soft —
    a cache is never the sole home of the secret — so an unreachable
    facility, an exhausted key quota or a refused permission change is
    logged and reported rather than raised.

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

    # Ensure this process *possesses* the key it is about to create, so
    # the keyctl_setperm below is permitted.  A fresh key grants the
    # possessor everything but the uid only ``view`` (0x3f010000); on a
    # host without a pam_keyinit-linked session keyring — a headless
    # supervisor, cron, CI — the process does not possess ``@u`` and so
    # falls to that uid class, and setperm (which needs ``setattr``)
    # fails EACCES.  Linking ``@u`` into the session keyring makes its
    # keys possessed for this process; it is an idempotent no-op where a
    # login session already did it.  Best-effort: if it fails, the
    # setperm below simply fails as it would have anyway.
    lib.keyctl_link(_KEY_SPEC_USER_KEYRING, _KEY_SPEC_SESSION_KEYRING)

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
    """Return the cached passphrase.

    Silent on every miss: an absent key and an unusable facility are
    both the ordinary "locked" outcome, which the next tier of the
    resolver chain handles.  Reach for
    [`is_cached`][terok_sandbox.vault.store.kernel_keyring.is_cached]
    when only presence matters — this materialises the secret.

    Returns:
        The cached passphrase, or None when nothing is cached here.
    """
    try:
        lib = _load_library()
    except _KeyutilsUnavailable:
        return None

    serial = _find_cached_key(lib)
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


def forget() -> bool:
    """Clear the cached passphrase.

    Backs ``vault lock``.  An already-absent key counts as success: the
    contract is the end state — nothing cached here — not the act of
    removing something.  Any same-uid terminal may call it, not only the
    one that cached the passphrase.

    Returns:
        True when no passphrase remains cached.
    """
    try:
        lib = _load_library()
    except _KeyutilsUnavailable:
        return True

    serial = _find_cached_key(lib)
    if serial is None:
        return True
    if lib.keyctl_unlink(serial, _KEY_SPEC_USER_KEYRING) == -1:
        _logger.warning("kernel keyring keyctl_unlink failed: %s", os.strerror(ctypes.get_errno()))
        return False
    return True


def is_cached() -> bool:
    """Whether a passphrase is currently cached here.

    The presence question every status surface asks — ``vault status``,
    the doctor checks, the TUI pill's poll — answered without reading
    the payload, so reporting *on* the secret never materialises it.

    Returns:
        True when the well-known key exists in the user keyring.
    """
    try:
        lib = _load_library()
    except _KeyutilsUnavailable:
        return False
    return _find_cached_key(lib) is not None


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


# ── Key lookup and library binding (private) ────────────────────────


def _find_cached_key(lib: ctypes.CDLL) -> int | None:
    """Serial of the cached passphrase key, or ``None`` when absent.

    Returns:
        The key's serial number, or None when the user keyring holds no
        key under the well-known description.
    """
    ctypes.set_errno(0)
    serial = lib.keyctl_search(_KEY_SPEC_USER_KEYRING, KEY_TYPE, KEY_DESCRIPTION, 0)
    return None if serial == -1 else serial


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
    "KEY_DESCRIPTION",
    "KEY_TYPE",
    "forget",
    "is_cached",
    "load",
    "store",
    "unavailable_reason",
]
