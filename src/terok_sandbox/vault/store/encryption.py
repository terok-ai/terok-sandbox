# SPDX-FileCopyrightText: 2026 Jiri Vyskocil
# SPDX-License-Identifier: Apache-2.0

"""Passphrase plumbing and SQLCipher helpers for at-rest credential encryption.

Walks the five-tier resolution chain — systemd-creds → desktop keyring →
session cache → ``passphrase_command`` helper → interactive prompt —
and exposes the SQLCipher open / rekey / migrate primitives the rest
of the package builds on.  The tier vocabulary lives in
[`tiers`][terok_sandbox.vault.store.tiers]; ``resolve_passphrase``
documents the chain order; ``open_sqlcipher`` is the only entry point
that ever calls ``sqlcipher3.connect``.

The setup-time plaintext→SQLCipher migration (deprecated in 0.8.0,
removed in 0.9.0) lives at the bottom of the file; nothing in the
runtime chain touches it.
"""

from __future__ import annotations

import logging
import os
import secrets
import shlex
import sqlite3
import subprocess  # nosec B404 — operator-supplied passphrase_command helper — operator-supplied passphrase_command helper + systemd-creds
import sys
from concurrent.futures import ThreadPoolExecutor
from contextlib import closing, contextmanager
from dataclasses import dataclass
from functools import partial
from pathlib import Path
from typing import TYPE_CHECKING, Any

from . import session_cache as _session_cache, systemd_creds as _systemd_creds
from .tiers import PassphraseTier

if TYPE_CHECKING:
    from collections.abc import Callable, Iterator

    from keyring.backend import KeyringBackend

DESKTOP_KEYRING_SERVICE = "terok-sandbox"
DESKTOP_KEYRING_USERNAME = "credentials-db"
_SECRET_SERVICE_NO_PROMPT = "/"  # nosec B105 — D-Bus null object path, not a secret

#: ``token_urlsafe(32)`` ≈ 43 chars of URL-safe Base64 — 256 bits of
#: entropy from a 62-char alphabet plus ``-``/``_``, both shell-safe.
_GENERATED_PASSPHRASE_BYTES = 32

#: Wall-clock budget for a `passphrase_command` helper before the
#: resolver gives up.  Generous enough for the slow cloud CLIs
#: (``aws secretsmanager``, ``gcloud secrets``, ``az keyvault``) on a
#: cold cache, tight enough that a wedged helper doesn't pin the daemon
#: start.
_PASSPHRASE_COMMAND_TIMEOUT_S = 30.0

_logger = logging.getLogger(__name__)


class NoPassphraseError(RuntimeError):
    """No SQLCipher passphrase resolved — the DB cannot be opened."""


class WrongPassphraseError(RuntimeError):
    """SQLCipher could not decrypt the DB — passphrase doesn't match its encryption key."""


# ── Resolution chain ────────────────────────────────────────────────


def open_sqlcipher_via_chain(
    db_path: str | Path,
    *,
    systemd_creds_file: Path | None = None,
    use_desktop_keyring: bool = False,
    passphrase_command: str | None = None,
    prompt_on_tty: bool = False,
    **connect_kwargs: Any,
) -> Any:
    """Resolve the passphrase through the runtime chain and open *db_path*.

    Raises [`NoPassphraseError`][terok_sandbox.vault.store.encryption.NoPassphraseError]
    when the chain yields nothing.  *prompt_on_tty* turns on the
    interactive fallback for CLI consumers; daemons leave it ``False``.
    """
    passphrase = resolve_passphrase(
        credentials_db=db_path,
        systemd_creds_file=systemd_creds_file,
        use_desktop_keyring=use_desktop_keyring,
        passphrase_command=passphrase_command,
        prompt_on_tty=prompt_on_tty,
    )
    if passphrase is None:
        raise NoPassphraseError(f"no SQLCipher passphrase available for {db_path}")
    return open_sqlcipher(db_path, passphrase, **connect_kwargs)


def resolve_passphrase_with_source(
    *,
    credentials_db: str | Path,
    systemd_creds_file: Path | None = None,
    use_desktop_keyring: bool = False,
    passphrase_command: str | None = None,
    prompt_on_tty: bool = False,
) -> tuple[str | None, PassphraseTier | None]:
    """Walk the runtime resolution chain; return ``(passphrase, source)``.

    Single source of truth for the resolution order — see
    [`resolve_passphrase`][terok_sandbox.vault.store.encryption.resolve_passphrase]
    for the tier semantics.  Both elements of the tuple are ``None``
    when no tier had a passphrase.

    *credentials_db* is the vault the passphrase is *for*: it scopes the
    session-cache lookup to that DB's key (see
    [`kernel_keyring.key_description`][terok_sandbox.vault.store.kernel_keyring.key_description]),
    so the cache for one vault never resolves another's — pass the same
    path the caller is about to open.

    The source half feeds a TUI/CLI status display — keep the labels
    stable, callers dispatch on them.
    """
    # Truthy checks throughout: an empty string anywhere in the chain
    # is SQLCipher's no-encryption sentinel; treat it as "not present"
    # rather than letting it overrule a real later tier.
    if systemd_creds_file is not None and systemd_creds_file.is_file():
        sealed_pw = _systemd_creds.unseal(systemd_creds_file)
        if sealed_pw:
            return sealed_pw, PassphraseTier.SYSTEMD_CREDS
        # Fail closed: silently falling through would demote a
        # machine-bound tier to desktop keyring / plaintext-on-disk without
        # the operator's knowledge.
        raise WrongPassphraseError(
            f"sealed systemd-creds credential present at {systemd_creds_file}"
            " but could not be unsealed"
        )
    if use_desktop_keyring:
        desktop_keyring_pw = load_passphrase_from_desktop_keyring(allow_prompt=prompt_on_tty)
        if desktop_keyring_pw:
            return desktop_keyring_pw, PassphraseTier.DESKTOP_KEYRING
    # Volatile unlock cache, below the zero-friction durable tiers and
    # above the helper: fail-*open* like the desktop keyring above it — an
    # absent or expired key falls through rather than masking the
    # durable ``passphrase_command`` beneath.  Read via the module
    # namespace so tests can monkeypatch the session cache away.
    session_pw = _session_cache.load(credentials_db)
    if session_pw:
        return session_pw, PassphraseTier.SESSION_CACHE
    if passphrase_command:
        cmd_pw = load_passphrase_from_command(passphrase_command)
        if cmd_pw:
            return cmd_pw, PassphraseTier.PASSPHRASE_COMMAND
        # Fail closed for the same reason as systemd-creds above; the
        # command string itself is omitted because operators sometimes
        # inline AWS ARNs / vault paths there and this exception reaches
        # doctor output and journals.
        raise WrongPassphraseError(
            "passphrase_command produced no passphrase; run it manually to diagnose"
            " (see WARNING in the vault journal)"
        )
    if prompt_on_tty and sys.stdin.isatty():
        return prompt_passphrase(), PassphraseTier.PROMPT
    return None, None


def resolve_passphrase(
    *,
    credentials_db: str | Path,
    systemd_creds_file: Path | None = None,
    use_desktop_keyring: bool = False,
    passphrase_command: str | None = None,
    prompt_on_tty: bool = False,
) -> str | None:
    """Walk the runtime resolution chain; return ``None`` if nothing has it.

    Order:

    1. *systemd_creds_file* — sealed credential decrypted via
       ``systemd-creds(1)``.  Machine-bound (TPM2 or host key), survives
       reboot, no desktop keyring required.  See
       [`terok_sandbox.vault.store.systemd_creds`][terok_sandbox.vault.store.systemd_creds].
    2. Desktop keyring — only when *use_desktop_keyring* is true; off by default because
       Linux Secret Service grants access per-collection, not per-item.
    3. Session cache — the volatile unlock cache
       ([`terok_sandbox.vault.store.session_cache`][terok_sandbox.vault.store.session_cache]):
       the kernel keyring, or a tmpfs session file where the kernel
       facility is unusable.  Consulted unconditionally (an
       absent/expired/unavailable cache just yields ``None``);
       positioned so it never shadows a zero-friction durable tier
       above but still spares re-running the helper below.
    4. *passphrase_command* — operator-supplied shell command
       (``pass show …``, ``bw get``, ``op read``, cloud secret-manager
       CLIs).  Delegates retrieval without per-backend integration code,
       same shape as ``git config credential.helper`` or
       ``BORG_PASSCOMMAND``.  Configured-but-broken fails closed so a
       misbehaving helper can't silently demote security to a weaker tier.
    5. Interactive prompt — only when *prompt_on_tty* and ``sys.stdin.isatty()``.

    *passphrase_command* is threaded through as a parameter rather than
    read here so this module stays free of any dependency on the
    sandbox config layer — the config module already imports from
    credentials.db, and the back-edge would close a tach cycle.
    """
    passphrase, _source = resolve_passphrase_with_source(
        credentials_db=credentials_db,
        systemd_creds_file=systemd_creds_file,
        use_desktop_keyring=use_desktop_keyring,
        passphrase_command=passphrase_command,
        prompt_on_tty=prompt_on_tty,
    )
    return passphrase


@dataclass(frozen=True)
class TierPresence:
    """Whether one passphrase-chain tier currently holds material — for ``vault status``.

    A diagnostic, non-short-circuiting counterpart to
    [`resolve_passphrase_with_source`][terok_sandbox.vault.store.encryption.resolve_passphrase_with_source]:
    that walker stops at the first tier that resolves, so it can only
    ever name the *winner*.  ``vault status`` needs the whole chain to
    show every tier that currently holds material, not just the one that
    would unlock the vault.
    """

    source: PassphraseTier
    present: bool
    detail: str


def probe_passphrase_chain(
    *,
    credentials_db: str | Path,
    systemd_creds_file: Path | None = None,
    use_desktop_keyring: bool = False,
    passphrase_command: str | None = None,
) -> tuple[TierPresence, ...]:
    """Report per-tier presence across the resolution chain without short-circuiting.

    Presence is judged from *material on hand*, not by resolving the
    secret: the sealed systemd-creds credential is never unsealed and
    the ``passphrase_command`` is never executed (both can be slow or
    have side effects), so their mere configuration counts as present.
    The desktop-keyring and session-cache tiers are cheap to read, so those are
    probed for real.  Tiers appear in resolution order; the first
    ``present`` one is the tier that would unlock the vault.  The
    interactive ``prompt`` tier is omitted — it stores nothing, so it
    can never be "present".
    """
    # Presence only — the status chain reports *that* a tier holds
    # material, never its value, so this must not read the passphrase.
    session_cached = _session_cache.is_cached(credentials_db)
    return (
        TierPresence(
            PassphraseTier.SYSTEMD_CREDS,
            bool(systemd_creds_file and systemd_creds_file.is_file()),
            _systemd_creds_detail(systemd_creds_file),
        ),
        TierPresence(
            PassphraseTier.DESKTOP_KEYRING,
            # Truthy, not ``is not None``: an empty string is the resolver's
            # "no passphrase" sentinel, so status must treat it as absent too.
            use_desktop_keyring and bool(load_passphrase_from_desktop_keyring()),
            "desktop keyring" if use_desktop_keyring else "use_desktop_keyring off",
        ),
        TierPresence(
            PassphraseTier.SESSION_CACHE,
            session_cached,
            _session_cache.backing_detail(cached=session_cached),
        ),
        TierPresence(
            PassphraseTier.PASSPHRASE_COMMAND,
            bool(passphrase_command),
            "configured (not executed)" if passphrase_command else "not configured",
        ),
    )


def _systemd_creds_detail(path: Path | None) -> str:
    """Human detail for the systemd-creds tier in the ``vault status`` chain.

    The tier row used to print the bare configured path regardless of
    whether anything was sealed or whether the tier could even run here,
    so an absent credential looked identical to a present-but-outranked
    one, and a host too old for the non-root ``--user`` path (systemd
    < 257, no TPM needed) was listed as if it were a live option. This
    separates the three states the path blurred together:

    - unconfigured → ``not configured`` (matches the other tiers' phrasing);
    - configured but nothing sealed → ``not sealed (<path>)``;
    - sealed/configured but the tier can't run on this host →
      the path plus ``— unusable here: <reason>``.

    Uses the cheap, cached availability probe
    ([`unavailable_reason`][terok_sandbox.vault.store.systemd_creds.unavailable_reason]) —
    it never unseals and has no side effects, so it's safe on the
    diagnostic path that the rest of ``probe_passphrase_chain`` keeps
    free of secret resolution.
    """
    if path is None:
        return "not configured"
    base = str(path) if path.is_file() else f"not sealed ({path})"
    reason = _systemd_creds.unavailable_reason()
    return f"{base} — unusable here: {reason}" if reason else base


# ── Tier primitives ─────────────────────────────────────────────────


#: Ceiling on one desktop-keyring read.  A healthy backend answers in
#: milliseconds.  Only a wedged D-Bus round-trip comes near this limit.
_DESKTOP_KEYRING_READ_TIMEOUT_S = 3.0


def load_passphrase_from_desktop_keyring(*, allow_prompt: bool = False) -> str | None:
    """Return the desktop-keyring passphrase, or ``None`` when a read cannot succeed.

    A read from a locked Secret Service collection triggers the
    desktop's unlock dialog and waits for the answer.  That wait is
    legitimate exactly once: an interactive caller (*allow_prompt*) on
    a host with a graphical session, where the operator sees the dialog
    and answers or cancels it.  Every other read — a status probe, a
    poll, a daemon, a headless or SSH session — skips a locked
    collection instead of waiting on a dialog nobody sees.  A timeout
    guards the promptless read, so a misbehaving backend degrades this
    tier instead of freezing its caller.
    """

    interactive = allow_prompt and _graphical_session_present()

    def _read() -> str | None:
        import keyring  # noqa: PLC0415

        backend = keyring.get_keyring()
        if interactive:
            return backend.get_password(DESKTOP_KEYRING_SERVICE, DESKTOP_KEYRING_USERNAME)
        return _read_desktop_keyring_backend(backend)

    # The probe is always bounded — ``dbus_init`` can block exactly like
    # the read, and a wedged D-Bus must not block any caller.  Only the
    # interactive read is unbounded: the unlock dialog may legitimately
    # wait on the operator, and a timeout would cancel a dialog they
    # are looking at.
    try:
        if not interactive:
            return _call_with_timeout(_read, _DESKTOP_KEYRING_READ_TIMEOUT_S)
        blocked = _call_with_timeout(
            lambda: desktop_keyring_read_blocked(allow_prompt=True),
            _DESKTOP_KEYRING_READ_TIMEOUT_S,
        )
        if blocked is not None:
            return None
        return _read()
    except Exception:  # noqa: BLE001 — timeout or backend error: the tier degrades
        return None


def desktop_keyring_read_blocked(*, allow_prompt: bool = False) -> str | None:
    """Explain why a desktop-keyring read would block or fail, or ``None`` when safe.

    A Secret Service read from a locked collection can trigger a D-Bus
    unlock prompt that waits for a desktop dialog.  This probe reads the
    lock state without a prompt.  A locked collection does not block an
    *allow_prompt* read when a graphical session is present — the operator answers the
    dialog.  A probe error (no D-Bus session, no Secret Service daemon)
    also makes the tier unusable and returns a reason.  The generic
    libsecret backend cannot disable implicit unlocking, so background
    reads reject it.  Other configured backends retain their own semantics.
    """
    try:
        import keyring  # noqa: PLC0415
        from keyring.backends.SecretService import Keyring as _SecretService  # noqa: PLC0415

        for backend in _desktop_keyring_backends(keyring.get_keyring()):
            if not (allow_prompt and _graphical_session_present()):
                _reject_implicit_unlock_backend(backend)
            if isinstance(backend, _SecretService):
                with _secret_service_collection(backend) as collection:
                    if (
                        collection is not None
                        and collection.is_locked()
                        and not (allow_prompt and _graphical_session_present())
                    ):
                        return (
                            "desktop keyring locked "
                            "(unlock it in a desktop session, or use another tier)"
                        )
    except Exception as exc:  # noqa: BLE001
        return f"desktop keyring unreachable ({type(exc).__name__})"
    return None


def _desktop_keyring_backends(backend: KeyringBackend) -> Iterator[KeyringBackend]:
    """Yield concrete backends in the selected desktop keyring's resolution order."""
    from keyring.backends.chainer import ChainerBackend  # noqa: PLC0415

    if isinstance(backend, ChainerBackend):
        for member in backend.backends:
            yield from _desktop_keyring_backends(member)
    else:
        yield backend


@contextmanager
def _secret_service_collection(backend: KeyringBackend) -> Iterator[Any]:
    """Open the configured collection, or yield ``None`` if it does not exist.

    ``get_default_collection`` creates a missing collection and may prompt.
    Constructing ``Collection`` directly instead fails without side effects.
    """
    import secretstorage  # noqa: PLC0415

    with closing(secretstorage.dbus_init()) as connection:
        try:
            if hasattr(backend, "preferred_collection"):
                collection = secretstorage.Collection(connection, backend.preferred_collection)
            else:
                collection = secretstorage.Collection(connection)
        except secretstorage.exceptions.ItemNotFoundException:
            collection = None
        yield collection


def _read_desktop_keyring_backend(backend: KeyringBackend) -> str | None:
    """Read without Secret Service unlock prompts; propagate failures, not false absence.

    The generic Secret Service getter unlocks both collections and items.
    A prior lock probe cannot prevent a later lock race, so background reads
    use SecretStorage's non-unlocking primitives on the same credential query.
    """
    from keyring.backends.SecretService import Keyring as _SecretService  # noqa: PLC0415

    for member in _desktop_keyring_backends(backend):
        _reject_implicit_unlock_backend(member)
        if isinstance(member, _SecretService):
            with _secret_service_collection(member) as collection:
                if collection is None:
                    continue
                collection.ensure_not_locked()
                for item in collection.search_items(
                    member._query(DESKTOP_KEYRING_SERVICE, DESKTOP_KEYRING_USERNAME)
                ):
                    return item.get_secret().decode("utf-8")
        elif (
            passphrase := member.get_password(DESKTOP_KEYRING_SERVICE, DESKTOP_KEYRING_USERNAME)
        ) is not None:
            return passphrase
    return None


def _delete_desktop_keyring_backend(backend: KeyringBackend) -> str | None:
    """Delete without Secret Service prompts, or explain why confirmation is needed.

    SecretStorage's public ``Item.delete`` executes returned prompts.  Its
    transport is used here only to omit that interactive step; an unexecuted
    confirmation request cannot establish deletion and must report failure.
    """
    from keyring.backends.SecretService import Keyring as _SecretService  # noqa: PLC0415
    from keyring.errors import PasswordDeleteError  # noqa: PLC0415

    if not isinstance(backend, _SecretService):
        backend.delete_password(DESKTOP_KEYRING_SERVICE, DESKTOP_KEYRING_USERNAME)
        return None
    with _secret_service_collection(backend) as collection:
        if collection is not None:
            collection.ensure_not_locked()
            for item in collection.search_items(
                backend._query(DESKTOP_KEYRING_SERVICE, DESKTOP_KEYRING_USERNAME)
            ):
                item.ensure_not_locked()
                (prompt,) = item._item.call("Delete", "")
                if prompt != _SECRET_SERVICE_NO_PROMPT:
                    return "desktop keyring requires deletion confirmation"
                return None
    raise PasswordDeleteError("No such password")


def _reject_implicit_unlock_backend(backend: KeyringBackend) -> None:
    """Reject native libsecret reads that cannot suppress implicit unlock prompts."""
    if any(cls.__module__ == "keyring.backends.libsecret" for cls in type(backend).__mro__):
        raise RuntimeError("the libsecret backend does not support promptless access")


def _graphical_session_present() -> bool:
    """Return ``True`` when a display server can show the unlock dialog."""
    return bool(os.environ.get("DISPLAY") or os.environ.get("WAYLAND_DISPLAY"))


#: One worker serializes every bounded desktop-keyring access.  A wedged
#: call occupies the single slot; later calls wait in the queue for at
#: most their own timeout and then cancel out of it, so the process
#: never accumulates threads against one wedged backend.  Created on
#: first use and retired by
#: [`retire_desktop_keyring_worker`][terok_sandbox.vault.store.encryption.retire_desktop_keyring_worker],
#: so a process that is done reading can be single-threaded again.
_desktop_keyring_executor: ThreadPoolExecutor | None = None
_desktop_keyring_worker_wedged = False


def _desktop_keyring_worker() -> ThreadPoolExecutor:
    """The shared desktop-keyring worker, started on first use.

    A fresh worker holds no wedged call: the wedge belongs to the executor
    that was retired around it, so the flag clears with the new one.
    """
    global _desktop_keyring_executor, _desktop_keyring_worker_wedged
    if _desktop_keyring_executor is None:
        _desktop_keyring_executor = ThreadPoolExecutor(
            max_workers=1, thread_name_prefix="desktop-keyring"
        )
        _desktop_keyring_worker_wedged = False
    return _desktop_keyring_executor


def retire_desktop_keyring_worker() -> None:
    """Join the desktop-keyring worker so the process is single-threaded again.

    A process that confines itself after its passphrase chain needs
    this: Landlock below ABI 8 restricts one thread, and the helper
    refuses a process with two.  A worker abandoned in a wedged read
    cannot be joined and is left where it is — the confinement then
    reports the thread, which is the honest outcome.  The next read
    starts a fresh worker.
    """
    global _desktop_keyring_executor
    executor, _desktop_keyring_executor = _desktop_keyring_executor, None
    if executor is not None and not _desktop_keyring_worker_wedged:
        executor.shutdown(wait=True)


def _call_with_timeout(fn: Callable[[], str | None], timeout: float) -> str | None:
    """Run *fn* on the shared desktop-keyring worker; raise ``TimeoutError`` past *timeout*.

    The caller abandons an overrun call instead of killing it, because
    Python has no safe thread kill.  *fn* must therefore be a read with
    no state to corrupt.  A worker exception re-raises here, in the
    caller's thread.
    """
    global _desktop_keyring_worker_wedged
    future = _desktop_keyring_worker().submit(fn)
    try:
        return future.result(timeout)
    except TimeoutError:
        # A queued call leaves the queue; a running one is abandoned to
        # the single slot it already occupies.
        future.cancel()
        _desktop_keyring_worker_wedged = True
        _logger.warning("Desktop keyring access exceeded %.0fs; skipping the tier", timeout)
        raise


def desktop_keyring_backend_available() -> bool:
    """Return ``True`` iff a usable desktop keyring backend is reachable.

    Availability probe for setup frontends (the TUI tier chooser)
    deciding whether to *offer* the desktop-keyring tier at all.  A probe, not
    a guarantee — the definitive answer stays with
    [`store_passphrase_in_desktop_keyring`][terok_sandbox.vault.store.encryption.store_passphrase_in_desktop_keyring]'s
    return value at provisioning time.  The ``fail`` and ``null``
    backends both answer "no": they accept calls but hold nothing.
    """
    try:
        import keyring  # noqa: PLC0415
        from keyring.backends import fail, null  # noqa: PLC0415

        for backend in _desktop_keyring_backends(keyring.get_keyring()):
            if not isinstance(backend, (fail.Keyring, null.Keyring)):
                _reject_implicit_unlock_backend(backend)
                return True
        return False
    except Exception:  # noqa: BLE001
        return False


def store_passphrase_in_desktop_keyring(passphrase: str) -> bool:
    """Persist *passphrase* in the desktop keyring and verify promptless retrieval.

    ``True`` requires the same backend to return the exact value within
    the read timeout.  ``False`` does not prove that nothing was written:
    callers must preserve their source until verification succeeds.

    Refuses to store an empty value — SQLCipher interprets it as
    "no encryption", and a later resolve hit on a blank desktop-keyring entry
    would silently open the DB plaintext.
    """
    if not passphrase:
        raise ValueError("refusing to store an empty passphrase in the desktop keyring")
    try:
        import keyring  # noqa: PLC0415

        selected = keyring.get_keyring()
        for backend in _desktop_keyring_backends(selected):
            _reject_implicit_unlock_backend(backend)
            try:
                backend.set_password(DESKTOP_KEYRING_SERVICE, DESKTOP_KEYRING_USERNAME, passphrase)
            except NotImplementedError:
                continue
            stored = _call_with_timeout(
                partial(_read_desktop_keyring_backend, backend), _DESKTOP_KEYRING_READ_TIMEOUT_S
            )
            if stored is None or not secrets.compare_digest(
                stored.encode("utf-8"), passphrase.encode("utf-8")
            ):
                return False
            if selected is backend:
                return True
            resolved = _call_with_timeout(
                lambda: _read_desktop_keyring_backend(selected), _DESKTOP_KEYRING_READ_TIMEOUT_S
            )
            return resolved is not None and secrets.compare_digest(
                resolved.encode("utf-8"), passphrase.encode("utf-8")
            )
        return False
    except Exception:  # noqa: BLE001
        return False


def forget_passphrase_in_desktop_keyring() -> str | None:
    """Remove the desktop-keyring entry; return ``None`` only after verified absence.

    "Gone" covers a successful delete and an entry that never existed —
    the caller's goal is absence, not the delete call.  A locked desktop keyring
    cannot prove absence and never prompts here, so it returns its
    reason; so does a backend that rejects the delete while the entry
    still reads back.  Callers render the reason instead of guessing.
    """
    try:
        import keyring  # noqa: PLC0415
        from keyring.errors import PasswordDeleteError  # noqa: PLC0415

        if (
            blocked := _call_with_timeout(
                desktop_keyring_read_blocked, _DESKTOP_KEYRING_READ_TIMEOUT_S
            )
        ) is not None:
            return blocked
        selected = keyring.get_keyring()
        for backend in _desktop_keyring_backends(selected):
            try:
                if (reason := _delete_desktop_keyring_backend(backend)) is not None:
                    return reason
            except NotImplementedError:
                continue
            except PasswordDeleteError:
                pass
            break
        remaining = _call_with_timeout(
            lambda: _read_desktop_keyring_backend(selected), _DESKTOP_KEYRING_READ_TIMEOUT_S
        )
        return None if remaining is None else "the backend did not remove the passphrase"
    except Exception as exc:  # noqa: BLE001
        return f"desktop keyring unreachable ({type(exc).__name__})"


def load_passphrase_from_command(
    command: str, *, timeout: float = _PASSPHRASE_COMMAND_TIMEOUT_S
) -> str | None:
    """Run *command*, return its stdout with the trailing newline removed, or ``None`` on any failure.

    Same shape as the other tier primitives
    ([`load_passphrase_from_desktop_keyring`][terok_sandbox.vault.store.encryption.load_passphrase_from_desktop_keyring]):
    silent on every failure path so the resolver can decide whether
    ``None`` means "skip this tier" or "fail closed". Warnings report
    only the failure stage, exception type, exit code, or timeout.
    Command text, helper output, and exception messages may contain
    secrets and never enter the log.

    Same vocabulary as ``git config credential.helper``, ssh pinentry,
    ``BORG_PASSCOMMAND``: one field plugs any credential backend into
    the resolver — ``pass show …``, ``bw get password …``,
    ``op read op://…``, ``vault kv get -field=passphrase …``,
    ``aws secretsmanager get-secret-value …`` — without per-backend
    integration code in the sandbox.
    """
    try:
        argv = shlex.split(command)
    except ValueError as exc:
        _logger.warning("passphrase_command shlex parse failed (%s)", type(exc).__name__)
        return None
    if not argv:
        return None
    try:
        result = subprocess.run(  # noqa: S603 — argv is operator-configured  # nosec B603 — argv is a fixed list controlled by this module — argv is a fixed list controlled by this module
            argv, capture_output=True, text=True, timeout=timeout, check=False
        )
    except OSError as exc:
        _logger.warning("passphrase_command failed to spawn (%s)", type(exc).__name__)
        return None
    except UnicodeError as exc:
        _logger.warning("passphrase_command text conversion failed (%s)", type(exc).__name__)
        return None
    except subprocess.TimeoutExpired:
        _logger.warning("passphrase_command timed out after %.0fs", timeout)
        return None
    if result.returncode != 0:
        _logger.warning("passphrase_command exited %d", result.returncode)
        return None
    # rstrip only the line ending the helper appends — leading/trailing
    # whitespace inside the passphrase is legitimate secret material and
    # must reach SQLCipher verbatim.
    passphrase = result.stdout.rstrip("\r\n")
    return passphrase or None


def _write_to_controlling_tty(message: str, *, required: bool = True) -> None:
    """Write *message* to ``/dev/tty`` so a redirected stdout can't capture it.

    Fails closed by default when no controlling TTY is reachable (CI,
    headless automation): refuses rather than letting an irrecoverable
    generated passphrase fall on the floor.  Operators automating
    setup must either pre-provide the passphrase via a tier the
    resolver can find, or pass ``--echo-passphrase`` so the value
    reaches stdout — in which case the caller passes ``required=False``
    here and the missing-tty error becomes a silent skip.
    """
    try:
        with Path("/dev/tty").open("w", encoding="utf-8") as tty:
            tty.write(message)
    except OSError as exc:
        if not required:
            return
        raise SystemExit(
            "Refusing to print the generated vault passphrase: no controlling TTY"
            f" ({exc.strerror}).\n"
            "Re-run setup from an interactive terminal, or pre-provide the"
            " passphrase (vault unlock / credentials.passphrase / sealed credential)"
            " before re-running."
        ) from exc


def _read_from_controlling_tty(prompt: str) -> str | None:
    """Read a single line from ``/dev/tty`` after writing *prompt* to it.

    Mirrors [`_write_to_controlling_tty`][terok_sandbox.vault.store.encryption._write_to_controlling_tty]
    but in the opposite direction — used by the ack flow to ask the
    operator "type SAVED" on the same channel where the generated
    passphrase was just displayed, regardless of how stdin/stdout
    are redirected.  Returns ``None`` when no controlling TTY is
    reachable so callers can fall through to a no-ack path on truly
    headless runs (the announcement step has its own failure mode
    there).
    """
    try:
        with Path("/dev/tty").open("r+", encoding="utf-8") as tty:
            tty.write(prompt)
            tty.flush()
            return tty.readline().rstrip("\n")
    except OSError:
        return None


def prompt_passphrase(*, confirm: bool = False) -> str:
    """Read a passphrase from the controlling TTY with ``*``-masked echo.

    Mirrors the ``_prompt_api_key`` helper in [`terok_executor.credentials.auth`][terok_executor.credentials.auth]:
    ``prompt_toolkit.prompt(is_password=True)`` for the TTY path —
    proper terminal raw-mode handling, ``Ctrl+C`` raises
    ``KeyboardInterrupt`` cleanly, every character is masked.  Non-TTY
    input (e.g. ``terok-sandbox credentials encrypt-db < passphrase.txt``)
    falls back to a plain ``readline`` so pipe-fed automation still
    works.

    Empty entries are SQLCipher's no-encryption sentinel and never
    return a blank string.  In *confirm* mode (setup-time provisioning
    of a brand-new passphrase) hitting ``Enter`` is treated as
    "generate one for me": a fresh random passphrase is minted, echoed
    once so the operator can copy it out, and returned.  In single-shot
    mode (unlocking an existing DB) an empty entry raises — generating
    here would produce a wrong key that fails to decrypt the DB.
    """
    if sys.stdin.isatty():
        if confirm:
            typed = prompt_new_passphrase()
            if typed is not None:
                return typed
            # Empty + confirm = "mint one for me".  Write to
            # ``/dev/tty`` (not stdout) so a redirected install
            # — ``terok-sandbox setup > install.log`` or CI —
            # can't capture the recovery key.  ``commands._announce_generated_passphrase``
            # does the same thing for non-``prompt_passphrase``
            # paths; this is the foundation-layer mirror (we
            # can't import from the surface layer per tach).
            passphrase = generate_passphrase()
            _write_to_controlling_tty(
                f"\nVault passphrase: {passphrase}\n"
                "  Write this down — it's your recovery key for rebuilds and other hosts.\n"
            )
            return passphrase
        from prompt_toolkit import prompt as ptk_prompt  # noqa: PLC0415

        try:
            passphrase = ptk_prompt("credentials.db passphrase: ", is_password=True)
        except (KeyboardInterrupt, EOFError):
            raise SystemExit("passphrase entry cancelled.") from None
    else:
        passphrase = sys.stdin.readline().rstrip("\n")
    if not passphrase:
        raise ValueError("empty passphrase")
    return passphrase


def prompt_new_passphrase() -> str | None:
    """Read a typed-and-confirmed *new* passphrase from the TTY, or ``None`` on empty entry.

    The entry half of [`prompt_passphrase`][terok_sandbox.vault.store.encryption.prompt_passphrase]'s
    ``confirm`` mode, split out so flows that mint-and-announce on
    their own schedule (``vault passphrase change`` reveals only after
    the rekey succeeded) can reuse the exact same typed-entry
    experience: masked echo, a confirmation read, a mismatch error.
    ``None`` means the operator hit ``Enter`` on the first prompt —
    the established "generate one for me" gesture.
    """
    from prompt_toolkit import prompt as ptk_prompt  # noqa: PLC0415

    try:
        passphrase = ptk_prompt("credentials.db passphrase: ", is_password=True)
        if not passphrase:
            return None
        again = ptk_prompt("confirm passphrase:        ", is_password=True)
        if passphrase != again:
            raise ValueError("passphrases do not match")
        return passphrase
    except (KeyboardInterrupt, EOFError):
        raise SystemExit("passphrase entry cancelled.") from None


# ── SQLCipher primitives ────────────────────────────────────────────


def open_sqlcipher(db_path: str | Path, passphrase: str, **connect_kwargs: Any) -> Any:
    """Return a sqlcipher3 connection with *passphrase* applied.

    Rejects an empty passphrase at the lowest level — ``set_key("")``
    is SQLCipher's "open me plaintext" sentinel and would silently
    produce or read an unencrypted DB.  All higher-level call paths
    already screen for empties; this is the load-bearing guard.
    """
    if not passphrase:
        raise ValueError("empty passphrase would disable SQLCipher encryption")
    import sqlcipher3  # noqa: PLC0415

    conn = sqlcipher3.connect(str(db_path), **connect_kwargs)
    conn.set_key(passphrase)
    conn.execute("PRAGMA cipher_compatibility = 4")
    return conn


def generate_passphrase() -> str:
    """Return a freshly-randomised url-safe passphrase."""
    return secrets.token_urlsafe(_GENERATED_PASSPHRASE_BYTES)


def rekey_in_place(db_path: Path, old_passphrase: str, new_passphrase: str) -> None:
    """Re-encrypt *db_path* under *new_passphrase* via SQLCipher ``PRAGMA rekey``.

    The change-passphrase counterpart of
    [`encrypt_in_place`][terok_sandbox.vault.store.encryption.encrypt_in_place]
    — same journal discipline, different starting point: an
    already-encrypted DB instead of a legacy plaintext one, re-encrypted
    page by page inside SQLCipher's own journaled transaction (no temp
    copy, no backup — the operator's off-host passphrase copy is the
    recovery story).

    The WAL is drained and journaling switched to ``DELETE`` first: WAL
    frames are encrypted with the *old* key, so rekeying around a
    populated WAL would leave frames the new key cannot read.  Both
    steps report contention through their *result rows* rather than by
    raising — ``wal_checkpoint`` returns a busy flag and
    ``journal_mode`` echoes the old mode when it couldn't switch — so
    each is verified explicitly, and a live per-container supervisor
    still holding the DB surfaces as ``database is locked`` *before*
    anything is modified.

    Raises [`WrongPassphraseError`][terok_sandbox.vault.store.encryption.WrongPassphraseError]
    when *old_passphrase* doesn't open the DB and [`ValueError`][ValueError]
    on an empty new passphrase (SQLCipher's no-encryption sentinel).
    """
    if not new_passphrase:
        raise ValueError("empty passphrase would disable SQLCipher encryption")
    import sqlcipher3  # noqa: PLC0415

    conn = open_sqlcipher(db_path, old_passphrase)
    try:
        try:
            # A wrong key only surfaces on the first page read — force
            # one before touching anything.
            conn.execute("SELECT count(*) FROM sqlite_master")
        except sqlcipher3.DatabaseError as exc:
            if "file is not a database" not in str(exc):
                raise  # a real fault ("database is locked", I/O) — not a key mismatch
            raise WrongPassphraseError(f"the current passphrase does not open {db_path}") from exc
        busy, _log_pages, _moved_pages = conn.execute("PRAGMA wal_checkpoint(FULL)").fetchone()
        if busy:
            raise RuntimeError(
                f"database is locked — cannot drain the WAL of {db_path};"
                " another connection is still holding it"
            )
        (mode,) = conn.execute("PRAGMA journal_mode=DELETE").fetchone()
        if str(mode).lower() != "delete":
            raise RuntimeError(
                f"database is locked — cannot take exclusive hold of {db_path};"
                " another connection is still open"
            )
        # PRAGMA statements cannot take bound parameters; single-quote
        # doubling is SQL's exact string-literal escape, valid for any
        # passphrase content.
        conn.execute("PRAGMA rekey = '{}'".format(new_passphrase.replace("'", "''")))
        conn.execute("PRAGMA journal_mode=WAL")
        # No sidecar cleanup after this point: the DELETE-mode switch
        # already removed the old-key WAL, and an unlink after close
        # would race a fresh supervisor's brand-new (new-key) sidecars.
    finally:
        conn.close()


# ── Setup-time migration ────────────────────────────────────────────
#
# Everything below is a one-shot plaintext→SQLCipher migration path
# for users upgrading from pre-encryption releases.  Fresh installs
# never enter this code — the DB is created encrypted on first write.
#
# Deprecated in 0.8.0 (warning surfaced at setup time).
# Removed in 0.9.0 — after which any leftover plaintext DB stops
# being recognised and the operator must restore from the
# ``.plaintext-backup-<stamp>.tar.gz`` snapshot or reinitialise.


def is_plaintext_sqlite(db_path: Path) -> bool:
    """Return ``True`` if *db_path* is a legacy plaintext sqlite DB.

    Stdlib sqlite refuses to open SQLCipher files with ``DatabaseError:
    file is not a database``; a successful ``PRAGMA quick_check`` means
    the file is plain sqlite.  Used only by the one-shot setup
    migration — not on any runtime open path.
    """
    if not db_path.exists() or db_path.stat().st_size == 0:
        return False
    try:
        conn = sqlite3.connect(str(db_path))
        try:
            conn.execute("PRAGMA quick_check").fetchone()
        finally:
            conn.close()
    except sqlite3.DatabaseError:
        return False
    return True


_SQLITE_SIDECAR_SUFFIXES = ("-wal", "-shm", "-journal")


def _unlink_sidecars(db_path: Path) -> None:
    """Remove ``-wal`` / ``-shm`` / ``-journal`` files next to *db_path*.

    Best-effort: any of them may legitimately be absent.  Called twice
    in the migration — once for the plaintext source (so leftover WAL
    pages don't keep secrets on disk) and once for the encrypted temp
    DB (so a half-finished export leaves no debris).
    """
    for suffix in _SQLITE_SIDECAR_SUFFIXES:
        Path(str(db_path) + suffix).unlink(missing_ok=True)


def encrypt_in_place(db_path: Path, passphrase: str) -> None:
    """Convert plaintext *db_path* into a SQLCipher-encrypted DB.

    Deprecated in 0.8.0; scheduled for removal in 0.9.0.  After
    removal, this function and its CLI surface
    (``terok-sandbox credentials encrypt-db``) disappear — installs
    older than 0.8.0 must migrate before upgrading past 0.9.0.

    Atomic: a crash between export and rename leaves the original
    plaintext file untouched, so a re-run starts cleanly.

    WAL-aware: the legacy DB may have been opened in WAL mode (the
    daemon sets ``journal_mode=WAL`` on every connection), so its
    pages can live in ``.db-wal`` rather than the main file.  Before
    exporting we force a full checkpoint and switch to ``DELETE``
    journaling, then unlink the ``-wal`` / ``-shm`` / ``-journal``
    sidecars; otherwise plaintext secrets would survive the migration
    in the leftover sidecars even after the main file is encrypted.

    Permission-tight: the temp file is created up-front at 0o600 so
    SQLCipher's ``ATTACH`` doesn't materialise a world-readable
    encrypted DB under a permissive umask.
    """
    if not passphrase:
        raise ValueError("empty passphrase would produce a plaintext DB")
    if not db_path.exists():
        raise FileNotFoundError(db_path)

    tmp_path = db_path.with_suffix(db_path.suffix + ".encrypting")
    tmp_path.unlink(missing_ok=True)
    # Materialise tmp_path at 0o600 before ATTACH so SQLCipher inherits
    # those bits instead of the umask default — the file is empty so
    # SQLCipher will populate it freely.
    os.close(os.open(tmp_path, os.O_CREAT | os.O_EXCL | os.O_WRONLY, 0o600))

    import sqlcipher3  # noqa: PLC0415

    try:
        conn = sqlcipher3.connect(str(db_path))
        try:
            # Drain WAL into the main file and stop journaling so the
            # subsequent sidecar unlink genuinely removes plaintext data.
            conn.execute("PRAGMA wal_checkpoint(FULL)")
            conn.execute("PRAGMA journal_mode=DELETE")

            conn.execute(
                "ATTACH DATABASE ? AS encrypted KEY ?",
                (str(tmp_path), passphrase),
            )
            conn.execute("PRAGMA encrypted.cipher_compatibility = 4")
            (result,) = conn.execute("SELECT sqlcipher_export('encrypted')").fetchone() or (None,)
            conn.execute("DETACH DATABASE encrypted")
        finally:
            conn.close()

        if result is not None and result != 0:
            raise RuntimeError(f"sqlcipher_export returned {result!r}")
    except BaseException:
        # Any failure between pre-create and replace must scrub the
        # ``.encrypting`` temp file and its sidecars so a re-run starts
        # clean.  ``BaseException`` covers SystemExit / KeyboardInterrupt
        # too — leaking a zero-byte tmp is the failure mode the user
        # actually hits ("database is locked" with a stale temp left
        # behind on disk).
        tmp_path.unlink(missing_ok=True)
        _unlink_sidecars(tmp_path)
        raise

    tmp_path.replace(db_path)
    # Sidecars under both names: plaintext leftovers from the legacy
    # connection (now next to the encrypted file) and any encrypted-side
    # sidecars that briefly accompanied the temp file.
    _unlink_sidecars(db_path)
    _unlink_sidecars(tmp_path)


__all__ = [
    "DESKTOP_KEYRING_SERVICE",
    "DESKTOP_KEYRING_USERNAME",
    "NoPassphraseError",
    "PassphraseTier",
    "WrongPassphraseError",
    "encrypt_in_place",
    "forget_passphrase_in_desktop_keyring",
    "generate_passphrase",
    "is_plaintext_sqlite",
    "desktop_keyring_backend_available",
    "load_passphrase_from_command",
    "load_passphrase_from_desktop_keyring",
    "open_sqlcipher",
    "open_sqlcipher_via_chain",
    "prompt_new_passphrase",
    "prompt_passphrase",
    "rekey_in_place",
    "resolve_passphrase",
    "resolve_passphrase_with_source",
    "store_passphrase_in_desktop_keyring",
]
