# SPDX-FileCopyrightText: 2026 Jiri Vyskocil
# SPDX-License-Identifier: Apache-2.0

"""Passphrase plumbing and SQLCipher helpers for at-rest credential encryption.

Walks the six-tier resolution chain — session-unlock file →
systemd-creds → OS keyring → ``passphrase_command`` helper →
plaintext config fallback → interactive prompt — and exposes the
SQLCipher open / migrate primitives the rest of the package builds on.
``resolve_passphrase`` documents the chain order; ``open_sqlcipher``
is the only entry point that ever calls ``sqlcipher3.connect``.

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
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Literal

from . import systemd_creds as _systemd_creds

KEYRING_SERVICE = "terok-sandbox"
KEYRING_USERNAME = "credentials-db"

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

#: Closed set of tier labels — feeds the vault status display's
#: passphrase-source label and the setup-chooser's
#: [`SetupTier`][terok_sandbox.commands.credentials.SetupTier] subset.
PassphraseSource = Literal[
    "session-file",
    "systemd-creds",
    "keyring",
    "passphrase-command",
    "config",
    "prompt",
]


class NoPassphraseError(RuntimeError):
    """No SQLCipher passphrase resolved — the DB cannot be opened."""


class WrongPassphraseError(RuntimeError):
    """SQLCipher could not decrypt the DB — passphrase doesn't match its encryption key."""


# ── Resolution chain ────────────────────────────────────────────────


def open_sqlcipher_via_chain(
    db_path: str | Path,
    *,
    passphrase_file: Path | None = None,
    systemd_creds_file: Path | None = None,
    use_keyring: bool = False,
    passphrase_command: str | None = None,
    config_fallback: str | None = None,
    prompt_on_tty: bool = False,
    **connect_kwargs: Any,
) -> Any:
    """Resolve the passphrase through the runtime chain and open *db_path*.

    Raises [`NoPassphraseError`][terok_sandbox.vault.store.encryption.NoPassphraseError]
    when the chain yields nothing.  *prompt_on_tty* turns on the
    interactive fallback for CLI consumers; daemons leave it ``False``.
    """
    passphrase = resolve_passphrase(
        passphrase_file=passphrase_file,
        systemd_creds_file=systemd_creds_file,
        use_keyring=use_keyring,
        passphrase_command=passphrase_command,
        config_fallback=config_fallback,
        prompt_on_tty=prompt_on_tty,
    )
    if passphrase is None:
        raise NoPassphraseError(f"no SQLCipher passphrase available for {db_path}")
    return open_sqlcipher(db_path, passphrase, **connect_kwargs)


def resolve_passphrase_with_source(
    *,
    passphrase_file: Path | None = None,
    systemd_creds_file: Path | None = None,
    use_keyring: bool = False,
    passphrase_command: str | None = None,
    config_fallback: str | None = None,
    prompt_on_tty: bool = False,
) -> tuple[str | None, PassphraseSource | None]:
    """Walk the runtime resolution chain; return ``(passphrase, source)``.

    Single source of truth for the resolution order — see
    [`resolve_passphrase`][terok_sandbox.vault.store.encryption.resolve_passphrase]
    for the tier semantics.  Both elements of the tuple are ``None``
    when no tier had a passphrase.

    The source half feeds a TUI/CLI status display — keep the labels
    stable, callers dispatch on them.
    """
    # Truthy checks throughout: an empty string anywhere in the chain
    # is SQLCipher's no-encryption sentinel; treat it as "not present"
    # rather than letting it overrule a real later tier.
    if passphrase_file is not None:
        file_pw = load_passphrase_from_file(passphrase_file)
        if file_pw:
            return file_pw, "session-file"
    if systemd_creds_file is not None and systemd_creds_file.is_file():
        sealed_pw = _systemd_creds.unseal(systemd_creds_file)
        if sealed_pw:
            return sealed_pw, "systemd-creds"
        # Fail closed: silently falling through would demote a
        # machine-bound tier to keyring / plaintext-on-disk without
        # the operator's knowledge.
        raise WrongPassphraseError(
            f"sealed systemd-creds credential present at {systemd_creds_file}"
            " but could not be unsealed"
        )
    if use_keyring:
        keyring_pw = load_passphrase_from_keyring()
        if keyring_pw:
            return keyring_pw, "keyring"
    if passphrase_command:
        cmd_pw = load_passphrase_from_command(passphrase_command)
        if cmd_pw:
            return cmd_pw, "passphrase-command"
        # Fail closed for the same reason as systemd-creds above; the
        # command string itself is omitted because operators sometimes
        # inline AWS ARNs / vault paths there and this exception reaches
        # doctor output and journals.
        raise WrongPassphraseError(
            "passphrase_command produced no passphrase; run it manually to diagnose"
            " (see WARNING in the vault journal)"
        )
    if config_fallback:
        return config_fallback, "config"
    if prompt_on_tty and sys.stdin.isatty():
        return prompt_passphrase(), "prompt"
    return None, None


def resolve_passphrase(
    *,
    passphrase_file: Path | None = None,
    systemd_creds_file: Path | None = None,
    use_keyring: bool = False,
    passphrase_command: str | None = None,
    config_fallback: str | None = None,
    prompt_on_tty: bool = False,
) -> str | None:
    """Walk the runtime resolution chain; return ``None`` if nothing has it.

    Order:

    1. *passphrase_file* — session-unlock tmpfs file (cleared on reboot).
    2. *systemd_creds_file* — sealed credential decrypted via
       ``systemd-creds(1)``.  Machine-bound (TPM2 or host key), survives
       reboot, no OS keyring required.  See
       [`terok_sandbox.vault.store.systemd_creds`][terok_sandbox.vault.store.systemd_creds].
    3. OS keyring — only when *use_keyring* is true; off by default because
       Linux Secret Service grants access per-collection, not per-item.
    4. *passphrase_command* — operator-supplied shell command
       (``pass show …``, ``bw get``, ``op read``, cloud secret-manager
       CLIs).  Delegates retrieval without per-backend integration code,
       same shape as ``git config credential.helper`` or
       ``BORG_PASSCOMMAND``.  Configured-but-broken fails closed so a
       misbehaving helper can't silently demote security to plaintext.
    5. *config_fallback* — ``credentials.passphrase`` from ``config.yml``.
       Plaintext-on-disk trust boundary: the operator accepts that
       filesystem-level protection (LUKS / signed image / permissions)
       is their security perimeter.  Sandbox#282 surfaces a permanent
       WARNING in ``vault status`` and sickbay whenever this tier is
       set, regardless of which tier actually unlocked the call.
    6. Interactive prompt — only when *prompt_on_tty* and ``sys.stdin.isatty()``.

    *config_fallback* and *passphrase_command* are threaded through as
    parameters rather than read here so this module stays free of any
    dependency on the sandbox config layer — the config module already
    imports from credentials.db, and the back-edge would close a tach
    cycle.
    """
    passphrase, _source = resolve_passphrase_with_source(
        passphrase_file=passphrase_file,
        systemd_creds_file=systemd_creds_file,
        use_keyring=use_keyring,
        passphrase_command=passphrase_command,
        config_fallback=config_fallback,
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
    show when a high-priority tier (typically the session file) is
    *shadowing* a durable tier underneath — the operator's "why is my
    TPM2 box reading a RAM-backed file?" question.
    """

    source: PassphraseSource
    present: bool
    detail: str


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


def probe_passphrase_chain(
    *,
    passphrase_file: Path | None = None,
    systemd_creds_file: Path | None = None,
    use_keyring: bool = False,
    passphrase_command: str | None = None,
    config_fallback: str | None = None,
) -> tuple[TierPresence, ...]:
    """Report per-tier presence across the resolution chain without short-circuiting.

    Presence is judged from *material on hand*, not by resolving the
    secret: the sealed systemd-creds credential is never unsealed and
    the ``passphrase_command`` is never executed (both can be slow or
    have side effects), so their mere configuration counts as present.
    The session-file and keyring tiers are cheap to read, so those are
    probed for real.  Tiers appear in resolution order; the first
    ``present`` one is the tier that would unlock the vault.  The
    interactive ``prompt`` tier is omitted — it stores nothing, so it
    can neither be "present" nor shadow anything.
    """
    session_value = load_passphrase_from_file(passphrase_file) if passphrase_file else None
    session_detail = str(passphrase_file) if passphrase_file else "no session file"
    # A file that exists but yields nothing is a fault (permissions,
    # SELinux, empty write), not a locked vault — say so in the detail
    # line; the silent variant cost us a debugging session already.
    if passphrase_file is not None and session_value is None and passphrase_file.exists():
        session_detail = f"{passphrase_file} (exists but unreadable or empty)"
    return (
        TierPresence(
            "session-file",
            bool(session_value),
            session_detail,
        ),
        TierPresence(
            "systemd-creds",
            bool(systemd_creds_file and systemd_creds_file.is_file()),
            _systemd_creds_detail(systemd_creds_file),
        ),
        TierPresence(
            "keyring",
            # Truthy, not ``is not None``: an empty string is the resolver's
            # "no passphrase" sentinel, so status must treat it as absent too.
            use_keyring and bool(load_passphrase_from_keyring()),
            "OS keyring" if use_keyring else "use_keyring off",
        ),
        TierPresence(
            "passphrase-command",
            bool(passphrase_command),
            "configured (not executed)" if passphrase_command else "not configured",
        ),
        TierPresence(
            "config",
            bool(config_fallback),
            "plaintext in config.yml" if config_fallback else "not set",
        ),
    )


# ── Tier primitives ─────────────────────────────────────────────────


def load_passphrase_from_file(path: Path) -> str | None:
    """Return the passphrase stored at *path*, or ``None`` if absent or unreadable.

    An absent file is the normal locked state and stays silent.  Any
    *other* ``OSError`` (EACCES from a permissions slip, an SELinux
    denial, an unmounted tmpfs) also degrades to ``None`` so the chain
    can fall through — but it logs a warning first: without the log a
    blocked read is indistinguishable from "locked" on every surface,
    which buries the actual fault.
    """
    try:
        return path.read_text(encoding="utf-8").rstrip("\n") or None
    except FileNotFoundError:
        return None
    except OSError as exc:
        _logger.warning("session passphrase file %s exists but is unreadable: %s", path, exc)
        return None


def load_passphrase_from_keyring() -> str | None:
    """Return the keyring-stored passphrase, or ``None`` if no backend is reachable."""
    try:
        import keyring  # noqa: PLC0415

        return keyring.get_password(KEYRING_SERVICE, KEYRING_USERNAME)
    except Exception:  # noqa: BLE001
        return None


def store_passphrase_in_keyring(passphrase: str) -> bool:
    """Persist *passphrase* in the OS keyring; return ``True`` on success.

    Refuses to store an empty value — SQLCipher interprets it as
    "no encryption", and a later resolve hit on a blank keyring entry
    would silently open the DB plaintext.
    """
    if not passphrase:
        raise ValueError("refusing to store an empty passphrase in the keyring")
    try:
        import keyring  # noqa: PLC0415

        keyring.set_password(KEYRING_SERVICE, KEYRING_USERNAME, passphrase)
        return True
    except Exception:  # noqa: BLE001
        return False


def forget_passphrase_in_keyring() -> bool:
    """Remove the keyring entry; return ``True`` on success."""
    try:
        import keyring  # noqa: PLC0415

        keyring.delete_password(KEYRING_SERVICE, KEYRING_USERNAME)
        return True
    except Exception:  # noqa: BLE001
        return False


def load_passphrase_from_command(
    command: str, *, timeout: float = _PASSPHRASE_COMMAND_TIMEOUT_S
) -> str | None:
    """Run *command*, return its stdout with the trailing newline removed, or ``None`` on any failure.

    Same shape as the other tier primitives ([`load_passphrase_from_file`][terok_sandbox.vault.store.encryption.load_passphrase_from_file],
    [`load_passphrase_from_keyring`][terok_sandbox.vault.store.encryption.load_passphrase_from_keyring]):
    silent on every failure path so the resolver can decide whether
    ``None`` means "skip this tier" or "fail closed".  Diagnostic
    detail (parse error, exec failure, non-zero exit, helper stderr,
    timeout) is logged at WARNING so operators can triage their helper
    via ``journalctl --user -u terok-vault`` without us crashing the
    chain walk.

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
        _logger.warning("passphrase_command shlex parse failed: %s", exc)
        return None
    if not argv:
        return None
    try:
        result = subprocess.run(  # noqa: S603 — argv is operator-configured  # nosec B603 — argv is a fixed list controlled by this module — argv is a fixed list controlled by this module
            argv, capture_output=True, text=True, timeout=timeout, check=False
        )
    except OSError as exc:
        _logger.warning("passphrase_command %r failed to spawn: %s", argv[0], exc)
        return None
    except subprocess.TimeoutExpired:
        _logger.warning("passphrase_command %r timed out after %.0fs", argv[0], timeout)
        return None
    if result.returncode != 0:
        _logger.warning(
            "passphrase_command %r exited %d: %s",
            argv[0],
            result.returncode,
            result.stderr.strip() or "(no stderr)",
        )
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
        from prompt_toolkit import prompt as ptk_prompt  # noqa: PLC0415

        try:
            passphrase = ptk_prompt("credentials.db passphrase: ", is_password=True).strip()
            if not passphrase and confirm:
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
            if confirm:
                again = ptk_prompt("confirm passphrase:        ", is_password=True).strip()
                if passphrase != again:
                    raise ValueError("passphrases do not match")
        except (KeyboardInterrupt, EOFError):
            raise SystemExit("passphrase entry cancelled.") from None
    else:
        passphrase = sys.stdin.readline().rstrip("\n")
    if not passphrase:
        raise ValueError("empty passphrase")
    return passphrase


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
    "KEYRING_SERVICE",
    "KEYRING_USERNAME",
    "NoPassphraseError",
    "PassphraseSource",
    "WrongPassphraseError",
    "encrypt_in_place",
    "forget_passphrase_in_keyring",
    "generate_passphrase",
    "is_plaintext_sqlite",
    "load_passphrase_from_command",
    "load_passphrase_from_file",
    "load_passphrase_from_keyring",
    "open_sqlcipher",
    "open_sqlcipher_via_chain",
    "prompt_passphrase",
    "resolve_passphrase",
    "resolve_passphrase_with_source",
    "store_passphrase_in_keyring",
]
