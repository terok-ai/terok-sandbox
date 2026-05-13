# SPDX-FileCopyrightText: 2026 Jiri Vyskocil
# SPDX-License-Identifier: Apache-2.0

"""Passphrase plumbing and SQLCipher helpers for at-rest credential encryption.

Owns the runtime passphrase resolution chain (tmpfs session-unlock
file → OS keyring → ``credentials.passphrase`` config-file fallback →
interactive prompt) and the one-shot setup migration from legacy
plaintext sqlite files.
"""

from __future__ import annotations

import os
import secrets
import sqlite3
import sys
from pathlib import Path
from typing import Any, Literal

from . import systemd_creds as _systemd_creds

KEYRING_SERVICE = "terok-sandbox"
KEYRING_USERNAME = "credentials-db"

#: ``token_urlsafe(32)`` ≈ 43 chars of URL-safe Base64 — 256 bits of
#: entropy from a 62-char alphabet plus ``-``/``_``, both shell-safe.
_GENERATED_PASSPHRASE_BYTES = 32

#: Where in the chain a passphrase came from — domain vocabulary so callers can
#: dispatch on a closed set instead of stringly-typed branches.  ``"prompt"``
#: covers both the runtime-prompt fallback and the setup-time chooser path.
PassphraseSource = Literal["session-file", "systemd-creds", "keyring", "config", "prompt"]


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
    config_fallback: str | None = None,
    prompt_on_tty: bool = False,
    **connect_kwargs: Any,
) -> Any:
    """Resolve the passphrase through the runtime chain and open *db_path*.

    Raises [`NoPassphraseError`][terok_sandbox.credentials.encryption.NoPassphraseError]
    when the chain yields nothing.  *prompt_on_tty* turns on the
    interactive fallback for CLI consumers; daemons leave it ``False``.
    """
    passphrase = resolve_passphrase(
        passphrase_file=passphrase_file,
        systemd_creds_file=systemd_creds_file,
        use_keyring=use_keyring,
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
    config_fallback: str | None = None,
    prompt_on_tty: bool = False,
) -> tuple[str | None, PassphraseSource | None]:
    """Walk the runtime resolution chain; return ``(passphrase, source)``.

    Single source of truth for the resolution order — see
    [`resolve_passphrase`][terok_sandbox.credentials.encryption.resolve_passphrase]
    for the tier semantics.  Both elements of the tuple are ``None``
    when no tier had a passphrase.

    The source half is what feeds
    [`VaultStatus.passphrase_source`][terok_sandbox.VaultStatus] and the
    TUI status pill — keep the labels stable, callers dispatch on them.
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
        # Fail closed: a present-but-unsealable credential is a
        # downgrade vector — silent fall-through to keyring / config
        # would change the security posture (machine-bound → keyring
        # or plaintext-on-disk) without the operator's knowledge.
        # Surface as ``locked`` so the caller sees the broken state
        # instead of a working-but-weaker vault.
        raise WrongPassphraseError(
            f"sealed systemd-creds credential present at {systemd_creds_file}"
            " but could not be unsealed"
        )
    if use_keyring:
        keyring_pw = load_passphrase_from_keyring()
        if keyring_pw:
            return keyring_pw, "keyring"
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
    config_fallback: str | None = None,
    prompt_on_tty: bool = False,
) -> str | None:
    """Walk the runtime resolution chain; return ``None`` if nothing has it.

    Order:

    1. *passphrase_file* — session-unlock tmpfs file (cleared on reboot).
    2. *systemd_creds_file* — sealed credential decrypted via
       ``systemd-creds(1)``.  Machine-bound (TPM2 or host key), survives
       reboot, no OS keyring required.  See
       [`terok_sandbox.credentials.systemd_creds`][terok_sandbox.credentials.systemd_creds].
    3. OS keyring — only when *use_keyring* is true; off by default because
       Linux Secret Service grants access per-collection, not per-item.
    4. *config_fallback* — ``credentials.passphrase`` from ``config.yml``.
       Plaintext-on-disk trust boundary: the operator accepts that
       filesystem-level protection (LUKS / signed image / permissions)
       is their security perimeter.  Sandbox#282 surfaces a permanent
       WARNING in ``vault status`` and sickbay whenever this tier is
       set, regardless of which tier actually unlocked the call.
    5. Interactive prompt — only when *prompt_on_tty* and ``sys.stdin.isatty()``.

    *config_fallback* is threaded through as a parameter rather than
    read here so this module stays free of any dependency on the
    sandbox config layer — the config module already imports from
    credentials.db, and the back-edge would close a tach cycle.
    """
    passphrase, _source = resolve_passphrase_with_source(
        passphrase_file=passphrase_file,
        systemd_creds_file=systemd_creds_file,
        use_keyring=use_keyring,
        config_fallback=config_fallback,
        prompt_on_tty=prompt_on_tty,
    )
    return passphrase


# ── Tier primitives ─────────────────────────────────────────────────


def load_passphrase_from_file(path: Path) -> str | None:
    """Return the passphrase stored at *path*, or ``None`` if absent or unreadable."""
    try:
        return path.read_text(encoding="utf-8").rstrip("\n") or None
    except OSError:
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


def _write_to_controlling_tty(message: str) -> None:
    """Write *message* to ``/dev/tty`` so a redirected stdout can't capture it.

    Fails closed when no controlling TTY is reachable (CI, headless
    automation): refuses rather than letting an irrecoverable
    generated passphrase fall on the floor.  Operators automating
    setup must pre-provide the passphrase via a tier the resolver
    can find.
    """
    try:
        with Path("/dev/tty").open("w", encoding="utf-8") as tty:
            tty.write(message)
    except OSError as exc:
        raise SystemExit(
            "Refusing to print the generated vault passphrase: no controlling TTY"
            f" ({exc.strerror}).\n"
            "Re-run setup from an interactive terminal, or pre-provide the"
            " passphrase (vault unlock / credentials.passphrase / sealed credential)"
            " before re-running."
        ) from exc


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
# Deprecated in 0.9.0 (warning surfaced at setup time).
# Removed in 0.10.0 — after which any leftover plaintext DB stops
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

    Deprecated in 0.9.0; scheduled for removal in 0.10.0.  After
    removal, this function and its CLI surface
    (``terok-sandbox credentials encrypt-db``) disappear — installs
    older than 0.9.0 must migrate before upgrading past 0.10.0.

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
    # Now that the encrypted file is canonical at db_path, clean up
    # any plaintext sidecars left behind by the legacy connection plus
    # any encrypted-side sidecars under the tmp name.
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
    "load_passphrase_from_file",
    "load_passphrase_from_keyring",
    "open_sqlcipher",
    "open_sqlcipher_via_chain",
    "prompt_passphrase",
    "resolve_passphrase",
    "resolve_passphrase_with_source",
    "store_passphrase_in_keyring",
]
