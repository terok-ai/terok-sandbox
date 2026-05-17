# SPDX-FileCopyrightText: 2026 Jiri Vyskocil
# SPDX-License-Identifier: Apache-2.0

"""Credentials-DB at-rest encryption — chooser, provisioning, migration.

Three passphrase storage modes are chosen interactively when
``systemd-creds`` isn't available; with it, the chooser is skipped and
the credential is sealed silently.  Once chosen, the mode is persisted
into ``config.yml`` so the resolution chain picks it up on the next
daemon start — session mode self-describes via the tmpfs file's
presence; keyring sets ``credentials.use_keyring=true``; config writes
the passphrase itself into the file.

The plaintext→encrypted migration is deprecated in 0.9.0 and slated
for removal in 0.10.0.  After that release fresh installs stay the
only supported entry point; operators with a stale plaintext DB must
restore from the ``.plaintext-backup-<stamp>.tar.gz`` snapshot this
phase writes before re-keying.
"""

from __future__ import annotations

import contextlib
import sys
from pathlib import Path
from typing import Any, Literal

from .._yaml import update_section as _yaml_update_section
from ..config import SandboxConfig
from ..vault.store.encryption import PassphraseSource
from ._types import CommandDef

#: The chooser-offered subset of [`PassphraseSource`][terok_sandbox.vault.store.encryption.PassphraseSource].
#: ``systemd-creds`` is auto-detected (not offered); ``passphrase-command``
#: is an opt-in config-file edit (not offered); ``prompt`` is a
#: runtime-only fallback.  Same vocabulary as the resolver so a chooser
#: pick and a resolver hit speak the same language.
SetupTier = Literal["session-file", "keyring", "config"]

#: Tiers the operator may force via ``--passphrase-tier`` (or the
#: equivalent Python kwarg).  ``systemd-creds`` is included here so
#: ``--passphrase-tier=systemd-creds`` can override the auto-detect
#: when the operator wants to be explicit, and refuse silently if
#: systemd-creds isn't available.  ``config`` is allowed but the
#: confirmation banner still gates it.
_EXPLICIT_TIERS: frozenset[str] = frozenset({"session-file", "keyring", "config", "systemd-creds"})

_NON_TTY_TIER_HINT = """\
terok-sandbox setup: running non-interactively but no passphrase tier was chosen.

  systemd-creds is unavailable on this host (needs systemd ≥ 257), so
  setup would otherwise fall through to the session-unlock tmpfs file
  — a fresh random passphrase you would never see, lost on the first
  reboot.  Pick a tier explicitly:

    --passphrase-tier keyring        (recommended on a single-user host)
    --passphrase-tier session-file   (re-run `vault unlock` after each boot)
    --passphrase-tier config         (plaintext-on-disk; needs confirmation)

  Or install systemd ≥ 257 (Fedora ≥ 42, Debian ≥ 13) and re-run setup
  so the systemd-creds auto-tier becomes available."""

_CHOOSER_PROMPT = """\

systemd-creds isn't available on this host (needs systemd ≥ 257 with
the user Varlink service).  Where should terok store the passphrase
to encrypt the vault?

  [k] keyring — your login keyring (recommended; auto-unlocks at login)
  [s] session-unlock — terok-sandbox vault unlock after each boot
  [c] config file — plaintext on disk; same as encrypted DB (requires confirmation)

For the strongest protection, install systemd ≥ 257 and re-run setup.
Choice [k]:"""

# Operator's first character maps to the tier.  Empty input picks the
# recommended default (keyring); anything outside this set falls back
# to keyring too — safer than guessing the operator meant ``[s]``.
_CHOICE_TO_TIER: dict[str, SetupTier] = {"s": "session-file", "k": "keyring", "c": "config"}
_DEFAULT_TIER: SetupTier = "keyring"

_CONFIG_TIER_CONFIRMATION = """\

You picked the config-file tier.  This stores the SQLCipher passphrase
as plaintext on the same disk as the encrypted database itself, so
your encryption is only as strong as the filesystem layer
(LUKS / signed image / per-user permissions).

  - Do NOT enable on shared or multi-tenant machines.
  - The keyring tier ([k]) and systemd-creds tier (separate verb)
    bind the passphrase to your login session or TPM; both are
    safer defaults on a single-user host.
  - `terok-sandbox vault status` and sickbay will permanently warn
    that plaintext-on-disk is configured until you remove it.

Type `yes` to confirm, anything else to choose a different tier:"""


def _handle_credentials_encrypt_db(
    *,
    cfg: SandboxConfig | None = None,
    echo_passphrase: bool = False,
    passphrase_tier: str | None = None,
) -> None:
    """Provision the credentials-DB passphrase; migrate any legacy plaintext file.

    Short-circuits when the DB is already SQLCipher-encrypted: minting
    a fresh passphrase here would overwrite whatever tier currently
    holds the working key and lock the operator out.

    Tier selection precedence:

    1. *passphrase_tier* (CLI: ``--passphrase-tier``) — the operator
       said exactly which tier to use, so honour it.  ``systemd-creds``
       refuses if unavailable; ``session-file`` skips the silent-default
       refusal below.
    2. ``systemd-creds`` auto-detect — the strongest option when present;
       asking when the answer is unambiguous just slows the operator down.
    3. Interactive chooser on a TTY; otherwise hard-fail with an
       actionable hint.  Earlier releases silently fell through to
       ``session-file`` here, which generates a fresh passphrase that
       the operator never sees and that evaporates on the next reboot.

    *echo_passphrase* mirrors the announce path: when ``True``, any
    auto-generated passphrase is also printed to stdout so
    non-interactive bootstraps (CI, Ansible) can capture it into their
    own secret store.  Default ``False`` so a routine ``setup > log``
    can't leak it.

    Whenever a passphrase is *auto-generated* (any tier where the
    operator didn't type it themselves) the ack flow runs after
    provisioning — see
    [`_maybe_acknowledge_recovery`][terok_sandbox.commands.credentials._maybe_acknowledge_recovery]
    — and writes a sidecar marker so the unconfirmed-recovery warning
    in the TUI / sickbay / doctor turns off.
    """
    from ..vault.store.encryption import encrypt_in_place, is_plaintext_sqlite

    if cfg is None:
        cfg = SandboxConfig()

    db_path = cfg.db_path
    if db_path.exists() and not is_plaintext_sqlite(db_path):
        print(f"  {db_path} is already SQLCipher-encrypted.")
        return

    passphrase, source, auto_generated = _select_and_provision(
        cfg,
        passphrase_tier=passphrase_tier,
        echo_passphrase=echo_passphrase,
    )
    print(f"  passphrase source: {source}")

    if auto_generated:
        _maybe_acknowledge_recovery(cfg, passphrase, echo_to_stdout=echo_passphrase)

    if not db_path.exists():
        print(f"  no DB at {db_path}; will be created encrypted on first use.")
        return

    # Snapshot the plaintext DB before touching anything — a failed
    # migration without a fallback would leave the operator with a
    # possibly-clobbered DB and no recourse.  The tarball is
    # intentionally NOT auto-deleted; the operator must remove it
    # explicitly once they've verified the migration.
    backup_path = _back_up_plaintext_db(db_path)
    encrypt_in_place(db_path, passphrase)
    print(f"  encrypted {db_path} in place.")
    _warn_about_plaintext_backup(backup_path)


def _select_and_provision(
    cfg: SandboxConfig,
    *,
    passphrase_tier: str | None,
    echo_passphrase: bool,
) -> tuple[str, PassphraseSource, bool]:
    """Pick a tier, provision a passphrase, return ``(passphrase, source, auto_generated)``.

    Centralises the precedence logic so
    [`_handle_credentials_encrypt_db`][terok_sandbox.commands.credentials._handle_credentials_encrypt_db]
    stays linear and the unit tests can hit each branch directly.
    """
    from ..vault.store import systemd_creds as _systemd_creds

    if passphrase_tier is not None:
        return _provision_explicit_tier(cfg, tier=passphrase_tier, echo_passphrase=echo_passphrase)

    if _systemd_creds.is_available():
        passphrase, source = _provision_systemd_creds_tier(cfg, echo_passphrase=echo_passphrase)
        return passphrase, source, True

    mode = _ask_passphrase_mode()
    passphrase, source, auto_generated = _provision_passphrase(
        cfg, mode=mode, echo_passphrase=echo_passphrase
    )
    _persist_mode_choice(mode, passphrase)
    return passphrase, source, auto_generated


def _provision_explicit_tier(
    cfg: SandboxConfig, *, tier: str, echo_passphrase: bool
) -> tuple[str, PassphraseSource, bool]:
    """Honour an explicit ``--passphrase-tier`` choice.

    Unknown tiers fail fast (``SystemExit``) with the allowed vocabulary;
    ``systemd-creds`` checks availability before dispatching.  The
    config-file tier still threads through the chooser's confirmation
    banner — being explicit on the CLI doesn't waive the "plaintext on
    disk" acknowledgement.
    """
    from ..vault.store import systemd_creds as _systemd_creds

    if tier not in _EXPLICIT_TIERS:
        raise SystemExit(
            f"unknown --passphrase-tier {tier!r};"
            f" expected one of: {', '.join(sorted(_EXPLICIT_TIERS))}"
        )
    if tier == "systemd-creds":
        if not _systemd_creds.is_available():
            raise SystemExit(
                "--passphrase-tier=systemd-creds requested but systemd-creds is"
                " unavailable (needs systemd ≥ 257 with the Varlink"
                " io.systemd.Credentials interface)"
            )
        passphrase, source = _provision_systemd_creds_tier(cfg, echo_passphrase=echo_passphrase)
        return passphrase, source, True

    if tier == "config":
        # Same confirmation gate the interactive chooser applies; an
        # explicit CLI choice still requires the operator to type ``yes``
        # so plaintext-on-disk never lands without a deliberate ack.
        if not _confirm_config_tier():
            raise SystemExit("config tier not confirmed; pick a different --passphrase-tier")

    mode: SetupTier = tier  # type: ignore[assignment]  # narrowed by membership check above
    passphrase, source, auto_generated = _provision_passphrase(
        cfg, mode=mode, echo_passphrase=echo_passphrase
    )
    _persist_mode_choice(mode, passphrase)
    return passphrase, source, auto_generated


def _confirm_config_tier() -> bool:
    """Show the plaintext-on-disk warning; return ``True`` iff the operator typed ``yes``.

    Reads from ``stdin`` so the confirmation works both interactively
    and via a piped ``yes`` (the rare case of a CI bootstrap that
    really does want the plaintext tier).
    """
    print(_CONFIG_TIER_CONFIRMATION)
    return sys.stdin.readline().strip().lower() == "yes"


def _maybe_acknowledge_recovery(
    cfg: SandboxConfig, passphrase: str, *, echo_to_stdout: bool
) -> None:
    """Run the "I have saved my recovery key" gate after an auto-mint.

    The operator just had a fresh passphrase displayed and now needs
    to confirm they have an off-host copy — every keystore tier is
    bound to *this* machine / account / boot, so a hardware failure
    strands the vault without a written recovery key.

    *echo_to_stdout* is the automation opt-out: a CI / TUI bootstrap
    that passed ``--echo-passphrase`` is on its own to call
    ``vault passphrase acknowledge`` once it has captured the value
    into its own secret store.  Skipping the interactive prompt here
    keeps non-TTY runs from hanging on a confirmation they can't
    answer; the unconfirmed-recovery warning surfaces in sickbay /
    doctor / TUI pill until the out-of-band ack lands.

    On a TTY-less run *without* ``--echo-passphrase`` the announcement
    step would have already failed closed in
    [`_write_to_controlling_tty`][terok_sandbox.vault.store.encryption._write_to_controlling_tty],
    so this branch isn't reachable — we degrade to "no ack written"
    rather than re-prompting.
    """
    from ..vault.store.encryption import _read_from_controlling_tty
    from ..vault.store.recovery import acknowledge

    if echo_to_stdout:
        # CI / TUI flow: marker is written out-of-band by the operator
        # (or by the TUI) after they've captured the passphrase.  Don't
        # block here — the unconfirmed warning is the safety net.
        print(
            "  recovery key NOT auto-acknowledged (--echo-passphrase set);"
            " run `terok-sandbox vault passphrase acknowledge`"
            " once you have saved the value."
        )
        return

    response = _read_from_controlling_tty(
        "Type SAVED to confirm you have stored the recovery key elsewhere: "
    )
    if response is None:
        # No controlling TTY — the announcement step has its own
        # failure mode, this is just defence in depth.
        return
    if response.strip() == "SAVED":
        acknowledge(cfg.vault_recovery_marker_file, passphrase)
        print("  recovery key marked as saved.")
    else:
        print(
            "  recovery key NOT confirmed — sickbay / doctor / TUI will warn"
            " until you run `terok-sandbox vault passphrase reveal` and"
            " confirm."
        )


def _ask_passphrase_mode() -> SetupTier:
    """Return the operator's chosen mode; refuse non-TTY runs without an explicit tier.

    Reached only when ``systemd-creds`` isn't available AND no explicit
    ``--passphrase-tier`` was supplied — the higher layers short-circuit
    before us in both of those cases.  Earlier releases auto-picked
    ``session-file`` on non-TTY to keep installs from hanging; the
    side-effect was a silent fresh passphrase that the operator never
    saw, lost on the first reboot.  That convenience-vs-data-loss
    trade is wrong, so we now fail closed with an actionable hint.
    """
    if not sys.stdin.isatty():
        raise SystemExit(_NON_TTY_TIER_HINT)
    while True:
        print(_CHOOSER_PROMPT)
        choice = sys.stdin.readline().strip().lower()[:1]
        if not choice:
            return _DEFAULT_TIER
        mode = _CHOICE_TO_TIER.get(choice, _DEFAULT_TIER)
        if mode != "config":
            return mode
        if _confirm_config_tier():
            return mode


def _provision_systemd_creds_tier(
    cfg: SandboxConfig, *, echo_passphrase: bool = False
) -> tuple[str, PassphraseSource]:
    """Auto-detected systemd-creds branch: mint a passphrase and seal it.

    ``--with-key=auto`` lets the host's TPM2 / host-key combination
    pick itself: a TPM-equipped laptop seals as ``host+tpm2``, a
    headless server without TPM falls back to ``host``-only.

    Always auto-generates a fresh passphrase — by construction this
    tier has no existing material to reuse, so the caller treats the
    return as ``auto_generated=True`` and runs the ack flow over it.
    """
    from ..vault.store import systemd_creds as _systemd_creds
    from ..vault.store.encryption import generate_passphrase

    passphrase = generate_passphrase()
    _systemd_creds.seal(passphrase, cfg.vault_systemd_creds_file, key_mode="auto")
    _announce_generated_passphrase(passphrase, echo_to_stdout=echo_passphrase)
    return passphrase, "systemd-creds"


def _provision_passphrase(
    cfg: SandboxConfig, *, mode: SetupTier, echo_passphrase: bool = False
) -> tuple[str, PassphraseSource, bool]:
    """Resolve or mint a passphrase for *mode*; return ``(passphrase, source, auto_generated)``.

    The third element flags whether this call *minted* a fresh
    passphrase the operator hasn't typed themselves — used by
    [`_handle_credentials_encrypt_db`][terok_sandbox.commands.credentials._handle_credentials_encrypt_db]
    to decide whether the ack flow needs to run.  Reusing an existing
    session-file / keyring / config entry returns ``False`` because
    the operator (or the previous run) already saw the value.
    """
    from .._yaml import write_secret_text
    from ..vault.store.encryption import (
        generate_passphrase,
        load_passphrase_from_file,
        load_passphrase_from_keyring,
        prompt_passphrase,
        store_passphrase_in_keyring,
    )

    if mode == "session-file":
        existing = load_passphrase_from_file(cfg.vault_passphrase_file)
        if existing is not None:
            return existing, "session-file", False
        # ``prompt_passphrase(confirm=True)`` mints-and-announces an
        # auto-generated value on empty input, or echo-confirms a typed
        # one.  We can't distinguish the two outcomes from here, so we
        # report ``auto_generated=True`` conservatively — re-acking a
        # value the operator typed is a no-op the second time round
        # but missing the ack on an auto-mint loses the recovery key.
        new = prompt_passphrase(confirm=True)
        write_secret_text(cfg.vault_passphrase_file, new + "\n")
        return new, "session-file", True

    if mode == "keyring":
        existing = load_passphrase_from_keyring()
        if existing is not None:
            return existing, "keyring", False
        new = generate_passphrase()
        if store_passphrase_in_keyring(new):
            _announce_generated_passphrase(new, echo_to_stdout=echo_passphrase)
            return new, "keyring", True
        raise RuntimeError("OS keyring is unreachable or denied; choose a different storage mode")

    if mode == "config":
        if cfg.credentials_passphrase:
            return cfg.credentials_passphrase, "config", False
        # ``prompt_passphrase(confirm=True)`` mints-and-announces on
        # empty input, so this branch can yield either an operator-typed
        # passphrase or a freshly-generated one — we can't tell from the
        # return value.  Report ``auto_generated=True`` conservatively;
        # the ack flow is a mild "type SAVED" prompt at worst when the
        # operator typed the value, and the correct gate when they let
        # the helper mint one.  Same trade as the session-file branch.
        print("Enter a passphrase to write to credentials.passphrase in config.yml:")
        return prompt_passphrase(confirm=True), "prompt", True

    raise ValueError(f"unknown mode: {mode!r}")


def _announce_generated_passphrase(passphrase: str, *, echo_to_stdout: bool = False) -> None:
    """Show an auto-minted passphrase to the operator's controlling terminal.

    Routes through ``_write_to_controlling_tty`` so a redirected
    install — ``terok-sandbox setup > install.log``, a CI job, an
    Ansible play — can't capture the recovery key into a journal or
    log artifact.

    *echo_to_stdout* is the opt-in for the no-TTY case: CI / Ansible
    drivers explicitly pass ``--echo-passphrase`` to ``setup`` so they
    can capture the value into their own secret manager.  When set, the
    ``/dev/tty`` write becomes best-effort — a TTY-less CI run that
    *did* pass ``--echo-passphrase`` would otherwise hit the fail-closed
    ``SystemExit`` in [`_write_to_controlling_tty`][terok_sandbox.vault.store.encryption._write_to_controlling_tty]
    before the stdout copy ever lands, defeating the documented escape
    hatch.  Without ``--echo-passphrase`` the controlling-TTY write
    stays required so a redirected ``setup > install.log`` never
    silently drops the recovery key.
    """
    from ..vault.store.encryption import _write_to_controlling_tty

    message = (
        f"\nVault passphrase: {passphrase}\n"
        "  Write this down — it's your recovery key for rebuilds and other hosts.\n"
    )
    _write_to_controlling_tty(message, required=not echo_to_stdout)
    if echo_to_stdout:
        print(message, end="")


def _persist_mode_choice(mode: SetupTier, passphrase: str) -> None:
    """Write the chosen mode into config.yml so the chain re-resolves next time.

    Session mode needs no change — the tmpfs file is self-describing.
    Both fields (``use_keyring`` and ``passphrase``) are always written
    so switching modes leaves a clean, single-source state and the
    resolution chain can't see two tiers claiming ownership.
    """
    from .. import config as _config
    from ..paths import _config_file_paths

    user_config = next((p for label, p in _config_file_paths() if label == "user"), None)
    if user_config is None or mode == "session-file":
        return
    updates: dict[str, object | None] = (
        {"use_keyring": True, "passphrase": None}  # nosec: B105 — clearing a config key
        if mode == "keyring"
        else {"use_keyring": False, "passphrase": passphrase}
    )
    _yaml_update_section(user_config, "credentials", updates)
    # The chain reads through ``_credentials_section``'s lru_cache;
    # without invalidation the same process keeps seeing the
    # pre-setup state.
    _config._credentials_section.cache_clear()


def _back_up_plaintext_db(db_path: Path) -> Path:
    """Tar the plaintext DB + WAL/SHM sidecars next to it, return the tarball path.

    The tarball holds cleartext secrets; pre-create the file at 0o600
    via ``O_CREAT | O_EXCL`` and stream the tar into that fd.  A
    ``chmod`` after ``tarfile.open(path)`` would leave the secrets
    world-readable on hosts with a permissive umask for the duration
    of the write.
    """
    import datetime
    import os
    import tarfile

    stamp = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
    backup_path = db_path.with_name(f"{db_path.name}.plaintext-backup-{stamp}.tar.gz")
    fd = os.open(backup_path, os.O_CREAT | os.O_EXCL | os.O_WRONLY, 0o600)
    try:
        with os.fdopen(fd, "wb") as raw, tarfile.open(fileobj=raw, mode="w:gz") as tar:
            tar.add(db_path, arcname=db_path.name)
            for suffix in ("-wal", "-shm", "-journal"):
                sidecar = db_path.with_name(db_path.name + suffix)
                if sidecar.exists():
                    tar.add(sidecar, arcname=sidecar.name)
    except BaseException:
        backup_path.unlink(missing_ok=True)
        raise
    return backup_path


def _warn_about_plaintext_backup(backup_path: Path) -> None:
    """Surface the backup path in red so the operator can't miss it."""
    use_color = sys.stderr.isatty()
    red = "\033[1;31m" if use_color else ""
    reset = "\033[0m" if use_color else ""
    print(
        f"\n{red}WARNING: plaintext backup written to {backup_path}{reset}\n"
        f"{red}         It contains your secrets in cleartext."
        f" Once you have confirmed the migration is good, delete it:{reset}\n"
        f"{red}           rm {backup_path}{reset}",
        file=sys.stderr,
    )


def _run_credentials_setup_phase(
    cfg: SandboxConfig,
    *,
    echo_passphrase: bool = False,
    passphrase_tier: str | None = None,
) -> bool:
    """Migrate the credentials DB to SQLCipher; no-op on already-encrypted or absent.

    The vault daemon, if installed from a previous setup, holds a write
    lock on the plaintext DB (WAL mode); quiesce it so
    ``sqlcipher_export`` can ATTACH the source without colliding.
    ``run_vault_install_phase`` re-installs+starts the daemon a moment
    later, so this stop is transient by design.

    On "database is locked" the phase self-heals: socket activation can
    re-spawn the service between our stop and the migration's open, so
    we uninstall the units to kill the trigger entirely, then retry
    once before bailing.
    """
    from ..vault.daemon.lifecycle import VaultManager

    print("→ credentials", end="", flush=True)
    mgr = VaultManager(cfg)
    _quiesce_vault_for_migration(mgr)
    try:
        _handle_credentials_encrypt_db(
            cfg=cfg, echo_passphrase=echo_passphrase, passphrase_tier=passphrase_tier
        )
    except Exception as exc:  # noqa: BLE001
        if not _looks_like_db_lock(exc):
            print(f" — FAILED: {exc}")
            return False
        print(" — locked, auto-recovering by uninstalling vault units …", flush=True)
        _quiesce_vault_for_migration(mgr, force_uninstall=True)
        try:
            _handle_credentials_encrypt_db(
                cfg=cfg, echo_passphrase=echo_passphrase, passphrase_tier=passphrase_tier
            )
        except Exception as retry_exc:  # noqa: BLE001
            print(f"  recovery FAILED: {retry_exc}")
            if _looks_like_db_lock(retry_exc):
                print(
                    "  Hint: another process still holds the DB.  Find it with:\n"
                    "    fuser -v " + str(cfg.db_path) + "\n"
                    "  stop it, then re-run `terok setup`."
                )
            return False
        print("  recovered.")
    return True


def _quiesce_vault_for_migration(
    mgr: Any,
    *,
    force_uninstall: bool = False,
) -> None:
    """Best-effort quiesce of the vault daemon so it doesn't hold the DB.

    ``stop_daemon`` itself is idempotent (handles both systemd-managed
    and PID-file daemons); ``force_uninstall=True`` additionally removes
    the unit files so socket activation can't race-respawn the service.
    """
    with contextlib.suppress(Exception):
        mgr.stop_daemon()
    if force_uninstall:
        with contextlib.suppress(Exception):
            mgr.uninstall_systemd_units()


def _looks_like_db_lock(exc: BaseException) -> bool:
    """Return ``True`` if *exc* is sqlite's "database is locked" complaint."""
    return "database is locked" in str(exc).lower()


#: The credentials-DB management group exposed at sandbox's top level.
CREDENTIALS_COMMANDS: tuple[CommandDef, ...] = (
    CommandDef(
        name="credentials",
        help="Credentials DB management",
        children=(
            CommandDef(
                name="encrypt-db",
                help=(
                    "Migrate a legacy plaintext credentials DB to SQLCipher-encrypted "
                    "(deprecated in 0.9.0, removed in 0.10.0)"
                ),
                handler=_handle_credentials_encrypt_db,
            ),
        ),
    ),
)


__all__ = ["CREDENTIALS_COMMANDS", "_run_credentials_setup_phase"]
