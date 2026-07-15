# SPDX-FileCopyrightText: 2026 Jiri Vyskocil
# SPDX-License-Identifier: Apache-2.0

"""Vault passphrase CLI verbs — session unlock / lock plus passphrase management.

The unlock/lock pair drives the session-tier slot of the SQLCipher
passphrase resolution chain: ``unlock`` lands a passphrase on the
session-unlock tmpfs file; ``lock`` removes it.  Everything else lives
under ``vault passphrase``:

- ``vault passphrase seal`` promotes the current passphrase into a
  machine-bound ``systemd-creds`` credential.
- ``vault passphrase to-keyring`` moves it from whichever tier holds it
  now into the OS keyring (the recommended upgrade path off the
  session-file / plaintext-config tiers).
- ``vault passphrase reveal`` resolves and prints the current
  passphrase (to ``/dev/tty`` by default, or stdout with
  ``--allow-redirect``) and offers to mark the recovery key as saved.
- ``vault passphrase acknowledge`` marks the current passphrase as
  saved without displaying it — the silent ack a TUI / CI captures.
- ``vault passphrase change`` re-encrypts the DB under a new
  passphrase and rewrites every tier that stores the old one —
  [`change_passphrase`][terok_sandbox.commands.vault.change_passphrase]
  is the prompt-free core the TUI shares.
``vault lock`` clears every stored copy of the passphrase — session
file, keyring, sealed systemd-creds credential, and plaintext config —
so the vault becomes irrecoverable without an off-host copy.  The
machine-bound tiers are an automatic-unlock convenience on top of a
passphrase the operator is expected to have saved; locking peels them
away.  ``purge_passphrase_tiers`` is the prompt-free core ``lock`` and
panic share.

Each container mounts its own short-lived
[`VaultProxy`][terok_sandbox.vault.daemon.token_broker.VaultProxy]
that resolves the passphrase on demand.  ``vault unlock`` / ``vault
lock`` therefore only manage the passphrase tier; a supervisor that's
already running keeps the passphrase it resolved at spawn, so picking
up a changed tier means starting a fresh task (delete the matching
one — per the no-state-preservation rule).
"""

from __future__ import annotations

import sys
from dataclasses import dataclass
from typing import TYPE_CHECKING

from terok_util import LazyHandler, sanitize_tty

from ..vault.store.tiers import PassphraseTier
from ._types import ArgDef, CommandDef

if TYPE_CHECKING:
    from ..config import SandboxConfig
    from ..vault.store.status import VaultStatus
    from ..vault.store.systemd_creds import KeyMode


def _resolve_cfg(cfg: SandboxConfig | None) -> SandboxConfig:
    """Return *cfg*, or a default [`SandboxConfig`][terok_sandbox.SandboxConfig] built lazily.

    Keeps the default-config import (pydantic, the credential store) out
    of module import time so building the command registry stays cheap.
    """
    if cfg is not None:
        return cfg
    from ..config import SandboxConfig

    return SandboxConfig()


@dataclass(frozen=True)
class SessionProvisionResult:
    """Outcome of [`provision_session_passphrase`][terok_sandbox.commands.vault.provision_session_passphrase].

    ``written`` is the load-bearing bit: ``False`` means the write was
    *refused* because a durable tier already resolves the vault, so a
    session file would only shadow it (``shadowed_durable`` names that
    tier).  Refusal is a normal outcome, not an error — callers report
    it ("already unlocked via X") rather than raising.  ``validated``
    says whether the written value was test-opened against an existing
    DB (vs. an empty install where it becomes the key on first use).
    """

    written: bool
    validated: bool = False
    shadowed_durable: PassphraseTier | None = None


def provision_session_passphrase(
    cfg: SandboxConfig, passphrase: str, *, force: bool = False
) -> SessionProvisionResult:
    """Validate *passphrase* against the DB, then land it on the session tier.

    The single writer of the session-unlock tmpfs file — the CLI
    ``vault unlock`` and terok's TUI unlock modal both funnel through
    here, so the no-shadow and validation guards apply to every caller
    by construction; neither can store a value the DB rejects, nor
    silently shadow a working durable tier.

    Two guards, in order:

    1. **No-shadow.** The session tier is the highest-priority slot, so
       writing it masks any durable tier (systemd-creds / keyring /
       config) underneath — a reboot then wipes the session copy and the
       operator only *thought* the sealed key was in use.  When a durable
       tier already resolves and *force* is false, nothing is written and
       the result reports ``written=False`` + the shadowed tier.  *force*
       (re-key / deliberate override) skips this guard.
    2. **Validation.** When the DB exists (and isn't a legacy plaintext
       file) the value is test-opened first; a mismatch raises
       [`WrongPassphraseError`][terok_sandbox.WrongPassphraseError] and
       **nothing is written**.  A missing DB skips validation (opening it
       just to check would create it as a side effect) — the value
       becomes its key on first use.
    """
    from .._yaml import write_secret_text
    from ..vault.store.db import CredentialDB
    from ..vault.store.encryption import is_plaintext_sqlite
    from ..vault.store.status import active_durable_source

    if not force:
        shadowed = active_durable_source(cfg)
        if shadowed is not None:
            return SessionProvisionResult(written=False, shadowed_durable=shadowed)

    validated = False
    if cfg.db_path.exists() and not is_plaintext_sqlite(cfg.db_path):
        # Raises WrongPassphraseError / PlaintextDBFoundError on mismatch —
        # deliberately before the write so a bad value never lands.
        CredentialDB(cfg.db_path, passphrase=passphrase).close()
        validated = True
    write_secret_text(cfg.vault_passphrase_file, passphrase + "\n")
    return SessionProvisionResult(written=True, validated=validated)


def _handle_vault_unlock(*, cfg: SandboxConfig | None = None, force: bool = False) -> None:
    """Write the credentials-DB passphrase to the session-unlock tmpfs file.

    Both guards live in
    [`provision_session_passphrase`][terok_sandbox.commands.vault.provision_session_passphrase]
    (no-shadow + DB validation).  This handler adds only CLI ergonomics:
    a pre-prompt durable-tier check so the operator isn't asked to type a
    passphrase the writer would just refuse, and exit codes / messages
    for each outcome.
    """
    from ..vault.store.db import PlaintextDBFoundError
    from ..vault.store.encryption import WrongPassphraseError, prompt_passphrase
    from ..vault.store.status import active_durable_source

    cfg = _resolve_cfg(cfg)

    # Skip the prompt entirely when the writer would refuse anyway — same
    # guard the writer applies, run early purely so we don't ask for a
    # value we'd discard.
    if not force and (durable := active_durable_source(cfg)) is not None:
        print(
            f"vault already auto-unlocks via {durable}; not writing a session file"
            " (it would shadow the durable tier and be lost on the next reboot)."
            " Re-run with --force to override."
        )
        return

    try:
        result = provision_session_passphrase(cfg, prompt_passphrase(), force=force)
    except WrongPassphraseError:
        raise SystemExit(
            f"that passphrase does not open {cfg.db_path} — nothing was written.\n"
            "  If you never saved the passphrase, see"
            " `terok-sandbox vault passphrase reveal` (while a tier still"
            " resolves it) or the recovery section of the docs."
        ) from None
    except PlaintextDBFoundError as exc:
        raise SystemExit(str(exc)) from exc

    print(f"→ wrote passphrase to {cfg.vault_passphrase_file} (RAM-backed, cleared on reboot)")
    if result.validated:
        print("  verified: the value opens the credentials DB")
    else:
        print(f"  no DB at {cfg.db_path} yet — this value becomes its key on first use")


def _forget_config_tier_updates(cfg: SandboxConfig) -> dict[str, object | None]:
    """Return the config-section patch ``purge_passphrase_tiers`` should apply.

    The ``passphrase_command`` wiring is an auto-resolution hook —
    leaving it would let a future supervisor re-unlock from disk and
    defeat the lock.
    """
    updates: dict[str, object | None] = {}
    if cfg.credentials_passphrase_command:
        updates["passphrase_command"] = None
    return updates


def purge_passphrase_tiers(cfg: SandboxConfig) -> None:
    """Remove every stored copy of the credentials-DB passphrase.

    Clears the session-unlock tmpfs file, the OS keyring entry, the
    sealed systemd-creds credential, and the
    ``credentials.passphrase_command`` wiring in ``config.yml`` — then
    drops the recovery-acknowledged marker, since it's meaningless once
    no tier remains.  After this the
    vault can only be reopened by re-supplying the passphrase
    (``vault unlock``); it is **unrecoverable** without an off-host copy.

    No prompts and no acknowledgement check: this is the raw destructive
    action.  The ``lock`` verb gates it behind a typed-``SAVED``
    confirmation when recovery is unacknowledged; panic calls it
    directly — no questions asked.
    """
    from ..vault.store.encryption import forget_passphrase_in_keyring, load_passphrase_from_keyring

    path = cfg.vault_passphrase_file
    if path.exists():
        path.unlink()
        print(f"→ removed {path}")

    if cfg.credentials_use_keyring:
        if forget_passphrase_in_keyring():
            print("→ cleared keyring entry")
        elif load_passphrase_from_keyring() is None:
            # ``keyring.delete_password`` raises on a missing entry on most
            # backends, which the helper folds to False — a residual entry
            # after that means the backend rejected the delete.
            print("→ keyring entry already absent")
        else:
            raise SystemExit(
                "failed to clear keyring entry;"
                " future supervisors may still auto-unlock from keyring"
            )

    config_updates = _forget_config_tier_updates(cfg)
    if config_updates:
        from .. import config as _config
        from .._yaml import update_section as _yaml_update_section
        from ..paths import config_file_paths

        user_config = next((p for label, p in config_file_paths() if label == "user"), None)
        if user_config is not None and user_config.exists():
            _yaml_update_section(user_config, "credentials", config_updates)
            _config._credentials_section.cache_clear()
            for key in config_updates:
                print(f"→ cleared credentials.{key} from config.yml")

    sealed_cred = cfg.vault_systemd_creds_file
    if sealed_cred.exists():
        try:
            sealed_cred.unlink()
        except OSError as exc:
            raise SystemExit(f"failed to remove sealed credential at {sealed_cred}: {exc}") from exc
        print(f"→ removed sealed credential at {sealed_cred}")

    from ..vault.store.recovery import forget as forget_recovery_marker

    forget_recovery_marker(cfg.vault_recovery_marker_file)


def _confirm_lock_when_unacknowledged(cfg: SandboxConfig, *, force: bool) -> None:
    """Require a typed ``SAVED`` before locking an unconfirmed vault.

    Locking clears every stored copy, so it's irreversible without an
    off-host passphrase — the escrow-before-destroy gate the survey of
    BitLocker / FileVault recommends.  Skipped when the operator has
    already acknowledged saving the key (then a re-supply is trivial) or
    passed ``--force`` (CI / scripted).  Fails closed on a headless run:
    no controlling TTY to answer the prompt means the lock is aborted.
    """
    from ..vault.store.encryption import _read_from_controlling_tty
    from ..vault.store.recovery import acknowledged

    if force or acknowledged(cfg.vault_recovery_marker_file):
        return

    print(
        "Locking clears EVERY stored copy of the vault passphrase (session file,"
        " keyring, sealed systemd-creds credential, config.yml).  You have not"
        " confirmed an off-host copy, so the vault would become UNRECOVERABLE."
    )
    response = _read_from_controlling_tty("Type SAVED to confirm you have it stored elsewhere: ")
    if response != "SAVED":
        raise SystemExit(
            "lock aborted — passphrase not confirmed saved."
            "  Save it (see `vault passphrase reveal`) then retry, or pass --force."
        )


def _handle_vault_lock(*, cfg: SandboxConfig | None = None, force: bool = False) -> None:
    """Lock the vault: clear every stored copy of the passphrase.

    "Locked" means what an operator expects — the next open needs the
    passphrase again.  Against a machine-bound tier (systemd-creds /
    keyring) there is no honest half-measure: a soft-lock that leaves the
    sealed key in place still auto-unlocks on any access (the
    BitLocker-Suspend trap), so locking removes the stored copies
    outright.  The systemd-creds / keyring tiers are an *automatic-unlock
    convenience* layered on top of a passphrase you are expected to have
    saved — locking peels them away.

    Reversible only by re-supplying that passphrase via ``vault unlock``;
    unconfirmed vaults are gated behind a typed confirmation.
    """
    cfg = _resolve_cfg(cfg)
    _confirm_lock_when_unacknowledged(cfg, force=force)
    purge_passphrase_tiers(cfg)


#: CLI verbs for ``vault seal --key=``, mapped to systemd-creds' own
#: ``--with-key=`` vocabulary.  ``"auto"`` is the natural default —
#: systemd already picks ``host+tpm2`` on TPM-equipped hosts and
#: ``host`` otherwise, and second-guessing that decision here would
#: silently weaken the dual-factor default.
_SEAL_KEY_MODES: dict[str, KeyMode] = {
    "auto": "auto",
    "tpm": "tpm2",
    "host": "host",
    "tpm+host": "host+tpm2",
}


def _require_recovery_acknowledged(cfg: SandboxConfig, *, tier: str) -> None:
    """Refuse to enable a machine-bound auto-unlock tier until recovery is acknowledged.

    Escrow-before-enable, the BitLocker / FileVault model: a
    systemd-creds or keyring tier auto-unlocks the vault on *this*
    machine / account, so an off-host copy of the passphrase is the
    operator's only recovery if the hardware or account is lost.  Block
    the upgrade until they've confirmed they saved it.
    """
    from ..vault.store.recovery import acknowledged

    if acknowledged(cfg.vault_recovery_marker_file):
        return
    raise SystemExit(
        f"cannot enable the {tier} tier: the recovery passphrase isn't marked as saved.\n"
        "  This tier auto-unlocks the vault on this machine, so an off-host copy of the\n"
        "  passphrase is your only recovery if the hardware fails.  Save it first, then\n"
        "  `terok-sandbox vault passphrase reveal` (shows it + offers to confirm) or\n"
        "  `terok-sandbox vault passphrase acknowledge`."
    )


def handle_vault_seal(*, cfg: SandboxConfig | None = None, key: str = "auto") -> None:
    """Seal the credentials-DB passphrase into a systemd-creds credential.

    Adds the systemd-creds tier to the resolution chain: machine-bound
    (TPM2 + host key, or either alone), survives reboot, no OS
    keyring required.  After sealing, every new supervisor resolves the
    passphrase via ``systemd-creds decrypt`` on start — no operator
    interaction needed at boot, no plaintext-on-disk.

    Requires an already-resolvable passphrase — typically from a fresh
    ``vault unlock`` in the current session.
    """
    from ..vault.store import systemd_creds
    from ..vault.store.encryption import WrongPassphraseError

    cfg = _resolve_cfg(cfg)

    if not systemd_creds.is_available():
        raise SystemExit(
            "systemd-creds unavailable: needs systemd ≥ 257 with the Varlink"
            " io.systemd.Credentials interface (Fedora ≥ 42, Debian ≥ 13)"
        )

    key_mode = _SEAL_KEY_MODES.get(key)
    if key_mode is None:
        choices = ", ".join(sorted(_SEAL_KEY_MODES))
        raise SystemExit(f"unknown --key value: {key!r} (expected one of: {choices})")

    # A prompt here would accept a freshly-typed value and seal *that*,
    # leaving the next chain walk holding a key that doesn't open the DB.
    try:
        passphrase = cfg.resolve_passphrase()
    except WrongPassphraseError as exc:
        raise SystemExit(f"cannot seal: {exc}") from exc
    if passphrase is None:
        raise SystemExit("no current passphrase to seal — run `terok-sandbox vault unlock` first")
    _require_recovery_acknowledged(cfg, tier="systemd-creds")

    try:
        systemd_creds.seal(passphrase, cfg.vault_systemd_creds_file, key_mode=key_mode)
    except RuntimeError as exc:
        # ``tpm2`` requested on a TPM-less host surfaces as a CalledProcessError
        # bubbled to RuntimeError — pass it through with the hint attached.
        raise SystemExit(str(exc)) from exc

    print(f"→ sealed passphrase to {cfg.vault_systemd_creds_file} (--with-key={key_mode})")

    # The passphrase now lives in the durable sealed credential, so the
    # session file is redundant — and, being higher priority, it would
    # *shadow* the seal until the next reboot wiped it.  Drop it so the
    # chain resolves from the tier the operator just established (same
    # cleanup ``to-keyring`` does).
    if cfg.vault_passphrase_file.exists():
        cfg.vault_passphrase_file.unlink()
        print(f"→ removed now-redundant session file {cfg.vault_passphrase_file}")

    print(
        "  the resolution chain will pick this up the next time a supervisor"
        " starts; no restart required"
    )


def handle_vault_to_keyring(*, cfg: SandboxConfig | None = None) -> None:
    """Move the current passphrase from its current tier into the OS keyring.

    Resolves the passphrase via the chain (or prompts as a last resort),
    writes it to the keyring, flips ``credentials.use_keyring`` to true
    in ``config.yml``, clears any plaintext ``credentials.passphrase`` /
    ``credentials.passphrase_command`` wiring, and removes the
    session-file and sealed systemd-creds copies.

    The validate-before-destroy ordering is deliberate: if the keyring
    write fails, the source tier is still intact.
    """
    from .. import config as _config
    from .._yaml import update_section as _yaml_update_section
    from ..vault.store.encryption import (
        WrongPassphraseError,
        store_passphrase_in_keyring,
    )

    cfg = _resolve_cfg(cfg)

    try:
        passphrase, source = cfg.resolve_passphrase_with_source(prompt_on_tty=True)
    except WrongPassphraseError as exc:
        raise SystemExit(f"cannot move to keyring: {exc}") from exc

    if not passphrase:
        raise SystemExit("no current passphrase resolvable; run `terok-sandbox vault unlock` first")
    if source == "keyring":
        print("→ passphrase is already in the keyring; nothing to do")
        return
    _require_recovery_acknowledged(cfg, tier="keyring")

    if not store_passphrase_in_keyring(passphrase):
        raise SystemExit("OS keyring is unreachable or denied; aborting (nothing was changed)")
    print(f"→ stored passphrase in keyring (was: {source})")

    # Switch the config's tier wiring atomically: flip use_keyring on,
    # drop the plaintext + helper fallbacks so the chain can't re-resolve
    # via a stale lower tier.
    from ..paths import config_file_paths

    user_config = next((p for label, p in config_file_paths() if label == "user"), None)
    if user_config is not None:
        # nosec: B105 — clearing config keys to None, not hardcoding secrets
        updates = {  # nosec: B105
            "use_keyring": True,
            "passphrase": None,  # nosec: B105
            "passphrase_command": None,  # nosec: B105
        }
        _yaml_update_section(user_config, "credentials", updates)
        _config._credentials_section.cache_clear()
        print(f"→ updated {user_config} (use_keyring: true, plaintext fields cleared)")

    # Remove the old tier's persistent copy.  Session file is removed
    # because the chain prefers it over keyring; sealed systemd-creds
    # likewise outranks keyring on the resolution order.
    for stale in (cfg.vault_passphrase_file, cfg.vault_systemd_creds_file):
        if stale.exists():
            stale.unlink()
            print(f"→ removed {sanitize_tty(str(stale))}")


def _handle_vault_passphrase_reveal(
    *, cfg: SandboxConfig | None = None, allow_redirect: bool = False
) -> None:
    """Print the currently-resolvable vault passphrase and offer to ack the recovery.

    The cleartext routes to ``/dev/tty`` by default so a routine
    ``terok-sandbox vault passphrase reveal > out`` can't capture the
    recovery key into a file by accident — the same channel the auto-mint
    flow uses for its announcement.  ``--allow-redirect`` flips the
    output to stdout for the operator who actually does want to pipe
    the value into another tool (``pass insert``, ``op item create``);
    in that mode stdout carries **only** the passphrase so the pipe
    receives clean payload — every banner, reminder, and ack message
    is diverted to stderr / ``/dev/tty`` so the redirected output
    stays usable as the secret itself.

    After a successful reveal the operator is prompted whether to mark
    the current passphrase as saved.  An affirmative answer writes the
    fingerprint marker the unconfirmed-recovery warning checks for,
    closing the loop without a separate ``vault passphrase acknowledge``
    invocation.
    """
    from ..vault.store.encryption import (
        WrongPassphraseError,
        _read_from_controlling_tty,
        _write_to_controlling_tty,
    )
    from ..vault.store.recovery import acknowledge, acknowledged

    cfg = _resolve_cfg(cfg)

    try:
        passphrase, source = cfg.resolve_passphrase_with_source(prompt_on_tty=True)
    except WrongPassphraseError as exc:
        raise SystemExit(f"cannot reveal passphrase: {exc}") from exc
    if not passphrase:
        raise SystemExit("no current passphrase resolvable; run `terok-sandbox vault unlock` first")

    if allow_redirect:
        # Pipe-friendly: stdout carries *only* the passphrase string
        # + trailing newline, suitable for ``| pass insert -e ...``
        # or ``| op item create``.  The stderr banner deliberately
        # does NOT echo the passphrase — many CI/log setups persist
        # stderr by default and an operator piping stdout into a
        # secret manager would expect stdout to be the sole carrier
        # of the secret (audit finding #1 on PR #325).
        print(passphrase)
        safe_banner = (
            f"\nVault passphrase ({source}) printed to stdout.\n"
            "  Recovery key — save it off-host"
            " (1Password emergency kit, paper safe, sealed envelope).\n"
        )
        print(safe_banner, end="", file=sys.stderr)
    else:
        if sys.stdout.isatty():
            print(
                "  (passphrase routed to /dev/tty so a redirected stdout"
                " can't capture it; pass --allow-redirect to print to stdout)"
            )
        _write_to_controlling_tty(
            f"\nVault passphrase ({source}): {passphrase}\n"
            "  Recovery key — save it off-host"
            " (1Password emergency kit, paper safe, sealed envelope).\n"
        )

    # In ``--allow-redirect`` mode every subsequent UX line also has to
    # go to stderr so the stdout pipe stays a pure-secret payload.  The
    # closure below picks the right sink once and reuses it.
    def _say(text: str) -> None:
        if allow_redirect:
            print(text, file=sys.stderr)
        else:
            print(text)

    if acknowledged(cfg.vault_recovery_marker_file):
        _say("  recovery key already marked as saved.")
        return

    response = _read_from_controlling_tty("Mark recovery key as saved? Type SAVED to confirm: ")
    if response is None:
        _say(
            "  no controlling TTY for confirmation;"
            " run `terok-sandbox vault passphrase acknowledge` separately"
            " once you have saved the value."
        )
        return
    if response.strip() == "SAVED":
        acknowledge(cfg.vault_recovery_marker_file)
        _say("  recovery key marked as saved.")
    else:
        _say("  recovery key NOT confirmed; unconfirmed-recovery warning stays on.")


def _handle_vault_passphrase_acknowledge(*, cfg: SandboxConfig | None = None) -> None:
    """Mark the recovery key as saved, without displaying any passphrase.

    The silent counterpart of
    [`_handle_vault_passphrase_reveal`][terok_sandbox.commands.vault._handle_vault_passphrase_reveal]
    — used by the TUI after it has shown the passphrase in its own
    modal, and by CI bootstraps that captured the value via
    ``--echo-passphrase`` and have now stashed it in their secret
    manager.  Independent of the vault-lock state: the marker is a
    zero-byte file, so we can flip it on without resolving anything.
    Idempotent — calling again with the marker already in place is a
    silent no-op.
    """
    from ..vault.store.recovery import acknowledge, acknowledged

    cfg = _resolve_cfg(cfg)

    if acknowledged(cfg.vault_recovery_marker_file):
        print("recovery key already marked as saved.")
        return
    acknowledge(cfg.vault_recovery_marker_file)
    print("recovery key marked as saved.")


@dataclass(frozen=True)
class TierRewrite:
    """What happened to one passphrase-holding tier during a change.

    ``ok`` means the tier now holds the *new* passphrase.  A failed
    rewrite (``ok=False``) is reported, never raised — by the time the
    fan-out runs the DB is already rekeyed, so aborting would only
    hide which tiers still need the operator's attention.
    """

    tier: PassphraseTier
    ok: bool
    detail: str


@dataclass(frozen=True)
class PassphraseChangeResult:
    """Outcome of [`change_passphrase`][terok_sandbox.commands.vault.change_passphrase].

    ``generated`` carries the same follow-up duty as
    [`TierProvisionResult`][terok_sandbox.commands.credentials.TierProvisionResult]:
    a minted value must be revealed to the operator.  The recovery
    marker has been dropped either way — whoever renders this result
    owns re-running the acknowledgement flow.
    """

    passphrase: str
    """The new passphrase — mint or caller-supplied."""

    generated: bool
    """``True`` iff this call minted the value (caller passed ``None``)."""

    rekeyed: bool
    """``True`` iff an existing DB was re-encrypted (``False`` on a
    fresh install where the new value simply becomes the key on first
    use)."""

    rewrites: tuple[TierRewrite, ...]
    """Per-tier outcomes, in resolution-chain order."""

    @property
    def problems(self) -> tuple[TierRewrite, ...]:
        """The rewrites that failed — non-empty means the operator has cleanup to do."""
        return tuple(rewrite for rewrite in self.rewrites if not rewrite.ok)


def change_passphrase(
    cfg: SandboxConfig | None = None,
    *,
    old: str | None = None,
    new: str | None = None,
) -> PassphraseChangeResult:
    """Re-encrypt the vault under a new passphrase and rewrite every tier holding the old one.

    The prompt-free core shared by the ``vault passphrase change`` CLI
    verb and the TUI — same contract shape as
    [`provision_passphrase_tier`][terok_sandbox.commands.credentials.provision_passphrase_tier]:
    no ``/dev/tty``, no ack prompt; the caller owns the conversation.

    *old* is only consulted when the resolution chain can't produce the
    current passphrase (locked vault) — when a caller supplies it
    explicitly it wins over the chain, so a stale tier value can't
    override an operator who knows better.  *new* ``None`` mints a
    fresh [`generate_passphrase`][terok_sandbox.vault.store.encryption.generate_passphrase]
    value (``generated=True`` in the result — reveal it!).

    The ordering is the safety argument:

    1. Verify + rekey the DB first ([`rekey_in_place`][terok_sandbox.vault.store.encryption.rekey_in_place]).
       Every failure here — wrong old passphrase, a live supervisor
       holding the WAL ("database is locked") — aborts with **nothing
       changed anywhere**.
    2. Only then fan the new value out to every tier that currently
       holds material, in resolution order, collecting per-tier
       outcomes instead of raising: a tier that can't take the new
       value (keyring denied, systemd-creds host regressed) is purged
       where possible so no tier keeps resolving the *old* passphrase,
       and reported either way.
    3. Drop the recovery-acknowledged marker — the saved copy the
       operator confirmed is now the wrong passphrase.

    Refuses up front (``RuntimeError``) while ``passphrase_command`` is
    configured: that tier's secret lives in a store the operator owns,
    so the sandbox rewriting everything else would leave the helper
    resolving a stale value that fails closed on the next boot.  Update
    the external store first, or remove the wiring.

    Raises [`NoPassphraseError`][terok_sandbox.NoPassphraseError] when
    neither the chain nor *old* yields the current passphrase,
    [`WrongPassphraseError`][terok_sandbox.WrongPassphraseError] when
    that value doesn't open the DB, and [`ValueError`][ValueError] for
    an empty or unchanged *new*.
    """
    from ..vault.store.encryption import (
        NoPassphraseError,
        generate_passphrase,
        is_plaintext_sqlite,
        probe_passphrase_chain,
        rekey_in_place,
    )
    from ..vault.store.recovery import forget as forget_recovery_marker

    cfg = _resolve_cfg(cfg)

    if cfg.credentials_passphrase_command:
        raise RuntimeError(
            "credentials.passphrase_command is configured — the passphrase lives in"
            " an external secret store terok cannot write to.  Update the secret"
            " there first (or remove passphrase_command from config.yml), then"
            " re-run the change."
        )
    if new == "":  # nosec: B105 — rejecting the empty sentinel, not comparing a secret
        raise ValueError("refusing an empty passphrase (SQLCipher reads it as no encryption)")
    if cfg.db_path.exists() and is_plaintext_sqlite(cfg.db_path):
        raise RuntimeError(
            f"{cfg.db_path} is still the legacy plaintext format — run the"
            " encrypt-db migration before changing the passphrase"
        )

    # Explicit *old* wins; the chain fills in for the common unlocked case.
    current = old or cfg.resolve_passphrase()  # raises WrongPassphraseError on a broken tier
    db_exists = cfg.db_path.exists()
    if current is None:
        if db_exists:
            raise NoPassphraseError(
                "the vault is locked — supply the current passphrase to change it"
            )
        raise NoPassphraseError(
            "no credentials DB and no stored passphrase — provision the vault"
            " (setup) instead of changing it"
        )

    generated = new is None
    if new is None:
        new = generate_passphrase()
    if new == current:
        raise ValueError("the new passphrase is identical to the current one")

    if db_exists:
        rekey_in_place(cfg.db_path, current, new)

    present = [
        row.source
        for row in probe_passphrase_chain(
            passphrase_file=cfg.vault_passphrase_file,
            systemd_creds_file=cfg.vault_systemd_creds_file,
            use_keyring=cfg.credentials_use_keyring,
            passphrase_command=cfg.credentials_passphrase_command,
        )
        if row.present
    ]
    if not present:
        # Locked vault changed via an explicitly-supplied *old*: nothing
        # holds material yet, so land the new value where `vault unlock`
        # would — otherwise the change succeeds and nobody can open the DB.
        present = [PassphraseTier.SESSION_FILE]
    rewrites = tuple(_rewrite_tier(cfg, tier, new) for tier in present)

    # The confirmed-saved copy (if any) is now the wrong passphrase —
    # the marker doesn't auto-invalidate (deliberately fingerprint-free,
    # see vault.store.recovery), so the change flow must drop it.
    forget_recovery_marker(cfg.vault_recovery_marker_file)

    return PassphraseChangeResult(
        passphrase=new, generated=generated, rekeyed=db_exists, rewrites=rewrites
    )


def _rewrite_tier(cfg: SandboxConfig, tier: PassphraseTier, passphrase: str) -> TierRewrite:
    """Land *passphrase* on one tier that currently holds material; report, never raise.

    Failure handling is tier-shaped: a tier that can't take the new
    value is *purged* where possible, because a tier left holding the
    old passphrase would either shadow the new one or fail closed on
    every future resolve — a stale copy is strictly worse than a
    missing one.
    """
    from .._yaml import write_secret_text
    from ..vault.store import systemd_creds as _systemd_creds
    from ..vault.store.encryption import (
        forget_passphrase_in_keyring,
        store_passphrase_in_keyring,
    )

    try:
        if tier is PassphraseTier.SESSION_FILE:
            write_secret_text(cfg.vault_passphrase_file, passphrase + "\n")
            return TierRewrite(tier, ok=True, detail="session file rewritten")
        if tier is PassphraseTier.SYSTEMD_CREDS:
            if not _systemd_creds.is_available():
                reason = _systemd_creds.unavailable_reason() or "unavailable"
                cfg.vault_systemd_creds_file.unlink(missing_ok=True)
                return TierRewrite(
                    tier,
                    ok=False,
                    detail=(
                        f"cannot re-seal ({reason}) — stale sealed credential removed;"
                        " re-run `vault passphrase seal` on a capable host"
                    ),
                )
            _systemd_creds.seal(passphrase, cfg.vault_systemd_creds_file, key_mode="auto")
            return TierRewrite(
                tier, ok=True, detail="credential re-sealed under the new passphrase"
            )
        if tier is PassphraseTier.KEYRING:
            if store_passphrase_in_keyring(passphrase):
                return TierRewrite(tier, ok=True, detail="keyring entry rewritten")
            if forget_passphrase_in_keyring():
                return TierRewrite(
                    tier, ok=False, detail="keyring write failed — stale entry removed"
                )
            return TierRewrite(
                tier,
                ok=False,
                detail=(
                    "keyring write failed and the stale entry could not be removed —"
                    " it still holds the OLD passphrase and will fail to open the vault"
                ),
            )
        return TierRewrite(tier, ok=False, detail="tier cannot be rewritten programmatically")
    except Exception as exc:  # noqa: BLE001 — the DB is already rekeyed; report every tier
        return TierRewrite(tier, ok=False, detail=str(exc))


def _handle_vault_passphrase_change(*, cfg: SandboxConfig | None = None) -> None:
    """Interactively change the vault passphrase.

    CLI ergonomics over [`change_passphrase`][terok_sandbox.commands.vault.change_passphrase]:
    the current passphrase is prompted only when no tier resolves it
    (retyping a value the same shell can print with
    ``vault passphrase reveal`` would be security theatre); the new one
    is typed-and-confirmed, or minted on empty entry and announced to
    ``/dev/tty`` *after* the rekey succeeded.  Ends with the standard
    recovery acknowledgement — the old confirmation died with the old
    passphrase.

    Piped usage reads one line per needed value from stdin: the current
    passphrase first (only when the vault is locked), then the new one
    (empty line = generate).
    """
    from ..vault.store.encryption import (
        WrongPassphraseError,
        prompt_new_passphrase,
        prompt_passphrase,
    )
    from .credentials import _announce_generated_passphrase, _maybe_acknowledge_recovery

    cfg = _resolve_cfg(cfg)

    try:
        current = cfg.resolve_passphrase()
    except WrongPassphraseError as exc:
        raise SystemExit(
            f"cannot change the passphrase: {exc}\n"
            "  Fix or remove the broken tier first — `terok-sandbox vault status`"
            " names it."
        ) from exc
    if current is None:
        if not cfg.db_path.exists():
            raise SystemExit(
                "nothing to change: no credentials DB and no stored passphrase —"
                " run setup to provision the vault instead"
            )
        print("The vault is locked — enter the CURRENT passphrase first.")
        try:
            current = prompt_passphrase()
        except ValueError as exc:
            raise SystemExit(f"nothing was changed: {exc}") from None

    if sys.stdin.isatty():
        print("Enter the NEW passphrase (leave empty to generate one):")
        try:
            new = prompt_new_passphrase()
        except ValueError as exc:
            raise SystemExit(f"nothing was changed: {exc}") from None
    else:
        new = sys.stdin.readline().rstrip("\n") or None
        if new is None:
            # A minted value must be displayed on /dev/tty, and the
            # fail-closed announce would only fire *after* the rekey —
            # refuse before anything changes instead.
            raise SystemExit(
                "a generated passphrase needs a terminal to be displayed on —"
                " supply the new passphrase on stdin, or run interactively."
            )

    try:
        result = change_passphrase(cfg, old=current, new=new)
    except WrongPassphraseError:
        raise SystemExit(
            f"that passphrase does not open {cfg.db_path} — nothing was changed."
        ) from None
    except ValueError as exc:
        raise SystemExit(f"nothing was changed: {exc}") from None
    except RuntimeError as exc:
        raise SystemExit(str(exc)) from None
    except Exception as exc:
        if "database is locked" not in str(exc).lower():
            raise
        import shlex

        raise SystemExit(
            "cannot re-encrypt the credentials DB while it is in use — nothing was"
            " changed.\n"
            "  A running task's supervisor still holds it open.  Find it with:\n"
            f"    fuser -v {shlex.quote(str(cfg.db_path))}\n"
            "  stop it (delete or stop the matching task), then re-run"
            " `vault passphrase change`."
        ) from exc

    if result.rekeyed:
        print(f"→ re-encrypted {cfg.db_path} under the new passphrase")
    for rewrite in result.rewrites:
        marker = "→" if rewrite.ok else "✗"
        print(f"{marker} {rewrite.tier}: {sanitize_tty(rewrite.detail)}")
    print(
        "  already-running tasks may still hold the old passphrase — restart them"
        " to pick up the new one; new tasks use it automatically"
    )

    if result.generated:
        _announce_generated_passphrase(result.passphrase)
    _maybe_acknowledge_recovery(cfg, echo_to_stdout=False)

    if result.problems:
        raise SystemExit(
            "the vault now uses the new passphrase, but the tiers marked ✗ above"
            " could not be rewritten — fix them before the next reboot."
        )


def _handle_vault_status(*, cfg: SandboxConfig | None = None, as_json: bool = False) -> None:
    """Show the vault's lock state, passphrase resolution chain, and stored secrets.

    Read-only host-side diagnostic — all the facts come from
    [`VaultStatus.load`][terok_sandbox.vault.store.status.VaultStatus.load]
    (the same snapshot the TUI and sickbay render), so this handler is
    pure presentation.  Unlike the daemon-startup log it reports the
    *whole* chain rather than only the winning tier, so a session file
    shadowing a durable systemd-creds / keyring tier is visible.
    """
    import json

    from ..vault.store.status import VaultState, VaultStatus

    status = VaultStatus.load(_resolve_cfg(cfg))

    if as_json:
        print(
            json.dumps(
                {
                    "state": status.state,
                    "locked": status.state is not VaultState.UNLOCKED,
                    "lock_reason": status.lock_reason,
                    "db_error": status.db_error,
                    "passphrase_source": status.source,
                    "chain": [
                        {
                            "source": row.tier,
                            "present": row.present,
                            "active": row.active,
                            "shadowed": row.shadowed,
                            "detail": row.detail,
                        }
                        for row in status.chain
                    ],
                    "shadowed_tiers": [row.tier for row in status.chain if row.shadowed],
                    "session_shadow": (
                        {
                            "durable_source": status.shadow.durable_source,
                            "redundant": status.shadow.redundant,
                        }
                        if status.shadow
                        else None
                    ),
                    "recovery_acknowledged": status.recovery.acknowledged,
                    "warnings": [
                        {
                            "kind": warning.kind,
                            "severity": warning.severity,
                            "message": warning.message,
                        }
                        for warning in status.warnings
                    ],
                    "credentials": list(status.providers) if status.providers is not None else None,
                    "db_path": str(status.db_path),
                },
                indent=2,
            )
        )
        return

    _print_vault_status(status)


def _print_vault_status(status: VaultStatus) -> None:
    """Render the human-readable ``vault status`` report to stdout.

    The JSON branch carries the same facts; this is purely presentation.
    Every operator-influenced string (tier detail, provider slugs, paths)
    flows through [`sanitize_tty`][terok_util.sanitize_tty] — they trace
    back to on-disk config / DB content.
    """
    from ..vault.store.status import VaultState

    if status.db_error is not None:
        header = f"ERROR — {sanitize_tty(status.db_error)}"
    elif status.state is VaultState.UNPROVISIONED:
        header = "UNPROVISIONED — no credentials DB and no stored passphrase yet"
    elif status.lock_reason is not None:
        header = f"LOCKED — {sanitize_tty(status.lock_reason)}"
    else:
        header = f"unlocked — passphrase via {status.source}"
    print(f"Vault: {header}")
    db_note = "" if status.db_exists else " (created encrypted on first use)"
    print(f"  DB:  {sanitize_tty(str(status.db_path))}{db_note}")
    print("  Passphrase resolution chain:")
    for row in status.chain:
        marker = "active" if row.active else "shadowed" if row.shadowed else "–"
        print(f"    {row.tier:<19} {marker:<9} {sanitize_tty(row.detail)}")
    if status.state is VaultState.UNPROVISIONED:
        print("  run setup (or the TUI) to provision a vault passphrase")
    elif status.state is VaultState.LOCKED:
        print("  the vault is locked — run `terok-sandbox vault unlock` to provision a passphrase")
    recovery_line = (
        "acknowledged"
        if status.recovery.acknowledged
        else "NOT acknowledged — save the passphrase off-host before you rely on a machine-bound tier"
    )
    print(f"  Recovery key: {recovery_line}")
    for warning in status.warnings:
        label = "note" if warning.severity == "info" else warning.severity
        print(f"  {label}: {sanitize_tty(warning.message)}")
    if status.providers is None:
        detail = "DB unreadable — see the error above" if status.db_error else "vault locked"
        print(f"  Credentials: {detail}")
    else:
        listing = (
            f" ({', '.join(sanitize_tty(p) for p in status.providers)})" if status.providers else ""
        )
        print(f"  Credentials: {len(status.providers)} stored{listing}")


def _handle_vault_list(
    *,
    cfg: SandboxConfig | None = None,
    include_tokens: bool = False,
    as_json: bool = False,
) -> None:
    """List every credential row in the vault (and optionally proxy tokens).

    Read-only inspection.  Secret values never leave the DB: only field
    *names* are surfaced for credential payloads and only an 8-char
    prefix of each proxy token.  Locked-vault failures surface as a
    one-line error pointing at ``vault unlock``.
    """
    import json

    cfg = _resolve_cfg(cfg)

    try:
        db = cfg.open_credential_db(prompt_on_tty=True)
    except (Exception, SystemExit) as exc:  # noqa: BLE001 — bubble out as a friendly message
        print(f"vault unreachable: {type(exc).__name__}: {exc}", file=sys.stderr)
        print(
            "hint: run `terok-sandbox vault unlock` if the passphrase isn't provisioned.",
            file=sys.stderr,
        )
        raise SystemExit(2) from exc

    try:
        credentials: list[dict[str, object]] = []
        for credential_set in db.list_credential_sets():
            for provider in db.list_credentials(credential_set):
                payload = db.load_credential(credential_set, provider) or {}
                credentials.append(
                    {
                        "credential_set": credential_set,
                        "provider": provider,
                        "type": str(payload.get("type") or "—"),
                        "fields": sorted(k for k in payload if k != "type"),
                    }
                )
        tokens = db.list_tokens() if include_tokens else []
    finally:
        db.close()

    if as_json:
        out: dict[str, object] = {"credentials": credentials}
        if include_tokens:
            out["tokens"] = [
                {
                    "token_prefix": _mask_token(row["token"]),
                    "scope": row["scope"],
                    "subject": row["subject"],
                    "credential_set": row["credential_set"],
                    "provider": row["provider"],
                }
                for row in tokens
            ]
        print(json.dumps(out, indent=2))
        return

    _print_credentials_table(credentials)
    if include_tokens:
        print()
        _print_tokens_table(tokens)


def _mask_token(token: str) -> str:
    """Return a display-safe prefix of *token* — keeps the ``terok-p-`` namespace + 8 random chars."""
    if token.startswith("terok-p-"):
        return token[: len("terok-p-") + 8] + "…"
    return token[:8] + "…"


def _print_credentials_table(rows: list[dict[str, object]]) -> None:
    """Render credentials inventory as a fixed-width table to stdout.

    Every cell flows through [`sanitize_tty`][terok_util.sanitize_tty] —
    credential-set / provider / field names originate in the on-disk
    vault DB which is operator-writable, so a hostile value injecting
    ANSI escapes or BEL would otherwise hit the operator's terminal
    directly.
    """
    if not rows:
        print("(no credentials stored)")
        return
    headers = ("credential_set", "provider", "type", "fields")
    formatted = [
        (
            sanitize_tty(str(r["credential_set"])),
            sanitize_tty(str(r["provider"])),
            sanitize_tty(str(r["type"])),
            sanitize_tty(", ".join(r["fields"])) if r["fields"] else "—",  # type: ignore[arg-type]
        )
        for r in rows
    ]
    widths = [max(len(h), *(len(row[i]) for row in formatted)) for i, h in enumerate(headers)]
    print("  ".join(h.ljust(widths[i]) for i, h in enumerate(headers)))
    print("  ".join("-" * w for w in widths))
    for row in formatted:
        print("  ".join(row[i].ljust(widths[i]) for i in range(len(headers))))


def _print_tokens_table(rows: list[dict]) -> None:
    """Render proxy-token inventory as a fixed-width table, with masked tokens.

    Cells are sanitised — see [`_print_credentials_table`][terok_sandbox.commands.vault._print_credentials_table].
    """
    if not rows:
        print("(no proxy tokens issued)")
        return
    print("proxy tokens (token values masked):")
    headers = ("token", "scope", "subject", "credential_set", "provider")
    formatted = [
        (
            sanitize_tty(_mask_token(str(r["token"]))),
            sanitize_tty(str(r["scope"])),
            sanitize_tty(str(r["subject"])),
            sanitize_tty(str(r["credential_set"])),
            sanitize_tty(str(r["provider"])),
        )
        for r in rows
    ]
    widths = [max(len(h), *(len(row[i]) for row in formatted)) for i, h in enumerate(headers)]
    print("  ".join(h.ljust(widths[i]) for i, h in enumerate(headers)))
    print("  ".join("-" * w for w in widths))
    for row in formatted:
        print("  ".join(row[i].ljust(widths[i]) for i in range(len(headers))))


#: Subverbs of ``vault passphrase``.  Live inside the vault group as a
#: nested [`CommandDef`][terok_util.cli_types.CommandDef] so the
#: passphrase verbs reach the CLI exclusively via the vault subparser —
#: no separate top-level tuple to keep in sync.
_PASSPHRASE_GROUP = CommandDef(
    name="passphrase",
    help="Manage where the vault passphrase lives",
    children=(
        CommandDef(
            name="seal",
            help="Seal the current passphrase into a systemd-creds credential",
            handler=LazyHandler("terok_sandbox.commands.vault:handle_vault_seal"),
            args=(
                ArgDef(
                    name="--key",
                    default="auto",
                    help=(
                        "Sealing key: 'auto' (host+TPM2 if a TPM is present,"
                        " host alone otherwise), 'tpm' (require TPM2),"
                        " 'host' (host key only), 'tpm+host' (pin both)"
                    ),
                ),
            ),
        ),
        CommandDef(
            name="to-keyring",
            help="Move the current passphrase from its current tier into the OS keyring",
            handler=LazyHandler("terok_sandbox.commands.vault:handle_vault_to_keyring"),
        ),
        CommandDef(
            name="reveal",
            help=(
                "Display the current vault passphrase (to /dev/tty by default)"
                " and offer to mark the recovery key as saved"
            ),
            handler=LazyHandler("terok_sandbox.commands.vault:_handle_vault_passphrase_reveal"),
            args=(
                ArgDef(
                    name="--allow-redirect",
                    action="store_true",
                    help=(
                        "Print to stdout instead of /dev/tty — opt-in for the"
                        " operator who really does want to pipe the value"
                        " (`| pass insert`, `| op item create`)"
                    ),
                ),
            ),
        ),
        CommandDef(
            name="acknowledge",
            help=(
                "Mark the current passphrase as saved (silent ack from TUI / CI"
                " after the value has been captured out-of-band)"
            ),
            handler=LazyHandler(
                "terok_sandbox.commands.vault:_handle_vault_passphrase_acknowledge"
            ),
        ),
        CommandDef(
            name="change",
            help=(
                "Change the vault passphrase: re-encrypt the DB and rewrite"
                " every tier that stores it"
            ),
            handler=LazyHandler("terok_sandbox.commands.vault:_handle_vault_passphrase_change"),
        ),
    ),
)


#: The vault command group exposed at the package's top level — a
#: single [`CommandDef`][terok_util.cli_types.CommandDef] whose
#: ``children`` are the session passphrase verbs plus the nested
#: ``passphrase`` group.  Consumers wire the whole subtree via
#: [`CommandTree`][terok_util.cli_types.CommandTree]; the structural
#: nesting is what makes ``vault passphrase X`` work without manual
#: subparser plumbing.
VAULT_COMMANDS: tuple[CommandDef, ...] = (
    CommandDef(
        name="vault",
        help="Vault passphrase management",
        children=(
            CommandDef(
                name="status",
                help="Show lock state, the passphrase resolution chain, and stored secrets",
                handler=LazyHandler("terok_sandbox.commands.vault:_handle_vault_status"),
                args=(
                    ArgDef(
                        name="--json",
                        dest="as_json",
                        action="store_true",
                        help="Machine-readable JSON output",
                    ),
                ),
            ),
            CommandDef(
                name="unlock",
                help="Provision the credentials-DB passphrase for this session (tmpfs file)",
                handler=LazyHandler("terok_sandbox.commands.vault:_handle_vault_unlock"),
                args=(
                    ArgDef(
                        name="--force",
                        action="store_true",
                        help="Write the session file even if a durable tier already unlocks the vault",
                    ),
                ),
            ),
            CommandDef(
                name="lock",
                help="Clear every stored copy of the passphrase — you'll need it to unlock again",
                handler=LazyHandler("terok_sandbox.commands.vault:_handle_vault_lock"),
                args=(
                    ArgDef(
                        name="--force",
                        action="store_true",
                        help="Skip the 'have you saved the passphrase?' confirmation",
                    ),
                ),
            ),
            CommandDef(
                name="list",
                help="Inventory stored credentials (and optionally proxy tokens)",
                handler=LazyHandler("terok_sandbox.commands.vault:_handle_vault_list"),
                args=(
                    ArgDef(
                        name="--include-tokens",
                        action="store_true",
                        help="Also show proxy-token rows (token values are masked)",
                    ),
                    ArgDef(
                        name="--json",
                        dest="as_json",
                        action="store_true",
                        help="Machine-readable JSON output",
                    ),
                ),
            ),
            _PASSPHRASE_GROUP,
        ),
    ),
)

#: Per-verb lazy-dispatch entry point resolved by ``commands.COMMANDS``
#: via its ``source`` string (see that module).  Co-located with the
#: registry tuple above so the verb definition stays the single source.
VAULT: CommandDef = VAULT_COMMANDS[0]


__all__ = ["VAULT", "VAULT_COMMANDS"]
