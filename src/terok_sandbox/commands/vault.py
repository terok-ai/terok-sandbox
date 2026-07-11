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

from ._types import ArgDef, CommandDef

if TYPE_CHECKING:
    from collections.abc import Sequence
    from pathlib import Path

    from ..config import SandboxConfig
    from ..vault.store.encryption import TierPresence
    from ..vault.store.recovery import RecoveryStatus
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


#: Tiers that survive a reboot.  A higher-priority *volatile* tier (the
#: session file) sitting on top of one of these is "shadowing" it: the
#: vault auto-resolves from the session copy and silently ignores the
#: durable one — how a TPM2-sealed box ends up reading a RAM-backed file.
_DURABLE_TIERS = frozenset({"systemd-creds", "keyring", "passphrase-command", "config"})


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
    shadowed_durable: str | None = None


def _active_durable_source(cfg: SandboxConfig) -> str | None:
    """Name the durable tier that already resolves the vault, or ``None``.

    Probes the chain *minus* the session file (file presence only — no
    unseal, no command exec): if a reboot-surviving tier is present, a
    session write would merely shadow it.  The single source of truth
    for the no-shadow guard, shared by the writer and the CLI's
    skip-the-prompt early-out.
    """
    from ..vault.store.encryption import probe_passphrase_chain

    for tier in probe_passphrase_chain(
        systemd_creds_file=cfg.vault_systemd_creds_file,
        use_keyring=cfg.credentials_use_keyring,
        passphrase_command=cfg.credentials_passphrase_command,
        config_fallback=cfg.credentials_passphrase,
    ):
        if tier.present and tier.source in _DURABLE_TIERS:
            return tier.source
    return None


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

    if not force:
        shadowed = _active_durable_source(cfg)
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

    cfg = _resolve_cfg(cfg)

    # Skip the prompt entirely when the writer would refuse anyway — same
    # guard the writer applies, run early purely so we don't ask for a
    # value we'd discard.
    if not force and (durable := _active_durable_source(cfg)) is not None:
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


@dataclass(frozen=True)
class SessionShadow:
    """A session-file tier sitting on top of a durable tier.

    ``redundant`` is the actionable bit:

    - ``True`` — the session copy is byte-identical to what the durable
      tier resolves to, so it's pure residue (a past ``unlock`` on a
      box that already auto-unlocks).  Safe to delete; nothing is lost.
    - ``False`` — the two differ.  Either a deliberate re-key in
      progress or a stale unlock masking the durable value; never
      auto-removed.
    - ``None`` — the durable tier is present but couldn't be read to
      compare (broken seal, dead helper), so the session file may be
      doing real work.  Left alone.
    """

    durable_source: str
    redundant: bool | None


def session_shadow_state(cfg: SandboxConfig) -> SessionShadow | None:
    """Describe a session-file-over-durable-tier shadow, or ``None`` if there is none.

    Returns ``None`` in the common cases — no session file, or no durable
    tier beneath it — without resolving anything.  Only when both are
    present does it resolve the *durable* chain (omitting the session
    file) to compare values; that one comparison is the only place
    status / remediation pay an unseal.  The session secret never leaves
    the process — this reads two tiers and compares, exactly what the
    resolver already does internally.
    """
    from ..vault.store.encryption import (
        NoPassphraseError,
        WrongPassphraseError,
        load_passphrase_from_file,
        resolve_passphrase_with_source,
    )

    session_value = load_passphrase_from_file(cfg.vault_passphrase_file)
    if not session_value:
        return None
    durable_source = _active_durable_source(cfg)
    if durable_source is None:
        return None
    try:
        durable_value, _ = resolve_passphrase_with_source(
            systemd_creds_file=cfg.vault_systemd_creds_file,
            use_keyring=cfg.credentials_use_keyring,
            passphrase_command=cfg.credentials_passphrase_command,
            config_fallback=cfg.credentials_passphrase,
            # ``passphrase_file`` omitted on purpose — resolve the durable
            # chain *under* the session file so we can compare against it.
        )
    except (WrongPassphraseError, NoPassphraseError):
        durable_value = None
    if not durable_value:
        return SessionShadow(durable_source, redundant=None)
    return SessionShadow(durable_source, redundant=(durable_value == session_value))


def clear_redundant_session_file(cfg: SandboxConfig) -> str | None:
    """Remove the session-unlock file iff it merely duplicates a durable tier.

    Same-key residue only: a session file whose passphrase differs from
    the durable tier is a deliberate override (or a re-key mid-flight)
    and is kept.  Re-verifies state at call time rather than trusting a
    caller's stale read.  Returns the durable tier it deduplicated
    against, or ``None`` when there was nothing safe to remove.
    """
    shadow = session_shadow_state(cfg)
    if shadow is None or shadow.redundant is not True:
        return None
    cfg.vault_passphrase_file.unlink(missing_ok=True)
    return shadow.durable_source


def _forget_config_tier_updates(cfg: SandboxConfig) -> dict[str, object | None]:
    """Return the config-section patch ``purge_passphrase_tiers`` should apply.

    Both fields are auto-resolution wirings — leaving either would let
    a future supervisor re-unlock from disk and defeat the lock.
    """
    updates: dict[str, object | None] = {}
    if cfg.credentials_passphrase:
        updates["passphrase"] = None
    if cfg.credentials_passphrase_command:
        updates["passphrase_command"] = None
    return updates


def purge_passphrase_tiers(cfg: SandboxConfig) -> None:
    """Remove every stored copy of the credentials-DB passphrase.

    Clears the session-unlock tmpfs file, the OS keyring entry, the
    sealed systemd-creds credential, and the plaintext
    ``credentials.passphrase`` / ``credentials.passphrase_command``
    wiring in ``config.yml`` — then drops the recovery-acknowledged
    marker, since it's meaningless once no tier remains.  After this the
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


def _classify_vault_access(
    cfg: SandboxConfig, recovery: RecoveryStatus
) -> tuple[str | None, list[str] | None, str | None]:
    """One best-effort DB open, three answers: ``(lock_reason, providers, db_error)``.

    "Locked" hides three different operator problems, each with a
    different remedy — collapsing them is how a wrong passphrase, a
    broken sealed credential, and a genuinely empty chain all read as
    the same word on every surface:

    - ``lock_reason`` set — the vault can't be opened *because of the
      passphrase*: ``no passphrase in any tier`` (provision one), the
      resolved value doesn't open the DB (typo / foreign DB — re-enter
      the right one), or a configured tier is unreadable (broken seal /
      dead helper — fail-closed from the resolver, carried in from
      [`RecoveryStatus.resolve_error`][terok_sandbox.RecoveryStatus]).
    - ``providers`` set — the DB opened; sorted provider slugs.
    - ``db_error`` set — the DB failed for a non-passphrase reason
      (schema drift, permissions); surfaced verbatim.

    Never prompts: a status read must not block on stdin.
    """
    from ..vault.store.encryption import NoPassphraseError, WrongPassphraseError

    # Plain prose, not a credential — named so credential-heuristic
    # scanners (Sonar S2068) don't misread the assignment.
    no_tier_reason = "no passphrase in any tier"
    if recovery.resolve_error is not None:
        return f"a configured tier is unreadable — {recovery.resolve_error}", None, None
    if recovery.source is None:
        return no_tier_reason, None, None
    try:
        db = cfg.open_credential_db(prompt_on_tty=False)
    except WrongPassphraseError:
        return (
            f"the passphrase via {recovery.source} does not open the DB"
            " — wrong key, or a DB from another install",
            None,
            None,
        )
    except NoPassphraseError:
        # Tier vanished between the resolve and the open — plain lock.
        return no_tier_reason, None, None
    # ``Exception`` only: with ``prompt_on_tty=False`` no prompt path can
    # raise ``SystemExit`` here, and catching it would stringify an
    # explicit exit from a lower layer into a status line.
    except Exception as exc:  # noqa: BLE001 - non-passphrase failure, surfaced verbatim
        return None, None, str(exc)
    try:
        providers = sorted(
            {provider for cs in db.list_credential_sets() for provider in db.list_credentials(cs)}
        )
    except Exception as exc:  # noqa: BLE001 - a mid-read failure is a DB fault, not a lock
        return None, None, str(exc)
    finally:
        db.close()
    return None, providers, None


def _handle_vault_status(*, cfg: SandboxConfig | None = None, as_json: bool = False) -> None:
    """Show the vault's lock state, passphrase resolution chain, and stored secrets.

    Read-only host-side diagnostic: it never opens a write transaction
    and never resolves the live passphrase beyond a best-effort DB open
    for the credential count.  Unlike the daemon-startup log, it reports
    the *whole* chain rather than only the winning tier, so a session
    file shadowing a durable systemd-creds / keyring tier is visible —
    and it re-states the plaintext-on-disk and unconfirmed-recovery
    warnings the TUI and sickbay share.
    """
    import json

    from ..paths import plaintext_passphrase_config_path
    from ..vault.store.encryption import probe_passphrase_chain
    from ..vault.store.recovery import RecoveryStatus

    cfg = _resolve_cfg(cfg)

    chain = probe_passphrase_chain(
        passphrase_file=cfg.vault_passphrase_file,
        systemd_creds_file=cfg.vault_systemd_creds_file,
        use_keyring=cfg.credentials_use_keyring,
        passphrase_command=cfg.credentials_passphrase_command,
        config_fallback=cfg.credentials_passphrase,
    )
    active_index = next((i for i, tier in enumerate(chain) if tier.present), None)
    active_source = chain[active_index].source if active_index is not None else None
    # Shadowing only matters when a *volatile* tier (the session file) sits on
    # top of a durable one.  A durable active tier legitimately outranks the
    # tiers below it — that's not a shadow, just the resolution order.
    active_is_durable = active_source in _DURABLE_TIERS
    rows = [
        (
            tier,
            i == active_index,
            tier.present
            and active_index is not None
            and i > active_index
            and tier.source in _DURABLE_TIERS
            and not active_is_durable,
        )
        for i, tier in enumerate(chain)
    ]
    shadowed = [tier.source for tier, _active, is_shadowed in rows if is_shadowed]

    recovery = RecoveryStatus.load(cfg)
    plaintext = plaintext_passphrase_config_path()
    # When the chain table shows a session-over-durable shadow, resolve it
    # once more to learn whether the session copy is redundant (same key)
    # or a real override (different key) — turns the bare "shadowing"
    # warning into an actionable one.  Cheap no-op unless a shadow exists.
    shadow = session_shadow_state(cfg) if shadowed else None
    # Lock state comes from *access* (can the DB actually be opened?),
    # not from chain presence — a sealed credential that exists but
    # won't unseal is "present" in the table yet locked in the header,
    # and that contradiction is exactly the diagnostic.
    lock_reason, providers, db_error = _classify_vault_access(cfg, recovery)

    if as_json:
        print(
            json.dumps(
                {
                    "locked": lock_reason is not None,
                    "lock_reason": lock_reason,
                    "db_error": db_error,
                    "passphrase_source": recovery.source,
                    "chain": [
                        {
                            "source": tier.source,
                            "present": tier.present,
                            "active": is_active,
                            "shadowed": is_shadowed,
                            "detail": tier.detail,
                        }
                        for tier, is_active, is_shadowed in rows
                    ],
                    "shadowed_tiers": shadowed,
                    "session_shadow": (
                        {"durable_source": shadow.durable_source, "redundant": shadow.redundant}
                        if shadow
                        else None
                    ),
                    "recovery_acknowledged": recovery.acknowledged,
                    "plaintext_passphrase_path": str(plaintext) if plaintext else None,
                    "credentials": providers,
                    "db_path": str(cfg.db_path),
                },
                indent=2,
            )
        )
        return

    _print_vault_status(
        rows=rows,
        active_source=recovery.source,
        lock_reason=lock_reason,
        db_error=db_error,
        shadow=shadow,
        recovery_acknowledged=recovery.acknowledged,
        plaintext=plaintext,
        providers=providers,
        db_path=cfg.db_path,
    )


def _print_vault_status(
    *,
    rows: Sequence[tuple[TierPresence, bool, bool]],
    active_source: str | None,
    lock_reason: str | None,
    db_error: str | None,
    shadow: SessionShadow | None,
    recovery_acknowledged: bool,
    plaintext: Path | None,
    providers: list[str] | None,
    db_path: Path,
) -> None:
    """Render the human-readable ``vault status`` report to stdout.

    The JSON branch carries the same facts; this is purely presentation.
    Every operator-influenced string (tier detail, provider slugs, paths)
    flows through [`sanitize_tty`][terok_util.sanitize_tty] — they trace
    back to on-disk config / DB content.
    """
    locked = lock_reason is not None
    if db_error is not None:
        header = f"ERROR — {sanitize_tty(db_error)}"
    elif lock_reason is not None:
        header = f"LOCKED — {sanitize_tty(lock_reason)}"
    else:
        header = f"unlocked — passphrase via {active_source}"
    print(f"Vault: {header}")
    print(f"  DB:  {sanitize_tty(str(db_path))}")
    print("  Passphrase resolution chain:")
    for tier, is_active, is_shadowed in rows:
        marker = "active" if is_active else "shadowed" if is_shadowed else "–"
        print(f"    {tier.source:<19} {marker:<9} {sanitize_tty(tier.detail)}")
    if locked:
        print("  the vault is locked — run `terok-sandbox vault unlock` to provision a passphrase")
    if shadow is not None:
        src = sanitize_tty(shadow.durable_source)
        if shadow.redundant is True:
            print(
                f"  note: the session-file tier duplicates the durable {src} tier"
                " (same passphrase) — redundant residue; it clears on the next reboot"
            )
        elif shadow.redundant is False:
            print(
                f"  warning: the session-file tier shadows the durable {src} tier with a"
                " DIFFERENT passphrase — a deliberate override, or a stale unlock masking"
                " the durable value; the durable tier resumes once the session file is gone"
            )
        else:
            print(
                f"  warning: the session-file tier shadows {src}, but {src} could not be"
                " read to compare — fix or remove the durable tier"
            )
    recovery_line = (
        "acknowledged"
        if recovery_acknowledged
        else "NOT acknowledged — save the passphrase off-host before you rely on a machine-bound tier"
    )
    print(f"  Recovery key: {recovery_line}")
    if plaintext is not None:
        print(f"  warning: vault passphrase stored in plaintext at {sanitize_tty(str(plaintext))}")
    if providers is None:
        detail = "DB unreadable — see the error above" if db_error else "vault locked"
        print(f"  Credentials: {detail}")
    else:
        listing = f" ({', '.join(sanitize_tty(p) for p in providers)})" if providers else ""
        print(f"  Credentials: {len(providers)} stored{listing}")


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
