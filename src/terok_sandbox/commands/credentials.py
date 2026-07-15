# SPDX-FileCopyrightText: 2026 Jiri Vyskocil
# SPDX-License-Identifier: Apache-2.0

"""Credentials-DB at-rest encryption — chooser, provisioning, migration.

Two passphrase storage modes are chosen interactively when
``systemd-creds`` isn't available; with it, the chooser is skipped and
the credential is sealed silently.  Once chosen, the mode is persisted
so the resolution chain picks it up on the next daemon start — session
mode self-describes via the tmpfs file's presence; keyring sets
``credentials.use_keyring=true`` in ``config.yml``.
[`plan_provisioning`][terok_sandbox.commands.credentials.plan_provisioning]
is the shared decision core: the CLI chooser here and the TUI's modal
flow are two renderings of the same plan.

The plaintext→encrypted migration is deprecated in 0.8.0 and slated
for removal in 0.9.0.  After that release fresh installs stay the
only supported entry point; operators with a stale plaintext DB must
restore from the ``.plaintext-backup-<stamp>.tar.gz`` snapshot this
phase writes before re-keying.
"""

from __future__ import annotations

import sys
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING

from terok_util import LazyHandler

from ..operator_cli import setup_invocation
from ..vault.store.tiers import CHOOSER_TIERS, PROVISIONABLE_TIERS, PassphraseTier
from ._types import CommandDef

if TYPE_CHECKING:
    from ..config import SandboxConfig

_NON_TTY_TIER_HINT = """\
{setup}: running non-interactively but no passphrase tier was chosen.

  systemd-creds is unavailable on this host (needs systemd ≥ 257), so
  setup would otherwise fall through to the session-unlock tmpfs file
  — a fresh random passphrase you would never see, lost on the first
  reboot.  Pick a tier explicitly:

    --passphrase-tier keyring        (recommended on a single-user host)
    --passphrase-tier session-file   (re-run `vault unlock` after each boot)

  Or install systemd ≥ 257 (Fedora ≥ 42, Debian ≥ 13) and re-run `{setup}`
  so the systemd-creds auto-tier becomes available.  For a headless
  file-based store, point credentials.passphrase_command at your own
  secret file — see the credentials-encryption docs."""

_CHOOSER_PROMPT = """\

systemd-creds isn't available on this host (needs systemd ≥ 257 with
the user Varlink service).  Where should terok store the passphrase
to encrypt the vault?

  [k] keyring — your login keyring (recommended; auto-unlocks at login)
  [s] session-unlock — terok-sandbox vault unlock after each boot

For the strongest protection, install systemd ≥ 257 and re-run setup.
Choice [k]:"""

# Operator's first character maps to the tier.  Empty input picks the
# recommended default (keyring); anything outside this set falls back
# to keyring too — safer than guessing the operator meant ``[s]``.
_CHOICE_TO_TIER: dict[str, PassphraseTier] = {
    "s": PassphraseTier.SESSION_FILE,
    "k": PassphraseTier.KEYRING,
}
_DEFAULT_TIER = PassphraseTier.KEYRING


@dataclass(frozen=True)
class TierProvisionResult:
    """Outcome of [`provision_passphrase_tier`][terok_sandbox.commands.credentials.provision_passphrase_tier].

    ``generated`` drives the caller's follow-up duty: a minted value
    must be *revealed* to the operator (it is their recovery key) and
    acknowledged via
    [`terok_sandbox.vault.store.recovery.acknowledge`][terok_sandbox.vault.store.recovery.acknowledge]
    once they confirm it is saved off-host.
    """

    passphrase: str
    """The value now landed on the tier — mint or caller-supplied."""

    source: PassphraseTier
    """Which tier holds it, in resolution-chain vocabulary."""

    generated: bool
    """``True`` iff this call minted the value (caller passed ``None``)."""


def provision_passphrase_tier(
    cfg: SandboxConfig | None = None,
    *,
    tier: str,
    passphrase: str | None = None,
) -> TierProvisionResult:
    """Land a passphrase on *tier* with no terminal interaction whatsoever.

    The programmatic sibling of the setup chooser, built for GUI/TUI
    frontends that own the conversation themselves: no ``/dev/tty``
    announce, no ack prompt, no stdout side channel.  The caller shows
    the returned value in its own reveal surface and records the
    operator's confirmation through
    [`terok_sandbox.vault.store.recovery.acknowledge`][terok_sandbox.vault.store.recovery.acknowledge].

    *passphrase* ``None`` mints a fresh
    [`generate_passphrase`][terok_sandbox.vault.store.encryption.generate_passphrase]
    value.  When the credentials DB already exists encrypted, the
    supplied value must open it — a mint could never match, so ``None``
    raises [`NoPassphraseError`][terok_sandbox.vault.store.encryption.NoPassphraseError]
    and a mismatch raises
    [`WrongPassphraseError`][terok_sandbox.vault.store.encryption.WrongPassphraseError]
    *before* anything lands on the tier.  This closes the fresh-install
    trapdoor where an "unlock"-shaped prompt silently keys a brand-new
    vault to an unvalidated string.

    Raises [`ValueError`][ValueError] for a tier outside
    [`PROVISIONABLE_TIERS`][terok_sandbox.vault.store.tiers.PROVISIONABLE_TIERS]
    or an explicit empty passphrase (SQLCipher reads ``""`` as "no
    encryption"), and [`RuntimeError`][RuntimeError] when the chosen
    backend (systemd-creds, OS keyring) is unreachable.
    """
    from ..config import SandboxConfig
    from ..vault.store import systemd_creds as _systemd_creds
    from ..vault.store.db import CredentialDB
    from ..vault.store.encryption import (
        NoPassphraseError,
        generate_passphrase,
        is_plaintext_sqlite,
        store_passphrase_in_keyring,
    )

    if tier not in PROVISIONABLE_TIERS:
        raise ValueError(
            f"cannot provision tier {tier!r};"
            f" expected one of: {', '.join(sorted(PROVISIONABLE_TIERS))}"
        )
    tier = PassphraseTier(tier)
    if passphrase == "":  # nosec: B105 — rejecting the empty sentinel, not comparing a secret
        # The keyring and systemd-creds writers refuse an empty value
        # themselves; guard the session-file tier to the same standard
        # so no branch can report success while leaving nothing usable.
        raise ValueError("refusing to provision an empty passphrase")
    if cfg is None:
        cfg = SandboxConfig()

    generated = passphrase is None
    if cfg.db_path.exists() and not is_plaintext_sqlite(cfg.db_path):
        if passphrase is None:
            raise NoPassphraseError(
                f"{cfg.db_path} is already encrypted — a freshly minted passphrase"
                " could never open it; pass the existing passphrase explicitly"
            )
        # Raises WrongPassphraseError on mismatch — before the write, so
        # a bad value never lands on any tier.
        CredentialDB(cfg.db_path, passphrase=passphrase).close()
    if passphrase is None:
        passphrase = generate_passphrase()

    if tier is PassphraseTier.SYSTEMD_CREDS:
        if not _systemd_creds.is_available():
            raise RuntimeError(
                "systemd-creds is unavailable (needs systemd ≥ 257 with the"
                " Varlink io.systemd.Credentials interface); choose a different tier"
            )
        _systemd_creds.seal(passphrase, cfg.vault_systemd_creds_file, key_mode="auto")
        return TierProvisionResult(passphrase, PassphraseTier.SYSTEMD_CREDS, generated)

    if tier is PassphraseTier.KEYRING:
        if not store_passphrase_in_keyring(passphrase):
            raise RuntimeError(
                "OS keyring is unreachable or denied; choose a different storage mode"
            )
        _persist_mode_choice(PassphraseTier.KEYRING)
        return TierProvisionResult(passphrase, PassphraseTier.KEYRING, generated)

    from .._yaml import write_secret_text

    write_secret_text(cfg.vault_passphrase_file, passphrase + "\n")
    return TierProvisionResult(passphrase, PassphraseTier.SESSION_FILE, generated)


def credentials_provisioned(cfg: SandboxConfig | None = None) -> bool:
    """Return ``True`` iff setup's credentials phase has nothing left to provision.

    The pre-flight probe for frontends that want to collect the tier
    choice *before* dispatching ``setup`` non-interactively: ``True``
    when the DB is already SQLCipher-encrypted or some tier of the
    resolution chain already holds a passphrase, ``False`` when a
    non-TTY setup run would fail closed asking for a tier.

    Not infallible: a configured-but-broken durable tier (an
    unsealable systemd-creds credential, a dead ``passphrase_command``)
    propagates its fail-closed
    [`WrongPassphraseError`][terok_sandbox.vault.store.encryption.WrongPassphraseError]
    — callers should surface that as a hard failure, not read it as
    ``False``.
    """
    from ..config import SandboxConfig
    from ..vault.store.encryption import is_plaintext_sqlite

    if cfg is None:
        cfg = SandboxConfig()
    if cfg.db_path.exists() and not is_plaintext_sqlite(cfg.db_path):
        return True
    return _resolve_existing(cfg) is not None


def _resolve_existing(cfg: SandboxConfig) -> tuple[str, PassphraseTier] | None:
    """Return the ``(passphrase, source)`` an existing tier resolves, or ``None``.

    A thin cfg-threading wrapper over
    [`resolve_passphrase_with_source`][terok_sandbox.vault.store.encryption.resolve_passphrase_with_source]
    (never the interactive prompt tier), so provisioning reuses whatever
    a previous run — or a frontend that provisioned in-process — already
    landed, instead of minting over it.  Configured-but-broken durable
    tiers propagate their fail-closed ``WrongPassphraseError``.
    """
    from ..vault.store.encryption import resolve_passphrase_with_source

    passphrase, source = resolve_passphrase_with_source(
        passphrase_file=cfg.vault_passphrase_file,
        systemd_creds_file=cfg.vault_systemd_creds_file,
        use_keyring=cfg.credentials_use_keyring,
        passphrase_command=cfg.credentials_passphrase_command,
    )
    if passphrase is None or source is None:
        return None
    return passphrase, source


@dataclass(frozen=True)
class ProvisioningPlan:
    """The decision half of first-run passphrase provisioning, frontend-free.

    Every frontend renders exactly this: skip when ``provisioned``,
    provision ``auto_tier`` silently when set, otherwise put
    ``choices`` to the operator.  ``keyring_available`` lets a frontend
    grey out the keyring choice up front instead of failing after the
    pick.  The CLI chooser below and the TUI's modal flow are two
    renderings of one plan — the decisions themselves are made here,
    once.
    """

    provisioned: bool
    auto_tier: PassphraseTier | None
    choices: tuple[PassphraseTier, ...]
    keyring_available: bool


def plan_provisioning(cfg: SandboxConfig | None = None) -> ProvisioningPlan:
    """Probe the host and return the provisioning decisions a frontend should render.

    Propagates the fail-closed ``WrongPassphraseError`` of a
    configured-but-broken durable tier (see
    [`credentials_provisioned`][terok_sandbox.commands.credentials.credentials_provisioned])
    — surface it as a hard failure, not as "unprovisioned".
    """
    from ..config import SandboxConfig
    from ..vault.store import systemd_creds as _systemd_creds
    from ..vault.store.encryption import keyring_backend_available

    if cfg is None:
        cfg = SandboxConfig()
    keyring_ok = keyring_backend_available()
    if credentials_provisioned(cfg):
        return ProvisioningPlan(
            provisioned=True, auto_tier=None, choices=(), keyring_available=keyring_ok
        )
    auto_tier = PassphraseTier.SYSTEMD_CREDS if _systemd_creds.is_available() else None
    return ProvisioningPlan(
        provisioned=False,
        auto_tier=auto_tier,
        choices=CHOOSER_TIERS,
        keyring_available=keyring_ok,
    )


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
    2. A tier that already resolves — reuse it silently.  A frontend
       (the TUI chooser) may have provisioned in-process just before
       dispatching setup, and a previous run's keyring entry or sealed
       credential is equally authoritative; minting over either would
       orphan the value the operator already saved.
    3. ``systemd-creds`` auto-detect — the strongest option when present;
       asking when the answer is unambiguous just slows the operator down.
    4. Interactive chooser on a TTY; otherwise hard-fail with an
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
    import warnings

    from ..config import SandboxConfig
    from ..vault.store.encryption import encrypt_in_place, is_plaintext_sqlite

    if cfg is None:
        cfg = SandboxConfig()

    db_path = cfg.db_path
    if db_path.exists() and not is_plaintext_sqlite(db_path):
        print(f"  {db_path} is already SQLCipher-encrypted.")
        return

    warnings.warn(
        "the plaintext→SQLCipher migration path is deprecated in 0.8.0 and "
        "will be removed in 0.9.0; run this migration before upgrading past 0.8.x",
        DeprecationWarning,
        stacklevel=2,
    )

    passphrase, source, auto_generated = _select_and_provision(
        cfg,
        passphrase_tier=passphrase_tier,
        echo_passphrase=echo_passphrase,
    )
    print(f"  passphrase source: {source}")

    if auto_generated:
        _maybe_acknowledge_recovery(cfg, echo_to_stdout=echo_passphrase)

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
) -> tuple[str, PassphraseTier, bool]:
    """Pick a tier, provision a passphrase, return ``(passphrase, source, auto_generated)``.

    Centralises the precedence logic so
    [`_handle_credentials_encrypt_db`][terok_sandbox.commands.credentials._handle_credentials_encrypt_db]
    stays linear and the unit tests can hit each branch directly.  The
    auto-vs-chooser decision comes from
    [`plan_provisioning`][terok_sandbox.commands.credentials.plan_provisioning]
    — the same plan the TUI renders.
    """
    if passphrase_tier is not None:
        return _provision_explicit_tier(cfg, tier=passphrase_tier, echo_passphrase=echo_passphrase)

    if (existing := _resolve_existing(cfg)) is not None:
        passphrase, source = existing
        return passphrase, source, False

    plan = plan_provisioning(cfg)
    if plan.auto_tier is PassphraseTier.SYSTEMD_CREDS:
        passphrase, source = _provision_systemd_creds_tier(cfg, echo_passphrase=echo_passphrase)
        return passphrase, source, True

    mode = _ask_passphrase_mode()
    passphrase, source, auto_generated = _provision_passphrase(
        cfg, mode=mode, echo_passphrase=echo_passphrase
    )
    _persist_mode_choice(mode)
    return passphrase, source, auto_generated


def _provision_explicit_tier(
    cfg: SandboxConfig, *, tier: str, echo_passphrase: bool
) -> tuple[str, PassphraseTier, bool]:
    """Honour an explicit ``--passphrase-tier`` choice.

    Unknown tiers fail fast (``SystemExit``) with the allowed vocabulary;
    ``systemd-creds`` checks availability before dispatching.
    """
    from ..vault.store import systemd_creds as _systemd_creds

    if tier not in PROVISIONABLE_TIERS:
        raise SystemExit(
            f"unknown --passphrase-tier {tier!r};"
            f" expected one of: {', '.join(sorted(PROVISIONABLE_TIERS))}"
        )
    mode = PassphraseTier(tier)
    if mode is PassphraseTier.SYSTEMD_CREDS:
        if not _systemd_creds.is_available():
            raise SystemExit(
                "--passphrase-tier=systemd-creds requested but systemd-creds is"
                " unavailable (needs systemd ≥ 257 with the Varlink"
                " io.systemd.Credentials interface)"
            )
        passphrase, source = _provision_systemd_creds_tier(cfg, echo_passphrase=echo_passphrase)
        return passphrase, source, True

    passphrase, source, auto_generated = _provision_passphrase(
        cfg, mode=mode, echo_passphrase=echo_passphrase
    )
    _persist_mode_choice(mode)
    return passphrase, source, auto_generated


def _maybe_acknowledge_recovery(cfg: SandboxConfig, *, echo_to_stdout: bool) -> None:
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
        acknowledge(cfg.vault_recovery_marker_file)
        print("  recovery key marked as saved.")
    else:
        print(
            "  recovery key NOT confirmed — sickbay / doctor / TUI will warn"
            " until you run `terok-sandbox vault passphrase reveal` and"
            " confirm."
        )


def _ask_passphrase_mode() -> PassphraseTier:
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
        raise SystemExit(_NON_TTY_TIER_HINT.format(setup=setup_invocation()))
    print(_CHOOSER_PROMPT)
    choice = sys.stdin.readline().strip().lower()[:1]
    if not choice:
        return _DEFAULT_TIER
    return _CHOICE_TO_TIER.get(choice, _DEFAULT_TIER)


def _provision_systemd_creds_tier(
    cfg: SandboxConfig, *, echo_passphrase: bool = False
) -> tuple[str, PassphraseTier]:
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
    return passphrase, PassphraseTier.SYSTEMD_CREDS


def _provision_passphrase(
    cfg: SandboxConfig, *, mode: PassphraseTier, echo_passphrase: bool = False
) -> tuple[str, PassphraseTier, bool]:
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

    if mode is PassphraseTier.SESSION_FILE:
        existing = load_passphrase_from_file(cfg.vault_passphrase_file)
        if existing is not None:
            return existing, PassphraseTier.SESSION_FILE, False
        # ``prompt_passphrase(confirm=True)`` mints-and-announces an
        # auto-generated value on empty input, or echo-confirms a typed
        # one.  We can't distinguish the two outcomes from here, so we
        # report ``auto_generated=True`` conservatively — re-acking a
        # value the operator typed is a no-op the second time round
        # but missing the ack on an auto-mint loses the recovery key.
        new = prompt_passphrase(confirm=True)
        write_secret_text(cfg.vault_passphrase_file, new + "\n")
        return new, PassphraseTier.SESSION_FILE, True

    if mode is PassphraseTier.KEYRING:
        existing = load_passphrase_from_keyring()
        if existing is not None:
            return existing, PassphraseTier.KEYRING, False
        new = generate_passphrase()
        if store_passphrase_in_keyring(new):
            _announce_generated_passphrase(new, echo_to_stdout=echo_passphrase)
            return new, PassphraseTier.KEYRING, True
        raise RuntimeError("OS keyring is unreachable or denied; choose a different storage mode")

    raise ValueError(f"unknown mode: {mode!r}")


def _announce_generated_passphrase(passphrase: str, *, echo_to_stdout: bool = False) -> None:
    """Show an auto-minted passphrase to the operator's controlling terminal.

    Routes through ``_write_to_controlling_tty`` so a redirected
    install — ``terok-sandbox setup > install.log``, a CI job, an
    Ansible play — can't capture the recovery key into a journal or
    log artifact.

    Colours the cleartext line in bold yellow so it stands out from
    the rest of the setup-time scroll; without the colour the line
    blends into the surrounding ``→`` / ``ok`` rows and is the first
    thing operators miss in screenshots and logs they paste back.
    A trailing reminder lands at the end of ``terok setup`` via
    [`_post_setup_recovery_hint`][terok_sandbox.commands.credentials._post_setup_recovery_hint]
    as a belt-and-braces nudge.

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
    from .._stage import bold, yellow
    from ..vault.store.encryption import _write_to_controlling_tty

    # The passphrase line itself goes bold-yellow; the surrounding
    # explanation stays plain so the cleartext is the visual centre
    # of the announce block.
    highlighted = bold(yellow(f"Vault passphrase: {passphrase}"))
    message = (
        f"\n{highlighted}\n"
        "  Write this down — it's your recovery key for rebuilds and other hosts.\n"
    )
    _write_to_controlling_tty(message, required=not echo_to_stdout)
    if echo_to_stdout:
        print(message, end="")


def _post_setup_recovery_hint(cfg: SandboxConfig | None = None) -> None:
    """End-of-setup reminder pointing at reveal / acknowledge.

    Setup output is long and the auto-mint banner is buried somewhere
    in the middle of it; operators who get distracted (Slack ping,
    boss walks by) routinely miss the announce line entirely and
    only notice the WARN rows surfacing days later.  This trailing
    block calls out the two verbs the operator needs — ``reveal`` to
    re-display the value, ``acknowledge`` to clear the warning once
    they've saved it — so the breadcrumb survives the scroll.

    Gated on "marker absent" rather than "minted this run": a re-run
    of setup against a host where the operator never acked still needs
    the nudge, while an already-acked host stays quiet.
    """
    from .._stage import bold
    from ..config import SandboxConfig
    from ..vault.store.recovery import acknowledged

    if cfg is None:
        cfg = SandboxConfig()
    if acknowledged(cfg.vault_recovery_marker_file):
        return

    print()
    print(bold("Recovery key — save it off-host"))
    print(
        f"  • {bold('terok vault passphrase reveal')}     "
        "show the passphrase again (route to /dev/tty by default)"
    )
    print(
        f"  • {bold('terok vault passphrase acknowledge')}  "
        "confirm you've stored it — clears the sickbay / doctor / TUI warning"
    )


def _persist_mode_choice(mode: PassphraseTier) -> None:
    """Write the chosen mode into config.yml so the chain re-resolves next time.

    Session mode needs no change — the tmpfs file is self-describing.
    ``use_keyring`` is written even though it defaults on, so an
    explicit ``use_keyring: false`` in the user config can't silently
    disable the tier the operator just chose.
    """
    from .. import config as _config
    from .._yaml import update_section as _yaml_update_section
    from ..paths import config_file_paths

    if mode is not PassphraseTier.KEYRING:
        return
    user_config = next((p for label, p in config_file_paths() if label == "user"), None)
    if user_config is None:
        return
    _yaml_update_section(user_config, "credentials", {"use_keyring": True})
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

    Each per-container supervisor opens the DB read-mostly for the
    lifetime of one container, so the migration writer contends for the
    WAL lock only with whatever containers are currently live.  A
    still-running container would surface as a transient "database is
    locked"; the operator's expected response is "delete old tasks and
    re-run" per the no-state-preservation rule.
    """
    print("→ credentials", end="", flush=True)
    try:
        _handle_credentials_encrypt_db(
            cfg=cfg, echo_passphrase=echo_passphrase, passphrase_tier=passphrase_tier
        )
    except SystemExit as exc:
        # The chooser and the explicit-tier guards refuse via SystemExit
        # carrying a multi-line operator hint.  Let it escape and the
        # stage line above never terminates — the hint glues onto the
        # aggregator's failure banner as one garbled line.  Finish the
        # line, print the hint on its own lines, report phase failure.
        print(" — FAILED")
        if exc.code not in (None, 0):
            print(exc.code if isinstance(exc.code, str) else f"  exit code {exc.code}")
        return False
    except Exception as exc:  # noqa: BLE001
        print(f" — FAILED: {exc}")
        if "database is locked" in str(exc).lower():
            import shlex

            print(
                "  Hint: a per-container supervisor still holds the DB.  Find it with:\n"
                f"    fuser -v {shlex.quote(str(cfg.db_path))}\n"
                "  stop it (delete the matching task), then re-run `terok setup`."
            )
        return False
    return True


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
                    "(deprecated in 0.8.0, removed in 0.9.0)"
                ),
                handler=LazyHandler(
                    "terok_sandbox.commands.credentials:_handle_credentials_encrypt_db"
                ),
            ),
        ),
    ),
)

#: Per-verb lazy-dispatch entry point resolved by ``commands.COMMANDS``
#: via its ``source`` string (see that module).  Co-located with the
#: registry tuple above so the verb definition stays the single source.
CREDENTIALS: CommandDef = CREDENTIALS_COMMANDS[0]


__all__ = [
    "CREDENTIALS",
    "CREDENTIALS_COMMANDS",
    "ProvisioningPlan",
    "TierProvisionResult",
    "_run_credentials_setup_phase",
    "credentials_provisioned",
    "plan_provisioning",
    "provision_passphrase_tier",
]
