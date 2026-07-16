# SPDX-FileCopyrightText: 2026 Jiri Vyskocil
# SPDX-License-Identifier: Apache-2.0

"""The passphrase-tier vocabulary — one registry, every derived subset.

Each tier is one place the vault's SQLCipher passphrase can live.  The
enum's declaration order **is** the resolution-chain order that
[`resolve_passphrase_with_source`][terok_sandbox.vault.store.encryption.resolve_passphrase_with_source]
walks, and the traits table answers the questions the other modules
used to hard-code their own copies of: does the tier survive a reboot,
and can a value be landed on it programmatically?

Adding a tier is: one enum member, one traits row, one branch in each
of the resolver / prober / writer switchboards
([`encryption`][terok_sandbox.vault.store.encryption],
[`provision_passphrase_tier`][terok_sandbox.commands.credentials.provision_passphrase_tier]).
The derived sets below propagate it everywhere else — chooser
vocabulary, ``--passphrase-tier`` validation, shadow detection, the
change-passphrase rewrite fan-out — by construction.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import StrEnum


class PassphraseTier(StrEnum):
    """Where the vault passphrase can live, in resolution-chain priority order.

    A ``StrEnum`` so members compare, hash, and serialise as their
    plain string values — status JSON, config knobs, and CLI arguments
    all speak the same vocabulary without conversion shims.
    """

    SESSION_FILE = "session-file"
    """Tmpfs session-unlock file — RAM-backed, cleared on reboot."""

    SYSTEMD_CREDS = "systemd-creds"
    """Sealed machine-bound credential (TPM2 / host key); needs systemd ≥ 257."""

    KEYRING = "keyring"
    """OS keyring entry, unlocked together with the login session."""

    PASSPHRASE_COMMAND = "passphrase-command"  # nosec: B105 — a tier label, not a secret
    """Operator-supplied helper command that prints the passphrase
    (``pass show …``, ``bw get …``, ``op read …``, cloud secret CLIs)."""

    PROMPT = "prompt"
    """Interactive TTY entry — stores nothing, ever."""

    @property
    def durable(self) -> bool:
        """``True`` iff the tier survives a reboot.

        A volatile tier resolving on top of a durable one is
        *shadowing* it — the vault silently reads the copy that dies
        on the next boot.
        """
        return _TRAITS[self].durable

    @property
    def provisionable(self) -> bool:
        """``True`` iff a value can be written into the tier programmatically.

        ``passphrase-command`` is the counter-example: the secret lives
        in a store the *operator* owns (pass / bitwarden / a cloud
        secret manager), so the sandbox can read it but never write it.
        ``prompt`` stores nothing at all.
        """
        return _TRAITS[self].provisionable

    @property
    def chooser_offered(self) -> bool:
        """``True`` iff the interactive setup chooser lists the tier.

        ``systemd-creds`` is deliberately not offered — it auto-selects
        whenever the host supports it, so listing it would only add a
        dead option to the menu.
        """
        return _TRAITS[self].chooser_offered


@dataclass(frozen=True)
class TierTraits:
    """The per-tier facts every derived subset is built from."""

    durable: bool
    provisionable: bool
    chooser_offered: bool


_TRAITS: dict[PassphraseTier, TierTraits] = {
    PassphraseTier.SESSION_FILE: TierTraits(
        durable=False, provisionable=True, chooser_offered=True
    ),
    PassphraseTier.SYSTEMD_CREDS: TierTraits(
        durable=True, provisionable=True, chooser_offered=False
    ),
    PassphraseTier.KEYRING: TierTraits(durable=True, provisionable=True, chooser_offered=True),
    PassphraseTier.PASSPHRASE_COMMAND: TierTraits(
        durable=True, provisionable=False, chooser_offered=False
    ),
    PassphraseTier.PROMPT: TierTraits(durable=False, provisionable=False, chooser_offered=False),
}

#: Tiers that survive a reboot — the no-shadow guard and the status
#: display's shadowing column both key off this set.
DURABLE_TIERS: frozenset[PassphraseTier] = frozenset(t for t in PassphraseTier if t.durable)

#: Tiers a frontend (setup, TUI, change-passphrase) may write to
#: programmatically.  Doubles as the ``--passphrase-tier`` vocabulary.
PROVISIONABLE_TIERS: frozenset[PassphraseTier] = frozenset(
    t for t in PassphraseTier if t.provisionable
)

#: Tiers the interactive setup chooser offers, in resolution order.
CHOOSER_TIERS: tuple[PassphraseTier, ...] = tuple(t for t in PassphraseTier if t.chooser_offered)


__all__ = [
    "CHOOSER_TIERS",
    "DURABLE_TIERS",
    "PROVISIONABLE_TIERS",
    "PassphraseTier",
    "TierTraits",
]
