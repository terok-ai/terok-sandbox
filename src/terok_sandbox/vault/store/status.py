# SPDX-FileCopyrightText: 2026 Jiri Vyskocil
# SPDX-License-Identifier: Apache-2.0

"""One vault-state picture for every surface — CLI status, TUI pill, sickbay.

[`VaultStatus.load`][terok_sandbox.vault.store.status.VaultStatus.load]
computes everything the frontends used to derive independently (and
with independently-drifting wording): the lock classification, the
per-tier chain table with shadowing, the session-shadow comparison,
and a catalog of [`VaultWarning`][terok_sandbox.vault.store.status.VaultWarning]s
whose text is authored exactly once.  A renderer's whole job is
picking which fields to show — never re-deciding what they mean.

"Locked" alone hides three different operator problems with three
different remedies, so the classification keeps them apart:

- [`VaultState.UNPROVISIONED`][terok_sandbox.vault.store.status.VaultState]
  — fresh install: no DB and no tier holds anything.  The remedy is
  the provisioning flow, not an unlock prompt.
- [`VaultState.LOCKED`][terok_sandbox.vault.store.status.VaultState]
  — a passphrase problem; ``lock_reason`` says which one (empty
  chain, wrong key, or a broken fail-closed tier).
- [`VaultState.ERROR`][terok_sandbox.vault.store.status.VaultState]
  — the DB failed for a non-passphrase reason (schema drift,
  permissions); ``db_error`` carries it verbatim.
- [`VaultState.UNLOCKED`][terok_sandbox.vault.store.status.VaultState]
  — a tier resolves and (when the DB exists) opens it.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from enum import StrEnum
from pathlib import Path
from types import MappingProxyType
from typing import TYPE_CHECKING, Literal

# The function calls go through the module namespace (not by-value
# imports) so the test suite's established ``monkeypatch.setattr(
# encryption, ...)`` stubbing reaches this module too — a by-value
# binding here would silently read real tiers under stubbed tests.
from . import encryption as _encryption
from .encryption import NoPassphraseError, WrongPassphraseError
from .recovery import RecoveryStatus
from .tiers import DURABLE_TIERS, PassphraseTier

if TYPE_CHECKING:
    from ...config import SandboxConfig


def _resolve_cfg(cfg: SandboxConfig | None) -> SandboxConfig:
    """Return *cfg*, or a default [`SandboxConfig`][terok_sandbox.SandboxConfig] built lazily."""
    if cfg is not None:
        return cfg
    from ...config import SandboxConfig  # noqa: PLC0415

    return SandboxConfig()


# ── Durable-tier probe ──────────────────────────────────────────────


def active_durable_source(cfg: SandboxConfig) -> PassphraseTier | None:
    """Name the durable tier that already resolves the vault, or ``None``.

    Probes the chain for a reboot-surviving tier (presence only — no
    unseal, no command exec).  The volatile kernel-keyring cache is
    ``durable=False`` and so never counts here.  The single source of
    truth for the no-cache guard, shared by the passphrase-cache writer
    and the CLI's skip-the-prompt early-out: if a durable tier is
    already present, there is nothing worth caching on top of it.
    """
    for tier in _encryption.probe_passphrase_chain(
        systemd_creds_file=cfg.vault_systemd_creds_file,
        use_keyring=cfg.credentials_use_keyring,
        passphrase_command=cfg.credentials_passphrase_command,
    ):
        if tier.present and tier.source in DURABLE_TIERS:
            return tier.source
    return None


# ── Classification ──────────────────────────────────────────────────


class VaultState(StrEnum):
    """The four operator-distinct answers to "can I use the vault?"."""

    UNPROVISIONED = "unprovisioned"
    """Fresh install — no credentials DB and no tier holds a passphrase.
    The remedy is the provisioning flow (setup / the TUI tier chooser),
    not an unlock prompt keying a brand-new vault to a typo."""

    LOCKED = "locked"
    """A passphrase problem — ``lock_reason`` names which of the three:
    empty chain, wrong key, or a broken fail-closed tier."""

    UNLOCKED = "unlocked"
    """A tier resolves the passphrase and, when the DB exists, opens it."""

    ERROR = "error"
    """The DB failed for a non-passphrase reason — ``db_error`` has it verbatim."""


@dataclass(frozen=True)
class ChainRow:
    """One tier of the resolution chain, annotated for display.

    ``active`` marks the tier the resolver would use right now — the
    first ``present`` tier in resolution order.
    """

    tier: PassphraseTier
    present: bool
    active: bool
    detail: str


class VaultWarningKind(StrEnum):
    """Stable identifiers for the warning catalog — frontends dispatch on these."""

    BROKEN_TIER = "broken-tier"
    RECOVERY_UNCONFIRMED = "recovery-unconfirmed"
    RECOVERY_VOLATILE = "recovery-volatile"


@dataclass(frozen=True)
class VaultWarning:
    """One vault warning, authored here and only here.

    ``brief`` is the compact form for pills / one-line summaries;
    ``message`` the full sentence for notifications and status pages.
    ``kind`` is the semantic identifier — a renderer that wants to
    attach a remedy (a CLI verb, a TUI button) maps it per frontend
    rather than reading command strings out of the library layer.
    """

    kind: VaultWarningKind
    severity: Literal["info", "warning", "error"]
    brief: str
    message: str


def _build_warnings(recovery: RecoveryStatus) -> tuple[VaultWarning, ...]:
    """Derive the warning catalog from the loaded facts — the single wording source."""
    warnings: list[VaultWarning] = []
    if recovery.resolve_error is not None:
        warnings.append(
            VaultWarning(
                VaultWarningKind.BROKEN_TIER,
                "error",
                "broken passphrase tier",
                f"a configured passphrase tier is unreadable — {recovery.resolve_error}",
            )
        )
    if recovery.urgent:
        warnings.append(
            VaultWarning(
                VaultWarningKind.RECOVERY_VOLATILE,
                "error",
                "recovery key UNSAVED, vault dies at logout",
                "the only copy of the vault passphrase is the kernel-keyring cache,"
                " which is cleared at logout and never survives a reboot — save it"
                " off-host now or the vault becomes unrecoverable the next time this"
                " login session ends",
            )
        )
    elif not recovery.acknowledged and recovery.source is not None:
        warnings.append(
            VaultWarning(
                VaultWarningKind.RECOVERY_UNCONFIRMED,
                "warning",
                "recovery key UNCONFIRMED",
                "the vault passphrase is not confirmed saved off-host — every"
                " storage tier is bound to this machine, account, or boot, so a"
                " hardware failure strands the vault without a written copy",
            )
        )
    return tuple(warnings)


@dataclass(frozen=True)
class VaultStatus:
    """Everything a frontend needs to render the vault — loaded in one call."""

    state: VaultState
    lock_reason: str | None
    """Why the vault counts as locked, in operator language.  ``None``
    unless ``state`` is ``LOCKED``."""

    db_error: str | None
    """Verbatim non-passphrase DB failure.  ``None`` unless ``state`` is ``ERROR``."""

    source: PassphraseTier | None
    """The tier the resolver would use right now, or ``None``."""

    chain: tuple[ChainRow, ...]
    recovery: RecoveryStatus
    db_path: Path
    db_exists: bool
    """``False`` on a fresh install — the DB is created encrypted on
    first use, so "unlocked" then means "the key is ready", not "the
    DB opened"."""

    providers: tuple[str, ...] | None
    """Sorted provider slugs stored in the vault; ``None`` when the DB
    couldn't be read (locked / error)."""

    credential_types: Mapping[str, str] | None
    """``provider → type`` (``api_key`` / ``oauth_token`` / …), read in
    the same DB pass as ``providers`` so no renderer ever pays a second
    SQLCipher key derivation just to label a row.  ``None`` exactly
    when ``providers`` is."""

    ssh_keys: int | None
    """Count of stored SSH keypairs, from the same DB pass.  ``None``
    exactly when ``providers`` is."""

    warnings: tuple[VaultWarning, ...]

    @classmethod
    def load(cls, cfg: SandboxConfig | None = None) -> VaultStatus:
        """Compute the full vault picture for *cfg* (defaults if ``None``).

        Read-only by construction: unlike a bare
        ``cfg.open_credential_db()`` this never *creates* the DB — a
        fresh install stays fresh no matter how often status is
        rendered.  Never prompts (a status read must not block on
        stdin).
        """
        cfg = _resolve_cfg(cfg)
        recovery = RecoveryStatus.load(cfg)
        chain = _encryption.probe_passphrase_chain(
            systemd_creds_file=cfg.vault_systemd_creds_file,
            use_keyring=cfg.credentials_use_keyring,
            passphrase_command=cfg.credentials_passphrase_command,
        )
        active_index = next((i for i, tier in enumerate(chain) if tier.present), None)
        rows = tuple(
            ChainRow(
                tier=presence.source,
                present=presence.present,
                active=(i == active_index),
                detail=presence.detail,
            )
            for i, presence in enumerate(chain)
        )

        db_exists = cfg.db_path.exists()
        access = _classify_db_access(cfg, recovery, db_exists=db_exists)
        if access.db_error is not None:
            state = VaultState.ERROR
        elif access.lock_reason is None:
            state = VaultState.UNLOCKED
        elif not db_exists and recovery.source is None and recovery.resolve_error is None:
            state = VaultState.UNPROVISIONED
        else:
            state = VaultState.LOCKED

        return cls(
            state=state,
            lock_reason=access.lock_reason,
            db_error=access.db_error,
            source=recovery.source,
            chain=rows,
            recovery=recovery,
            db_path=cfg.db_path,
            db_exists=db_exists,
            providers=access.providers,
            credential_types=access.credential_types,
            ssh_keys=access.ssh_keys,
            warnings=_build_warnings(recovery),
        )


@dataclass(frozen=True)
class _DbAccess:
    """What one best-effort DB open established — the classifier's answers."""

    lock_reason: str | None = None
    providers: tuple[str, ...] | None = None
    credential_types: Mapping[str, str] | None = None
    ssh_keys: int | None = None
    db_error: str | None = None


def _classify_db_access(
    cfg: SandboxConfig, recovery: RecoveryStatus, *, db_exists: bool
) -> _DbAccess:
    """One best-effort DB open, every answer the status needs.

    - ``lock_reason`` set — the vault can't be opened *because of the
      passphrase*: nothing in any tier (provision one), the resolved
      value doesn't open the DB (typo / foreign DB — re-enter the right
      one), or a configured tier is unreadable (fail-closed from the
      resolver, carried in via ``recovery.resolve_error``).
    - ``providers`` / ``credential_types`` / ``ssh_keys`` set — the DB
      opened (or doesn't exist yet while a tier is ready); all three
      come from the same open so no caller pays a second SQLCipher key
      derivation for the details.
    - ``db_error`` set — the DB failed for a non-passphrase reason
      (schema drift, permissions); surfaced verbatim.

    Never prompts, and never opens a DB that doesn't exist — SQLite
    would create it as a side effect, turning a status *read* into the
    write that defines the vault's encryption key.
    """
    # Plain prose, not a credential — named so credential-heuristic
    # scanners (Sonar S2068) don't misread the assignment.
    no_tier_reason = "no passphrase in any tier"
    if recovery.resolve_error is not None:
        return _DbAccess(lock_reason=f"a configured tier is unreadable — {recovery.resolve_error}")
    if recovery.source is None:
        return _DbAccess(lock_reason=no_tier_reason)
    if not db_exists:
        return _DbAccess(providers=(), credential_types=MappingProxyType({}), ssh_keys=0)
    try:
        db = cfg.open_credential_db(prompt_on_tty=False)
    except WrongPassphraseError:
        return _DbAccess(
            lock_reason=(
                f"the passphrase via {recovery.source} does not open the DB"
                " — wrong key, or a DB from another install"
            )
        )
    except NoPassphraseError:
        # Tier vanished between the resolve and the open — plain lock.
        return _DbAccess(lock_reason=no_tier_reason)
    # ``Exception`` only: with ``prompt_on_tty=False`` no prompt path can
    # raise ``SystemExit`` here, and catching it would stringify an
    # explicit exit from a lower layer into a status line.
    # Non-passphrase failure — surfaced verbatim.
    except Exception as exc:  # noqa: BLE001
        return _DbAccess(db_error=str(exc))
    try:
        types: dict[str, str] = {}
        for credential_set in db.list_credential_sets():
            for provider in db.list_credentials(credential_set):
                row = db.load_credential(credential_set, provider)
                types.setdefault(provider, str(row.get("type", "unknown")) if row else "unknown")
        ssh_keys = db.count_ssh_keys()
    # A mid-read failure is a DB fault, not a lock.
    except Exception as exc:  # noqa: BLE001
        return _DbAccess(db_error=str(exc))
    finally:
        db.close()
    return _DbAccess(
        providers=tuple(sorted(types)),
        credential_types=MappingProxyType(types),
        ssh_keys=ssh_keys,
    )


__all__ = [
    "ChainRow",
    "VaultState",
    "VaultStatus",
    "VaultWarning",
    "VaultWarningKind",
    "active_durable_source",
]
