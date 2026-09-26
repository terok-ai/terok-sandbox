# SPDX-FileCopyrightText: 2026 Jiri Vyskocil
# SPDX-License-Identifier: Apache-2.0

"""Sandbox-wide setup / uninstall — single-call bootstrap of the full stack.

Composes the supervisor OCI hooks + shield hooks + gate install phases
plus the credentials-DB encryption phase into one idempotent ``setup``
verb and the symmetric teardown verb.  Each phase runs its own
idempotent install cycle so a re-run after a pipx upgrade picks up the
new code.  A legacy cleanup phase sweeps systemd
units / sockets installed by pre-supervisor versions.

Higher-level frontends (``terok setup``, ``terok-executor setup``) reuse
this so they install everything in one call.
"""

from __future__ import annotations

import dataclasses
from typing import TYPE_CHECKING

from terok_util import LazyHandler, require_no_downgrade, require_setup

from ._types import ArgDef, CommandDef

if TYPE_CHECKING:
    from ..config import SandboxConfig


def _handle_sandbox_setup(
    *,
    component: str | None = None,
    show: bool = False,
    no_shield: bool = False,
    no_vault: bool = False,
    echo_passphrase: bool = False,
    passphrase_tier: str | None = None,
    cfg: SandboxConfig | None = None,
) -> int | None:
    """Install supervisor hooks + shield in one idempotent bootstrap.

    With a *component* (``setup selinux`` / ``setup apparmor``) the
    aggregate bootstrap is skipped entirely and the interactive
    per-component installer runs instead — see
    [`handle_setup_component`][terok_sandbox._setup_manual.handle_setup_component].

    Preflights downgrades before writes, installs the lower-owned Shield
    hooks, then refreshes sandbox artifacts and removes obsolete machinery.
    Exits non-zero if any required setup is incomplete.

    Certifies only sandbox-owned setup after verifying required artifacts.
    Dependency receipts remain owned by their packages.

    The git gate lives in each container's supervisor, so there is no
    host-side gate install phase.

    Args:
        no_shield: Skip the shield install phase.
        no_vault: Skip the credentials-DB encryption phase (per-container
            vault has no host-side install of its own; the flag controls
            credentials provisioning only).
        echo_passphrase: Print any auto-generated vault passphrase to
            stdout in addition to ``/dev/tty``.  Required for
            non-interactive bootstraps (CI, Ansible) that need to
            capture the passphrase into their own secret manager —
            without it, the recovery key only reaches the controlling
            terminal and a no-TTY run drops it silently.  Off by
            default so a routine ``setup > install.log`` can't leak it.
        passphrase_tier: Force the credentials-DB passphrase storage
            tier.  One of ``systemd-creds``, ``keyring``,
            ``kernel-keyring``.  Default ``None`` runs the auto-detect / chooser
            chain — on a non-TTY without systemd-creds that now fails
            closed, so headless bootstraps must pass this explicitly.
        cfg: Optional [`SandboxConfig`][terok_sandbox.config.SandboxConfig]
            override.  Defaults to the layered config — passed through
            so terok's config stays the single source of truth for paths.
    """
    from .._exit_codes import EXIT_MANUAL_STEP_NEEDED
    from .._setup import (
        print_apparmor_install_hint,
        print_selinux_install_hint,
        run_legacy_install_cleanup_phase,
        run_prereq_report,
        run_shield_install_phase,
        run_supervisor_install_phase,
    )
    from ..config import SandboxConfig, credentials_use_keyring
    from ..integrations.shield import ShieldHooks
    from ..setup import check_artifacts, check_host_tools, check_setup, setup_receipt
    from .credentials import _run_credentials_setup_phase

    if cfg is None:
        cfg = SandboxConfig()

    if component is not None or show:
        # Only meaningful against a named component; without one, the flow
        # below answers the real problem ("--show needs a component").
        rejected = (
            []
            if component is None
            else [
                flag
                for flag, given in (
                    ("--no-shield", no_shield),
                    ("--no-vault", no_vault),
                    ("--echo-passphrase", echo_passphrase),
                    ("--passphrase-tier", passphrase_tier is not None),
                )
                if given
            ]
        )
        if rejected:
            raise SystemExit(
                f"{', '.join(rejected)} belongs to the full setup, not to 'setup {component}'"
            )
        from .._setup_manual import handle_setup_component

        return handle_setup_component(component, show_only=show, cfg=cfg)

    # Fail-fast on an unknown / unsupported ``--passphrase-tier`` *before*
    # any host-mutating phase runs.  Without this check, a typo would let
    # the shield install land its hooks and only blow up several phases
    # later when the credentials provisioning rejected the tier — leaving
    # a half-installed sandbox that's harder to back out of than to
    # re-attempt cleanly.  When ``--no-vault`` is set the credentials
    # phase is skipped so the tier value is irrelevant.
    if passphrase_tier is not None and not no_vault:
        _validate_passphrase_tier(passphrase_tier)

    require_no_downgrade(check_setup(cfg))
    require_setup(check_host_tools())
    receipt = setup_receipt(cfg)

    selinux_result, apparmor_result = run_prereq_report(cfg)
    print()
    print("Services:")

    if not no_shield and not cfg.shield_disabled and not run_shield_install_phase():
        raise SystemExit(1)
    # Child setup succeeds before invalidating or changing our own artifacts.
    receipt.clear()
    failed = not run_legacy_install_cleanup_phase()
    # Credentials DB migration runs *before* the per-container vault
    # will need it.  After the credentials phase we refresh the
    # credential fields on ``cfg`` so any downstream phase sees the tier
    # choice the operator just made; ``dataclasses.replace`` preserves
    # any non-default paths the caller (e.g. terok-executor) constructed
    # the config with.
    if not no_vault:
        failed |= not _run_credentials_setup_phase(
            cfg, echo_passphrase=echo_passphrase, passphrase_tier=passphrase_tier
        )
        cfg = dataclasses.replace(
            cfg,
            credentials_use_keyring=credentials_use_keyring(),
        )
    # The git gate lives in each container's supervisor — no host-side
    # install phase.  Clearance has nothing to install on the host — every
    # container's supervisor composes the hub + verdict + notifier
    # inline.  The legacy clearance unit files are removed by the
    # ``Legacy install cleanup`` phase that ran above.
    # Supervisor hooks land last so a half-installed prereq doesn't leave
    # a fire-able OCI hook pointing at a non-existent ``terok-sandbox``
    # binary.  An unresolved companion leaves setup incomplete.
    failed |= not run_supervisor_install_phase(root=cfg.state_dir)

    # Re-surface the SELinux install command at the bottom of output
    # so it isn't scrolled away by service install banners.  Sandbox#854.
    print_selinux_install_hint(selinux_result)
    # Likewise the AppArmor dnsmasq-profile addendum (non-fatal: shield
    # falls back to the lookup tier without it, so no exit-code change).
    print_apparmor_install_hint(apparmor_result)

    if failed:
        raise SystemExit(1)
    if selinux_result.status.action_needed:
        # All install phases succeeded but the host still can't reach
        # the sockets without the policy — missing entirely, or a stale
        # revision lacking the supervisor's rule.  Either way setup is
        # functionally incomplete: exit 5 ("manual host configuration
        # needed") so scripts and the TUI can distinguish this from a
        # phase failure and offer the specific remediation.
        raise SystemExit(EXIT_MANUAL_STEP_NEEDED)

    children = () if cfg.shield_disabled else ShieldHooks.check_setup(live=True)
    require_setup((*children, *check_artifacts(cfg, live=True)))
    receipt.write()
    print(f"→ setup receipt written: {receipt.path}")

    # Trailing recovery-key reminder — fires only when the marker is
    # absent, so re-runs on an already-acked host stay quiet.  The
    # auto-mint announce lands mid-flow and is easy to scroll past;
    # this block survives the scroll.
    if not no_vault:
        from .credentials import _post_setup_recovery_hint

        _post_setup_recovery_hint(cfg)
    return None


def _validate_passphrase_tier(tier: str) -> None:
    """Reject an unknown / unavailable ``--passphrase-tier`` value early.

    Mirrors the validation the credentials phase does internally, but
    runs *before* any host-mutating phase so a typo can't leave shield
    hooks installed against a sandbox that will fail at the credentials
    step.  Validates against the tier registry's
    [`PROVISIONABLE_TIERS`][terok_sandbox.vault.store.tiers.PROVISIONABLE_TIERS]
    so the two paths stay in lockstep by construction.
    """
    from ..vault.store import systemd_creds as _systemd_creds
    from ..vault.store.tiers import PROVISIONABLE_TIERS

    if tier not in PROVISIONABLE_TIERS:
        raise SystemExit(
            f"unknown --passphrase-tier {tier!r};"
            f" expected one of: {', '.join(sorted(PROVISIONABLE_TIERS))}"
        )
    if tier == "systemd-creds" and not _systemd_creds.is_available():
        raise SystemExit(
            "--passphrase-tier=systemd-creds requested but systemd-creds is"
            " unavailable (needs systemd ≥ 257 with the Varlink"
            " io.systemd.Credentials interface)"
        )


def _handle_sandbox_uninstall(
    *,
    no_shield: bool = False,
    cfg: SandboxConfig | None = None,
) -> None:
    """Tear down the stack in reverse install order.

    Losing supervisor hooks mid-flight is recoverable, but losing shield
    hooks while containers are live is the most disruptive — shield goes
    last so live containers stay firewalled as long as possible.

    Best-effort across phases: a failing phase reports the error and
    the next phase runs anyway, so a partial-install teardown still
    removes what it can instead of leaving orphans behind.  Exits
    non-zero only after every phase has had its attempt.

    The git gate has no host-side install, so there is no gate uninstall
    phase — the legacy sweep removes any pre-supervisor gate units.
    """
    from .._setup import (
        run_legacy_install_cleanup_phase,
        run_shield_uninstall_phase,
        run_supervisor_uninstall_phase,
    )
    from ..config import SandboxConfig
    from ..setup import check_setup, setup_receipt

    cfg = cfg or SandboxConfig()
    require_no_downgrade(check_setup(cfg))
    setup_receipt(cfg).clear()

    print("Services:")

    failed = False
    # Supervisor hooks come down first so a slow uninstall on lower
    # layers can't surprise-fire a still-installed OCI hook.
    failed |= not run_supervisor_uninstall_phase(root=cfg.state_dir)
    if not no_shield:
        failed |= not run_shield_uninstall_phase()
    # Legacy-install sweep also runs at uninstall so a host that's
    # being decommissioned doesn't leave pre-supervisor systemd units
    # behind for a future operator to puzzle over.
    failed |= not run_legacy_install_cleanup_phase()

    if failed:
        raise SystemExit(1)


SETUP_COMMANDS: tuple[CommandDef, ...] = (
    CommandDef(
        name="setup",
        help="Install supervisor hooks + shield hooks in one step",
        handler=LazyHandler("terok_sandbox.commands.sandbox:_handle_sandbox_setup"),
        args=(
            ArgDef(
                name="component",
                nargs="?",
                help=(
                    "Install one hardening prerequisite interactively"
                    " (selinux | apparmor): shows the exact sudo command and"
                    " the rules before anything runs"
                ),
            ),
            ArgDef(
                name="--show",
                action="store_true",
                help="With a component: print the rules it would install, then exit",
            ),
            ArgDef(name="--no-shield", action="store_true", help="Skip shield install"),
            ArgDef(
                name="--no-vault",
                action="store_true",
                help="Skip the credentials-DB encryption phase",
            ),
            ArgDef(
                name="--echo-passphrase",
                action="store_true",
                help=(
                    "Also print any auto-generated vault passphrase to stdout"
                    " (default off — the value otherwise only reaches /dev/tty,"
                    " so non-interactive bootstraps must opt in to capture it)"
                ),
            ),
            ArgDef(
                name="--passphrase-tier",
                default=None,
                help=(
                    "Force credentials-DB passphrase storage to a specific tier"
                    " (systemd-creds | keyring | kernel-keyring) instead of"
                    " the auto-detect / chooser chain.  Required on a non-TTY host"
                    " without systemd-creds — the silent volatile fallback was"
                    " removed in v0.0.100 because it minted a passphrase the"
                    " operator never saw and lost it at logout."
                ),
            ),
        ),
    ),
    CommandDef(
        name="uninstall",
        help="Remove supervisor hooks + shield hooks in one step",
        handler=LazyHandler("terok_sandbox.commands.sandbox:_handle_sandbox_uninstall"),
        args=(ArgDef(name="--no-shield", action="store_true", help="Skip shield uninstall"),),
    ),
)

#: Per-verb lazy-dispatch entry point resolved by ``commands.COMMANDS``
#: via its ``source`` string (see that module).  Co-located with the
#: registry tuple above so the verb definition stays the single source.
SETUP: CommandDef = SETUP_COMMANDS[0]
UNINSTALL: CommandDef = SETUP_COMMANDS[1]


__all__ = ["SETUP", "UNINSTALL", "SETUP_COMMANDS"]
