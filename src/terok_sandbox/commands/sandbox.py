# SPDX-FileCopyrightText: 2026 Jiri Vyskocil
# SPDX-License-Identifier: Apache-2.0

"""Sandbox-wide setup / uninstall — single-call bootstrap of the full stack.

Composes shield + vault + gate + clearance install phases into one
idempotent ``setup`` verb and the symmetric teardown verb.  Each phase
runs its own stop → uninstall → install → verify cycle so a re-run after
a pipx upgrade guarantees the on-disk units pick up the new code.
Higher-level frontends (``terok setup``, ``terok-executor setup``) reuse
this so they install everything in one call.
"""

from __future__ import annotations

import dataclasses

from .._setup import (
    EXIT_MANUAL_STEP_NEEDED,
    print_selinux_install_hint,
    run_clearance_install_phase,
    run_clearance_uninstall_phase,
    run_gate_install_phase,
    run_gate_uninstall_phase,
    run_prereq_report,
    run_shield_install_phase,
    run_shield_uninstall_phase,
    run_vault_install_phase,
    run_vault_uninstall_phase,
)
from .._util._selinux import SelinuxStatus
from ..config import SandboxConfig, credentials_passphrase, credentials_use_keyring
from ._types import ArgDef, CommandDef
from .credentials import _run_credentials_setup_phase


def _handle_sandbox_setup(
    *,
    root: bool = False,
    no_shield: bool = False,
    no_vault: bool = False,
    no_gate: bool = False,
    no_clearance: bool = False,
    echo_passphrase: bool = False,
    passphrase_tier: str | None = None,
    cfg: SandboxConfig | None = None,
) -> None:
    """Install shield + vault + gate + clearance in one idempotent bootstrap.

    Runs a prereq report first (host binaries, firewall binaries,
    SELinux — report-only, never blocks).  Then each service phase
    does the full stop → uninstall → install → verify cycle so a
    re-run after a pipx upgrade guarantees the running units pick up
    the new code, not just the rewritten on-disk files.  Clearance is
    installed when ``terok_clearance`` is importable; headless hosts
    skip it silently.  Exits non-zero if any mandatory phase fails —
    the clearance phase is optional by design.

    On success, writes ``setup.stamp`` with the currently-installed
    package versions — the TUI's startup probe reads it to decide
    whether to nudge the user toward setup.

    Args:
        root: Install shield hooks system-wide (requires sudo); vault
            and gate stay per-user.
        no_shield: Skip the shield install phase.
        no_vault: Skip the vault install phase.
        no_gate: Skip the gate install phase.
        no_clearance: Skip the clearance (hub + verdict + notifier) phase.
        echo_passphrase: Print any auto-generated vault passphrase to
            stdout in addition to ``/dev/tty``.  Required for
            non-interactive bootstraps (CI, Ansible) that need to
            capture the passphrase into their own secret manager —
            without it, the recovery key only reaches the controlling
            terminal and a no-TTY run drops it silently.  Off by
            default so a routine ``setup > install.log`` can't leak it.
        passphrase_tier: Force the credentials-DB passphrase storage
            tier.  One of ``systemd-creds``, ``keyring``, ``session-file``,
            ``config``.  Default ``None`` runs the auto-detect / chooser
            chain — on a non-TTY without systemd-creds that now fails
            closed, so headless bootstraps must pass this explicitly.
        cfg: Optional [`SandboxConfig`][terok_sandbox.config.SandboxConfig]
            override.  Defaults to the layered config — passed through
            so terok's config stays the single source of truth for paths.
    """
    from ..setup_stamp import write_stamp

    if cfg is None:
        cfg = SandboxConfig()

    # Fail-fast on an unknown / unsupported ``--passphrase-tier`` *before*
    # any host-mutating phase runs.  Without this check, a typo would let
    # the shield install land its hooks and only blow up several phases
    # later when the credentials provisioning rejected the tier — leaving
    # a half-installed sandbox that's harder to back out of than to
    # re-attempt cleanly.  When ``--no-vault`` is set, the credentials
    # phase is skipped so the tier value is irrelevant and we don't
    # validate it (passing the kwarg in that mode would be a documented
    # quirk; refusing here would just block the no-vault escape hatch).
    if passphrase_tier is not None and not no_vault:
        _validate_passphrase_tier(passphrase_tier)

    selinux_result = run_prereq_report(cfg)
    print()
    print("Services:")

    failed = False
    if not no_shield:
        failed |= not run_shield_install_phase(root=root)
    # Credentials DB migration runs *before* vault — the daemon needs a
    # decryptable DB to start.  After the credentials phase we refresh
    # the credential fields on ``cfg`` so the vault phase sees the tier
    # choice the operator just made; ``dataclasses.replace`` preserves
    # any non-default paths the caller (e.g. terok-executor) constructed
    # the config with.
    if not no_vault:
        failed |= not _run_credentials_setup_phase(
            cfg, echo_passphrase=echo_passphrase, passphrase_tier=passphrase_tier
        )
        cfg = dataclasses.replace(
            cfg,
            credentials_passphrase=credentials_passphrase(),
            credentials_use_keyring=credentials_use_keyring(),
        )
        failed |= not run_vault_install_phase(cfg)
    if not no_gate:
        failed |= not run_gate_install_phase(cfg)
    if not no_clearance:
        failed |= not run_clearance_install_phase()

    # Re-surface the SELinux install command at the bottom of output
    # so it isn't scrolled away by service install banners.  Sandbox#854.
    print_selinux_install_hint(selinux_result)

    if failed:
        raise SystemExit(1)
    if selinux_result.status is SelinuxStatus.POLICY_MISSING:
        # All install phases succeeded but the host still can't reach
        # the sockets without the policy — setup is functionally
        # incomplete.  Exit 5 ("manual host configuration needed") so
        # scripts and the TUI can distinguish this from a phase failure
        # and offer the specific remediation.
        raise SystemExit(EXIT_MANUAL_STEP_NEEDED)

    stamp = write_stamp()
    print(f"→ setup stamp written: {stamp}")


def _validate_passphrase_tier(tier: str) -> None:
    """Reject an unknown / unavailable ``--passphrase-tier`` value early.

    Mirrors the validation the credentials phase does internally, but
    runs *before* any host-mutating phase so a typo can't leave shield
    hooks installed against a sandbox that will fail at the credentials
    step.  Imports the validator out of ``commands.credentials`` rather
    than re-declaring the tier set so the two paths stay in lockstep.
    """
    from ..vault.store import systemd_creds as _systemd_creds
    from .credentials import _EXPLICIT_TIERS

    if tier not in _EXPLICIT_TIERS:
        raise SystemExit(
            f"unknown --passphrase-tier {tier!r};"
            f" expected one of: {', '.join(sorted(_EXPLICIT_TIERS))}"
        )
    if tier == "systemd-creds" and not _systemd_creds.is_available():
        raise SystemExit(
            "--passphrase-tier=systemd-creds requested but systemd-creds is"
            " unavailable (needs systemd ≥ 257 with the Varlink"
            " io.systemd.Credentials interface)"
        )


def _handle_sandbox_uninstall(
    *,
    root: bool = False,
    no_shield: bool = False,
    no_vault: bool = False,
    no_gate: bool = False,
    no_clearance: bool = False,
    cfg: SandboxConfig | None = None,
) -> None:
    """Tear down the stack in reverse install order.

    Losing gate/vault mid-flight is recoverable, but losing shield
    hooks while containers are live is the most disruptive — shield
    goes last so live containers stay firewalled as long as possible.
    Clearance has no dependants so it goes first.

    Best-effort across phases: a failing phase reports the error and
    the next phase runs anyway, so a partial-install teardown still
    removes what it can instead of leaving orphans behind.  Exits
    non-zero only after every phase has had its attempt.
    """
    from ..setup_stamp import clear_stamp

    if cfg is None:
        cfg = SandboxConfig()

    print("Services:")

    failed = False
    if not no_clearance:
        failed |= not run_clearance_uninstall_phase()
    if not no_gate:
        failed |= not run_gate_uninstall_phase(cfg)
    if not no_vault:
        failed |= not run_vault_uninstall_phase(cfg)
    if not no_shield:
        failed |= not run_shield_uninstall_phase(root=root)

    if clear_stamp():
        print("→ setup stamp removed")
    if failed:
        raise SystemExit(1)


SETUP_COMMANDS: tuple[CommandDef, ...] = (
    CommandDef(
        name="setup",
        help="Install shield hooks + vault + gate in one step",
        handler=_handle_sandbox_setup,
        args=(
            ArgDef(
                name="--root",
                action="store_true",
                help="Install shield hooks system-wide (requires sudo); vault and gate stay per-user",
            ),
            ArgDef(name="--no-shield", action="store_true", help="Skip shield install"),
            ArgDef(name="--no-vault", action="store_true", help="Skip vault install"),
            ArgDef(name="--no-gate", action="store_true", help="Skip gate install"),
            ArgDef(
                name="--no-clearance",
                action="store_true",
                help="Skip clearance hub/verdict/notifier install",
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
                    " (systemd-creds | keyring | session-file | config) instead of"
                    " the auto-detect / chooser chain.  Required on a non-TTY host"
                    " without systemd-creds — the silent session-file fallback was"
                    " removed in v0.0.100 because it minted a passphrase the"
                    " operator never saw and lost it on the first reboot."
                ),
            ),
        ),
    ),
    CommandDef(
        name="uninstall",
        help="Remove shield hooks + vault + gate in one step",
        handler=_handle_sandbox_uninstall,
        args=(
            ArgDef(
                name="--root",
                action="store_true",
                help="Remove shield hooks from the system hooks directory (requires sudo)",
            ),
            ArgDef(name="--no-shield", action="store_true", help="Skip shield uninstall"),
            ArgDef(name="--no-vault", action="store_true", help="Skip vault uninstall"),
            ArgDef(name="--no-gate", action="store_true", help="Skip gate uninstall"),
            ArgDef(
                name="--no-clearance",
                action="store_true",
                help="Skip clearance hub/verdict/notifier uninstall",
            ),
        ),
    ),
)


__all__ = ["SETUP_COMMANDS"]
