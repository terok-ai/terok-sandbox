# SPDX-FileCopyrightText: 2026 Jiri Vyskocil
# SPDX-License-Identifier: Apache-2.0

"""Command registry for terok-sandbox.

Follows the same [`CommandDef`][terok_sandbox.commands.CommandDef] / [`ArgDef`][terok_sandbox.commands.ArgDef] pattern as
``terok_shield.registry``.  Higher-level consumers (terok, terok-executor)
can import ``COMMANDS`` to build their own CLI frontends without
duplicating argument definitions or handler logic.

Shield commands are delegated to terok-shield's own registry —
``SHIELD_COMMANDS`` re-exports the non-standalone subset.
"""

from __future__ import annotations

import contextlib
import dataclasses
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Any, Literal, NamedTuple

from ._setup import (
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
from ._util import sanitize_tty
from ._yaml import update_section as _yaml_update_section
from .config import SandboxConfig, credentials_passphrase, credentials_use_keyring
from .credentials.encryption import PassphraseSource
from .credentials.systemd_creds import KeyMode

if TYPE_CHECKING:
    from collections.abc import Callable

    from .credentials.db import CredentialDB


class KeyRow(NamedTuple):
    """One registered SSH key, fully resolved for display and matching."""

    scope: str
    comment: str
    key_type: str
    fingerprint: str
    private_key: str
    public_key: str


@dataclass(frozen=True)
class ArgDef:
    """Definition of a single CLI argument."""

    name: str
    help: str = ""
    type: Callable[[str], Any] | None = None
    default: Any = None
    action: str | None = None
    dest: str | None = None
    nargs: int | str | None = None
    required: bool = False


@dataclass(frozen=True)
class CommandDef:
    """Definition of a sandbox subcommand.

    Attributes:
        name: Subcommand name (e.g. ``"gate start"``).
        help: One-line help string.
        handler: Callable implementing the command.
        args: Argument definitions.
        group: Command group (e.g. ``"gate"``, ``"shield"``).
        epilog: Optional long-form text rendered after the argparse
            argument list in ``--help`` output.
    """

    name: str
    help: str = ""
    handler: Callable[..., None] | None = None
    args: tuple[ArgDef, ...] = ()
    group: str = ""
    epilog: str = ""


# ---------------------------------------------------------------------------
# Sandbox-wide setup and uninstall
#
# Single-call bootstrap/teardown for the shield+vault+gate stack.  Consumed
# by higher-level frontends (``terok setup``, ``terok-executor setup``)
# so they can install everything with one call and tear it down
# symmetrically.  Individual services still have their own install /
# uninstall verbs in the groups below.
# ---------------------------------------------------------------------------


def _handle_sandbox_setup(
    *,
    root: bool = False,
    no_shield: bool = False,
    no_vault: bool = False,
    no_gate: bool = False,
    no_clearance: bool = False,
    cfg: SandboxConfig | None = None,
) -> None:
    """Install shield + vault + gate + clearance in one idempotent bootstrap.

    Runs a prereq report first (host binaries, firewall binaries, SELinux
    — report-only, never blocks).  Then each service phase does the full
    stop → uninstall → install → verify cycle so a re-run after a pipx
    upgrade guarantees the running units pick up the new code, not just
    the rewritten on-disk files.  Clearance (hub + verdict + notifier)
    is installed when ``terok_clearance`` is importable; headless hosts
    skip it silently.  Exits non-zero if any mandatory phase fails —
    the clearance phase is optional by design.

    On success (every attempted phase reached its end), writes
    ``setup.stamp`` with the currently-installed package versions —
    the TUI's startup probe reads it to decide whether to nudge the
    user toward setup.  Phases skipped via ``--no-*`` are the user's
    explicit choice and don't block stamping; only an actual failure
    holds the stamp back.

    Args:
        root: Install shield hooks system-wide (requires sudo); vault
            and gate stay per-user.
        no_shield: Skip the shield install phase.
        no_vault: Skip the vault install phase.
        no_gate: Skip the gate install phase.
        no_clearance: Skip the clearance (hub + verdict + notifier) phase.
        cfg: Optional [`SandboxConfig`][terok_sandbox.commands.SandboxConfig] override.  Defaults to the
            layered config — passed through so terok's config stays
            the single source of truth for paths.
    """
    from .setup_stamp import write_stamp

    if cfg is None:
        cfg = SandboxConfig()

    run_prereq_report(cfg)
    print()
    print("Services:")

    failed = False
    # Shield's install installs the OCI hook pair, the bridge hook pair,
    # and the standalone NFLOG reader resource in one go — no separate
    # bridge phase any more.  The bridge hooks soft-fail when clearance
    # isn't running (no socket to deliver events to), so installing them
    # unconditionally costs nothing on shield-only deployments.
    if not no_shield:
        failed |= not run_shield_install_phase(root=root)
    # Credentials DB migration runs *before* vault — the daemon needs a
    # decryptable DB to start, so doing it here keeps a single failed
    # phase from masking the actual cause.  Skipped along with the vault
    # phase when ``--no-vault`` is set; otherwise we refresh the
    # credential fields on ``cfg`` so the vault phase sees the tier
    # choice the operator just made.  ``dataclasses.replace`` preserves
    # any non-default paths the caller (e.g. terok-executor) constructed
    # the config with.
    if not no_vault:
        failed |= not _run_credentials_setup_phase(cfg)
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

    if failed:
        raise SystemExit(1)

    stamp = write_stamp()
    print(f"→ setup stamp written: {stamp}")


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

    A running container can lose its gate and vault without immediate
    blast, but losing shield hooks mid-flight is the most disruptive —
    shield goes last so live containers stay firewalled as long as
    possible.  Clearance (hub + verdict + notifier) is torn down first
    because its only connection is to the hub's varlink socket; it
    has no dependants.

    Best-effort across phases: a failing phase reports the error and
    the next phase runs anyway, so a partial-install teardown still
    removes what it can instead of leaving orphans behind.  Exits
    non-zero only after every phase has had its attempt.

    Output shape mirrors `_handle_sandbox_setup` — one stage line
    per phase under a ``Services:`` heading — so the two commands read
    as symmetric halves of the same log when run back-to-back.
    """
    from .setup_stamp import clear_stamp

    if cfg is None:
        cfg = SandboxConfig()

    print("Services:")

    failed = False
    # Shield's uninstall removes both hook pairs and the reader resource
    # together — no separate bridge phase any more.
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


def _handle_gate_install() -> None:
    """Install gate server systemd units, refusing hosts without systemd-user."""
    from .gate.lifecycle import GateServerManager

    mgr = GateServerManager()
    if not mgr.is_systemd_available():
        print("Error: systemd user services are not available on this host.")
        raise SystemExit(1)
    mgr.install_systemd_units()
    print("Gate server installed via systemd socket activation.")


def _handle_gate_uninstall() -> None:
    """Remove gate server systemd units, stopping any stray daemon first."""
    from .gate.lifecycle import GateServerManager

    mgr = GateServerManager()
    if mgr.get_status().mode == "daemon":
        mgr.stop_daemon()
    if mgr.is_systemd_available():
        mgr.uninstall_systemd_units()
    print("Gate server systemd units removed.")


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


# ---------------------------------------------------------------------------
# Gate handlers
# ---------------------------------------------------------------------------


def _handle_gate_start(
    *, port: int | None = None, daemon: bool = False, cfg: SandboxConfig | None = None
) -> None:
    """Start the gate server (systemd preferred, daemon fallback)."""
    from .gate.lifecycle import GateServerManager

    mgr = GateServerManager(cfg)
    if mgr.is_systemd_available() and not daemon:
        mgr.install_systemd_units()
        print("Gate server started via systemd socket activation.")
    else:
        mgr.start_daemon(port=port)
        print("Gate server daemon started.")


def _handle_gate_stop(*, cfg: SandboxConfig | None = None) -> None:
    """Stop the gate server."""
    from .gate.lifecycle import GateServerManager

    mgr = GateServerManager(cfg)
    status = mgr.get_status()
    if status.mode == "systemd":
        mgr.uninstall_systemd_units()
        print("Gate server systemd units removed.")
    elif status.mode == "daemon":
        mgr.stop_daemon()
        print("Gate server daemon stopped.")
    else:
        print("Gate server is not running.")


def _handle_gate_status(*, cfg: SandboxConfig | None = None) -> None:
    """Show gate server status."""
    from .gate.lifecycle import GateServerManager

    mgr = GateServerManager(cfg)
    status = mgr.get_status()
    print(f"Mode:      {status.mode}")
    print(f"Running:   {'yes' if status.running else 'no'}")
    print(f"Port:      {status.port}")
    print(f"Base path: {mgr.gate_base_path}")

    warning = mgr.check_units_outdated()
    if warning:
        import sys

        print(f"\nWarning: {warning}", file=sys.stderr)
        print("Run 'terok-sandbox gate start' to update.", file=sys.stderr)


# ---------------------------------------------------------------------------
# Shield handlers (thin wrappers around terok_sandbox.shield)
# ---------------------------------------------------------------------------


def _handle_shield_setup(*, root: bool = False, user: bool = False) -> None:
    """Install OCI hooks for the shield firewall.

    Validates the ``--root`` / ``--user`` choice at the CLI layer so
    the library function ([`shield.run_setup`][terok_sandbox.shield.run_setup]) can stay UX-agnostic:
    it raises ``ValueError`` on invalid combinations; this handler turns
    that into a ``SystemExit`` with CLI-specific remediation hints.
    """
    if not root and not user:
        raise SystemExit(
            "Specify --root (system-wide, uses sudo) or --user (user-local).\n"
            "  shield install-hooks --root   # /etc/containers/oci/hooks.d\n"
            "  shield install-hooks --user   # ~/.local/share/containers/oci/hooks.d"
        )
    from .shield import run_setup

    run_setup(root=root, user=user)


def _handle_shield_uninstall(*, root: bool = False, user: bool = False) -> None:
    """Remove the OCI hooks previously installed by ``shield install-hooks``.

    Idempotent — missing files are treated as success.  Symmetric to
    `_handle_shield_setup`: ``--root`` uses sudo, ``--user``
    touches the user hooks directory.
    """
    if not root and not user:
        raise SystemExit(
            "Specify --root (system-wide, uses sudo) or --user (user-local).\n"
            "  shield uninstall-hooks --root   # /etc/containers/oci/hooks.d\n"
            "  shield uninstall-hooks --user   # ~/.local/share/containers/oci/hooks.d"
        )
    from .shield import run_uninstall

    run_uninstall(root=root, user=user)
    scope = "system" if root else "user"
    print(f"Shield hooks removed from {scope} hooks directory.")


def _handle_shield_status() -> None:
    """Show shield configuration and environment check."""
    import sys

    from .shield import check_environment, status

    env = check_environment()
    cfg = status()

    print(f"Shield mode:    {cfg.get('mode', '?')}")
    print(f"Profiles:       {', '.join(cfg.get('profiles', []))}")
    print(f"Audit:          {'enabled' if cfg.get('audit_enabled') else 'disabled'}")
    print(f"Hooks:          {env.hooks}")
    print(f"Health:         {env.health}")
    if env.needs_setup:
        print(f"\n{env.setup_hint}", file=sys.stderr)


# ---------------------------------------------------------------------------
# Command definitions
# ---------------------------------------------------------------------------

GATE_COMMANDS: tuple[CommandDef, ...] = (
    CommandDef(
        name="start",
        help="Start the gate server",
        handler=_handle_gate_start,
        group="gate",
        args=(
            ArgDef(name="--port", type=int, default=None, help="Override port (default: 9418)"),
            ArgDef(name="--daemon", action="store_true", help="Force daemon mode (skip systemd)"),
        ),
    ),
    CommandDef(
        name="stop",
        help="Stop the gate server",
        handler=_handle_gate_stop,
        group="gate",
    ),
    CommandDef(
        name="status",
        help="Show gate server status",
        handler=_handle_gate_status,
        group="gate",
    ),
)

SHIELD_COMMANDS: tuple[CommandDef, ...] = (
    CommandDef(
        name="install-hooks",
        help="Install OCI hooks for the shield firewall",
        handler=_handle_shield_setup,
        group="shield",
        args=(
            ArgDef(name="--root", action="store_true", help="Install system-wide (requires sudo)"),
            ArgDef(name="--user", action="store_true", help="Install to user hooks directory"),
        ),
    ),
    CommandDef(
        name="uninstall-hooks",
        help="Remove OCI hooks previously installed by install-hooks",
        handler=_handle_shield_uninstall,
        group="shield",
        args=(
            ArgDef(name="--root", action="store_true", help="Remove system-wide (requires sudo)"),
            ArgDef(name="--user", action="store_true", help="Remove from user hooks directory"),
        ),
    ),
    CommandDef(
        name="status",
        help="Show shield status",
        handler=_handle_shield_status,
        group="shield",
    ),
)

# ---------------------------------------------------------------------------
# Vault handlers
# ---------------------------------------------------------------------------


def _handle_vault_start() -> None:
    """Start the vault daemon."""
    from .vault.lifecycle import VaultManager

    mgr = VaultManager()
    if mgr.get_status().running:
        print("Vault is already running.")
        return
    mgr.start_daemon()
    print("Vault started.")


def _handle_vault_stop() -> None:
    """Stop the vault daemon."""
    from .vault.lifecycle import VaultManager

    mgr = VaultManager()
    if not mgr.is_daemon_running():
        print("Vault is not running.")
        return
    mgr.stop_daemon()
    print("Vault stopped.")


def _handle_vault_status() -> None:
    """Show vault status."""
    from .vault.lifecycle import VaultManager

    status = VaultManager().get_status()
    state = "running" if status.running else "stopped"
    print(f"Status: {state}")
    print(f"Socket: {sanitize_tty(str(status.socket_path))}")
    print(f"DB:     {sanitize_tty(str(status.db_path))}")
    print(
        f"Routes: {sanitize_tty(str(status.routes_path))} ({status.routes_configured} configured)"
    )
    if status.credentials_stored:
        print(f"Credentials: {', '.join(sanitize_tty(c) for c in status.credentials_stored)}")
    else:
        print("Credentials: none stored")


def _handle_vault_install() -> None:
    """Install and start systemd socket activation for the vault."""
    from .vault.lifecycle import VaultManager

    mgr = VaultManager()
    if not mgr.is_systemd_available():
        print("Error: systemd user services are not available on this host.")
        raise SystemExit(1)
    mgr.install_systemd_units()
    print("Vault installed via systemd socket activation.")


def _handle_vault_uninstall() -> None:
    """Remove vault systemd units."""
    from .vault.lifecycle import VaultManager

    mgr = VaultManager()
    if not mgr.is_systemd_available():
        print("Error: systemd user services are not available. Nothing to uninstall.")
        raise SystemExit(1)
    mgr.uninstall_systemd_units()
    print("Vault systemd units removed.")


def _handle_vault_unlock(*, cfg: SandboxConfig | None = None) -> None:
    """Write the credentials-DB passphrase to the session-unlock tmpfs file; restart the daemon."""
    from ._yaml import write_secret_text
    from .credentials.encryption import prompt_passphrase
    from .vault.lifecycle import VaultManager

    if cfg is None:
        cfg = SandboxConfig()

    write_secret_text(cfg.vault_passphrase_file, prompt_passphrase() + "\n")
    print(f"→ wrote passphrase to {cfg.vault_passphrase_file} (RAM-backed, cleared on reboot)")

    mgr = VaultManager(cfg)
    if mgr.is_daemon_running():
        mgr.stop_daemon()
        mgr.start_daemon()
        print("→ vault daemon restarted")
    else:
        print("→ vault daemon is not running; start it with `terok-sandbox vault start`")


def _handle_vault_lock(*, cfg: SandboxConfig | None = None, forget: bool = False) -> None:
    """Delete the session-unlock tmpfs file and stop the vault daemon.

    By default this only clears the *session* tier; any of keyring,
    ``credentials.passphrase``, or a sealed systemd-creds credential
    can re-unlock the daemon at the next socket activation — and the
    operator may not realise that's not "locked".  Pass
    ``forget=True`` (``--forget`` on the CLI) to also remove the
    keyring entry, clear ``credentials.passphrase`` from ``config.yml``,
    and delete the sealed systemd-creds credential so the next daemon
    start *must* have an explicit ``vault unlock``.
    """
    from .credentials.encryption import forget_passphrase_in_keyring
    from .vault.lifecycle import VaultManager

    if cfg is None:
        cfg = SandboxConfig()

    path = cfg.vault_passphrase_file
    if path.exists():
        path.unlink()
        print(f"→ removed {path}")
    else:
        print(f"→ {path} was not present")

    sealed_cred = cfg.vault_systemd_creds_file
    if forget:
        if cfg.credentials_use_keyring:
            from .credentials.encryption import load_passphrase_from_keyring

            if forget_passphrase_in_keyring():
                print("→ cleared keyring entry")
            elif load_passphrase_from_keyring() is None:
                # ``keyring.delete_password`` raises on a missing entry on most
                # backends, which the helper folds to False — treat that as
                # "nothing to clear", but a residual entry after a False
                # return means the backend rejected the delete.
                print("→ keyring entry already absent")
            else:
                raise SystemExit(
                    "failed to clear keyring entry; daemon may still auto-unlock from keyring"
                )
        if cfg.credentials_passphrase:
            from . import config as _config
            from .paths import _config_file_paths

            user_config = next((p for label, p in _config_file_paths() if label == "user"), None)
            if user_config is not None and user_config.exists():
                _yaml_update_section(
                    user_config,
                    "credentials",
                    {"passphrase": None},  # nosec: B105 — clearing a config key
                )
                _config._credentials_section.cache_clear()
                print("→ cleared credentials.passphrase from config.yml")
        if sealed_cred.exists():
            try:
                sealed_cred.unlink()
            except OSError as exc:
                raise SystemExit(
                    f"failed to remove sealed credential at {sealed_cred}: {exc}"
                ) from exc
            print(f"→ removed sealed credential at {sealed_cred}")
    else:
        active = []
        if cfg.credentials_use_keyring:
            active.append("keyring")
        if cfg.credentials_passphrase:
            active.append("config.yml")
        if sealed_cred.is_file():
            active.append("systemd-creds")
        if active:
            print(
                f"warning: non-session passphrase tiers still active ({', '.join(active)});"
                " the daemon may auto-unlock on next socket activation.\n"
                "         Use `terok-sandbox vault lock --forget` to clear them too.",
                file=sys.stderr,
            )

    mgr = VaultManager(cfg)
    if mgr.is_daemon_running():
        mgr.stop_daemon()
        print("→ vault daemon stopped")


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


def _handle_vault_seal(*, cfg: SandboxConfig | None = None, key: str = "auto") -> None:
    """Seal the credentials-DB passphrase into a systemd-creds credential.

    Adds the systemd-creds tier to the resolution chain: machine-bound
    (TPM2 + host key, or either alone), survives reboot, no OS
    keyring required.  After sealing, the daemon resolves the
    passphrase via ``systemd-creds decrypt`` on every start — no
    operator interaction needed at boot, no plaintext-on-disk.

    *key=auto* (default) lets systemd pick the strongest available
    combination: ``host+tpm2`` on TPM-equipped hosts, ``host`` on
    hosts without a TPM.  *key=tpm* / *key=host* pin a single factor;
    *key=tpm+host* requires both (defense in depth, explicit).

    Requires an already-resolvable passphrase — typically from a fresh
    ``vault unlock`` in the current session.  Doesn't touch other tiers
    or restart the daemon; the new tier is picked up on the next chain
    walk.
    """
    from .credentials import systemd_creds
    from .credentials.encryption import resolve_passphrase

    if cfg is None:
        cfg = SandboxConfig()

    if not systemd_creds.is_available():
        raise SystemExit(
            "systemd-creds unavailable: needs systemd ≥ 257 with the Varlink"
            " io.systemd.Credentials interface (Fedora ≥ 42, Debian ≥ 13)"
        )

    key_mode = _SEAL_KEY_MODES.get(key)
    if key_mode is None:
        choices = ", ".join(sorted(_SEAL_KEY_MODES))
        raise SystemExit(f"unknown --key value: {key!r} (expected one of: {choices})")

    # Seal must reuse an already-resolved passphrase — a prompt here
    # would accept a fresh-typed value and seal *that*, leaving the
    # next chain walk holding a key that doesn't open the DB.  Every
    # persistent tier is included so re-sealing with a different
    # ``--key=`` mode on a headless host (where systemd-creds may be
    # the only resolvable source) is idempotent: the resolver returns
    # the same passphrase regardless of which tier it came from.
    passphrase = resolve_passphrase(
        passphrase_file=cfg.vault_passphrase_file,
        systemd_creds_file=cfg.vault_systemd_creds_file,
        use_keyring=cfg.credentials_use_keyring,
        config_fallback=cfg.credentials_passphrase,
        prompt_on_tty=False,
    )
    if passphrase is None:
        raise SystemExit("no current passphrase to seal — run `terok-sandbox vault unlock` first")

    try:
        systemd_creds.seal(passphrase, cfg.vault_systemd_creds_file, key_mode=key_mode)
    except RuntimeError as exc:
        # ``tpm2`` requested on a TPM-less host surfaces as a CalledProcessError
        # bubbled to RuntimeError; pass it through with the operator-facing
        # hint we'd otherwise have had to duplicate up here.
        raise SystemExit(str(exc)) from exc

    print(f"→ sealed passphrase to {cfg.vault_systemd_creds_file} (--with-key={key_mode})")
    print("  the resolution chain will pick this up on the next daemon start; no restart required")


VAULT_COMMANDS: tuple[CommandDef, ...] = (
    CommandDef(
        name="start",
        help="Start the vault daemon",
        handler=_handle_vault_start,
        group="vault",
    ),
    CommandDef(
        name="stop",
        help="Stop the vault daemon",
        handler=_handle_vault_stop,
        group="vault",
    ),
    CommandDef(
        name="status",
        help="Show vault status",
        handler=_handle_vault_status,
        group="vault",
    ),
    CommandDef(
        name="install",
        help="Install systemd socket activation",
        handler=_handle_vault_install,
        group="vault",
    ),
    CommandDef(
        name="uninstall",
        help="Remove systemd units",
        handler=_handle_vault_uninstall,
        group="vault",
    ),
    CommandDef(
        name="unlock",
        help="Provision the credentials-DB passphrase for this session (tmpfs file)",
        handler=_handle_vault_unlock,
        group="vault",
    ),
    CommandDef(
        name="lock",
        help="Remove the session-unlock tmpfs file and stop the vault daemon",
        handler=_handle_vault_lock,
        group="vault",
        args=(
            ArgDef(
                name="--forget",
                action="store_true",
                help=(
                    "Also clear keyring + credentials.passphrase so the daemon"
                    " cannot auto-unlock from non-session tiers"
                ),
            ),
        ),
    ),
    CommandDef(
        name="seal",
        help="Seal the current passphrase into a systemd-creds credential",
        handler=_handle_vault_seal,
        group="vault",
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
)

# ---------------------------------------------------------------------------
# SSH handlers
# ---------------------------------------------------------------------------


def _open_db(cfg: SandboxConfig) -> CredentialDB:
    """Open the vault credential DB for SSH operations (CLI flavour, TTY-prompt enabled)."""
    return cfg.open_credential_db(prompt_on_tty=True)


def _build_key_rows(cfg: SandboxConfig) -> list[KeyRow]:
    """Enumerate every registered SSH key as a displayable [`KeyRow`][terok_sandbox.commands.KeyRow].

    Shared by ``list`` and ``remove`` so both present identical
    information.  Returns an empty list when no keys are registered.
    """
    db = _open_db(cfg)
    try:
        rows: list[KeyRow] = []
        for scope in db.list_scopes_with_ssh_keys():
            for r in db.list_ssh_keys_for_scope(scope):
                rows.append(
                    KeyRow(
                        scope=scope,
                        comment=r.comment or f"id={r.id}",
                        key_type=r.key_type,
                        fingerprint=r.fingerprint,
                        private_key=f"db:ssh_keys/{r.id}",
                        public_key=f"db:ssh_keys/{r.id}",
                    )
                )
        rows.sort(key=lambda r: (r.scope, r.comment))
        return rows
    finally:
        db.close()


def _print_key_table(rows: list[KeyRow], *, numbered: bool = False) -> None:
    """Print a formatted table of SSH key rows.

    All untrusted fields are sanitized before display to prevent
    terminal escape injection from crafted key comments or paths.

    Args:
        rows: Key rows to display.
        numbered: Prefix each row with a 1-based index for interactive selection.
    """
    if not rows:
        print("No SSH keys registered.")
        return

    headers = ("SCOPE", "KEY", "TYPE", "FINGERPRINT", "PATH")
    # Sanitize untrusted fields before computing widths and formatting
    display = [
        tuple(
            sanitize_tty(f) for f in (r.scope, r.comment, r.key_type, r.fingerprint, r.public_key)
        )
        for r in rows
    ]
    widths = [max(len(h), *(len(d[i]) for d in display)) for i, h in enumerate(headers)]

    if numbered:
        idx_w = len(str(len(display)))
        prefix_w = idx_w + 2  # "N) " or "   "
        fmt = f"{{:<{prefix_w}}}" + "  ".join(f"{{:<{w}}}" for w in widths)
        print(fmt.format("", *headers))
        for i, d in enumerate(display, 1):
            print(fmt.format(f"{i})", *d))
    else:
        fmt = "  ".join(f"{{:<{w}}}" for w in widths)
        print(fmt.format(*headers))
        for d in display:
            print(fmt.format(*d))


def _validate_scope_name(scope: str) -> None:
    """Raise [`SystemExit`][SystemExit] if *scope* is not a safe identifier.

    Delegates to the canonical DB-layer validator so the character set
    *and* the length bound (derived from the AF_UNIX socket-path budget)
    stay co-located with the write sites that depend on them.
    """
    from .credentials.db import InvalidScopeName, _require_safe_scope

    try:
        _require_safe_scope(scope)
    except InvalidScopeName as exc:
        raise SystemExit(str(exc)) from exc


def _handle_ssh_import(
    *,
    scope: str,
    private_key: str,
    public_key: str | None = None,
    comment: str | None = None,
    cfg: SandboxConfig | None = None,
) -> None:
    """Import an OpenSSH keypair from files into the vault DB for *scope*."""
    from pathlib import Path

    from .config import SandboxConfig as _SandboxConfig
    from .credentials.db import UnsafeCommentError
    from .credentials.ssh_keypair import (
        KeypairMismatchError,
        PasswordProtectedKeyError,
        import_ssh_keypair,
    )

    _validate_scope_name(scope)
    if cfg is None:
        cfg = _SandboxConfig()

    priv_path = Path(private_key).expanduser().resolve()
    pub_path = Path(public_key).expanduser().resolve() if public_key else None

    if not priv_path.is_file():
        raise SystemExit(f"Private key not found: {priv_path}")
    if pub_path is not None and not pub_path.is_file():
        raise SystemExit(f"Public key not found: {pub_path}")

    db = _open_db(cfg)
    try:
        try:
            result = import_ssh_keypair(
                db,
                scope,
                priv_path,
                pub_path=pub_path,
                comment=comment,
            )
        except PasswordProtectedKeyError as exc:
            # The library message is diagnostic; append the CLI remediation here.
            raise SystemExit(
                f"{exc}  Run `ssh-keygen -p -f {priv_path}` to strip the passphrase."
            ) from exc
        except (KeypairMismatchError, UnsafeCommentError, ValueError) as exc:
            raise SystemExit(f"Import failed: {exc}") from exc

        pretty_scope = sanitize_tty(scope)
        if not result.already_present:
            headline = f"Imported new key to scope '{pretty_scope}':"
        elif result.scope_was_assigned:
            headline = f"Key already linked to scope '{pretty_scope}' — nothing to do:"
        else:
            headline = f"Linked existing vault key to scope '{pretty_scope}':"
        print(
            f"{headline}\n"
            f"  id:          {result.key_id}\n"
            f"  fingerprint: {sanitize_tty(result.fingerprint)}\n"
            f"  comment:     {sanitize_tty(result.comment)}"
        )
    finally:
        db.close()


def _handle_ssh_add(
    *,
    scope: str,
    key_type: str = "ed25519",
    comment: str | None = None,
    force: bool = False,
    cfg: SandboxConfig | None = None,
) -> None:
    """Generate a new SSH keypair in the vault for *scope*."""
    from .config import SandboxConfig as _SandboxConfig
    from .credentials.ssh import SSHManager

    _validate_scope_name(scope)
    if key_type not in ("ed25519", "rsa"):
        raise SystemExit("Unsupported --key-type. Use 'ed25519' or 'rsa'.")
    if cfg is None:
        cfg = _SandboxConfig()

    db = _open_db(cfg)
    try:
        manager = SSHManager(scope=scope, db=db)
        result = manager.init(key_type=key_type, comment=comment, force=force)
        print(f"SSH key ready for scope '{sanitize_tty(scope)}':")
        print(f"  id:          {result['key_id']}")
        print(f"  type:        {sanitize_tty(result['key_type'])}")
        print(f"  fingerprint: {sanitize_tty(result['fingerprint'])}")
        print(f"  comment:     {sanitize_tty(result['comment'])}")
        print("Public key (register as a deploy key):")
        print(f"  {sanitize_tty(result['public_line'])}")
    finally:
        db.close()


def _handle_ssh_export(
    *,
    scope: str,
    out_dir: str,
    key_id: int | None = None,
    out_name: str | None = None,
    cfg: SandboxConfig | None = None,
) -> None:
    """Write a scope's key back to a standard OpenSSH file pair."""
    from pathlib import Path

    from .config import SandboxConfig as _SandboxConfig
    from .credentials.ssh_keypair import export_ssh_keypair

    _validate_scope_name(scope)
    if cfg is None:
        cfg = _SandboxConfig()

    db = _open_db(cfg)
    try:
        try:
            result = export_ssh_keypair(
                db,
                scope,
                Path(out_dir).expanduser(),
                key_id=key_id,
                out_name=out_name,
            )
        except ValueError as exc:
            raise SystemExit(str(exc)) from exc
        except FileExistsError as exc:
            raise SystemExit(f"Refusing to overwrite existing file: {exc.filename}") from exc

        print(f"Exported key id={result.key_id} ({sanitize_tty(result.fingerprint)}):")
        print(f"  private key: {sanitize_tty(str(result.private_path))}")
        print(f"  public key:  {sanitize_tty(str(result.public_path))}")
    finally:
        db.close()


def _handle_ssh_pub(
    *,
    scope: str,
    key_id: int | None = None,
    all_keys: bool = False,
    cfg: SandboxConfig | None = None,
) -> None:
    """Print a scope's public key line(s) to stdout.

    Default: the most recently assigned key — the one likely to be the
    primary deploy key.  ``--all`` prints every key assigned to the scope
    (one per line, newest last); ``--key-id`` targets a specific row.
    """
    from .config import SandboxConfig as _SandboxConfig
    from .credentials.ssh_keypair import public_line_of

    _validate_scope_name(scope)
    if cfg is None:
        cfg = _SandboxConfig()

    if all_keys and key_id is not None:
        raise SystemExit("--all and --key-id are mutually exclusive")

    db = _open_db(cfg)
    try:
        records = db.load_ssh_keys_for_scope(scope)
        if not records:
            raise SystemExit(f"scope {scope!r} has no SSH keys assigned")
        if all_keys:
            for record in records:
                print(public_line_of(record))
            return
        if key_id is None:
            record = records[-1]
        else:
            matches = [r for r in records if r.id == key_id]
            if not matches:
                raise SystemExit(f"key_id {key_id} is not assigned to scope {scope!r}")
            record = matches[0]
        print(public_line_of(record))
    finally:
        db.close()


def _handle_ssh_link(
    *,
    key_id: int,
    scope: str,
    cfg: SandboxConfig | None = None,
) -> None:
    """Assign an already-stored ssh_keys row to an additional scope.

    The inverse of ``ssh remove`` — adds a row in ``ssh_key_assignments``
    linking *scope* to *key_id*.  Idempotent: re-linking an existing
    pair is a no-op.  Useful when several projects legitimately share a
    single deploy key.
    """
    from .config import SandboxConfig as _SandboxConfig

    _validate_scope_name(scope)
    if cfg is None:
        cfg = _SandboxConfig()

    db = _open_db(cfg)
    try:
        # Existence check up front — ``assign_ssh_key`` would otherwise
        # bubble a raw foreign-key error for a stale id.
        key_exists = db._conn.execute(  # noqa: SLF001 — one-shot read
            "SELECT 1 FROM ssh_keys WHERE id = ?", (key_id,)
        ).fetchone()
        if not key_exists:
            raise SystemExit(f"No ssh_keys row with id={key_id}")

        already_linked = db._conn.execute(  # noqa: SLF001
            "SELECT 1 FROM ssh_key_assignments WHERE scope = ? AND key_id = ?",
            (scope, key_id),
        ).fetchone()
        if already_linked:
            print(f"Scope '{sanitize_tty(scope)}' is already linked to key id={key_id}")
            return

        db.assign_ssh_key(scope, key_id)
        print(f"Linked key id={key_id} to scope '{sanitize_tty(scope)}'")
    finally:
        db.close()


def _handle_ssh_list(
    *,
    scope: str | None = None,
    cfg: SandboxConfig | None = None,
) -> None:
    """List SSH keys registered in the auth proxy's key store."""
    from .config import SandboxConfig as _SandboxConfig

    if cfg is None:
        cfg = _SandboxConfig()

    rows = _build_key_rows(cfg)
    if scope:
        filtered = [r for r in rows if r.scope == scope]
        if not filtered:
            raise SystemExit(f"No keys registered for scope {scope!r}")
        rows = filtered

    _print_key_table(rows)


def _filter_key_rows(
    rows: list[KeyRow],
    *,
    scope: str | None = None,
    comment: str | None = None,
    fingerprint: str | None = None,
) -> list[KeyRow]:
    """Narrow key rows by scope (exact), comment (glob), and fingerprint (prefix)."""
    from fnmatch import fnmatch

    if scope:
        rows = [r for r in rows if r.scope == scope]
    if comment:
        rows = [r for r in rows if fnmatch(r.comment, comment)]
    if fingerprint:
        fp = fingerprint.removeprefix("SHA256:")
        rows = [r for r in rows if r.fingerprint.removeprefix("SHA256:").startswith(fp)]
    return rows


def _key_id_from_row(row: KeyRow) -> int:
    """Extract the ``ssh_keys.id`` from a row's pseudo-path ``db:ssh_keys/<id>``."""
    return int(row.private_key.rsplit("/", 1)[-1])


def _handle_ssh_remove(
    *,
    scope: str | None = None,
    comment: str | None = None,
    fingerprint: str | None = None,
    yes: bool = False,
    cfg: SandboxConfig | None = None,
) -> None:
    """Unassign SSH keys from their scope(s); cascade-delete orphaned key rows.

    Two modes: interactive selection when called without filters, or
    direct matching when any of ``--scope``, ``--comment``, or
    ``--fingerprint`` is provided.
    """
    from .config import SandboxConfig as _SandboxConfig

    if cfg is None:
        cfg = _SandboxConfig()

    all_rows = _build_key_rows(cfg)
    if not all_rows:
        raise SystemExit("No SSH keys registered.")

    has_filters = any((scope, comment, fingerprint))

    if has_filters:
        candidates = _filter_key_rows(
            all_rows,
            scope=scope,
            comment=comment,
            fingerprint=fingerprint,
        )
        if not candidates:
            raise SystemExit("No keys match the given filters.")
        if not yes:
            n = len(candidates)
            if n > 1:
                print(f"Multiple keys match ({n}):\n")
            _print_key_table(candidates)
            prompt = f"\nRemove all {n} keys? [y/N]: " if n > 1 else "\nRemove this key? [y/N]: "
            try:
                answer = input(prompt).strip().lower()
            except (EOFError, KeyboardInterrupt):
                print()
                raise SystemExit("Aborted.") from None
            if answer not in ("y", "yes"):
                raise SystemExit("Aborted.")
    else:
        if yes:
            raise SystemExit("Cannot use --yes without at least one filter flag.")
        _print_key_table(all_rows, numbered=True)
        print()
        try:
            selection = input("Select key(s) to remove (number, comma-separated, or 'all'): ")
        except (EOFError, KeyboardInterrupt):
            print()
            raise SystemExit("Aborted.") from None
        selection = selection.strip().lower()
        if not selection:
            raise SystemExit("Aborted.")
        if selection == "all":
            candidates = list(all_rows)
        else:
            indices: list[int] = []
            for part in selection.split(","):
                part = part.strip()
                if not part.isdigit() or not (1 <= int(part) <= len(all_rows)):
                    raise SystemExit(
                        f"Invalid selection {part!r}. "
                        f"Enter a number 1–{len(all_rows)}, comma-separated, or 'all'."
                    )
                indices.append(int(part) - 1)
            candidates = [all_rows[i] for i in dict.fromkeys(indices)]

    db = _open_db(cfg)
    try:
        for row in candidates:
            db.unassign_ssh_key(row.scope, _key_id_from_row(row))
    finally:
        db.close()

    n = len(candidates)
    # Keys still assigned to *other* scopes survive the DB — unassign,
    # not remove, is the truthful verb for a possibly-shared key.
    print(f"Unassigned {n} key{'s' if n != 1 else ''} from their scope(s).")


SSH_COMMANDS: tuple[CommandDef, ...] = (
    CommandDef(
        name="list",
        help="List SSH keys stored in the vault",
        handler=_handle_ssh_list,
        group="ssh",
        args=(
            ArgDef(
                name="--scope",
                help="Show keys for a specific credential scope only",
                default=None,
            ),
        ),
    ),
    CommandDef(
        name="import",
        help="Import an OpenSSH keypair from files into the vault DB",
        handler=_handle_ssh_import,
        group="ssh",
        args=(
            ArgDef(name="scope", help="Credential scope to associate the key with"),
            ArgDef(
                name="--private-key",
                help="Path to the private key file",
                dest="private_key",
                required=True,
            ),
            ArgDef(
                name="--public-key",
                help="Path to the .pub file (default: derive from the private key)",
                default=None,
                dest="public_key",
            ),
            ArgDef(
                name="--comment",
                help="Override the key's comment string",
                default=None,
            ),
        ),
    ),
    CommandDef(
        name="add",
        help="Generate a new SSH keypair in the vault for a credential scope",
        handler=_handle_ssh_add,
        group="ssh",
        args=(
            ArgDef(name="scope", help="Credential scope to associate the key with"),
            ArgDef(
                name="--key-type",
                help="Key algorithm: ed25519 (default) or rsa",
                default="ed25519",
                dest="key_type",
            ),
            ArgDef(
                name="--comment",
                help="Comment embedded in the public key (default: tk-main:<scope>)",
                default=None,
            ),
            ArgDef(
                name="--force",
                help="Rotate — unassign all existing keys from the scope and generate fresh",
                action="store_true",
            ),
        ),
    ),
    CommandDef(
        name="export",
        help="Export a scope's SSH keypair to standard OpenSSH files",
        handler=_handle_ssh_export,
        group="ssh",
        args=(
            ArgDef(name="scope", help="Credential scope to export"),
            ArgDef(
                name="--out-dir",
                help="Directory to write files into",
                dest="out_dir",
                required=True,
            ),
            ArgDef(
                name="--key-id",
                help="Export a specific ssh_keys.id (default: most recently added)",
                default=None,
                dest="key_id",
                type=int,
            ),
            ArgDef(
                name="--out-name",
                help="Override the output filename stem (default: id_<type>_<fp8>)",
                default=None,
                dest="out_name",
            ),
        ),
    ),
    CommandDef(
        name="pub",
        help="Print a scope's public key to stdout",
        handler=_handle_ssh_pub,
        group="ssh",
        args=(
            ArgDef(name="scope", help="Credential scope"),
            ArgDef(
                name="--key-id",
                help="Specific ssh_keys.id (default: most recently added)",
                default=None,
                dest="key_id",
                type=int,
            ),
            ArgDef(
                name="--all",
                help="Print every key assigned to the scope, one per line",
                action="store_true",
                dest="all_keys",
            ),
        ),
    ),
    CommandDef(
        name="link",
        help="Link an existing vault key to an additional scope",
        handler=_handle_ssh_link,
        group="ssh",
        args=(
            ArgDef(name="scope", help="Credential scope to link the key to"),
            ArgDef(
                name="--key-id",
                help="ssh_keys.id of the key already stored in the vault",
                dest="key_id",
                type=int,
                required=True,
            ),
        ),
    ),
    CommandDef(
        name="remove",
        help="Unassign SSH keys from scopes (orphaned keys cascade-delete)",
        handler=_handle_ssh_remove,
        group="ssh",
        args=(
            ArgDef(
                name="--scope",
                help="Filter by credential scope (exact match)",
                default=None,
            ),
            ArgDef(
                name="--comment",
                help="Filter by comment (supports glob wildcards)",
                default=None,
            ),
            ArgDef(
                name="--fingerprint",
                help="Filter by fingerprint prefix (min 8 chars recommended)",
                default=None,
            ),
            ArgDef(
                name="--yes",
                help="Skip confirmation prompts",
                action="store_true",
                dest="yes",
            ),
        ),
    ),
)

# ---------------------------------------------------------------------------
# Doctor handler
# ---------------------------------------------------------------------------


def _handle_doctor(*, cfg: SandboxConfig | None = None) -> None:
    """Run sandbox-level health checks and print results.

    This is the standalone host-side doctor — it runs on the host, not
    inside a container.  For non-host_side checks (network probes), we
    execute the probe_cmd directly via subprocess instead of ``podman exec``.
    For host_side checks (e.g. shield), we delegate to ``evaluate`` which
    performs the check itself using Python APIs.
    """
    import subprocess
    import sys

    from .config import SandboxConfig as _SandboxConfig
    from .doctor import sandbox_doctor_checks
    from .vault.lifecycle import VaultManager

    if cfg is None:
        cfg = _SandboxConfig()
    mgr = VaultManager(cfg)
    checks = sandbox_doctor_checks(
        token_broker_port=mgr.token_broker_port,
        ssh_signer_port=mgr.ssh_signer_port,
        desired_shield_state=None,  # standalone mode — no task context
    )
    worst = "ok"
    markers = {"ok": "ok", "warn": "WARN", "error": "ERROR"}
    for check in checks:
        if check.host_side:
            # Host-side checks perform the check inside evaluate() itself.
            verdict = check.evaluate(0, "", "")
        elif check.probe_cmd:
            # Non-host_side checks: run probe_cmd directly on the host
            # (the command targets host.containers.internal which resolves
            # to localhost when not inside a container, so we rewrite to
            # localhost for standalone execution).
            try:
                result = subprocess.run(  # noqa: S603
                    check.probe_cmd,
                    capture_output=True,
                    text=True,
                    timeout=5,
                )
                verdict = check.evaluate(result.returncode, result.stdout, result.stderr)
            except (FileNotFoundError, subprocess.TimeoutExpired):
                verdict = check.evaluate(1, "", "probe command unavailable or timed out")
        else:
            verdict = check.evaluate(0, "", "")
        tag = markers.get(verdict.severity, verdict.severity)
        print(f"  {check.label} .... {tag} ({verdict.detail})")
        if verdict.severity == "error" or worst == "error":
            worst = "error"
        elif verdict.severity == "warn" or worst == "warn":
            worst = "warn"

    if worst == "error":
        sys.exit(2)
    elif worst == "warn":
        sys.exit(1)


DOCTOR_COMMANDS: tuple[CommandDef, ...] = (
    CommandDef(
        name="doctor",
        help="Run sandbox health checks",
        handler=_handle_doctor,
        group="doctor",
    ),
)


# ---------------------------------------------------------------------------
# Credentials DB at-rest encryption — chooser, provisioning, migration
#
# Three passphrase storage modes (chosen interactively, or session-unlock
# by default for automated installs):
#   - session-unlock: passphrase lives in $XDG_RUNTIME_DIR/.../vault.passphrase
#     (tmpfs, cleared on reboot; daemon reads at startup)
#   - OS keyring: passphrase lives in libsecret/Keychain/Credential Manager
#   - config file: passphrase inlined into config.yml (UNSAFE, on-disk)
#
# Once chosen, the mode is persisted so the resolution chain picks it
# up next time — session mode self-describes via the tmpfs file's
# presence; keyring sets credentials.use_keyring=true in config.yml;
# config writes the passphrase itself into config.yml.
#
# The plaintext→encrypted migration code path is deprecated in 0.9.0
# and slated for removal in 0.10.0: after that release fresh installs
# stay the only supported entry point, and the ``credentials
# encrypt-db`` CLI verb plus ``_handle_credentials_encrypt_db``'s
# ``is_plaintext_sqlite`` branch + ``encrypt_in_place`` call all go
# away.  Operators with a stale plaintext DB after 0.10.0 must restore
# from the ``.plaintext-backup-<stamp>.tar.gz`` snapshot the setup
# phase writes before re-keying.
# ---------------------------------------------------------------------------


_PassphraseMode = Literal["session", "keyring", "config"]

_CHOOSER_PROMPT = """\

Where should terok store the passphrase to encrypt the vault?
  [s] session-unlock — terok-sandbox vault unlock after each boot (default)
  [k] keyring — store passphrase in your login keyring
  [c] config file — UNSAFE, same disk as the encrypted DB

Choice [s]:"""

_CHOICE_TO_MODE: dict[str, _PassphraseMode] = {"s": "session", "k": "keyring", "c": "config"}


def _handle_credentials_encrypt_db(*, cfg: SandboxConfig | None = None) -> None:
    """Provision the credentials-DB passphrase; migrate any legacy plaintext file.

    An already-encrypted DB is short-circuited *before* we provision a
    new passphrase — minting a fresh one here would overwrite whatever
    tier currently holds the working key and lock the operator out.
    """
    from .credentials.encryption import encrypt_in_place, is_plaintext_sqlite

    if cfg is None:
        cfg = SandboxConfig()

    db_path = cfg.db_path
    if db_path.exists() and not is_plaintext_sqlite(db_path):
        print(f"  {db_path} is already SQLCipher-encrypted.")
        return

    mode = _ask_passphrase_mode()
    passphrase, source = _provision_passphrase(cfg, mode=mode)
    _persist_mode_choice(mode, passphrase)
    print(f"  passphrase source: {source}")

    if not db_path.exists():
        print(f"  no DB at {db_path}; will be created encrypted on first use.")
        return

    # Snapshot the plaintext DB (and any sidecars) before we touch
    # anything.  A failed migration would otherwise leave the operator
    # with a possibly-clobbered DB and no fallback — having a tarred
    # copy is cheap insurance.  The tarball is intentionally NOT
    # auto-deleted: the operator must explicitly remove it once they
    # have verified migration succeeded.
    backup_path = _back_up_plaintext_db(db_path)
    encrypt_in_place(db_path, passphrase)
    print(f"  encrypted {db_path} in place.")
    _warn_about_plaintext_backup(backup_path)


def _back_up_plaintext_db(db_path: Path) -> Path:
    """Tar the plaintext DB + WAL/SHM sidecars next to it, return the tarball path.

    The tarball contains cleartext secrets, so it must never go
    through a window of umask-default permissions: pre-create the
    file via ``O_CREAT | O_EXCL`` at 0o600 and stream the tar into
    that fd.  ``chmod`` after ``tarfile.open(path)`` would leave the
    secrets world-readable on hosts with a permissive umask for the
    duration of the write.
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


def _run_credentials_setup_phase(cfg: SandboxConfig) -> bool:
    """Migrate the credentials DB to SQLCipher; no-op on already-encrypted or absent.

    The vault daemon, if installed from a previous setup, holds a write
    lock on the plaintext DB (WAL mode).  Quiesce it before the
    migration so ``sqlcipher_export`` can ATTACH the source without
    colliding.  ``run_vault_install_phase`` re-installs+starts the
    daemon a moment later, so this stop is transient by design.

    On "database is locked" the phase self-heals: the socket-activation
    unit can re-spawn the service between our stop and the migration's
    open.  Uninstalling the units removes the trigger entirely, then we
    retry once before bailing — saves the operator from a manual
    ``systemctl --user stop`` + re-run.
    """
    from .vault.lifecycle import VaultManager

    print("→ credentials", end="", flush=True)
    mgr = VaultManager(cfg)
    _quiesce_vault_for_migration(mgr)
    try:
        _handle_credentials_encrypt_db(cfg=cfg)
    except Exception as exc:  # noqa: BLE001
        if not _looks_like_db_lock(exc):
            print(f" — FAILED: {exc}")
            return False
        # Socket activation may have respawned the daemon between our
        # quiesce and the migration's open; uninstalling the units kills
        # the trigger.
        print(" — locked, auto-recovering by uninstalling vault units …", flush=True)
        _quiesce_vault_for_migration(mgr, force_uninstall=True)
        try:
            _handle_credentials_encrypt_db(cfg=cfg)
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
    and PID-file daemons), but socket activation can race-respawn the
    service between stop and migration.  ``force_uninstall`` removes
    the unit files entirely — same pattern ``run_vault_install_phase``
    uses, just earlier in the timeline.
    """
    with contextlib.suppress(Exception):
        mgr.stop_daemon()
    if force_uninstall:
        with contextlib.suppress(Exception):
            mgr.uninstall_systemd_units()


def _looks_like_db_lock(exc: BaseException) -> bool:
    """Return ``True`` if *exc* is sqlite's "database is locked" complaint."""
    return "database is locked" in str(exc).lower()


def _ask_passphrase_mode() -> _PassphraseMode:
    """Return the operator's chosen mode; default to session on non-TTY runs."""
    if not sys.stdin.isatty():
        return "session"
    print(_CHOOSER_PROMPT)
    choice = sys.stdin.readline().strip().lower()[:1] or "s"
    return _CHOICE_TO_MODE.get(choice, "session")


def _provision_passphrase(
    cfg: SandboxConfig, *, mode: _PassphraseMode
) -> tuple[str, PassphraseSource]:
    """Resolve or mint a passphrase for *mode*; return ``(passphrase, source)``."""
    from ._yaml import write_secret_text
    from .credentials.encryption import (
        generate_passphrase,
        load_passphrase_from_file,
        load_passphrase_from_keyring,
        prompt_passphrase,
        store_passphrase_in_keyring,
    )

    if mode == "session":
        existing = load_passphrase_from_file(cfg.vault_passphrase_file)
        if existing is not None:
            return existing, "session-file"
        # Auto-generating silently would lock the operator out at next
        # boot — the tmpfs file vanishes, and ``vault unlock`` has no
        # way to re-type a passphrase they never saw.  Route through
        # ``prompt_passphrase(confirm=True)``: empty entry still mints
        # a fresh passphrase but echoes it once for the operator to
        # copy out before it lands on the tmpfs file.
        new = prompt_passphrase(confirm=True)
        write_secret_text(cfg.vault_passphrase_file, new + "\n")
        return new, "session-file"

    if mode == "keyring":
        existing = load_passphrase_from_keyring()
        if existing is not None:
            return existing, "keyring"
        new = generate_passphrase()
        if store_passphrase_in_keyring(new):
            return new, "keyring"
        raise RuntimeError("OS keyring is unreachable or denied; choose a different storage mode")

    if mode == "config":
        if cfg.credentials_passphrase:
            return cfg.credentials_passphrase, "config"
        print("Enter a passphrase to write to credentials.passphrase in config.yml:")
        return prompt_passphrase(confirm=True), "prompt"

    raise ValueError(f"unknown mode: {mode!r}")


def _persist_mode_choice(mode: _PassphraseMode, passphrase: str) -> None:
    """Write the chosen mode into config.yml so the chain re-resolves next time.

    Session mode needs no change — the tmpfs file is self-describing.
    Both fields (``use_keyring`` and ``passphrase``) are always written
    so that switching modes leaves a clean, single-source state — the
    previous mode's marker doesn't linger and the resolution chain
    can't see two tiers claiming ownership.
    """
    from . import config as _config
    from .paths import _config_file_paths

    user_config = next((p for label, p in _config_file_paths() if label == "user"), None)
    if user_config is None or mode == "session":
        return
    updates: dict[str, object | None] = (
        {"use_keyring": True, "passphrase": None}  # nosec: B105 — clearing a config key
        if mode == "keyring"
        else {"use_keyring": False, "passphrase": passphrase}
    )
    _yaml_update_section(user_config, "credentials", updates)
    # The resolution chain reads through ``_credentials_section`` which
    # is lru-cached; without invalidation the same process keeps seeing
    # the pre-setup state.
    _config._credentials_section.cache_clear()


CREDENTIALS_COMMANDS: tuple[CommandDef, ...] = (
    CommandDef(
        name="encrypt-db",
        help=(
            "Migrate a legacy plaintext credentials DB to SQLCipher-encrypted "
            "(deprecated in 0.9.0, removed in 0.10.0)"
        ),
        handler=_handle_credentials_encrypt_db,
        group="credentials",
    ),
)


# ---------------------------------------------------------------------------
# Container wiring — prepare / run / cleanup
#
# Compose (or exec into) the podman flags that wire a user-owned container
# into sandbox services.  Mirrors terok-shield's prepare/run shape and
# extends it with vault SSH signer, vault token broker, gate token, and
# bridge-resource volume wiring.  Container lifecycle stays with the user;
# sandbox owns only the services and per-container ancillary state.
# ---------------------------------------------------------------------------


_BRIDGES_EPILOG = """\
Container-side contract:
  The image must have `socat` installed and source the bridge script in
  its startup.  Two equally supported delivery paths:

    * Build-time:  COPY the bridge scripts into the image (any path);
                   RUN apt install -y socat;  source ensure-bridges.sh
                   from your entrypoint.
    * Runtime:     image already has socat; sandbox bind-mounts the
                   bridges at /usr/local/share/terok-sandbox/bridges/;
                   source ensure-bridges.sh from your entrypoint.

Without socat the container is still sandboxed (shield/userns apply)
but the broker/gate/SSH bridges cannot connect.
"""


def _handle_prepare(
    container: str,
    *,
    no_shield: bool = False,
    no_gate: bool = False,
    no_broker: bool = False,
    scope: str | None = None,
    profiles: list[str] | None = None,
    output_json: bool = False,
    cfg: SandboxConfig | None = None,
) -> None:
    """Print podman flags for sandboxing *container*.

    Mints any tokens needed for the active subsystems (broker/gate/ssh)
    and persists per-container state so [`_handle_cleanup`][terok_sandbox.commands._handle_cleanup]
    can reverse this invocation later.

    Args:
        container: Container name; becomes ``--name`` in the emitted args.
        no_shield: Disable the egress firewall (default: on).
        no_gate: Disable the git gate (default: on).
        no_broker: Disable the vault token broker (default: on).
        scope: Credential scope.  Required for gate/broker/ssh; omit for
            a shield-only run.
        profiles: Override shield profiles for this container.
        output_json: Emit a JSON array instead of a shell-quoted string.
        cfg: Optional [`SandboxConfig`][terok_sandbox.SandboxConfig] override.
    """
    from .launch import compose, format_args

    if cfg is None:
        cfg = SandboxConfig()
    args, _plan = compose(
        container,
        cfg=cfg,
        shield=not no_shield,
        gate=not no_gate,
        broker=not no_broker,
        scope=scope,
        profiles=tuple(profiles) if profiles else None,
    )
    print(format_args(args, output_json=output_json))


def _handle_run(
    container: str,
    *,
    no_shield: bool = False,
    no_gate: bool = False,
    no_broker: bool = False,
    scope: str | None = None,
    profiles: list[str] | None = None,
    podman_args: list[str] | None = None,
    cfg: SandboxConfig | None = None,
) -> None:
    """Launch *container* by exec-ing into ``podman run``.

    Same composition as [`_handle_prepare`][terok_sandbox.commands._handle_prepare]
    plus a collision check on the user-supplied trailing podman args and
    an ``os.execv`` into the podman binary.  Caller does not return.
    """
    from .launch import compose, exec_podman

    if cfg is None:
        cfg = SandboxConfig()
    sandbox_args, _plan = compose(
        container,
        cfg=cfg,
        shield=not no_shield,
        gate=not no_gate,
        broker=not no_broker,
        scope=scope,
        profiles=tuple(profiles) if profiles else None,
    )
    exec_podman(sandbox_args, podman_args or [])


def _handle_cleanup(container: str, *, cfg: SandboxConfig | None = None) -> None:
    """Reverse a prior `prepare`/`run` for *container*.

    Revokes minted tokens, calls [`shield.down`][terok_sandbox.shield.down],
    and removes the per-container state directory.  Idempotent — exits
    quietly when no state is found.
    """
    from .launch import cleanup

    if cfg is None:
        cfg = SandboxConfig()
    found = cleanup(container, cfg=cfg)
    if found:
        print(f"Cleaned up sandbox state for {container}.")
    else:
        print(f"No sandbox state found for {container}; nothing to clean up.")


LAUNCH_COMMANDS: tuple[CommandDef, ...] = (
    CommandDef(
        name="prepare",
        help="Print podman flags for sandboxing a user-owned container",
        handler=_handle_prepare,
        epilog=_BRIDGES_EPILOG,
        args=(
            ArgDef(name="container", help="Container name (becomes --name)"),
            ArgDef(
                name="--no-shield",
                action="store_true",
                help="Disable egress firewall (default: on)",
                dest="no_shield",
            ),
            ArgDef(
                name="--no-gate",
                action="store_true",
                help="Disable git gate (default: on; requires --scope)",
                dest="no_gate",
            ),
            ArgDef(
                name="--no-broker",
                action="store_true",
                help="Disable vault token broker (default: on; requires --scope)",
                dest="no_broker",
            ),
            ArgDef(
                name="--scope",
                help="Credential scope; enables vault SSH agent and is required by gate/broker",
            ),
            ArgDef(
                name="--profiles",
                nargs="+",
                help="Override shield profiles for this container",
            ),
            ArgDef(
                name="--json",
                action="store_true",
                dest="output_json",
                help="Output JSON array instead of a shell-quoted string",
            ),
        ),
    ),
    CommandDef(
        name="run",
        help="Launch a sandboxed user-owned container (exec into podman run)",
        handler=_handle_run,
        epilog=_BRIDGES_EPILOG,
        args=(
            ArgDef(name="container", help="Container name (becomes --name)"),
            ArgDef(
                name="--no-shield",
                action="store_true",
                help="Disable egress firewall (default: on)",
                dest="no_shield",
            ),
            ArgDef(
                name="--no-gate",
                action="store_true",
                help="Disable git gate (default: on; requires --scope)",
                dest="no_gate",
            ),
            ArgDef(
                name="--no-broker",
                action="store_true",
                help="Disable vault token broker (default: on; requires --scope)",
                dest="no_broker",
            ),
            ArgDef(
                name="--scope",
                help="Credential scope; enables vault SSH agent and is required by gate/broker",
            ),
            ArgDef(
                name="--profiles",
                nargs="+",
                help="Override shield profiles for this container",
            ),
        ),
    ),
    CommandDef(
        name="cleanup",
        help="Revoke tokens and drop shield rules for a sandboxed container",
        handler=_handle_cleanup,
        args=(ArgDef(name="container", help="Container name to clean up"),),
    ),
)


#: All sandbox commands, grouped by subsystem.
COMMANDS: tuple[CommandDef, ...] = (
    SETUP_COMMANDS
    + GATE_COMMANDS
    + SHIELD_COMMANDS
    + VAULT_COMMANDS
    + SSH_COMMANDS
    + DOCTOR_COMMANDS
    + LAUNCH_COMMANDS
)
