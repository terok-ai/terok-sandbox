# SPDX-FileCopyrightText: 2026 Jiri Vyskocil
# SPDX-License-Identifier: Apache-2.0

"""Command registry for terok-sandbox.

Follows the same :class:`CommandDef` / :class:`ArgDef` pattern as
``terok_shield.registry``.  Higher-level consumers (terok, terok-executor)
can import ``COMMANDS`` to build their own CLI frontends without
duplicating argument definitions or handler logic.

Shield commands are delegated to terok-shield's own registry —
``SHIELD_COMMANDS`` re-exports the non-standalone subset.
"""

from __future__ import annotations

import sys
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, NamedTuple

from ._util import sanitize_tty

if TYPE_CHECKING:
    from collections.abc import Callable

    from .config import SandboxConfig


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
    """

    name: str
    help: str = ""
    handler: Callable[..., None] | None = None
    args: tuple[ArgDef, ...] = ()
    group: str = ""


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
) -> None:
    """Install shield hooks, vault, and gate as one idempotent bootstrap."""
    if not no_shield:
        print("→ shield install-hooks")
        _handle_shield_setup(user=not root, root=root)
    if not no_vault:
        print("→ vault install")
        _handle_vault_install()
    if not no_gate:
        print("→ gate install")
        _handle_gate_install()


def _handle_sandbox_uninstall(
    *,
    root: bool = False,
    no_shield: bool = False,
    no_vault: bool = False,
    no_gate: bool = False,
) -> None:
    """Tear down the stack in reverse install order.

    A running container can lose its gate and vault without immediate
    blast, but losing shield hooks mid-flight is the most disruptive —
    shield goes last so live containers stay firewalled as long as
    possible.

    Best-effort across phases: a failing phase reports the error and
    the next phase runs anyway, so a partial-install teardown still
    removes what it can instead of leaving orphans behind.  Exits
    non-zero only after every phase has had its attempt.
    """
    failed = False
    if not no_gate:
        print("→ gate uninstall")
        failed |= _try_phase(_handle_gate_uninstall)
    if not no_vault:
        print("→ vault uninstall")
        failed |= _try_phase(_handle_vault_uninstall)
    if not no_shield:
        print("→ shield uninstall-hooks")
        failed |= _try_phase(lambda: _handle_shield_uninstall(user=not root, root=root))
    if failed:
        raise SystemExit(1)


def _try_phase(phase: Callable[[], None]) -> bool:
    """Run one uninstall phase, reporting but not re-raising on failure.

    Returns True when the phase failed so the aggregator can decide
    the exit code after every phase has had its attempt.
    """
    try:
        phase()
    except SystemExit as exc:
        print(f"  phase failed: {exc}", file=sys.stderr)
        return True
    return False


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
    the library function (:func:`shield.run_setup`) can stay UX-agnostic:
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
    :func:`_handle_shield_setup`: ``--root`` uses sudo, ``--user``
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
)

# ---------------------------------------------------------------------------
# SSH handlers
# ---------------------------------------------------------------------------


def _open_db(cfg: SandboxConfig):
    """Open the vault credential DB for SSH operations."""
    from .credentials.db import CredentialDB

    return CredentialDB(cfg.db_path)


def _build_key_rows(cfg: SandboxConfig) -> list[KeyRow]:
    """Enumerate every registered SSH key as a displayable :class:`KeyRow`.

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
    """Raise :class:`SystemExit` if *scope* is not a safe identifier.

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
    from .credentials.ssh_keypair import (
        KeypairMismatchError,
        PasswordProtectedKeyError,
        UnsafeCommentError,
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

#: All sandbox commands, grouped by subsystem.
COMMANDS: tuple[CommandDef, ...] = (
    SETUP_COMMANDS
    + GATE_COMMANDS
    + SHIELD_COMMANDS
    + VAULT_COMMANDS
    + SSH_COMMANDS
    + DOCTOR_COMMANDS
)
