# SPDX-FileCopyrightText: 2026 Jiri Vyskocil
# SPDX-License-Identifier: Apache-2.0

"""Command registry for terok-sandbox.

Follows the same :class:`CommandDef` / :class:`ArgDef` pattern as
``terok_shield.registry``.  Higher-level consumers (terok, terok-agent)
can import ``COMMANDS`` to build their own CLI frontends without
duplicating argument definitions or handler logic.

Shield commands are delegated to terok-shield's own registry —
``SHIELD_COMMANDS`` re-exports the non-standalone subset.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from collections.abc import Callable

    from .config import SandboxConfig


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
# Gate handlers
# ---------------------------------------------------------------------------


def _handle_gate_start(
    *, port: int | None = None, daemon: bool = False, cfg: SandboxConfig | None = None
) -> None:
    """Start the gate server (systemd preferred, daemon fallback)."""
    from .gate_server import install_systemd_units, is_systemd_available, start_daemon

    if is_systemd_available() and not daemon:
        install_systemd_units(cfg=cfg)
        print("Gate server started via systemd socket activation.")
    else:
        start_daemon(port=port, cfg=cfg)
        print("Gate server daemon started.")


def _handle_gate_stop(*, cfg: SandboxConfig | None = None) -> None:
    """Stop the gate server."""
    from .gate_server import get_server_status, stop_daemon, uninstall_systemd_units

    status = get_server_status(cfg=cfg)
    if status.mode == "systemd":
        uninstall_systemd_units(cfg=cfg)
        print("Gate server systemd units removed.")
    elif status.mode == "daemon":
        stop_daemon(cfg=cfg)
        print("Gate server daemon stopped.")
    else:
        print("Gate server is not running.")


def _handle_gate_status(*, cfg: SandboxConfig | None = None) -> None:
    """Show gate server status."""
    from .gate_server import check_units_outdated, get_gate_base_path, get_server_status

    status = get_server_status(cfg=cfg)
    print(f"Mode:      {status.mode}")
    print(f"Running:   {'yes' if status.running else 'no'}")
    print(f"Port:      {status.port}")
    print(f"Base path: {get_gate_base_path(cfg=cfg)}")

    warning = check_units_outdated(cfg=cfg)
    if warning:
        import sys

        print(f"\nWarning: {warning}", file=sys.stderr)


# ---------------------------------------------------------------------------
# Shield handlers (thin wrappers around terok_sandbox.shield)
# ---------------------------------------------------------------------------


def _handle_shield_setup(*, root: bool = False, user: bool = False) -> None:
    """Install OCI hooks for the shield firewall."""
    from .shield import run_setup

    run_setup(root=root, user=user)


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
        name="setup",
        help="Install OCI hooks for the shield firewall",
        handler=_handle_shield_setup,
        group="shield",
        args=(
            ArgDef(name="--root", action="store_true", help="Install system-wide (requires sudo)"),
            ArgDef(name="--user", action="store_true", help="Install to user hooks directory"),
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
# Credential proxy handlers
# ---------------------------------------------------------------------------


def _handle_proxy_start() -> None:
    """Start the credential proxy daemon."""
    from .credential_proxy_lifecycle import get_proxy_status, start_daemon

    status = get_proxy_status()
    if status.running:
        print("Credential proxy is already running.")
        return
    start_daemon()
    print("Credential proxy started.")


def _handle_proxy_stop() -> None:
    """Stop the credential proxy daemon."""
    from .credential_proxy_lifecycle import is_daemon_running, stop_daemon

    if not is_daemon_running():
        print("Credential proxy is not running.")
        return
    stop_daemon()
    print("Credential proxy stopped.")


def _handle_proxy_status() -> None:
    """Show credential proxy status."""
    from .credential_proxy_lifecycle import get_proxy_status

    status = get_proxy_status()
    state = "running" if status.running else "stopped"
    print(f"Status: {state}")
    print(f"Socket: {status.socket_path}")
    print(f"DB:     {status.db_path}")
    print(f"Routes: {status.routes_path} ({status.routes_configured} configured)")
    if status.credentials_stored:
        print(f"Credentials: {', '.join(status.credentials_stored)}")
    else:
        print("Credentials: none stored")


def _handle_proxy_install() -> None:
    """Install and start systemd socket activation for the credential proxy."""
    from .credential_proxy_lifecycle import install_systemd_units, is_systemd_available

    if not is_systemd_available():
        print("Error: systemd user services are not available on this host.")
        raise SystemExit(1)
    install_systemd_units()
    print("Credential proxy systemd socket installed and started.")


def _handle_proxy_uninstall() -> None:
    """Remove credential proxy systemd units."""
    from .credential_proxy_lifecycle import is_systemd_available, uninstall_systemd_units

    if not is_systemd_available():
        print("Error: systemd user services are not available. Nothing to uninstall.")
        raise SystemExit(1)
    uninstall_systemd_units()
    print("Credential proxy systemd units removed.")


PROXY_COMMANDS: tuple[CommandDef, ...] = (
    CommandDef(
        name="start",
        help="Start the credential proxy daemon",
        handler=_handle_proxy_start,
        group="proxy",
    ),
    CommandDef(
        name="stop",
        help="Stop the credential proxy daemon",
        handler=_handle_proxy_stop,
        group="proxy",
    ),
    CommandDef(
        name="status",
        help="Show credential proxy status",
        handler=_handle_proxy_status,
        group="proxy",
    ),
    CommandDef(
        name="install",
        help="Install systemd socket activation",
        handler=_handle_proxy_install,
        group="proxy",
    ),
    CommandDef(
        name="uninstall",
        help="Remove systemd units",
        handler=_handle_proxy_uninstall,
        group="proxy",
    ),
)

# ---------------------------------------------------------------------------
# SSH handlers
# ---------------------------------------------------------------------------


def _handle_ssh_import(
    *,
    project: str,
    private_key: str,
    public_key: str | None = None,
    cfg: SandboxConfig | None = None,
) -> None:
    """Copy an SSH keypair into the managed key store and register it in ssh-keys.json."""
    import os
    import re
    import shutil
    from pathlib import Path

    from .config import SandboxConfig as _SandboxConfig
    from .ssh import SSHInitResult, update_ssh_keys_json

    if not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9._-]*", project):
        raise SystemExit(
            f"Invalid project ID {project!r}: must start with a letter or digit "
            "and contain only [A-Za-z0-9._-]"
        )

    priv_src = Path(private_key).expanduser().resolve()
    pub_src = Path(public_key).expanduser().resolve() if public_key else Path(f"{priv_src}.pub")

    if not priv_src.exists():
        raise SystemExit(f"Private key not found: {priv_src}")
    if not pub_src.exists():
        raise SystemExit(
            f"Public key not found: {pub_src} (use --public-key to specify explicitly)"
        )

    if cfg is None:
        cfg = _SandboxConfig()
    dest_dir = cfg.ssh_keys_dir / project
    dest_dir.mkdir(parents=True, exist_ok=True)

    def _unique_dst(src: Path) -> Path:
        """Return dest path for *src* inside *dest_dir*, with numeric suffix on collision.

        A destination that already exists with identical content is considered
        the same key (re-import) and returns the existing path unchanged.
        """
        p = dest_dir / src.name
        if not p.exists() or p.read_bytes() == src.read_bytes():
            return p
        stem, suffix = p.stem, p.suffix
        n = 1
        while True:
            p = dest_dir / f"{stem}_{n}{suffix}"
            if not p.exists() or p.read_bytes() == src.read_bytes():
                return p
            n += 1

    priv_dst = _unique_dst(priv_src)
    pub_dst = _unique_dst(pub_src)

    print(f"Copying private key: {priv_src}")
    print(f"              → {priv_dst}")
    shutil.copy2(str(priv_src), str(priv_dst))
    os.chmod(priv_dst, 0o600)

    print(f"Copying public key:  {pub_src}")
    print(f"              → {pub_dst}")
    shutil.copy2(str(pub_src), str(pub_dst))

    result = SSHInitResult(
        dir=str(dest_dir),
        private_key=str(priv_dst),
        public_key=str(pub_dst),
        config_path="",
        key_name=priv_dst.name,
    )
    keys_path = cfg.ssh_keys_json_path
    update_ssh_keys_json(keys_path, project, result)
    print(f"Registered key for project '{project}': {priv_dst}")


def _handle_ssh_add_key(
    *,
    project: str,
    name: str | None = None,
    key_type: str = "ed25519",
    cfg: SandboxConfig | None = None,
) -> None:
    """Generate a new SSH keypair and register it for a project."""
    import re

    from .config import SandboxConfig as _SandboxConfig
    from .ssh import (
        SSHInitResult,
        _harden_permissions,
        _next_key_number,
        generate_keypair,
        update_ssh_keys_json,
    )

    if not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9._-]*", project):
        raise SystemExit(
            f"Invalid project ID {project!r}: must start with a letter or digit "
            "and contain only [A-Za-z0-9._-]"
        )
    if key_type not in ("ed25519", "rsa"):
        raise SystemExit("Unsupported --key-type. Use 'ed25519' or 'rsa'.")

    algo = "ed25519" if key_type == "ed25519" else "rsa"
    if cfg is None:
        cfg = _SandboxConfig()
    dest_dir = cfg.ssh_keys_dir / project
    dest_dir.mkdir(parents=True, exist_ok=True)

    if name is not None:
        if not re.fullmatch(r"[a-zA-Z_-]+", name):
            raise SystemExit(
                f"Invalid key name {name!r}: must contain only letters, underscores, and hyphens"
            )
        key_name = name
    else:
        key_name = f"key-{_next_key_number(dest_dir, algo)}"

    filename = f"id_{algo}_{key_name}"
    priv_path = dest_dir / filename
    pub_path = dest_dir / f"{filename}.pub"

    existing = priv_path if priv_path.exists() else pub_path if pub_path.exists() else None
    if existing:
        raise SystemExit(
            f"Key file already exists: {existing}\n"
            "Use a different --name or remove the existing key first."
        )

    comment = f"tk-side:{project} {key_name}"
    generate_keypair(key_type, priv_path, pub_path, comment)

    try:
        _harden_permissions(dest_dir, priv_path, pub_path, dest_dir / "config")
    except OSError as e:
        raise SystemExit(f"Failed to set permissions: {e}") from e

    result = SSHInitResult(
        dir=str(dest_dir),
        private_key=str(priv_path),
        public_key=str(pub_path),
        config_path="",
        key_name=filename,
    )
    update_ssh_keys_json(cfg.ssh_keys_json_path, project, result)

    print(f"SSH key generated for project '{project}':")
    print(f"  name:        {key_name}")
    print(f"  private key: {priv_path}")
    print(f"  public key:  {pub_path}")
    print(f"  comment:     {comment}")
    try:
        pub_text = pub_path.read_text(encoding="utf-8", errors="ignore").strip()
        if pub_text:
            print("Public key (add as deploy key):")
            print(f"  {pub_text}")
    except Exception:
        pass


SSH_COMMANDS: tuple[CommandDef, ...] = (
    CommandDef(
        name="import",
        help="Register an existing SSH keypair in ssh-keys.json",
        handler=_handle_ssh_import,
        group="ssh",
        args=(
            ArgDef(name="project", help="Project ID to associate the key with"),
            ArgDef(
                name="--private-key",
                help="Path to the private key file",
                dest="private_key",
                required=True,
            ),
            ArgDef(
                name="--public-key",
                help="Path to the .pub file (default: <private-key>.pub)",
                default=None,
                dest="public_key",
            ),
        ),
    ),
    CommandDef(
        name="add-key",
        help="Generate a new SSH keypair for a project",
        handler=_handle_ssh_add_key,
        group="ssh",
        args=(
            ArgDef(name="project", help="Project ID to associate the key with"),
            ArgDef(
                name="--name",
                help="Key name ([a-zA-Z_-]; auto-generates key-1, key-2, ... if omitted)",
                default=None,
            ),
            ArgDef(
                name="--key-type",
                help="Key algorithm: ed25519 (default) or rsa",
                default="ed25519",
                dest="key_type",
            ),
        ),
    ),
)

#: All sandbox commands, grouped by subsystem.
COMMANDS: tuple[CommandDef, ...] = GATE_COMMANDS + SHIELD_COMMANDS + PROXY_COMMANDS + SSH_COMMANDS
