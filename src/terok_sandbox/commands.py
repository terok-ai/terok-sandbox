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


def _handle_gate_start(*, port: int | None = None, daemon: bool = False) -> None:
    """Start the gate server (systemd preferred, daemon fallback)."""
    from .gate_server import install_systemd_units, is_systemd_available, start_daemon

    if is_systemd_available() and not daemon:
        install_systemd_units()
        print("Gate server started via systemd socket activation.")
    else:
        start_daemon(port=port)
        print("Gate server daemon started.")


def _handle_gate_stop() -> None:
    """Stop the gate server."""
    from .gate_server import get_server_status, stop_daemon, uninstall_systemd_units

    status = get_server_status()
    if status.mode == "systemd":
        uninstall_systemd_units()
        print("Gate server systemd units removed.")
    elif status.mode == "daemon":
        stop_daemon()
        print("Gate server daemon stopped.")
    else:
        print("Gate server is not running.")


def _handle_gate_status() -> None:
    """Show gate server status."""
    from .gate_server import check_units_outdated, get_gate_base_path, get_server_status

    status = get_server_status()
    print(f"Mode:      {status.mode}")
    print(f"Running:   {'yes' if status.running else 'no'}")
    print(f"Port:      {status.port}")
    print(f"Base path: {get_gate_base_path()}")

    warning = check_units_outdated()
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
)

#: All sandbox commands, grouped by subsystem.
COMMANDS: tuple[CommandDef, ...] = GATE_COMMANDS + SHIELD_COMMANDS + PROXY_COMMANDS
