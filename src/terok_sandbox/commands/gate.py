# SPDX-FileCopyrightText: 2026 Jiri Vyskocil
# SPDX-License-Identifier: Apache-2.0

"""Gate-server CLI verbs — install, uninstall, start, stop, status.

Thin handlers that delegate to
[`GateServerManager`][terok_sandbox.gate.lifecycle.GateServerManager];
this module owns only the CLI shape.
"""

from __future__ import annotations

import sys
from typing import TYPE_CHECKING

from ._types import ArgDef, CommandDef

if TYPE_CHECKING:
    from ..config import SandboxConfig


def _handle_gate_install(*, cfg: SandboxConfig | None = None) -> None:
    """Install gate server systemd units, refusing hosts without systemd-user."""
    from ..gate.lifecycle import GateServerManager

    mgr = GateServerManager(cfg)
    if not mgr.is_systemd_available():
        print("Error: systemd user services are not available on this host.")
        raise SystemExit(1)
    mgr.install_systemd_units()
    print("Gate server installed via systemd socket activation.")


def _handle_gate_uninstall(*, cfg: SandboxConfig | None = None) -> None:
    """Remove gate server systemd units, stopping any stray daemon first."""
    from ..gate.lifecycle import GateServerManager

    mgr = GateServerManager(cfg)
    if mgr.get_status().mode == "daemon":
        mgr.stop_daemon()
    if mgr.is_systemd_available():
        mgr.uninstall_systemd_units()
    print("Gate server systemd units removed.")


def _handle_gate_start(
    *, port: int | None = None, daemon: bool = False, cfg: SandboxConfig | None = None
) -> None:
    """Start the gate server (systemd preferred, daemon fallback)."""
    from ..gate.lifecycle import GateServerManager

    mgr = GateServerManager(cfg)
    if mgr.is_systemd_available() and not daemon:
        mgr.install_systemd_units()
        print("Gate server started via systemd socket activation.")
    else:
        mgr.start_daemon(port=port)
        print("Gate server daemon started.")


def _handle_gate_stop(*, cfg: SandboxConfig | None = None) -> None:
    """Stop the gate server."""
    from ..gate.lifecycle import GateServerManager

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
    from ..gate.lifecycle import GateServerManager

    mgr = GateServerManager(cfg)
    status = mgr.get_status()
    print(f"Mode:      {status.mode}")
    print(f"Running:   {'yes' if status.running else 'no'}")
    print(f"Port:      {status.port}")
    print(f"Base path: {mgr.gate_base_path}")

    warning = mgr.check_units_outdated()
    if warning:
        print(f"\nWarning: {warning}", file=sys.stderr)
        print("Run 'terok-sandbox gate start' to update.", file=sys.stderr)


GATE_COMMANDS: tuple[CommandDef, ...] = (
    CommandDef(
        name="install",
        help="Install systemd socket activation for the gate server",
        handler=_handle_gate_install,
        group="gate",
    ),
    CommandDef(
        name="uninstall",
        help="Remove gate server systemd units",
        handler=_handle_gate_uninstall,
        group="gate",
    ),
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


__all__ = ["GATE_COMMANDS"]
