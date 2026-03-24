# SPDX-FileCopyrightText: 2025 Jiri Vyskocil
# SPDX-FileCopyrightText: 2026 Jiri Vyskocil
# SPDX-License-Identifier: Apache-2.0

"""CLI entry point for terok-sandbox.

Subcommands delegate directly to the package's module-level functions —
no logic lives here beyond argument parsing and error presentation.
"""

from __future__ import annotations

import argparse
import sys
from importlib.metadata import PackageNotFoundError, version as _meta_version

try:
    __version__ = _meta_version("terok-sandbox")
except PackageNotFoundError:
    __version__ = "0.0.0"

# ---------------------------------------------------------------------------
# shield
# ---------------------------------------------------------------------------


def _cmd_shield_setup(args: argparse.Namespace) -> None:
    """Install OCI hooks for the shield firewall."""
    from .shield import run_setup

    run_setup(root=args.root, user=args.user)


def _cmd_shield_status(_args: argparse.Namespace) -> None:
    """Show shield configuration and environment check."""
    from .shield import check_environment, status

    env = check_environment()
    cfg = status()

    print(f"Shield mode:    {cfg.get('mode', '?')}")
    print(f"Profiles:       {', '.join(cfg.get('profiles', []))}")
    print(f"Audit:          {cfg.get('audit', '?')}")
    print(f"Hooks:          {env.hooks}")
    print(f"Health:         {env.health}")
    if env.needs_setup:
        print(f"\n{env.setup_hint}", file=sys.stderr)


# ---------------------------------------------------------------------------
# gate
# ---------------------------------------------------------------------------


def _cmd_gate_start(args: argparse.Namespace) -> None:
    """Start the gate server."""
    from .gate_server import is_systemd_available, start_daemon

    if is_systemd_available() and not args.daemon:
        from .gate_server import install_systemd_units

        install_systemd_units()
        print("Gate server started via systemd socket activation.")
    else:
        start_daemon(port=args.port)
        print("Gate server daemon started.")


def _cmd_gate_stop(_args: argparse.Namespace) -> None:
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


def _cmd_gate_status(_args: argparse.Namespace) -> None:
    """Show gate server status."""
    from .gate_server import check_units_outdated, get_gate_base_path, get_server_status

    status = get_server_status()
    print(f"Mode:      {status.mode}")
    print(f"Running:   {'yes' if status.running else 'no'}")
    print(f"Port:      {status.port}")
    print(f"Base path: {get_gate_base_path()}")

    warning = check_units_outdated()
    if warning:
        print(f"\nWarning: {warning}", file=sys.stderr)


# ---------------------------------------------------------------------------
# main
# ---------------------------------------------------------------------------


def main() -> None:
    """Entry point for the ``terok-sandbox`` command."""
    parser = argparse.ArgumentParser(
        prog="terok-sandbox",
        description="Hardened Podman container runtime with shield firewall and git gate",
    )
    parser.add_argument("--version", action="version", version=f"terok-sandbox {__version__}")
    sub = parser.add_subparsers()

    # -- shield --
    shield_p = sub.add_parser("shield", help="Egress firewall management")
    shield_sub = shield_p.add_subparsers()

    setup_p = shield_sub.add_parser("setup", help="Install OCI hooks")
    setup_p.add_argument(
        "--root", action="store_true", help="Install system-wide hooks (requires sudo)"
    )
    setup_p.add_argument("--user", action="store_true", help="Install user-level hooks")
    setup_p.set_defaults(func=_cmd_shield_setup)

    status_p = shield_sub.add_parser("status", help="Show shield status")
    status_p.set_defaults(func=_cmd_shield_status)

    shield_p.set_defaults(func=lambda _: shield_p.print_help())

    # -- gate --
    gate_p = sub.add_parser("gate", help="Git gate server management")
    gate_sub = gate_p.add_subparsers()

    start_p = gate_sub.add_parser("start", help="Start gate server")
    start_p.add_argument("--port", type=int, default=None, help="Override port (default: 9418)")
    start_p.add_argument("--daemon", action="store_true", help="Force daemon mode (skip systemd)")
    start_p.set_defaults(func=_cmd_gate_start)

    stop_p = gate_sub.add_parser("stop", help="Stop gate server")
    stop_p.set_defaults(func=_cmd_gate_stop)

    gstatus_p = gate_sub.add_parser("status", help="Show gate status")
    gstatus_p.set_defaults(func=_cmd_gate_status)

    gate_p.set_defaults(func=lambda _: gate_p.print_help())

    args = parser.parse_args()
    if hasattr(args, "func"):
        args.func(args)
    else:
        parser.print_help()
        raise SystemExit(1)


if __name__ == "__main__":
    main()
