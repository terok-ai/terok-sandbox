# SPDX-FileCopyrightText: 2026 Jiri Vyskocil
# SPDX-License-Identifier: Apache-2.0

"""Shield CLI verbs — install-hooks, uninstall-hooks, status.

Thin wrappers around the shield adapter; the install/uninstall handlers
turn library ``ValueError``s into ``SystemExit`` with CLI-specific
remediation hints so the library function stays UX-agnostic.
"""

from __future__ import annotations

import sys

from ._types import ArgDef, CommandDef


def _handle_shield_setup(*, root: bool = False, user: bool = False) -> None:
    """Install OCI hooks for the shield firewall."""
    if not root and not user:
        raise SystemExit(
            "Specify --root (system-wide, uses sudo) or --user (user-local).\n"
            "  shield install-hooks --root   # /etc/containers/oci/hooks.d\n"
            "  shield install-hooks --user   # ~/.local/share/containers/oci/hooks.d"
        )
    from ..shield import run_setup

    run_setup(root=root, user=user)


def _handle_shield_uninstall(*, root: bool = False, user: bool = False) -> None:
    """Remove the OCI hooks previously installed by ``shield install-hooks`` (idempotent)."""
    if not root and not user:
        raise SystemExit(
            "Specify --root (system-wide, uses sudo) or --user (user-local).\n"
            "  shield uninstall-hooks --root   # /etc/containers/oci/hooks.d\n"
            "  shield uninstall-hooks --user   # ~/.local/share/containers/oci/hooks.d"
        )
    from ..shield import run_uninstall

    run_uninstall(root=root, user=user)
    scope = "system" if root else "user"
    print(f"Shield hooks removed from {scope} hooks directory.")


def _handle_shield_status() -> None:
    """Show shield configuration and environment check."""
    from ..shield import check_environment, status

    env = check_environment()
    cfg = status()

    print(f"Shield mode:    {cfg.get('mode', '?')}")
    print(f"Profiles:       {', '.join(cfg.get('profiles', []))}")
    print(f"Audit:          {'enabled' if cfg.get('audit_enabled') else 'disabled'}")
    print(f"Hooks:          {env.hooks}")
    print(f"Health:         {env.health}")
    if env.needs_setup:
        print(f"\n{env.setup_hint}", file=sys.stderr)


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


__all__ = ["SHIELD_COMMANDS"]
