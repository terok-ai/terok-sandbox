# SPDX-FileCopyrightText: 2026 Jiri Vyskocil
# SPDX-License-Identifier: Apache-2.0

"""Shield CLI surface — sandbox-side install + the full shield registry.

Composes sandbox's own ``install-hooks`` / ``uninstall-hooks`` admin
verbs with every non-standalone-only entry from
[`terok_shield.COMMANDS`][terok_shield.COMMANDS].  ``standalone_only=True``
on shield's CommandDefs is the explicit "skip me when consumed
downstream" marker; we honour it (filters ``setup``, ``prepare``,
``run``, ``resolve``).

Per-container shield verbs (``allow``, ``deny``, ``down``, ``up``,
``block``, ``rules``, ``watch``, ``simple-clearance``, ``logs``) bind
a [`terok_shield.Shield`][terok_shield.Shield] via shield's own
``resolve_state_dir`` so the standalone CLI and the sandbox-wrapped
form behave identically.
"""

from __future__ import annotations

import functools
from typing import TYPE_CHECKING, Any

from terok_shield import COMMANDS as _SHIELD_REGISTRY, Shield
from terok_shield.cli.main import _build_config as _shield_build_config  # noqa: PLC2701

from ._types import ArgDef, CommandDef

if TYPE_CHECKING:
    from collections.abc import Callable


def _handle_shield_setup(*, root: bool = False, user: bool = False) -> None:
    """Install OCI hooks for the shield firewall.

    Sandbox-specific entry point — shield's own ``setup`` verb is
    ``standalone_only`` (its CLI argv shape needs ``--check``,
    ``--root``, ``--user`` handling that doesn't fit a generic
    registry handler).  This verb is the integration-friendly form.
    """
    if not root and not user:
        raise SystemExit(
            "Specify --root (system-wide, uses sudo) or --user (user-local).\n"
            "  shield install-hooks --root   # /etc/containers/oci/hooks.d\n"
            "  shield install-hooks --user   # ~/.local/share/containers/oci/hooks.d"
        )
    from ..shield import run_setup

    run_setup(root=root, user=user)


def _handle_shield_uninstall(*, root: bool = False, user: bool = False) -> None:
    """Remove the OCI hooks previously installed by ``shield install-hooks``."""
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


def _wrap_shield_handler(
    handler: Callable[..., Any], *, needs_container: bool
) -> Callable[..., Any]:
    """Bind a per-container [`Shield`][terok_shield.Shield] around *handler*.

    Shield's registry handlers expect ``(shield, [container], **kwargs)``.
    Sandbox dispatch hands the handler ``**kwargs`` from argparse — we
    extract ``container`` (when required), resolve its ``state_dir``
    via shield's own helper, build a ``Shield``, and forward in the
    shape shield expects.
    """

    @functools.wraps(handler)
    def wrapped(**kwargs: Any) -> Any:
        container = kwargs.pop("container", None)
        # Reach for shield's private ``_build_config`` until it's
        # promoted to public API — sandbox needs the same state-dir
        # resolution shield's standalone CLI uses (podman annotation
        # lookup for per-container ops, default slot for global), and
        # re-implementing here would drift.  Tracked for cleanup in
        # terok-ai/terok-sandbox#111.
        config = _shield_build_config(container)
        shield = Shield(config)
        if needs_container:
            if not container:
                raise SystemExit(f"shield {handler.__name__}: container argument required")
            return handler(shield, container, **kwargs)
        return handler(shield, **kwargs)

    return wrapped


def _adapt_shield_command(cmd: Any) -> CommandDef:
    """Convert one shield CommandDef into sandbox's vocabulary.

    Prepends an implicit ``container`` positional arg when
    ``needs_container=True`` — shield's standalone CLI adds it the
    same way at parse time, so the user-facing argv is identical.
    """
    args = tuple(
        ArgDef(
            name=arg.name,
            help=arg.help,
            type=arg.type,
            default=arg.default,
            action=arg.action,
            dest=arg.dest,
            nargs=arg.nargs,
        )
        for arg in cmd.args
    )
    if cmd.needs_container:
        args = (ArgDef(name="container", help="Container name"), *args)
    return CommandDef(
        name=cmd.name,
        help=cmd.help,
        handler=_wrap_shield_handler(cmd.handler, needs_container=cmd.needs_container),
        args=args,
    )


def _imported_shield_children() -> tuple[CommandDef, ...]:
    """Filter shield's registry to verbs that make sense via sandbox.

    Excludes ``standalone_only`` (the registry's explicit
    "skip downstream" marker — ``setup`` / ``prepare`` / ``run`` /
    ``resolve`` carry custom CLI logic that doesn't lift cleanly into
    the integration surface) and handlerless entries (defensive — same
    set, but the check makes the filter intent unambiguous).
    """
    return tuple(
        _adapt_shield_command(cmd)
        for cmd in _SHIELD_REGISTRY
        if not cmd.standalone_only and cmd.handler is not None
    )


# Sandbox-specific install/uninstall verbs prepended to the imported set,
# so ``terok-sandbox shield install-hooks`` keeps working alongside the
# inherited ``shield status``, ``shield allow``, etc.
_SANDBOX_VERBS: tuple[CommandDef, ...] = (
    CommandDef(
        name="install-hooks",
        help="Install OCI hooks for the shield firewall",
        handler=_handle_shield_setup,
        args=(
            ArgDef(name="--root", action="store_true", help="Install system-wide (requires sudo)"),
            ArgDef(name="--user", action="store_true", help="Install to user hooks directory"),
        ),
    ),
    CommandDef(
        name="uninstall-hooks",
        help="Remove OCI hooks previously installed by install-hooks",
        handler=_handle_shield_uninstall,
        args=(
            ArgDef(name="--root", action="store_true", help="Remove system-wide (requires sudo)"),
            ArgDef(name="--user", action="store_true", help="Remove from user hooks directory"),
        ),
    ),
)


#: The shield command group exposed at sandbox's top level.  Composes
#: sandbox's own ``install-hooks`` / ``uninstall-hooks`` admin verbs
#: with every non-standalone-only entry from shield's own registry.
#: Adding a new shield verb (e.g. ``terok-shield`` grows a new
#: per-container action) flows into sandbox CLI zero-edit.
SHIELD_COMMANDS: tuple[CommandDef, ...] = (
    CommandDef(
        name="shield",
        help="Egress firewall management",
        children=_SANDBOX_VERBS + _imported_shield_children(),
    ),
)


__all__ = ["SHIELD_COMMANDS"]
