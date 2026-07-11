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

This module — and therefore terok-shield — is imported only when the
``shield`` verb is actually dispatched: ``COMMANDS`` references it by a
lazy ``source`` string, so a plain ``import terok_sandbox`` or a
``terok-sandbox vault …`` run never pays for the shield stack.  As the
sole (import-linter-sanctioned) importer of ``terok_shield`` besides the
integrations adapter, keeping the imports at module top is fine here.
"""

from __future__ import annotations

import functools
from typing import TYPE_CHECKING, Any

from terok_shield import COMMANDS as _SHIELD_REGISTRY, Shield
from terok_shield.cli.main import _build_config as _shield_build_config  # noqa: PLC2701
from terok_shield.commands import (
    needs_container as _shield_needs_container,
    standalone_only as _shield_standalone_only,
)
from terok_util import LazyHandler

from ._types import ArgDef, CommandDef

if TYPE_CHECKING:
    from collections.abc import Callable


def _handle_shield_setup() -> None:
    """Install OCI hooks for the shield firewall.

    Sandbox-specific entry point — shield's own ``setup`` verb is
    ``standalone_only`` (its CLI argv shape doesn't lift cleanly into
    a generic registry handler).  This verb is the integration-
    friendly form.
    """
    from ..integrations.shield import ShieldHooks

    ShieldHooks.install()


def _handle_shield_uninstall() -> None:
    """Remove the OCI hooks previously installed by ``shield install-hooks``."""
    from ..integrations.shield import ShieldHooks

    ShieldHooks.uninstall()
    print("Shield hooks removed.")


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


def _resolve_shield_command(cmd: Any) -> Any:
    """Materialise a shield registry entry, importing its module if it is lazy.

    terok-shield's ``COMMANDS`` are themselves lazy references (only
    ``name``/``help`` plus a ``source``), so ``handler`` / ``args`` /
    ``standalone_only`` aren't populated until the entry is resolved.
    Sandbox needs the full definition to wrap the handler and mirror the
    args, so resolve here — this runs only when the ``shield`` verb is
    dispatched (this module is itself lazily loaded).
    """
    resolve = getattr(cmd, "resolve", None)
    return resolve() if callable(resolve) else cmd


def _adapt_shield_command(cmd: Any) -> CommandDef:
    """Convert one (resolved) shield CommandDef into sandbox's vocabulary.

    Shield's own verb defs already declare the ``container`` positional
    for the per-container verbs, so the args copy across as-is — sandbox
    only re-wraps the handler to bind a per-container ``Shield``.
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
    needs_container = _shield_needs_container(cmd)
    return CommandDef(
        name=cmd.name,
        help=cmd.help,
        handler=_wrap_shield_handler(cmd.handler, needs_container=needs_container),
        args=args,
    )


def _imported_shield_children() -> tuple[CommandDef, ...]:
    """Filter shield's registry to verbs that make sense via sandbox.

    Excludes ``standalone_only`` (the registry's explicit
    "skip downstream" marker — ``setup`` / ``prepare`` / ``run`` /
    ``resolve`` carry custom CLI logic that doesn't lift cleanly into
    the integration surface) and handlerless entries (defensive — same
    set, but the check makes the filter intent unambiguous).

    Each registry entry is resolved first — terok-shield's ``COMMANDS``
    are lazy references whose ``handler`` / ``standalone_only`` only
    materialise on resolution.
    """
    resolved = (_resolve_shield_command(cmd) for cmd in _SHIELD_REGISTRY)
    return tuple(
        _adapt_shield_command(cmd)
        for cmd in resolved
        if not _shield_standalone_only(cmd) and cmd.handler is not None
    )


# Sandbox-specific install/uninstall verbs prepended to the imported set,
# so ``terok-sandbox shield install-hooks`` keeps working alongside the
# inherited ``shield status``, ``shield allow``, etc.
_SANDBOX_VERBS: tuple[CommandDef, ...] = (
    CommandDef(
        name="install-hooks",
        help="Install OCI hooks for the shield firewall",
        handler=LazyHandler("terok_sandbox.commands.shield:_handle_shield_setup"),
    ),
    CommandDef(
        name="uninstall-hooks",
        help="Remove OCI hooks previously installed by install-hooks",
        handler=LazyHandler("terok_sandbox.commands.shield:_handle_shield_uninstall"),
    ),
)


#: The shield command group — sandbox's own admin verbs plus every
#: non-standalone-only entry from terok-shield's own registry.  Adding a
#: new shield verb flows into the sandbox CLI zero-edit.  Referenced by a
#: lazy ``source`` from [`commands.COMMANDS`][terok_sandbox.commands.COMMANDS],
#: so this whole module (and terok-shield) loads only for ``shield`` verbs.
SHIELD: CommandDef = CommandDef(
    name="shield",
    help="Egress firewall management",
    children=_SANDBOX_VERBS + _imported_shield_children(),
)

SHIELD_COMMANDS: tuple[CommandDef, ...] = (SHIELD,)


__all__ = ["SHIELD", "SHIELD_COMMANDS"]
