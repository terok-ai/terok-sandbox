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

The inherited children are materialised **lazily** — see
[`_LazyShieldChildren`][terok_sandbox.commands.shield._LazyShieldChildren].
Building the sandbox command forest (and the per-container supervisor
spawn that starts by importing it) must not import terok-shield; the
registry import waits until the shield subtree is actually wired or
walked.  Every ``from terok_shield …`` therefore lives inside a function
body, keeping this module the sole (import-linter-sanctioned) importer
without paying the cost at import time.
"""

from __future__ import annotations

import functools
from typing import TYPE_CHECKING, Any, cast

from terok_util import LazyHandler

from ._types import ArgDef, CommandDef

if TYPE_CHECKING:
    from collections.abc import Callable, Iterator


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
        from terok_shield import Shield
        from terok_shield.cli.main import _build_config as _shield_build_config  # noqa: PLC2701

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
    from terok_shield.commands import needs_container as _shield_needs_container

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
    if needs_container:
        args = (ArgDef(name="container", help="Container name"), *args)
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
    """
    from terok_shield import COMMANDS as _SHIELD_REGISTRY
    from terok_shield.commands import standalone_only as _shield_standalone_only

    return tuple(
        _adapt_shield_command(cmd)
        for cmd in _SHIELD_REGISTRY
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


class _LazyShieldChildren:
    """Shield subverbs, materialised from terok-shield's registry on first traversal.

    Building the sandbox command forest must not import terok-shield —
    that keeps ``import terok_sandbox`` and the per-container supervisor
    spawn off pydantic and the shield stack.  The inherited shield verbs
    are only needed when the tree is actually wired, walked, or overlaid,
    so this stand-in defers the registry import until it is first
    iterated or indexed.  It quacks like the ``tuple[CommandDef, ...]``
    the [`CommandDef.children`][terok_util.cli_types.CommandDef] field
    expects (hence the [`cast`][typing.cast] at the wiring site).
    """

    __slots__ = ("_resolved",)

    def __init__(self) -> None:
        """Start unresolved — no terok-shield import yet."""
        self._resolved: tuple[CommandDef, ...] | None = None

    def _materialise(self) -> tuple[CommandDef, ...]:
        """Build (once) and return the static verbs plus the inherited shield subtree."""
        if self._resolved is None:
            self._resolved = _SANDBOX_VERBS + _imported_shield_children()
        return self._resolved

    def __iter__(self) -> Iterator[CommandDef]:
        """Yield each child, resolving the registry on first call."""
        return iter(self._materialise())

    def __len__(self) -> int:
        """Number of shield subverbs (resolves the registry)."""
        return len(self._materialise())

    def __getitem__(self, index: int) -> CommandDef:
        """Return the child at *index* (resolves the registry)."""
        return self._materialise()[index]

    def __add__(self, other: tuple[CommandDef, ...]) -> tuple[CommandDef, ...]:
        """Concatenate the resolved children with *other* (used by ``extend_at``)."""
        return self._materialise() + tuple(other)

    def __radd__(self, other: tuple[CommandDef, ...]) -> tuple[CommandDef, ...]:
        """Concatenate *other* with the resolved children."""
        return tuple(other) + self._materialise()

    def __bool__(self) -> bool:
        """Always truthy — the static install/uninstall verbs are always present.

        Answering without materialising lets ``wire``'s ``if
        cmd.children:`` guard stay off terok-shield until it actually
        descends into the group.
        """
        return True


#: The shield command group exposed at sandbox's top level.  Composes
#: sandbox's own ``install-hooks`` / ``uninstall-hooks`` admin verbs
#: with every non-standalone-only entry from shield's own registry.
#: Adding a new shield verb (e.g. ``terok-shield`` grows a new
#: per-container action) flows into sandbox CLI zero-edit.  The children
#: are the lazy stand-in so importing this module — and building
#: ``COMMANDS`` — never pulls in terok-shield.
SHIELD_COMMANDS: tuple[CommandDef, ...] = (
    CommandDef(
        name="shield",
        help="Egress firewall management",
        children=cast("tuple[CommandDef, ...]", _LazyShieldChildren()),
    ),
)


__all__ = ["SHIELD_COMMANDS"]
