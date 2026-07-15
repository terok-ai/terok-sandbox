# SPDX-FileCopyrightText: 2026 Jiri Vyskocil
# SPDX-License-Identifier: Apache-2.0

"""Command registry for terok-sandbox — one module per subsystem.

Follows the same [`CommandDef`][terok_util.cli_types.CommandDef] /
[`ArgDef`][terok_util.cli_types.ArgDef] pattern as
``terok_shield.registry``.  Higher-level consumers (terok,
terok-executor) import ``COMMANDS`` to build their own CLI frontends
without duplicating argument definitions or handler logic.

Per-subsystem modules:

- [`sandbox`][terok_sandbox.commands.sandbox] — setup/uninstall
  (composes shield + vault + gate + clearance into one verb).
- [`gate`][terok_sandbox.commands.gate] — gate server lifecycle.
- [`shield`][terok_sandbox.commands.shield] — egress-firewall hooks.
- [`vault`][terok_sandbox.commands.vault] — vault passphrase verbs
  (the unlock/lock/seal trio that drives the SQLCipher chain).
- [`ssh`][terok_sandbox.commands.ssh] — SSH-key CRUD against the
  credentials DB.
- [`doctor`][terok_sandbox.commands.doctor] — host-side health checks.
- [`credentials`][terok_sandbox.commands.credentials] — credentials-DB
  encryption chooser, provisioning, and migration phase.
- [`launch`][terok_sandbox.commands.launch] — prepare/run/cleanup for
  user-owned containers.

``COMMANDS`` is a forest of **lazy roots**: each top-level verb is a
[`CommandDef`][terok_util.cli_types.CommandDef] carrying only its
``name``/``help`` plus a ``source`` string pointing at the fully
populated verb definition in its module.  Building ``COMMANDS`` imports
none of the per-subsystem modules; ``CommandTree.wire(parser, argv=…)``
resolves and imports **only the invoked verb's module** (and a bare
``--help`` lists every verb without importing any).  This is what keeps
``import terok_sandbox`` and the per-container supervisor spawn off the
config / SQLCipher / cryptography / terok-shield stacks.

The ``*_COMMANDS`` registries and the handler functions the executor
splice and tests reach for are re-exported **lazily** through
[`__getattr__`][terok_sandbox.commands.__getattr__], so importing this
package to read ``COMMANDS`` never pulls a handler module behind it.
"""

from __future__ import annotations

from importlib import import_module
from typing import TYPE_CHECKING, Any

from ._types import ArgDef, CommandDef, CommandTree, KeyRow

#: Sandbox's top-level command forest — one **lazy** root per verb.
#: Each root defers to a ``"module:qualname"`` ``source`` resolved by
#: [`CommandTree.wire`][terok_util.cli_types.CommandTree.wire] only when
#: that verb is the one actually invoked.  Order is the operator-facing
#: ``--help`` order.  Out-of-tree consumers (terok, terok-executor) walk
#: this tree via [`CommandTree.overlay`][terok_util.cli_types.CommandTree.overlay]
#: / [`extend_at`][terok_util.cli_types.CommandTree.extend_at]; a lazy
#: root exposes its full subtree via
#: [`CommandDef.resolve`][terok_util.cli_types.CommandDef.resolve].
COMMANDS: CommandTree = CommandTree(
    [
        CommandDef(
            name="setup",
            help="Install supervisor hooks + shield hooks in one step",
            source="terok_sandbox.commands.sandbox:SETUP",
        ),
        CommandDef(
            name="uninstall",
            help="Remove supervisor hooks + shield hooks in one step",
            source="terok_sandbox.commands.sandbox:UNINSTALL",
        ),
        CommandDef(
            name="gate",
            help="Git gate inspection",
            source="terok_sandbox.commands.gate:GATE",
        ),
        CommandDef(
            name="shield",
            help="Egress firewall management",
            source="terok_sandbox.commands.shield:SHIELD",
        ),
        CommandDef(
            name="vault",
            help="Vault passphrase management",
            source="terok_sandbox.commands.vault:VAULT",
        ),
        CommandDef(
            name="ssh",
            help="SSH keypair management",
            source="terok_sandbox.commands.ssh:SSH",
        ),
        CommandDef(
            name="credentials",
            help="Credentials DB management",
            source="terok_sandbox.commands.credentials:CREDENTIALS",
        ),
        CommandDef(
            name="prepare",
            help="Print podman flags for sandboxing a user-owned container",
            source="terok_sandbox.commands.launch:PREPARE",
        ),
        CommandDef(
            name="run",
            help="Launch a sandboxed user-owned container (exec into podman run)",
            source="terok_sandbox.commands.launch:RUN",
        ),
        CommandDef(
            name="cleanup",
            help="Revoke tokens and drop shield rules for a sandboxed container",
            source="terok_sandbox.commands.launch:CLEANUP",
        ),
        CommandDef(
            name="doctor",
            help="Run sandbox health checks",
            source="terok_sandbox.commands.doctor:DOCTOR",
        ),
        CommandDef(
            name="supervisor",
            help="Run the per-container supervisor (internal; spawned by the OCI hook)",
            source="terok_sandbox.commands.supervisor:SUPERVISOR",
        ),
        CommandDef(
            name="supervise-child",
            help="Run one hardened supervisor service (internal; spawned by the supervisor)",
            source="terok_sandbox.commands.supervisor:SUPERVISE_CHILD",
        ),
    ]
)


#: Lazily re-exported name → ``"submodule:attr"`` source.  The
#: ``*_COMMANDS`` registries and the underscore handlers the executor
#: splice / tests import stay reachable, but resolving one imports its
#: owning module only on first access — so ``from terok_sandbox.commands
#: import COMMANDS`` (and the supervisor spawn behind it) pulls no
#: handler module.
_LAZY: dict[str, str] = {
    # credentials
    "CREDENTIALS_COMMANDS": "credentials:CREDENTIALS_COMMANDS",
    "ProvisioningPlan": "credentials:ProvisioningPlan",
    "TierProvisionResult": "credentials:TierProvisionResult",
    "_ask_passphrase_mode": "credentials:_ask_passphrase_mode",
    "_back_up_plaintext_db": "credentials:_back_up_plaintext_db",
    "_handle_credentials_encrypt_db": "credentials:_handle_credentials_encrypt_db",
    "_persist_mode_choice": "credentials:_persist_mode_choice",
    "_provision_passphrase": "credentials:_provision_passphrase",
    "_run_credentials_setup_phase": "credentials:_run_credentials_setup_phase",
    "credentials_provisioned": "credentials:credentials_provisioned",
    "plan_provisioning": "credentials:plan_provisioning",
    "provision_passphrase_tier": "credentials:provision_passphrase_tier",  # nosec: B105 — export-map import paths, never secrets
    # doctor
    "DOCTOR_COMMANDS": "doctor:DOCTOR_COMMANDS",
    "_handle_doctor": "doctor:_handle_doctor",
    # gate
    "GATE_COMMANDS": "gate:GATE_COMMANDS",
    "_handle_gate_path": "gate:_handle_gate_path",
    # launch
    "LAUNCH_COMMANDS": "launch:LAUNCH_COMMANDS",
    "_handle_cleanup": "launch:_handle_cleanup",
    "_handle_prepare": "launch:_handle_prepare",
    "_handle_run": "launch:_handle_run",
    # sandbox
    "SETUP_COMMANDS": "sandbox:SETUP_COMMANDS",
    "_handle_sandbox_setup": "sandbox:_handle_sandbox_setup",
    "_handle_sandbox_uninstall": "sandbox:_handle_sandbox_uninstall",
    # shield
    "SHIELD_COMMANDS": "shield:SHIELD_COMMANDS",
    "_handle_shield_setup": "shield:_handle_shield_setup",
    "_handle_shield_uninstall": "shield:_handle_shield_uninstall",
    # ssh
    "SSH_COMMANDS": "ssh:SSH_COMMANDS",
    "_build_key_rows": "ssh:_build_key_rows",
    "_filter_key_rows": "ssh:_filter_key_rows",
    "_handle_ssh_add": "ssh:_handle_ssh_add",
    "_handle_ssh_export": "ssh:_handle_ssh_export",
    "_handle_ssh_import": "ssh:_handle_ssh_import",
    "_handle_ssh_link": "ssh:_handle_ssh_link",
    "_handle_ssh_list": "ssh:_handle_ssh_list",
    "_handle_ssh_pub": "ssh:_handle_ssh_pub",
    "_handle_ssh_remove": "ssh:_handle_ssh_remove",
    "_handle_ssh_rename": "ssh:_handle_ssh_rename",
    "_key_id_from_row": "ssh:_key_id_from_row",
    "_open_db": "ssh:_open_db",
    "_print_key_table": "ssh:_print_key_table",
    "_validate_scope_name": "ssh:_validate_scope_name",
    # supervisor
    "SUPERVISOR_COMMANDS": "supervisor:SUPERVISOR_COMMANDS",
    "SUPERVISE_CHILD": "supervisor:SUPERVISE_CHILD",
    "_handle_supervisor": "supervisor:_handle_supervisor",
    "_handle_supervise_child": "supervisor:_handle_supervise_child",
    # vault
    "VAULT_COMMANDS": "vault:VAULT_COMMANDS",
    "PassphraseChangeResult": "vault:PassphraseChangeResult",  # nosec: B105 — export-map import paths, never secrets
    "SessionProvisionResult": "vault:SessionProvisionResult",
    "TierRewrite": "vault:TierRewrite",
    "_forget_config_tier_updates": "vault:_forget_config_tier_updates",
    "_handle_vault_list": "vault:_handle_vault_list",
    "_handle_vault_lock": "vault:_handle_vault_lock",
    "_handle_vault_unlock": "vault:_handle_vault_unlock",
    "change_passphrase": "vault:change_passphrase",  # nosec: B105 — export-map import paths, never secrets
    "handle_vault_seal": "vault:handle_vault_seal",
    "handle_vault_to_keyring": "vault:handle_vault_to_keyring",
    "provision_session_passphrase": "vault:provision_session_passphrase",
    "purge_passphrase_tiers": "vault:purge_passphrase_tiers",
}


def __getattr__(name: str) -> Any:
    """Resolve a re-exported registry / handler by importing its module on first access.

    Looked up in the module's ``_LAZY`` map and cached on the
    package, so ``COMMANDS`` can be built (and the supervisor spawned)
    without importing any handler module, while
    ``from terok_sandbox.commands import _handle_sandbox_setup`` still
    works for the executor splice.
    """
    target = _LAZY.get(name)
    if target is None:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
    module_name, _, attr = target.partition(":")
    value = getattr(import_module(f".{module_name}", __name__), attr)
    globals()[name] = value  # cache — the next access skips __getattr__
    return value


def __dir__() -> list[str]:
    """List eager globals plus every lazily re-exported name."""
    return sorted({*globals(), *_LAZY})


if TYPE_CHECKING:
    # Eager view of the lazily re-exported surface for type checkers / IDEs.
    from .credentials import CREDENTIALS_COMMANDS, ProvisioningPlan, plan_provisioning
    from .doctor import DOCTOR_COMMANDS
    from .gate import GATE_COMMANDS
    from .launch import LAUNCH_COMMANDS
    from .sandbox import SETUP_COMMANDS
    from .shield import SHIELD_COMMANDS
    from .ssh import SSH_COMMANDS
    from .supervisor import SUPERVISOR_COMMANDS
    from .vault import (
        VAULT_COMMANDS,
        PassphraseChangeResult,
        SessionProvisionResult,
        TierRewrite,
        change_passphrase,
        handle_vault_seal,
        handle_vault_to_keyring,
        provision_session_passphrase,
        purge_passphrase_tiers,
    )


#: The registry's stable surface.  Underscore-prefixed handlers stay
#: resolvable via [`__getattr__`][terok_sandbox.commands.__getattr__] for
#: the executor's CLI splice and the tests that mock them, but they are
#: deliberately absent from ``__all__``: the ``*_COMMANDS`` registries
#: (and ``COMMANDS``) are the public entry point, not the handlers.
__all__ = [
    # Vocabulary
    "ArgDef",
    "CommandDef",
    "CommandTree",
    "KeyRow",
    # Aggregated registries
    "COMMANDS",
    "CREDENTIALS_COMMANDS",
    "DOCTOR_COMMANDS",
    "GATE_COMMANDS",
    "LAUNCH_COMMANDS",
    "SETUP_COMMANDS",
    "SHIELD_COMMANDS",
    "SSH_COMMANDS",
    "SUPERVISOR_COMMANDS",
    "VAULT_COMMANDS",
    # Vault passphrase workflows (public, non-underscore surface)
    "PassphraseChangeResult",
    "ProvisioningPlan",
    "SessionProvisionResult",
    "TierRewrite",
    "change_passphrase",
    "handle_vault_seal",
    "handle_vault_to_keyring",
    "plan_provisioning",
    "provision_session_passphrase",
    "purge_passphrase_tiers",
]
