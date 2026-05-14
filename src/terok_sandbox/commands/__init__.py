# SPDX-FileCopyrightText: 2026 Jiri Vyskocil
# SPDX-License-Identifier: Apache-2.0

"""Command registry for terok-sandbox — one module per subsystem.

Follows the same [`CommandDef`][terok_sandbox.commands.CommandDef] /
[`ArgDef`][terok_sandbox.commands.ArgDef] pattern as
``terok_shield.registry``.  Higher-level consumers (terok,
terok-executor) import ``COMMANDS`` to build their own CLI frontends
without duplicating argument definitions or handler logic.

Per-subsystem modules:

- [`sandbox`][terok_sandbox.commands.sandbox] — setup/uninstall
  (composes shield + vault + gate + clearance into one verb).
- [`gate`][terok_sandbox.commands.gate] — gate server lifecycle.
- [`shield`][terok_sandbox.commands.shield] — egress-firewall hooks.
- [`vault`][terok_sandbox.commands.vault] — vault daemon + the
  unlock/lock/seal trio that drives the SQLCipher passphrase chain.
- [`ssh`][terok_sandbox.commands.ssh] — SSH-key CRUD against the
  credentials DB.
- [`doctor`][terok_sandbox.commands.doctor] — host-side health checks.
- [`credentials`][terok_sandbox.commands.credentials] — credentials-DB
  encryption chooser, provisioning, and migration phase.
- [`launch`][terok_sandbox.commands.launch] — prepare/run/cleanup for
  user-owned containers.

Shield commands are delegated to terok-shield's own registry —
``SHIELD_COMMANDS`` re-exports the non-standalone subset.
"""

from __future__ import annotations

from ._types import ArgDef, CommandDef, CommandTree, KeyRow
from .credentials import (
    CREDENTIALS_COMMANDS,
    _ask_passphrase_mode,
    _back_up_plaintext_db,
    _handle_credentials_encrypt_db,
    _persist_mode_choice,
    _provision_passphrase,
    _run_credentials_setup_phase,
)
from .doctor import DOCTOR_COMMANDS, _handle_doctor
from .gate import (
    GATE_COMMANDS,
    _handle_gate_install,
    _handle_gate_start,
    _handle_gate_status,
    _handle_gate_stop,
    _handle_gate_uninstall,
)
from .launch import LAUNCH_COMMANDS, _handle_cleanup, _handle_prepare, _handle_run
from .sandbox import SETUP_COMMANDS, _handle_sandbox_setup, _handle_sandbox_uninstall
from .shield import (
    SHIELD_COMMANDS,
    _handle_shield_setup,
    _handle_shield_uninstall,
)
from .ssh import (
    SSH_COMMANDS,
    _build_key_rows,
    _filter_key_rows,
    _handle_ssh_add,
    _handle_ssh_export,
    _handle_ssh_import,
    _handle_ssh_link,
    _handle_ssh_list,
    _handle_ssh_pub,
    _handle_ssh_remove,
    _handle_ssh_rename,
    _key_id_from_row,
    _open_db,
    _print_key_table,
    _validate_scope_name,
)
from .vault import (
    VAULT_COMMANDS,
    _forget_config_tier_updates,
    _handle_vault_destroy_passphrase,
    _handle_vault_install,
    _handle_vault_lock,
    _handle_vault_start,
    _handle_vault_status,
    _handle_vault_stop,
    _handle_vault_uninstall,
    _handle_vault_unlock,
    _print_plaintext_passphrase_warning,
    handle_vault_seal,
    handle_vault_to_keyring,
)

#: Sandbox's top-level command forest — a [`CommandTree`][terok_sandbox.commands.CommandTree]
#: of every verb the package exposes.  Each per-subsystem ``*_COMMANDS``
#: tuple contributes one or more root verbs; subsystem groups (gate,
#: shield, vault, ssh, credentials) each contribute exactly one root
#: holding their subverbs as ``children`` so the structural nesting
#: matches the operator-facing CLI surface.  Out-of-tree consumers
#: (terok, terok-executor) walk this tree via
#: [`CommandTree.overlay`][terok_sandbox.commands.CommandTree.overlay]
#: and [`CommandTree.wire`][terok_sandbox.commands.CommandTree.wire].
COMMANDS: CommandTree = CommandTree(
    SETUP_COMMANDS
    + GATE_COMMANDS
    + SHIELD_COMMANDS
    + VAULT_COMMANDS
    + SSH_COMMANDS
    + CREDENTIALS_COMMANDS
    + LAUNCH_COMMANDS
    + DOCTOR_COMMANDS
)


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
    "VAULT_COMMANDS",
    # Handlers re-exported for testing and out-of-tree consumers (terok
    # frontends sometimes call them directly).  The underscore prefix
    # marks them as "registry-only" — the registry is the public entry
    # point, not the handler functions themselves.
    "_ask_passphrase_mode",
    "_back_up_plaintext_db",
    "_build_key_rows",
    "_filter_key_rows",
    "_forget_config_tier_updates",
    "_handle_cleanup",
    "_handle_credentials_encrypt_db",
    "_handle_doctor",
    "_handle_gate_install",
    "_handle_gate_start",
    "_handle_gate_status",
    "_handle_gate_stop",
    "_handle_gate_uninstall",
    "_handle_prepare",
    "_handle_run",
    "_handle_sandbox_setup",
    "_handle_sandbox_uninstall",
    "_handle_shield_setup",
    "_handle_shield_uninstall",
    "_handle_ssh_add",
    "_handle_ssh_export",
    "_handle_ssh_import",
    "_handle_ssh_link",
    "_handle_ssh_list",
    "_handle_ssh_pub",
    "_handle_ssh_remove",
    "_handle_ssh_rename",
    "_handle_vault_destroy_passphrase",
    "_handle_vault_install",
    "_handle_vault_lock",
    "handle_vault_seal",
    "_handle_vault_start",
    "_handle_vault_status",
    "_handle_vault_stop",
    "handle_vault_to_keyring",
    "_handle_vault_uninstall",
    "_handle_vault_unlock",
    "_key_id_from_row",
    "_open_db",
    "_persist_mode_choice",
    "_print_key_table",
    "_print_plaintext_passphrase_warning",
    "_provision_passphrase",
    "_run_credentials_setup_phase",
    "_validate_scope_name",
]
