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

Shield commands are delegated to terok-shield's own registry —
``SHIELD_COMMANDS`` re-exports the non-standalone subset.
"""

from __future__ import annotations

# Underscore-prefixed handlers are re-exported (not advertised in
# ``__all__``) for terok-executor's CLI splice and the tests that mock
# them; the F401 suppressions below mark those intentional re-exports.
from ._types import ArgDef, CommandDef, CommandTree, KeyRow
from .credentials import (  # noqa: F401
    CREDENTIALS_COMMANDS,
    _ask_passphrase_mode,
    _back_up_plaintext_db,
    _handle_credentials_encrypt_db,
    _persist_mode_choice,
    _provision_passphrase,
    _run_credentials_setup_phase,
)
from .doctor import DOCTOR_COMMANDS, _handle_doctor  # noqa: F401
from .gate import GATE_COMMANDS, _handle_gate_path  # noqa: F401
from .launch import (  # noqa: F401
    LAUNCH_COMMANDS,
    _handle_cleanup,
    _handle_prepare,
    _handle_run,
)
from .sandbox import (  # noqa: F401
    SETUP_COMMANDS,
    _handle_sandbox_setup,
    _handle_sandbox_uninstall,
)
from .shield import (  # noqa: F401
    SHIELD_COMMANDS,
    _handle_shield_setup,
    _handle_shield_uninstall,
)
from .ssh import (  # noqa: F401
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
from .supervisor import SUPERVISOR_COMMANDS, _handle_supervisor  # noqa: F401
from .vault import (  # noqa: F401
    VAULT_COMMANDS,
    SessionProvisionResult,
    SessionShadow,
    _forget_config_tier_updates,
    _handle_vault_list,
    _handle_vault_lock,
    _handle_vault_unlock,
    clear_redundant_session_file,
    handle_vault_seal,
    handle_vault_to_keyring,
    provision_session_passphrase,
    purge_passphrase_tiers,
    session_shadow_state,
)

#: Sandbox's top-level command forest — a [`CommandTree`][terok_util.cli_types.CommandTree]
#: of every verb the package exposes.  Each per-subsystem ``*_COMMANDS``
#: tuple contributes one or more root verbs; subsystem groups (gate,
#: shield, vault, ssh, credentials) each contribute exactly one root
#: holding their subverbs as ``children`` so the structural nesting
#: matches the operator-facing CLI surface.  Out-of-tree consumers
#: (terok, terok-executor) walk this tree via
#: [`CommandTree.overlay`][terok_util.cli_types.CommandTree.overlay]
#: and [`CommandTree.wire`][terok_util.cli_types.CommandTree.wire].
COMMANDS: CommandTree = CommandTree(
    SETUP_COMMANDS
    + GATE_COMMANDS
    + SHIELD_COMMANDS
    + VAULT_COMMANDS
    + SSH_COMMANDS
    + CREDENTIALS_COMMANDS
    + LAUNCH_COMMANDS
    + DOCTOR_COMMANDS
    + SUPERVISOR_COMMANDS
)


#: The registry's stable surface.  Underscore-prefixed handlers
#: (``_handle_sandbox_setup``, ``_open_db``, …) stay importable for the
#: executor's CLI splice and the tests that mock them, but they are
#: deliberately absent from ``__all__``: the ``*_COMMANDS`` registries
#: are the public entry point, not the handler functions themselves.
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
    "SessionProvisionResult",
    "SessionShadow",
    "clear_redundant_session_file",
    "handle_vault_seal",
    "handle_vault_to_keyring",
    "provision_session_passphrase",
    "purge_passphrase_tiers",
    "session_shadow_state",
]
