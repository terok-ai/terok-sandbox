# SPDX-FileCopyrightText: 2026 Jiri Vyskocil
# SPDX-License-Identifier: Apache-2.0

"""Adapter for terok-clearance — desktop notifier + verdict hub unit installers.

Re-export catalog: every ``from terok_clearance …`` import in
``terok_sandbox`` lives here.  The contract is enforced by
``.importlinter`` (``terok_clearance`` is a protected module with
``terok_sandbox.integrations.clearance`` as the sole allowed importer).

Headless installs typically skip the clearance phase entirely
(``terok_clearance`` not installed); callers handle ``ImportError`` on
this module's first attribute access by surfacing a stage-line
``skip`` line.  No symbol here raises on import — that would defeat
the soft-skip behaviour.
"""

from terok_clearance.runtime.installer import (  # noqa: F401 — re-exported public API
    HUB_UNIT_NAME,
    NOTIFIER_UNIT_NAME,
    VERDICT_UNIT_NAME,
    install_notifier_service,
    install_service,
    uninstall_notifier_service,
    uninstall_service,
)
