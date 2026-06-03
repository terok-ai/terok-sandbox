# SPDX-FileCopyrightText: 2026 Jiri Vyskocil
# SPDX-License-Identifier: Apache-2.0

"""Adapter for terok-clearance — clearance hub + verdict server + subscribers.

Re-export catalog: every ``from terok_clearance …`` import in
``terok_sandbox`` lives here.  The contract is enforced by
``.importlinter`` (``terok_clearance`` is a protected module with
``terok_sandbox.integrations.clearance`` as the sole allowed importer).

Headless installs typically skip the clearance phase entirely
(``terok_clearance`` not installed); callers handle ``ImportError`` on
this module's first attribute access by surfacing a stage-line
``skip`` line.  No symbol here raises on import — that would defeat
the soft-skip behaviour.

Clearance runs as a single in-process composition that the supervisor
builds at startup:
[`ClearanceHub`][terok_clearance.ClearanceHub] +
[`VerdictServer`][terok_clearance.VerdictServer] +
[`EventSubscriber`][terok_clearance.EventSubscriber] (driven by
[`create_notifier`][terok_clearance.create_notifier]).  TUI clients
multiplex across per-container hub sockets via
[`MultiSocketSubscriber`][terok_clearance.MultiSocketSubscriber].  The
supervisor composes these objects directly, so there is no per-service
systemd install.
"""

from terok_clearance import (  # noqa: F401 — re-exported public API
    ALL_NOTIFY_CATEGORIES,
    NOTIFY_BLOCKED,
    NOTIFY_VERDICT,
    ClearanceClient,
    ClearanceEvent,
    ClearanceHub,
    EventSubscriber,
    MultiSocketSubscriber,
    VerdictClient,
    VerdictServer,
    create_notifier,
    default_clearance_socket_path,
)
