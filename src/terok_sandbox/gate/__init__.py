# SPDX-FileCopyrightText: 2026 Jiri Vyskocil
# SPDX-License-Identifier: Apache-2.0

"""Code access gate — authenticated git serving and mirror management.

Collaborators:

- :mod:`~.server` — standalone HTTP server wrapping ``git http-backend``
  with per-task token auth.  Zero terok imports; runs as a separate process.
- :mod:`~.lifecycle` — host-side systemd socket activation and daemon
  fallback for the gate server.
- :mod:`~.tokens` — per-task token CRUD (create, revoke, file I/O).
- :mod:`~.mirror` — host-side bare git mirror (clone, sync, staleness
  detection vs upstream).
"""
