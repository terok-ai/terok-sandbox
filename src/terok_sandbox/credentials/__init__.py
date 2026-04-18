# SPDX-FileCopyrightText: 2026 Jiri Vyskocil
# SPDX-License-Identifier: Apache-2.0

"""Credential management — credential DB and SSH key provisioning.

Collaborators:

- :mod:`~.db` — SQLite credential store (provider secrets, phantom token
  registry).
- :mod:`~.ssh` — SSH keypair generation, config rendering, and key
  registry (``ssh-keys.json``).
"""
