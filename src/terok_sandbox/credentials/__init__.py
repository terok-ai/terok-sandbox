# SPDX-FileCopyrightText: 2026 Jiri Vyskocil
# SPDX-License-Identifier: Apache-2.0

"""Credential management — secret injection and SSH key provisioning.

Collaborators:

- :mod:`~.proxy` — standalone HTTP reverse proxy that injects real API
  credentials into container requests.  Includes an SSH agent protocol
  handler for transparent key signing.  Zero terok imports; runs as a
  separate process.
- :mod:`~.lifecycle` — host-side systemd socket activation and daemon
  fallback for the credential proxy.
- :mod:`~.db` — SQLite credential store (provider secrets, phantom token
  registry).
- :mod:`~.ssh` — SSH keypair generation, config rendering, and key
  registry (``ssh-keys.json``).
"""
