# SPDX-FileCopyrightText: 2026 Jiri Vyskocil
# SPDX-License-Identifier: Apache-2.0

"""Credential management — credential DB and SSH key provisioning.

Collaborators:

- :mod:`~.db` — SQLite credential store (provider secrets, SSH keys, phantom
  token registry).
- :mod:`~.ssh` — :class:`~.ssh.SSHManager`, the per-scope key provisioning
  entry point.
- :mod:`~.ssh_keypair` — generation, import, and export of OpenSSH keypairs
  against the credential DB.
"""
