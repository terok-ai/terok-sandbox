# SPDX-FileCopyrightText: 2026 Jiri Vyskocil
# SPDX-License-Identifier: Apache-2.0

"""Credential management — credential DB and SSH key provisioning.

Collaborators:

- [`db`][terok_sandbox.credentials.db] — SQLite credential store (provider secrets, SSH keys, phantom
  token registry).
- [`ssh`][terok_sandbox.credentials.ssh] — [`SSHManager`][terok_sandbox.credentials.ssh.SSHManager], the per-scope key provisioning
  entry point.
- [`ssh_keypair`][terok_sandbox.credentials.ssh_keypair] — generation, import, and export of OpenSSH keypairs
  against the credential DB.
"""
