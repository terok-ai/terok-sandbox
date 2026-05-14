# SPDX-FileCopyrightText: 2026 Jiri Vyskocil
# SPDX-License-Identifier: Apache-2.0

"""Vault — unified credential service: store, SSH, daemon.

The vault protects API credentials and SSH keys behind phantom tokens.
Containers never see real secrets; they present phantom tokens that
the daemon validates against the at-rest store, injects real
credentials, and forwards requests upstream.

Three sub-packages under one namespace:

- [`store`][terok_sandbox.vault.store] — the at-rest SQLCipher database and the six-tier
  passphrase resolution chain that unlocks it.
- [`ssh`][terok_sandbox.vault.ssh] — keypair I/O, scope provisioning, and the SSH-agent
  protocol handler.
- [`daemon`][terok_sandbox.vault.daemon] — the long-running process: lifecycle, the HTTP
  token broker, audit logging.
"""
