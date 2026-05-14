# SPDX-FileCopyrightText: 2026 Jiri Vyskocil
# SPDX-License-Identifier: Apache-2.0

"""SSH key handling — keypair I/O, scope management, and the agent protocol.

Consolidates the SSH story end to end: keypair generation and
OpenSSH-file parsing on one side, DB-backed scope provisioning in the
middle, and the host-side SSH-agent protocol on the other.

Collaborators:

- [`keypair`][terok_sandbox.vault.ssh.keypair] — pure-bytes primitives: generate / import / export
  OpenSSH keypairs, fingerprint computation, PEM encoding.
- [`manager`][terok_sandbox.vault.ssh.manager] — [`SSHManager`][terok_sandbox.vault.ssh.manager.SSHManager],
  the per-scope key-provisioning façade over [`store.db`][terok_sandbox.vault.store.db]
  and [`keypair`][terok_sandbox.vault.ssh.keypair].
- [`signer`][terok_sandbox.vault.ssh.signer] — SSH-agent protocol handler that signs git data
  using vault-stored private keys.
- [`scope_sockets`][terok_sandbox.vault.ssh.scope_sockets] — per-scope UID-gated Unix sockets that gate-sync
  consumes as ``SSH_AUTH_SOCK``.
"""
