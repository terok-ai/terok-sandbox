# SPDX-FileCopyrightText: 2026 Jiri Vyskocil
# SPDX-License-Identifier: Apache-2.0

"""Vault — unified credential service with token-broker and ssh-signer faces.

The vault daemon protects API credentials and SSH keys behind phantom tokens.
Containers never see real secrets; they present phantom tokens that the vault
validates against a SQLite database, injects real credentials, and forwards
requests upstream.

Two protocol faces:

- [`token_broker`][terok_sandbox.vault.token_broker] — HTTP reverse proxy that swaps phantom tokens for
  real API credentials (Anthropic, Mistral, GitHub, etc.).
- [`ssh_signer`][terok_sandbox.vault.ssh_signer] — SSH agent protocol handler that signs git data
  with host-side private keys.

Both faces run in a single daemon process managed by [`lifecycle`][terok_sandbox.vault.lifecycle].
"""
