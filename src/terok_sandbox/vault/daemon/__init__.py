# SPDX-FileCopyrightText: 2026 Jiri Vyskocil
# SPDX-License-Identifier: Apache-2.0

"""Long-running vault daemon — lifecycle, protocol services, audit.

The runtime side of the vault: the process that exposes the store
over network protocols and the manager that controls its lifecycle.

Collaborators:

- [`lifecycle`][terok_sandbox.vault.daemon.lifecycle] — [`VaultManager`][terok_sandbox.vault.daemon.lifecycle.VaultManager]:
  start / stop / install systemd units / probe / report
  [`VaultStatus`][terok_sandbox.vault.daemon.lifecycle.VaultStatus].
- [`token_broker`][terok_sandbox.vault.daemon.token_broker] — aiohttp HTTP+WebSocket reverse proxy that swaps
  phantom tokens for real API credentials before forwarding upstream.
- [`audit`][terok_sandbox.vault.daemon.audit] — append-only JSONL audit log for every
  credential-bearing broker request.
- [`constants`][terok_sandbox.vault.daemon.constants] — shared marker strings (health path, phantom-creds
  marker, codex-shared-oauth marker).
"""
