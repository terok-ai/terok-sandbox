# SPDX-FileCopyrightText: 2026 Jiri Vyskocil
# SPDX-License-Identifier: Apache-2.0

"""Code access gate — authenticated git serving and mirror management.

Collaborators:

- [`server`][terok_sandbox.gate.server] — the [`GateServer`][terok_sandbox.gate.server.GateServer]
  component wrapping ``git http-backend`` with single-token auth.  Composed
  by the per-container supervisor; zero terok imports beyond the SELinux
  socket-labelling helper.
- [`tokens`][terok_sandbox.gate.tokens] — [`mint_gate_token`][terok_sandbox.gate.tokens.mint_gate_token],
  the per-task token minter.
- [`mirror`][terok_sandbox.gate.mirror] — host-side bare git mirror (clone, sync, staleness
  detection vs upstream).
"""
