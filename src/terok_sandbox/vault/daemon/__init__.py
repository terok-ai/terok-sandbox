# SPDX-FileCopyrightText: 2026 Jiri Vyskocil
# SPDX-License-Identifier: Apache-2.0

"""Per-container vault runtime — embeddable proxy, protocol services, audit.

The runtime side of the vault: the aiohttp app each per-container
supervisor mounts to swap phantom tokens for real credentials.  Every
container gets a fresh
[`VaultProxy`][terok_sandbox.vault.daemon.token_broker.VaultProxy] that
lives only as long as the container.

Collaborators:

- [`token_broker`][terok_sandbox.vault.daemon.token_broker] —
  [`VaultProxy`][terok_sandbox.vault.daemon.token_broker.VaultProxy]:
  the embeddable aiohttp HTTP+WebSocket reverse proxy that swaps
  phantom tokens for real API credentials before forwarding upstream.
- [`audit`][terok_sandbox.vault.daemon.audit] — append-only JSONL audit log for every
  credential-bearing broker request.

Three shared marker constants live here directly: the health-check path
the broker serves and two phantom-credential markers (Claude shared
``.credentials.json`` and Codex shared ``auth.json``).  Earlier
iterations had these in a sibling ``constants`` submodule; they're
small enough that the extra module just added tach noise.
"""

#: Static marker token written to ``.credentials.json`` in the shared Claude
#: config mount.  Claude Code reads subscription metadata from this file and
#: uses the ``accessToken`` for API calls.  The vault's token broker recognises
#: this marker and swaps it for the real OAuth credential — no per-task phantom
#: token needed.
#:
#: **Why this exists (workaround):**  Claude Code's JS binary determines the
#: subscription display ("Claude Max" vs "Claude API") based on where the
#: token *comes from*.  When ``CLAUDE_CODE_OAUTH_TOKEN`` env var is the
#: detected source, it always shows "Claude API" regardless of
#: ``.credentials.json`` content.  By not injecting that env var and
#: letting Claude Code read the token from the file instead, subscription
#: mode works correctly.  The token broker must then accept this static token.
PHANTOM_CREDENTIALS_MARKER = "terok-proxy-phantom-token:vault-handles-real-auth"

#: Static marker token written to Codex's shared ``auth.json``.  Codex's
#: built-in ChatGPT auth flow is file-based, so task containers all share one
#: synthetic auth store under ``~/.codex``.  The token broker recognises this
#: marker and swaps it for the real host-side OAuth credential.
CODEX_SHARED_OAUTH_MARKER = "terok-proxy-codex-oauth-marker:vault-handles-real-auth"

#: Unauthenticated health-check path served by the vault's token broker.
#: Used by the per-container vault proxy and in-container doctor checks.
HEALTH_PATH = "/-/health"
