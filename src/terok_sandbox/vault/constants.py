# SPDX-FileCopyrightText: 2026 Jiri Vyskocil
# SPDX-License-Identifier: Apache-2.0

"""Shared constants for the vault subsystem."""

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

#: Unauthenticated health-check path served by the vault's token broker.
#: Used by the server, lifecycle probes, and sickbay doctor checks.
HEALTH_PATH = "/-/health"
