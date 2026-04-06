# SPDX-FileCopyrightText: 2026 Jiri Vyskocil
# SPDX-License-Identifier: Apache-2.0

"""Shared constants for the credential proxy subsystem."""

#: Static marker token written to ``.credentials.json`` in the shared Claude
#: config mount.  Claude Code reads subscription metadata from this file and
#: uses the ``accessToken`` for API calls.  The proxy recognises this marker
#: and swaps it for the real OAuth credential — no per-task phantom token
#: needed.
#:
#: **Why this exists (workaround):**  Claude Code's JS binary determines the
#: subscription display ("Claude Max" vs "Claude API") based on where the
#: token *comes from*.  When ``CLAUDE_CODE_OAUTH_TOKEN`` env var is the
#: detected source, it always shows "Claude API" regardless of
#: ``.credentials.json`` content.  By not injecting that env var and
#: letting Claude Code read the token from the file instead, subscription
#: mode works correctly.  The proxy must then accept this static file token.
PHANTOM_CREDENTIALS_MARKER = "terok-proxy-phantom-token:credential-proxy-handles-real-auth"
