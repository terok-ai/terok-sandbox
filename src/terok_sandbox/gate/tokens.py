# SPDX-FileCopyrightText: 2026 Jiri Vyskocil
# SPDX-License-Identifier: Apache-2.0

"""Token minting for the per-container gate.

Each task gets a prefixed random 128-bit hex token.  The token travels
to the container only via the sidecar JSON the supervisor reads; there
is no on-disk token registry, and revocation is simply the supervisor's
death (the gate stops accepting requests when its process exits).

Token format: ``terok-g-<32 hex chars>`` (e.g. ``terok-g-a1b2c3…``).
"""

from __future__ import annotations

import secrets


def mint_gate_token() -> str:
    """Generate a fresh 128-bit hex gate token.

    Uses ``secrets.token_hex(16)`` for cryptographic randomness.  The
    supervisor validates this single token directly via
    [`_SingleTokenStore`][terok_sandbox.gate.server._SingleTokenStore],
    so there is nothing to persist.
    """
    return f"terok-g-{secrets.token_hex(16)}"
