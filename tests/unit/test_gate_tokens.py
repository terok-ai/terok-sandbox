# SPDX-FileCopyrightText: 2026 Jiri Vyskocil
# SPDX-License-Identifier: Apache-2.0

"""Tests for the gate token minter."""

from __future__ import annotations

from terok_sandbox.gate.tokens import mint_gate_token


class TestMintGateToken:
    """``mint_gate_token`` returns a prefixed 128-bit hex token."""

    def test_returns_prefixed_hex(self) -> None:
        token = mint_gate_token()
        assert token.startswith("terok-g-")
        assert len(token) == 8 + 32  # prefix + 32 hex chars
        # The suffix is valid hexadecimal.
        int(token.removeprefix("terok-g-"), 16)

    def test_tokens_are_unique(self) -> None:
        assert mint_gate_token() != mint_gate_token()
