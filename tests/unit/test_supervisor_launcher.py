# SPDX-FileCopyrightText: 2026 Jiri Vyskocil
# SPDX-License-Identifier: Apache-2.0

"""Tests for the supervisor's child spawn in
[`terok_sandbox.supervisor.launcher`][terok_sandbox.supervisor.launcher].

[`launch_child`][terok_sandbox.supervisor.launcher.launch_child] just
builds an argv and forks it; ``create_subprocess_exec`` is stubbed so the
test asserts the *shape* of the spawn — self-invocation on the parent
interpreter, the ``supervise-child`` verb — without forking a real child.
"""

from __future__ import annotations

import sys
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from terok_sandbox.supervisor.launcher import launch_child

_SIDECAR = Path("/state/sidecar/demo.json")


def _spy_exec() -> tuple[AsyncMock, list[list[str]]]:
    """An ``create_subprocess_exec`` stub recording each argv it is handed."""
    calls: list[list[str]] = []

    async def _fake(*argv: str, **_kw: object) -> MagicMock:
        calls.append(list(argv))
        return MagicMock(pid=999)

    return AsyncMock(side_effect=_fake), calls


class TestLaunchChild:
    """Spawns ``python -m terok_sandbox supervise-child <service> …`` on this interpreter."""

    @pytest.mark.asyncio
    async def test_launches_self_with_supervise_child_verb(self) -> None:
        spy, calls = _spy_exec()
        with patch("terok_sandbox.supervisor.launcher.asyncio.create_subprocess_exec", spy):
            handle = await launch_child("vault", "abc123", _SIDECAR)
        assert calls == [
            [
                sys.executable,
                "-P",
                "-m",
                "terok_sandbox",
                "supervise-child",
                "vault",
                "abc123",
                str(_SIDECAR),
            ]
        ]
        assert handle.service == "vault"
        assert handle.pid == 999
