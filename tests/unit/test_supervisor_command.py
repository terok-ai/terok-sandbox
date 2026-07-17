# SPDX-FileCopyrightText: 2026 Jiri Vyskocil
# SPDX-License-Identifier: Apache-2.0

"""Tests for the hidden ``supervisor`` CLI verb dispatcher.

[`_handle_supervisor`][terok_sandbox.commands.supervisor._handle_supervisor]
is a thin bridge: configure root logging to stderr, then
``asyncio.run(run_supervisor(container_id, Path(sidecar_path)))`` and
return its exit code.  The async ``run_supervisor`` itself is covered in
``test_supervisor_run.py``; here we pin only the dispatch contract —
args forwarded verbatim, return code propagated, logging configured.
"""

from __future__ import annotations

from pathlib import Path
from unittest.mock import MagicMock, patch

from terok_util import LazyHandler

from terok_sandbox.commands.supervisor import (
    SUPERVISE_CHILD,
    SUPERVISOR,
    SUPERVISOR_COMMANDS,
    _handle_supervise_child,
    _handle_supervisor,
)


class TestHandleSupervisor:
    """The CLI handler bridges to ``run_supervisor`` and returns its rc."""

    def test_forwards_args_and_returns_exit_code(self) -> None:
        """``container_id`` passes through; ``sidecar_path`` becomes a ``Path``; rc propagates."""
        captured: dict[str, object] = {}

        async def _fake_run(
            container_id: str, sidecar_path: Path, container_pid: int | None = None
        ) -> int:
            captured["container_id"] = container_id
            captured["sidecar_path"] = sidecar_path
            captured["container_pid"] = container_pid
            return 7

        with (
            patch("terok_sandbox.supervisor.run_supervisor", side_effect=_fake_run),
            patch("logging.basicConfig") as basic_config,
        ):
            rc = _handle_supervisor("abc123", "/run/terok/sidecar/demo.json", 4242)

        assert rc == 7
        assert captured["container_id"] == "abc123"
        assert captured["sidecar_path"] == Path("/run/terok/sidecar/demo.json")
        assert captured["container_pid"] == 4242
        assert isinstance(captured["sidecar_path"], Path)
        # Root logging is configured so module loggers reach the wrapper's
        # per-container stderr log file.
        basic_config.assert_called_once()

    def test_runs_under_asyncio_run(self) -> None:
        """The coroutine is driven via ``asyncio.run`` (not awaited inline).

        ``run_supervisor`` is a coroutine function, so ``patch`` swaps it for
        an ``AsyncMock`` and calling it yields a coroutine; the handler must
        hand that coroutine to ``asyncio.run`` rather than awaiting inline.
        """
        sentinel = MagicMock(name="run_supervisor")
        with (
            patch("terok_sandbox.supervisor.run_supervisor", new=sentinel) as run,
            patch("terok_sandbox.commands.supervisor.asyncio.run", return_value=0) as asyncio_run,
            patch("logging.basicConfig"),
        ):
            rc = _handle_supervisor("cid", "/sidecar.json")

        assert rc == 0
        run.assert_called_once_with("cid", Path("/sidecar.json"), None)
        # The object asyncio.run drove is exactly what run_supervisor returned.
        asyncio_run.assert_called_once_with(sentinel.return_value)


class TestHandleSuperviseChild:
    """The CLI handler bridges to ``run_child`` and returns its exit code."""

    def test_forwards_args_and_returns_exit_code(self) -> None:
        """``service`` / ``container_id`` pass through; ``sidecar_path`` becomes a ``Path``."""
        with (
            patch("terok_sandbox.supervisor.children.run_child", return_value=4) as run_child,
            patch("logging.basicConfig") as basic_config,
        ):
            rc = _handle_supervise_child("vault", "abc123", "/run/terok/sidecar/demo.json")

        assert rc == 4
        run_child.assert_called_once_with("vault", "abc123", Path("/run/terok/sidecar/demo.json"))
        basic_config.assert_called_once()


class TestRegistration:
    """Both internal verbs are registered under the hidden ``internal`` group."""

    def test_registers_supervisor_and_supervise_child(self) -> None:
        assert SUPERVISOR_COMMANDS == (SUPERVISOR, SUPERVISE_CHILD)
        assert all(cmd.group == "internal" for cmd in SUPERVISOR_COMMANDS)

    def test_supervisor_verb_shape(self) -> None:
        assert SUPERVISOR.name == "supervisor"
        # Registered lazily — the target resolves to the real handler at
        # dispatch, keeping the spawn path off the handler's imports.
        assert SUPERVISOR.handler == LazyHandler(
            "terok_sandbox.commands.supervisor:_handle_supervisor"
        )
        assert tuple(a.name for a in SUPERVISOR.args) == (
            "container_id",
            "sidecar_path",
            "container_pid",
        )

    def test_supervise_child_verb_shape(self) -> None:
        assert SUPERVISE_CHILD.name == "supervise-child"
        assert SUPERVISE_CHILD.handler == LazyHandler(
            "terok_sandbox.commands.supervisor:_handle_supervise_child"
        )
        assert tuple(a.name for a in SUPERVISE_CHILD.args) == (
            "service",
            "container_id",
            "sidecar_path",
        )
