# SPDX-FileCopyrightText: 2026 Jiri Vyskocil
# SPDX-License-Identifier: Apache-2.0

"""Tests for the supervisor's process launchers in
[`terok_sandbox.supervisor.launcher`][terok_sandbox.supervisor.launcher].

The launchers just build an argv and spawn it; ``create_subprocess_exec``
is stubbed so the tests assert the *shape* of the spawn (self-invocation,
the ``supervise-child`` verb, the systemd scope wrapping) without forking
a real child.
"""

from __future__ import annotations

import sys
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from terok_sandbox.supervisor.launcher import (
    DirectLauncher,
    SystemdRunLauncher,
    default_launcher,
)

_SIDECAR = Path("/state/sidecar/demo.json")


def _spy_exec() -> tuple[AsyncMock, list[list[str]]]:
    """An ``create_subprocess_exec`` stub recording each argv it is handed."""
    calls: list[list[str]] = []

    async def _fake(*argv: str, **_kw: object) -> MagicMock:
        calls.append(list(argv))
        return MagicMock(pid=999)

    return AsyncMock(side_effect=_fake), calls


class TestDirectLauncher:
    """Spawns ``python -m terok_sandbox supervise-child <service> …`` unmodified."""

    @pytest.mark.asyncio
    async def test_launches_self_with_supervise_child_verb(self) -> None:
        spy, calls = _spy_exec()
        with patch("terok_sandbox.supervisor.launcher.asyncio.create_subprocess_exec", spy):
            handle = await DirectLauncher().launch("vault", "abc123", _SIDECAR)
        assert calls == [
            [
                sys.executable,
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


class TestSystemdRunLauncher:
    """Wraps the child in ``systemd-run --user --scope`` when it is available."""

    @pytest.mark.asyncio
    async def test_wraps_child_in_a_transient_scope(self) -> None:
        spy, calls = _spy_exec()
        with (
            patch(
                "terok_sandbox.supervisor.launcher.shutil.which",
                return_value="/usr/bin/systemd-run",
            ),
            patch("terok_sandbox.supervisor.launcher.asyncio.create_subprocess_exec", spy),
        ):
            await SystemdRunLauncher().launch("vault", "abc123def456", _SIDECAR)
        argv = calls[0]
        assert argv[0] == "/usr/bin/systemd-run"
        assert "--user" in argv and "--scope" in argv
        # The child self-invocation is preserved after the scope flags.
        tail = argv[argv.index("-m") - 1 :]
        assert tail == [
            sys.executable,
            "-m",
            "terok_sandbox",
            "supervise-child",
            "vault",
            "abc123def456",
            str(_SIDECAR),
        ]
        # Per-service unit name keyed on the short container id.
        assert any(a == "--unit=terok-abc123def456-vault" for a in argv)

    @pytest.mark.asyncio
    async def test_falls_back_to_direct_when_systemd_run_absent(self) -> None:
        spy, calls = _spy_exec()
        with (
            patch("terok_sandbox.supervisor.launcher.shutil.which", return_value=None),
            patch("terok_sandbox.supervisor.launcher.asyncio.create_subprocess_exec", spy),
        ):
            await SystemdRunLauncher().launch("gate", "abc123", _SIDECAR)
        # No systemd wrapping — a plain self-spawn.
        assert calls[0][0] == sys.executable
        assert "systemd-run" not in calls[0][0]

    def _patch_signals(self, *, on_path: bool, is_init: bool, user_mgr: bool):
        """Patch the three availability signals to the given booleans."""
        return (
            patch(
                "terok_sandbox.supervisor.launcher.shutil.which",
                return_value="/usr/bin/systemd-run" if on_path else None,
            ),
            patch(
                "terok_sandbox.supervisor.launcher._SYSTEMD_INIT_MARKER",
                MagicMock(is_dir=MagicMock(return_value=is_init)),
            ),
            patch(
                "terok_sandbox.supervisor.launcher._user_manager_socket",
                return_value=MagicMock(exists=MagicMock(return_value=user_mgr)),
            ),
        )

    def test_available_when_all_three_signals_present(self) -> None:
        which_p, init_p, mgr_p = self._patch_signals(on_path=True, is_init=True, user_mgr=True)
        with which_p, init_p, mgr_p:
            assert SystemdRunLauncher.is_available() is True

    @pytest.mark.parametrize(
        ("on_path", "is_init", "user_mgr"),
        [
            (False, True, True),  # no systemd-run binary
            (True, False, True),  # non-systemd init (no /run/systemd/system)
            (True, True, False),  # systemd, but no reachable user manager
        ],
    )
    def test_unavailable_when_any_signal_missing(
        self, on_path: bool, is_init: bool, user_mgr: bool
    ) -> None:
        which_p, init_p, mgr_p = self._patch_signals(
            on_path=on_path, is_init=is_init, user_mgr=user_mgr
        )
        with which_p, init_p, mgr_p:
            assert SystemdRunLauncher.is_available() is False


class TestDefaultLauncher:
    """``default_launcher`` probes the host and honours the env override."""

    def test_auto_prefers_systemd_when_available(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.delenv("TEROK_SUPERVISOR_LAUNCHER", raising=False)
        with patch(
            "terok_sandbox.supervisor.launcher.SystemdRunLauncher.is_available", return_value=True
        ):
            assert isinstance(default_launcher(), SystemdRunLauncher)

    def test_auto_falls_back_to_direct_on_non_systemd(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.delenv("TEROK_SUPERVISOR_LAUNCHER", raising=False)
        with patch(
            "terok_sandbox.supervisor.launcher.SystemdRunLauncher.is_available", return_value=False
        ):
            assert isinstance(default_launcher(), DirectLauncher)

    def test_env_forces_direct_even_when_systemd_available(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.setenv("TEROK_SUPERVISOR_LAUNCHER", "direct")
        with patch(
            "terok_sandbox.supervisor.launcher.SystemdRunLauncher.is_available", return_value=True
        ):
            assert isinstance(default_launcher(), DirectLauncher)

    def test_env_forces_systemd_even_when_probe_says_no(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.setenv("TEROK_SUPERVISOR_LAUNCHER", "systemd")
        with patch(
            "terok_sandbox.supervisor.launcher.SystemdRunLauncher.is_available", return_value=False
        ):
            assert isinstance(default_launcher(), SystemdRunLauncher)
