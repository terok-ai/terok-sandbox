# SPDX-FileCopyrightText: 2026 Jiri Vyskocil
# SPDX-License-Identifier: Apache-2.0

"""The placement fact and the user-manager verbs the hook and the package share."""

from __future__ import annotations

import socket
from pathlib import Path
from types import SimpleNamespace

import pytest

from terok_sandbox._util._placement import SupervisorPlacement, supervisor_placement
from terok_sandbox.resources.hooks import _supervisor_state as state

#: The real predicate, captured at import — the autouse conftest fixture
#: stubs it to True for the suite at large.
_REACHABLE = state.user_manager_reachable


def _plant_manager_socket(runtime_dir: Path) -> None:
    """A per-user manager leaves its private socket where systemd-run looks."""
    (runtime_dir / "systemd").mkdir(parents=True)
    sock = socket.socket(socket.AF_UNIX)
    sock.bind(str(runtime_dir / "systemd" / "private"))
    sock.close()


class TestUserManagerReachable:
    def test_a_private_socket_and_systemd_run_mean_reachable(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        _plant_manager_socket(tmp_path)
        monkeypatch.setattr(state, "find_host_tool", lambda _name: "/usr/bin/systemd-run")
        assert _REACHABLE(tmp_path) is True

    def test_no_socket_is_not_reachable(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.setattr(state, "find_host_tool", lambda _name: "/usr/bin/systemd-run")
        assert _REACHABLE(tmp_path) is False

    def test_a_socket_without_the_tool_is_not_reachable(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        _plant_manager_socket(tmp_path)
        monkeypatch.setattr(state, "find_host_tool", lambda _name: None)
        assert _REACHABLE(tmp_path) is False


class TestUnitNames:
    def test_unit_is_named_by_the_short_container_id(self) -> None:
        assert state.unit_name("abc123def456789") == "terok-supervisor-abc123def456.service"

    def test_pattern_reaches_every_unit(self) -> None:
        assert state.unit_pattern() == "terok-supervisor-*"

    def test_runtime_dir_is_logind_s(self) -> None:
        assert state.user_runtime_dir(1017) == Path("/run/user/1017")


class TestManagerVerbs:
    """Every verb is one quiet ``systemctl --user`` call; the exit status is the answer."""

    @pytest.fixture
    def calls(self, monkeypatch: pytest.MonkeyPatch) -> list[list[str]]:
        seen: list[list[str]] = []

        def fake_run(argv: list[str], **_kw: object) -> SimpleNamespace:
            seen.append(argv)
            return SimpleNamespace(returncode=0 if "is-active" in argv else 5)

        monkeypatch.setattr(state.subprocess, "run", fake_run)
        monkeypatch.setattr(state, "find_host_tool", lambda _: "/usr/bin/systemctl")
        return seen

    def test_unit_active_asks_is_active(self, calls: list[list[str]]) -> None:
        assert state.unit_active("terok-supervisor-abc.service") is True
        assert calls == [
            ["/usr/bin/systemctl", "--user", "--quiet", "is-active", "terok-supervisor-abc.service"]
        ]

    def test_stop_and_kill_send_their_verbs(self, calls: list[list[str]]) -> None:
        state.stop_unit("terok-supervisor-abc.service")
        state.kill_units("terok-supervisor-*")
        assert calls[0][3:] == ["stop", "terok-supervisor-abc.service"]
        assert calls[1][3:] == ["kill", "--signal=SIGKILL", "terok-supervisor-*"]

    def test_a_missing_systemctl_is_a_no(self, monkeypatch: pytest.MonkeyPatch) -> None:
        def _raise(*_a: object, **_k: object) -> None:
            raise OSError("no systemctl")

        monkeypatch.setattr(state.subprocess, "run", _raise)
        assert state.unit_active("terok-supervisor-abc.service") is False


class TestSupervisorPlacement:
    @pytest.mark.parametrize(
        ("reachable", "placement"),
        [(True, SupervisorPlacement.USER_UNIT), (False, SupervisorPlacement.NAMESPACE_DAEMON)],
        ids=["user unit", "namespace daemon"],
    )
    def test_follows_the_manager(
        self, monkeypatch: pytest.MonkeyPatch, reachable: bool, placement: SupervisorPlacement
    ) -> None:
        monkeypatch.setattr(state, "user_manager_reachable", lambda _dir: reachable)
        assert supervisor_placement() is placement
