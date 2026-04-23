# SPDX-FileCopyrightText: 2026 Jiri Vyskocil
# SPDX-License-Identifier: Apache-2.0

"""Top-level ``sandbox setup`` and ``sandbox uninstall`` orchestration.

The aggregators compose shield, vault, gate, and clearance lifecycle
into a single bootstrap and a symmetric teardown.  Phase ordering,
exit-code propagation, and opt-out flags are the contract under test —
the individual phase implementations have their own tests below.
"""

from __future__ import annotations

from unittest.mock import patch

import pytest

from terok_sandbox.commands import (
    _handle_gate_install,
    _handle_gate_uninstall,
    _handle_sandbox_setup,
    _handle_sandbox_uninstall,
    _handle_shield_uninstall,
)

# ── Fixtures ──────────────────────────────────────────────────────────────


@pytest.fixture
def install_spies():
    """Replace every ``run_*_install_phase`` with a MagicMock so order is observable.

    Default return ``True`` so the aggregator walks the happy path
    unless a test overrides a phase.  Prereq reporting is stubbed out
    — it shells out for host binaries which would noisily poll the CI
    runner's PATH.
    """
    with (
        patch("terok_sandbox.commands.run_prereq_report") as prereq,
        patch("terok_sandbox.commands.run_shield_install_phase", return_value=True) as shield,
        patch("terok_sandbox.commands.run_vault_install_phase", return_value=True) as vault,
        patch("terok_sandbox.commands.run_gate_install_phase", return_value=True) as gate,
        patch("terok_sandbox.commands.run_clearance_install_phase", return_value=True) as clearance,
    ):
        yield {
            "prereq": prereq,
            "shield": shield,
            "vault": vault,
            "gate": gate,
            "clearance": clearance,
        }


@pytest.fixture
def uninstall_spies():
    """Replace every uninstall phase handler with a MagicMock so order is observable."""
    with (
        patch("terok_sandbox.commands._handle_shield_uninstall") as shield_uninstall,
        patch("terok_sandbox.commands._handle_vault_uninstall") as vault_uninstall,
        patch("terok_sandbox.commands._handle_gate_uninstall") as gate_uninstall,
        patch("terok_sandbox.commands._handle_clearance_uninstall") as clearance_uninstall,
    ):
        yield {
            "shield_uninstall": shield_uninstall,
            "vault_uninstall": vault_uninstall,
            "gate_uninstall": gate_uninstall,
            "clearance_uninstall": clearance_uninstall,
        }


# ── Setup aggregator ──────────────────────────────────────────────────────


class TestSandboxSetup:
    """``sandbox setup`` orchestrates prereq → shield → vault → gate → clearance."""

    def test_default_runs_all_phases_in_order(self, install_spies) -> None:
        order: list[str] = []
        install_spies["prereq"].side_effect = lambda _cfg: order.append("prereq")
        install_spies["shield"].side_effect = lambda **_: order.append("shield") or True
        install_spies["vault"].side_effect = lambda _cfg: order.append("vault") or True
        install_spies["gate"].side_effect = lambda _cfg: order.append("gate") or True
        install_spies["clearance"].side_effect = lambda: order.append("clearance") or True

        _handle_sandbox_setup()

        assert order == ["prereq", "shield", "vault", "gate", "clearance"]

    def test_root_flag_threaded_to_shield_phase(self, install_spies) -> None:
        _handle_sandbox_setup(root=True)
        install_spies["shield"].assert_called_once_with(root=True)

    @pytest.mark.parametrize(
        ("skip_kwarg", "skipped_spy"),
        [
            ("no_shield", "shield"),
            ("no_vault", "vault"),
            ("no_gate", "gate"),
            ("no_clearance", "clearance"),
        ],
    )
    def test_opt_out_flag_skips_exactly_its_phase(
        self, install_spies, skip_kwarg: str, skipped_spy: str
    ) -> None:
        _handle_sandbox_setup(**{skip_kwarg: True})
        install_spies[skipped_spy].assert_not_called()
        for key in ("shield", "vault", "gate", "clearance"):
            if key != skipped_spy:
                assert install_spies[key].called, f"{key} should still run"

    def test_failing_phase_exits_nonzero_after_others_run(self, install_spies) -> None:
        """A vault failure must not short-circuit the gate + clearance phases."""
        install_spies["vault"].return_value = False
        with pytest.raises(SystemExit) as exc:
            _handle_sandbox_setup()
        assert exc.value.code == 1
        install_spies["gate"].assert_called_once()
        install_spies["clearance"].assert_called_once()

    def test_happy_path_does_not_raise(self, install_spies) -> None:
        """Every phase reports ``ok=True`` → the aggregator returns normally."""
        _handle_sandbox_setup()  # no SystemExit expected


# ── Uninstall aggregator ──────────────────────────────────────────────────


class TestSandboxUninstall:
    """``sandbox uninstall`` runs clearance → gate → vault → shield (reverse of install)."""

    def test_default_uninstalls_all_four_phases(self, uninstall_spies) -> None:
        _handle_sandbox_uninstall()
        uninstall_spies["clearance_uninstall"].assert_called_once_with()
        uninstall_spies["gate_uninstall"].assert_called_once_with()
        uninstall_spies["vault_uninstall"].assert_called_once_with()
        uninstall_spies["shield_uninstall"].assert_called_once_with(user=True, root=False)

    def test_phases_run_in_reverse_install_order(self, uninstall_spies) -> None:
        """Clearance first (no dependants), shield last (most disruptive to live containers)."""
        order: list[str] = []
        uninstall_spies["clearance_uninstall"].side_effect = lambda: order.append("clearance")
        uninstall_spies["gate_uninstall"].side_effect = lambda: order.append("gate")
        uninstall_spies["vault_uninstall"].side_effect = lambda: order.append("vault")
        uninstall_spies["shield_uninstall"].side_effect = lambda **_: order.append("shield")

        _handle_sandbox_uninstall()

        assert order == ["clearance", "gate", "vault", "shield"]

    def test_root_flag_removes_shield_hooks_system_wide(self, uninstall_spies) -> None:
        _handle_sandbox_uninstall(root=True)
        uninstall_spies["shield_uninstall"].assert_called_once_with(user=False, root=True)

    @pytest.mark.parametrize(
        ("skip_kwarg", "skipped_spy_key"),
        [
            ("no_shield", "shield_uninstall"),
            ("no_vault", "vault_uninstall"),
            ("no_gate", "gate_uninstall"),
            ("no_clearance", "clearance_uninstall"),
        ],
    )
    def test_opt_out_flag_skips_exactly_its_phase(
        self, uninstall_spies, skip_kwarg: str, skipped_spy_key: str
    ) -> None:
        _handle_sandbox_uninstall(**{skip_kwarg: True})
        uninstall_spies[skipped_spy_key].assert_not_called()
        for key in uninstall_spies:
            if key != skipped_spy_key:
                assert uninstall_spies[key].called, f"{key} should still run"

    def test_failing_phase_does_not_abort_subsequent_phases(self, uninstall_spies) -> None:
        """A vault uninstall that raises SystemExit must not skip shield cleanup."""
        uninstall_spies["vault_uninstall"].side_effect = SystemExit("vault teardown broken")
        with pytest.raises(SystemExit):
            _handle_sandbox_uninstall()
        # Gate ran before the failure; shield must run after the failure.
        uninstall_spies["gate_uninstall"].assert_called_once()
        uninstall_spies["shield_uninstall"].assert_called_once()

    def test_all_phases_succeeding_does_not_exit_nonzero(self, uninstall_spies) -> None:
        """Happy path is not wrapped in an exit-1 just because of the try/except."""
        _handle_sandbox_uninstall()


# ── Gate install/uninstall handlers (aggregator phase primitives) ────────


class TestHandleGateInstall:
    """Gate install handler — refuses hosts without systemd-user."""

    def test_installs_when_systemd_available(self, capsys: pytest.CaptureFixture[str]) -> None:
        from terok_sandbox.gate.lifecycle import GateServerManager

        with (
            patch.object(GateServerManager, "is_systemd_available", return_value=True),
            patch.object(GateServerManager, "install_systemd_units") as install,
        ):
            _handle_gate_install()
        install.assert_called_once()
        assert "installed" in capsys.readouterr().out.lower()

    def test_refuses_when_systemd_unavailable(self, capsys: pytest.CaptureFixture[str]) -> None:
        from terok_sandbox.gate.lifecycle import GateServerManager

        with patch.object(GateServerManager, "is_systemd_available", return_value=False):
            with pytest.raises(SystemExit) as exc:
                _handle_gate_install()
        assert exc.value.code == 1
        assert "systemd" in capsys.readouterr().out.lower()


class TestHandleGateUninstall:
    """Gate uninstall handler — tolerates both daemon-started and systemd-installed state."""

    def test_stops_daemon_when_running_as_daemon(self, capsys: pytest.CaptureFixture[str]) -> None:
        from terok_sandbox.gate.lifecycle import GateServerManager, GateServerStatus

        status = GateServerStatus(mode="daemon", running=True, port=9418)
        with (
            patch.object(GateServerManager, "get_status", return_value=status),
            patch.object(GateServerManager, "stop_daemon") as stop,
            patch.object(GateServerManager, "is_systemd_available", return_value=False),
        ):
            _handle_gate_uninstall()
        stop.assert_called_once()
        assert "removed" in capsys.readouterr().out.lower()

    def test_uninstalls_systemd_units_when_available(
        self, capsys: pytest.CaptureFixture[str]
    ) -> None:
        from terok_sandbox.gate.lifecycle import GateServerManager, GateServerStatus

        status = GateServerStatus(mode="systemd", running=False, port=9418)
        with (
            patch.object(GateServerManager, "get_status", return_value=status),
            patch.object(GateServerManager, "is_systemd_available", return_value=True),
            patch.object(GateServerManager, "uninstall_systemd_units") as un,
        ):
            _handle_gate_uninstall()
        un.assert_called_once()


# ── Shield uninstall CLI flag validation ─────────────────────────────────


class TestHandleShieldUninstall:
    """The ``shield uninstall-hooks`` handler's user/root flag contract."""

    def test_missing_flags_exits_with_usage_hint(self, capsys: pytest.CaptureFixture[str]) -> None:
        with pytest.raises(SystemExit) as exc:
            _handle_shield_uninstall()
        msg = str(exc.value)
        assert "--root" in msg and "--user" in msg

    def test_user_flag_invokes_library_uninstall(self, capsys: pytest.CaptureFixture[str]) -> None:
        with patch("terok_sandbox.shield.run_uninstall") as mock_run:
            _handle_shield_uninstall(user=True)
        mock_run.assert_called_once_with(root=False, user=True)
        assert "user" in capsys.readouterr().out

    def test_root_flag_invokes_library_uninstall(self, capsys: pytest.CaptureFixture[str]) -> None:
        with patch("terok_sandbox.shield.run_uninstall") as mock_run:
            _handle_shield_uninstall(root=True)
        mock_run.assert_called_once_with(root=True, user=False)
        assert "system" in capsys.readouterr().out
