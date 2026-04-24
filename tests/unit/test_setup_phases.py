# SPDX-FileCopyrightText: 2026 Jiri Vyskocil
# SPDX-License-Identifier: Apache-2.0

"""Tests for the sandbox-wide setup phase functions.

Covers the individual phases ``_handle_sandbox_setup`` wires together:
prereq reporting, shield / vault / gate / clearance install, the
shared reinstall skeleton, and the lifecycle helpers.  The aggregator
orchestration itself is tested in ``test_setup_aggregator.py``.
"""

from __future__ import annotations

from unittest.mock import MagicMock, patch

import pytest

from terok_sandbox._setup import (
    Marker,
    SelinuxStatus,
    _enable_and_restart_user_unit,
    _reinstall_systemd_service,
    _stage,
    _stop_and_uninstall,
    run_clearance_install_phase,
    run_gate_install_phase,
    run_prereq_report,
    run_shield_install_phase,
    run_vault_install_phase,
)
from terok_sandbox.config import SandboxConfig


@pytest.fixture
def bare_cfg() -> SandboxConfig:
    """A spec'd mock of :class:`SandboxConfig` — no XDG I/O, no port registry.

    Phases under test read ``cfg`` as an opaque handle that's passed to
    a manager constructor (which is itself patched away); they never
    reach into its fields.  A ``MagicMock(spec=…)`` still rejects typos
    (attribute access on an unknown name raises) while skipping the
    real ``__post_init__`` that resolves TCP ports via the registry.
    """
    return MagicMock(spec=SandboxConfig)


# ── Stage primitive ──────────────────────────────────────────────────


class TestStage:
    """The ``_stage`` helper is the one place unit output formatting lives."""

    def test_writes_label_marker_and_detail(self, capsys: pytest.CaptureFixture[str]) -> None:
        _stage("Vault", Marker.OK, "systemd, tcp, reachable")
        out = capsys.readouterr().out
        assert "Vault" in out
        assert " ok " in out
        assert "(systemd, tcp, reachable)" in out

    def test_blank_detail_emits_no_parens(self, capsys: pytest.CaptureFixture[str]) -> None:
        _stage("Shield hooks", Marker.OK)
        assert "()" not in capsys.readouterr().out

    def test_label_padded_to_consistent_column(self, capsys: pytest.CaptureFixture[str]) -> None:
        _stage("x", Marker.OK, "a")
        _stage("a_longer_label", Marker.OK, "b")
        lines = capsys.readouterr().out.splitlines()
        assert lines[0].index(" ok ") == lines[1].index(" ok ")


# ── Prereq report ────────────────────────────────────────────────────


class TestPrereqReport:
    """Prereq report writes stage lines for every probe — never raises."""

    def test_reports_host_binaries(
        self,
        bare_cfg: SandboxConfig,
        capsys: pytest.CaptureFixture[str],
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        monkeypatch.setattr("terok_sandbox._setup.shutil.which", lambda name: f"/usr/bin/{name}")
        with (
            patch("terok_shield.check_firewall_binaries", return_value=()),
            patch(
                "terok_sandbox._setup.check_selinux_status",
                return_value=MagicMock(status=SelinuxStatus.NOT_APPLICABLE_TCP_MODE),
            ),
        ):
            run_prereq_report(bare_cfg)
        out = capsys.readouterr().out
        assert "podman" in out
        assert "git" in out
        assert "ssh-keygen" in out

    def test_reports_firewall_binaries_via_shield(
        self,
        bare_cfg: SandboxConfig,
        capsys: pytest.CaptureFixture[str],
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        monkeypatch.setattr("terok_sandbox._setup.shutil.which", lambda _n: None)
        fake_check = MagicMock(path="/usr/sbin/nft", purpose="ruleset enforcement", ok=True)
        fake_check.name = "nft"  # ``name=`` is a MagicMock constructor kwarg, not an attr
        with (
            patch("terok_shield.check_firewall_binaries", return_value=(fake_check,)),
            patch(
                "terok_sandbox._setup.check_selinux_status",
                return_value=MagicMock(status=SelinuxStatus.NOT_APPLICABLE_TCP_MODE),
            ),
        ):
            run_prereq_report(bare_cfg)
        out = capsys.readouterr().out
        assert "nft" in out
        assert "/usr/sbin/nft" in out

    def test_selinux_ok_renders_a_stage_line(
        self,
        bare_cfg: SandboxConfig,
        capsys: pytest.CaptureFixture[str],
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        monkeypatch.setattr("terok_sandbox._setup.shutil.which", lambda _n: None)
        with (
            patch("terok_shield.check_firewall_binaries", return_value=()),
            patch(
                "terok_sandbox._setup.check_selinux_status",
                return_value=MagicMock(status=SelinuxStatus.OK),
            ),
        ):
            run_prereq_report(bare_cfg)
        assert "SELinux policy" in capsys.readouterr().out

    def test_selinux_not_applicable_stays_silent(
        self,
        bare_cfg: SandboxConfig,
        capsys: pytest.CaptureFixture[str],
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        """Hosts where SELinux isn't enforcing shouldn't see a policy stage line."""
        monkeypatch.setattr("terok_sandbox._setup.shutil.which", lambda _n: None)
        with (
            patch("terok_shield.check_firewall_binaries", return_value=()),
            patch(
                "terok_sandbox._setup.check_selinux_status",
                return_value=MagicMock(status=SelinuxStatus.NOT_APPLICABLE_PERMISSIVE),
            ),
        ):
            run_prereq_report(bare_cfg)
        assert "SELinux" not in capsys.readouterr().out


# ── Shield install phase ─────────────────────────────────────────────


class TestShieldInstallPhase:
    """Shield phase: install hooks + verify health."""

    def test_clean_install_reports_ok(self, capsys: pytest.CaptureFixture[str]) -> None:
        with (
            patch("terok_sandbox.shield.run_setup") as setup,
            patch(
                "terok_sandbox.shield.check_environment",
                return_value=MagicMock(health="ok"),
            ),
        ):
            assert run_shield_install_phase(root=False) is True
        setup.assert_called_once_with(root=False, user=True)
        assert "ok" in capsys.readouterr().out

    def test_bypass_mode_reports_warn_but_still_ok(
        self, capsys: pytest.CaptureFixture[str]
    ) -> None:
        """A bypass-firewall host lands as WARN but counts as ``ok`` for the aggregator."""
        with (
            patch("terok_sandbox.shield.run_setup"),
            patch(
                "terok_sandbox.shield.check_environment",
                return_value=MagicMock(health="bypass"),
            ),
        ):
            assert run_shield_install_phase(root=False) is True
        assert "WARN" in capsys.readouterr().out

    def test_install_raises_reports_fail(self, capsys: pytest.CaptureFixture[str]) -> None:
        with patch("terok_sandbox.shield.run_setup", side_effect=RuntimeError("sudo required")):
            assert run_shield_install_phase(root=False) is False
        assert "FAIL" in capsys.readouterr().out

    def test_unhealthy_post_install_reports_fail(self, capsys: pytest.CaptureFixture[str]) -> None:
        """Install succeeded on the surface but ``check_environment`` disagrees."""
        with (
            patch("terok_sandbox.shield.run_setup"),
            patch(
                "terok_sandbox.shield.check_environment",
                return_value=MagicMock(health="setup-needed"),
            ),
        ):
            assert run_shield_install_phase(root=False) is False
        assert "FAIL" in capsys.readouterr().out


# ── Shared reinstall skeleton ────────────────────────────────────────


class TestReinstallSystemdService:
    """Stop → uninstall → install → verify, with both exception tuples covered."""

    def test_happy_path_invokes_full_lifecycle(self, capsys: pytest.CaptureFixture[str]) -> None:
        mgr = MagicMock()
        mgr.get_status.return_value = MagicMock(mode="systemd", transport="tcp")
        assert _reinstall_systemd_service(label="Vault", mgr=mgr) is True
        mgr.stop_daemon.assert_called_once()
        mgr.uninstall_systemd_units.assert_called_once()
        mgr.install_systemd_units.assert_called_once()
        mgr.ensure_reachable.assert_called_once()
        assert "reachable" in capsys.readouterr().out

    def test_install_systemexit_reports_fail(self, capsys: pytest.CaptureFixture[str]) -> None:
        mgr = MagicMock()
        mgr.install_systemd_units.side_effect = SystemExit("no ports")
        assert _reinstall_systemd_service(label="Vault", mgr=mgr) is False
        assert "FAIL" in capsys.readouterr().out

    def test_install_generic_exception_reports_fail(
        self, capsys: pytest.CaptureFixture[str]
    ) -> None:
        mgr = MagicMock()
        mgr.install_systemd_units.side_effect = RuntimeError("template missing")
        assert _reinstall_systemd_service(label="Gate server", mgr=mgr) is False
        assert "install:" in capsys.readouterr().out

    def test_verify_failure_reports_installed_but_unreachable(
        self, capsys: pytest.CaptureFixture[str]
    ) -> None:
        mgr = MagicMock()
        mgr.ensure_reachable.side_effect = SystemExit("connection refused")
        assert _reinstall_systemd_service(label="Gate server", mgr=mgr) is False
        assert "NOT reachable" in capsys.readouterr().out

    def test_custom_reachable_exc_tuple_catches_narrower_error(
        self, capsys: pytest.CaptureFixture[str]
    ) -> None:
        """Vault's caller passes ``(VaultUnreachableError, SystemExit)`` so both types reach FAIL."""

        class VaultUnreachable(RuntimeError):
            pass

        mgr = MagicMock()
        mgr.ensure_reachable.side_effect = VaultUnreachable("socket silent")
        ok = _reinstall_systemd_service(
            label="Vault", mgr=mgr, reachable_exc=(VaultUnreachable, SystemExit)
        )
        assert ok is False
        assert "NOT reachable" in capsys.readouterr().out

    def test_stop_or_uninstall_exceptions_soft_fail(
        self, capsys: pytest.CaptureFixture[str]
    ) -> None:
        """Dangling daemon / unit file from a broken install is tolerated."""
        mgr = MagicMock()
        mgr.stop_daemon.side_effect = RuntimeError("no pid")
        mgr.uninstall_systemd_units.side_effect = RuntimeError("no units")
        mgr.get_status.return_value = MagicMock(mode="systemd", transport="tcp")
        assert _reinstall_systemd_service(label="Vault", mgr=mgr) is True


# ── Vault / gate install phase adapters ──────────────────────────────


class TestVaultInstallPhase:
    """Vault's entry-point wires the VaultUnreachableError exception into the reinstall skeleton."""

    def test_clean_reinstall_invokes_full_lifecycle(
        self, bare_cfg: SandboxConfig, capsys: pytest.CaptureFixture[str]
    ) -> None:
        from terok_sandbox.vault.lifecycle import VaultManager

        status = MagicMock(mode="systemd", transport="tcp")
        with (
            patch.object(VaultManager, "stop_daemon") as stop,
            patch.object(VaultManager, "uninstall_systemd_units") as uninstall,
            patch.object(VaultManager, "install_systemd_units") as install,
            patch.object(VaultManager, "ensure_reachable") as verify,
            patch.object(VaultManager, "get_status", return_value=status),
        ):
            assert run_vault_install_phase(bare_cfg) is True
        stop.assert_called_once()
        uninstall.assert_called_once()
        install.assert_called_once()
        verify.assert_called_once()
        assert "reachable" in capsys.readouterr().out

    def test_vault_unreachable_error_is_reported(
        self, bare_cfg: SandboxConfig, capsys: pytest.CaptureFixture[str]
    ) -> None:
        """Vault's typed ``VaultUnreachableError`` reaches FAIL through the custom exc tuple."""
        from pathlib import Path

        from terok_sandbox.vault.lifecycle import VaultManager, VaultUnreachableError

        unreachable = VaultUnreachableError(
            socket_path=Path("/tmp/vault.sock"), db_path=Path("/tmp/vault.db")
        )
        with (
            patch.object(VaultManager, "stop_daemon"),
            patch.object(VaultManager, "uninstall_systemd_units"),
            patch.object(VaultManager, "install_systemd_units"),
            patch.object(VaultManager, "ensure_reachable", side_effect=unreachable),
        ):
            assert run_vault_install_phase(bare_cfg) is False
        assert "NOT reachable" in capsys.readouterr().out


class TestGateInstallPhase:
    """Gate's entry-point adds the systemd-availability preflight."""

    def test_systemd_unavailable_is_warning_not_failure(
        self, bare_cfg: SandboxConfig, capsys: pytest.CaptureFixture[str]
    ) -> None:
        """Hosts without user systemd (CI containers) skip the phase cleanly."""
        from terok_sandbox.gate.lifecycle import GateServerManager

        with patch.object(GateServerManager, "is_systemd_available", return_value=False):
            assert run_gate_install_phase(bare_cfg) is True
        assert "WARN" in capsys.readouterr().out

    def test_clean_reinstall_invokes_full_lifecycle(
        self, bare_cfg: SandboxConfig, capsys: pytest.CaptureFixture[str]
    ) -> None:
        from terok_sandbox.gate.lifecycle import GateServerManager

        status = MagicMock(mode="systemd", transport="tcp")
        with (
            patch.object(GateServerManager, "is_systemd_available", return_value=True),
            patch.object(GateServerManager, "stop_daemon") as stop,
            patch.object(GateServerManager, "uninstall_systemd_units") as uninstall,
            patch.object(GateServerManager, "install_systemd_units") as install,
            patch.object(GateServerManager, "ensure_reachable") as verify,
            patch.object(GateServerManager, "get_status", return_value=status),
        ):
            assert run_gate_install_phase(bare_cfg) is True
        stop.assert_called_once()
        uninstall.assert_called_once()
        install.assert_called_once()
        verify.assert_called_once()


# ── Clearance install phase ──────────────────────────────────────────


class TestClearanceInstallPhase:
    """Clearance phase: install the hub + verdict + notifier; soft-skip on missing import."""

    def test_happy_path_installs_hub_and_notifier(self, capsys: pytest.CaptureFixture[str]) -> None:
        with (
            patch("terok_clearance.runtime.installer.install_service") as install_hub,
            patch("terok_clearance.runtime.installer.install_notifier_service") as install_notifier,
            patch("terok_sandbox._setup._systemctl.run_best_effort"),
            patch("terok_sandbox._setup._enable_and_restart_user_unit"),
        ):
            assert run_clearance_install_phase() is True
        install_hub.assert_called_once()
        install_notifier.assert_called_once()
        out = capsys.readouterr().out
        assert "Clearance hub" in out
        assert "Clearance notifier" in out

    def test_batched_daemon_reload_runs_once_per_unit_pair(
        self, capsys: pytest.CaptureFixture[str]
    ) -> None:
        """Hub install batches ``daemon-reload`` once for the hub + verdict units.

        Otherwise every ``--user enable`` + ``--user restart`` cascade would
        pay its own daemon-reload round-trip — three per install before the
        batching fix.
        """
        with (
            patch("terok_clearance.runtime.installer.install_service"),
            patch("terok_clearance.runtime.installer.install_notifier_service"),
            patch("terok_sandbox._setup._systemctl.run_best_effort") as run,
            patch("terok_sandbox._setup._enable_and_restart_user_unit"),
        ):
            run_clearance_install_phase()
        # One daemon-reload per clearance install call (hub/verdict + notifier),
        # batched so a three-unit install doesn't pay three round-trips.
        reloads = [call for call in run.call_args_list if call.args == ("daemon-reload",)]
        assert len(reloads) == 2

    def test_hub_failure_reports_fail(self, capsys: pytest.CaptureFixture[str]) -> None:
        with (
            patch(
                "terok_clearance.runtime.installer.install_service",
                side_effect=RuntimeError("template missing"),
            ),
            patch("terok_clearance.runtime.installer.install_notifier_service"),
            patch("terok_sandbox._setup._systemctl.run_best_effort"),
            patch("terok_sandbox._setup._enable_and_restart_user_unit"),
        ):
            assert run_clearance_install_phase() is False
        assert "FAIL" in capsys.readouterr().out

    def test_notifier_failure_does_not_flip_exit_code(
        self, capsys: pytest.CaptureFixture[str]
    ) -> None:
        """Notifier is non-critical — a failure WARNs without failing the phase."""
        with (
            patch("terok_clearance.runtime.installer.install_service"),
            patch(
                "terok_clearance.runtime.installer.install_notifier_service",
                side_effect=RuntimeError("session bus missing"),
            ),
            patch("terok_sandbox._setup._systemctl.run_best_effort"),
            patch("terok_sandbox._setup._enable_and_restart_user_unit"),
        ):
            assert run_clearance_install_phase() is True


# ── Lifecycle helpers ────────────────────────────────────────────────


class TestStopAndUninstall:
    """Both steps soft-fail — authoritative install is next, dangling bits reported by verify."""

    def test_both_succeed(self) -> None:
        stop, uninstall = MagicMock(), MagicMock()
        _stop_and_uninstall(stop, uninstall)
        stop.assert_called_once()
        uninstall.assert_called_once()

    def test_stop_raises_uninstall_still_runs(self) -> None:
        stop = MagicMock(side_effect=RuntimeError("no pid"))
        uninstall = MagicMock()
        _stop_and_uninstall(stop, uninstall)
        uninstall.assert_called_once()

    def test_both_raise_no_propagation(self) -> None:
        stop = MagicMock(side_effect=RuntimeError("no pid"))
        uninstall = MagicMock(side_effect=RuntimeError("no units"))
        _stop_and_uninstall(stop, uninstall)  # must not raise


class TestEnableAndRestartUserUnit:
    """``_enable_and_restart_user_unit`` — enable + restart, no daemon-reload."""

    def test_invokes_enable_then_restart_without_reload(self) -> None:
        """Both verbs run in order; daemon-reload is the caller's batched responsibility."""
        with patch("terok_sandbox._setup._systemctl.run_best_effort") as run:
            _enable_and_restart_user_unit("terok-vault")
        verbs = [call.args[0] for call in run.call_args_list]
        assert verbs == ["enable", "restart"]
        for call in run.call_args_list:
            assert call.args[1] == "terok-vault"
