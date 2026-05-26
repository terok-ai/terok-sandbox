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
    SelinuxStatus,
    _reinstall_systemd_service,
    _start_managed_daemon,
    _stop_and_uninstall,
    run_gate_install_phase,
    run_gate_uninstall_phase,
    run_legacy_install_cleanup_phase,
    run_prereq_report,
    run_shield_install_phase,
    run_shield_uninstall_phase,
)
from terok_sandbox.config import SandboxConfig


@pytest.fixture
def bare_cfg() -> SandboxConfig:
    """A spec'd mock of [`SandboxConfig`][terok_sandbox.SandboxConfig] — no XDG I/O, no port registry.

    Phases under test read ``cfg`` as an opaque handle that's passed to
    a manager constructor (which is itself patched away); they only
    reach into one real field — ``services_mode``, for the SELinux
    check — which we set explicitly.  A ``MagicMock(spec=…)`` still
    rejects typos (attribute access on an unknown name raises) while
    skipping the real ``__post_init__`` that resolves TCP ports via
    the registry.
    """
    mock = MagicMock(spec=SandboxConfig)
    mock.with_resolved_ports.return_value = mock
    mock.services_mode = "socket"
    mock.experimental = False
    return mock


# ── Stage primitive ──────────────────────────────────────────────────
# Formatter tests live in ``test_stage.py`` where the renderer does —
# ``_setup`` only composes stage lines, so testing the same formatting
# rules here was a duplication trap.


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
            patch("terok_sandbox.integrations.shield.check_firewall_binaries", return_value=()),
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
            patch(
                "terok_sandbox.integrations.shield.check_firewall_binaries",
                return_value=(fake_check,),
            ),
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
            patch("terok_sandbox.integrations.shield.check_firewall_binaries", return_value=()),
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
            patch("terok_sandbox.integrations.shield.check_firewall_binaries", return_value=()),
            patch(
                "terok_sandbox._setup.check_selinux_status",
                return_value=MagicMock(status=SelinuxStatus.NOT_APPLICABLE_PERMISSIVE),
            ),
        ):
            run_prereq_report(bare_cfg)
        assert "SELinux" not in capsys.readouterr().out

    @pytest.mark.parametrize(
        ("status", "detail_fragment"),
        [
            (SelinuxStatus.POLICY_MISSING, "install:"),
            (SelinuxStatus.LIBSELINUX_MISSING, "libselinux.so.1"),
        ],
    )
    def test_selinux_problem_statuses_render_missing_with_hint(
        self,
        bare_cfg: SandboxConfig,
        capsys: pytest.CaptureFixture[str],
        monkeypatch: pytest.MonkeyPatch,
        status: SelinuxStatus,
        detail_fragment: str,
    ) -> None:
        """Problem statuses route to MISSING with a pointer-to-fix in the detail."""
        monkeypatch.setattr("terok_sandbox._setup.shutil.which", lambda _n: None)
        with (
            patch("terok_sandbox.integrations.shield.check_firewall_binaries", return_value=()),
            patch(
                "terok_sandbox._setup.check_selinux_status",
                return_value=MagicMock(status=status),
            ),
        ):
            run_prereq_report(bare_cfg)
        out = capsys.readouterr().out
        assert "SELinux policy" in out
        assert "MISSING" in out
        assert detail_fragment in out

    def test_missing_firewall_binary_renders_missing_line(
        self,
        bare_cfg: SandboxConfig,
        capsys: pytest.CaptureFixture[str],
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        """A firewall binary with ``ok=False`` surfaces its ``purpose`` as the MISSING detail."""
        monkeypatch.setattr("terok_sandbox._setup.shutil.which", lambda name: f"/usr/bin/{name}")
        bad_check = MagicMock(path="", purpose="DNS resolver", ok=False)
        bad_check.name = "dnsmasq"  # ``name=`` is a MagicMock kwarg, not an attr
        with (
            patch(
                "terok_sandbox.integrations.shield.check_firewall_binaries",
                return_value=(bad_check,),
            ),
            patch(
                "terok_sandbox._setup.check_selinux_status",
                return_value=MagicMock(status=SelinuxStatus.NOT_APPLICABLE_TCP_MODE),
            ),
        ):
            run_prereq_report(bare_cfg)
        out = capsys.readouterr().out
        assert "dnsmasq" in out and "MISSING" in out and "DNS resolver" in out

    def test_krun_binaries_skipped_when_experimental_off(
        self,
        bare_cfg: SandboxConfig,
        capsys: pytest.CaptureFixture[str],
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        """Default operators (experimental off) don't see a krun probe row.

        Reporting ``ip`` as missing would be confusing noise for the 99%
        of users who never touch the krun runtime.
        """
        bare_cfg.experimental = False
        monkeypatch.setattr("terok_sandbox._setup.shutil.which", lambda _n: None)
        with (
            patch("terok_sandbox.integrations.shield.check_firewall_binaries", return_value=()),
            patch("terok_sandbox.integrations.shield.check_krun_binaries") as krun_probe,
            patch(
                "terok_sandbox._setup.check_selinux_status",
                return_value=MagicMock(status=SelinuxStatus.NOT_APPLICABLE_TCP_MODE),
            ),
        ):
            run_prereq_report(bare_cfg)
        krun_probe.assert_not_called()
        assert "ip" not in capsys.readouterr().out.split()

    def test_krun_binaries_reported_when_experimental_on(
        self,
        bare_cfg: SandboxConfig,
        capsys: pytest.CaptureFixture[str],
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        """``experimental: true`` in the config flips the krun probe on."""
        bare_cfg.experimental = True
        monkeypatch.setattr("terok_sandbox._setup.shutil.which", lambda _n: None)
        ip_check = MagicMock(path="/sbin/ip", purpose="krun in-netns IP", ok=True)
        ip_check.name = "ip"
        with (
            patch("terok_sandbox.integrations.shield.check_firewall_binaries", return_value=()),
            patch(
                "terok_sandbox.integrations.shield.check_krun_binaries", return_value=(ip_check,)
            ),
            patch(
                "terok_sandbox._setup.check_selinux_status",
                return_value=MagicMock(status=SelinuxStatus.NOT_APPLICABLE_TCP_MODE),
            ),
        ):
            run_prereq_report(bare_cfg)
        out = capsys.readouterr().out
        assert "ip" in out
        assert "/sbin/ip" in out


# ── Shield install phase ─────────────────────────────────────────────


class TestShieldInstallPhase:
    """Shield phase: install hooks + verify health."""

    def test_clean_install_reports_ok(self, capsys: pytest.CaptureFixture[str]) -> None:
        with (
            patch("terok_sandbox.integrations.shield.ShieldHooks.install") as setup,
            patch(
                "terok_sandbox.integrations.shield.check_environment",
                return_value=MagicMock(health="ok"),
            ),
        ):
            assert run_shield_install_phase() is True
        setup.assert_called_once_with()
        assert "ok" in capsys.readouterr().out

    def test_bypass_mode_reports_warn_but_still_ok(
        self, capsys: pytest.CaptureFixture[str]
    ) -> None:
        """A bypass-firewall host lands as WARN but counts as ``ok`` for the aggregator."""
        with (
            patch("terok_sandbox.integrations.shield.ShieldHooks.install"),
            patch(
                "terok_sandbox.integrations.shield.check_environment",
                return_value=MagicMock(health="bypass"),
            ),
        ):
            assert run_shield_install_phase() is True
        assert "WARN" in capsys.readouterr().out

    def test_install_raises_reports_fail(self, capsys: pytest.CaptureFixture[str]) -> None:
        with patch(
            "terok_sandbox.integrations.shield.ShieldHooks.install",
            side_effect=RuntimeError("install failed"),
        ):
            assert run_shield_install_phase() is False
        assert "FAIL" in capsys.readouterr().out

    def test_unhealthy_post_install_reports_fail(self, capsys: pytest.CaptureFixture[str]) -> None:
        """Install succeeded on the surface but ``check_environment`` disagrees."""
        with (
            patch("terok_sandbox.integrations.shield.ShieldHooks.install"),
            patch(
                "terok_sandbox.integrations.shield.check_environment",
                return_value=MagicMock(health="setup-needed"),
            ),
        ):
            assert run_shield_install_phase() is False
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


class TestStartManagedDaemon:
    """Stop → start → verify cycle for the no-systemd code path."""

    def test_happy_path_starts_and_verifies(self, capsys: pytest.CaptureFixture[str]) -> None:
        mgr = MagicMock()
        mgr.get_status.return_value = MagicMock(mode="daemon", transport="socket")
        assert _start_managed_daemon(label="Vault", mgr=mgr) is True
        mgr.stop_daemon.assert_called_once()
        mgr.start_daemon.assert_called_once()
        mgr.ensure_reachable.assert_called_once()
        out = capsys.readouterr().out
        assert "reachable" in out
        assert "no systemd" in out

    def test_start_systemexit_reports_fail(self, capsys: pytest.CaptureFixture[str]) -> None:
        mgr = MagicMock()
        mgr.start_daemon.side_effect = SystemExit("port busy")
        assert _start_managed_daemon(label="Vault", mgr=mgr) is False
        assert "port busy" in capsys.readouterr().out

    def test_start_generic_exception_reports_fail(self, capsys: pytest.CaptureFixture[str]) -> None:
        mgr = MagicMock()
        mgr.start_daemon.side_effect = RuntimeError("crashed")
        assert _start_managed_daemon(label="Vault", mgr=mgr) is False
        assert "daemon start" in capsys.readouterr().out

    def test_verify_failure_reports_started_but_unreachable(
        self, capsys: pytest.CaptureFixture[str]
    ) -> None:
        mgr = MagicMock()
        mgr.ensure_reachable.side_effect = SystemExit("socket silent")
        assert _start_managed_daemon(label="Vault", mgr=mgr) is False
        assert "started but NOT reachable" in capsys.readouterr().out

    def test_stop_exception_soft_fail(self, capsys: pytest.CaptureFixture[str]) -> None:
        """A failed stop on a stale daemon doesn't prevent a fresh start."""
        mgr = MagicMock()
        mgr.stop_daemon.side_effect = RuntimeError("no pid")
        mgr.get_status.return_value = MagicMock(mode="daemon", transport="tcp")
        assert _start_managed_daemon(label="Vault", mgr=mgr) is True
        mgr.start_daemon.assert_called_once()


# ── Gate install phase adapter ───────────────────────────────


class TestGateInstallPhase:
    """Gate's entry-point adds the systemd-availability preflight."""

    def test_systemd_unavailable_is_warning_not_failure(
        self, bare_cfg: SandboxConfig, capsys: pytest.CaptureFixture[str]
    ) -> None:
        """Hosts without user systemd warn-skip the phase — gate has no daemon fallback.

        The skip is intentional: the gate's inetd-style architecture has no
        managed-daemon counterpart yet, so callers (sickbay, preflight) are
        expected to detect ``mode="none"`` and degrade gracefully.  The
        message names the consequence so the operator isn't left guessing
        why a feature went missing.
        """
        from terok_sandbox.gate.lifecycle import GateServerManager

        with patch.object(GateServerManager, "is_systemd_available", return_value=False):
            assert run_gate_install_phase(bare_cfg) is True
        out = capsys.readouterr().out
        assert "WARN" in out
        # Consequence is named, not just "skipping".
        assert "git push channel" in out

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

    def test_systemexit_from_stop_is_swallowed(self) -> None:
        """``SystemExit`` is BaseException-derived — must be caught too (issue #310).

        Before this fix, a ``_systemctl.run`` raising ``SystemExit`` out
        of ``stop`` would escape the ``suppress(Exception)`` guard and
        block the subsequent ``install_systemd_units`` call, defeating
        the "best-effort, never block the install that follows"
        contract.
        """
        stop = MagicMock(side_effect=SystemExit("wedged unit"))
        uninstall = MagicMock()
        _stop_and_uninstall(stop, uninstall)  # must not raise
        # ``uninstall`` still runs — that's the whole point: the install
        # that follows expects a clean unit-file slate, and a wedged
        # stop must not prevent the uninstall sweep.
        uninstall.assert_called_once()

    def test_systemexit_from_uninstall_is_swallowed(self) -> None:
        """Symmetric coverage on the uninstall leg."""
        stop = MagicMock()
        uninstall = MagicMock(side_effect=SystemExit("wedged unit-file remove"))
        _stop_and_uninstall(stop, uninstall)  # must not raise
        stop.assert_called_once()


# ── Uninstall phases ─────────────────────────────────────────────────


class TestShieldUninstallPhase:
    """Shield uninstall: removes hooks from the canonical terok-owned dir."""

    def test_reports_ok(self, capsys: pytest.CaptureFixture[str]) -> None:
        with patch("terok_sandbox.integrations.shield.ShieldHooks.uninstall") as run:
            assert run_shield_uninstall_phase() is True
        run.assert_called_once_with()
        assert "removed" in capsys.readouterr().out

    def test_uninstall_raises_reports_fail(self, capsys: pytest.CaptureFixture[str]) -> None:
        with patch(
            "terok_sandbox.integrations.shield.ShieldHooks.uninstall",
            side_effect=RuntimeError("hook dir missing"),
        ):
            assert run_shield_uninstall_phase() is False
        out = capsys.readouterr().out
        assert "FAIL" in out and "hook dir missing" in out


class TestGateUninstallPhase:
    """Gate uninstall: stop daemon-mode process first, then remove units."""

    def test_daemon_mode_stopped_before_unit_removal(
        self, capsys: pytest.CaptureFixture[str]
    ) -> None:
        from terok_sandbox.gate.lifecycle import GateServerManager, GateServerStatus

        status = GateServerStatus(mode="daemon", running=True, port=9418)
        with (
            patch.object(GateServerManager, "get_status", return_value=status),
            patch.object(GateServerManager, "stop_daemon") as stop,
            patch.object(GateServerManager, "is_systemd_available", return_value=True),
            patch.object(GateServerManager, "uninstall_systemd_units") as un,
        ):
            assert run_gate_uninstall_phase(SandboxConfig(services_mode="socket")) is True
        stop.assert_called_once()
        un.assert_called_once()
        assert "removed" in capsys.readouterr().out

    def test_systemd_only_skips_daemon_stop(self, capsys: pytest.CaptureFixture[str]) -> None:
        """When only systemd is active, no daemon stop happens."""
        from terok_sandbox.gate.lifecycle import GateServerManager, GateServerStatus

        status = GateServerStatus(mode="systemd", running=True, port=9418)
        with (
            patch.object(GateServerManager, "get_status", return_value=status),
            patch.object(GateServerManager, "stop_daemon") as stop,
            patch.object(GateServerManager, "is_systemd_available", return_value=True),
            patch.object(GateServerManager, "uninstall_systemd_units") as un,
        ):
            assert run_gate_uninstall_phase(SandboxConfig(services_mode="socket")) is True
        stop.assert_not_called()
        un.assert_called_once()

    def test_no_systemd_skips_unit_removal(self, capsys: pytest.CaptureFixture[str]) -> None:
        """Hosts without systemd still report ok — nothing to uninstall is not a failure."""
        from terok_sandbox.gate.lifecycle import GateServerManager, GateServerStatus

        status = GateServerStatus(mode="none", running=False, port=None)
        with (
            patch.object(GateServerManager, "get_status", return_value=status),
            patch.object(GateServerManager, "is_systemd_available", return_value=False),
            patch.object(GateServerManager, "uninstall_systemd_units") as un,
        ):
            assert run_gate_uninstall_phase(SandboxConfig(services_mode="socket")) is True
        un.assert_not_called()

    def test_systemexit_during_removal_is_reported(
        self, capsys: pytest.CaptureFixture[str]
    ) -> None:
        from terok_sandbox.gate.lifecycle import GateServerManager, GateServerStatus

        status = GateServerStatus(mode="systemd", running=True, port=9418)
        with (
            patch.object(GateServerManager, "get_status", return_value=status),
            patch.object(GateServerManager, "is_systemd_available", return_value=True),
            patch.object(
                GateServerManager, "uninstall_systemd_units", side_effect=SystemExit("permission")
            ),
        ):
            assert run_gate_uninstall_phase(SandboxConfig(services_mode="socket")) is False
        out = capsys.readouterr().out
        assert "FAIL" in out and "permission" in out

    def test_generic_exception_adds_uninstall_prefix(
        self, capsys: pytest.CaptureFixture[str]
    ) -> None:
        from terok_sandbox.gate.lifecycle import GateServerManager, GateServerStatus

        status = GateServerStatus(mode="systemd", running=True, port=9418)
        with (
            patch.object(GateServerManager, "get_status", return_value=status),
            patch.object(GateServerManager, "is_systemd_available", return_value=True),
            patch.object(
                GateServerManager, "uninstall_systemd_units", side_effect=RuntimeError("fs busy")
            ),
        ):
            assert run_gate_uninstall_phase(SandboxConfig(services_mode="socket")) is False
        out = capsys.readouterr().out
        assert "FAIL" in out and "uninstall: fs busy" in out


class TestLegacyInstallCleanupPhase:
    """Idempotent sweep of pre-supervisor systemd units, unit files, and sockets.

    The phase is one-way: it removes hosts-side artefacts the new
    architecture doesn't manage any more.  Soft-fail throughout so a
    missing ``systemctl`` / missing unit / stale socket cannot abort
    the rest of the sweep.
    """

    def test_disables_every_legacy_unit_and_removes_global_socket(
        self, monkeypatch: pytest.MonkeyPatch, tmp_path, capsys: pytest.CaptureFixture[str]
    ) -> None:
        """Every legacy clearance + vault unit gets a best-effort disable+stop pass."""
        from terok_sandbox._setup import _LEGACY_SYSTEMD_UNITS

        # Set XDG_RUNTIME_DIR so the global shield-events socket gets
        # unlinked at a predictable path; place a stub file there so
        # the unlink path is exercised.
        runtime = tmp_path / "run"
        runtime.mkdir()
        legacy_sock = runtime / "terok-shield-events.sock"
        legacy_sock.write_text("stub")
        monkeypatch.setenv("XDG_RUNTIME_DIR", str(runtime))

        with patch("terok_sandbox._setup._systemctl.run_best_effort") as run:
            assert run_legacy_install_cleanup_phase() is True

        targets = {call.args[2] for call in run.call_args_list if call.args[0] == "disable"}
        for unit in _LEGACY_SYSTEMD_UNITS:
            assert unit in targets, f"{unit} should be disabled"

        # Stub socket is gone.
        assert not legacy_sock.exists()

        out = capsys.readouterr().out
        assert "Legacy install cleanup" in out
        assert "swept" in out

    def test_idempotent_when_nothing_to_clean(
        self, monkeypatch: pytest.MonkeyPatch, tmp_path, capsys: pytest.CaptureFixture[str]
    ) -> None:
        """A fresh host without pre-supervisor artefacts still reports ok."""

        runtime = tmp_path / "run"
        runtime.mkdir()
        monkeypatch.setenv("XDG_RUNTIME_DIR", str(runtime))

        with patch("terok_sandbox._setup._systemctl.run_best_effort"):
            assert run_legacy_install_cleanup_phase() is True
            # Re-run is also ok (idempotent).
            assert run_legacy_install_cleanup_phase() is True

        # Second invocation must not have raised over the still-absent socket.
        assert "Legacy install cleanup" in capsys.readouterr().out

    def test_missing_xdg_runtime_dir_does_not_crash(
        self, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
    ) -> None:
        """A missing ``XDG_RUNTIME_DIR`` skips the shield-events socket sweep cleanly."""

        monkeypatch.delenv("XDG_RUNTIME_DIR", raising=False)

        with patch("terok_sandbox._setup._systemctl.run_best_effort"):
            assert run_legacy_install_cleanup_phase() is True
        assert "swept" in capsys.readouterr().out
