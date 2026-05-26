# SPDX-FileCopyrightText: 2026 Jiri Vyskocil
# SPDX-License-Identifier: Apache-2.0

"""Tests for the sandbox-wide setup phase functions.

Covers the individual phases ``_handle_sandbox_setup`` wires together:
prereq reporting, shield install/uninstall, and the legacy install
cleanup sweep.  The aggregator orchestration itself is tested in
``test_setup_aggregator.py``.
"""

from __future__ import annotations

from unittest.mock import MagicMock, patch

import pytest

from terok_sandbox._setup import (
    SelinuxStatus,
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


# The gate install-phase tests that lived here covered the retired host
# gate systemd install.  The gate now lives in the per-container
# supervisor and has no host-side install phase, so there is nothing to
# exercise.


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


# The gate uninstall-phase tests that lived here covered the retired
# host gate systemd uninstall.  The gate has no host-side install, so
# there is no uninstall phase to exercise; the legacy sweep below
# removes any pre-supervisor gate units.


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
