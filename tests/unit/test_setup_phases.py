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
    print_selinux_install_hint,
    run_legacy_install_cleanup_phase,
    run_prereq_report,
    run_shield_install_phase,
    run_shield_uninstall_phase,
    run_supervisor_install_phase,
    run_supervisor_uninstall_phase,
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

    def test_reports_catatonit_path_when_found(
        self,
        bare_cfg: SandboxConfig,
        capsys: pytest.CaptureFixture[str],
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        monkeypatch.setattr("terok_sandbox._setup.shutil.which", lambda _n: None)
        with (
            patch("terok_sandbox.integrations.shield.check_firewall_binaries", return_value=()),
            patch(
                "terok_sandbox.runtime.podman.find_init_binary",
                return_value="/usr/libexec/podman/catatonit",
            ),
            patch(
                "terok_sandbox._setup.check_selinux_status",
                return_value=MagicMock(status=SelinuxStatus.NOT_APPLICABLE_TCP_MODE),
            ),
        ):
            run_prereq_report(bare_cfg)
        out = capsys.readouterr().out
        assert "catatonit" in out
        assert "/usr/libexec/podman/catatonit" in out

    def test_warns_on_missing_catatonit(
        self,
        bare_cfg: SandboxConfig,
        capsys: pytest.CaptureFixture[str],
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        """Missing catatonit warns about degraded stops — setup still proceeds."""
        monkeypatch.setattr("terok_sandbox._setup.shutil.which", lambda _n: None)
        with (
            patch("terok_sandbox.integrations.shield.check_firewall_binaries", return_value=()),
            patch("terok_sandbox.runtime.podman.find_init_binary", return_value=None),
            patch(
                "terok_sandbox._setup.check_selinux_status",
                return_value=MagicMock(status=SelinuxStatus.NOT_APPLICABLE_TCP_MODE),
            ),
        ):
            run_prereq_report(bare_cfg)
        out = capsys.readouterr().out
        assert "catatonit" in out
        assert "degrade" in out

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
            (SelinuxStatus.POLICY_OUTDATED, "rebuild:"),
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

    def test_unlink_oserror_is_swallowed(
        self, monkeypatch: pytest.MonkeyPatch, tmp_path, capsys: pytest.CaptureFixture[str]
    ) -> None:
        """A permission/IO error while unlinking a stale artefact never aborts the sweep.

        Every ``_unlink_legacy_*`` helper guards its ``unlink`` with a
        soft ``except OSError`` — a read-only runtime dir (or a racing
        removal) must not stop the phase from finishing.
        """
        runtime = tmp_path / "run"
        runtime.mkdir()
        monkeypatch.setenv("XDG_RUNTIME_DIR", str(runtime))
        monkeypatch.setenv("XDG_DATA_HOME", str(tmp_path / "share"))

        with (
            patch("terok_sandbox._setup._systemctl.run_best_effort"),
            patch("pathlib.Path.unlink", side_effect=OSError("planted")),
        ):
            assert run_legacy_install_cleanup_phase() is True
        assert "swept" in capsys.readouterr().out


# ── Supervisor install / uninstall phases ─────────────────────────────


class TestSupervisorInstallPhase:
    """The supervisor-hooks install/uninstall phases frame the installer.

    Each delegates to ``supervisor.install`` and reports a stage line;
    an installer exception is caught and surfaced as a ``fail`` → False
    so the aggregator keeps running the remaining phases.
    """

    def test_install_reports_ok(self, capsys: pytest.CaptureFixture[str]) -> None:
        with patch("terok_sandbox.supervisor.install.install_supervisor_hooks") as install:
            assert run_supervisor_install_phase() is True
            install.assert_called_once_with()
        out = capsys.readouterr().out
        assert "Supervisor hooks" in out
        assert "OCI hook" in out

    def test_install_failure_reports_fail(self, capsys: pytest.CaptureFixture[str]) -> None:
        with patch(
            "terok_sandbox.supervisor.install.install_supervisor_hooks",
            side_effect=RuntimeError("no entry point"),
        ):
            assert run_supervisor_install_phase() is False
        assert "no entry point" in capsys.readouterr().out

    def test_uninstall_reports_ok(self, capsys: pytest.CaptureFixture[str]) -> None:
        with patch("terok_sandbox.supervisor.install.uninstall_supervisor_hooks") as uninstall:
            assert run_supervisor_uninstall_phase() is True
            uninstall.assert_called_once_with()
        assert "removed" in capsys.readouterr().out

    def test_uninstall_failure_reports_fail(self, capsys: pytest.CaptureFixture[str]) -> None:
        with patch(
            "terok_sandbox.supervisor.install.uninstall_supervisor_hooks",
            side_effect=OSError("permission denied"),
        ):
            assert run_supervisor_uninstall_phase() is False
        assert "permission denied" in capsys.readouterr().out


# ── Legacy sweep helpers — exercised against real tmp files ────────────


class TestSweepLegacyUnitFiles:
    """``_sweep_legacy_unit_files`` unlinks stray glob-matched user units."""

    def test_unlinks_matching_units_and_disables_them(
        self, tmp_path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Files matching the legacy globs are disabled then unlinked; others stay."""
        from terok_sandbox._setup import _sweep_legacy_unit_files

        unit_dir = tmp_path / "units"
        unit_dir.mkdir()
        # Two match the legacy globs; one is an unrelated user unit.
        matched = unit_dir / "terok-clearance-hub.service"
        matched.write_text("[Unit]")
        matched_vault = unit_dir / "terok-vault-socket.service"
        matched_vault.write_text("[Unit]")
        keep = unit_dir / "my-other.service"
        keep.write_text("[Unit]")

        monkeypatch.setattr("terok_sandbox._util.systemd_user_unit_dir", lambda: unit_dir)
        with patch("terok_sandbox._setup._systemctl.run_best_effort") as run:
            _sweep_legacy_unit_files()

        assert not matched.exists()
        assert not matched_vault.exists()
        assert keep.exists()
        disabled = {c.args[2] for c in run.call_args_list if c.args[0] == "disable"}
        assert "terok-clearance-hub.service" in disabled
        assert "terok-vault-socket.service" in disabled

    def test_unlink_oserror_does_not_abort_remaining_sweep(
        self, tmp_path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """A failed unlink (e.g. EBUSY) is swallowed so the rest of the sweep runs.

        Both files match the legacy globs; the first unlink raises OSError —
        the sweep must still attempt (and disable) the second.
        """
        from terok_sandbox._setup import _sweep_legacy_unit_files

        unit_dir = tmp_path / "units"
        unit_dir.mkdir()
        (unit_dir / "terok-vault.service").write_text("[Unit]")
        (unit_dir / "terok-vault.socket").write_text("[Unit]")

        monkeypatch.setattr("terok_sandbox._util.systemd_user_unit_dir", lambda: unit_dir)
        with (
            patch("terok_sandbox._setup._systemctl.run_best_effort") as run,
            patch("pathlib.Path.unlink", side_effect=OSError("device or resource busy")),
        ):
            _sweep_legacy_unit_files()  # must not raise

        # Both matched units still get a disable pass despite unlink raising.
        disabled = {c.args[2] for c in run.call_args_list if c.args[0] == "disable"}
        assert "terok-vault.service" in disabled
        assert "terok-vault.socket" in disabled

    def test_missing_unit_dir_is_a_noop(self, tmp_path, monkeypatch: pytest.MonkeyPatch) -> None:
        """A non-existent systemd user unit dir → nothing to sweep, no crash."""
        from terok_sandbox._setup import _sweep_legacy_unit_files

        monkeypatch.setattr(
            "terok_sandbox._util.systemd_user_unit_dir", lambda: tmp_path / "absent"
        )
        with patch("terok_sandbox._setup._systemctl.run_best_effort") as run:
            _sweep_legacy_unit_files()  # must not raise
        run.assert_not_called()

    def test_unit_dir_resolution_systemexit_is_swallowed(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """A ``SystemExit`` from ``systemd_user_unit_dir`` (no XDG home) is caught."""
        from terok_sandbox._setup import _sweep_legacy_unit_files

        def _boom() -> object:
            raise SystemExit("no config home")

        monkeypatch.setattr("terok_sandbox._util.systemd_user_unit_dir", _boom)
        _sweep_legacy_unit_files()  # must not raise


class TestUnlinkLegacyRuntimeSockets:
    """``_unlink_legacy_runtime_sockets`` sweeps host-global daemon sockets."""

    def test_removes_legacy_sockets_keeps_passphrase(
        self, tmp_path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """The three pre-supervisor socket names go; ``vault.passphrase`` stays."""
        from terok_sandbox._setup import _unlink_legacy_runtime_sockets

        rt = tmp_path / "rt"
        rt.mkdir()
        for name in ("vault.sock", "ssh-agent.sock", "gate-server.sock"):
            (rt / name).write_text("stale")
        passphrase = rt / "vault.passphrase"
        passphrase.write_text("live-secret")

        monkeypatch.setattr("terok_sandbox.paths.runtime_root", lambda: rt)
        _unlink_legacy_runtime_sockets()

        assert not (rt / "vault.sock").exists()
        assert not (rt / "ssh-agent.sock").exists()
        assert not (rt / "gate-server.sock").exists()
        # Live session-tier credential must survive.
        assert passphrase.exists()

    def test_runtime_root_oserror_is_swallowed(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """A ``runtime_root`` OSError (no resolvable runtime dir) → soft no-op."""
        from terok_sandbox._setup import _unlink_legacy_runtime_sockets

        def _boom() -> object:
            raise OSError("no runtime dir")

        monkeypatch.setattr("terok_sandbox.paths.runtime_root", _boom)
        _unlink_legacy_runtime_sockets()  # must not raise


class TestUnlinkLegacyXdgDataFiles:
    """``_unlink_legacy_xdg_data_files`` removes the pre-paths.root shield copy."""

    def test_removes_stale_reader_and_prunes_empty_dirs(
        self, tmp_path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """The orphaned ``nflog-reader.py`` is removed; empty terok dirs pruned."""
        from terok_sandbox._setup import _unlink_legacy_xdg_data_files

        data_home = tmp_path / "data"
        shield_root = data_home / "terok" / "shield"
        shield_root.mkdir(parents=True)
        (shield_root / "nflog-reader.py").write_text("# stale")
        monkeypatch.setenv("XDG_DATA_HOME", str(data_home))

        _unlink_legacy_xdg_data_files()

        assert not shield_root.exists()
        # The now-empty ``terok`` parent is pruned too.
        assert not (data_home / "terok").exists()

    def test_leaves_non_empty_terok_parent_intact(
        self, tmp_path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """A ``terok`` dir holding other files is not rmdir'd (rmdir fails soft)."""
        from terok_sandbox._setup import _unlink_legacy_xdg_data_files

        data_home = tmp_path / "data"
        terok_dir = data_home / "terok"
        shield_root = terok_dir / "shield"
        shield_root.mkdir(parents=True)
        (shield_root / "nflog-reader.py").write_text("# stale")
        sibling = terok_dir / "other"
        sibling.mkdir()
        monkeypatch.setenv("XDG_DATA_HOME", str(data_home))

        _unlink_legacy_xdg_data_files()

        assert not shield_root.exists()
        assert terok_dir.exists()  # non-empty → preserved

    def test_absent_shield_dir_is_a_noop(self, tmp_path, monkeypatch: pytest.MonkeyPatch) -> None:
        """No legacy shield dir → nothing happens, no crash."""
        from terok_sandbox._setup import _unlink_legacy_xdg_data_files

        monkeypatch.setenv("XDG_DATA_HOME", str(tmp_path / "empty"))
        _unlink_legacy_xdg_data_files()  # must not raise


class TestUnlinkLegacyShieldGlobalHooks:
    """``_unlink_legacy_shield_global_hooks`` sweeps the master-branch hooks.d copy."""

    def test_removes_only_terok_owned_files(
        self, tmp_path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Terok's own scripts + descriptors go; operator-owned siblings stay."""
        from terok_sandbox._setup import _unlink_legacy_shield_global_hooks

        hooks_d = tmp_path / ".local" / "share" / "containers" / "oci" / "hooks.d"
        hooks_d.mkdir(parents=True)
        terok_files = (
            "_oci_state.py",
            "terok-shield-hook",
            "terok-shield-createRuntime.json",
            "terok-shield-poststop.json",
        )
        for name in terok_files:
            (hooks_d / name).write_text("terok")
        foreign = hooks_d / "someone-elses-hook.json"
        foreign.write_text("not ours")

        monkeypatch.setattr("pathlib.Path.home", lambda: tmp_path)
        _unlink_legacy_shield_global_hooks()

        for name in terok_files:
            assert not (hooks_d / name).exists(), f"{name} should be swept"
        assert foreign.exists()  # operator-owned, left alone

    def test_absent_hooks_dir_is_a_noop(self, tmp_path, monkeypatch: pytest.MonkeyPatch) -> None:
        """No legacy hooks.d → unlink(missing_ok) makes it a clean no-op."""
        from terok_sandbox._setup import _unlink_legacy_shield_global_hooks

        monkeypatch.setattr("pathlib.Path.home", lambda: tmp_path)
        _unlink_legacy_shield_global_hooks()  # must not raise


class TestSelinuxOutdatedReporting:
    """The end-of-setup banner explains a stale (outdated) terok_socket policy.

    The stage-line branch is covered by ``TestPrereqReport`` above; this
    pins the supervisor-era ``print_selinux_install_hint`` banner added
    with the Fedora 44 fixes.
    """

    def test_install_hint_banner_for_outdated_policy(
        self, capsys: pytest.CaptureFixture[str]
    ) -> None:
        """The banner names the stale policy and the rebuild command."""
        print_selinux_install_hint(MagicMock(status=SelinuxStatus.POLICY_OUTDATED))
        out = capsys.readouterr().out
        assert "predates the per-container" in out
        assert "Rebuild the policy" in out
