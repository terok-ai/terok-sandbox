# SPDX-FileCopyrightText: 2026 Jiri Vyskocil
# SPDX-License-Identifier: Apache-2.0

"""Top-level ``sandbox setup`` and ``sandbox uninstall`` orchestration.

After the per-container-supervisor refactor (May 2026) the install
phases that remain are: legacy-install cleanup, shield hooks, the
credentials-DB encryption phase, the clearance soft-skip reporter, and
the supervisor OCI hook chain.  The git gate moved into the
per-container supervisor, so there is no host-side gate phase.  The
aggregator wires them together; the individual phase implementations
live in ``test_setup_phases.py``.
"""

from __future__ import annotations

from unittest.mock import patch

import pytest

from terok_sandbox.commands import (
    _handle_sandbox_setup,
    _handle_sandbox_uninstall,
    _handle_shield_uninstall,
)
from terok_sandbox.config import SandboxConfig

# ── Fixtures ──────────────────────────────────────────────────────────────


@pytest.fixture
def install_spies():
    """Replace every ``run_*_install_phase`` with a MagicMock so order is observable.

    Default return ``True`` so the aggregator walks the happy path
    unless a test overrides a phase.  Prereq reporting is stubbed out
    — it shells out for host binaries which would noisily poll the CI
    runner's PATH.
    """
    from terok_sandbox._util._selinux import SelinuxCheckResult, SelinuxStatus

    # Default prereq result: TCP-mode equivalent ("no SELinux policy needed").
    # Tests that exercise the SELinux POLICY_MISSING branch override this.
    _no_selinux_concern = SelinuxCheckResult(SelinuxStatus.NOT_APPLICABLE_TCP_MODE)
    with (
        patch(
            "terok_sandbox._setup.run_prereq_report",
            return_value=_no_selinux_concern,
        ) as prereq,
        patch(
            "terok_sandbox._setup.run_legacy_install_cleanup_phase",
            return_value=True,
        ) as legacy,
        patch("terok_sandbox._setup.run_shield_install_phase", return_value=True) as shield,
        # The credentials phase is the one non-``run_*_install_phase``
        # step the aggregator drives in-line.  It needs stubbing for
        # the same reason as the others — without it, the real
        # passphrase-tier resolution runs and (under proper $HOME
        # isolation) exits via the non-TTY tier-choice prompt.
        patch(
            "terok_sandbox.commands.credentials._run_credentials_setup_phase", return_value=True
        ) as credentials,
        patch("terok_sandbox._setup.run_supervisor_install_phase", return_value=True) as supervisor,
    ):
        yield {
            "prereq": prereq,
            "legacy": legacy,
            "shield": shield,
            "credentials": credentials,
            "supervisor": supervisor,
        }


@pytest.fixture
def uninstall_spies():
    """Replace every ``run_*_uninstall_phase`` with a MagicMock so order is observable."""
    with (
        patch(
            "terok_sandbox._setup.run_supervisor_uninstall_phase", return_value=True
        ) as supervisor_uninstall,
        patch(
            "terok_sandbox._setup.run_shield_uninstall_phase", return_value=True
        ) as shield_uninstall,
        patch(
            "terok_sandbox._setup.run_legacy_install_cleanup_phase", return_value=True
        ) as legacy_cleanup,
    ):
        yield {
            "supervisor_uninstall": supervisor_uninstall,
            "shield_uninstall": shield_uninstall,
            "legacy_cleanup": legacy_cleanup,
        }


# ── Setup aggregator ──────────────────────────────────────────────────────


class TestSandboxSetup:
    """``sandbox setup`` orchestrates the install phases in fixed order."""

    def test_default_runs_all_phases_in_order(self, install_spies) -> None:
        from terok_sandbox._util._selinux import SelinuxCheckResult, SelinuxStatus

        order: list[str] = []
        no_selinux = SelinuxCheckResult(SelinuxStatus.NOT_APPLICABLE_TCP_MODE)
        install_spies["prereq"].side_effect = lambda _cfg: order.append("prereq") or no_selinux
        install_spies["legacy"].side_effect = lambda: order.append("legacy") or True
        install_spies["shield"].side_effect = lambda **_: order.append("shield") or True
        install_spies["credentials"].side_effect = lambda *_a, **_kw: (
            order.append("credentials") or True
        )
        install_spies["supervisor"].side_effect = lambda: order.append("supervisor") or True

        _handle_sandbox_setup()

        # Legacy cleanup runs first (so a leftover pre-supervisor unit
        # can't fight a fresh install).  Supervisor lands last (the OCI
        # hook should only fire once the rest of the stack is ready).
        assert order == [
            "prereq",
            "legacy",
            "shield",
            "credentials",
            "supervisor",
        ]

    @pytest.mark.parametrize(
        ("skip_kwarg", "skipped_spy"),
        [
            ("no_shield", "shield"),
            ("no_vault", "credentials"),  # --no-vault now skips the credentials phase
        ],
    )
    def test_opt_out_flag_skips_exactly_its_phase(
        self, install_spies, skip_kwarg: str, skipped_spy: str
    ) -> None:
        _handle_sandbox_setup(**{skip_kwarg: True})
        install_spies[skipped_spy].assert_not_called()
        # Every other install phase still fires.
        for key in ("shield", "credentials", "supervisor", "legacy"):
            if key != skipped_spy:
                assert install_spies[key].called, f"{key} should still run"

    def test_failing_phase_exits_nonzero_after_others_run(self, install_spies) -> None:
        """A credentials failure must not short-circuit the supervisor phase."""
        install_spies["credentials"].return_value = False
        with pytest.raises(SystemExit) as exc:
            _handle_sandbox_setup()
        assert exc.value.code == 1
        install_spies["supervisor"].assert_called_once()

    def test_happy_path_does_not_raise(self, install_spies) -> None:
        """Every phase reports ``ok=True`` → the aggregator returns normally."""
        _handle_sandbox_setup()  # no SystemExit expected

    def test_policy_missing_exits_5_with_actionable_hint(
        self, install_spies, capsys: pytest.CaptureFixture[str]
    ) -> None:
        """``POLICY_MISSING`` re-surfaces the install command + TCP alternative and exits 5."""
        from terok_sandbox._util._selinux import SelinuxCheckResult, SelinuxStatus

        install_spies["prereq"].return_value = SelinuxCheckResult(SelinuxStatus.POLICY_MISSING)

        with pytest.raises(SystemExit) as exc:
            _handle_sandbox_setup()
        # Exit code 5 ("manual host configuration needed") distinguishes
        # this from a phase failure (1) so the TUI can offer the
        # specific fix instead of a generic "setup failed" banner.
        assert exc.value.code == 5

        out = capsys.readouterr().out
        # Both the install command and the TCP-mode alternative land
        # at the end of output, each on its own line so the operator
        # can copy-paste either without bleed.
        assert "SELinux policy required" in out
        assert "install_policy.sh" in out
        assert 'services.mode = "tcp"' in out

    def test_policy_missing_skipped_when_phases_already_failed(self, install_spies) -> None:
        """A prior phase failure exits 1; the SELinux exit-5 path doesn't override that."""
        from terok_sandbox._util._selinux import SelinuxCheckResult, SelinuxStatus

        install_spies["prereq"].return_value = SelinuxCheckResult(SelinuxStatus.POLICY_MISSING)
        install_spies["credentials"].return_value = False

        with pytest.raises(SystemExit) as exc:
            _handle_sandbox_setup()
        # Phase failures take precedence — exit 1, not 5.  The SELinux
        # hint is still printed (operator-facing) but the contract for
        # the exit code is "the first thing that went wrong".
        assert exc.value.code == 1


# ── Uninstall aggregator ──────────────────────────────────────────────────


class TestSandboxUninstall:
    """``sandbox uninstall`` runs the remaining phases in reverse install order."""

    def test_default_uninstalls_all_phases(self, uninstall_spies) -> None:
        cfg = SandboxConfig()
        _handle_sandbox_uninstall(cfg=cfg)
        uninstall_spies["supervisor_uninstall"].assert_called_once_with()
        uninstall_spies["shield_uninstall"].assert_called_once_with()
        uninstall_spies["legacy_cleanup"].assert_called_once_with()

    def test_phases_run_in_reverse_install_order(self, uninstall_spies) -> None:
        """Supervisor first (its hook would outlive the rest), legacy cleanup last."""
        order: list[str] = []
        uninstall_spies["supervisor_uninstall"].side_effect = lambda: (
            order.append("supervisor") or True
        )
        uninstall_spies["shield_uninstall"].side_effect = lambda: order.append("shield") or True
        uninstall_spies["legacy_cleanup"].side_effect = lambda: order.append("legacy") or True

        _handle_sandbox_uninstall()

        assert order == ["supervisor", "shield", "legacy"]

    def test_no_shield_skips_exactly_its_phase(self, uninstall_spies) -> None:
        _handle_sandbox_uninstall(no_shield=True)
        uninstall_spies["shield_uninstall"].assert_not_called()
        for key in uninstall_spies:
            if key != "shield_uninstall":
                assert uninstall_spies[key].called, f"{key} should still run"

    def test_failing_phase_does_not_abort_subsequent_phases(self, uninstall_spies) -> None:
        """A shield uninstall that returns ``False`` must not skip the legacy cleanup."""
        uninstall_spies["shield_uninstall"].return_value = False
        with pytest.raises(SystemExit):
            _handle_sandbox_uninstall()
        # Supervisor ran before; legacy cleanup must still run after the failure.
        uninstall_spies["supervisor_uninstall"].assert_called_once()
        uninstall_spies["legacy_cleanup"].assert_called_once()

    def test_all_phases_succeeding_does_not_exit_nonzero(self, uninstall_spies) -> None:
        """Happy path is not wrapped in an exit-1 just because of the try/except."""
        _handle_sandbox_uninstall()


# The gate install/uninstall handler tests that lived here covered the
# retired host gate daemon.  The gate now lives in the per-container
# supervisor; ``gate path`` is the only remaining verb (covered in
# ``test_cli_commands.py``).


# ── Shield uninstall CLI flag validation ─────────────────────────────────


class TestHandleShieldUninstall:
    """The ``shield uninstall-hooks`` handler delegates to ``ShieldHooks.uninstall``."""

    def test_delegates_to_library(self, capsys: pytest.CaptureFixture[str]) -> None:
        with patch("terok_sandbox.integrations.shield.ShieldHooks.uninstall") as mock_run:
            _handle_shield_uninstall()
        mock_run.assert_called_once_with()
        assert "removed" in capsys.readouterr().out
