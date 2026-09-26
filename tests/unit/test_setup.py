# SPDX-FileCopyrightText: 2026 Jiri Vyskocil
# SPDX-License-Identifier: Apache-2.0

"""Owned sandbox receipts compose downward and cannot certify partial setup."""

import json
from dataclasses import replace
from unittest.mock import Mock

import pytest
from terok_util import (
    SetupCheck,
    SetupDowngradeError,
    SetupRequiredError,
    SetupStatus,
    setup_status,
)

from terok_sandbox import setup
from terok_sandbox._util._apparmor import AppArmorCheckResult, AppArmorStatus
from terok_sandbox._util._selinux import SelinuxCheckResult, SelinuxStatus
from terok_sandbox.commands.sandbox import _handle_sandbox_setup
from terok_sandbox.config import SandboxConfig
from terok_sandbox.supervisor import install


@pytest.fixture
def installed(tmp_path, monkeypatch):
    """Install genuine supervisor artifacts around an isolated credentials fixture."""
    cfg = SandboxConfig(state_dir=tmp_path / "sandbox", vault_dir=tmp_path / "vault")
    monkeypatch.setattr(setup, "version", lambda _: "0.6.0")
    monkeypatch.setattr(install, "ensure_user_hooks_dir_configured", lambda _: None)
    monkeypatch.setattr(install, "user_hooks_dir_configured", lambda _: True)
    companion = tmp_path / "bin" / "terok-sandbox"
    companion.parent.mkdir()
    companion.write_text("#!/bin/sh\nexit 0\n")
    companion.chmod(0o755)
    monkeypatch.setattr(install, "_resolve_sandbox_argv", lambda: [str(companion)])
    monkeypatch.setattr(setup.ShieldHooks, "check_setup", lambda **_kw: ())
    monkeypatch.setattr(setup, "find_host_tool", lambda name: str(tmp_path / "bin" / name))
    install.install_supervisor_hooks(root=cfg.state_dir)
    cfg.db_path.parent.mkdir(parents=True, exist_ok=True)
    cfg.db_path.write_bytes(b"encrypted database fixture")
    setup.setup_receipt(cfg).write()
    return cfg


def test_receipt_records_only_its_owner(installed):
    """Sandbox never inspects an executor, terok, or clearance package version."""
    assert setup_status(setup.check_setup(installed)) == SetupStatus.READY
    assert setup.setup_receipt(installed).owner == "terok-sandbox"


def test_legacy_aggregate_stamp_does_not_certify(installed):
    """A clean upgrade requires real setup, not migration of an aggregate marker."""
    setup.setup_receipt(installed).clear()
    from terok_sandbox.paths import namespace_state_dir

    legacy = namespace_state_dir() / "setup.stamp"
    legacy.parent.mkdir(parents=True, exist_ok=True)
    legacy.write_text('{"packages": {"terok": "99"}}')
    assert setup_status(setup.check_setup(installed)) == SetupStatus.MISSING


def test_changed_python_requires_setup(installed, monkeypatch):
    """Replacing the setup-bound Python invalidates this owner's receipt."""
    monkeypatch.setattr(setup, "python_identity", lambda: "different interpreter")
    assert setup_status(setup.check_setup(installed)) == SetupStatus.STALE


@pytest.mark.parametrize("artifact", ["hooks/supervisor_hook.py", "supervisor_wrapper.py"])
def test_missing_required_artifact_invalidates_readiness(installed, artifact):
    """A present receipt is not sufficient when required hook files disappear."""
    (installed.state_dir / artifact).unlink()
    assert setup_status(setup.check_setup(installed)) != SetupStatus.READY


def test_missing_credentials_are_not_certified(installed):
    """Skipping credential provisioning cannot hide an absent vault."""
    installed.db_path.unlink()
    assert any(
        c.component == "credentials" and c.status == SetupStatus.MISSING
        for c in setup.check_setup(installed)
    )


def test_disabled_shield_does_not_require_child_install(installed, monkeypatch):
    """An explicitly disabled optional component is not missing required setup."""
    child = Mock(return_value=(SetupCheck("terok-shield", "receipt", SetupStatus.MISSING),))
    monkeypatch.setattr(setup.ShieldHooks, "check_setup", child)
    assert (
        setup_status(setup.check_setup(replace(installed, shield_disabled=True)))
        == SetupStatus.READY
    )


def test_children_are_public_results_not_file_reads(installed, monkeypatch):
    """The parent preserves the lower owner's precise readiness result."""
    child = SetupCheck("terok-shield", "receipt", SetupStatus.DOWNGRADE, "newer shield")
    monkeypatch.setattr(setup.ShieldHooks, "check_setup", lambda **_kw: (child,))
    assert child in setup.check_setup(installed)


@pytest.fixture
def phases(monkeypatch):
    """Keep setup phases observable without touching host services."""
    from terok_sandbox import _setup

    calls = {}
    for name in (
        "run_legacy_install_cleanup_phase",
        "run_shield_install_phase",
        "run_supervisor_install_phase",
    ):
        calls[name] = Mock(return_value=True)
        monkeypatch.setattr(_setup, name, calls[name])
    monkeypatch.setattr(
        _setup,
        "run_prereq_report",
        lambda _: (
            SelinuxCheckResult(SelinuxStatus.NOT_APPLICABLE_TCP_MODE),
            AppArmorCheckResult(AppArmorStatus.NOT_APPLICABLE),
        ),
    )
    monkeypatch.setattr(
        "terok_sandbox.commands.credentials._post_setup_recovery_hint", lambda _: None
    )
    return calls


def test_downgrade_preflight_precedes_any_write(installed, phases, monkeypatch):
    """A newer child protects all existing artifacts and the parent's receipt."""
    before = setup.setup_receipt(installed).path.read_bytes()
    child = SetupCheck("terok-shield", "receipt", SetupStatus.DOWNGRADE)
    monkeypatch.setattr(setup.ShieldHooks, "check_setup", lambda **_kw: (child,))
    with pytest.raises(SetupDowngradeError):
        _handle_sandbox_setup(cfg=installed, no_vault=True)
    assert setup.setup_receipt(installed).path.read_bytes() == before
    assert not any(spy.called for spy in phases.values())


def test_failed_phase_removes_own_receipt(installed, phases):
    """A failed update cannot leave an earlier success receipt behind."""
    phases["run_supervisor_install_phase"].return_value = False
    with pytest.raises(SystemExit):
        _handle_sandbox_setup(cfg=installed, no_vault=True)
    assert not setup.setup_receipt(installed).path.exists()


def test_partial_setup_cannot_certify_missing_credentials(installed, phases):
    """All attempted phases can succeed while skipped required work is still missing."""
    installed.db_path.unlink()
    with pytest.raises(SetupRequiredError, match="credentials"):
        _handle_sandbox_setup(cfg=installed, no_vault=True)
    assert not setup.setup_receipt(installed).path.exists()


def test_skipped_existing_credentials_can_be_certified(installed, phases):
    """Skipping a phase is safe when its actual required artifacts already exist."""
    _handle_sandbox_setup(cfg=installed, no_vault=True)
    assert setup_status(setup.check_setup(installed)) == SetupStatus.READY


def test_start_checks_setup_before_runtime_changes(installed, monkeypatch):
    """A live setup fault must not create mount points or start the container."""
    from terok_sandbox.sandbox import Sandbox

    check = Mock(return_value=(SetupCheck("terok-sandbox", "python", SetupStatus.MISSING),))
    monkeypatch.setattr("terok_sandbox.sandbox.check_setup", check)
    runtime = Mock()
    sandbox = Sandbox(installed, runtime=runtime)
    with pytest.raises(SetupRequiredError):
        sandbox.start("task")
    runtime.container.assert_not_called()
    check.assert_called_once_with(sandbox.config, live=True)


def test_disabled_child_still_refuses_downgrade(installed, phases, monkeypatch):
    """Disabling an optional service does not authorize downgrading its install."""
    child = SetupCheck("terok-shield", "receipt", SetupStatus.DOWNGRADE)
    monkeypatch.setattr(setup.ShieldHooks, "check_setup", lambda **_kw: (child,))
    with pytest.raises(SetupDowngradeError):
        _handle_sandbox_setup(cfg=replace(installed, shield_disabled=True), no_vault=True)
    assert not any(spy.called for spy in phases.values())


def test_unregistered_hooks_require_setup(installed, monkeypatch):
    """A receipt cannot certify hooks that Podman will no longer dispatch."""
    monkeypatch.setattr(install, "user_hooks_dir_configured", lambda _: False)
    assert setup_status(setup.check_setup(installed)) == SetupStatus.STALE


def test_shared_lookup_change_requires_setup(installed, monkeypatch):
    """A utility upgrade that changes copied source invalidates the hook install."""
    monkeypatch.setattr(setup, "host_tools_source", lambda: "different implementation")
    assert setup_status(setup.check_setup(installed)) == SetupStatus.STALE


def test_compose_checks_setup_before_writing_state(installed, monkeypatch):
    """Standalone prepare/run uses the same pre-mutation setup boundary."""
    from terok_sandbox import launch

    check = Mock(return_value=(SetupCheck("terok-sandbox", "python", SetupStatus.MISSING),))
    monkeypatch.setattr(launch, "check_setup", check)
    with pytest.raises(SetupRequiredError):
        launch.compose("task", cfg=installed, shield=True, gate=False, broker=False, scope=None)
    assert not launch.run_state_dir(installed, "task").exists()


def test_plaintext_credentials_are_not_ready(installed):
    """The cheap readiness probe reads only the header; never opens SQLite for writes."""
    installed.db_path.write_bytes(b"SQLite format 3\x00" + bytes(100))
    assert setup_status(setup.check_setup(installed)) == SetupStatus.MISSING


def test_missing_live_tool_fails_before_mutation(installed, phases, monkeypatch):
    """Setup doesn't begin cleanup when the initiating PATH lacks required tools."""
    before = setup.setup_receipt(installed).path.read_bytes()
    monkeypatch.setattr(setup, "find_host_tool", lambda _: None)
    with pytest.raises(SetupRequiredError, match="podman"):
        _handle_sandbox_setup(cfg=installed, no_vault=True)
    assert not any(spy.called for spy in phases.values())
    assert setup.setup_receipt(installed).path.read_bytes() == before


def test_changed_hook_binding_requires_setup(installed):
    """Even unchanged package versions cannot certify a moved bootstrap binding."""
    descriptor = installed.state_dir / "hooks" / "terok-sandbox-supervisor-createRuntime.json"
    payload = json.loads(descriptor.read_text())
    payload["hook"]["path"] = str(installed.state_dir / "old-python")
    descriptor.write_text(json.dumps(payload))
    assert setup_status(setup.check_setup(installed)) == SetupStatus.STALE


def test_live_bootstrap_failure_is_typed_readiness(installed, monkeypatch):
    """An interpreter that cannot execute is rejected before a managed task changes."""
    monkeypatch.setattr(install.subprocess, "run", Mock(side_effect=OSError("bootstrap vanished")))
    checks = setup.check_setup(installed, live=True)
    assert any(
        check.status == SetupStatus.STALE and "bootstrap vanished" in check.diagnostic
        for check in checks
    )


def test_fresh_setup_creates_the_database_it_certifies(installed, phases, monkeypatch):
    """Real credential provisioning must produce the artifact required by readiness."""
    from terok_sandbox.commands import credentials
    from terok_sandbox.vault.store.db import CredentialDB
    from terok_sandbox.vault.store.tiers import PassphraseTier

    installed.db_path.unlink()
    monkeypatch.setattr(
        credentials,
        "_resolve_existing",
        lambda _: ("test-passphrase", PassphraseTier.PASSPHRASE_COMMAND),
    )
    _handle_sandbox_setup(cfg=installed)
    assert setup_status(setup.check_setup(installed)) == SetupStatus.READY
    CredentialDB(installed.db_path, passphrase="test-passphrase").close()
