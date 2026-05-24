# SPDX-FileCopyrightText: 2025 Jiri Vyskocil
# SPDX-License-Identifier: Apache-2.0

"""Tests for the terok-shield adapter (``terok_sandbox.integrations.shield``)."""

from __future__ import annotations

from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest
from terok_shield import (
    EnvironmentCheck,
    NftNotFoundError,
    Shield,
    ShieldMode,
    ShieldNeedsSetup,
    ShieldState,
)

from terok_sandbox.config import SandboxConfig
from terok_sandbox.integrations.shield import (
    _BYPASS_WARNING,
    ShieldHooks,
    ShieldManager,
    check_environment,
)
from tests.constants import (
    GATE_PORT,
    MOCK_BASE,
    MOCK_CONFIG_ROOT,
    MOCK_TASK_DIR,
)

CUSTOM_GATE_PORT = GATE_PORT + 1


def make_mock_shield(
    *,
    shield_state: ShieldState = ShieldState.UP,
    pre_start_args: list[str] | None = None,
) -> MagicMock:
    """Create a mock ``Shield`` instance with useful defaults."""
    mock_shield = MagicMock(spec=Shield)
    mock_shield.state.return_value = shield_state
    mock_shield.pre_start.return_value = (
        ["--network", "hook-net"] if pre_start_args is None else pre_start_args
    )
    return mock_shield


# ── ShieldManager.shield (the lazy Shield builder) ───────────────────────


@pytest.mark.parametrize(
    ("cfg_kwargs", "expected_profiles", "expected_port", "audit_enabled"),
    [
        pytest.param(
            {"gate_port": GATE_PORT, "token_broker_port": 18731, "ssh_signer_port": 18732},
            ("dev-standard",),
            GATE_PORT,
            True,
            id="defaults",
        ),
        pytest.param(
            {
                "shield_profiles": ("custom-a", "custom-b"),
                "shield_audit": False,
                "gate_port": CUSTOM_GATE_PORT,
                "token_broker_port": 18741,
                "ssh_signer_port": 18742,
            },
            ("custom-a", "custom-b"),
            CUSTOM_GATE_PORT,
            False,
            id="custom-values",
        ),
        pytest.param(
            {
                "shield_profiles": ("single-profile",),
                "gate_port": GATE_PORT,
                "token_broker_port": 18731,
                "ssh_signer_port": 18732,
            },
            ("single-profile",),
            GATE_PORT,
            True,
            id="single-profile",
        ),
    ],
)
def test_shield_property_maps_config_to_shield_config(
    cfg_kwargs: dict[str, object],
    expected_profiles: tuple[str, ...],
    expected_port: int,
    audit_enabled: bool,
) -> None:
    """SandboxConfig values are translated into the per-task ``ShieldConfig``."""
    cfg = SandboxConfig(**cfg_kwargs)
    with (
        patch("terok_shield.run.SubprocessRunner", autospec=True),
        patch("terok_sandbox.paths.namespace_config_root", return_value=MOCK_CONFIG_ROOT),
    ):
        shield = ShieldManager(MOCK_TASK_DIR, cfg=cfg).shield

    assert isinstance(shield, Shield)
    config = shield.config
    assert config.mode == ShieldMode.HOOK
    assert config.default_profiles == expected_profiles
    assert config.loopback_ports == (expected_port, cfg.token_broker_port, cfg.ssh_signer_port)
    assert config.audit_enabled is audit_enabled
    assert config.state_dir == MOCK_TASK_DIR / "shield"
    assert config.profiles_dir == MOCK_CONFIG_ROOT / "shield" / "profiles"


def test_shield_property_resolves_ports_for_auto_allocated_config(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A tcp-mode cfg with unset ports still produces a non-empty ``loopback_ports``.

    Regression guard for the post-#315 bug where Shield construction
    read the raw ``cfg.gate_port`` / ``token_broker_port`` /
    ``ssh_signer_port`` fields without calling
    [`SandboxConfig.with_resolved_ports`][terok_sandbox.SandboxConfig.with_resolved_ports]
    first.  Side-effect-free ``SandboxConfig`` construction leaves
    those fields ``None`` for auto-allocated configs, which silently
    produced an empty ``loopback_ports`` tuple — and a bypass ruleset
    with no ``tcp dport <p> ip daddr 169.254.1.2 accept`` rules, so
    every container→host loopback access fell through to the
    private-range reject after ``shield down``.
    """
    monkeypatch.setattr(
        "terok_sandbox.port_registry._read_installed_ports", lambda: {}, raising=False
    )

    cfg = SandboxConfig(state_dir=MOCK_BASE / "state-tcp-auto", services_mode="tcp")
    assert cfg.gate_port is None
    assert cfg.token_broker_port is None
    assert cfg.ssh_signer_port is None

    with (
        patch("terok_shield.run.SubprocessRunner", autospec=True),
        patch("terok_sandbox.paths.namespace_config_root", return_value=MOCK_CONFIG_ROOT),
    ):
        shield = ShieldManager(MOCK_TASK_DIR, cfg=cfg).shield

    assert len(shield.config.loopback_ports) == 3
    assert all(isinstance(p, int) for p in shield.config.loopback_ports)
    assert len(set(shield.config.loopback_ports)) == 3  # gate, broker, signer all distinct


def test_shield_property_socket_mode_skips_port_resolution(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Socket-mode configs short-circuit ``with_resolved_ports`` and emit no loopback rules."""
    monkeypatch.setattr(
        "terok_sandbox.port_registry._read_installed_ports", lambda: {}, raising=False
    )
    cfg = SandboxConfig(state_dir=MOCK_BASE / "state-socket", services_mode="socket")

    with (
        patch("terok_shield.run.SubprocessRunner", autospec=True),
        patch("terok_sandbox.paths.namespace_config_root", return_value=MOCK_CONFIG_ROOT),
    ):
        shield = ShieldManager(MOCK_TASK_DIR, cfg=cfg).shield

    assert shield.config.loopback_ports == ()


def test_shield_property_is_cached() -> None:
    """The shield instance is built once per manager — repeated reads return the same object."""
    cfg = SandboxConfig(gate_port=GATE_PORT, token_broker_port=18731, ssh_signer_port=18732)
    with (
        patch("terok_shield.run.SubprocessRunner", autospec=True),
        patch("terok_sandbox.paths.namespace_config_root", return_value=MOCK_CONFIG_ROOT),
    ):
        manager = ShieldManager(MOCK_TASK_DIR, cfg=cfg)
        first = manager.shield
        second = manager.shield
    assert first is second


# ── Re-exports ───────────────────────────────────────────────────────────


def test_nft_not_found_is_reexported() -> None:
    """``NftNotFoundError`` is re-exported from the adapter module."""
    from terok_sandbox.integrations.shield import NftNotFoundError as error_type

    assert error_type is NftNotFoundError


def test_shield_state_is_reexported() -> None:
    """``ShieldState`` is re-exported from the adapter module."""
    from terok_sandbox.integrations.shield import ShieldState as shield_state_type

    assert shield_state_type is ShieldState


# ── ShieldManager delegation to the underlying Shield ───────────────────


@pytest.mark.parametrize(
    ("method_name", "delegated_call", "expected"),
    [
        pytest.param("up", ("up", "ctr"), None, id="up"),
        pytest.param("quarantine", ("quarantine", "ctr"), None, id="quarantine"),
        pytest.param("state", ("state", "ctr"), ShieldState.UP, id="state"),
    ],
)
def test_manager_methods_delegate_to_shield(
    method_name: str,
    delegated_call: tuple[str, str],
    expected: object,
) -> None:
    """Each manager method forwards to the corresponding ``Shield`` method."""
    mock_shield = make_mock_shield()
    manager = ShieldManager(MOCK_TASK_DIR)
    with patch.object(ShieldManager, "shield", new=mock_shield):
        result = getattr(manager, method_name)("ctr")
    getattr(mock_shield, delegated_call[0]).assert_called_once_with(delegated_call[1])
    if expected is not None:
        assert result == expected


def test_manager_down_passes_allow_all() -> None:
    """``ShieldManager.down`` forwards ``allow_all=True`` to the underlying Shield."""
    mock_shield = make_mock_shield()
    manager = ShieldManager(MOCK_TASK_DIR)
    with patch.object(ShieldManager, "shield", new=mock_shield):
        manager.down("ctr", allow_all=True)
    mock_shield.down.assert_called_once_with("ctr", allow_all=True)


def test_manager_pre_start_threads_runtime_into_shield() -> None:
    """The runtime kwarg flows into the ShieldConfig the Shield is built from."""
    from terok_shield import ShieldRuntime

    cfg = SandboxConfig(gate_port=GATE_PORT, token_broker_port=18731, ssh_signer_port=18732)
    with (
        patch("terok_shield.run.SubprocessRunner", autospec=True),
        patch("terok_sandbox.paths.namespace_config_root", return_value=MOCK_CONFIG_ROOT),
    ):
        manager = ShieldManager(MOCK_TASK_DIR, cfg=cfg, runtime=ShieldRuntime.KRUN)
        assert manager.shield.config.runtime is ShieldRuntime.KRUN


def test_status_defaults() -> None:
    """Status reflects the default configured shield state."""
    assert ShieldManager(MOCK_TASK_DIR, SandboxConfig()).status() == {
        "mode": "hook",
        "profiles": ["dev-standard"],
        "audit_enabled": True,
    }


def test_status_custom_config() -> None:
    """Status reflects custom configured profiles and audit settings."""
    cfg = SandboxConfig(shield_profiles=("custom",), shield_audit=False)
    assert ShieldManager(MOCK_TASK_DIR, cfg).status() == {
        "mode": "hook",
        "profiles": ["custom"],
        "audit_enabled": False,
    }


# ── Bypass handling ─────────────────────────────────────────────────────


@pytest.mark.parametrize("method_name", ["down", "up"])
def test_bypass_makes_down_and_up_noops(method_name: str) -> None:
    """Bypass mode makes ``ShieldManager.up`` / ``.down`` no-ops without touching Shield."""
    mock_shield = MagicMock(spec=Shield)
    manager = ShieldManager(MOCK_TASK_DIR, SandboxConfig(shield_bypass=True))
    with patch.object(ShieldManager, "shield", new=mock_shield):
        getattr(manager, method_name)("ctr")
    mock_shield.up.assert_not_called()
    mock_shield.down.assert_not_called()


def test_quarantine_ignores_bypass() -> None:
    """Quarantine overrides bypass — panic must always work."""
    mock_shield = make_mock_shield()
    manager = ShieldManager(MOCK_TASK_DIR, SandboxConfig(shield_bypass=True))
    with patch.object(ShieldManager, "shield", new=mock_shield):
        manager.quarantine("ctr")
    mock_shield.quarantine.assert_called_once_with("ctr")


def test_bypass_pre_start_returns_empty_with_warning() -> None:
    """Bypass mode returns no pre-start podman args and warns loudly."""
    manager = ShieldManager(MOCK_TASK_DIR, SandboxConfig(shield_bypass=True))
    with pytest.warns(UserWarning) as caught:
        assert manager.pre_start("ctr") == []
    assert any(_BYPASS_WARNING in str(item.message) for item in caught)


def test_bypass_state_still_queries_real_shield() -> None:
    """State lookup still queries the real shield to handle pre-bypass containers."""
    mock_shield = make_mock_shield(shield_state=ShieldState.UP)
    manager = ShieldManager(MOCK_TASK_DIR, SandboxConfig(shield_bypass=True))
    with patch.object(ShieldManager, "shield", new=mock_shield):
        assert manager.state("ctr") == ShieldState.UP
    mock_shield.state.assert_called_once_with("ctr")


@pytest.mark.parametrize(
    ("bypass_enabled", "expected_key"),
    [
        pytest.param(True, True, id="bypass-active"),
        pytest.param(False, False, id="bypass-disabled"),
    ],
)
def test_status_includes_bypass_flag_only_when_active(
    bypass_enabled: bool,
    expected_key: bool,
) -> None:
    """Status output surfaces the dangerous bypass flag only when it is active."""
    result = ShieldManager(MOCK_TASK_DIR, SandboxConfig(shield_bypass=bypass_enabled)).status()
    assert ("bypass_firewall_no_protection" in result) is expected_key
    assert result["mode"] == "hook"
    assert "profiles" in result


# ── Environment probe ──────────────────────────────────────────────────


def test_check_environment_forwards_result() -> None:
    """The free check_environment delegates to a throwaway ``ShieldManager``."""
    expected = EnvironmentCheck(ok=True, health="ok", podman_version=(5, 6, 0))
    mock_shield = make_mock_shield()
    mock_shield.check_environment.return_value = expected
    with patch.object(ShieldManager, "shield", new=mock_shield):
        assert check_environment() == expected
    mock_shield.check_environment.assert_called_once()


def test_check_environment_bypass_returns_synthetic_result() -> None:
    """Bypass mode surfaces a synthetic degraded environment result."""
    result = check_environment(cfg=SandboxConfig(shield_bypass=True))
    assert not result.ok
    assert result.health == "bypass"
    assert any("bypass" in issue for issue in result.issues)


def test_pre_start_converts_shield_needs_setup_to_system_exit() -> None:
    """``ShieldNeedsSetup`` is converted into a diagnostic SystemExit."""
    mock_shield = make_mock_shield()
    mock_shield.pre_start.side_effect = ShieldNeedsSetup("hooks not installed")
    manager = ShieldManager(MOCK_TASK_DIR)
    with (
        patch.object(ShieldManager, "shield", new=mock_shield),
        pytest.raises(SystemExit, match="hooks not installed"),
    ):
        manager.pre_start("ctr")


# ── ShieldHooks (host-wide installer) ───────────────────────────────────


@pytest.mark.parametrize(
    ("kwargs", "expected_calls"),
    [
        pytest.param({}, None, id="missing-flags"),
        pytest.param({"user": True}, [("user",)], id="user-only"),
        pytest.param({"root": True}, [("system",)], id="root-only"),
        pytest.param({"root": True, "user": True}, [("user",), ("system",)], id="both-scopes"),
    ],
)
def test_shield_hooks_install_dispatches_per_scope(
    kwargs: dict[str, bool],
    expected_calls: list[tuple[str]] | None,
) -> None:
    """``ShieldHooks.install`` calls the matching ``HooksInstaller`` factories."""
    with (
        patch("terok_sandbox.integrations.shield.HooksInstaller.user") as user_factory,
        patch("terok_sandbox.integrations.shield.HooksInstaller.system") as system_factory,
    ):
        if expected_calls is None:
            with pytest.raises(ValueError, match="root=True or user=True"):
                ShieldHooks.install(**kwargs)
            user_factory.assert_not_called()
            system_factory.assert_not_called()
            return
        ShieldHooks.install(**kwargs)
        factories = {"user": user_factory, "system": system_factory}
        invoked = {scope for (scope,) in expected_calls}
        for scope in invoked:
            factories[scope].assert_called_once()
            factories[scope].return_value.install.assert_called_once()
        # Non-selected scopes must not be touched — guards against
        # accidentally installing into both scopes when only one flag
        # was set.
        for scope in set(factories) - invoked:
            factories[scope].assert_not_called()


@pytest.mark.parametrize(
    ("kwargs", "expected_calls"),
    [
        pytest.param({}, None, id="missing-flags"),
        pytest.param({"user": True}, [("user",)], id="user-only"),
        pytest.param({"root": True}, [("system",)], id="root-only"),
        pytest.param({"root": True, "user": True}, [("user",), ("system",)], id="both-scopes"),
    ],
)
def test_shield_hooks_uninstall_dispatches_per_scope(
    kwargs: dict[str, bool],
    expected_calls: list[tuple[str]] | None,
) -> None:
    """``ShieldHooks.uninstall`` calls the matching ``HooksInstaller`` factories."""
    with (
        patch("terok_sandbox.integrations.shield.HooksInstaller.user") as user_factory,
        patch("terok_sandbox.integrations.shield.HooksInstaller.system") as system_factory,
    ):
        if expected_calls is None:
            with pytest.raises(ValueError, match="root=True or user=True"):
                ShieldHooks.uninstall(**kwargs)
            user_factory.assert_not_called()
            system_factory.assert_not_called()
            return
        ShieldHooks.uninstall(**kwargs)
        factories = {"user": user_factory, "system": system_factory}
        invoked = {scope for (scope,) in expected_calls}
        for scope in invoked:
            factories[scope].assert_called_once()
            factories[scope].return_value.uninstall.assert_called_once()
        # Non-selected scopes must not be touched — same guard as the
        # install path above.
        for scope in set(factories) - invoked:
            factories[scope].assert_not_called()


# ── Session helpers ─────────────────────────────────────────────────────


@patch("terok_shield.simple_clearance.run_simple_clearance")
def test_interactive_session_delegates_to_simple_clearance(
    mock_run_simple_clearance: MagicMock,
) -> None:
    """Session helper forwards the shield state_dir and container to the terminal fallback."""
    ShieldManager(MOCK_TASK_DIR).interactive_session("task-ctr")
    mock_run_simple_clearance.assert_called_once_with(MOCK_TASK_DIR / "shield", "task-ctr")


@patch("terok_shield.watch.run_watch")
def test_watch_session_delegates_to_run_watch(mock_run_watch: MagicMock) -> None:
    """Session helper forwards the shield state_dir and container to terok-shield."""
    ShieldManager(MOCK_TASK_DIR).watch_session("task-ctr")
    mock_run_watch.assert_called_once_with(MOCK_TASK_DIR / "shield", "task-ctr")


# ── Round-trip ──────────────────────────────────────────────────────────


def test_install_then_uninstall_round_trip(tmp_path: Path) -> None:
    """``HooksInstaller.user().install()`` + ``.uninstall()`` round-trip leaves no residue."""
    from terok_shield import HooksInstaller

    hooks_dir = tmp_path / "hooks.d"
    installer = HooksInstaller(target_dir=hooks_dir)
    installer.install()
    assert hooks_dir.is_dir()
    assert any(hooks_dir.iterdir())
    installer.uninstall()
    # The directory itself is left in place (other tools may share it);
    # only our files are gone.
    assert list(hooks_dir.iterdir()) == []
