# SPDX-FileCopyrightText: 2026 Jiri Vyskocil
# SPDX-License-Identifier: Apache-2.0

"""Tests for [`SandboxConfig`][terok_sandbox.SandboxConfig] — notably the services-mode branching."""

from __future__ import annotations

import unittest.mock

import pytest

from terok_sandbox.config import SandboxConfig


class TestSocketModeSkipsPortResolution:
    """``services.mode: socket`` must bypass the TCP port registry entirely."""

    def test_socket_mode_leaves_ports_unset(self) -> None:
        """No auto-allocation happens; port fields remain ``None``."""
        with (
            unittest.mock.patch("terok_sandbox.config.services_mode", return_value="socket"),
            unittest.mock.patch(
                "terok_sandbox.port_registry._default.resolve_service_ports"
            ) as resolve,
        ):
            cfg = SandboxConfig()
        resolve.assert_not_called()
        assert cfg.gate_port is None
        assert cfg.token_broker_port is None
        assert cfg.ssh_signer_port is None

    def test_invalid_mode_warns_and_falls_back(self, capsys: pytest.CaptureFixture[str]) -> None:
        """A typo in ``services.mode`` emits stderr warning and defaults to schema value.

        The fallback follows ``RawServicesSection`` — one schema default
        across both terok's and sandbox's readers (since the refactor,
        the hand-rolled ``"tcp"`` fallback is gone in favor of whatever
        the pydantic schema declares).
        """
        from terok_sandbox.config import services_mode
        from terok_sandbox.config_schema import RawServicesSection

        with unittest.mock.patch(
            "terok_sandbox.config.read_config_section", return_value={"mode": "soket"}
        ):
            assert services_mode() == RawServicesSection().mode
        captured = capsys.readouterr()
        assert "invalid services section" in captured.err

    def test_tcp_mode_construction_does_not_allocate(self) -> None:
        """Constructing ``SandboxConfig()`` is side-effect-free, even in tcp mode.

        The port registry is consulted on demand by callers that need
        real listeners — see ``with_resolved_ports`` — not by
        ``__post_init__``.  The user-facing principle (sandbox #156)
        is that *constructing* a config never bind-tests or persists.
        """
        with (
            unittest.mock.patch("terok_sandbox.config.services_mode", return_value="tcp"),
            unittest.mock.patch(
                "terok_sandbox.port_registry._default.resolve_service_ports",
            ) as resolve,
        ):
            cfg = SandboxConfig()
        resolve.assert_not_called()
        assert cfg.gate_port is None
        assert cfg.token_broker_port is None
        assert cfg.ssh_signer_port is None

    def test_with_resolved_ports_populates_via_registry_in_tcp_mode(self) -> None:
        """``with_resolved_ports`` is the explicit allocation step for consumers."""
        from terok_sandbox.port_registry import ServicePorts

        fake_ports = ServicePorts(gate=18700, proxy=18701, ssh_agent=18702)
        with (
            unittest.mock.patch("terok_sandbox.config.services_mode", return_value="tcp"),
            unittest.mock.patch(
                "terok_sandbox.port_registry._default.resolve_service_ports",
                return_value=fake_ports,
            ) as resolve,
        ):
            cfg = SandboxConfig().with_resolved_ports()
        resolve.assert_called_once()
        assert cfg.gate_port == 18700
        assert cfg.token_broker_port == 18701
        assert cfg.ssh_signer_port == 18702

    def test_with_resolved_ports_is_noop_in_socket_mode(self) -> None:
        """Socket-mode cfgs return ``self`` (no copy, no registry pass)."""
        with (
            unittest.mock.patch("terok_sandbox.config.services_mode", return_value="socket"),
            unittest.mock.patch(
                "terok_sandbox.port_registry._default.resolve_service_ports",
            ) as resolve,
        ):
            original = SandboxConfig()
            resolved = original.with_resolved_ports()
        resolve.assert_not_called()
        assert resolved is original

    def test_with_resolved_ports_is_idempotent_once_all_set(self) -> None:
        """A cfg with every port already set short-circuits — no registry pass."""
        with (
            unittest.mock.patch("terok_sandbox.config.services_mode", return_value="tcp"),
            unittest.mock.patch(
                "terok_sandbox.port_registry._default.resolve_service_ports",
            ) as resolve,
        ):
            cfg = SandboxConfig(gate_port=1, token_broker_port=2, ssh_signer_port=3)
            resolved = cfg.with_resolved_ports()
        resolve.assert_not_called()
        assert resolved is cfg


class TestExperimentalEnabled:
    """Tests for ``experimental_enabled`` — the top-level opt-in reader.

    Gates the krun runtime in terok and ``check_krun_binaries`` in
    sandbox's prereq report.  Single declaration on
    ``SandboxConfigView``; the reader walks the layered config and
    coerces defensively.
    """

    def test_returns_true_when_set(self) -> None:
        """``experimental: true`` in config.yml flips the bit on."""
        from terok_sandbox.config import experimental_enabled

        with unittest.mock.patch("terok_sandbox.config.read_config_top_level", return_value=True):
            assert experimental_enabled() is True

    def test_returns_false_when_unset(self) -> None:
        """Missing top-level key collapses to ``False`` (the safe default)."""
        from terok_sandbox.config import experimental_enabled

        with unittest.mock.patch("terok_sandbox.config.read_config_top_level", return_value=None):
            assert experimental_enabled() is False

    @pytest.mark.parametrize(
        "raw",
        [
            pytest.param("true", id="string-true"),
            pytest.param(1, id="int-one"),
            pytest.param([True], id="list"),
            pytest.param({"nested": True}, id="dict"),
        ],
    )
    def test_returns_false_for_non_bool_values(self, raw: object) -> None:
        """A malformed value (string, int, list, dict) falls through to ``False``.

        Defensive against operators who write ``experimental: "yes"``
        thinking YAML coerces; never crash, never accidentally enable
        an experimental code path on a typo.
        """
        from terok_sandbox.config import experimental_enabled

        with unittest.mock.patch("terok_sandbox.config.read_config_top_level", return_value=raw):
            assert experimental_enabled() is False

    def test_sandbox_config_picks_up_value(self) -> None:
        """``SandboxConfig().experimental`` flows through the default-factory."""
        with unittest.mock.patch("terok_sandbox.config.experimental_enabled", return_value=True):
            cfg = SandboxConfig()
        assert cfg.experimental is True

    def test_explicit_override_wins(self) -> None:
        """``SandboxConfig(experimental=False)`` overrides the config-derived value."""
        with unittest.mock.patch("terok_sandbox.config.experimental_enabled", return_value=True):
            cfg = SandboxConfig(experimental=False)
        assert cfg.experimental is False
