# SPDX-FileCopyrightText: 2026 Jiri Vyskocil
# SPDX-License-Identifier: Apache-2.0

"""Tests for :class:`SandboxConfig` — notably the services-mode branching."""

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

    def test_tcp_mode_resolves_ports(self) -> None:
        """In tcp mode the registry is consulted and fields are populated."""
        from terok_sandbox.port_registry import ServicePorts

        fake_ports = ServicePorts(gate=18700, proxy=18701, ssh_agent=18702)
        with (
            unittest.mock.patch("terok_sandbox.config.services_mode", return_value="tcp"),
            unittest.mock.patch(
                "terok_sandbox.port_registry._default.resolve_service_ports",
                return_value=fake_ports,
            ) as resolve,
        ):
            cfg = SandboxConfig()
        resolve.assert_called_once()
        assert cfg.gate_port == 18700
        assert cfg.token_broker_port == 18701
        assert cfg.ssh_signer_port == 18702
