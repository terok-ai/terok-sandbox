# SPDX-FileCopyrightText: 2026 Jiri Vyskocil
# SPDX-License-Identifier: Apache-2.0

"""Pin the public API for the sandbox-wide install/uninstall aggregators.

External consumers (``terok``, ``terok-executor``) should reach the
aggregators through the stable unprefixed names ``sandbox_setup`` /
``sandbox_uninstall`` on the package root, not through the
``commands._handle_*`` implementation symbols.  This test makes sure
the two sets stay identity-linked so a renamed implementation can't
silently leave the public names pointing at stale callables.
"""

from __future__ import annotations

from unittest.mock import patch

import terok_sandbox
from terok_sandbox.commands import _handle_sandbox_setup, _handle_sandbox_uninstall


def test_sandbox_setup_is_public_alias_of_aggregator() -> None:
    assert terok_sandbox.sandbox_setup is _handle_sandbox_setup


def test_sandbox_uninstall_is_public_alias_of_aggregator() -> None:
    assert terok_sandbox.sandbox_uninstall is _handle_sandbox_uninstall


def test_aggregators_are_in_all() -> None:
    """Listing in ``__all__`` — the contract ``from terok_sandbox import *`` promises."""
    assert "sandbox_setup" in terok_sandbox.__all__
    assert "sandbox_uninstall" in terok_sandbox.__all__


def test_vault_systemd_wrappers_delegate_to_manager() -> None:
    """Public vault systemd helpers are thin ``VaultManager`` delegators."""
    cfg = object()
    with patch("terok_sandbox.VaultManager") as manager_cls:
        manager = manager_cls.return_value

        terok_sandbox.install_vault_systemd(cfg)
        terok_sandbox.uninstall_vault_systemd(cfg)

    assert manager_cls.call_args_list[0].args == (cfg,)
    assert manager_cls.call_args_list[1].args == (cfg,)
    manager.install_systemd_units.assert_called_once_with()
    manager.uninstall_systemd_units.assert_called_once_with()


def test_vault_systemd_wrappers_are_public() -> None:
    """The star-import contract includes the vault systemd helpers."""
    assert "install_vault_systemd" in terok_sandbox.__all__
    assert "uninstall_vault_systemd" in terok_sandbox.__all__
