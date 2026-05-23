# SPDX-FileCopyrightText: 2026 Jiri Vyskocil
# SPDX-License-Identifier: Apache-2.0

"""Pin the public API for the sandbox-wide uninstall aggregator.

External consumers (``terok``, ``terok-executor``) reach the
sandbox-wide install/uninstall aggregators in two shapes:

- ``sandbox_uninstall`` is published on the package root for the
  ``terok`` CLI's ``uninstall`` command, which calls it directly.
- The corresponding setup verb (``_handle_sandbox_setup``) is imported
  by ``terok-executor`` from ``terok_sandbox.commands`` via its
  integrations adapter — no top-level alias is published for it.

This test fixes the identity link between the top-level
``sandbox_uninstall`` and the implementation handler so a renamed
implementation can't silently leave the public name pointing at a
stale callable.
"""

from __future__ import annotations

import terok_sandbox
from terok_sandbox.commands import _handle_sandbox_uninstall


def test_sandbox_uninstall_is_public_alias_of_aggregator() -> None:
    assert terok_sandbox.sandbox_uninstall is _handle_sandbox_uninstall


def test_sandbox_uninstall_listed_in_all() -> None:
    """``__all__`` lists ``sandbox_uninstall`` — pins the star-import contract."""
    assert "sandbox_uninstall" in terok_sandbox.__all__
