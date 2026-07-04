# SPDX-FileCopyrightText: 2026 Jiri Vyskocil
# SPDX-License-Identifier: Apache-2.0

"""Operator-facing invocation identity for composing actionable hints.

Sandbox operations are embedded under other front-ends' verbs: the
credentials provisioning runs inside ``terok setup`` and
``terok-executor setup`` as well as ``terok-sandbox setup``.  A hint
that tells the operator to re-run setup must name an invocation the
consuming front-end actually exposes, and only that front-end knows its
own verb layout — so front-ends declare the spelling in the
``TEROK_SETUP_INVOCATION`` environment variable at CLI entry, and hints
fall back to sandbox's own spelling when nothing is declared.  The
fallback is always a valid remedy: ``terok-sandbox setup`` ships with
every install.

The contract is the environment variable name, not an import — consuming
packages define the same string as their own constant, so no version
coupling exists in either direction.
"""

from __future__ import annotations

import os

#: Environment variable through which a front-end declares the
#: invocation under which sandbox setup is reachable in its own CLI.
SETUP_INVOCATION_ENV = "TEROK_SETUP_INVOCATION"

_SANDBOX_SETUP = "terok-sandbox setup"


def setup_invocation() -> str:
    """Return the setup invocation re-run hints should name."""
    return os.environ.get(SETUP_INVOCATION_ENV) or _SANDBOX_SETUP
