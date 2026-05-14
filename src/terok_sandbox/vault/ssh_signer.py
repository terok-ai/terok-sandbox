# SPDX-FileCopyrightText: 2026 Jiri Vyskocil
# SPDX-License-Identifier: Apache-2.0

"""Back-compat re-export — moved to ``terok_sandbox.vault.ssh.signer``.

Cleaned up once cross-repo consumers (terok, terok-executor) update
their imports.
"""

from terok_sandbox.vault.ssh.signer import *  # noqa: F401,F403
