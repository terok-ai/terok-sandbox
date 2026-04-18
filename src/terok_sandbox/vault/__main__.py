# SPDX-FileCopyrightText: 2026 Jiri Vyskocil
# SPDX-License-Identifier: Apache-2.0

"""Allow ``python -m terok_sandbox.vault`` to start the vault daemon."""

from .token_broker import main

if __name__ == "__main__":
    main()
