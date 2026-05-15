# SPDX-FileCopyrightText: 2025 Jiri Vyskocil
# SPDX-License-Identifier: Apache-2.0

"""Allow ``python -m terok_sandbox.gate`` to start the gate server."""

from .server import main

if __name__ == "__main__":
    main()
