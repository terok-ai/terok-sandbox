# SPDX-FileCopyrightText: 2026 Jiri Vyskocil
# SPDX-License-Identifier: Apache-2.0

"""``python -m terok_sandbox`` entry point.

Equivalent to the ``terok-sandbox`` console script, but reachable
through the interpreter that is already running the package.  The split
supervisor spawns its children this way
([`launch_child`][terok_sandbox.supervisor.launcher.launch_child] uses
``[sys.executable, "-m", "terok_sandbox", …]``) so a child lands on the
exact interpreter and environment of its parent, with no dependence on
``terok-sandbox`` being resolvable on ``$PATH``.
"""

from __future__ import annotations

from terok_sandbox.cli import main

if __name__ == "__main__":
    # ``main`` dispatches the verb and exits via ``SystemExit`` carrying
    # the handler's return code (that is how the wrapper reads a child's
    # exit), so there is nothing to wrap here.
    main()
