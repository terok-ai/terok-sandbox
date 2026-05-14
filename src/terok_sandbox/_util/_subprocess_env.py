# SPDX-FileCopyrightText: 2026 Jiri Vyskocil
# SPDX-License-Identifier: Apache-2.0

"""Environment helper for spawned ``terok_sandbox`` child processes.

Centralises the ``PYTHONPATH`` shim every ``subprocess.run`` /
``Popen`` of ``sys.executable`` must use, so adding a new spawn site
can't silently regress the Nix-wrapped-Python fix.
"""

from __future__ import annotations

import os
import sys


def child_process_env(overrides: dict[str, str] | None = None) -> dict[str, str]:
    """Build the environment for a spawned ``terok_sandbox`` child process.

    Threads the parent's ``sys.path`` through as ``PYTHONPATH`` so the
    child can import ``terok_sandbox`` regardless of how the parent was
    launched.  Under Nix, ``sys.executable`` is a wrapper script that
    rewrites the env on startup — but spawning it directly via
    ``subprocess`` / ``create_subprocess_exec`` bypasses that wrapper,
    leaving the child unable to find the ``terok_sandbox`` package.
    This shim restores it.

    *overrides* are applied on top of the parent env; ``PYTHONPATH``
    always wins so a stray ambient value can't shadow the parent's
    real import path.
    """
    return {**os.environ, **(overrides or {}), "PYTHONPATH": os.pathsep.join(sys.path)}
