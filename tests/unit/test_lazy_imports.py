# SPDX-FileCopyrightText: 2026 Jiri Vyskocil
# SPDX-License-Identifier: Apache-2.0

"""Guard the import-laziness contract of the top-level barrel + command registry.

Importing ``terok_sandbox`` — or its command registry, the entry point
the per-container supervisor spawn walks through ``cli:main`` — must not
eagerly pull in the heavy leaves: pydantic (config models), SQLCipher
(the credential store), cryptography (SSH keypairs), or terok-shield.
Each is paid for only when its subsystem is first used, which is what
keeps the supervisor spawn cheap.

Measured in a **fresh interpreter** per case (``sys.executable``): a
same-process ``sys.modules`` check would be polluted by whatever the
test session already imported.
"""

from __future__ import annotations

import json
import os
import subprocess  # nosec B404 — fixed argv, no shell, running our own interpreter
import sys
from pathlib import Path

import terok_sandbox

#: Heavy leaves that must stay off the import path of a bare
#: ``import terok_sandbox`` and of building the command registry.
HEAVY_LEAVES = ("pydantic", "sqlcipher3", "cryptography", "terok_shield")

#: The ``src`` root holding the importable package, so the fresh
#: subprocess resolves ``terok_sandbox`` the same way this session did
#: (covers the PYTHONPATH=src layout as well as an installed wheel).
_SRC_ROOT = str(Path(terok_sandbox.__file__).resolve().parent.parent)


def _heavy_leaves_after(snippet: str) -> list[str]:
    """Return which [`HEAVY_LEAVES`][tests.unit.test_lazy_imports.HEAVY_LEAVES] *snippet* pulled in.

    Runs ``snippet`` in a fresh interpreter and reports the subset of the
    heavy leaves present in its ``sys.modules`` afterwards.
    """
    code = (
        "import sys, json\n"
        f"{snippet}\n"
        f"print(json.dumps([m for m in {HEAVY_LEAVES!r} if m in sys.modules]))"
    )
    env = {
        **os.environ,
        "PYTHONPATH": os.pathsep.join([_SRC_ROOT, os.environ.get("PYTHONPATH", "")]),
    }
    result = subprocess.run(  # noqa: S603 — fixed interpreter + inline code, no shell
        [sys.executable, "-c", code],
        capture_output=True,
        text=True,
        env=env,
        check=True,
    )
    return json.loads(result.stdout.strip().splitlines()[-1])


def test_bare_import_pulls_no_heavy_leaves() -> None:
    """``import terok_sandbox`` stays off pydantic / SQLCipher / cryptography / terok-shield."""
    assert _heavy_leaves_after("import terok_sandbox") == []


def test_building_commands_pulls_no_heavy_leaves() -> None:
    """Building the command forest — the supervisor-spawn path — stays lightweight."""
    assert _heavy_leaves_after("from terok_sandbox.commands import COMMANDS") == []
