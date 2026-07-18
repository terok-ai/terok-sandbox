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
#: ``importlib.metadata`` earns its place: the ``__version__`` lookup
#: pulls ``inspect``/``email``/``zipfile`` (~3–4 MiB RSS), which every
#: supervisor child would pay if it crept back to import time.
HEAVY_LEAVES = ("pydantic", "sqlcipher3", "cryptography", "terok_shield", "importlib.metadata")

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


#: The per-subsystem command modules, by leaf name — the set lazy
#: dispatch keeps out of ``sys.modules`` until their verb is invoked.
COMMAND_MODULES = (
    "gate",
    "vault",
    "ssh",
    "credentials",
    "shield",
    "doctor",
    "launch",
    "sandbox",
    "supervisor",
)


def _command_modules_after(argv: list[str]) -> list[str]:
    """Return which command modules a fresh ``terok-sandbox`` *argv* run loaded.

    Drives [`terok_sandbox.cli.main`][terok_sandbox.cli.main] with *argv*
    in a fresh interpreter and reports the per-subsystem modules present
    in ``sys.modules`` afterwards.
    """
    code = (
        "import sys\n"
        "from terok_sandbox.cli import main\n"
        "try:\n"
        f"    main({argv!r})\n"
        "except SystemExit:\n"
        "    pass\n"
        f"loaded = [n for n in {COMMAND_MODULES!r} "
        "if f'terok_sandbox.commands.{n}' in sys.modules]\n"
        "import json; print(json.dumps(loaded))"
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


def test_verb_loads_only_its_own_module() -> None:
    """``terok-sandbox gate …`` imports the gate module and no other subsystem."""
    assert _command_modules_after(["gate", "--help"]) == ["gate"]


def test_supervisor_spawn_loads_only_supervisor() -> None:
    """The per-container spawn (``supervisor <id> <sidecar>``) stays off every other module."""
    assert _command_modules_after(["supervisor", "cid", "/sidecar.json"]) == ["supervisor"]
    # …and specifically never touches terok-shield.
    assert (
        _heavy_leaves_after(
            "from terok_sandbox.cli import main\n"
            "import contextlib\n"
            "with contextlib.suppress(SystemExit, BaseException):\n"
            "    main(['supervisor', 'cid', '/sidecar.json'])"
        )
        == []
    )


def test_top_level_help_imports_no_command_module() -> None:
    """``terok-sandbox --help`` lists every verb without importing any subsystem module."""
    assert _command_modules_after(["--help"]) == []


def test_version_still_resolves_lazily() -> None:
    """``terok_sandbox.__version__`` and ``--version`` still work, just paid on demand."""
    env = {
        **os.environ,
        "PYTHONPATH": os.pathsep.join([_SRC_ROOT, os.environ.get("PYTHONPATH", "")]),
    }
    code = (
        "import terok_sandbox\n"
        "assert isinstance(terok_sandbox.__version__, str) and terok_sandbox.__version__\n"
        "from terok_sandbox.cli import main\n"
        "try:\n"
        "    main(['--version'])\n"
        "except SystemExit:\n"
        "    pass"
    )
    result = subprocess.run(  # noqa: S603 — fixed interpreter + inline code, no shell
        [sys.executable, "-c", code],
        capture_output=True,
        text=True,
        env=env,
        check=True,
    )
    assert result.stdout.startswith("terok-sandbox ")


def test_supervisor_verb_skips_shield_wiring() -> None:
    """The `supervisor` fast-path wires only its own subtree, so the shield
    subtree never materialises and terok-shield stays unimported — this is
    the last heavy leaf the live per-container spawn would otherwise pay."""
    snippet = (
        "from terok_sandbox.cli import main\n"
        "try:\n"
        "    main(['supervisor', '--help'])\n"
        "except SystemExit:\n"
        "    pass"
    )
    assert _heavy_leaves_after(snippet) == []
