# SPDX-FileCopyrightText: 2026 Jiri Vyskocil
# SPDX-License-Identifier: Apache-2.0

"""Smoke test: every ``__main__.py`` shim must import cleanly.

These shims are launched by systemd via ``python -m terok_sandbox.<pkg>``
and *only* by that path — no other test loads them, because plain
``import terok_sandbox.<pkg>`` only evaluates ``__init__.py``.  The vault
service silently broke once after a refactor moved ``token_broker.py``
under ``vault/daemon/`` and left ``vault/__main__.py`` pointing at the
old path; the regression only surfaced on a real systemd start.

This test walks every ``__main__.py`` in the source tree and imports it
via the dotted module name.  Any ``ModuleNotFoundError`` /
``ImportError`` raised at module load time fails the test with the
offending file path attached.

Entry-point shims must therefore guard side effects with
``if __name__ == "__main__":`` — calling ``main()`` unconditionally would
trigger argparse/socket binding here.  That guard is the convention this
test enforces by side-channel.
"""

from __future__ import annotations

import importlib
from pathlib import Path

import pytest

_SRC_ROOT = Path(__file__).resolve().parents[2] / "src"


def _entry_point_modules() -> list[str]:
    """Return the dotted names of every ``__main__.py`` shim under ``src/``."""
    return [
        ".".join(path.relative_to(_SRC_ROOT).with_suffix("").parts)
        for path in _SRC_ROOT.rglob("__main__.py")
    ]


@pytest.mark.parametrize("module_name", _entry_point_modules())
def test_entry_point_module_imports_cleanly(module_name: str) -> None:
    """Importing the ``__main__`` shim must not raise.

    Catches the failure mode where a refactor moves a sibling module
    but leaves the entry-point shim's ``from .x import y`` stale.
    """
    importlib.import_module(module_name)
