# SPDX-FileCopyrightText: 2026 Jiri Vyskocil
# SPDX-License-Identifier: Apache-2.0

"""Static check: every ``from .x`` import resolves to a real module.

The smoke test in
[`test_entry_point_modules_importable`][tests.unit.test_entry_point_modules_importable]
catches stale relative imports at module *load* time — but only for
imports written at top level.  Lazy imports inside function bodies
(``from .ssh_signer import start_ssh_signer`` deep inside an async
``_run_multi``) hide from that test until the live code path runs.
The vault daemon refactor (#292) shipped two such stale lazy imports
that crashed in production weeks after the refactor merged.

This test walks every ``.py`` file under ``src/terok_sandbox/`` and
finds every ``ast.ImportFrom`` node — at any depth — then resolves the
relative target against the file's own package using
``importlib.util.find_spec``.  Any unresolved module (whether sibling,
parent, or child) fails the test with the source location.

We don't *execute* anything: ``find_spec`` is a pure name lookup, so
this is safe to run in any environment.
"""

from __future__ import annotations

import ast
import importlib.util
from pathlib import Path

import terok_sandbox

_SRC_ROOT = Path(terok_sandbox.__file__).resolve().parent


def _file_to_package(path: Path) -> str:
    """Return the dotted package name of the *containing* directory of *path*."""
    rel = path.relative_to(_SRC_ROOT.parent).with_suffix("")
    parts = rel.parts[:-1] if rel.name == "__init__" else rel.parts[:-1]
    return ".".join(parts)


def _resolve_relative(level: int, module: str | None, package: str) -> str | None:
    """Mirror Python's relative-import resolution; ``None`` if it underflows."""
    if level == 0:
        return module
    parts = package.split(".")
    if level > len(parts):
        return None
    base = ".".join(parts[: len(parts) - level + 1])
    return f"{base}.{module}" if module else base


def _collect_relative_imports(path: Path) -> list[tuple[int, str]]:
    """Return ``(lineno, absolute_target_module)`` for every ``from .X`` in *path*."""
    package = _file_to_package(path)
    tree = ast.parse(path.read_text())
    out: list[tuple[int, str]] = []
    for node in ast.walk(tree):
        if not isinstance(node, ast.ImportFrom) or node.level == 0:
            continue
        target = _resolve_relative(node.level, node.module, package)
        if target is None:
            out.append((node.lineno, f"<underflow: level={node.level} module={node.module}>"))
            continue
        out.append((node.lineno, target))
    return out


def _iter_python_files(root: Path):
    """Yield every ``.py`` file under *root*, excluding ``__pycache__``."""
    for path in root.rglob("*.py"):
        if "__pycache__" in path.parts:
            continue
        yield path


def _stale_imports() -> list[str]:
    """Return formatted ``file:line — target`` strings for unresolvable imports."""
    offenders: list[str] = []
    for path in _iter_python_files(_SRC_ROOT):
        for lineno, target in _collect_relative_imports(path):
            try:
                spec = importlib.util.find_spec(target)
            except (ModuleNotFoundError, ValueError):
                spec = None
            if spec is None:
                rel = path.relative_to(_SRC_ROOT.parent)
                offenders.append(f"  {rel}:{lineno} — {target!r} does not resolve")
    return offenders


def test_every_relative_import_resolves() -> None:
    """``from .x import y`` must point at a module that actually exists.

    Catches the failure mode where a refactor moves ``x`` and leaves
    callers (especially lazy imports inside function bodies) pointing
    at the old path.  Static — no module evaluation, just name lookup.
    """
    offenders = _stale_imports()
    assert not offenders, (
        "Found relative imports that don't resolve.  These will crash "
        "with ModuleNotFoundError when the importing code runs (which "
        "may be far from module load — e.g. inside an async function "
        "guarded by a config flag):\n" + "\n".join(offenders)
    )
