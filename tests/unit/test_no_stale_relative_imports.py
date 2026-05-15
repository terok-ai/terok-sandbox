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
relative target against the file's own package.  Any unresolved module
(whether sibling, parent, or child) fails the test with the source
location.

Resolution uses [`PathFinder.find_spec`][importlib.machinery.PathFinder.find_spec]
with an explicit *parent* path on the source tree, not
``importlib.util.find_spec``.  The latter would import every parent
``__init__.py`` of a dotted target as a side effect of name lookup,
and we deliberately want this test to stay side-effect-free so it
remains safe to run before any project state has been initialised.
"""

from __future__ import annotations

import ast
from importlib.machinery import PathFinder
from pathlib import Path

import terok_sandbox

_SRC_ROOT = Path(terok_sandbox.__file__).resolve().parent
_REPO_ROOT = _SRC_ROOT.parent


def _file_to_package(path: Path) -> str:
    """Return the dotted package name of the *containing* directory of *path*."""
    rel = path.relative_to(_REPO_ROOT).with_suffix("")
    parts = rel.parts[:-1]
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
    """Return ``(lineno, absolute_target_module)`` for every relative import in *path*.

    Handles both forms equally:

    * ``from .x import a, b`` → resolves ``package + .x`` once
    * ``from . import a, b`` → resolves ``package + .a`` and ``package + .b``
      (each alias is its own potential submodule and can independently rot)
    """
    package = _file_to_package(path)
    tree = ast.parse(path.read_text())
    out: list[tuple[int, str]] = []
    for node in ast.walk(tree):
        if not isinstance(node, ast.ImportFrom) or node.level == 0:
            continue
        if node.module is not None:
            target = _resolve_relative(node.level, node.module, package)
            if target is None:
                out.append((node.lineno, f"<underflow: level={node.level} module={node.module}>"))
                continue
            out.append((node.lineno, target))
            continue
        # ``from . import a, b`` — each alias is a candidate submodule.
        # ``from . import *`` is a separate beast: it doesn't name any
        # submodule, just star-imports the package's ``__all__``.  The
        # candidate to validate there is the package itself.
        for alias in node.names:
            if alias.name == "*":
                target = _resolve_relative(node.level, None, package)
            else:
                target = _resolve_relative(node.level, alias.name, package)
            if target is None:
                out.append((node.lineno, f"<underflow: level={node.level} name={alias.name}>"))
                continue
            out.append((node.lineno, target))
    return out


def _iter_python_files(root: Path):
    """Yield every ``.py`` file under *root*, excluding ``__pycache__``."""
    for path in root.rglob("*.py"):
        if "__pycache__" in path.parts:
            continue
        yield path


def _resolves_statically(target: str) -> bool:
    """Return True when *target* exists on disk under the source tree.

    Walks the dotted name down the source tree and uses
    ``PathFinder.find_spec(target, [parent_dir])`` for the leaf lookup —
    a finder restricted to the parent directory cannot trigger any
    import side effects (no parent ``__init__`` evaluation, no
    ``sys.modules`` writes).  Names outside ``terok_sandbox.*`` are
    treated as resolving (third-party + stdlib are not our concern
    here).  Underflow sentinels emitted by the collector — strings
    starting with ``<underflow:`` — are non-modules and must surface
    as failures rather than getting swept up by the early return.
    """
    if target.startswith("<underflow:"):
        return False
    if not target.startswith("terok_sandbox"):
        return True
    parent_dotted, _, _ = target.rpartition(".")
    parent_rel = Path(*parent_dotted.split("."))
    parent_dir = _REPO_ROOT / parent_rel
    if not parent_dir.is_dir():
        return False
    spec = PathFinder.find_spec(target, [str(parent_dir)])
    return spec is not None


def _stale_imports() -> list[str]:
    """Return formatted ``file:line — target`` strings for unresolvable imports."""
    offenders: list[str] = []
    for path in _iter_python_files(_SRC_ROOT):
        for lineno, target in _collect_relative_imports(path):
            if _resolves_statically(target):
                continue
            rel = path.relative_to(_REPO_ROOT)
            offenders.append(f"  {rel}:{lineno} — {target!r} does not resolve")
    return offenders


def test_every_relative_import_resolves() -> None:
    """Every ``from .X import …`` and ``from . import X`` must point at a real module.

    Catches the failure mode where a refactor moves ``X`` and leaves
    callers — especially lazy imports inside function bodies — pointing
    at the old path.  Static: no module evaluation, just on-disk lookup
    via [`PathFinder`][importlib.machinery.PathFinder].
    """
    offenders = _stale_imports()
    assert not offenders, (
        "Found relative imports that don't resolve.  These will crash "
        "with ModuleNotFoundError when the importing code runs (which "
        "may be far from module load — e.g. inside an async function "
        "guarded by a config flag):\n" + "\n".join(offenders)
    )
