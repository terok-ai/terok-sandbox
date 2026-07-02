# Agent Instructions

## Verification

Run `make check` before declaring work done — it covers lint, unit
tests, module boundaries (tach), security, docstring coverage, dead
code, and SPDX compliance.  Skip tests that need podman or docker;
those run on a dedicated test machine.

## Code style

- Domain-first docstrings, public entry points above private helpers,
  top-down reading order.
- Cross-references in docstrings use mkdocstrings autoref syntax
  `` [`Name`][module.path.Name] `` — never the Sphinx
  ``:class:`Name``` / ``:func:`name``` forms.  Sphinx roles render as
  literal text on the rendered docs site (mkdocstrings doesn't process
  them).  Prefer the explicit full path over the bare `` [`Name`][] ``
  autoref form: explicit paths keep `properdocs build --strict` green
  even when the symbol's short name isn't unique.  For external symbols
  use the dependency's own path
  (`` [`Sandbox`][terok_sandbox.Sandbox] ``,
  `` [`StreamReader`][asyncio.StreamReader] ``) — those resolve via the
  inventories listed in `properdocs.yml`.
- SPDX copyright: author name "Jiri Vyskocil".  Add a new
  `SPDX-FileCopyrightText` line only for a previously unlisted
  contributor making a substantive change.
- Public API surface: ``__init__.py`` + ``__all__`` is the contract.
  Symbols listed in ``__all__`` are stable across minor releases;
  anything underscore-prefixed or absent from ``__all__`` is internal
  and may change without notice.  Review the list before each release
  — stable APIs stay small because growing them costs.

## Tests

- **Path isolation**: tests must never touch the operator's real
  filesystem.  All on-disk fixtures route through `tmp_path` /
  `tmp_path_factory`.  The autouse `_isolate_user_paths` fixture in
  `tests/unit/conftest.py` redirects `HOME` and the `XDG_*` chain to
  a fresh tmp dir, so default-config code paths (`SandboxConfig()`,
  `handle_*(cfg=None)`) land in tmp by construction — don't bypass
  it, and add to `_TEROK_PATH_OVERRIDE_ENV_VARS` when introducing a
  new `TEROK_*_DIR` knob.

## Docs

- Markdown files under `docs/` are lowercase by convention; root-level
  files (`README.md`, `AGENTS.md`) are not.

## Dependency Pinning & `pyproject.toml` Hygiene

**Version pinning policy.** Runtime/production dependencies — those pulled in
by a plain `pip install` / `pipx install` of this package (the
`[project].dependencies` table) — are pinned by the dependency's major
version:

- **Third-party, major 0 (`0.y.z`)** → pin to an **exact patch**
  (`pkg==0.y.z`). Pre-1.0 packages promise no compatibility across either
  minors *or* patches, so a floating range invites silent breakage.
- **Third-party, major ≥ 1** → pin by **range** (e.g. `pkg>=2.6`), trusting
  the package to honour semver. If a specific `>=1` dependency is known to
  break semver, tighten it deliberately.
- **Sibling `terok-*` deps** → **exempt**: keep ranges (or their
  release-wheel URL pin). We guarantee patch-level API stability across the
  sibling packages, so a `0.y` range there will not silently break — do
  *not* exact-pin them (it would fight the multi-repo release/PR-chain flow).

Dev / test / docs / tooling dependencies (the `[tool.poetry.group.*]` groups)
are **exempt** — they are not shipped to installers and exact-pinning them is
an unwarranted maintenance burden the developers can absorb. After changing
any pin, run `poetry lock` and commit `pyproject.toml` and `poetry.lock`
together.

**No comments in `pyproject.toml`.** Do **not** add comments to
`pyproject.toml`, with the single exception of the standing dependency-pinning
policy note above the `dependencies` table. In particular **never** add a
comment about a dependency that is temporarily pinned to a git branch during a
multi-repo PR chain, and never mention the PR-chain workflow in
`pyproject.toml` at all. Cross-repo merges are performed by a script that does
not understand comments, so any stray dev-cycle comment is carried straight
into a production release. Keep such rationale in commit messages, PR
descriptions, or this file.
