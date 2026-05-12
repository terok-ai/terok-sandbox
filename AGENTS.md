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

## Docs

- Markdown files under `docs/` are lowercase by convention; root-level
  files (`README.md`, `AGENTS.md`) are not.
