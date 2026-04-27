# SPDX-FileCopyrightText: 2026 Jiri Vyskocil
# SPDX-License-Identifier: Apache-2.0

"""Setup-stamp primitive: cheap "did the user run setup?" check for the TUI startup path.

Background
==========

The TUI's start-up budget is ≤2 s.  A real ``sickbay`` health check
takes seconds — too slow for an every-launch probe.  Instead we drop
a small JSON marker after a successful ``setup`` run and compare it
against the currently-installed package versions on each launch:

- match → ``OK``, no surface
- absent → ``FIRST_RUN``, blocking modal nudges the user to run setup
- installed > stamped → ``STALE_AFTER_UPDATE``, non-blocking banner
- installed < stamped → ``STALE_AFTER_DOWNGRADE``, blocking warning
  (downgrades aren't tested; explicit override required)
- stamp can't be parsed → ``STAMP_CORRUPT``, treated as ``FIRST_RUN``

Lives in sandbox because it's the lowest layer — every other
ecosystem package depends on sandbox, so the primitive is reachable
from any frontend without inverting the dep graph.

Resolution detail
=================

Package versions are read via [`importlib.metadata.version`][importlib.metadata.version] —
no subprocess, no parsing, sub-millisecond per package.  Total cost
of [`needs_setup`][terok_sandbox.setup_stamp.needs_setup] on a happy path: one ``Path.is_file``, one
JSON decode, four to five ``importlib.metadata`` lookups.  Well under
the 2 s budget.

The set of tracked packages is the v0 ecosystem (``terok``,
``terok-executor``, ``terok-sandbox``, ``terok-shield``,
``terok-clearance``).  A package missing from the local install is
ignored on the read side — standalone sandbox installations don't
have ``terok`` and shouldn't see ``STALE_AFTER_DOWNGRADE`` because of it.
"""

from __future__ import annotations

import contextlib
import json
from datetime import UTC, datetime
from enum import Enum
from importlib.metadata import PackageNotFoundError, version as _meta_version
from pathlib import Path

from packaging.version import InvalidVersion, Version

from .paths import namespace_state_dir

# ── Public API ────────────────────────────────────────────────────────


class SetupVerdict(Enum):
    """Result of [`needs_setup`][terok_sandbox.setup_stamp.needs_setup] — five possible states a launch can be in."""

    OK = "ok"
    """Stamp matches all installed package versions exactly."""

    FIRST_RUN = "first_run"
    """No stamp on disk — the user has never run setup (or wiped state)."""

    STALE_AFTER_UPDATE = "stale_after_update"
    """At least one installed package is newer than the stamped version."""

    STALE_AFTER_DOWNGRADE = "stale_after_downgrade"
    """At least one installed package is older than the stamped version.

    Downgrades aren't tested and can leave systemd units / state DB in
    forms the older code can't interpret.  Frontends should treat this
    as a hard stop until the user explicitly overrides.
    """

    STAMP_CORRUPT = "stamp_corrupt"
    """Stamp file exists but can't be parsed.  Frontends should treat as FIRST_RUN."""


_STAMP_SCHEMA_VERSION = 1
_STAMP_FILENAME = "setup.stamp"
_TRACKED_PACKAGES: tuple[str, ...] = (
    "terok",
    "terok-executor",
    "terok-sandbox",
    "terok-shield",
    "terok-clearance",
)


def stamp_path() -> Path:
    """Return the canonical on-disk location of the setup stamp.

    Honours the umbrella ``paths.root`` resolver so a user who relocates
    the state tree (``paths.root: /virt/terok`` in ``config.yml``) sees
    the stamp move with it — same place every package would look.
    """
    return namespace_state_dir() / _STAMP_FILENAME


def needs_setup() -> SetupVerdict:
    """Compare the on-disk stamp against currently-installed package versions.

    See [`SetupVerdict`][terok_sandbox.setup_stamp.SetupVerdict] for the five possible outcomes.  Designed
    to be cheap enough to call on every TUI startup.
    """
    path = stamp_path()
    if not path.exists():
        return SetupVerdict.FIRST_RUN
    if not path.is_file():
        # Something at the stamp location, but not a regular file — a
        # directory or device left there by a misbehaving sync tool.
        # That's not "user hasn't run setup" (FIRST_RUN); it's a corrupt
        # state the next ``write_stamp`` would also fail on.
        return SetupVerdict.STAMP_CORRUPT

    try:
        stamped = _read_stamp(path)
    except (OSError, json.JSONDecodeError, ValueError):
        return SetupVerdict.STAMP_CORRUPT

    installed = _installed_versions()
    return _compare(stamped, installed)


def write_stamp() -> Path:
    """Capture the currently-installed versions to disk and return the path.

    Called by ``_handle_sandbox_setup`` after a successful run.  An
    atomic temp-file + rename keeps a partial write from leaving a
    half-stamp that ``needs_setup`` would later flag as ``STAMP_CORRUPT``.
    """
    path = stamp_path()
    path.parent.mkdir(parents=True, exist_ok=True)

    payload = {
        "version": _STAMP_SCHEMA_VERSION,
        "completed_at": datetime.now(UTC).isoformat(timespec="seconds"),
        "packages": _installed_versions(),
    }
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")
    tmp.replace(path)
    return path


def clear_stamp() -> bool:
    """Remove the stamp file if present.  Returns True if a file was removed.

    EAFP rather than ``is_file`` + ``unlink``: keeps the function
    race-safe under a (rare) concurrent ``terok uninstall`` instead of
    leaking a ``FileNotFoundError`` between the existence check and the
    unlink.  Used by ``terok uninstall`` and by tests that want to
    simulate a fresh-install state without nuking the rest of the
    state tree.
    """
    try:
        stamp_path().unlink()
    except FileNotFoundError:
        return False
    return True


# ── Internals ─────────────────────────────────────────────────────────


def _installed_versions() -> dict[str, str]:
    """Return ``{package: version}`` for every tracked package present in the install.

    Missing packages are silently dropped — a standalone ``terok-sandbox``
    install doesn't have ``terok`` available, and that's fine.  The
    invariant we check on the read side is that every package the *stamp*
    knows about is also installed (and at the right version).
    """
    out: dict[str, str] = {}
    for pkg in _TRACKED_PACKAGES:
        with contextlib.suppress(PackageNotFoundError):
            out[pkg] = _meta_version(pkg)
    return out


def _read_stamp(path: Path) -> dict[str, str]:
    """Parse the stamp file, returning the ``packages`` mapping.

    Raises [`ValueError`][ValueError] if the schema version doesn't match — a
    schema bump should be handled explicitly, not silently coerced.
    """
    raw = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(raw, dict):
        raise ValueError(f"stamp root is not an object: {type(raw).__name__}")
    if raw.get("version") != _STAMP_SCHEMA_VERSION:
        raise ValueError(f"unsupported stamp schema version: {raw.get('version')!r}")
    pkgs = raw.get("packages")
    if not isinstance(pkgs, dict):
        raise ValueError(f"stamp packages is not an object: {type(pkgs).__name__}")
    # Coerce values to str so a malformed stamp with an int can't sneak past.
    return {str(k): str(v) for k, v in pkgs.items()}


def _compare(stamped: dict[str, str], installed: dict[str, str]) -> SetupVerdict:
    """Compare stamped versions against installed; return the matching verdict.

    The stamp is the source of truth for "what setup expected to find".
    For each stamped package we look up the installed version and
    compare.  A package missing from the install while stamped counts
    as STALE_AFTER_DOWNGRADE — the install lost a package the stamp
    expected, which means setup state may be inconsistent.
    """
    saw_update = False
    for pkg, stamp_ver in stamped.items():
        if pkg not in installed:
            return SetupVerdict.STALE_AFTER_DOWNGRADE
        installed_ver = installed[pkg]
        cmp = _compare_versions(installed_ver, stamp_ver)
        if cmp < 0:
            return SetupVerdict.STALE_AFTER_DOWNGRADE
        if cmp > 0:
            saw_update = True
    return SetupVerdict.STALE_AFTER_UPDATE if saw_update else SetupVerdict.OK


def _compare_versions(a: str, b: str) -> int:
    """Return -1 / 0 / 1 for ``a`` vs ``b`` using PEP 440 ordering.

    Falls back to string comparison if either side fails to parse —
    keeps ``needs_setup`` from blowing up on a hand-edited stamp with
    a non-PEP-440 version string.
    """
    try:
        va, vb = Version(a), Version(b)
    except InvalidVersion:
        return (a > b) - (a < b)
    return (va > vb) - (va < vb)


__all__ = [
    "SetupVerdict",
    "clear_stamp",
    "needs_setup",
    "stamp_path",
    "write_stamp",
]
