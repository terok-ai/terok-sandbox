# SPDX-FileCopyrightText: 2026 Jiri Vyskocil
# SPDX-License-Identifier: Apache-2.0

"""Stage-line rendering — one format, one colour palette, one column width.

The setup aggregator prints one ``  <label>  <marker> (<detail>)``
line per phase.  Frontends (terok's ``terok setup``,
``terok-executor`` CLIs, future CI reporters) can mix stage lines of
their own — desktop-entry install, credential-DB purge, etc. — by
importing the public symbols from :mod:`terok_sandbox`; keeping the
renderer central guarantees the mixed log reads as one continuous
column with aligned status markers and coherent colours.

Kept as an underscore-prefixed submodule because the package's
``__init__`` already re-exports the surface — a public
``terok_sandbox.stage`` submodule name would collide with the
re-exported ``stage()`` function.

Colour is auto-detected from ``NO_COLOR`` / ``FORCE_COLOR`` /
``sys.stdout.isatty()`` (the no-color.org contract).  Terminals that
don't report TTY — typical CI logs — fall through to plain text.
"""

from __future__ import annotations

import os
import sys
from enum import StrEnum

# Label column width: widest shipped label is ``"Clearance notifier"``
# (18 chars); 21 leaves 3 chars of gutter before the marker.  Recompute
# when a new phase ships with a longer label.  Terok front-ends use
# the same value so their supplementary stages stay column-aligned.
STAGE_WIDTH = 21


class Marker(StrEnum):
    """Status tokens rendered in each stage line.

    A :class:`StrEnum` so a typo (``"Warn"`` vs ``"WARN"``) is a
    load-time error, not silent drift that test assertions happen to
    keep passing.  Each value maps to an ANSI colour in :data:`_PALETTE`.
    """

    OK = "ok"
    WARN = "WARN"
    FAIL = "FAIL"
    MISSING = "MISSING"
    SKIP = "skip"


# ANSI SGR codes keyed by marker.  ``None`` means "render uncoloured
# even when colour is on" — ``skip`` is a user-chosen soft state, not
# a success or failure, so it stays neutral.
_PALETTE: dict[Marker, str | None] = {
    Marker.OK: "32",  # green
    Marker.WARN: "33",  # yellow
    Marker.FAIL: "31",  # red
    Marker.MISSING: "31",  # red — a missing binary will fail the install
    Marker.SKIP: None,
}


def stage(label: str, marker: Marker, detail: str = "") -> None:
    """Write one complete stage line: ``'  <label>  <marker>[ (<detail>)]'``.

    Matches :func:`stage_begin` + :func:`stage_end` when the caller
    doesn't need progressive output.  The marker is ANSI-coloured
    according to :data:`_PALETTE` when colour is enabled.
    """
    suffix = f" ({detail})" if detail else ""
    print(f"  {label:<{STAGE_WIDTH}} {_render_marker(marker)}{suffix}")


def stage_begin(label: str) -> None:
    """Write the label column and flush — no newline, no marker.

    Pairs with :func:`stage_end`.  Use when the phase takes long enough
    that the operator benefits from seeing *which* step is running
    before the marker lands.  Without this, a slow
    ``systemctl --user restart`` looks like a frozen terminal.
    """
    print(f"  {label:<{STAGE_WIDTH}}", end="", flush=True)


def stage_end(marker: Marker, detail: str = "") -> None:
    """Write the marker and optional detail with trailing newline.

    The sibling of :func:`stage_begin`; together they render the same
    line :func:`stage` would.
    """
    suffix = f" ({detail})" if detail else ""
    print(f" {_render_marker(marker)}{suffix}")


def supports_color() -> bool:
    """Return whether ANSI colour should be emitted to stdout.

    Follows the `no-color.org <https://no-color.org>`_ contract:
    ``NO_COLOR`` always wins; ``FORCE_COLOR`` (set to anything but
    ``"0"``) opts back in even on non-TTY streams; otherwise
    ``sys.stdout.isatty()`` decides.  Cached at module-import time so
    the verdict is stable for the life of the process — tests that
    need a different answer set the env vars before importing.
    """
    return _COLOUR_ON


def bold(text: str) -> str:
    """Return *text* wrapped in ANSI bold when :func:`supports_color` is true.

    Exposed because frontend callers print titles and closing banners
    that sit outside the stage-line column — keeping the bold helper
    here means those callers don't grow a second colour import.
    """
    return _color(text, "1")


def _render_marker(marker: Marker) -> str:
    """Return *marker*'s string value, coloured when the palette has a code."""
    code = _PALETTE[marker]
    return _color(marker.value, code) if code else marker.value


def _color(text: str, code: str | None) -> str:
    """Wrap *text* in ANSI SGR *code* when colour is on; passthrough otherwise."""
    if not _COLOUR_ON or code is None:
        return text
    return f"\x1b[{code}m{text}\x1b[0m"


def _detect_colour() -> bool:
    """Apply the NO_COLOR / FORCE_COLOR / isatty precedence."""
    if "NO_COLOR" in os.environ:
        return False
    force = os.environ.get("FORCE_COLOR")
    if force is not None and force != "0":
        return True
    return sys.stdout.isatty()


_COLOUR_ON = _detect_colour()
