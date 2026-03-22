# SPDX-FileCopyrightText: 2025 Jiri Vyskocil
# SPDX-License-Identifier: Apache-2.0

"""CLI entry point for terok-sandbox."""

from . import __version__


def main() -> None:
    """Entry point for the ``terok-sandbox`` command."""
    print(f"terok-sandbox {__version__}")
    raise SystemExit(0)
