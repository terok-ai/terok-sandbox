# SPDX-FileCopyrightText: 2025 Jiri Vyskocil
# SPDX-FileCopyrightText: 2026 Jiri Vyskocil
# SPDX-License-Identifier: Apache-2.0

"""Sandbox-specific filesystem helpers.

Generic helpers (``ensure_dir``, ``ensure_dir_writable``,
``write_sensitive_file``) live in [`terok_util.fs`][terok_util.fs];
sandbox's ``_util/__init__.py`` re-exports them so the existing
``from .._util import ensure_dir`` callsites keep working.  Only
sandbox-specific helpers stay in this module.
"""

import os
from pathlib import Path


def systemd_user_unit_dir() -> Path:
    """Return the systemd user unit directory, validated against path traversal.

    Refuses to run as root (``euid == 0``) and resolves ``$XDG_CONFIG_HOME``
    to ensure the result stays beneath the user's home directory.

    Raises [`SystemExit`][SystemExit] on validation failure.
    """
    if os.geteuid() == 0:
        raise SystemExit(
            "Refusing to manage user systemd units as root. "
            "Run without sudo or use a system-level unit instead."
        )
    xdg = os.environ.get("XDG_CONFIG_HOME")
    config_home = Path(xdg).resolve() if xdg else Path.home() / ".config"
    home = Path.home().resolve()
    if not str(config_home).startswith(str(home)):
        raise SystemExit(
            f"XDG_CONFIG_HOME ({config_home}) resolves outside the home directory ({home}). "
            "Refusing to install systemd units to a potentially untrusted location."
        )
    return config_home / "systemd" / "user"
