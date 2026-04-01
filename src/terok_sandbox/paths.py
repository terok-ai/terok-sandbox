# SPDX-FileCopyrightText: 2025 Jiri Vyskocil
# SPDX-FileCopyrightText: 2026 Jiri Vyskocil
# SPDX-License-Identifier: Apache-2.0

"""Platform-aware path resolution for sandbox directories.

Vendored from ``core.paths`` — zero internal dependencies.  Provides
the same XDG / FHS resolution logic so that ``terok-sandbox`` works
identically whether embedded in terok or used standalone.
"""

import getpass
import os
from pathlib import Path

try:
    from platformdirs import (
        user_config_dir as _user_config_dir,
        user_data_dir as _user_data_dir,
    )
except ImportError:  # optional dependency
    _user_config_dir = _user_data_dir = None  # type: ignore[assignment]


APP_NAME = "terok-sandbox"
CREDENTIALS_APP_NAME = "terok-credentials"

_UMBRELLA = "terok"
_SUBDIR = "sandbox"
_CRED_SUBDIR = "credentials"


def _is_root() -> bool:
    """Return True if the current process is running as root."""
    try:
        return os.geteuid() == 0  # type: ignore[attr-defined]
    except AttributeError:
        return getpass.getuser() == "root"


def config_root() -> Path:
    """Base directory for configuration.

    Priority: ``TEROK_SANDBOX_CONFIG_DIR`` → ``/etc/terok/sandbox`` (root)
    → ``~/.config/terok/sandbox``.
    """
    env = os.getenv("TEROK_SANDBOX_CONFIG_DIR")
    if env:
        return Path(env).expanduser()
    if _is_root():
        return Path("/etc") / _UMBRELLA / _SUBDIR
    if _user_config_dir is not None:
        return Path(_user_config_dir(_UMBRELLA)) / _SUBDIR
    return Path.home() / ".config" / _UMBRELLA / _SUBDIR


def state_root() -> Path:
    """Writable state root (tasks, tokens, caches).

    Priority: ``TEROK_SANDBOX_STATE_DIR`` → ``/var/lib/terok/sandbox`` (root)
    → ``~/.local/share/terok/sandbox``.
    """
    env = os.getenv("TEROK_SANDBOX_STATE_DIR")
    if env:
        return Path(env).expanduser()
    if _is_root():
        return Path("/var/lib") / _UMBRELLA / _SUBDIR
    if _user_data_dir is not None:
        return Path(_user_data_dir(_UMBRELLA)) / _SUBDIR
    xdg = os.getenv("XDG_DATA_HOME")
    if xdg:
        return Path(xdg) / _UMBRELLA / _SUBDIR
    return Path.home() / ".local" / "share" / _UMBRELLA / _SUBDIR


def runtime_root() -> Path:
    """Transient runtime directory (PID files, sockets).

    Priority: ``TEROK_SANDBOX_RUNTIME_DIR`` → ``/run/terok/sandbox`` (root) →
    ``$XDG_RUNTIME_DIR/terok/sandbox`` → ``$XDG_STATE_HOME/terok/sandbox`` →
    ``~/.local/state/terok/sandbox``.
    """
    env = os.getenv("TEROK_SANDBOX_RUNTIME_DIR")
    if env:
        return Path(env).expanduser()
    if _is_root():
        return Path("/run") / _UMBRELLA / _SUBDIR
    xdg_runtime = os.getenv("XDG_RUNTIME_DIR")
    if xdg_runtime:
        return Path(xdg_runtime) / _UMBRELLA / _SUBDIR
    xdg_state = os.getenv("XDG_STATE_HOME")
    if xdg_state:
        return Path(xdg_state) / _UMBRELLA / _SUBDIR
    return Path.home() / ".local" / "state" / _UMBRELLA / _SUBDIR


def credentials_root() -> Path:
    """Shared credentials directory used by all terok ecosystem packages.

    Priority: ``TEROK_CREDENTIALS_DIR`` → ``/var/lib/terok/credentials`` (root)
    → XDG data dir.
    """
    env = os.getenv("TEROK_CREDENTIALS_DIR")
    if env:
        return Path(env).expanduser()
    if _is_root():
        return Path("/var/lib") / _UMBRELLA / _CRED_SUBDIR
    if _user_data_dir is not None:
        return Path(_user_data_dir(_UMBRELLA)) / _CRED_SUBDIR
    xdg = os.getenv("XDG_DATA_HOME")
    if xdg:
        return Path(xdg) / _UMBRELLA / _CRED_SUBDIR
    return Path.home() / ".local" / "share" / _UMBRELLA / _CRED_SUBDIR


def umbrella_config_root() -> Path:
    """Return the top-level terok config root (umbrella, not sandbox-scoped).

    Used for cross-package paths like shield profiles that live under
    the shared ``~/.config/terok/`` umbrella rather than under any single
    package's config directory.
    """
    env = os.getenv("TEROK_CONFIG_DIR")
    if env:
        return Path(env).expanduser()
    if _is_root():
        return Path("/etc") / _UMBRELLA
    if _user_config_dir is not None:
        return Path(_user_config_dir(_UMBRELLA))
    return Path.home() / ".config" / _UMBRELLA
