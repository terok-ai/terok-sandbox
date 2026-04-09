# SPDX-FileCopyrightText: 2025 Jiri Vyskocil
# SPDX-FileCopyrightText: 2026 Jiri Vyskocil
# SPDX-License-Identifier: Apache-2.0

"""Platform-aware path resolution for the terok ecosystem.

Provides generic **umbrella resolvers** that any sibling package can call
to place its state/config/runtime under the shared ``terok/`` namespace,
plus sandbox-specific thin wrappers for backward compatibility.
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


_UMBRELLA = "terok"


def _is_root() -> bool:
    """Return True if the current process is running as root."""
    try:
        return os.geteuid() == 0  # type: ignore[attr-defined]
    except AttributeError:
        return getpass.getuser() == "root"


# ---------------------------------------------------------------------------
# Generic umbrella resolvers (DRY: used by sandbox, agent, and terok)
# ---------------------------------------------------------------------------


def umbrella_state_dir(subdir: str = "", env_var: str | None = None) -> Path:
    """Resolve a state directory under the ``terok/`` umbrella namespace.

    Priority: *env_var* → ``/var/lib/terok/<subdir>`` (root) → platformdirs
    → ``$XDG_DATA_HOME/terok/<subdir>`` → ``~/.local/share/terok/<subdir>``.
    """
    if env_var:
        val = os.getenv(env_var)
        if val:
            return Path(val).expanduser()
    base: Path
    if _is_root():
        base = Path("/var/lib") / _UMBRELLA
    elif _user_data_dir is not None:
        base = Path(_user_data_dir(_UMBRELLA))
    else:
        xdg = os.getenv("XDG_DATA_HOME")
        base = Path(xdg) / _UMBRELLA if xdg else Path.home() / ".local" / "share" / _UMBRELLA
    return base / subdir if subdir else base


def umbrella_config_dir(subdir: str = "", env_var: str | None = None) -> Path:
    """Resolve a config directory under the ``terok/`` umbrella namespace.

    Priority: *env_var* → ``/etc/terok/<subdir>`` (root) → platformdirs
    → ``~/.config/terok/<subdir>``.
    """
    if env_var:
        val = os.getenv(env_var)
        if val:
            return Path(val).expanduser()
    base: Path
    if _is_root():
        base = Path("/etc") / _UMBRELLA
    elif _user_config_dir is not None:
        base = Path(_user_config_dir(_UMBRELLA))
    else:
        base = Path.home() / ".config" / _UMBRELLA
    return base / subdir if subdir else base


def umbrella_runtime_dir(subdir: str = "", env_var: str | None = None) -> Path:
    """Resolve a runtime directory under the ``terok/`` umbrella namespace.

    Priority: *env_var* → ``/run/terok/<subdir>`` (root)
    → ``$XDG_RUNTIME_DIR/terok/<subdir>`` → ``$XDG_STATE_HOME/terok/<subdir>``
    → ``~/.local/state/terok/<subdir>``.
    """
    if env_var:
        val = os.getenv(env_var)
        if val:
            return Path(val).expanduser()
    base: Path
    if _is_root():
        base = Path("/run") / _UMBRELLA
    else:
        xdg_runtime = os.getenv("XDG_RUNTIME_DIR")
        if xdg_runtime:
            base = Path(xdg_runtime) / _UMBRELLA
        else:
            xdg_state = os.getenv("XDG_STATE_HOME")
            base = (
                Path(xdg_state) / _UMBRELLA
                if xdg_state
                else Path.home() / ".local" / "state" / _UMBRELLA
            )
    return base / subdir if subdir else base


# ---------------------------------------------------------------------------
# Sandbox-specific thin wrappers (preserve existing API)
# ---------------------------------------------------------------------------


def config_root() -> Path:
    """Base directory for sandbox configuration.

    Priority: ``TEROK_SANDBOX_CONFIG_DIR`` → ``/etc/terok/sandbox`` (root)
    → ``~/.config/terok/sandbox``.
    """
    return umbrella_config_dir("sandbox", "TEROK_SANDBOX_CONFIG_DIR")


def state_root() -> Path:
    """Writable state root for sandbox (tasks, tokens, caches).

    Priority: ``TEROK_SANDBOX_STATE_DIR`` → ``/var/lib/terok/sandbox`` (root)
    → ``~/.local/share/terok/sandbox``.
    """
    return umbrella_state_dir("sandbox", "TEROK_SANDBOX_STATE_DIR")


def runtime_root() -> Path:
    """Transient runtime directory for sandbox (PID files, sockets).

    Priority: ``TEROK_SANDBOX_RUNTIME_DIR`` → ``/run/terok/sandbox`` (root) →
    ``$XDG_RUNTIME_DIR/terok/sandbox`` → ``$XDG_STATE_HOME/terok/sandbox`` →
    ``~/.local/state/terok/sandbox``.
    """
    return umbrella_runtime_dir("sandbox", "TEROK_SANDBOX_RUNTIME_DIR")


def credentials_root() -> Path:
    """Shared credentials directory used by all terok ecosystem packages.

    Priority: ``TEROK_CREDENTIALS_DIR`` → ``/var/lib/terok/credentials`` (root)
    → XDG data dir.
    """
    return umbrella_state_dir("credentials", "TEROK_CREDENTIALS_DIR")


def umbrella_config_root() -> Path:
    """Return the top-level terok config root (umbrella, not sandbox-scoped).

    Used for cross-package paths like shield profiles that live under
    the shared ``~/.config/terok/`` umbrella rather than under any single
    package's config directory.
    """
    return umbrella_config_dir("", "TEROK_CONFIG_DIR")
