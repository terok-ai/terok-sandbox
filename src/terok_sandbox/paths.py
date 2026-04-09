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

_TEROK_ROOT_ENV = "TEROK_ROOT"
"""Env var overriding the umbrella state root for all ecosystem packages."""


def _is_root() -> bool:
    """Return True if the current process is running as root."""
    try:
        return os.geteuid() == 0  # type: ignore[attr-defined]
    except AttributeError:
        return getpass.getuser() == "root"


# ---------------------------------------------------------------------------
# Shared config reader (Podman model: all packages read one config.yml)
# ---------------------------------------------------------------------------

_config_paths_cache: dict[str, str] | None = None


def _read_config_paths() -> dict[str, str]:
    """Read ``paths:`` from the shared terok config (cached, fail-silent).

    Follows the Podman model: one config file (``~/.config/terok/config.yml``)
    is the source of truth for directory layout across all ecosystem packages.
    Only the ``paths`` section is read — other keys are package-specific and
    ignored here.
    """
    global _config_paths_cache  # noqa: PLW0603
    if _config_paths_cache is not None:
        return _config_paths_cache

    _config_paths_cache = {}
    cfg = _config_file_path()
    if not cfg.is_file():
        return _config_paths_cache
    try:
        from yaml import safe_load  # lazy: PyYAML is a transitive dep via terok-shield

        data = safe_load(cfg.read_text(encoding="utf-8")) or {}
        paths = data.get("paths") or {}
        _config_paths_cache = {k: str(v) for k, v in paths.items() if v}
    except Exception:  # noqa: BLE001 — fail-silent; bad config should not crash path resolution
        pass
    return _config_paths_cache


def _config_file_path() -> Path:
    """Return the well-known terok config.yml location.

    ``TEROK_CONFIG_FILE`` → ``/etc/terok/config.yml`` (root)
    → ``~/.config/terok/config.yml``.
    """
    env = os.getenv("TEROK_CONFIG_FILE")
    if env:
        return Path(env).expanduser()
    if _is_root():
        return Path("/etc") / _UMBRELLA / "config.yml"
    if _user_config_dir is not None:
        return Path(_user_config_dir(_UMBRELLA)) / "config.yml"
    return Path.home() / ".config" / _UMBRELLA / "config.yml"


def _umbrella_root() -> Path | None:
    """Return the configured umbrella state root, or ``None`` for platform default.

    Resolution: ``TEROK_ROOT`` env var → ``config.yml`` ``paths.root``
    (with ``paths.state_dir`` as deprecated fallback).
    """
    env = os.getenv(_TEROK_ROOT_ENV)
    if env:
        return Path(env).expanduser()
    cfg = _read_config_paths()
    val = cfg.get("root") or cfg.get("state_dir")  # state_dir = deprecated alias
    return Path(val).expanduser().resolve() if val else None


# ---------------------------------------------------------------------------
# Generic umbrella resolvers (DRY: used by sandbox, agent, and terok)
# ---------------------------------------------------------------------------


def _safe_subdir(base: Path, subdir: str) -> Path:
    """Join *subdir* to *base*, rejecting absolute or parent-traversal paths."""
    if not subdir:
        return base
    if Path(subdir).is_absolute() or ".." in Path(subdir).parts:
        raise ValueError(f"subdir must be relative without '..', got {subdir!r}")
    return base / subdir


def _platform_state_base() -> Path:
    """Return the platform-default state base (no config override)."""
    if _is_root():
        return Path("/var/lib") / _UMBRELLA
    if _user_data_dir is not None:
        return Path(_user_data_dir(_UMBRELLA))
    xdg = os.getenv("XDG_DATA_HOME")
    return Path(xdg) / _UMBRELLA if xdg else Path.home() / ".local" / "share" / _UMBRELLA


def umbrella_state_dir(subdir: str = "", env_var: str | None = None) -> Path:
    """Resolve a state directory under the ``terok/`` umbrella namespace.

    Priority:

    1. *env_var* (package-specific override, e.g. ``TEROK_SANDBOX_STATE_DIR``)
    2. ``TEROK_ROOT`` env var (umbrella override)
    3. ``config.yml`` → ``paths.root`` (Podman model — all packages honor it)
    4. Platform default (``/var/lib/terok/<subdir>`` or XDG)
    """
    if env_var:
        val = os.getenv(env_var)
        if val:
            return Path(val).expanduser()
    root = _umbrella_root()
    base = root if root else _platform_state_base()
    return _safe_subdir(base, subdir)


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
    return _safe_subdir(base, subdir)


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
    return _safe_subdir(base, subdir)


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
