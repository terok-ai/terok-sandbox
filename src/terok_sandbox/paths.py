# SPDX-FileCopyrightText: 2025 Jiri Vyskocil
# SPDX-FileCopyrightText: 2026 Jiri Vyskocil
# SPDX-License-Identifier: Apache-2.0

"""Platform-aware path resolution for the terok ecosystem.

Provides generic **namespace resolvers** that any sibling package can call
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


_NAMESPACE = "terok"

_TEROK_ROOT_ENV = "TEROK_ROOT"
"""Env var overriding the namespace state root for all ecosystem packages."""


def _is_root() -> bool:
    """Return True if the current process is running as root."""
    try:
        return os.geteuid() == 0  # type: ignore[attr-defined]
    except AttributeError:
        return getpass.getuser() == "root"


# ---------------------------------------------------------------------------
# Shared config reader (Podman model: all packages read one config.yml)
# ---------------------------------------------------------------------------

_config_section_cache: dict[str, dict[str, str]] = {}


def _config_file_paths() -> list[tuple[str, Path]]:
    """Ordered config.yml locations with scope labels (lowest → highest priority).

    ``TEROK_CONFIG_FILE`` → single override (no layering).
    Otherwise: ``/etc/terok/config.yml`` (system) → ``~/.config/terok/config.yml`` (user).
    Root users get only the system path.
    """
    env = os.getenv("TEROK_CONFIG_FILE")
    if env:
        return [("override", Path(env).expanduser())]
    result: list[tuple[str, Path]] = [
        ("system", Path("/etc") / _NAMESPACE / "config.yml"),
    ]
    if not _is_root():
        user_dir = (
            Path(_user_config_dir(_NAMESPACE))
            if _user_config_dir
            else Path.home() / ".config" / _NAMESPACE
        )
        result.append(("user", user_dir / "config.yml"))
    return result


def read_config_section(section: str) -> dict[str, str]:
    """Read a top-level section from layered terok configs (cached, fail-silent).

    Merges system and user config files via [`ConfigStack`][terok_sandbox.ConfigStack] — user
    values override system defaults at the leaf level.
    """
    if section in _config_section_cache:
        return _config_section_cache[section]

    result: dict[str, str] = {}
    try:
        from .config_stack import ConfigStack, load_yaml_scope

        stack = ConfigStack()
        for label, path in _config_file_paths():
            stack.push(load_yaml_scope(label, path))
        merged = stack.resolve_section(section)
        result = {k: str(v) for k, v in merged.items() if v is not None}
    except Exception:  # noqa: BLE001 — fail-silent; bad config should not crash path resolution
        pass
    _config_section_cache[section] = result
    return result


def _read_config_paths() -> dict[str, str]:
    """Read ``paths:`` section — convenience wrapper."""
    return read_config_section("paths")


def _namespace_root() -> Path | None:
    """Return the configured namespace state root, or ``None`` for platform default.

    Resolution: ``TEROK_ROOT`` env var → ``config.yml`` ``paths.root``.
    """
    env = os.getenv(_TEROK_ROOT_ENV)
    if env:
        return Path(env).expanduser()
    val = _read_config_paths().get("root")
    return Path(val).expanduser().resolve() if val else None


# ---------------------------------------------------------------------------
# Generic namespace resolvers (DRY: used by sandbox, agent, and terok)
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
        return Path("/var/lib") / _NAMESPACE
    if _user_data_dir is not None:
        return Path(_user_data_dir(_NAMESPACE))
    xdg = os.getenv("XDG_DATA_HOME")
    return Path(xdg) / _NAMESPACE if xdg else Path.home() / ".local" / "share" / _NAMESPACE


def namespace_state_dir(subdir: str = "", env_var: str | None = None) -> Path:
    """Resolve a state directory under the ``terok/`` namespace.

    Priority:

    1. *env_var* (package-specific override, e.g. ``TEROK_SANDBOX_STATE_DIR``)
    2. ``TEROK_ROOT`` env var (namespace override)
    3. ``config.yml`` → ``paths.root`` (Podman model — all packages honor it)
    4. Platform default (``/var/lib/terok/<subdir>`` or XDG)
    """
    if env_var:
        val = os.getenv(env_var)
        if val:
            return Path(val).expanduser()
    root = _namespace_root()
    base = root if root else _platform_state_base()
    return _safe_subdir(base, subdir)


def namespace_config_dir(subdir: str = "", env_var: str | None = None) -> Path:
    """Resolve a config directory under the ``terok/`` namespace.

    Priority: *env_var* → ``/etc/terok/<subdir>`` (root) → platformdirs
    → ``~/.config/terok/<subdir>``.
    """
    if env_var:
        val = os.getenv(env_var)
        if val:
            return Path(val).expanduser()
    base: Path
    if _is_root():
        base = Path("/etc") / _NAMESPACE
    elif _user_config_dir is not None:
        base = Path(_user_config_dir(_NAMESPACE))
    else:
        base = Path.home() / ".config" / _NAMESPACE
    return _safe_subdir(base, subdir)


def namespace_runtime_dir(subdir: str = "", env_var: str | None = None) -> Path:
    """Resolve a runtime directory under the ``terok/`` namespace.

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
        base = Path("/run") / _NAMESPACE
    else:
        xdg_runtime = os.getenv("XDG_RUNTIME_DIR")
        if xdg_runtime:
            base = Path(xdg_runtime) / _NAMESPACE
        else:
            xdg_state = os.getenv("XDG_STATE_HOME")
            base = (
                Path(xdg_state) / _NAMESPACE
                if xdg_state
                else Path.home() / ".local" / "state" / _NAMESPACE
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
    return namespace_config_dir("sandbox", "TEROK_SANDBOX_CONFIG_DIR")


def state_root() -> Path:
    """Writable state root for sandbox (tasks, tokens, caches).

    Priority: ``TEROK_SANDBOX_STATE_DIR`` → ``/var/lib/terok/sandbox`` (root)
    → ``~/.local/share/terok/sandbox``.
    """
    return namespace_state_dir("sandbox", "TEROK_SANDBOX_STATE_DIR")


def port_registry_dir() -> Path:
    """Shared port registry directory (file-based multi-user isolation).

    Priority: ``TEROK_PORT_REGISTRY_DIR`` env var
    → ``config.yml`` ``paths.port_registry_dir``
    → ``/tmp/terok-ports``.

    Admins on multi-user hosts can point this at a persistent directory
    (e.g. ``/var/lib/terok/ports``) so that port claims survive reboots.
    """
    env = os.getenv("TEROK_PORT_REGISTRY_DIR")
    if env:
        return Path(env).expanduser()
    val = _read_config_paths().get("port_registry_dir")
    if val:
        return Path(val).expanduser().resolve()
    return Path("/tmp/terok-ports")  # nosec B108 — intentional shared registry


def runtime_root() -> Path:
    """Transient runtime directory for sandbox (PID files, sockets).

    Priority: ``TEROK_SANDBOX_RUNTIME_DIR`` → ``/run/terok/sandbox`` (root) →
    ``$XDG_RUNTIME_DIR/terok/sandbox`` → ``$XDG_STATE_HOME/terok/sandbox`` →
    ``~/.local/state/terok/sandbox``.
    """
    return namespace_runtime_dir("sandbox", "TEROK_SANDBOX_RUNTIME_DIR")


def vault_root() -> Path:
    """Shared vault directory used by all terok ecosystem packages.

    Priority: ``TEROK_VAULT_DIR`` → ``/var/lib/terok/vault`` (root)
    → XDG data dir.

    Migration: detects the pre-0.8 ``credentials/`` directory and the legacy
    ``TEROK_CREDENTIALS_DIR`` env var, emitting warnings when found.
    """
    env = os.getenv("TEROK_VAULT_DIR")
    if not env:
        old_env = os.getenv("TEROK_CREDENTIALS_DIR")
        if old_env:
            import logging

            logging.getLogger(__name__).warning(
                "TEROK_CREDENTIALS_DIR is deprecated — use TEROK_VAULT_DIR instead"
            )
            return Path(old_env).expanduser()
    if env:
        return Path(env).expanduser()
    path = namespace_state_dir("vault", "TEROK_VAULT_DIR")
    old = path.parent / "credentials"
    if old.is_dir() and not path.is_dir():
        import logging

        logging.getLogger(__name__).warning(
            "Pre-0.8 credentials directory detected at %s — "
            "run 'terok-migrate-vault' to migrate to %s",
            old,
            path,
        )
    return path


def namespace_config_root() -> Path:
    """Return the top-level terok config root (namespace, not sandbox-scoped).

    Used for cross-package paths like shield profiles that live under
    the shared ``~/.config/terok/`` namespace rather than under any single
    package's config directory.
    """
    return namespace_config_dir("", "TEROK_CONFIG_DIR")
