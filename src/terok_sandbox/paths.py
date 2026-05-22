# SPDX-FileCopyrightText: 2025 Jiri Vyskocil
# SPDX-FileCopyrightText: 2026 Jiri Vyskocil
# SPDX-License-Identifier: Apache-2.0

"""Platform-aware path resolution for the terok ecosystem.

Provides generic **namespace resolvers** that any sibling package can call
to place its state/config/runtime under the shared ``terok/`` namespace,
plus sandbox-specific thin wrappers that bind sandbox's own subsystems
(vault, gate, hooks) to those resolvers.
"""

import getpass
import os
from collections.abc import Callable
from pathlib import Path

try:
    from platformdirs import user_config_dir, user_data_dir

    _user_config_dir: Callable[..., str] | None = user_config_dir
    _user_data_dir: Callable[..., str] | None = user_data_dir
except ImportError:  # optional dependency
    _user_config_dir = None
    _user_data_dir = None


_NAMESPACE = "terok"

_TEROK_ROOT_ENV = "TEROK_ROOT"
"""Env var overriding the namespace state root for all ecosystem packages."""


def _is_root() -> bool:
    """Return True if the current process is running as root."""
    try:
        return os.geteuid() == 0
    except AttributeError:
        return getpass.getuser() == "root"


# ---------------------------------------------------------------------------
# Shared config reader (Podman model: all packages read one config.yml)
# ---------------------------------------------------------------------------

_config_section_cache: dict[str, dict[str, str]] = {}
_config_top_level_cache: dict[str, object | None] = {}


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
    except Exception:  # noqa: BLE001 — fail-silent; bad config should not crash path resolution  # nosec B110 — best-effort probe; failure is non-fatal
        pass
    _config_section_cache[section] = result
    return result


def read_config_top_level(key: str) -> object | None:
    """Read a top-level scalar/list from the layered config (cached, fail-silent).

    Counterpart to [`read_config_section`][terok_sandbox.paths.read_config_section]
    for keys whose value isn't a dict — e.g. the ecosystem-wide
    ``experimental: true`` opt-in.  Returns the merged value or
    ``None`` when the key is absent or the config files can't be loaded.
    """
    if key in _config_top_level_cache:
        return _config_top_level_cache[key]

    result: object | None = None
    try:
        from .config_stack import ConfigStack, load_yaml_scope

        stack = ConfigStack()
        for label, path in _config_file_paths():
            stack.push(load_yaml_scope(label, path))
        result = stack.resolve().get(key)
    except Exception:  # noqa: BLE001 — fail-silent; bad config should not crash field resolution  # nosec B110 — best-effort probe; failure is non-fatal
        pass
    _config_top_level_cache[key] = result
    return result


def _read_config_paths() -> dict[str, str]:
    """Read ``paths:`` section — convenience wrapper."""
    return read_config_section("paths")


def plaintext_passphrase_config_path() -> Path | None:
    """Locate the config file that sets ``credentials.passphrase`` (or ``None``).

    Walks the same layered files as
    [`read_config_section`][terok_sandbox.paths.read_config_section]
    and returns the *highest-priority* path that explicitly sets
    ``passphrase`` — useful for the visibility WARNING (sandbox#282),
    which needs to name the file so the operator can clean it up
    before moving to a sealed tier.

    Lives in ``paths`` rather than ``config`` so the tach foundation
    layer stays free of the ``config_stack`` import; the helper is a
    file-locator, not a schema-validated reader.

    Per-file failures (unreadable, malformed YAML, non-mapping
    top-level) are swallowed and the walk continues — same fail-silent
    contract as [`read_config_section`][terok_sandbox.paths.read_config_section],
    since this helper feeds visibility surfaces (``vault status``,
    sickbay) that must never crash on a bad config layer.
    """
    from .config_stack import load_yaml_scope

    found: Path | None = None
    for label, path in _config_file_paths():
        try:
            scope = load_yaml_scope(label, path)
            creds = scope.data.get("credentials") if isinstance(scope.data, dict) else None
            if isinstance(creds, dict) and creds.get("passphrase"):
                found = path
        except Exception:  # noqa: BLE001 — fail-silent; bad config layer must not crash the walk  # nosec B112 — best-effort iteration; skip-on-error keeps the scan going
            continue
    return found


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
