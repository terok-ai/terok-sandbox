# SPDX-FileCopyrightText: 2025 Jiri Vyskocil
# SPDX-FileCopyrightText: 2026 Jiri Vyskocil
# SPDX-License-Identifier: Apache-2.0

"""Platform-aware path resolution for the terok-sandbox subsystems.

Generic namespace resolvers ([`namespace_state_dir`][terok_util.paths.namespace_state_dir]
and friends) and the layered config readers
([`read_config_section`][terok_util.paths.read_config_section],
[`read_config_top_level`][terok_util.paths.read_config_top_level])
live in [`terok_util.paths`][terok_util.paths] now.  This module
re-exports them so the existing
``from .paths import namespace_state_dir`` callsites keep working,
and adds sandbox-specific thin wrappers that bind sandbox's own
subsystems (vault, gate, hooks) to those resolvers.
"""

import os
from pathlib import Path

from terok_util import (
    namespace_config_dir,
    namespace_runtime_dir,
    namespace_state_dir,
    read_config_section,
    read_config_top_level,
)
from terok_util.paths import _config_file_paths  # noqa: F401 — re-exported

__all__ = [
    "config_root",
    "namespace_config_dir",
    "namespace_config_root",
    "namespace_runtime_dir",
    "namespace_state_dir",
    "plaintext_passphrase_config_path",
    "port_registry_dir",
    "read_config_section",
    "read_config_top_level",
    "runtime_root",
    "state_root",
    "vault_root",
]


def _read_config_paths() -> dict[str, str]:
    """Read ``paths:`` section — convenience wrapper."""
    return read_config_section("paths")


def plaintext_passphrase_config_path() -> Path | None:
    """Locate the config file that sets ``credentials.passphrase`` (or ``None``).

    Walks the same layered files as
    [`read_config_section`][terok_util.paths.read_config_section]
    and returns the *highest-priority* path that explicitly sets
    ``passphrase`` — useful for the visibility WARNING (sandbox#282),
    which needs to name the file so the operator can clean it up
    before moving to a sealed tier.

    Per-file failures (unreadable, malformed YAML, non-mapping
    top-level) are swallowed and the walk continues — same fail-silent
    contract as [`read_config_section`][terok_util.paths.read_config_section],
    since this helper feeds visibility surfaces (``vault status``,
    sickbay) that must never crash on a bad config layer.
    """
    from terok_util.config_stack import load_yaml_scope

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


# ---------------------------------------------------------------------------
# Sandbox-specific thin wrappers (preserve existing API)
# ---------------------------------------------------------------------------


def config_root() -> Path:
    """Base directory for sandbox configuration.

    Priority: ``TEROK_SANDBOX_CONFIG_DIR`` → ``/etc/terok/sandbox`` (root)
    → ``~/.config/terok/sandbox``.
    """
    return namespace_config_dir("sandbox", env_var="TEROK_SANDBOX_CONFIG_DIR")


def state_root() -> Path:
    """Writable state root for sandbox (tasks, tokens, caches).

    Priority: ``TEROK_SANDBOX_STATE_DIR`` → ``/var/lib/terok/sandbox`` (root)
    → ``~/.local/share/terok/sandbox``.
    """
    return namespace_state_dir("sandbox", env_var="TEROK_SANDBOX_STATE_DIR")


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
    return namespace_runtime_dir("sandbox", env_var="TEROK_SANDBOX_RUNTIME_DIR")


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
    path = namespace_state_dir("vault", env_var="TEROK_VAULT_DIR")
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
    return namespace_config_dir("", env_var="TEROK_CONFIG_DIR")
