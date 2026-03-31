# SPDX-FileCopyrightText: 2026 Jiri Vyskocil
# SPDX-License-Identifier: Apache-2.0

"""Sandbox configuration — plain dataclass for standalone and embedded use.

:class:`SandboxConfig` captures directory paths and settings that sandbox
modules need.  In standalone ``terok-sandbox`` use, it is resolved from
environment variables and XDG defaults.  When embedded in terok, the
orchestration layer constructs it from :func:`core.config` values.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

from .paths import (
    config_root as _config_root,
    credentials_root as _credentials_root,
    runtime_root as _runtime_root,
    state_root as _state_root,
)


@dataclass(frozen=True)
class SandboxConfig:
    """Immutable configuration for the sandbox layer.

    All paths default to the XDG/FHS-resolved values from :mod:`paths`.
    Override individual fields when constructing from terok's global config
    or when using terok-sandbox standalone.
    """

    state_dir: Path = field(default_factory=_state_root)
    """Writable state root (tokens, gate repos, task data)."""

    runtime_dir: Path = field(default_factory=_runtime_root)
    """Transient runtime directory (PID files, sockets)."""

    config_dir: Path = field(default_factory=_config_root)
    """Configuration root (shield profiles)."""

    credentials_dir: Path = field(default_factory=_credentials_root)
    """Shared credentials directory (DB, routes, env mounts)."""

    gate_port: int = 9418
    """HTTP port for the gate server."""

    proxy_port: int = 18731
    """TCP port for the credential proxy (container access)."""

    ssh_agent_port: int = 18732
    """TCP port for the SSH agent proxy (container access)."""

    shield_profiles: tuple[str, ...] = ("dev-standard",)
    """Shield egress firewall profile names."""

    shield_audit: bool = True
    """Whether shield audit logging is enabled."""

    shield_bypass: bool = False
    """DANGEROUS: when True, the egress firewall is completely disabled."""

    @property
    def effective_envs_dir(self) -> Path:
        """Return the shared agent auth mount directory."""
        return self.credentials_dir / "envs"

    @property
    def gate_base_path(self) -> Path:
        """Return the gate server's repo base path."""
        return self.state_dir / "gate"

    @property
    def token_file_path(self) -> Path:
        """Return the path to the gate token file."""
        return self.state_dir / "gate" / "tokens.json"

    @property
    def pid_file_path(self) -> Path:
        """Return the PID file path for the managed gate daemon."""
        return self.runtime_dir / "gate-server.pid"

    @property
    def shield_profiles_dir(self) -> Path:
        """Return the directory for terok-managed shield profiles."""
        return self.config_dir / "shield" / "profiles"

    @property
    def proxy_db_path(self) -> Path:
        """Return the path to the credential proxy sqlite3 database."""
        return self.credentials_dir / "credentials.db"

    @property
    def proxy_socket_path(self) -> Path:
        """Return the Unix socket path for the credential proxy."""
        return self.runtime_dir / "credential-proxy.sock"

    @property
    def proxy_pid_file_path(self) -> Path:
        """Return the PID file path for the managed credential proxy daemon."""
        return self.runtime_dir / "credential-proxy.pid"

    @property
    def proxy_routes_path(self) -> Path:
        """Return the path to the proxy route configuration JSON."""
        return self.credentials_dir / "routes.json"

    @property
    def ssh_keys_dir(self) -> Path:
        """Return the base directory for per-project SSH keys."""
        return self.state_dir / "ssh-keys"

    @property
    def ssh_keys_json_path(self) -> Path:
        """Return the path to the SSH key mapping JSON."""
        return self.credentials_dir / "ssh-keys.json"
