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
    """Sandbox-scoped configuration root.

    Note: shield profiles are resolved by :attr:`shield_profiles_dir`
    via :func:`~terok_sandbox.paths.namespace_config_root`, not from
    this directory.
    """

    credentials_dir: Path = field(default_factory=_credentials_root)
    """Shared credentials directory (DB, routes, env mounts)."""

    gate_port: int | None = None
    """HTTP port for the gate server (``None`` = auto-allocate via registry)."""

    proxy_port: int | None = None
    """TCP port for the credential proxy (``None`` = auto-allocate via registry)."""

    ssh_agent_port: int | None = None
    """TCP port for the SSH agent proxy (``None`` = auto-allocate via registry)."""

    shield_profiles: tuple[str, ...] = ("dev-standard",)
    """Shield egress firewall profile names."""

    shield_audit: bool = True
    """Whether shield audit logging is enabled."""

    shield_bypass: bool = False
    """DANGEROUS: when True, the egress firewall is completely disabled."""

    def __post_init__(self) -> None:
        """Auto-resolve ``None`` ports via the shared port registry."""
        if self.gate_port is None or self.proxy_port is None or self.ssh_agent_port is None:
            from .port_registry import resolve_service_ports

            ports = resolve_service_ports(
                self.gate_port,
                self.proxy_port,
                self.ssh_agent_port,
                gate_explicit=self.gate_port is not None,
                proxy_explicit=self.proxy_port is not None,
                ssh_explicit=self.ssh_agent_port is not None,
                state_dir=self.state_dir,
            )
            if self.gate_port is None:
                object.__setattr__(self, "gate_port", ports.gate)
            if self.proxy_port is None:
                object.__setattr__(self, "proxy_port", ports.proxy)
            if self.ssh_agent_port is None:
                object.__setattr__(self, "ssh_agent_port", ports.ssh_agent)

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
        from .paths import namespace_config_root

        return namespace_config_root() / "shield" / "profiles"

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
    def clone_cache_base_path(self) -> Path:
        """Return the base directory for per-scope non-bare clone caches."""
        return self.state_dir / "clone-cache"

    @property
    def ssh_keys_dir(self) -> Path:
        """Return the base directory for per-scope SSH keys."""
        return self.state_dir / "ssh-keys"

    @property
    def ssh_keys_json_path(self) -> Path:
        """Return the path to the SSH key mapping JSON."""
        return self.credentials_dir / "ssh-keys.json"
