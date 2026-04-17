# SPDX-FileCopyrightText: 2025 Jiri Vyskocil
# SPDX-FileCopyrightText: 2026 Jiri Vyskocil
# SPDX-License-Identifier: Apache-2.0

"""terok-sandbox: hardened Podman container runner with gate and shield integration.

Delegates to domain subsystems:

- :mod:`~.gate` — authenticated git serving: HTTP server, token CRUD, upstream
  mirror management, systemd/daemon lifecycle.
- :mod:`~.credentials` — secret injection: reverse proxy with phantom tokens,
  SSH keypair provisioning, SQLite credential store, systemd/daemon lifecycle.
- :mod:`~.shield` — egress firewall adapter (delegates to terok-shield).
- :mod:`~.runtime` — Podman CLI wrapper (state queries, GPU, log streaming).
- :mod:`~.sandbox` — facade composing the above behind :class:`SandboxConfig`.
- :mod:`~.commands` — CLI command registry and handler implementations.
"""

from __future__ import annotations

__version__: str = "0.0.0"  # placeholder; replaced at build time

from importlib.metadata import PackageNotFoundError, version as _meta_version
from pathlib import Path
from typing import TYPE_CHECKING

from ._util._selinux import (
    SELINUX_SOCKET_TYPE,
    SelinuxCheckResult,
    SelinuxStatus,
    check_status as check_selinux_status,
    install_command as selinux_install_command,
    install_script_path as selinux_install_script,
    is_libselinux_available,
    is_policy_installed as is_selinux_policy_installed,
    is_selinux_enabled,
    is_selinux_enforcing,
    missing_policy_tools as missing_selinux_policy_tools,
    policy_source_path as selinux_policy_source,
)
from .commands import (
    COMMANDS as SANDBOX_COMMANDS,
    DOCTOR_COMMANDS,
    GATE_COMMANDS,
    PROXY_COMMANDS,
    SHIELD_COMMANDS,
    SSH_COMMANDS,
    CommandDef,
    KeyRow,
)
from .config import CONTAINER_RUNTIME_DIR, SandboxConfig
from .config_stack import ConfigScope, ConfigStack
from .credentials.db import CredentialDB
from .credentials.lifecycle import (
    CredentialProxyManager,
    CredentialProxyStatus,
    ProxyUnreachableError,
)
from .credentials.proxy.constants import PHANTOM_CREDENTIALS_MARKER
from .credentials.ssh import SSHManager, generate_keypair, update_ssh_keys_json
from .doctor import CheckVerdict, DoctorCheck, sandbox_doctor_checks
from .gate.lifecycle import GateServerManager, GateServerStatus
from .gate.mirror import GateStalenessInfo, GitGate
from .gate.tokens import TokenStore
from .paths import (
    credentials_root,
    namespace_config_dir,
    namespace_config_root,
    namespace_runtime_dir,
    namespace_state_dir,
    port_registry_dir,
)

# -- Port registry -----------------------------------------------------------
from .port_registry import (
    PORT_RANGE,
    SERVICE_GATE,
    SERVICE_PROXY,
    SERVICE_SSH_AGENT,
    PortRegistry,
    ServicePorts,
    claim_port,
    release_port,
    reset_cache as reset_port_cache,
    resolve_service_ports,
)
from .runtime import (
    ContainerRemoveResult,
    GpuConfigError,
    bypass_network_args,
    container_start,
    container_stop,
    find_free_port,
    get_container_rw_size,
    get_container_rw_sizes,
    get_container_state,
    get_container_states,
    gpu_run_args,
    is_container_running,
    login_command,
    podman_userns_args,
    redact_env_args,
    reserve_free_port,
    sandbox_exec,
    stop_task_containers,
    stream_initial_logs,
    wait_for_exit,
)
from .sandbox import READY_MARKER, LifecycleHooks, RunSpec, Sandbox, Sharing, VolumeSpec
from .shield import (
    EnvironmentCheck,
    NftNotFoundError,
    ShieldNeedsSetup,
    ShieldState,
    block,
    check_environment,
    down,
    make_shield,
    pre_start,
    run_setup,
    setup_hooks_direct,
    state,
    status,
    up,
)

if TYPE_CHECKING:
    pass  # all types already imported above

try:
    __version__ = _meta_version("terok-sandbox")
except PackageNotFoundError:
    pass  # editable install or running from source without metadata


# ---------------------------------------------------------------------------
# Convenience wrappers — default-config delegates for the top-level API.
#
# External consumers import ``from terok_sandbox import get_server_status``
# etc.  These thin wrappers instantiate the manager with the caller's
# (optional) config and delegate to the class method.
# ---------------------------------------------------------------------------


# -- Gate server wrappers ----------------------------------------------------


def check_units_outdated(cfg: SandboxConfig | None = None) -> str | None:
    """Return a warning string if installed systemd units are stale."""
    return GateServerManager(cfg).check_units_outdated()


def ensure_server_reachable(cfg: SandboxConfig | None = None) -> None:
    """Verify the gate server is running and configured correctly."""
    GateServerManager(cfg).ensure_reachable()


def get_gate_base_path(cfg: SandboxConfig | None = None) -> Path:
    """Return the gate base path."""
    return GateServerManager(cfg).gate_base_path


def get_gate_server_port(cfg: SandboxConfig | None = None) -> int | None:
    """Return the configured gate server TCP port, or ``None`` in socket mode."""
    return GateServerManager(cfg).server_port


def get_server_status(cfg: SandboxConfig | None = None) -> GateServerStatus:
    """Return the current gate server status."""
    return GateServerManager(cfg).get_status()


def install_systemd_units(
    cfg: SandboxConfig | None = None, *, transport: str | None = None
) -> None:
    """Render and install gate server systemd units.

    When *transport* is ``None`` (the default), reads ``services.mode``
    from the layered config so callers that don't thread the transport
    explicitly (e.g. the TUI's gate-install action) still pick up the
    user's ``socket`` vs ``tcp`` choice.  Pass an explicit string to
    override the config.
    """
    if transport is None:
        from .config import _services_mode

        transport = _services_mode()
    GateServerManager(cfg).install_systemd_units(transport=transport)


def is_daemon_running(cfg: SandboxConfig | None = None) -> bool:
    """Check whether the gate daemon is alive."""
    return GateServerManager(cfg).is_daemon_running()


def is_systemd_available() -> bool:
    """Check whether ``systemctl --user`` is usable."""
    return GateServerManager().is_systemd_available()


def start_daemon(port: int | None = None, cfg: SandboxConfig | None = None) -> None:
    """Start a gate daemon process."""
    GateServerManager(cfg).start_daemon(port)


def stop_daemon(cfg: SandboxConfig | None = None) -> None:
    """Stop the managed gate daemon."""
    GateServerManager(cfg).stop_daemon()


def uninstall_systemd_units(cfg: SandboxConfig | None = None) -> None:
    """Disable+stop the gate socket and remove unit files."""
    GateServerManager(cfg).uninstall_systemd_units()


# -- Gate token wrappers -----------------------------------------------------


def create_token(scope: str, task_id: str, cfg: SandboxConfig | None = None) -> str:
    """Generate a gate token for a task."""
    return TokenStore(cfg).create(scope, task_id)


def revoke_token_for_task(scope: str, task_id: str, cfg: SandboxConfig | None = None) -> None:
    """Remove all tokens for a scope+task pair."""
    TokenStore(cfg).revoke_for_task(scope, task_id)


# -- Credential proxy wrappers -----------------------------------------------


def ensure_proxy_reachable(cfg: SandboxConfig | None = None) -> None:
    """Verify the credential proxy is running and its TCP ports are up."""
    CredentialProxyManager(cfg).ensure_reachable()


def get_proxy_status(cfg: SandboxConfig | None = None) -> CredentialProxyStatus:
    """Return the current credential proxy status."""
    return CredentialProxyManager(cfg).get_status()


def get_proxy_port(cfg: SandboxConfig | None = None) -> int:
    """Return the configured credential proxy TCP port."""
    return CredentialProxyManager(cfg).proxy_port


def get_ssh_agent_port(cfg: SandboxConfig | None = None) -> int:
    """Return the configured SSH agent proxy TCP port."""
    return CredentialProxyManager(cfg).ssh_agent_port


def install_proxy_systemd(
    cfg: SandboxConfig | None = None, *, transport: str | None = None
) -> None:
    """Render and install credential proxy systemd units for the selected transport.

    When *transport* is ``None`` (the default), reads ``services.mode``
    from the layered config so callers that don't thread the transport
    explicitly (e.g. the TUI's proxy-install action) still pick up the
    user's ``socket`` vs ``tcp`` choice.  Pass an explicit string to
    override the config.
    """
    if transport is None:
        from .config import _services_mode

        transport = _services_mode()
    CredentialProxyManager(cfg).install_systemd_units(transport=transport)


def is_proxy_running(cfg: SandboxConfig | None = None) -> bool:
    """Check whether the managed proxy daemon is alive."""
    return CredentialProxyManager(cfg).is_daemon_running()


def is_proxy_service_active() -> bool:
    """Check whether the credential proxy service unit is active."""
    return CredentialProxyManager().is_service_active()


def is_proxy_socket_active() -> bool:
    """Check whether the credential proxy socket unit is active."""
    return CredentialProxyManager().is_socket_active()


def is_proxy_socket_installed() -> bool:
    """Check whether the credential proxy socket unit file exists."""
    return CredentialProxyManager().is_socket_installed()


def is_proxy_systemd_available() -> bool:
    """Check whether the systemd user session is reachable."""
    return CredentialProxyManager().is_systemd_available()


def start_proxy(cfg: SandboxConfig | None = None) -> None:
    """Start the credential proxy as a background daemon."""
    CredentialProxyManager(cfg).start_daemon()


def stop_proxy(cfg: SandboxConfig | None = None) -> None:
    """Stop the managed proxy daemon."""
    CredentialProxyManager(cfg).stop_daemon()


def uninstall_proxy_systemd(cfg: SandboxConfig | None = None) -> None:
    """Disable+stop the socket and remove unit files."""
    CredentialProxyManager(cfg).uninstall_systemd_units()


__all__ = [
    # Config
    "CONTAINER_RUNTIME_DIR",
    "ConfigScope",
    "ConfigStack",
    "SandboxConfig",
    # Lifecycle managers
    "CredentialProxyManager",
    "GateServerManager",
    "TokenStore",
    "credentials_root",
    "namespace_config_dir",
    "namespace_config_root",
    "namespace_runtime_dir",
    "namespace_state_dir",
    "port_registry_dir",
    # Runtime
    "ContainerRemoveResult",
    "GpuConfigError",
    "bypass_network_args",
    "container_start",
    "container_stop",
    "find_free_port",
    "get_container_rw_size",
    "get_container_rw_sizes",
    "get_container_state",
    "get_container_states",
    "gpu_run_args",
    "is_container_running",
    "login_command",
    "podman_userns_args",
    "redact_env_args",
    "reserve_free_port",
    "sandbox_exec",
    "stop_task_containers",
    "stream_initial_logs",
    "wait_for_exit",
    # Gate server
    "GateServerStatus",
    "check_units_outdated",
    "ensure_server_reachable",
    "get_gate_base_path",
    "get_gate_server_port",
    "get_server_status",
    "install_systemd_units",
    "is_daemon_running",
    "is_systemd_available",
    "start_daemon",
    "stop_daemon",
    "uninstall_systemd_units",
    # Gate tokens
    "create_token",
    "revoke_token_for_task",
    # Shield
    "EnvironmentCheck",
    "NftNotFoundError",
    "ShieldNeedsSetup",
    "ShieldState",
    "block",
    "check_environment",
    "down",
    "make_shield",
    "pre_start",
    "run_setup",
    "setup_hooks_direct",
    "state",
    "status",
    "up",
    # Git gate
    "GateStalenessInfo",
    "GitGate",
    # Credential constants
    "PHANTOM_CREDENTIALS_MARKER",
    # Credential DB
    "CredentialDB",
    # Credential proxy
    "CredentialProxyStatus",
    "ProxyUnreachableError",
    "ensure_proxy_reachable",
    "get_proxy_port",
    "get_proxy_status",
    "get_ssh_agent_port",
    "install_proxy_systemd",
    "is_proxy_running",
    "is_proxy_service_active",
    "is_proxy_socket_active",
    "is_proxy_socket_installed",
    "is_proxy_systemd_available",
    "start_proxy",
    "stop_proxy",
    "uninstall_proxy_systemd",
    # Command registry
    "CommandDef",
    "KeyRow",
    "DOCTOR_COMMANDS",
    "GATE_COMMANDS",
    "PROXY_COMMANDS",
    "SANDBOX_COMMANDS",
    "SHIELD_COMMANDS",
    "SSH_COMMANDS",
    # Facade
    "READY_MARKER",
    "LifecycleHooks",
    "RunSpec",
    "Sandbox",
    "Sharing",
    "VolumeSpec",
    # Doctor (container health checks)
    "CheckVerdict",
    "DoctorCheck",
    "sandbox_doctor_checks",
    # SSH
    "SSHManager",
    "generate_keypair",
    "update_ssh_keys_json",
    # Port registry
    "PORT_RANGE",
    "PortRegistry",
    "SERVICE_GATE",
    "SERVICE_PROXY",
    "SERVICE_SSH_AGENT",
    "ServicePorts",
    "claim_port",
    "release_port",
    "reset_port_cache",
    "resolve_service_ports",
    # SELinux
    "SELINUX_SOCKET_TYPE",
    "SelinuxCheckResult",
    "SelinuxStatus",
    "check_selinux_status",
    "is_libselinux_available",
    "is_selinux_enabled",
    "is_selinux_enforcing",
    "is_selinux_policy_installed",
    "missing_selinux_policy_tools",
    "selinux_install_command",
    "selinux_install_script",
    "selinux_policy_source",
    # Meta
    "__version__",
]
