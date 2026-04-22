# SPDX-FileCopyrightText: 2025 Jiri Vyskocil
# SPDX-FileCopyrightText: 2026 Jiri Vyskocil
# SPDX-License-Identifier: Apache-2.0

"""terok-sandbox: hardened Podman container runner with gate and shield integration.

Delegates to domain subsystems:

- :mod:`~.gate` — authenticated git serving: HTTP server, token CRUD, upstream
  mirror management, systemd/daemon lifecycle.
- :mod:`~.vault` — secret injection: token broker with phantom credentials,
  SSH signing proxy, SQLite credential store, systemd/daemon lifecycle.
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
    SETUP_COMMANDS,
    SHIELD_COMMANDS,
    SSH_COMMANDS,
    VAULT_COMMANDS,
    CommandDef,
    KeyRow,
)
from .config import CONTAINER_RUNTIME_DIR, SandboxConfig
from .config_stack import ConfigScope, ConfigStack
from .credentials.db import CredentialDB, SSHKeyRecord, SSHKeyRow
from .credentials.ssh import SSHInitResult, SSHManager
from .credentials.ssh_keypair import (
    DEFAULT_RSA_BITS,
    ExportResult,
    GeneratedKeypair,
    ImportResult,
    KeypairMismatchError,
    PasswordProtectedKeyError,
    UnsafeCommentError,
    export_ssh_keypair,
    fingerprint_of,
    generate_keypair,
    import_ssh_keypair,
    parse_openssh_keypair,
    public_line_of,
)
from .doctor import CheckVerdict, DoctorCheck, sandbox_doctor_checks
from .gate.lifecycle import GateServerManager, GateServerStatus
from .gate.mirror import GateAuthNotConfigured, GateStalenessInfo, GitGate, is_ssh_url
from .gate.tokens import TokenStore
from .paths import (
    namespace_config_dir,
    namespace_config_root,
    namespace_runtime_dir,
    namespace_state_dir,
    port_registry_dir,
    vault_root,
)
from .podman import ContainerInfo, PodmanInspector

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
    Container,
    ContainerRemoveResult,
    ContainerRuntime,
    ExecResult,
    GpuConfigError,
    Image,
    LogStream,
    NullRuntime,
    PodmanRuntime,
    PortReservation,
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
    install_shield_bridge,
    make_shield,
    pre_start,
    reader_script_path,
    resolve_container_state_dir,
    run_setup,
    run_uninstall,
    setup_hooks_direct,
    shield_interactive_session,
    shield_watch_session,
    state,
    status,
    uninstall_hooks_direct,
    uninstall_shield_bridge,
    up,
)
from .vault.constants import PHANTOM_CREDENTIALS_MARKER
from .vault.lifecycle import (
    VaultManager,
    VaultStatus,
    VaultUnreachableError,
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


# -- Vault wrappers ----------------------------------------------------------


def ensure_vault_reachable(cfg: SandboxConfig | None = None) -> None:
    """Verify the vault is running and its TCP ports are up."""
    VaultManager(cfg).ensure_reachable()


def get_vault_status(cfg: SandboxConfig | None = None) -> VaultStatus:
    """Return the current vault status."""
    return VaultManager(cfg).get_status()


def get_token_broker_port(cfg: SandboxConfig | None = None) -> int:
    """Return the configured token broker TCP port."""
    return VaultManager(cfg).token_broker_port


def get_ssh_signer_port(cfg: SandboxConfig | None = None) -> int:
    """Return the configured SSH signer TCP port."""
    return VaultManager(cfg).ssh_signer_port


def install_vault_systemd(
    cfg: SandboxConfig | None = None, *, transport: str | None = None
) -> None:
    """Render and install vault systemd units for the selected transport.

    When *transport* is ``None`` (the default), reads ``services.mode``
    from the layered config so callers that don't thread the transport
    explicitly (e.g. the TUI's vault-install action) still pick up the
    user's ``socket`` vs ``tcp`` choice.  Pass an explicit string to
    override the config.
    """
    if transport is None:
        from .config import _services_mode

        transport = _services_mode()
    VaultManager(cfg).install_systemd_units(transport=transport)


def is_vault_running(cfg: SandboxConfig | None = None) -> bool:
    """Check whether the managed vault daemon is alive."""
    return VaultManager(cfg).is_daemon_running()


def is_vault_service_active() -> bool:
    """Check whether the vault service unit is active."""
    return VaultManager().is_service_active()


def is_vault_socket_active() -> bool:
    """Check whether the vault socket unit is active."""
    return VaultManager().is_socket_active()


def is_vault_socket_installed() -> bool:
    """Check whether the vault socket unit file exists."""
    return VaultManager().is_socket_installed()


def is_vault_systemd_available() -> bool:
    """Check whether the systemd user session is reachable."""
    return VaultManager().is_systemd_available()


def start_vault(cfg: SandboxConfig | None = None) -> None:
    """Start the vault as a background daemon."""
    VaultManager(cfg).start_daemon()


def stop_vault(cfg: SandboxConfig | None = None) -> None:
    """Stop the managed vault daemon."""
    VaultManager(cfg).stop_daemon()


def uninstall_vault_systemd(cfg: SandboxConfig | None = None) -> None:
    """Disable+stop the socket and remove unit files."""
    VaultManager(cfg).uninstall_systemd_units()


__all__ = [
    # Config
    "CONTAINER_RUNTIME_DIR",
    "ConfigScope",
    "ConfigStack",
    "SandboxConfig",
    # Lifecycle managers
    "VaultManager",
    "GateServerManager",
    "TokenStore",
    "vault_root",
    "namespace_config_dir",
    "namespace_config_root",
    "namespace_runtime_dir",
    "namespace_state_dir",
    "port_registry_dir",
    # Runtime protocol + backends
    "Container",
    "ContainerRemoveResult",
    "ContainerRuntime",
    "ExecResult",
    "GpuConfigError",
    "Image",
    "LogStream",
    "NullRuntime",
    "PodmanRuntime",
    "PortReservation",
    # Podman inspector (container metadata lookup for terok-aware callers)
    "ContainerInfo",
    "PodmanInspector",
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
    "install_shield_bridge",
    "make_shield",
    "pre_start",
    "reader_script_path",
    "resolve_container_state_dir",
    "run_setup",
    "run_uninstall",
    "setup_hooks_direct",
    "shield_interactive_session",
    "shield_watch_session",
    "state",
    "status",
    "uninstall_hooks_direct",
    "uninstall_shield_bridge",
    "up",
    # Git gate
    "GateAuthNotConfigured",
    "GateStalenessInfo",
    "GitGate",
    "is_ssh_url",
    # Credential constants
    "PHANTOM_CREDENTIALS_MARKER",
    # Credential DB
    "CredentialDB",
    "SSHKeyRecord",
    "SSHKeyRow",
    # Vault
    "VaultStatus",
    "VaultUnreachableError",
    "ensure_vault_reachable",
    "get_token_broker_port",
    "get_vault_status",
    "get_ssh_signer_port",
    "install_vault_systemd",
    "is_vault_running",
    "is_vault_service_active",
    "is_vault_socket_active",
    "is_vault_socket_installed",
    "is_vault_systemd_available",
    "start_vault",
    "stop_vault",
    "uninstall_vault_systemd",
    # Command registry
    "CommandDef",
    "KeyRow",
    "DOCTOR_COMMANDS",
    "GATE_COMMANDS",
    "SETUP_COMMANDS",
    "VAULT_COMMANDS",
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
    "DEFAULT_RSA_BITS",
    "ExportResult",
    "GeneratedKeypair",
    "ImportResult",
    "KeypairMismatchError",
    "PasswordProtectedKeyError",
    "SSHInitResult",
    "SSHManager",
    "UnsafeCommentError",
    "export_ssh_keypair",
    "generate_keypair",
    "import_ssh_keypair",
    "fingerprint_of",
    "parse_openssh_keypair",
    "public_line_of",
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
