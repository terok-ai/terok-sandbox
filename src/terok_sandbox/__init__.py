# SPDX-FileCopyrightText: 2025 Jiri Vyskocil
# SPDX-FileCopyrightText: 2026 Jiri Vyskocil
# SPDX-License-Identifier: Apache-2.0

"""terok-sandbox: hardened Podman container runner with gate and shield integration.

Public API for standalone use and integration with terok.

The primary configuration type is :class:`SandboxConfig`:

    >>> from terok_sandbox import SandboxConfig
    >>> cfg = SandboxConfig(gate_port=9418)
"""

__version__: str = "0.0.0"  # placeholder; replaced at build time

from importlib.metadata import PackageNotFoundError, version as _meta_version

try:
    __version__ = _meta_version("terok-sandbox")
except PackageNotFoundError:
    pass  # editable install or running from source without metadata

# -- Config ------------------------------------------------------------------
# -- Command registry --------------------------------------------------------
from .commands import (
    COMMANDS as SANDBOX_COMMANDS,
    GATE_COMMANDS,
    PROXY_COMMANDS,
    SHIELD_COMMANDS,
    SSH_COMMANDS,
    CommandDef,
)
from .config import SandboxConfig

# -- Credential DB -----------------------------------------------------------
from .credential_db import CredentialDB

# -- Credential proxy lifecycle ----------------------------------------------
from .credential_proxy_lifecycle import (
    CredentialProxyStatus,
    ensure_proxy_reachable,
    get_proxy_port,
    get_proxy_status,
    get_ssh_agent_port,
    install_systemd_units as install_proxy_systemd,
    is_daemon_running as is_proxy_running,
    is_socket_active as is_proxy_socket_active,
    is_socket_installed as is_proxy_socket_installed,
    is_systemd_available as is_proxy_systemd_available,
    start_daemon as start_proxy,
    stop_daemon as stop_proxy,
    uninstall_systemd_units as uninstall_proxy_systemd,
)

# -- Gate server -------------------------------------------------------------
from .gate_server import (
    GateServerStatus,
    check_units_outdated,
    ensure_server_reachable,
    get_gate_base_path,
    get_gate_server_port,
    get_server_status,
    install_systemd_units,
    is_daemon_running,
    is_systemd_available,
    start_daemon,
    stop_daemon,
    uninstall_systemd_units,
)

# -- Gate tokens -------------------------------------------------------------
from .gate_tokens import create_token, revoke_token_for_task

# -- Git gate ----------------------------------------------------------------
from .git_gate import GateStalenessInfo, GitGate

# -- Paths -------------------------------------------------------------------
from .paths import credentials_root, umbrella_config_root

# -- Runtime -----------------------------------------------------------------
from .runtime import (
    GpuConfigError,
    bypass_network_args,
    find_free_port,
    get_container_state,
    get_project_container_states,
    gpu_run_args,
    is_container_running,
    podman_userns_args,
    redact_env_args,
    reserve_free_port,
    stop_task_containers,
    stream_initial_logs,
    wait_for_exit,
)

# -- Facade ------------------------------------------------------------------
from .sandbox import READY_MARKER, LifecycleHooks, RunSpec, Sandbox

# -- Shield ------------------------------------------------------------------
from .shield import (
    EnvironmentCheck,
    NftNotFoundError,
    ShieldNeedsSetup,
    ShieldState,
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

# -- SSH ---------------------------------------------------------------------
from .ssh import SSHManager, generate_keypair, update_ssh_keys_json

__all__ = [
    # Config
    "SandboxConfig",
    "credentials_root",
    "umbrella_config_root",
    # Runtime
    "GpuConfigError",
    "bypass_network_args",
    "find_free_port",
    "get_container_state",
    "get_project_container_states",
    "gpu_run_args",
    "is_container_running",
    "podman_userns_args",
    "redact_env_args",
    "reserve_free_port",
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
    # Credential DB
    "CredentialDB",
    # Credential proxy
    "CredentialProxyStatus",
    "ensure_proxy_reachable",
    "get_proxy_port",
    "get_proxy_status",
    "get_ssh_agent_port",
    "install_proxy_systemd",
    "is_proxy_running",
    "is_proxy_socket_active",
    "is_proxy_socket_installed",
    "is_proxy_systemd_available",
    "start_proxy",
    "stop_proxy",
    "uninstall_proxy_systemd",
    # Command registry
    "CommandDef",
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
    # SSH
    "SSHManager",
    "generate_keypair",
    "update_ssh_keys_json",
    # Meta
    "__version__",
]
