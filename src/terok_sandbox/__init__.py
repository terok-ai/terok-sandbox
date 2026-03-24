# SPDX-FileCopyrightText: 2025 Jiri Vyskocil
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
from .commands import COMMANDS as SANDBOX_COMMANDS, GATE_COMMANDS, SHIELD_COMMANDS, CommandDef
from .config import SandboxConfig

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

# -- Runtime -----------------------------------------------------------------
from .runtime import (
    find_free_port,
    get_container_state,
    get_project_container_states,
    gpu_run_args,
    is_container_running,
    reserve_free_port,
    stop_task_containers,
    stream_initial_logs,
    wait_for_exit,
)

# -- Facade ------------------------------------------------------------------
from .sandbox import READY_MARKER, RunSpec, Sandbox

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
from .ssh import SSHManager

__all__ = [
    # Config
    "SandboxConfig",
    # Runtime
    "find_free_port",
    "reserve_free_port",
    "get_container_state",
    "get_project_container_states",
    "gpu_run_args",
    "is_container_running",
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
    # Command registry
    "CommandDef",
    "GATE_COMMANDS",
    "SANDBOX_COMMANDS",
    "SHIELD_COMMANDS",
    # Facade
    "READY_MARKER",
    "RunSpec",
    "Sandbox",
    # SSH
    "SSHManager",
    # Meta
    "__version__",
]
