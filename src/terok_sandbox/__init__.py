# SPDX-FileCopyrightText: 2025 Jiri Vyskocil
# SPDX-License-Identifier: Apache-2.0

"""terok-sandbox: hardened Podman container runner with gate and shield integration.

Delegates to domain subsystems:

- [`gate`][terok_sandbox.gate] — authenticated git serving: HTTP server, token CRUD, upstream
  mirror management, systemd/daemon lifecycle.
- [`vault`][terok_sandbox.vault] — secret injection: token broker with phantom credentials,
  SSH signing proxy, SQLite credential store, systemd/daemon lifecycle.
- [`shield`][terok_sandbox.integrations.shield] — egress firewall adapter (delegates to terok-shield).
- [`runtime`][terok_sandbox.runtime] — Podman CLI wrapper (state queries, GPU, log streaming).
- [`sandbox`][terok_sandbox.sandbox] — facade composing the above behind [`SandboxConfig`][terok_sandbox.SandboxConfig].
- [`commands`][terok_sandbox.commands] — CLI command registry and handler implementations.
"""

from __future__ import annotations

__version__: str = "0.0.0"  # placeholder; replaced at build time

from importlib.metadata import PackageNotFoundError, version as _meta_version
from typing import TYPE_CHECKING

from terok_util import ConfigStack, sanitize_tty
from terok_util.config_stack import ConfigScope

from ._exit_codes import EXIT_MANUAL_STEP_NEEDED
from ._stage import (
    STAGE_WIDTH,
    Marker,
    StageLine,
    bold,
    red,
    stage,
    stage_begin,
    stage_end,
    stage_line,
    supports_color,
    yellow,
)
from ._util import BestEffortLogger
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
from ._yaml import update_section as yaml_update_section
from .commands import (
    COMMANDS as SANDBOX_COMMANDS,
    DOCTOR_COMMANDS,
    GATE_COMMANDS,
    SETUP_COMMANDS,
    SHIELD_COMMANDS,
    SSH_COMMANDS,
    VAULT_COMMANDS,
    ArgDef,
    CommandDef,
    CommandTree,
    KeyRow,
    _handle_sandbox_setup as sandbox_setup,
    _handle_sandbox_uninstall as sandbox_uninstall,
    handle_vault_seal,
    handle_vault_to_keyring,
)
from .config import CONTAINER_RUNTIME_DIR, SandboxConfig
from .config_schema import (
    SERVICES_TCP_OPTOUT_YAML,
    RawCredentialsSection,
    RawGateServerSection,
    RawHooksSection,
    RawNetworkSection,
    RawPathsSection,
    RawRunSection,
    RawServicesSection,
    RawShieldSection,
    RawSSHSection,
    RawVaultSection,
    SandboxConfigView,
    ServicesMode,
    gate_use_personal_ssh_default,
)
from .doctor import (
    CheckVerdict,
    DoctorCheck,
    make_recovery_acknowledged_check,
    sandbox_doctor_checks,
)
from .gate.lifecycle import GateServerManager, GateServerStatus
from .gate.mirror import GateAuthNotConfigured, GateStalenessInfo, GitGate, is_ssh_url
from .gate.tokens import TokenStore
from .integrations.shield import (
    EnvironmentCheck,
    NftNotFoundError,
    ShieldNeedsSetup,
    ShieldRuntime,
    ShieldState,
    check_environment,
    down,
    make_shield,
    pre_start,
    quarantine,
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
    up,
)
from .paths import (
    namespace_config_dir,
    namespace_config_root,
    namespace_runtime_dir,
    namespace_state_dir,
    port_registry_dir,
    vault_root,
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
    DEFAULT_GUEST_SSHD_PORT,
    DEFAULT_SSH_HOST,
    DEFAULT_SSH_USER,
    Container,
    ContainerRemoveResult,
    ContainerRuntime,
    ExecResult,
    FakeKrunTransport,
    GpuConfigError,
    Image,
    KrunContainer,
    KrunRuntime,
    KrunTransport,
    LogStream,
    NullRuntime,
    PodmanRuntime,
    PortReservation,
    TcpEndpoint,
    TcpSSHTransport,
    podman_port_resolver,
)
from .sandbox import READY_MARKER, LifecycleHooks, RunSpec, Sandbox, Sharing, VolumeSpec
from .setup_stamp import (
    SetupVerdict,
    clear_stamp,
    installed_versions,
    needs_setup,
    read_stamp,
    stamp_path,
    write_stamp,
)
from .vault.daemon.constants import CODEX_SHARED_OAUTH_MARKER, PHANTOM_CREDENTIALS_MARKER
from .vault.daemon.lifecycle import (
    VaultManager,
    VaultStatus,
    VaultUnreachableError,
)
from .vault.ssh.keypair import (
    DEFAULT_RSA_BITS,
    ExportResult,
    GeneratedKeypair,
    ImportResult,
    InfraKeypair,
    KeypairMismatchError,
    PasswordProtectedKeyError,
    ensure_infra_keypair,
    export_ssh_keypair,
    fingerprint_of,
    generate_keypair,
    import_ssh_keypair,
    openssh_pem_of,
    parse_openssh_keypair,
    public_line_of,
)
from .vault.ssh.manager import SSHInitResult, SSHManager
from .vault.store.db import CredentialDB, SSHKeyRecord, SSHKeyRow, UnsafeCommentError
from .vault.store.encryption import NoPassphraseError, PassphraseSource, WrongPassphraseError
from .vault.store.recovery import RecoveryStatus
from .vault.store.systemd_creds import (
    has_tpm2 as systemd_creds_has_tpm2,
    is_available as is_systemd_creds_available,
)

if TYPE_CHECKING:
    pass  # all types already imported above

try:
    __version__ = _meta_version("terok-sandbox")
except PackageNotFoundError:
    pass  # editable install or running from source without metadata


__all__ = [
    # Cross-package utilities
    "BestEffortLogger",
    "EXIT_MANUAL_STEP_NEEDED",
    "sanitize_tty",
    "yaml_update_section",
    # Config
    "CONTAINER_RUNTIME_DIR",
    "ConfigScope",
    "ConfigStack",
    "SandboxConfig",
    # Config schema (sandbox-owned slice of the shared config.yml)
    "RawCredentialsSection",
    "RawGateServerSection",
    "RawHooksSection",
    "RawNetworkSection",
    "RawPathsSection",
    "RawRunSection",
    "RawSSHSection",
    "RawServicesSection",
    "RawShieldSection",
    "RawVaultSection",
    "SERVICES_TCP_OPTOUT_YAML",
    "SandboxConfigView",
    "ServicesMode",
    "gate_use_personal_ssh_default",
    # Setup stamp (epic #685 phase 1 — TUI's cheap "needs_setup" probe)
    "SetupVerdict",
    "clear_stamp",
    "installed_versions",
    "needs_setup",
    "read_stamp",
    "stamp_path",
    "write_stamp",
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
    "DEFAULT_GUEST_SSHD_PORT",
    "DEFAULT_SSH_HOST",
    "DEFAULT_SSH_USER",
    "ExecResult",
    "FakeKrunTransport",
    "GpuConfigError",
    "Image",
    "KrunContainer",
    "KrunRuntime",
    "KrunTransport",
    "LogStream",
    "NullRuntime",
    "PodmanRuntime",
    "PortReservation",
    "TcpEndpoint",
    "TcpSSHTransport",
    "podman_port_resolver",
    # Gate server
    "GateServerStatus",
    # Shield
    "EnvironmentCheck",
    "NftNotFoundError",
    "ShieldNeedsSetup",
    "ShieldRuntime",
    "ShieldState",
    "check_environment",
    "down",
    "make_shield",
    "pre_start",
    "quarantine",
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
    "NoPassphraseError",
    "PassphraseSource",
    "SSHKeyRecord",
    "SSHKeyRow",
    "WrongPassphraseError",
    "is_systemd_creds_available",
    "systemd_creds_has_tpm2",
    # Vault
    "VaultStatus",
    "VaultUnreachableError",
    # Recovery-key acknowledgement view — bundle of the marker probe
    # and the resolved passphrase source.  Operators surface this via
    # ``RecoveryStatus.load()`` / ``.is_acknowledged()`` / ``.acknowledge()``.
    "RecoveryStatus",
    # Command registry
    "ArgDef",
    "CommandDef",
    "CommandTree",
    "CODEX_SHARED_OAUTH_MARKER",
    "KeyRow",
    "DOCTOR_COMMANDS",
    "GATE_COMMANDS",
    "SETUP_COMMANDS",
    "VAULT_COMMANDS",
    "SANDBOX_COMMANDS",
    "SHIELD_COMMANDS",
    "SSH_COMMANDS",
    "handle_vault_seal",
    "handle_vault_to_keyring",
    # Aggregator entry points — one-call install/teardown of the
    # full shield+vault+gate+clearance stack.
    "sandbox_setup",
    "sandbox_uninstall",
    # Stage-line rendering — shared with frontends so mixed logs
    # share one column width and colour palette.
    "Marker",
    "STAGE_WIDTH",
    "StageLine",
    "bold",
    "red",
    "stage",
    "stage_begin",
    "stage_end",
    "stage_line",
    "supports_color",
    "yellow",
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
    "make_recovery_acknowledged_check",
    "sandbox_doctor_checks",
    # SSH
    "DEFAULT_RSA_BITS",
    "ExportResult",
    "GeneratedKeypair",
    "ImportResult",
    "InfraKeypair",
    "KeypairMismatchError",
    "PasswordProtectedKeyError",
    "SSHInitResult",
    "SSHManager",
    "UnsafeCommentError",
    "ensure_infra_keypair",
    "export_ssh_keypair",
    "generate_keypair",
    "import_ssh_keypair",
    "fingerprint_of",
    "openssh_pem_of",
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
