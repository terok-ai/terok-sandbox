# SPDX-FileCopyrightText: 2025 Jiri Vyskocil
# SPDX-FileCopyrightText: 2026 Jiri Vyskocil
# SPDX-License-Identifier: Apache-2.0

"""terok-sandbox: hardened Podman container runner with gate and shield integration.

Delegates to domain subsystems:

- [`gate`][terok_sandbox.gate] — authenticated git serving: HTTP server, token CRUD, upstream
  mirror management, systemd/daemon lifecycle.
- [`vault`][terok_sandbox.vault] — secret injection: token broker with phantom credentials,
  SSH signing proxy, SQLite credential store, systemd/daemon lifecycle.
- [`shield`][terok_sandbox.shield] — egress firewall adapter (delegates to terok-shield).
- [`runtime`][terok_sandbox.runtime] — Podman CLI wrapper (state queries, GPU, log streaming).
- [`sandbox`][terok_sandbox.sandbox] — facade composing the above behind [`SandboxConfig`][terok_sandbox.SandboxConfig].
- [`commands`][terok_sandbox.commands] — CLI command registry and handler implementations.
"""

from __future__ import annotations

__version__: str = "0.0.0"  # placeholder; replaced at build time

import dataclasses
from importlib.metadata import PackageNotFoundError, version as _meta_version
from pathlib import Path
from typing import TYPE_CHECKING

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
from ._util import BestEffortLogger, sanitize_tty
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
from .config_stack import ConfigScope, ConfigStack
from .doctor import (
    CheckVerdict,
    DoctorCheck,
    make_recovery_acknowledged_check,
    sandbox_doctor_checks,
)
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
from .shield import (
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


def get_token_broker_port(cfg: SandboxConfig | None = None) -> int | None:
    """Return the configured token broker TCP port (``None`` in socket mode)."""
    return VaultManager(cfg).token_broker_port


def get_ssh_signer_port(cfg: SandboxConfig | None = None) -> int | None:
    """Return the configured SSH signer TCP port (``None`` in socket mode)."""
    return VaultManager(cfg).ssh_signer_port


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


def install_vault_systemd(cfg: SandboxConfig | None = None) -> None:
    """Install and start the vault's systemd user units."""
    VaultManager(cfg).install_systemd_units()


def uninstall_vault_systemd(cfg: SandboxConfig | None = None) -> None:
    """Remove the vault's systemd user units."""
    VaultManager(cfg).uninstall_systemd_units()


def start_vault(cfg: SandboxConfig | None = None) -> None:
    """Start the vault as a background daemon."""
    VaultManager(cfg).start_daemon()


def stop_vault(cfg: SandboxConfig | None = None) -> None:
    """Stop the managed vault daemon."""
    VaultManager(cfg).stop_daemon()


def is_recovery_acknowledged(cfg: SandboxConfig | None = None) -> bool:
    """Return ``True`` iff the operator has confirmed they saved the recovery key.

    The vault's resolver tiers (systemd-creds, keyring, session-file)
    are all bound to *this* machine, account, or boot — a hardware
    failure or TPM transplant strands the vault without an off-host
    copy of the passphrase.  This check is what surfaces the
    "unconfirmed recovery key" warning in sickbay / doctor / the TUI
    pill: a zero-byte marker file at
    [`vault_recovery_marker_file`][terok_sandbox.SandboxConfig.vault_recovery_marker_file]
    indicates the operator has acknowledged at some point.  Absence
    (or an unreadable marker) reports ``False`` — the warning is
    conservative by design.

    Operators close the warning by running ``terok-sandbox vault
    passphrase reveal`` (or the TUI's reveal modal) and confirming,
    or via the silent ``vault passphrase acknowledge`` after CI /
    TUI captured the value out-of-band.  A passphrase rotation does
    NOT auto-invalidate the marker; operators who rotate should
    re-ack against the new value.
    """
    from .vault.store.recovery import acknowledged

    if cfg is None:
        cfg = SandboxConfig()
    return acknowledged(cfg.vault_recovery_marker_file)


@dataclasses.dataclass(frozen=True)
class RecoveryStatus:
    """Combined marker + resolved-source view for the recovery-key warning surfaces.

    Returned by [`recovery_status`][terok_sandbox.recovery_status] so
    sickbay / doctor / TUI / post-launch CLI all paint the same picture
    of "is the operator one reboot away from losing their vault?".
    """

    acknowledged: bool
    """``True`` iff the zero-byte marker file is present."""

    source: PassphraseSource | None
    """Whichever resolver tier unlocked the chain right now, or ``None`` if locked."""

    @property
    def session_only(self) -> bool:
        """``True`` iff the passphrase lives only in the tmpfs session-unlock file.

        That tier dies on the next reboot — without an off-host copy
        the vault becomes unrecoverable the moment the machine
        restarts.  Severity should escalate accordingly on every
        surface that renders this status.
        """
        return self.source == "session-file"

    @property
    def urgent(self) -> bool:
        """``True`` iff unacknowledged AND session-only (one reboot away from loss)."""
        return not self.acknowledged and self.session_only


def recovery_status(cfg: SandboxConfig | None = None) -> RecoveryStatus:
    """Return the combined marker + resolved-source view in one call.

    Single seam for every "recovery key unconfirmed" surface — doctor,
    sickbay, TUI pill, post-task-launch CLI footer.  Walking the
    resolver chain to find the source is cheap (no DB open, just tier
    knobs) and bundling it with the marker check here means no caller
    has to repeat the "is this session-only?" lookup.
    """
    from .vault.store.encryption import NoPassphraseError, WrongPassphraseError
    from .vault.store.recovery import acknowledged

    if cfg is None:
        cfg = SandboxConfig()
    try:
        _passphrase, source = cfg.resolve_passphrase_with_source()
    except (NoPassphraseError, WrongPassphraseError):
        source = None
    return RecoveryStatus(
        acknowledged=acknowledged(cfg.vault_recovery_marker_file),
        source=source,
    )


def acknowledge_recovery(cfg: SandboxConfig | None = None) -> bool:
    """Mark the recovery key as saved (writes the zero-byte sidecar marker).

    Always succeeds and returns ``True`` — the marker is independent
    of the passphrase resolver, so a locked vault doesn't block
    acknowledgement.  The return value is kept for API stability with
    callers that previously had a "locked vault" failure path.
    """
    from .vault.store.recovery import acknowledge

    if cfg is None:
        cfg = SandboxConfig()
    acknowledge(cfg.vault_recovery_marker_file)
    return True


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
    "check_units_outdated",
    "ensure_server_reachable",
    "get_gate_base_path",
    "get_gate_server_port",
    "get_server_status",
    "is_daemon_running",
    "is_systemd_available",
    "start_daemon",
    "stop_daemon",
    # Gate tokens
    "create_token",
    "revoke_token_for_task",
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
    "ensure_vault_reachable",
    "get_token_broker_port",
    "get_vault_status",
    "get_ssh_signer_port",
    "is_vault_running",
    "is_vault_service_active",
    "is_vault_socket_active",
    "is_vault_socket_installed",
    "is_vault_systemd_available",
    "install_vault_systemd",
    "start_vault",
    "stop_vault",
    "uninstall_vault_systemd",
    # Recovery-key acknowledgement (operator confirmed they saved the
    # auto-generated passphrase off-host).  False until the operator
    # confirms via `vault passphrase reveal` / the TUI reveal modal /
    # the silent `acknowledge_recovery` wrapper.
    "RecoveryStatus",
    "acknowledge_recovery",
    "is_recovery_acknowledged",
    "recovery_status",
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
