# SPDX-FileCopyrightText: 2025 Jiri Vyskocil
# SPDX-License-Identifier: Apache-2.0

"""terok-sandbox: hardened Podman container runner with gate and shield integration.

Delegates to domain subsystems:

- [`gate`][terok_sandbox.gate] — authenticated git serving: HTTP server, token CRUD, upstream
  mirror management, systemd/daemon lifecycle.
- [`vault`][terok_sandbox.vault] — secret injection: per-container token broker with
  phantom credentials, SSH signing proxy, SQLite credential store.
- [`shield`][terok_sandbox.integrations.shield] — egress firewall adapter (delegates to terok-shield).
- [`runtime`][terok_sandbox.runtime] — Podman CLI wrapper (state queries, GPU, log streaming).
- [`sandbox`][terok_sandbox.sandbox] — facade composing the above behind [`SandboxConfig`][terok_sandbox.SandboxConfig].
- [`commands`][terok_sandbox.commands] — CLI command registry and handler implementations.

The top-level surface here is the published contract that
[`terok_executor`][terok_executor] and ``terok`` consume.  Internal
helpers (raw config schema fragments, runtime concrete types like
``Container``/``LogStream``/``PortReservation``, SSH keypair helpers,
selinux probe internals, port-registry primitives, shield error
classes) stay in their submodules; reach into ``terok_sandbox.<sub>``
when you need them.
"""

from __future__ import annotations

__version__: str = "0.0.0"  # placeholder; replaced at build time

from importlib.metadata import PackageNotFoundError, version as _meta_version
from typing import TYPE_CHECKING

from terok_util.config_stack import ConfigScope

from ._stage import bold, red, stage_line, yellow
from ._util._apparmor import (
    AppArmorCheckResult,
    AppArmorStatus,
    check_status as check_apparmor_status,
    install_command as apparmor_install_command,
    install_script_path as apparmor_install_script,
)
from ._util._selinux import (
    SelinuxCheckResult,
    SelinuxStatus,
    check_status as check_selinux_status,
    install_command as selinux_install_command,
    install_script_path as selinux_install_script,
)
from ._yaml import update_section as yaml_update_section
from .commands import (
    CommandTree,
    SessionProvisionResult,
    SessionShadow,
    _handle_sandbox_uninstall as sandbox_uninstall,
    clear_redundant_session_file,
    handle_vault_seal,
    handle_vault_to_keyring,
    provision_session_passphrase,
    purge_passphrase_tiers,
    session_shadow_state,
)
from .config import CONTAINER_RUNTIME_DIR, SandboxConfig
from .config_schema import (
    SERVICES_TCP_OPTOUT_YAML,
    RawRunSection,
    RawSSHSection,
    SandboxConfigView,
    ServicesMode,
    gate_use_personal_ssh_default,
)
from .diagnostics import ContainerDiagnostics, container_diagnostics
from .doctor import CheckVerdict, DoctorCheck, sandbox_doctor_checks
from .gate.mirror import GateAuthNotConfigured, GateStalenessInfo, GitGate, is_ssh_url
from .gate.server import GateServer
from .gate.tokens import mint_gate_token
from .integrations.shield import (
    EnvironmentCheck,
    ShieldHooks,
    ShieldManager,
    check_environment,
    resolve_container_state_dir,
)
from .launch import (
    PerContainerResources,
    allocate_per_container_resources,
    make_stray_sidecar_check,
    remove_container_state,
    write_sidecar,
)
from .port_registry import claim_port, release_port
from .runtime import (
    DEFAULT_GUEST_SSHD_PORT,
    DEFAULT_SSH_HOST,
    ContainerEvent,
    ContainerRuntime,
    ExecResult,
    GpuConfigError,
    Image,
    KrunRuntime,
    NullRuntime,
    PodmanEventStream,
    PodmanRuntime,
    TcpSSHTransport,
    check_gpu_available,
    podman_port_resolver,
)
from .sandbox import READY_MARKER, LifecycleHooks, RunSpec, Sandbox, Sharing, VolumeSpec
from .setup_stamp import SetupVerdict, installed_versions, needs_setup, read_stamp, stamp_path
from .vault.daemon import CODEX_SHARED_OAUTH_MARKER, PHANTOM_CREDENTIALS_MARKER
from .vault.ssh.keypair import ensure_infra_keypair, public_line_of
from .vault.ssh.manager import SSHInitResult, SSHManager
from .vault.store.db import CredentialDB, SSHKeyRow
from .vault.store.encryption import NoPassphraseError, WrongPassphraseError
from .vault.store.recovery import RecoveryStatus
from .vault.store.systemd_creds import has_tpm2 as systemd_creds_has_tpm2

if TYPE_CHECKING:
    pass  # all types already imported above

try:
    __version__ = _meta_version("terok-sandbox")
except PackageNotFoundError:
    pass  # editable install or running from source without metadata


__all__ = [
    # Config
    "CONTAINER_RUNTIME_DIR",
    "ConfigScope",
    "RawRunSection",
    "RawSSHSection",
    "SERVICES_TCP_OPTOUT_YAML",
    "Sandbox",
    "SandboxConfig",
    "SandboxConfigView",
    "ServicesMode",
    "gate_use_personal_ssh_default",
    # Setup stamp / aggregator
    "SetupVerdict",
    "installed_versions",
    "needs_setup",
    "read_stamp",
    "sandbox_uninstall",
    "stamp_path",
    # Lifecycle managers
    "GateServer",
    "PerContainerResources",
    "allocate_per_container_resources",
    "mint_gate_token",
    "remove_container_state",
    "write_sidecar",
    # Runtime + facade
    "ContainerRuntime",
    "DEFAULT_GUEST_SSHD_PORT",
    "DEFAULT_SSH_HOST",
    "ExecResult",
    "GpuConfigError",
    "Image",
    "KrunRuntime",
    "LifecycleHooks",
    "NullRuntime",
    "PodmanRuntime",
    "PodmanEventStream",
    "ContainerEvent",
    "READY_MARKER",
    "RunSpec",
    "Sharing",
    "TcpSSHTransport",
    "VolumeSpec",
    "check_gpu_available",
    "podman_port_resolver",
    # Shield (sandbox-side policy classes; the egress-firewall layer lives in terok-shield)
    "EnvironmentCheck",
    "ShieldHooks",
    "ShieldManager",
    "check_environment",
    "resolve_container_state_dir",
    # Git gate
    "GateAuthNotConfigured",
    "GateStalenessInfo",
    "GitGate",
    "is_ssh_url",
    # Credentials + vault
    "CODEX_SHARED_OAUTH_MARKER",
    "CredentialDB",
    "NoPassphraseError",
    "PHANTOM_CREDENTIALS_MARKER",
    "RecoveryStatus",
    "WrongPassphraseError",
    "systemd_creds_has_tpm2",
    "handle_vault_seal",
    "handle_vault_to_keyring",
    "SessionProvisionResult",
    "SessionShadow",
    "clear_redundant_session_file",
    "provision_session_passphrase",
    "purge_passphrase_tiers",
    "session_shadow_state",
    # SSH
    "SSHInitResult",
    "SSHKeyRow",
    "SSHManager",
    "ensure_infra_keypair",
    "public_line_of",
    # Port registry (one-call helpers; the underlying ``PortRegistry`` /
    # ``ServicePorts`` types stay in ``port_registry`` for callers that
    # need them).
    "claim_port",
    "release_port",
    # Doctor (container health checks)
    "CheckVerdict",
    "DoctorCheck",
    "make_stray_sidecar_check",
    "sandbox_doctor_checks",
    # Container diagnostics (on-host supervisor/sidecar artifact paths)
    "ContainerDiagnostics",
    "container_diagnostics",
    # SELinux (one-call probe + install plumbing; the granular probes
    # stay in ``_util._selinux``).
    "SelinuxCheckResult",
    "SelinuxStatus",
    "check_selinux_status",
    "selinux_install_command",
    "selinux_install_script",
    # AppArmor (dnsmasq DNS-tier profile addendum; granular probes stay
    # in ``_util._apparmor``).
    "AppArmorCheckResult",
    "AppArmorStatus",
    "check_apparmor_status",
    "apparmor_install_command",
    "apparmor_install_script",
    # CLI + cross-package utilities
    "CommandTree",
    "bold",
    "red",
    "stage_line",
    "yaml_update_section",
    "yellow",
    # Meta
    "__version__",
]
