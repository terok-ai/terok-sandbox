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

Every public name is re-exported **lazily**: the module-level
[`__getattr__`][terok_sandbox.__getattr__] (PEP 562) imports the owning
submodule only on first attribute access, so a bare ``import
terok_sandbox`` — or the per-container supervisor spawn that starts from
it — never eagerly drags in pydantic (config), cryptography (SSH
keypairs), SQLCipher (the credential store), or terok-shield.  Each
subsystem is paid for only when its symbol is first touched.
"""

from __future__ import annotations

from importlib import import_module
from typing import TYPE_CHECKING, Any

from terok_util.config_stack import ConfigScope

#: Public name → ``"submodule:attr"`` source.  ``submodule`` is relative
#: to this package; ``attr`` is the name inside that submodule (it
#: differs from the public name wherever the historical top-level export
#: renamed on import — e.g. ``sandbox_uninstall`` →
#: ``commands:_handle_sandbox_uninstall``).  Consumed by
#: [`__getattr__`][terok_sandbox.__getattr__]; the owning submodule is
#: imported on first access and never before.  Names absent from
#: ``__all__`` (``GateServer``, ``write_sidecar``, the vault-session
#: helpers, the AppArmor probes) stay resolvable here for back-compat
#: but are no longer advertised as stable API — reach into their
#: submodule directly.
_LAZY: dict[str, str] = {
    # Stage-line colour helpers
    "bold": "_stage:bold",
    "red": "_stage:red",
    "stage_line": "_stage:stage_line",
    "yellow": "_stage:yellow",
    # AppArmor probe (granular install plumbing stays in ``_util._apparmor``)
    "AppArmorCheckResult": "_util._apparmor:AppArmorCheckResult",
    "AppArmorStatus": "_util._apparmor:AppArmorStatus",
    "check_apparmor_status": "_util._apparmor:check_status",
    # SELinux probe + install plumbing
    "SelinuxCheckResult": "_util._selinux:SelinuxCheckResult",
    "SelinuxStatus": "_util._selinux:SelinuxStatus",
    "check_selinux_status": "_util._selinux:check_status",
    "selinux_install_command": "_util._selinux:install_command",
    "selinux_install_script": "_util._selinux:install_script_path",
    # YAML section writer
    "yaml_update_section": "_yaml:update_section",
    # CLI registry + vault passphrase workflows
    "CommandTree": "commands:CommandTree",
    "PassphraseChangeResult": "commands:PassphraseChangeResult",  # nosec: B105 — export-map import paths, never secrets
    "ProvisioningPlan": "commands:ProvisioningPlan",
    "TierProvisionResult": "commands:TierProvisionResult",
    "TierRewrite": "commands:TierRewrite",
    "change_passphrase": "commands:change_passphrase",  # nosec: B105 — export-map import paths, never secrets
    "credentials_provisioned": "commands:credentials_provisioned",
    "handle_vault_seal": "commands:handle_vault_seal",
    "handle_vault_to_keyring": "commands:handle_vault_to_keyring",
    "plan_provisioning": "commands:plan_provisioning",
    "provision_passphrase_tier": "commands:provision_passphrase_tier",  # nosec: B105 — export-map import paths, never secrets
    "provision_session_passphrase": "commands:provision_session_passphrase",
    "purge_passphrase_tiers": "commands:purge_passphrase_tiers",
    "sandbox_uninstall": "commands:_handle_sandbox_uninstall",
    # Config
    "CONTAINER_RUNTIME_DIR": "config:CONTAINER_RUNTIME_DIR",
    "SandboxConfig": "config:SandboxConfig",
    # Config schema view / fragments
    "RawRunSection": "config_schema:RawRunSection",
    "RawSSHSection": "config_schema:RawSSHSection",
    "SERVICES_TCP_OPTOUT_YAML": "config_schema:SERVICES_TCP_OPTOUT_YAML",
    "SandboxConfigView": "config_schema:SandboxConfigView",
    "ServicesMode": "config_schema:ServicesMode",
    "gate_use_personal_ssh_default": "config_schema:gate_use_personal_ssh_default",
    # Container diagnostics
    "ContainerDiagnostics": "diagnostics:ContainerDiagnostics",
    "container_diagnostics": "diagnostics:container_diagnostics",
    "SupervisorLiveness": "diagnostics:SupervisorLiveness",
    "supervisor_liveness": "diagnostics:supervisor_liveness",
    # Post-start supervision check
    "SupervisionStatus": "supervision:SupervisionStatus",
    "verify_supervision": "supervision:verify_supervision",
    "warn_unsupervised": "supervision:warn_unsupervised",
    # Doctor
    "CheckVerdict": "doctor:CheckVerdict",
    "DoctorCheck": "doctor:DoctorCheck",
    "sandbox_doctor_checks": "doctor:sandbox_doctor_checks",
    # Git gate
    "AppliedOp": "gate.mirror:AppliedOp",
    "ApplyPendingResult": "gate.mirror:ApplyPendingResult",
    "BackupRef": "gate.mirror:BackupRef",
    "GateAuthNotConfigured": "gate.mirror:GateAuthNotConfigured",
    "GateStalenessInfo": "gate.mirror:GateStalenessInfo",
    "GateSyncResult": "gate.mirror:GateSyncResult",
    "GitGate": "gate.mirror:GitGate",
    "PUSH_MARKER_FILENAME": "gate.hooks:PUSH_MARKER_FILENAME",
    "PendingOp": "gate.mirror:PendingOp",
    "RestoreBackupResult": "gate.mirror:RestoreBackupResult",
    "is_ssh_url": "gate.mirror:is_ssh_url",
    "GateServer": "gate.server:GateServer",
    "mint_gate_token": "gate.tokens:mint_gate_token",
    # Shield (sandbox-side policy classes)
    "EnvironmentCheck": "integrations.shield:EnvironmentCheck",
    "ShieldHooks": "integrations.shield:ShieldHooks",
    "ShieldManager": "integrations.shield:ShieldManager",
    "check_environment": "integrations.shield:check_environment",
    "resolve_container_state_dir": "integrations.shield:resolve_container_state_dir",
    # Per-container launch/state
    "PerContainerResources": "launch:PerContainerResources",
    "PASSTHROUGH_DENIED_FLAGS": "podman_args:PASSTHROUGH_DENIED_FLAGS",
    "SANDBOX_MANAGED_FLAGS": "podman_args:SANDBOX_MANAGED_FLAGS",
    "allocate_per_container_resources": "launch:allocate_per_container_resources",
    "reject_managed_flags": "podman_args:reject_managed_flags",
    "reject_managed_volumes": "podman_args:reject_managed_volumes",
    "validate_passthrough_args": "podman_args:validate_passthrough_args",
    "make_stray_sidecar_check": "launch:make_stray_sidecar_check",
    "remove_container_state": "launch:remove_container_state",
    "write_sidecar": "launch:write_sidecar",
    # Port registry (one-call helpers)
    "claim_port": "port_registry:claim_port",
    "release_port": "port_registry:release_port",
    # Runtime
    "ContainerEvent": "runtime:ContainerEvent",
    "ContainerRuntime": "runtime:ContainerRuntime",
    "DEFAULT_GUEST_SSHD_PORT": "runtime:DEFAULT_GUEST_SSHD_PORT",
    "DEFAULT_SSH_HOST": "runtime:DEFAULT_SSH_HOST",
    "ExecResult": "runtime:ExecResult",
    "GPU_VENDORS": "runtime:GPU_VENDORS",
    "GpuConfigError": "runtime:GpuConfigError",
    "GpuGrant": "runtime:GpuGrant",
    "GpuSelector": "runtime:GpuSelector",
    "GpuVendor": "runtime:GpuVendor",
    "Image": "runtime:Image",
    "KrunRuntime": "runtime:KrunRuntime",
    "NullRuntime": "runtime:NullRuntime",
    "PodmanEventStream": "runtime:PodmanEventStream",
    "PodmanRuntime": "runtime:PodmanRuntime",
    "TcpSSHTransport": "runtime:TcpSSHTransport",
    "check_gpu_available": "runtime:check_gpu_available",
    "detect_gpu_vendors": "runtime:detect_gpu_vendors",
    "gpu_device_addresses": "runtime:gpu_device_addresses",
    "normalize_gpus": "runtime:normalize_gpus",
    "podman_port_resolver": "runtime:podman_port_resolver",
    # Facade
    "GRANTABLE_CAPS": "sandbox:GRANTABLE_CAPS",
    "LifecycleHooks": "sandbox:LifecycleHooks",
    "READY_MARKER": "sandbox:READY_MARKER",
    "RunSpec": "sandbox:RunSpec",
    "Sandbox": "sandbox:Sandbox",
    "Sharing": "sandbox:Sharing",
    "VolumeSpec": "sandbox:VolumeSpec",
    # Setup stamp / aggregator
    "SetupVerdict": "setup_stamp:SetupVerdict",
    "installed_versions": "setup_stamp:installed_versions",
    "needs_setup": "setup_stamp:needs_setup",
    "read_stamp": "setup_stamp:read_stamp",
    "stamp_path": "setup_stamp:stamp_path",
    # Vault markers + SSH + credential store
    "CODEX_SHARED_OAUTH_MARKER": "vault.daemon:CODEX_SHARED_OAUTH_MARKER",
    "PHANTOM_CREDENTIALS_MARKER": "vault.daemon:PHANTOM_CREDENTIALS_MARKER",
    "ensure_infra_keypair": "vault.ssh.keypair:ensure_infra_keypair",
    "public_line_of": "vault.ssh.keypair:public_line_of",
    "SSHInitResult": "vault.ssh.manager:SSHInitResult",
    "SSHManager": "vault.ssh.manager:SSHManager",
    "CredentialDB": "vault.store.db:CredentialDB",
    "SSHKeyRow": "vault.store.db:SSHKeyRow",
    "NoPassphraseError": "vault.store.encryption:NoPassphraseError",
    "WrongPassphraseError": "vault.store.encryption:WrongPassphraseError",
    "keyring_backend_available": "vault.store.encryption:keyring_backend_available",
    "RecoveryStatus": "vault.store.recovery:RecoveryStatus",
    "ChainRow": "vault.store.status:ChainRow",
    "SessionShadow": "vault.store.status:SessionShadow",
    "VaultState": "vault.store.status:VaultState",
    "VaultStatus": "vault.store.status:VaultStatus",
    "VaultWarning": "vault.store.status:VaultWarning",
    "VaultWarningKind": "vault.store.status:VaultWarningKind",
    "clear_redundant_session_file": "vault.store.status:clear_redundant_session_file",
    "session_shadow_state": "vault.store.status:session_shadow_state",
    "systemd_creds_available": "vault.store.systemd_creds:is_available",
    "systemd_creds_has_tpm2": "vault.store.systemd_creds:has_tpm2",
    "PassphraseTier": "vault.store.tiers:PassphraseTier",
}


def __getattr__(name: str) -> Any:
    """Resolve a public symbol by importing its owning submodule on first access (PEP 562).

    Looked up in the module's ``_LAZY`` map; the resolved value is
    cached on the module so subsequent accesses are plain attribute
    reads.  Unknown names raise [`AttributeError`][AttributeError] the
    same way a missing module global would.

    ``__version__`` resolves here too: the
    [`importlib.metadata.version`][importlib.metadata.version] lookup
    drags in several MiB of stdlib (``inspect``, ``email``, ``zipfile``),
    so it is charged only to callers that actually ask — never to the
    bare import the per-container supervisor spawn starts from.
    """
    if name == "__version__":
        from importlib.metadata import PackageNotFoundError, version as meta_version

        try:
            value = meta_version("terok-sandbox")
        except PackageNotFoundError:
            value = "0.0.0"  # running from source without installed metadata
        globals()[name] = value
        return value
    target = _LAZY.get(name)
    if target is None:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
    module_name, _, attr = target.partition(":")
    value = getattr(import_module(f".{module_name}", __name__), attr)
    globals()[name] = value  # cache — the next access skips __getattr__
    return value


def __dir__() -> list[str]:
    """List eager globals plus every lazily-exported name for tab-completion."""
    return sorted({*globals(), *_LAZY, "__version__"})


if TYPE_CHECKING:
    # Eager view of the stable (``__all__``) surface for type checkers and
    # IDEs — never imported at runtime, so it costs nothing on ``import
    # terok_sandbox``.  Demoted names stay resolvable via ``_LAZY`` but
    # are intentionally omitted here; reach into their submodule for a
    # statically-typed handle.
    __version__: str

    from ._stage import bold, red, stage_line, yellow
    from ._util._selinux import (
        SelinuxCheckResult,
        SelinuxStatus,
        check_status as check_selinux_status,
        install_command as selinux_install_command,
        install_script_path as selinux_install_script,
    )
    from ._yaml import update_section as yaml_update_section
    from .commands import (
        PassphraseChangeResult,
        ProvisioningPlan,
        TierProvisionResult,
        TierRewrite,
        _handle_sandbox_uninstall as sandbox_uninstall,
        change_passphrase,
        credentials_provisioned,
        handle_vault_seal,
        handle_vault_to_keyring,
        plan_provisioning,
        provision_passphrase_tier,
        purge_passphrase_tiers,
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
    from .diagnostics import (
        ContainerDiagnostics,
        SupervisorLiveness,
        container_diagnostics,
        supervisor_liveness,
    )
    from .doctor import CheckVerdict, DoctorCheck, sandbox_doctor_checks
    from .gate.hooks import PUSH_MARKER_FILENAME
    from .gate.mirror import (
        AppliedOp,
        ApplyPendingResult,
        BackupRef,
        GateAuthNotConfigured,
        GateStalenessInfo,
        GateSyncResult,
        GitGate,
        PendingOp,
        RestoreBackupResult,
        is_ssh_url,
    )
    from .gate.tokens import mint_gate_token
    from .integrations.shield import (
        EnvironmentCheck,
        ShieldHooks,
        ShieldManager,
        check_environment,
        resolve_container_state_dir,
    )
    from .launch import PerContainerResources, allocate_per_container_resources
    from .podman_args import (
        PASSTHROUGH_DENIED_FLAGS,
        SANDBOX_MANAGED_FLAGS,
        reject_managed_flags,
        reject_managed_volumes,
        validate_passthrough_args,
    )
    from .port_registry import claim_port, release_port
    from .runtime import (
        DEFAULT_GUEST_SSHD_PORT,
        DEFAULT_SSH_HOST,
        GPU_VENDORS,
        ContainerEvent,
        ContainerRuntime,
        ExecResult,
        GpuConfigError,
        GpuGrant,
        GpuSelector,
        GpuVendor,
        Image,
        KrunRuntime,
        NullRuntime,
        PodmanEventStream,
        PodmanRuntime,
        TcpSSHTransport,
        check_gpu_available,
        detect_gpu_vendors,
        gpu_device_addresses,
        normalize_gpus,
        podman_port_resolver,
    )
    from .sandbox import (
        GRANTABLE_CAPS,
        READY_MARKER,
        LifecycleHooks,
        RunSpec,
        Sandbox,
        Sharing,
        VolumeSpec,
    )
    from .setup_stamp import (
        SetupVerdict,
        installed_versions,
        needs_setup,
        read_stamp,
        stamp_path,
    )
    from .supervision import SupervisionStatus, verify_supervision, warn_unsupervised
    from .vault.daemon import CODEX_SHARED_OAUTH_MARKER, PHANTOM_CREDENTIALS_MARKER
    from .vault.ssh.keypair import ensure_infra_keypair, public_line_of
    from .vault.ssh.manager import SSHInitResult, SSHManager
    from .vault.store.db import CredentialDB, SSHKeyRow
    from .vault.store.encryption import (
        NoPassphraseError,
        WrongPassphraseError,
        keyring_backend_available,
    )
    from .vault.store.recovery import RecoveryStatus
    from .vault.store.status import (
        ChainRow,
        SessionShadow,
        VaultState,
        VaultStatus,
        VaultWarning,
        VaultWarningKind,
    )
    from .vault.store.systemd_creds import (
        has_tpm2 as systemd_creds_has_tpm2,
        is_available as systemd_creds_available,
    )
    from .vault.store.tiers import PassphraseTier


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
    "PASSTHROUGH_DENIED_FLAGS",
    "PerContainerResources",
    "SANDBOX_MANAGED_FLAGS",
    "allocate_per_container_resources",
    "reject_managed_flags",
    "reject_managed_volumes",
    "validate_passthrough_args",
    "mint_gate_token",
    # Runtime + facade
    "ContainerRuntime",
    "DEFAULT_GUEST_SSHD_PORT",
    "DEFAULT_SSH_HOST",
    "ExecResult",
    "GPU_VENDORS",
    "GpuConfigError",
    "GpuGrant",
    "GpuSelector",
    "GpuVendor",
    "Image",
    "KrunRuntime",
    "LifecycleHooks",
    "NullRuntime",
    "PodmanRuntime",
    "PodmanEventStream",
    "ContainerEvent",
    "GRANTABLE_CAPS",
    "READY_MARKER",
    "RunSpec",
    "Sharing",
    "TcpSSHTransport",
    "VolumeSpec",
    "check_gpu_available",
    "detect_gpu_vendors",
    "gpu_device_addresses",
    "normalize_gpus",
    "podman_port_resolver",
    # Shield (sandbox-side policy classes; the egress-firewall layer lives in terok-shield)
    "EnvironmentCheck",
    "ShieldHooks",
    "ShieldManager",
    "check_environment",
    "resolve_container_state_dir",
    # Git gate
    "AppliedOp",
    "ApplyPendingResult",
    "BackupRef",
    "GateAuthNotConfigured",
    "GateStalenessInfo",
    "GateSyncResult",
    "GitGate",
    "PUSH_MARKER_FILENAME",
    "PendingOp",
    "RestoreBackupResult",
    "is_ssh_url",
    # Credentials + vault
    "CODEX_SHARED_OAUTH_MARKER",
    "ChainRow",
    "CredentialDB",
    "NoPassphraseError",
    "PHANTOM_CREDENTIALS_MARKER",
    "PassphraseChangeResult",
    "PassphraseTier",
    "ProvisioningPlan",
    "RecoveryStatus",
    "SessionShadow",
    "TierProvisionResult",
    "TierRewrite",
    "VaultState",
    "VaultStatus",
    "VaultWarning",
    "VaultWarningKind",
    "WrongPassphraseError",
    "change_passphrase",
    "credentials_provisioned",
    "keyring_backend_available",
    "plan_provisioning",
    "provision_passphrase_tier",
    "systemd_creds_available",
    "systemd_creds_has_tpm2",
    "handle_vault_seal",
    "handle_vault_to_keyring",
    "purge_passphrase_tiers",
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
    "sandbox_doctor_checks",
    # Container diagnostics (on-host supervisor/sidecar artifact paths)
    "ContainerDiagnostics",
    "container_diagnostics",
    "SupervisorLiveness",
    "supervisor_liveness",
    # Post-start supervision check
    "SupervisionStatus",
    "verify_supervision",
    "warn_unsupervised",
    # SELinux (one-call probe + install plumbing; the granular probes
    # stay in ``_util._selinux``).
    "SelinuxCheckResult",
    "SelinuxStatus",
    "check_selinux_status",
    "selinux_install_command",
    "selinux_install_script",
    # CLI + cross-package utilities
    "bold",
    "red",
    "stage_line",
    "yaml_update_section",
    "yellow",
    # Meta
    "__version__",
]
