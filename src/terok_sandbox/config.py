# SPDX-FileCopyrightText: 2026 Jiri Vyskocil
# SPDX-License-Identifier: Apache-2.0

"""Sandbox configuration — plain dataclass for standalone and embedded use.

:class:`SandboxConfig` captures directory paths and settings that sandbox
modules need.  In standalone ``terok-sandbox`` use, it is resolved from
environment variables and XDG defaults.  When embedded in terok, the
orchestration layer constructs it from :func:`core.config` values.
"""

from __future__ import annotations

import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING

from pydantic import ValidationError

from .paths import (
    config_root as _config_root,
    read_config_section,
    runtime_root as _runtime_root,
    state_root as _state_root,
    vault_root as _vault_root,
)

if TYPE_CHECKING:
    from .config_schema import ServicesMode

CONTAINER_RUNTIME_DIR = "/run/terok"
"""Container-side mount point for the host runtime directory (socket mode)."""


def services_mode() -> ServicesMode:
    """Resolve the ``services.mode`` setting through sandbox's own pydantic schema.

    Sandbox owns the ``services:`` section (see
    :class:`~terok_sandbox.config_schema.RawServicesSection`), so this
    reader and the composed ``RawGlobalConfig`` validator up in terok
    share one schema class: one default, one validator, one failure
    mode.  A missing section, a missing key, or a malformed value all
    collapse to the schema's own default — no hand-rolled fallback
    drifting from the schema's intent.

    An outright invalid value (a typo like ``soket``) still surfaces a
    stderr warning before the default kicks in, since the caller asked
    for a specific mode and silently ignoring the request would be
    worse than a default mismatch.
    """
    from .config_schema import RawServicesSection

    raw = read_config_section("services")
    try:
        return RawServicesSection.model_validate(raw).mode
    except ValidationError as exc:
        default = RawServicesSection().mode
        print(
            f"warning: invalid services section ({exc.errors()[0]['msg']}) "
            f"— falling back to {default!r}",
            file=sys.stderr,
        )
        return default


def _default_services_mode() -> ServicesMode:
    """Default-factory indirection for :attr:`SandboxConfig.services_mode`.

    Lets tests patch ``terok_sandbox.config.services_mode`` and see the
    patch take effect at construction time — a direct
    ``default_factory=services_mode`` would capture the original function
    reference at class-definition time and ignore later patches.
    """
    return services_mode()


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

    vault_dir: Path = field(default_factory=_vault_root)
    """Shared vault directory (DB, routes, env mounts)."""

    gate_port: int | None = None
    """HTTP port for the gate server (``None`` = auto-allocate via registry)."""

    token_broker_port: int | None = None
    """TCP port for the vault's token broker (``None`` = auto-allocate via registry)."""

    ssh_signer_port: int | None = None
    """TCP port for the vault's SSH signer (``None`` = auto-allocate via registry)."""

    shield_profiles: tuple[str, ...] = ("dev-standard",)
    """Shield egress firewall profile names."""

    shield_audit: bool = True
    """Whether shield audit logging is enabled."""

    shield_bypass: bool = False
    """DANGEROUS: when True, the egress firewall is completely disabled."""

    services_mode: ServicesMode = field(default_factory=_default_services_mode)
    """Transport for host↔container IPC, resolved once at construction.

    The default factory validates the layered ``config.yml`` through
    :class:`~terok_sandbox.config_schema.RawServicesSection` — the same
    schema that terok's ``RawGlobalConfig`` composes, so the standalone
    and embedded paths can't disagree on a mode value.

    Downstream sandbox operations (vault / gate install, SELinux checks)
    read this field exclusively.  Making it an instance attribute rather
    than a free-function call per site means the control flow can't
    bypass config resolution: you can't construct a manager without a
    ``SandboxConfig``, and every ``SandboxConfig`` carries a resolved
    mode.
    """

    def __post_init__(self) -> None:
        """Auto-resolve ``None`` ports via the shared port registry.

        Skipped entirely when ``services.mode`` is ``socket`` — in that
        transport the gate, vault token-broker, and vault SSH-signer all
        listen on Unix sockets, so TCP port claims would be wasted work
        (and, historically, a source of spurious collision errors when
        multiple ``SandboxConfig()`` constructions raced from TUI worker
        threads).
        """
        if self.services_mode == "socket":
            return
        if self.gate_port is None or self.token_broker_port is None or self.ssh_signer_port is None:
            from .port_registry import resolve_service_ports

            ports = resolve_service_ports(
                self.gate_port,
                self.token_broker_port,
                self.ssh_signer_port,
                gate_explicit=self.gate_port is not None,
                proxy_explicit=self.token_broker_port is not None,
                ssh_explicit=self.ssh_signer_port is not None,
                state_dir=self.state_dir,
            )
            if self.gate_port is None:
                object.__setattr__(self, "gate_port", ports.gate)
            if self.token_broker_port is None:
                object.__setattr__(self, "token_broker_port", ports.proxy)
            if self.ssh_signer_port is None:
                object.__setattr__(self, "ssh_signer_port", ports.ssh_agent)

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
    def db_path(self) -> Path:
        """Return the path to the vault sqlite3 database."""
        return self.vault_dir / "credentials.db"

    @property
    def vault_socket_path(self) -> Path:
        """Return the Unix socket path for the vault."""
        return self.runtime_dir / "vault.sock"

    @property
    def vault_pid_path(self) -> Path:
        """Return the PID file path for the managed vault daemon."""
        return self.runtime_dir / "vault.pid"

    @property
    def routes_path(self) -> Path:
        """Return the path to the vault route configuration JSON."""
        return self.vault_dir / "routes.json"

    @property
    def gate_socket_path(self) -> Path:
        """Return the Unix socket path for the gate server."""
        return self.runtime_dir / "gate-server.sock"

    @property
    def ssh_signer_socket_path(self) -> Path:
        """Return the Unix socket path for the vault's SSH signer.

        The vault binds this socket and serves the SSH-agent protocol on it
        (clients use it as ``$SSH_AUTH_SOCK``).  Filename uses the protocol
        name so its purpose is recognisable to anyone tracing socket activity.
        """
        return self.runtime_dir / "ssh-agent.sock"

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
        """Return the path to the SSH key mapping JSON.

        .. deprecated::
            SSH keys are stored in :attr:`db_path` (table ``ssh_keys``) and
            served via per-scope sockets at
            :meth:`ssh_signer_local_socket_path`.  This path is retained
            only for transitional callers in sibling packages; new code
            must not read or write it.
        """
        return self.vault_dir / "ssh-keys.json"

    def ssh_signer_local_socket_path(self, scope: str) -> Path:
        """Return the per-scope vault SSH-agent socket path for *scope*.

        The vault binds one 0600 Unix socket per scope with at least one
        assigned key, under the same ``runtime_dir`` as the main signer.
        Host-side ``gate-sync`` points ``SSH_AUTH_SOCK`` at this path.

        Rejects unsafe scope names with :class:`~.credentials.db.InvalidScopeName`
        as a belt-and-braces guard — writers in the DB layer enforce the
        same policy, but the socket path is public API and may be called
        without a preceding DB write.
        """
        from .credentials.db import _require_safe_scope

        _require_safe_scope(scope)
        return self.runtime_dir / f"ssh-agent-local-{scope}.sock"
