# SPDX-FileCopyrightText: 2026 Jiri Vyskocil
# SPDX-License-Identifier: Apache-2.0

"""Sandbox configuration — plain dataclass for standalone and embedded use.

[`SandboxConfig`][terok_sandbox.config.SandboxConfig] captures directory paths and settings that sandbox
modules need.  In standalone ``terok-sandbox`` use, it is resolved from
environment variables and XDG defaults.  When embedded in terok, the
orchestration layer constructs it from [`core.config`][terok.lib.core.config] values.
"""

from __future__ import annotations

import functools
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING, Any

from pydantic import BaseModel, ValidationError

from .paths import (
    config_root as _config_root,
    read_config_section,
    runtime_root as _runtime_root,
    state_root as _state_root,
    vault_root as _vault_root,
)

if TYPE_CHECKING:
    from .config_schema import RawCredentialsSection, ServicesMode
    from .credentials.db import CredentialDB
    from .credentials.encryption import PassphraseSource

CONTAINER_RUNTIME_DIR = "/run/terok"
"""Container-side mount point for the host runtime directory (socket mode)."""


def _validate_section[T: BaseModel](schema_cls: type[T], section: str) -> T:
    """Validate *section* against *schema_cls*; warn + return schema defaults on error.

    Sandbox owns several top-level sections of ``config.yml``.  Each
    reader runs the same dance: read → validate → on failure print a
    one-liner to stderr and fall through to the schema's defaults.
    This helper centralises that pattern so a missing/typo'd key
    collapses to the schema's default without hand-rolled fallbacks.
    """
    raw = read_config_section(section)
    try:
        return schema_cls.model_validate(raw)
    except ValidationError as exc:
        print(
            f"warning: invalid {section} section ({exc.errors()[0]['msg']}) "
            "— falling back to schema defaults",
            file=sys.stderr,
        )
        return schema_cls()


def services_mode() -> ServicesMode:
    """Resolve the ``services.mode`` setting through sandbox's own pydantic schema."""
    from .config_schema import RawServicesSection

    return _validate_section(RawServicesSection, "services").mode


def _default_services_mode() -> ServicesMode:
    """Default-factory indirection for [`SandboxConfig.services_mode`][terok_sandbox.config.SandboxConfig.services_mode].

    Lets tests patch ``terok_sandbox.config.services_mode`` and see the
    patch take effect at construction time — a direct
    ``default_factory=services_mode`` would capture the original function
    reference at class-definition time and ignore later patches.
    """
    return services_mode()


@functools.lru_cache(maxsize=1)
def _credentials_section() -> RawCredentialsSection:
    """Return a validated ``RawCredentialsSection`` from the layered config.

    Cached so the two field readers below share one pydantic pass per
    process — the daemon's per-scope-bind path re-resolves the chain
    on every reconcile event, and each resolution previously cost two
    validations.
    """
    from .config_schema import RawCredentialsSection

    return _validate_section(RawCredentialsSection, "credentials")


def credentials_passphrase() -> str | None:
    """Resolve the ``credentials.passphrase`` headless fallback through the schema."""
    return _credentials_section().passphrase


def credentials_use_keyring() -> bool:
    """Resolve the ``credentials.use_keyring`` opt-in flag through the schema."""
    return _credentials_section().use_keyring


def _default_credentials_passphrase() -> str | None:
    """Default-factory indirection so tests can patch ``credentials_passphrase``."""
    return credentials_passphrase()


def _default_credentials_use_keyring() -> bool:
    """Default-factory indirection so tests can patch ``credentials_use_keyring``."""
    return credentials_use_keyring()


@dataclass(frozen=True)
class SandboxConfig:
    """Immutable configuration for the sandbox layer.

    All paths default to the XDG/FHS-resolved values from [`paths`][terok_sandbox.paths].
    Override individual fields when constructing from terok's global config
    or when using terok-sandbox standalone.
    """

    state_dir: Path = field(default_factory=_state_root)
    """Writable state root (tokens, gate repos, task data)."""

    runtime_dir: Path = field(default_factory=_runtime_root)
    """Transient runtime directory (PID files, sockets)."""

    config_dir: Path = field(default_factory=_config_root)
    """Sandbox-scoped configuration root.

    Note: shield profiles are resolved by [`shield_profiles_dir`][terok_sandbox.config.SandboxConfig.shield_profiles_dir]
    via [`namespace_config_root`][terok_sandbox.paths.namespace_config_root], not from
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

    credentials_passphrase: str | None = field(default_factory=_default_credentials_passphrase)
    """Headless-no-keyring fallback for the SQLCipher passphrase.

    Read from ``credentials.passphrase`` in ``config.yml`` at construct
    time.  ``None`` (the default) means "no config-file fallback set"
    — callers fall through to the next tier in the resolution chain.
    """

    credentials_use_keyring: bool = field(default_factory=_default_credentials_use_keyring)
    """Opt-in switch for the OS keyring tier in the passphrase resolution chain.

    Off by default.  Linux Secret Service has per-collection (not
    per-item) ACLs, so authorising terok against the default collection
    grants read access to every other secret stored there.  Operators
    opt in via ``terok setup`` after weighing that trade-off.
    """

    services_mode: ServicesMode = field(default_factory=_default_services_mode)
    """Transport for host↔container IPC, resolved once at construction.

    The default factory validates the layered ``config.yml`` through
    [`RawServicesSection`][terok_sandbox.config_schema.RawServicesSection] — the same
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
    def vault_passphrase_file(self) -> Path:
        """Return the session-unlock tmpfs path for the SQLCipher passphrase.

        Lives under ``runtime_dir`` (``$XDG_RUNTIME_DIR/...``), so it is
        RAM-backed and cleared on reboot.  Written by
        ``terok-sandbox vault unlock``; read at daemon startup as the
        highest-priority tier of the passphrase resolution chain.
        """
        return self.runtime_dir / "vault.passphrase"

    @property
    def vault_systemd_creds_file(self) -> Path:
        """Return the sealed-credential path for the systemd-creds tier.

        Lives under ``vault_dir`` (persistent state, ``0o600``) — the
        credential is machine-bound (TPM2 or host key), so persistence
        across reboots is the whole point.  Written by
        ``terok-sandbox vault seal``; read on every chain walk via
        [`terok_sandbox.credentials.systemd_creds`][terok_sandbox.credentials.systemd_creds].
        """
        return self.vault_dir / "vault.passphrase.cred"

    def open_credential_db(
        self, db_path: Path | None = None, *, prompt_on_tty: bool = False
    ) -> Any:
        """Open the credentials DB with this config's resolution-chain knobs.

        Single seam over [`open_credential_db`][terok_sandbox.credentials.db.open_credential_db]
        so call sites never have to thread the tier-selection kwargs
        (``passphrase_file``, ``systemd_creds_file``, ``use_keyring``,
        ``config_fallback``) by hand — adding a new tier means
        editing *this* method only, no cross-package fan-out.

        *db_path* defaults to ``self.db_path``; callers that already
        hold a path (typically ``VaultStatus.db_path`` for the running
        daemon, or a test override) pass it explicitly so the open
        targets that DB while still using this config's tier policy.
        CLI consumers pass ``prompt_on_tty=True`` to unlock the
        interactive fallback; daemons leave it off.
        """
        from .credentials.db import open_credential_db  # noqa: PLC0415

        return open_credential_db(
            db_path if db_path is not None else self.db_path,
            passphrase_file=self.vault_passphrase_file,
            systemd_creds_file=self.vault_systemd_creds_file,
            use_keyring=self.credentials_use_keyring,
            config_fallback=self.credentials_passphrase,
            prompt_on_tty=prompt_on_tty,
        )

    def open_credential_db_with_source(
        self, db_path: Path | None = None, *, prompt_on_tty: bool = False
    ) -> tuple[CredentialDB, PassphraseSource]:
        """Same as [`open_credential_db`][terok_sandbox.SandboxConfig.open_credential_db]
        but also returns which tier of the chain hit.

        *db_path* override semantics match
        [`open_credential_db`][terok_sandbox.SandboxConfig.open_credential_db].
        The returned source flows into
        [`VaultStatus.passphrase_source`][terok_sandbox.VaultStatus] so
        callers don't have to second-guess the resolver.
        """
        from .credentials.db import open_credential_db_with_source  # noqa: PLC0415

        return open_credential_db_with_source(
            db_path if db_path is not None else self.db_path,
            passphrase_file=self.vault_passphrase_file,
            systemd_creds_file=self.vault_systemd_creds_file,
            use_keyring=self.credentials_use_keyring,
            config_fallback=self.credentials_passphrase,
            prompt_on_tty=prompt_on_tty,
        )

    def open_sqlcipher_connection(self, db_path: Path | None = None, **connect_kwargs: Any) -> Any:
        """Open a raw sqlcipher3 connection via the chain (vault daemon path)."""
        from .credentials.encryption import open_sqlcipher_via_chain  # noqa: PLC0415

        return open_sqlcipher_via_chain(
            db_path or self.db_path,
            passphrase_file=self.vault_passphrase_file,
            systemd_creds_file=self.vault_systemd_creds_file,
            use_keyring=self.credentials_use_keyring,
            config_fallback=self.credentials_passphrase,
            **connect_kwargs,
        )

    @property
    def routes_path(self) -> Path:
        """Return the path to the vault route configuration JSON."""
        return self.vault_dir / "routes.json"

    @property
    def credential_audit_log_path(self) -> Path:
        """Return the path to the credential-use audit JSONL.

        One file under the vault state dir, shared across every subject
        the broker has ever served — sandbox doesn't model
        "subject" semantically, so per-subject layout is the consumer's
        concern (terok's review CLI filters by ``scope`` / ``subject``).
        """
        return self.vault_dir / "credential_audit.jsonl"

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
            SSH keys are stored in [`db_path`][terok_sandbox.config.SandboxConfig.db_path] (table ``ssh_keys``) and
            served via per-scope sockets at
            [`ssh_signer_local_socket_path`][terok_sandbox.config.SandboxConfig.ssh_signer_local_socket_path].  This path is retained
            only for transitional callers in sibling packages; new code
            must not read or write it.
        """
        return self.vault_dir / "ssh-keys.json"

    def ssh_signer_local_socket_path(self, scope: str) -> Path:
        """Return the per-scope vault SSH-agent socket path for *scope*.

        The vault binds one 0600 Unix socket per scope with at least one
        assigned key, under the same ``runtime_dir`` as the main signer.
        Host-side ``gate-sync`` points ``SSH_AUTH_SOCK`` at this path.

        Rejects unsafe scope names with [`InvalidScopeName`][terok_sandbox.credentials.db.InvalidScopeName]
        as a belt-and-braces guard — writers in the DB layer enforce the
        same policy, but the socket path is public API and may be called
        without a preceding DB write.
        """
        from .credentials.db import _require_safe_scope

        _require_safe_scope(scope)
        return self.runtime_dir / f"ssh-agent-local-{scope}.sock"
