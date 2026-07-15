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
    read_config_top_level,
    runtime_root as _runtime_root,
    state_root as _state_root,
    vault_root as _vault_root,
)

if TYPE_CHECKING:
    from .config_schema import (
        RawCredentialsSection,
        RawGateServerSection,
        RawShieldSection,
        RawVaultSection,
        ServicesMode,
    )
    from .vault.store.db import CredentialDB
    from .vault.store.tiers import PassphraseTier

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
    process — the per-scope-bind path re-resolves the chain on every
    bind, and without the cache each resolution would cost two
    validations.
    """
    from .config_schema import RawCredentialsSection

    return _validate_section(RawCredentialsSection, "credentials")


def credentials_use_keyring() -> bool:
    """Resolve the ``credentials.use_keyring`` opt-in flag through the schema."""
    return _credentials_section().use_keyring


def credentials_passphrase_command() -> str | None:
    """Resolve the ``credentials.passphrase_command`` shell-helper recipe through the schema."""
    return _credentials_section().passphrase_command


def _default_credentials_use_keyring() -> bool:
    """Default-factory indirection so tests can patch ``credentials_use_keyring``."""
    return credentials_use_keyring()


def _default_credentials_passphrase_command() -> str | None:
    """Default-factory indirection so tests can patch ``credentials_passphrase_command``."""
    return credentials_passphrase_command()


@functools.lru_cache(maxsize=1)
def _shield_section() -> RawShieldSection:
    """Return a validated ``RawShieldSection`` from the layered config."""
    from .config_schema import RawShieldSection

    return _validate_section(RawShieldSection, "shield")


def shield_audit() -> bool:
    """Resolve the ``shield.audit`` setting through the schema."""
    return _shield_section().audit


def _default_shield_audit() -> bool:
    """Default-factory indirection so tests can patch ``shield_audit``."""
    return shield_audit()


def experimental_enabled() -> bool:
    """Resolve the top-level ``experimental:`` opt-in from the layered config.

    Ecosystem-wide flag: shared between sandbox (krun host-binary
    prereqs), executor (krun runtime construction), and terok (krun
    runtime selection at task launch).  Defaults to ``False`` when the
    key is absent or malformed.
    """
    raw = read_config_top_level("experimental")
    if isinstance(raw, bool):
        return raw
    return False


def _default_experimental() -> bool:
    """Default-factory indirection so tests can patch ``experimental_enabled``."""
    return experimental_enabled()


@functools.lru_cache(maxsize=1)
def _vault_section() -> RawVaultSection:
    """Return a validated ``RawVaultSection`` from the layered config."""
    from .config_schema import RawVaultSection

    return _validate_section(RawVaultSection, "vault")


@functools.lru_cache(maxsize=1)
def _gate_server_section() -> RawGateServerSection:
    """Return a validated ``RawGateServerSection`` from the layered config."""
    from .config_schema import RawGateServerSection

    return _validate_section(RawGateServerSection, "gate_server")


def gate_server_port() -> int | None:
    """Resolve ``gate_server.port`` through the schema; ``None`` = auto-allocate."""
    return _gate_server_section().port


def vault_token_broker_port() -> int | None:
    """Resolve ``vault.port`` through the schema; ``None`` = auto-allocate."""
    return _vault_section().port


def vault_ssh_signer_port() -> int | None:
    """Resolve ``vault.ssh_signer_port`` through the schema; ``None`` = auto-allocate."""
    return _vault_section().ssh_signer_port


def _default_gate_port() -> int | None:
    """Default-factory indirection so tests can patch ``gate_server_port``."""
    return gate_server_port()


def _default_token_broker_port() -> int | None:
    """Default-factory indirection so tests can patch ``vault_token_broker_port``."""
    return vault_token_broker_port()


def _default_ssh_signer_port() -> int | None:
    """Default-factory indirection so tests can patch ``vault_ssh_signer_port``."""
    return vault_ssh_signer_port()


# Deliberately not exposing a ``shield_bypass()`` reader nor a
# ``_default_shield_bypass`` factory.  ``shield.bypass_firewall_no_protection``
# is in the pydantic schema (orchestrators can pass it through their
# own resolution chain) but ``SandboxConfig.shield_bypass`` stays
# hardcoded ``False``: enabling bypass via a user-writable config
# scope (``~/.config/terok/config.yml``) or via ``TEROK_CONFIG_FILE``
# would let anything that can drop a file under ``$HOME`` silently
# disable the egress firewall.  Higher-layer orchestrators are
# trusted to acknowledge the risk explicitly when they set the field.


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

    gate_port: int | None = field(default_factory=_default_gate_port)
    """HTTP port for the gate server (``None`` = auto-allocate via registry).

    Default-factory reads ``gate_server.port`` from config.yml; missing
    or unset keys fall through to ``None`` so the port registry can
    pick one.  Direct ``SandboxConfig(gate_port=…)`` always wins.
    """

    token_broker_port: int | None = field(default_factory=_default_token_broker_port)
    """TCP port for the vault's token broker (``None`` = auto-allocate via registry).

    Default-factory reads ``vault.port`` from config.yml.
    """

    ssh_signer_port: int | None = field(default_factory=_default_ssh_signer_port)
    """TCP port for the vault's SSH signer (``None`` = auto-allocate via registry).

    Default-factory reads ``vault.ssh_signer_port`` from config.yml.
    """

    shield_profiles: tuple[str, ...] = ("dev-standard",)
    """Shield egress firewall profile names."""

    shield_audit: bool = field(default_factory=_default_shield_audit)
    """Whether shield audit logging is enabled.

    Default-factory reads ``shield.audit`` from the layered config.yml
    via the [`RawShieldSection`][terok_sandbox.config_schema.RawShieldSection]
    schema; missing/typo'd keys fall back to the schema's ``True``
    default.  Direct ``SandboxConfig(shield_audit=…)`` always wins.
    """

    shield_bypass: bool = False
    """DANGEROUS: when True, the egress firewall is completely disabled.

    Hardcoded ``False`` here — sandbox refuses to read this field
    from ``config.yml`` because the layered chain includes a
    user-writable scope (``~/.config/terok/config.yml``) and an
    ``$ENV``-controllable override (``TEROK_CONFIG_FILE``), so anything
    that drops a file in ``$HOME`` could silently disable the egress
    firewall.  Orchestrators that want bypass must pass it explicitly
    to ``SandboxConfig(shield_bypass=True)`` after resolving from
    their own trusted source.
    """

    credentials_use_keyring: bool = field(default_factory=_default_credentials_use_keyring)
    """Switch for the OS keyring tier in the passphrase resolution chain.

    On by default — an empty keyring simply doesn't resolve, so the
    tier costs nothing until something lands a value there.  Operators
    who want the chain to stay away from Secret Service entirely (its
    ACLs are per-collection, not per-item, so authorising terok against
    the default collection grants read access to every other secret
    stored there) set ``credentials.use_keyring: false``.
    """

    credentials_passphrase_command: str | None = field(
        default_factory=_default_credentials_passphrase_command
    )
    """Operator-supplied shell command that prints the SQLCipher passphrase on stdout.

    Resolver tier slotted between ``keyring`` and ``config``.  Canonical
    headless option for hosts without systemd ≥ 257 — same shape as
    ``git config credential.helper`` or ``BORG_PASSCOMMAND``.  Read
    from ``credentials.passphrase_command`` in ``config.yml`` at
    construct time; ``None`` (the default) means "no helper configured"
    and the resolver skips this tier.
    """

    services_mode: ServicesMode = field(default_factory=_default_services_mode)
    """Transport for host↔container IPC, resolved once at construction.

    Validated through the same
    [`RawServicesSection`][terok_sandbox.config_schema.RawServicesSection]
    schema terok's ``RawGlobalConfig`` composes, so standalone and
    embedded paths agree on the value.  Lives as an instance attribute
    rather than a free-function call per site so downstream code can't
    bypass config resolution — no manager without a ``SandboxConfig``,
    every ``SandboxConfig`` carries a resolved mode.
    """

    experimental: bool = field(default_factory=_default_experimental)
    """Whether the ecosystem-wide ``experimental:`` opt-in is on.

    Cross-package switch: gates terok's krun runtime at task launch
    and sandbox's krun-only prereq probes (currently just ``ip``) at
    ``terok-sandbox setup``.  Read from the top-level ``experimental:``
    key in the layered ``config.yml`` at construct time; missing /
    typo'd values fall back to ``False``.  Direct
    ``SandboxConfig(experimental=…)`` always wins.
    """

    def with_resolved_ports(self) -> SandboxConfig:
        """Return a copy with TCP ports allocated via the shared port registry.

        Idempotent — returns ``self`` (no copy) when there is nothing
        to allocate: socket mode never needs TCP listeners, and
        already-fully-resolved cfgs short-circuit.

        **Side-effectful**: allocation hits the shared port registry,
        bind-tests each candidate, and persists the claim to
        ``state_dir/port-claims.json``.  Keep this call OUT of
        construction paths that don't actually launch services
        (sickbay checks, config inspection, tests) — that's why it's
        opt-in rather than baked into ``__post_init__``.  The
        consumers that *do* need real ports (``ShieldManager``,
        ``Sandbox``) wrap their stored cfg in
        ``self._cfg = self._cfg.with_resolved_ports()`` at construction
        time so downstream code never sees ``None`` for the port it
        needs.
        """
        if self.services_mode == "socket":
            return self
        if (
            self.gate_port is not None
            and self.token_broker_port is not None
            and self.ssh_signer_port is not None
        ):
            return self
        from dataclasses import replace

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
        return replace(
            self,
            gate_port=self.gate_port if self.gate_port is not None else ports.gate,
            token_broker_port=(
                self.token_broker_port if self.token_broker_port is not None else ports.proxy
            ),
            ssh_signer_port=(
                self.ssh_signer_port if self.ssh_signer_port is not None else ports.ssh_agent
            ),
        )

    @property
    def gate_base_path(self) -> Path:
        """Return the gate server's repo base path."""
        return self.state_dir / "gate"

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
    def vault_rekey_stamp_file(self) -> Path:
        """Return the marker whose mtime records the last passphrase change.

        Touched by [`change_passphrase`][terok_sandbox.commands.vault.change_passphrase]
        so health surfaces can flag supervisors that were spawned *before*
        the rekey — they keep the passphrase they resolved at spawn and
        need a restart to pick up the new one.  Lives under
        ``runtime_dir`` deliberately: it vanishes on reboot together
        with every process it could possibly indict, so a stale stamp
        can never outlive the problem it describes.
        """
        return self.runtime_dir / "vault.rekeyed_at"

    def container_runtime_dir(self, container_name: str) -> Path:
        """Host-side per-container runtime dir, bind-mounted at ``/run/terok``.

        The single source of the ``runtime_dir/run/<name>`` convention:
        the launch path bind-mounts this directory into the container so
        the supervisor's ``vault.sock`` / ``ssh-agent.sock`` /
        ``gate-server.sock`` (socket mode) surface at the well-known
        ``/run/terok/`` paths inside it.  A method, not a property,
        because it is keyed on the container name.

        *container_name* must be a single safe path component — the
        returned path is ``mkdir``'d, ``chmod``'d, and ``rmtree``'d by
        callers ([`ensure_container_runtime_dir`][terok_sandbox.config.SandboxConfig.ensure_container_runtime_dir],
        [`remove_container_state`][terok_sandbox.launch.remove_container_state]),
        so a name carrying a separator or ``..`` (or an absolute path,
        which ``Path.__truediv__`` would let swallow the whole prefix)
        could redirect those mutations outside the runtime dir.  This is
        the same guard [`write_sidecar`][terok_sandbox.launch.write_sidecar]
        and the supervisor's ``load_sidecar`` apply at their entry points.

        Raises:
            ValueError: If *container_name* is empty or not a single path
                component (contains ``/`` or ``\\``, or is ``.``/``..``).
        """
        if (
            not container_name
            or container_name in (".", "..")
            or "/" in container_name
            or "\\" in container_name
        ):
            raise ValueError(
                f"unsafe container name (not a single path component): {container_name!r}"
            )
        return self.runtime_dir / "run" / container_name

    def ensure_container_runtime_dir(self, container_name: str) -> Path:
        """(Re)create [`container_runtime_dir`][terok_sandbox.config.SandboxConfig.container_runtime_dir] (mode 0700) and return it.

        Idempotent, and it must be re-callable: the directory is gone for
        two routine reasons by the time a stopped container restarts — it
        lives under ``runtime_dir`` (``$XDG_RUNTIME_DIR``), a tmpfs the OS
        clears on logout/reboot, and the per-container supervisor
        ``rmtree``s it on every stop.  ``podman start`` re-binds the
        ``/run/terok`` mount from this exact source, so it must exist
        first.  A plain stop/start survives because podman recreates the
        leaf while its parent lives; a reboot wipes the whole ``…/run``
        chain and ``mkdir(parents=True)`` is what rebuilds it.
        """
        run_dir = self.container_runtime_dir(container_name)
        run_dir.mkdir(parents=True, exist_ok=True)
        run_dir.chmod(0o700)
        return run_dir

    @property
    def vault_systemd_creds_file(self) -> Path:
        """Return the sealed-credential path for the systemd-creds tier.

        Lives under ``vault_dir`` (persistent state, ``0o600``) — the
        credential is machine-bound (TPM2 or host key), so persistence
        across reboots is the whole point.  Written by
        ``terok-sandbox vault seal``; read on every chain walk via
        [`terok_sandbox.vault.store.systemd_creds`][terok_sandbox.vault.store.systemd_creds].
        """
        return self.vault_dir / "vault.passphrase.cred"

    @property
    def vault_recovery_marker_file(self) -> Path:
        """Return the sidecar marker path for "operator saved the recovery passphrase".

        Lives next to the sealed-credential file (persistent state,
        ``0o600``).  A **zero-byte** file — deliberately no passphrase
        fingerprint (that would be an offline-guessing oracle, see
        [`terok_sandbox.vault.store.recovery`][terok_sandbox.vault.store.recovery]),
        so a re-key does not auto-invalidate it;
        [`change_passphrase`][terok_sandbox.commands.vault.change_passphrase]
        and ``vault lock`` drop the marker themselves.
        """
        return self.vault_dir / "vault.recovery_acknowledged"

    def open_credential_db(
        self, db_path: Path | None = None, *, prompt_on_tty: bool = False
    ) -> Any:
        """Open the credentials DB with this config's resolution-chain knobs.

        Single seam over [`open_credential_db`][terok_sandbox.vault.store.db.open_credential_db]
        so call sites never plumb tier-selection kwargs by hand — adding
        a new tier is one entry in the private ``_chain_kwargs`` helper,
        no cross-package fan-out.

        *db_path* defaults to ``self.db_path``; callers that already
        hold a path (a sidecar-pinned DB path, or a test override) pass
        it explicitly so the open targets that DB while still using
        this config's tier policy.  CLI consumers pass
        ``prompt_on_tty=True`` to unlock the interactive fallback;
        the per-container supervisor leaves it off.
        """
        from .vault.store.db import open_credential_db  # noqa: PLC0415

        return open_credential_db(
            db_path if db_path is not None else self.db_path,
            **self._chain_kwargs(prompt_on_tty=prompt_on_tty),
        )

    def open_credential_db_with_source(
        self, db_path: Path | None = None, *, prompt_on_tty: bool = False
    ) -> tuple[CredentialDB, PassphraseTier]:
        """Same as [`open_credential_db`][terok_sandbox.SandboxConfig.open_credential_db]
        but also returns which tier of the chain hit.

        *db_path* override semantics match
        [`open_credential_db`][terok_sandbox.SandboxConfig.open_credential_db].
        The returned source lets callers (status reports, the
        supervisor startup log) name which tier unlocked the vault
        instead of second-guessing the resolver.
        """
        from .vault.store.db import open_credential_db_with_source  # noqa: PLC0415

        return open_credential_db_with_source(
            db_path if db_path is not None else self.db_path,
            **self._chain_kwargs(prompt_on_tty=prompt_on_tty),
        )

    def open_sqlcipher_connection(self, db_path: Path | None = None, **connect_kwargs: Any) -> Any:
        """Open a raw sqlcipher3 connection via the chain (vault daemon path)."""
        from .vault.store.encryption import open_sqlcipher_via_chain  # noqa: PLC0415

        return open_sqlcipher_via_chain(
            db_path or self.db_path,
            **self._chain_kwargs(prompt_on_tty=False),
            **connect_kwargs,
        )

    def resolve_passphrase(self, *, prompt_on_tty: bool = False) -> str | None:
        """Walk the resolution chain with this config's knobs; return the passphrase or ``None``.

        Diagnostic seam — never opens the DB.  Used by host-side
        doctor / sickbay and by ``vault seal`` to reuse whatever tier
        currently has the key.  Same chain order as
        [`open_credential_db`][terok_sandbox.SandboxConfig.open_credential_db]
        because both delegate here.
        """
        from .vault.store.encryption import resolve_passphrase  # noqa: PLC0415

        return resolve_passphrase(**self._chain_kwargs(prompt_on_tty=prompt_on_tty))

    def resolve_passphrase_with_source(
        self, *, prompt_on_tty: bool = False
    ) -> tuple[str | None, PassphraseTier | None]:
        """Walk the resolution chain with this config's knobs; return ``(passphrase, source)``.

        Diagnostic counterpart to
        [`resolve_passphrase`][terok_sandbox.SandboxConfig.resolve_passphrase]
        — feeds the daemon startup log so the operator sees *which*
        tier unlocked the vault on this boot.
        """
        from .vault.store.encryption import resolve_passphrase_with_source  # noqa: PLC0415

        return resolve_passphrase_with_source(**self._chain_kwargs(prompt_on_tty=prompt_on_tty))

    def _chain_kwargs(self, *, prompt_on_tty: bool) -> dict[str, Any]:
        """Return the shared resolver kwargs every chain entry point threads through.

        Adding a new tier is one extra entry here rather than a fan-out
        across the five resolver entry points and their downstream call
        sites — the authoritative list of tier knobs lives here.
        """
        return {
            "passphrase_file": self.vault_passphrase_file,
            "systemd_creds_file": self.vault_systemd_creds_file,
            "use_keyring": self.credentials_use_keyring,
            "passphrase_command": self.credentials_passphrase_command,
            "prompt_on_tty": prompt_on_tty,
        }

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

    def ssh_signer_local_socket_path(self, scope: str) -> Path:
        """Return the per-scope vault SSH-agent socket path for *scope*.

        The vault binds one 0600 Unix socket per scope with at least one
        assigned key, under the same ``runtime_dir`` as the main signer.
        Host-side ``gate-sync`` points ``SSH_AUTH_SOCK`` at this path.

        Rejects unsafe scope names with [`InvalidScopeName`][terok_sandbox.vault.store.db.InvalidScopeName]
        as a belt-and-braces guard — writers in the DB layer enforce the
        same policy, but the socket path is public API and may be called
        without a preceding DB write.
        """
        from .vault.store.db import _require_safe_scope

        _require_safe_scope(scope)
        return self.runtime_dir / f"ssh-agent-local-{scope}.sock"
