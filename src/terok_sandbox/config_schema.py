# SPDX-FileCopyrightText: 2026 Jiri Vyskocil
# SPDX-License-Identifier: Apache-2.0

"""Pydantic schema for the sandbox-owned slice of the shared ``config.yml``.

The ecosystem uses one ``~/.config/terok/config.yml`` file shared across
every package (Podman model — see :mod:`terok_sandbox.paths` for the
prior decision around umbrella roots).  Each package owns the schema
for the sub-sections it consumes; higher-level packages compose the
full file by importing from their dependencies.

This module is sandbox's contribution: the eight top-level sections
sandbox actually reads (``paths``, ``credentials``, ``vault``,
``gate_server``, ``services``, ``shield``, ``network``, ``ssh``), each
strict on its own keys (``extra="forbid"``), wrapped in
:class:`SandboxConfigView` whose top level is *tolerant*
(``extra="allow"``) so unknown sections — those owned by terok-executor
or terok — pass through silently when sandbox is run standalone.

Validation strategy:

- **Owned sub-sections** are strict.  A typo inside ``paths.rooot``
  is rejected at load time with a clear pydantic error.
- **Unknown top-level sections** are tolerated.  Sandbox doesn't know
  about terok's ``tui:`` or executor's ``image:``; rejecting them
  would make the standalone ``python -m terok_sandbox`` flow crash
  on any complete ecosystem config.

Higher-level packages inherit from :class:`SandboxConfigView` and add
their own sections.  The topmost layer (terok) flips back to
``extra="forbid"`` because it knows every section in the v0 ecosystem.
"""

from __future__ import annotations

from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field, model_validator

# ── Owned top-level sections ──────────────────────────────────────────


class RawCredentialsSection(BaseModel):
    """The ``credentials:`` section — vault routing for proxy DB and agent mounts."""

    model_config = ConfigDict(extra="forbid")

    dir: str | None = Field(
        default=None,
        description="Shared credentials directory (proxy DB, agent config mounts)",
    )


class RawPathsSection(BaseModel):
    """The ``paths:`` section — umbrella state root and per-purpose overrides.

    ``root`` is the namespace state root read by every ecosystem package
    (Podman model — see also :func:`terok_sandbox.paths.umbrella_state_dir`).
    """

    model_config = ConfigDict(extra="forbid")

    root: str | None = Field(
        default=None,
        description=(
            "Namespace state root shared by all ecosystem packages"
            " (Podman model — one config, multiple readers)"
        ),
    )
    build_dir: str | None = Field(
        default=None, description="Build artifacts directory (generated Dockerfiles)"
    )
    sandbox_live_dir: str | None = Field(
        default=None,
        description=(
            "Container-writable runtime data (tasks, agent mounts)."
            " For hardened installs, mount the target with ``noexec,nosuid,nodev``"
        ),
    )
    user_projects_dir: str | None = Field(
        default=None, description="User projects directory (per-user project configs)"
    )
    user_presets_dir: str | None = Field(
        default=None, description="User presets directory (per-user preset configs)"
    )
    port_registry_dir: str | None = Field(
        default=None, description="Shared port registry directory for multi-user isolation"
    )


class RawShieldSection(BaseModel):
    """The ``shield:`` section — egress firewall policy + audit + task lifecycle defaults."""

    model_config = ConfigDict(extra="forbid")

    bypass_firewall_no_protection: bool = Field(
        default=False, description="**Dangerous**: disable egress firewall entirely"
    )
    profiles: dict[str, Any] | None = Field(
        default=None, description="Named shield profiles for per-project firewall rules"
    )
    audit: bool = Field(default=True, description="Enable shield audit logging")
    drop_on_task_run: bool = True
    on_task_restart: Literal["retain", "up"] = "retain"


SERVICES_TCP_OPTOUT_YAML = "services: {mode: tcp}"
"""User-facing opt-out snippet shown in SELinux hints — keep in one place
so setup, sickbay, tests and docs stay in sync."""


class RawServicesSection(BaseModel):
    """The ``services:`` section — transport mode for host ↔ container IPC."""

    model_config = ConfigDict(extra="forbid")

    mode: Literal["tcp", "socket"] = "socket"
    """Transport for host↔container IPC.  Default ``socket`` since 0.7.3;
    set to ``tcp`` to opt out.  See ``docs/selinux.md``."""


class RawVaultSection(BaseModel):
    """The ``vault:`` section — token broker and SSH signer ports.

    The container-side transport was previously configured via
    ``vault.transport``; since 0.7.4 it is derived from
    ``services.mode`` so the two knobs stay in lockstep (tcp listener
    ↔ direct routing, socket listener ↔ socket routing).  Any prior
    ``vault.transport:`` line in ``config.yml`` must be removed.
    """

    model_config = ConfigDict(extra="forbid")

    bypass_no_secret_protection: bool = False
    port: int | None = Field(default=None, ge=1, le=65535)
    ssh_signer_port: int | None = Field(default=None, ge=1, le=65535)


class RawGateServerSection(BaseModel):
    """The ``gate_server:`` section — host-side gate listen port + repo dir."""

    model_config = ConfigDict(extra="forbid")

    port: int | None = Field(default=None, ge=1, le=65535, description="Gate server listen port")
    repos_dir: str | None = Field(
        default=None,
        description="Override gate repo directory (default: ``state_dir/gate``)",
    )
    suppress_systemd_warning: bool = Field(
        default=False, description="Suppress the systemd unit installation suggestion"
    )


class RawNetworkSection(BaseModel):
    """The ``network:`` section — port range for service / container ports."""

    model_config = ConfigDict(extra="forbid")

    port_range_start: int = Field(default=18700, ge=1024, le=65535)
    port_range_end: int = Field(default=32700, ge=1024, le=65535)

    @model_validator(mode="after")
    def _check_port_range(self) -> RawNetworkSection:
        """Reject inverted port ranges before they reach the registry."""
        if self.port_range_start > self.port_range_end:
            raise ValueError("port_range_start must be <= port_range_end")
        return self


class RawSSHSection(BaseModel):
    """The ``ssh:`` section — auth strategy for the host-side gate.

    Default is ``None`` (not ``False``) so ``model_dump(exclude_none=True)``
    can distinguish *unset* from *explicitly false*.  Higher layers may
    layer this with a ``project.yml`` ``ssh:`` section of the same shape;
    the ``None`` sentinel keeps the project layer from stomping the
    global value when the user omits it.  The effective ``False`` default
    happens at the consumer end.
    """

    model_config = ConfigDict(extra="forbid")

    use_personal: bool | None = Field(
        default=None,
        description=(
            "Opt in to the user's ``~/.ssh`` keys for host-side ``gate-sync``. "
            "Default ``false`` — terok uses only its vault-managed key. "
            "Resolves through ConfigStack: ``terok-global config.yml`` → "
            "``project.yml`` → CLI ``--use-personal-ssh`` (highest)."
        ),
    )


# ── Sandbox's view of the global config ───────────────────────────────


class SandboxConfigView(BaseModel):
    """The slice of ``config.yml`` sandbox owns and validates.

    ``extra="allow"`` at the top level so unknown sections (executor's
    ``image:``, terok's ``tui:`` / ``logs:`` / ``tasks:`` / ``git:`` /
    ``hooks:``) pass through silently when sandbox is run standalone —
    the ecosystem's shared config file is expected to contain *every*
    package's keys, and rejecting them would make ``python -m
    terok_sandbox`` crash on any complete config.

    Higher layers compose by inheriting from this class and adding
    their own typed fields:

    - :class:`terok_executor.config_schema.ExecutorConfigView`
      inherits and adds the ``image:`` section.
    - terok's ``RawGlobalConfig`` inherits and adds the remaining
      five terok-owned sections, then flips to ``extra="forbid"`` —
      the topmost layer knows every section, so a typo at the top
      level is caught there.
    """

    model_config = ConfigDict(extra="allow")

    credentials: RawCredentialsSection = Field(default_factory=RawCredentialsSection)
    paths: RawPathsSection = Field(default_factory=RawPathsSection)
    shield: RawShieldSection = Field(default_factory=RawShieldSection)
    services: RawServicesSection = Field(default_factory=RawServicesSection)
    vault: RawVaultSection = Field(default_factory=RawVaultSection)
    gate_server: RawGateServerSection = Field(default_factory=RawGateServerSection)
    network: RawNetworkSection = Field(default_factory=RawNetworkSection)
    ssh: RawSSHSection = Field(default_factory=RawSSHSection)


# ── Section readers ───────────────────────────────────────────────────


def gate_use_personal_ssh_default() -> bool:
    """Resolve the host gate's ``ssh.use_personal`` global default.

    Reads the ``ssh:`` section from the shared ``config.yml``, validates
    via :class:`RawSSHSection`, and returns the bool.  An unset section,
    a missing key, or a malformed value collapses to ``False`` — the
    safe historical default ("terok never touches your real keys").

    Higher layers compose this with project-level and per-invocation
    overrides; the resolution chain ends up:

        CLI ``--use-personal-ssh``     (highest)
        project ``project.yml`` ssh
        global ``config.yml`` ssh      ← THIS function
        False                          (default)

    Lives in sandbox because the consumer
    (:func:`~terok_sandbox.gate.mirror._git_env_with_ssh`) is here too —
    same package owns the schema and the reader.
    """
    from .paths import read_config_section

    raw = read_config_section("ssh")
    if not raw:
        return False
    try:
        section = RawSSHSection.model_validate(raw)
    except Exception:  # noqa: BLE001 — malformed config falls back to safe default
        return False
    return bool(section.use_personal)


__all__ = [
    "SERVICES_TCP_OPTOUT_YAML",
    "RawCredentialsSection",
    "RawGateServerSection",
    "RawNetworkSection",
    "RawPathsSection",
    "RawSSHSection",
    "RawServicesSection",
    "RawShieldSection",
    "RawVaultSection",
    "SandboxConfigView",
    "gate_use_personal_ssh_default",
]
