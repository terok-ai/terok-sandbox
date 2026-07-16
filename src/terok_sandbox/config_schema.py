# SPDX-FileCopyrightText: 2026 Jiri Vyskocil
# SPDX-License-Identifier: Apache-2.0

"""Pydantic schema for the sandbox-owned slice of the shared ``config.yml``.

The ecosystem uses one ``~/.config/terok/config.yml`` file shared across
every package (Podman model — see [`terok_sandbox.paths`][terok_sandbox.paths] for the
prior decision around umbrella roots).  Each package owns the schema
for the sub-sections it consumes; higher-level packages compose the
full file by importing from their dependencies.

This module is sandbox's contribution: the nine top-level sections
sandbox actually reads (``paths``, ``credentials``, ``vault``,
``gate_server``, ``services``, ``shield``, ``network``, ``ssh``,
``run``), each strict on its own keys (``extra="forbid"``), wrapped
in [`SandboxConfigView`][terok_sandbox.config_schema.SandboxConfigView]
whose top level is *tolerant* (``extra="allow"``) so unknown
sections — those owned by terok-executor or terok — pass through
silently when sandbox is run standalone.

Validation strategy:

- **Owned sub-sections** are strict.  A typo inside ``paths.rooot``
  is rejected at load time with a clear pydantic error.
- **Unknown top-level sections** are tolerated.  Sandbox doesn't know
  about terok's ``tui:`` or executor's ``image:``; rejecting them
  would make the standalone ``python -m terok_sandbox`` flow crash
  on any complete ecosystem config.

Higher-level packages inherit from [`SandboxConfigView`][terok_sandbox.config_schema.SandboxConfigView] and add
their own sections.  The topmost layer (terok) flips back to
``extra="forbid"`` because it knows every section in the v0 ecosystem.
"""

from __future__ import annotations

import re
from collections.abc import Sequence
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

_MEMORY_RE = re.compile(r"\d+(\.\d+)? ?[kKmMgGtTpP]?[iI]?[bB]?")
"""Format check for ``run.memory`` — mirrors ``docker/go-units.sizeRegex``,
which is what podman's ``--memory`` flag accepts.

Accepts ``"4g"``, ``"4gb"``, ``"4gib"``, ``"4 G"``, ``"512m"``, plain
``"1024"`` (bytes), all case-insensitive.  Format only — host-availability
and cgroup-minimum checks stay with podman.
"""

_CPUS_RE = re.compile(r"\d+(\.\d+)?")
"""Format check for ``run.cpus`` — non-negative decimal."""

type GpuVendor = Literal["nvidia", "amd", "intel"]

GPU_VENDORS: tuple[GpuVendor, ...] = ("nvidia", "amd", "intel")
"""Recognised ``run.gpus`` vendor tokens, in canonical (emission) order."""

type GpuSelector = tuple[GpuVendor, ...] | Literal["all"] | None
"""Normalized ``run.gpus`` value: explicit vendors, auto-detect, or off.

The vocabulary lives here, next to the field it types; the host-probing
and podman-arg side of the story is [`terok_sandbox.runtime.gpu`][terok_sandbox.runtime.gpu].
"""


def normalize_gpus(value: bool | str | Sequence[str] | None) -> GpuSelector:
    """Normalize a raw ``run.gpus`` value into a [`GpuSelector`][terok_sandbox.config_schema.GpuSelector].

    Accepts the shapes a YAML config or CLI flag produces: booleans,
    a single token (``"all"``, ``"nvidia"``, …), a comma-separated
    token string, or a list of tokens.  Raises ``ValueError`` on an
    unknown vendor token so misconfiguration surfaces at parse time,
    not at launch.
    """
    if value is None or value is False:
        return None
    if value is True:
        return "all"
    parts = [value] if isinstance(value, str) else list(value)
    tokens = [tok for part in parts for raw in part.split(",") if (tok := raw.strip().lower())]
    if not tokens:
        return None
    if unknown := [tok for tok in tokens if tok != "all" and tok not in GPU_VENDORS]:
        raise ValueError(
            f"run.gpus: unknown GPU vendor(s) {unknown!r}; "
            f"expected 'all', true, or any of {list(GPU_VENDORS)}"
        )
    if "all" in tokens:
        return "all"
    return tuple(vendor for vendor in GPU_VENDORS if vendor in tokens)


ServicesMode = Literal["tcp", "socket"]
"""Type alias for the ``services.mode`` Literal; re-exported from
`RawServicesSection.model_fields['mode']` so downstream modules
(sandbox's [`SandboxConfig`][terok_sandbox.config.SandboxConfig], terok's
``make_sandbox_config``) can annotate without re-declaring the shape."""


# ── Owned top-level sections ──────────────────────────────────────────


class RawCredentialsSection(BaseModel):
    """The ``credentials:`` section — vault routing for proxy DB and agent mounts."""

    model_config = ConfigDict(extra="forbid")

    dir: str | None = Field(
        default=None,
        description="Shared credentials directory (proxy DB, agent config mounts)",
    )
    passphrase: str | None = Field(
        default=None,
        description=(
            "REMOVED — the plaintext config tier no longer exists.  The"
            " validator below rejects any set value with migration"
            " directions; the field itself stays so the error names the"
            " replacement instead of pydantic's generic extra-key refusal."
        ),
    )
    use_keyring: bool = Field(
        default=True,
        description=(
            "The OS keyring tier of the passphrase resolution chain."
            "  On by default — an empty keyring simply doesn't resolve;"
            " set ``false`` to keep the chain away from Secret Service"
            " entirely (its ACLs are per-collection, not per-item)."
        ),
    )
    passphrase_command: str | None = Field(
        default=None,
        description=(
            "Operator-supplied shell command (e.g. ``pass show terok-sandbox/vault-passphrase``)"
            " that prints the SQLCipher passphrase on stdout.  Tokenised with"
            " ``shlex.split``; resolver tier slots below the OS keyring."
            "  Canonical headless option for hosts without systemd ≥ 257 —"
            " covers a plain secret file (``cat /path/to/file``), ``pass``,"
            " ``bw``, ``op``, HashiCorp ``vault``, and the cloud"
            " secret-manager CLIs (AWS, GCP, Azure)."
        ),
    )

    @field_validator("passphrase")
    @classmethod
    def _reject_removed_plaintext_tier(cls, value: str | None) -> str | None:
        """Refuse the removed plaintext tier with directions instead of silence.

        The value used to be a supported chain tier (plaintext-on-disk
        by explicit operator acceptance); dropping the field outright
        would surface as pydantic's opaque "extra inputs are not
        permitted".  Rejecting it here keeps the error actionable.
        """
        if value is None:
            return None
        raise ValueError(
            "credentials.passphrase (the plaintext config tier) was removed —"
            " move the value into its own file (mode 600) and set"
            " `passphrase_command: cat /path/to/that/file` instead"
        )


class RawPathsSection(BaseModel):
    """The ``paths:`` section — umbrella state root and per-purpose overrides.

    ``root`` is the namespace state root read by every ecosystem package
    (Podman model — see also `terok_sandbox.paths.umbrella_state_dir`).
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

    mode: ServicesMode = "socket"
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


class RawHooksSection(BaseModel):
    """Task lifecycle hook commands.

    Run on the **host** (not inside the container) around container
    lifecycle events.  Sandbox owns them because the lifecycle events
    themselves are sandbox-mediated — the orchestrator just opts into
    being notified.  The four hook points map to sandbox-internal
    transitions:

    - ``pre_start``: before the container exists (host-side prep).
    - ``post_start``: after the container is created but possibly not ready.
    - ``post_ready``: after the readiness marker has been observed.
    - ``post_stop``: after the container has stopped (cleanup hook).

    Each value is a shell command string, run by the host shell with
    the orchestrator's environment.  ``None`` means no hook.
    """

    model_config = ConfigDict(extra="forbid")

    pre_start: str | None = None
    post_start: str | None = None
    post_ready: str | None = None
    post_stop: str | None = None


class RawRunSection(BaseModel):
    """The ``run:`` section — "how the container runs".

    Covers OCI-runtime selection, container resource limits,
    capability toggles, environment, and lifecycle hooks.  Sandbox
    owns this because every field translates to a podman/runtime
    flag or annotation sandbox emits at launch time.

    Inheritable in both directions:

    - At the **global** level, defaults apply to every project
      (e.g. set ``runtime: krun`` once to opt the whole installation
      into microVM isolation).
    - At the **project** level, fields override the global default
      one-by-one via the orchestrator's merge logic.
    """

    model_config = ConfigDict(extra="forbid")

    shutdown_timeout: int = Field(
        default=10, description="Seconds to wait before SIGKILL on container stop"
    )
    gpus: str | bool | list[str] | None = Field(
        default=None,
        description=(
            'GPU passthrough: ``"all"``/``true`` (every vendor detected on the '
            'host), a vendor name (``"nvidia"``, ``"amd"``, ``"intel"``), or a '
            "list of vendor names; omit to disable"
        ),
    )
    memory: str | None = Field(
        default=None,
        description=(
            'Podman ``--memory`` value (e.g. ``"4g"``, ``"512m"``, ``"4gib"``, '
            'plain ``"1024"`` for bytes); ``None`` = unlimited.  Format mirrors '
            "what podman accepts — see ``man podman-run(1)`` --memory."
        ),
    )
    cpus: str | None = Field(
        default=None,
        description=(
            'Podman ``--cpus`` value (e.g. ``"2.0"``, ``"0.5"``); ``None`` '
            "= unlimited.  Non-negative decimal."
        ),
    )
    nested_containers: bool = Field(
        default=False,
        description=(
            "Declares that the project runs podman/docker inside its container. "
            "When true, the outer container is launched with ``--security-opt "
            "label=nested`` and ``--device /dev/fuse`` so rootless nested "
            "containers work under SELinux without disabling labels wholesale."
        ),
    )
    runtime: Literal["crun", "krun"] | None = Field(
        default=None,
        description=(
            "OCI runtime: ``crun`` (default) for conventional containers, "
            "or ``krun`` for KVM-microVM isolation (experimental).  ``None`` "
            "resolves to ``crun`` — the OCI runtime podman picks by default "
            "on every supported distro.  ``krun`` requires the global "
            "``experimental: true`` flag at task launch."
        ),
    )
    timezone: str | None = Field(
        default=None,
        description=(
            "IANA timezone for the task container (e.g. ``Europe/Prague``, "
            "``UTC``).  Propagated as ``TZ`` — resolved against the image's "
            "``tzdata``.  Unset (default) means follow the host's timezone."
        ),
    )
    hooks: RawHooksSection = Field(default_factory=RawHooksSection)

    @field_validator("gpus", mode="before")
    @classmethod
    def _reject_numeric_gpus(cls, v: Any) -> Any:
        """Reject numeric YAML shapes (``gpus: 1``) before bool coercion.

        Pydantic's lax mode would coerce ``0``/``1``/``1.0`` through the
        ``bool`` branch and silently enable (or disable) every GPU; only
        explicit booleans, selector strings, and lists carry meaning here.
        """
        if isinstance(v, int | float) and not isinstance(v, bool):
            raise ValueError(
                f"gpus {v!r}: expected true/false, 'all', vendor names, or a list of vendors"
            )
        return v

    @field_validator("gpus", mode="after")
    @classmethod
    def _validate_gpus_tokens(
        cls, v: str | bool | list[str] | None
    ) -> str | bool | list[str] | None:
        """Reject unknown GPU vendor tokens at parse time.

        Delegates to [`normalize_gpus`][terok_sandbox.config_schema.normalize_gpus]
        (the launch-path normalizer) so config validation and launch
        behaviour can never disagree; the raw shape is preserved.
        """
        normalize_gpus(v)
        return v

    @field_validator("memory", "cpus", mode="before")
    @classmethod
    def _normalise(cls, v: Any) -> Any:
        """Coerce numeric YAML inputs to str; blank strings to ``None``.

        Accepts ``cpus: 2`` / ``memory: 1024`` (YAML int/float) by
        stringifying — both shapes are valid podman input.  ``bool`` is
        an ``int`` subclass, so reject it explicitly to keep ``cpus: true``
        from coercing to ``"True"``.
        """
        if isinstance(v, bool):
            return v  # let pydantic reject as not-str
        if isinstance(v, int | float):
            return str(v)
        if isinstance(v, str) and not v.strip():
            return None
        return v

    @field_validator("memory", mode="after")
    @classmethod
    def _validate_memory_format(cls, v: str | None) -> str | None:
        """Reject malformed ``memory`` at parse time.

        Format only — semantic checks (host RAM, cgroup minimum) stay
        with podman.
        """
        if v is None:
            return v
        if not _MEMORY_RE.fullmatch(v):
            raise ValueError(
                f"memory {v!r}: expected podman-style size (e.g. ``4g``, "
                "``512m``, ``4gib``); see man podman-run(1) --memory"
            )
        return v

    @field_validator("cpus", mode="after")
    @classmethod
    def _validate_cpus_format(cls, v: str | None) -> str | None:
        """Reject malformed ``cpus`` at parse time."""
        if v is None:
            return v
        if not _CPUS_RE.fullmatch(v):
            raise ValueError(
                f"cpus {v!r}: expected non-negative decimal (see man podman-run(1) --cpus)"
            )
        return v

    @model_validator(mode="before")
    @classmethod
    def _coerce_none_subsections(cls, data: Any) -> Any:
        """Coerce a ``None`` ``hooks:`` value to the empty defaults dict."""
        if isinstance(data, dict) and data.get("hooks") is None:
            data["hooks"] = {}
        return data


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

    - [`terok_executor.config_schema.ExecutorConfigView`][terok_executor.config_schema.ExecutorConfigView]
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
    run: RawRunSection = Field(default_factory=RawRunSection)
    experimental: bool = Field(
        default=False,
        description=(
            "Cross-package opt-in for experimental features.  Gates terok's "
            "krun runtime and sandbox's krun-only host-binary prereq probes "
            "(``ip``).  Lives on the top level rather than in any one "
            "section because it's shared between sandbox, executor, and "
            "terok — the topmost layer (terok) inherits this declaration."
        ),
    )


# ── Section readers ───────────────────────────────────────────────────


def gate_use_personal_ssh_default() -> bool:
    """Resolve the host gate's ``ssh.use_personal`` global default.

    Reads the ``ssh:`` section from the shared ``config.yml``, validates
    via [`RawSSHSection`][terok_sandbox.config_schema.RawSSHSection], and returns the bool.  An unset section,
    a missing key, or a malformed value collapses to ``False`` — the
    safe historical default ("terok never touches your real keys").

    Higher layers compose this with project-level and per-invocation
    overrides; the resolution chain ends up:

        CLI ``--use-personal-ssh``     (highest)
        project ``project.yml`` ssh
        global ``config.yml`` ssh      ← THIS function
        False                          (default)

    Lives in sandbox because the consumer
    (`_git_env_with_ssh`) is here too —
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
    "ServicesMode",
    "gate_use_personal_ssh_default",
]
