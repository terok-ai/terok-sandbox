# SPDX-FileCopyrightText: 2025 Jiri Vyskocil
# SPDX-License-Identifier: Apache-2.0

"""Adapter for terok-shield egress firewall.

Two classes carry the sandbox-side policy layer over terok-shield:

* [`ShieldManager`][terok_sandbox.integrations.shield.ShieldManager] —
  per-task wrapper around [`Shield`][terok_shield.Shield].  Caches the
  underlying instance.  Bypassable methods (``pre_start``, ``up``,
  ``down``, ``check_environment``) short-circuit when
  ``shield_bypass`` is set; non-bypassable methods (``quarantine``,
  ``state``) always hit the live shield because panic overrides every
  safety bypass and state probes report what nft actually sees.
  ``status`` is config-level only and surfaces the bypass flag in
  its dict rather than short-circuiting.
* [`ShieldHooks`][terok_sandbox.integrations.shield.ShieldHooks] — the
  host-wide OCI hooks installer, scoped to the root/user dual-scope
  flag pair the ``terok setup`` and ``terok-sandbox`` CLIs expose.
  Delegates to terok-shield's [`HooksInstaller`][terok_shield.HooksInstaller]
  for the actual file writes; terok-shield owns the on-disk install
  layout, so sandbox carries no private mirror of it.
"""

from __future__ import annotations

import tempfile
import warnings
from functools import cached_property
from pathlib import Path
from typing import TYPE_CHECKING

from terok_shield import (
    EnvironmentCheck,  # noqa: F401 — re-exported
    HooksInstaller,
    Shield,
    ShieldConfig,
    ShieldMode,
    ShieldRuntime,
    ShieldState,  # noqa: F401 — re-exported
)
from terok_shield.container import (
    resolve_state_dir as resolve_container_state_dir,  # noqa: F401 — re-exported
)

# ``ensure_user_hooks_dir_configured`` lives behind shield's lazy ``__getattr__``
# (so the package's top-level import stays light); pull it from the owning
# submodule for a concrete callable type.
from terok_shield.hooks.install import (
    ensure_user_hooks_dir_configured,  # noqa: F401 — re-exported with concrete type
)

# Several symbols are exposed via the top-level ``terok_shield.__getattr__``
# lazy importer which returns ``object`` to type-checkers — that breaks
# both ``except`` narrowing on the error classes and concrete typing on
# the prereqs dataclasses.  Pull them straight from the owning submodules.
from terok_shield.prereqs import (  # noqa: F401 — re-exported with concrete types
    BinaryCheck,
    check_firewall_binaries,
    check_krun_binaries,
)
from terok_shield.run import NftNotFoundError, ShieldNeedsSetup  # noqa: F401

from ..config import SandboxConfig

if TYPE_CHECKING:
    from terok_shield import HOOK_ENTRYPOINT_NAME  # noqa: F401 — re-export typing

#: Warning emitted by every bypassable [`ShieldManager`][terok_sandbox.integrations.shield.ShieldManager]
#: method when ``shield_bypass`` is set.  DANGEROUS TRANSITIONAL OVERRIDE —
#: will be removed once terok-shield supports all target podman versions
#: (see terok-shield#71, #101).
_BYPASS_WARNING = (
    "WARNING: shield.bypass_firewall_no_protection is set — "
    "the egress firewall is DISABLED.  Containers have unrestricted "
    "network access.  Remove this setting once your podman version "
    "is compatible with terok-shield."
)


# ── Per-task shield manager ─────────────────────────────


class ShieldManager:
    """Per-task wrapper around [`Shield`][terok_shield.Shield].

    Holds the (task_dir, cfg, runtime) tuple a Shield is built from
    and caches the constructed instance — the previous free-function
    surface rebuilt a Shield on every call, which paid the
    ``ShieldConfig`` + collaborator-wiring cost twice for every
    transition pair (``pre_start`` → ``up``, ``up`` → ``down``, …).

    Bypassable methods (``pre_start``, ``up``, ``down``) short-circuit
    when ``shield_bypass`` is set on the configuration.
    Non-bypassable methods (``quarantine``, ``state``) always run —
    panic overrides every safety bypass, and state probes report what
    nft actually sees regardless of operator intent.
    """

    def __init__(
        self,
        task_dir: Path,
        cfg: SandboxConfig | None = None,
        *,
        runtime: ShieldRuntime = ShieldRuntime.DEFAULT,
        loopback_ports_override: tuple[int, ...] | None = None,
    ) -> None:
        """Bind the manager to a task directory and shield configuration.

        *runtime* selects the container runtime category — ``DEFAULT``
        for crun/runc/youki (dnsmasq on netns ``127.0.0.1``), ``KRUN``
        for the libkrun microVM path (dnsmasq on a link-local address
        the guest can reach via passt).  Callers that drive the launch
        path map their runtime string (``RunSpec.runtime``) to the
        enum.

        *loopback_ports_override* replaces the cfg-derived
        ``(gate_port, token_broker_port, ssh_signer_port)`` triple — the
        per-container launch path passes the freshly-allocated broker
        and signer ports so shield's nft rules allow the actual host
        ports the supervisor binds.
        """
        self._task_dir = task_dir
        self._cfg = cfg or SandboxConfig()
        self._runtime = runtime
        self._loopback_ports_override = loopback_ports_override

    @property
    def state_dir(self) -> Path:
        """Per-task shield state directory: ``{task_dir}/shield``."""
        return self._task_dir / "shield"

    @property
    def bypass(self) -> bool:
        """True when ``shield_bypass`` is set on the sandbox configuration."""
        return self._cfg.shield_bypass

    @cached_property
    def shield(self) -> Shield:
        """Lazily constructed [`Shield`][terok_shield.Shield] instance.

        Built from a [`ShieldConfig`][terok_shield.ShieldConfig] whose
        ``loopback_ports`` reflect the *actual* gate/broker/signer
        ports — auto-allocated configs default those fields to ``None``,
        which would otherwise silently produce an empty tuple and a
        shield ruleset with no
        ``tcp dport <p> ip daddr 169.254.1.2 accept`` rules, causing
        container→host TCP traffic to fall through to the
        private-range reject (#156 regression follow-up).
        """
        resolved = self._cfg.with_resolved_ports()
        # ``loopback_ports_override`` wins when set (the per-container
        # launch path's freshly-allocated broker/signer ports).  Falls
        # back to the cfg-resolved triple for callers that haven't been
        # plumbed through yet.  Socket-mode transports emit no loopback
        # traffic; filter ``None`` so the nft rule generator only sees
        # ports that actually exist.
        loopback = (
            self._loopback_ports_override
            if self._loopback_ports_override is not None
            else tuple(
                p
                for p in (resolved.gate_port, resolved.token_broker_port, resolved.ssh_signer_port)
                if p is not None
            )
        )
        config = ShieldConfig(
            state_dir=self.state_dir,
            mode=ShieldMode.HOOK,
            default_profiles=resolved.shield_profiles,
            loopback_ports=loopback,
            audit_enabled=resolved.shield_audit,
            profiles_dir=resolved.shield_profiles_dir,
            runtime=self._runtime,
        )
        return Shield(config)

    # ── Bypassable lifecycle operations ─────────────────

    def pre_start(
        self,
        container: str,
        *,
        security_deny: tuple[str, ...] = (),
        provider_allow: tuple[str, ...] = (),
        project_allow: tuple[str, ...] = (),
        override: tuple[str, ...] = (),
    ) -> list[str]:
        """Return extra ``podman run`` args for egress firewalling.

        The four tier arguments are the orchestrator's generated policy tiers,
        which shield writes into the bundle so this layer only carries the data:
        *security_deny* → t20 (deny direct-to-vault-host), *provider_allow* → t30
        (provider egress), *project_allow* → t40 (git remote + custom, merged
        with the composed profiles), *override* → t10 (break-glass allow above
        the deny).  Empty tuples (the default) leave a tier untouched.

        Returns an empty list (no firewall args) when the dangerous
        ``bypass_firewall_no_protection`` override is active.

        Raises [`SystemExit`][SystemExit] with setup instructions when
        the podman environment requires one-time hook installation.
        """
        if self.bypass:
            warnings.warn(_BYPASS_WARNING, stacklevel=2)
            return []
        try:
            return self.shield.pre_start(
                container,
                security_deny=security_deny,
                provider_allow=provider_allow,
                project_allow=project_allow,
                override=override,
            )
        except ShieldNeedsSetup as exc:
            raise SystemExit(str(exc)) from None

    def up(self, container: str, container_id: str) -> None:
        """Set shield to deny-all mode for a running container.

        *container* is the operator-facing podman name (audit-log key);
        *container_id* is the full podman UUID — terok-shield's per-
        container hub socket is keyed on it.  Both are mandatory:
        terok-shield removed the global-hub fallback in
        ``feat/per-container-supervisor``.
        """
        if self.bypass:
            return
        self.shield.up(container, container_id)

    def down(self, container: str, container_id: str, *, allow_all: bool = False) -> None:
        """Set shield to bypass mode (allow egress) for a running container.

        *container* / *container_id* — see
        [`up`][terok_sandbox.integrations.shield.ShieldManager.up].  When
        *allow_all* is True, also permits private-range (RFC 1918)
        traffic.
        """
        if self.bypass:
            return
        self.shield.down(container, container_id, allow_all=allow_all)

    # ── Non-bypassable operations ───────────────────────

    def quarantine(self, container: str) -> None:
        """Total network blackout — drop all traffic, log dropped traffic.

        Ignores ``shield_bypass`` because panic overrides every safety bypass.
        """
        self.shield.quarantine(container)

    def state(self, container: str) -> ShieldState:
        """Return the live shield state for a running container.

        Queries actual nft state even when bypass is set, because
        containers started *before* bypass was enabled may still have
        active rules.
        """
        return self.shield.state(container)

    # ── Configuration probes ────────────────────────────

    def status(self) -> dict:
        """Return shield status dict from the sandbox configuration.

        Reads only the sandbox configuration — does not instantiate
        the underlying Shield, so callers that only want
        configuration-level shape don't pay the Shield wire-up cost.
        """
        result: dict = {
            "mode": "hook",
            "profiles": list(self._cfg.shield_profiles),
            "audit_enabled": self._cfg.shield_audit,
        }
        if self.bypass:
            result["bypass_firewall_no_protection"] = True
        return result

    def check_environment(self) -> EnvironmentCheck:
        """Check the podman environment for shield compatibility.

        Returns a synthetic [`EnvironmentCheck`][terok_shield.EnvironmentCheck]
        with bypass info when the dangerous bypass override is active.
        """
        if self.bypass:
            return EnvironmentCheck(
                ok=False,
                health="bypass",
                issues=["bypass_firewall_no_protection is set — egress firewall disabled"],
            )
        return self.shield.check_environment()

    # ── Operator-facing session helpers ─────────────────

    def interactive_session(self, container: str) -> None:
        """Run the terminal clearance fallback for this task's shield.

        Thin wrapper that spares callers from reaching into
        [`terok_shield.simple_clearance`][terok_shield.simple_clearance]
        and rebuilding the ``state_dir`` themselves.  Refuses to run
        when the D-Bus clearance hub is already handling the session.
        """
        from terok_shield.simple_clearance import run_simple_clearance

        run_simple_clearance(self.state_dir, container)

    def watch_session(self, container: str) -> None:
        """Stream shield blocked-access events for this task as JSON lines.

        Thin wrapper that spares callers from reaching into
        [`terok_shield.watch`][terok_shield.watch] and rebuilding the
        ``state_dir`` themselves.
        """
        from terok_shield.watch import run_watch

        run_watch(self.state_dir, container)


# ── Host-wide OCI hooks ──────────────────────────────────


class ShieldHooks:
    """Host-wide OCI hooks installer — no task context.

    Thin pass-through to terok-shield's
    [`HooksInstaller`][terok_shield.HooksInstaller].  Kept as a class
    so the sandbox setup aggregator can swap it out in tests without
    poking around terok-shield internals.
    """

    @staticmethod
    def install() -> None:
        """Install global OCI hooks for shield egress firewalling.

        Global hooks are required on all podman versions to survive
        container stop/start cycles (terok-shield#122).  Single
        layout: scripts, ballast, and JSON descriptors all land in
        ``namespace_state_dir("shield") / "hooks"``;
        ``containers.conf`` is patched to register that path.
        """
        HooksInstaller().install()

    @staticmethod
    def uninstall() -> None:
        """Remove the global OCI hooks [`install`][terok_sandbox.integrations.shield.ShieldHooks.install] writes.

        Idempotent — missing files are tolerated.
        """
        HooksInstaller().uninstall()


# ── Bypass-aware environment probe (no task context) ────


def check_environment(cfg: SandboxConfig | None = None) -> EnvironmentCheck:
    """Probe the podman environment with no task context.

    Returns a synthetic [`EnvironmentCheck`][terok_shield.EnvironmentCheck]
    when ``shield_bypass`` is set; otherwise constructs a throwaway
    [`ShieldManager`][terok_sandbox.integrations.shield.ShieldManager]
    bound to a temp directory and delegates to its
    [`check_environment`][terok_sandbox.integrations.shield.ShieldManager.check_environment].
    Kept as a free function because the setup CLI runs before any
    task directory exists.
    """
    with tempfile.TemporaryDirectory() as tmp:
        return ShieldManager(Path(tmp), cfg).check_environment()
