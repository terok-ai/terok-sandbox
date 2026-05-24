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
  for the actual file writes — sandbox no longer carries a private
  ``_HOOK_FILES`` mirror of the on-disk install layout.
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
    ) -> None:
        """Bind the manager to a task directory and shield configuration.

        *runtime* selects the container runtime category — ``DEFAULT``
        for crun/runc/youki (dnsmasq on netns ``127.0.0.1``), ``KRUN``
        for the libkrun microVM path (dnsmasq on a link-local address
        the guest can reach via passt).  Callers that drive the launch
        path map their runtime string (``RunSpec.runtime``) to the
        enum.
        """
        self._task_dir = task_dir
        self._cfg = cfg or SandboxConfig()
        self._runtime = runtime

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
        # Socket-mode transports emit no loopback traffic; filter ``None`` so
        # the nftables rule generator only sees ports that actually exist.
        loopback = tuple(
            p
            for p in (resolved.gate_port, resolved.token_broker_port, resolved.ssh_signer_port)
            if p is not None
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

    def pre_start(self, container: str) -> list[str]:
        """Return extra ``podman run`` args for egress firewalling.

        Returns an empty list (no firewall args) when the dangerous
        ``bypass_firewall_no_protection`` override is active.

        Raises [`SystemExit`][SystemExit] with setup instructions when
        the podman environment requires one-time hook installation.
        """
        if self.bypass:
            warnings.warn(_BYPASS_WARNING, stacklevel=2)
            return []
        try:
            return self.shield.pre_start(container)
        except ShieldNeedsSetup as exc:
            raise SystemExit(str(exc)) from None

    def up(self, container: str) -> None:
        """Set shield to deny-all mode for a running container."""
        if self.bypass:
            return
        self.shield.up(container)

    def down(self, container: str, *, allow_all: bool = False) -> None:
        """Set shield to bypass mode (allow egress) for a running container.

        When *allow_all* is True, also permits private-range (RFC 1918) traffic.
        """
        if self.bypass:
            return
        self.shield.down(container, allow_all=allow_all)

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

    Wraps terok-shield's [`HooksInstaller`][terok_shield.HooksInstaller]
    in the root/user dual-scope flag pattern the sandbox setup CLI
    exposes: a single ``ShieldHooks.install(root=…, user=…)`` call
    can target one or both scopes, mirroring the symmetric
    ``ShieldHooks.uninstall`` path.
    """

    @staticmethod
    def install(*, root: bool = False, user: bool = False) -> None:
        """Install global OCI hooks for shield egress firewalling.

        Global hooks are required on all podman versions to survive
        container stop/start cycles (terok-shield#122).  At least one
        of *root* or *user* must be true; passing both installs into
        both scopes so callers that want system-wide and per-user
        coverage can do it in a single call.

        Raises [`ValueError`][ValueError] when neither flag is true.
        """
        if not root and not user:
            raise ValueError("ShieldHooks.install requires either root=True or user=True")
        if user:
            HooksInstaller.user().install()
        if root:
            HooksInstaller.system().install()

    @staticmethod
    def uninstall(*, root: bool = False, user: bool = False) -> None:
        """Remove the global OCI hooks [`install`][terok_sandbox.integrations.shield.ShieldHooks.install] writes.

        At least one of *root* or *user* must be true; passing both is
        valid and removes hooks from both scopes.  Missing files are
        tolerated so the uninstaller is idempotent.
        """
        if not root and not user:
            raise ValueError("ShieldHooks.uninstall requires either root=True or user=True")
        if user:
            HooksInstaller.user().uninstall()
        if root:
            HooksInstaller.system().uninstall()


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
