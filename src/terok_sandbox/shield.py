# SPDX-FileCopyrightText: 2025 Jiri Vyskocil
# SPDX-FileCopyrightText: 2026 Jiri Vyskocil
# SPDX-License-Identifier: Apache-2.0

"""Adapter for terok-shield egress firewall.

Creates per-task :class:`Shield` instances from the sandbox configuration.
Each task gets its own ``state_dir`` under ``{task_dir}/shield/``.
"""

import tempfile
import warnings
from pathlib import Path

from terok_shield import (
    USER_HOOKS_DIR,
    EnvironmentCheck,  # noqa: F401 — re-exported
    NftNotFoundError,  # noqa: F401 — re-exported
    Shield,
    ShieldConfig,
    ShieldMode,
    ShieldNeedsSetup,  # noqa: F401 — re-exported
    ShieldState,  # noqa: F401 — re-exported
    ensure_containers_conf_hooks_dir,
    setup_global_hooks,
    system_hooks_dir,
)

from .config import SandboxConfig

# DANGEROUS TRANSITIONAL OVERRIDE — will be removed once terok-shield
# supports all target podman versions (see terok-shield#71, #101).
_BYPASS_WARNING = (
    "WARNING: shield.bypass_firewall_no_protection is set — "
    "the egress firewall is DISABLED.  Containers have unrestricted "
    "network access.  Remove this setting once your podman version "
    "is compatible with terok-shield."
)


def _cfg(cfg: SandboxConfig | None = None) -> SandboxConfig:
    """Return *cfg* or a default :class:`SandboxConfig`."""
    return cfg or SandboxConfig()


def make_shield(task_dir: Path, cfg: SandboxConfig | None = None) -> Shield:
    """Construct a per-task :class:`Shield` from sandbox configuration.

    Builds a :class:`ShieldConfig` with ``state_dir`` scoped to *task_dir*.
    """
    c = _cfg(cfg)
    # Socket-mode transports emit no loopback traffic; filter ``None`` so
    # the nftables rule generator only sees ports that actually exist.
    loopback = tuple(
        p for p in (c.gate_port, c.token_broker_port, c.ssh_signer_port) if p is not None
    )
    config = ShieldConfig(
        state_dir=task_dir / "shield",
        mode=ShieldMode.HOOK,
        default_profiles=c.shield_profiles,
        loopback_ports=loopback,
        audit_enabled=c.shield_audit,
        profiles_dir=c.shield_profiles_dir,
    )
    return Shield(config)


def pre_start(container: str, task_dir: Path, cfg: SandboxConfig | None = None) -> list[str]:
    """Return extra ``podman run`` args for egress firewalling.

    Returns an empty list (no firewall args) when the dangerous
    ``bypass_firewall_no_protection`` override is active.

    Raises :class:`SystemExit` with setup instructions when the
    podman environment requires one-time hook installation.
    """
    if _cfg(cfg).shield_bypass:
        warnings.warn(_BYPASS_WARNING, stacklevel=2)
        return []
    try:
        return make_shield(task_dir, cfg).pre_start(container)
    except ShieldNeedsSetup as exc:
        raise SystemExit(str(exc)) from None


def down(
    container: str, task_dir: Path, *, allow_all: bool = False, cfg: SandboxConfig | None = None
) -> None:
    """Set shield to bypass mode (allow egress) for a running container.

    When *allow_all* is True, also permits private-range (RFC 1918) traffic.
    """
    if _cfg(cfg).shield_bypass:
        return
    make_shield(task_dir, cfg).down(container, allow_all=allow_all)


def up(container: str, task_dir: Path, cfg: SandboxConfig | None = None) -> None:
    """Set shield to deny-all mode for a running container."""
    if _cfg(cfg).shield_bypass:
        return
    make_shield(task_dir, cfg).up(container)


def block(container: str, task_dir: Path, cfg: SandboxConfig | None = None) -> None:
    """Total network blackout — drop all traffic, log for forensics.

    Unlike :func:`up` and :func:`down`, this ignores ``shield_bypass``
    because panic overrides all safety bypasses.
    """
    make_shield(task_dir, cfg).block(container)


def state(container: str, task_dir: Path, cfg: SandboxConfig | None = None) -> ShieldState:
    """Return the live shield state for a running container.

    Queries actual nft state even when bypass is set, because containers
    started *before* bypass was enabled may still have active rules.
    """
    return make_shield(task_dir, cfg).state(container)


def status(cfg: SandboxConfig | None = None) -> dict:
    """Return shield status dict from the sandbox configuration."""
    c = _cfg(cfg)
    result: dict = {
        "mode": "hook",
        "profiles": list(c.shield_profiles),
        "audit_enabled": c.shield_audit,
    }
    if c.shield_bypass:
        result["bypass_firewall_no_protection"] = True
    return result


def check_environment(cfg: SandboxConfig | None = None) -> EnvironmentCheck:
    """Check the podman environment for shield compatibility.

    Returns a synthetic :class:`EnvironmentCheck` with bypass info when the
    dangerous bypass override is active.
    """
    if _cfg(cfg).shield_bypass:
        return EnvironmentCheck(
            ok=False,
            health="bypass",
            issues=["bypass_firewall_no_protection is set — egress firewall disabled"],
        )
    with tempfile.TemporaryDirectory() as tmp:
        return make_shield(Path(tmp), cfg).check_environment()


def shield_interactive_session(
    container: str,
    task_dir: Path,
    *,
    raw: bool = False,
    cfg: SandboxConfig | None = None,
) -> None:
    """Run an interactive verdict session for a task's shield.

    Thin wrapper that spares callers from reaching into
    :mod:`terok_shield.cli.interactive` and rebuilding the
    ``state_dir`` themselves.
    """
    from terok_shield.cli.interactive import run_interactive

    run_interactive(make_shield(task_dir, cfg).config.state_dir, container, raw=raw)


def shield_watch_session(
    container: str,
    task_dir: Path,
    cfg: SandboxConfig | None = None,
) -> None:
    """Stream shield blocked-access events for a task as JSON lines.

    Thin wrapper that spares callers from reaching into
    :mod:`terok_shield.cli.watch` and rebuilding the ``state_dir``
    themselves.
    """
    from terok_shield.cli.watch import run_watch

    run_watch(make_shield(task_dir, cfg).config.state_dir, container)


def run_setup(*, root: bool = False, user: bool = False) -> None:
    """Install global OCI hooks for shield egress firewalling.

    Global hooks are required on all podman versions to survive
    container stop/start cycles (terok-shield#122).

    Raises :class:`ValueError` when neither ``root`` nor ``user`` is true.
    The CLI layer (``_handle_shield_setup`` in :mod:`.commands`) maps this
    to a ``SystemExit`` with actionable ``shield install-hooks …`` hints;
    the library stays UX-agnostic.
    """
    if not root and not user:
        raise ValueError("run_setup requires either root=True or user=True")
    setup_hooks_direct(root=root)


def setup_hooks_direct(*, root: bool = False) -> None:
    """Install global hooks via the terok-shield Python API (no subprocess).

    Suitable for TUI callers that need direct control.  Installs hooks
    to the system directory (with sudo) when *root* is True, otherwise
    to the user directory.
    """
    if root:
        target = system_hooks_dir()
        setup_global_hooks(target, use_sudo=True)
    else:
        target = Path(USER_HOOKS_DIR).expanduser()
        setup_global_hooks(target)
        ensure_containers_conf_hooks_dir(target)
