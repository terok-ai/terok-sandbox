# SPDX-FileCopyrightText: 2026 Jiri Vyskocil
# SPDX-License-Identifier: Apache-2.0

"""Sandbox-wide setup orchestration — the phases ``_handle_sandbox_setup`` runs.

Each phase is self-contained and idempotent:

* Prereq probes are report-only.  A missing ``nft`` or ``podman`` later
  fails the relevant service with a clearer message; reporting here
  lets the operator spot the root cause before scrolling past install
  noise.
* Service install phases do the full stop → uninstall → install →
  verify cycle so a re-run after ``pipx install terok-sandbox``
  guarantees the running unit picks up the new code, not just the
  rewritten on-disk unit file.
* The clearance phase is optional — headless servers that skip the
  desktop bridge still get a working shield+vault+gate install.

Printed output is plain-text stage lines; higher-level frontends that
want ANSI colour can wrap the aggregator call in their own renderer.
Kept internal (underscore-prefixed module) because every public
entry point goes through :func:`commands._handle_sandbox_setup`.
"""

from __future__ import annotations

import contextlib
import shutil
import sys
from collections.abc import Callable, Iterable
from enum import StrEnum

from ._util import _systemctl
from ._util._selinux import (
    SelinuxStatus,
    check_status as check_selinux_status,
    install_command as selinux_install_command,
)
from .config import SandboxConfig, services_mode

# Widest label is "Clearance notifier" (18); recompute the gutter if
# a new phase ships with a longer label so the status markers align.
_STAGE_WIDTH = 20

_HOST_BINARIES: tuple[str, ...] = ("podman", "git", "ssh-keygen")


class Marker(StrEnum):
    """Status tokens rendered in each stage line.

    Typed so a typo (`"Warn"` vs `"WARN"`) is a load-time error, not a
    silent drift in output the test assertions keep passing against.
    """

    OK = "ok"
    WARN = "WARN"
    FAIL = "FAIL"
    MISSING = "MISSING"
    SKIP = "skip"


# ── Prereq reporting (host binaries, firewall binaries, SELinux) ─────


def run_prereq_report(cfg: SandboxConfig) -> None:
    """Print host prerequisites.  Never blocks — purely informational."""
    print("Prerequisites:")
    _report_host_binaries()
    _report_firewall_binaries()
    _report_selinux(cfg)


def _report_host_binaries() -> None:
    for name in _HOST_BINARIES:
        path = shutil.which(name)
        if path:
            _stage(name, Marker.OK, path)
        else:
            _stage(name, Marker.MISSING, "not on PATH")


def _report_firewall_binaries() -> None:
    """Delegate the nft / dnsmasq / dig probes to terok-shield's own list."""
    from terok_shield import check_firewall_binaries

    for check in check_firewall_binaries():
        if check.ok:
            _stage(check.name, Marker.OK, check.path)
        else:
            _stage(check.name, Marker.MISSING, check.purpose)


def _report_selinux(cfg: SandboxConfig) -> None:
    """Print SELinux policy status; stay silent when the host doesn't need one."""
    result = check_selinux_status(services_mode=services_mode())
    match result.status:
        case SelinuxStatus.NOT_APPLICABLE_TCP_MODE | SelinuxStatus.NOT_APPLICABLE_PERMISSIVE:
            return
        case SelinuxStatus.OK:
            _stage("SELinux policy", Marker.OK, "installed")
        case SelinuxStatus.POLICY_MISSING:
            _stage("SELinux policy", Marker.MISSING, f"install: {selinux_install_command()}")
        case SelinuxStatus.LIBSELINUX_MISSING:
            _stage("SELinux policy", Marker.MISSING, "libselinux.so.1 not loadable")


# ── Service install phases ────────────────────────────────────────────


def run_shield_install_phase(*, root: bool) -> bool:
    """Install shield OCI hooks — per-user or system-wide depending on *root*."""
    from .shield import check_environment, run_setup

    try:
        run_setup(root=root, user=not root)
    except Exception as exc:  # noqa: BLE001 — aggregator reports all failures uniformly
        _stage("Shield hooks", Marker.FAIL, str(exc))
        return False

    env = check_environment()
    if env.health == "ok":
        _stage("Shield hooks", Marker.OK, "active")
        return True
    if env.health == "bypass":
        _stage("Shield hooks", Marker.WARN, "bypass_firewall_no_protection is active")
        return True
    _stage("Shield hooks", Marker.FAIL, f"installed but health: {env.health}")
    return False


def run_vault_install_phase(cfg: SandboxConfig) -> bool:
    """Clean reinstall of the vault systemd units; verify reachability."""
    from .vault.lifecycle import VaultManager, VaultUnreachableError

    return _reinstall_systemd_service(
        label="Vault",
        mgr=VaultManager(cfg),
        reachable_exc=(VaultUnreachableError, SystemExit),
    )


def run_gate_install_phase(cfg: SandboxConfig) -> bool:
    """Clean reinstall of the gate systemd units; verify reachability."""
    from .gate.lifecycle import GateServerManager

    mgr = GateServerManager(cfg)
    if not mgr.is_systemd_available():
        _stage("Gate server", Marker.WARN, "systemd unavailable, skipping")
        return True
    return _reinstall_systemd_service(label="Gate server", mgr=mgr)


def run_clearance_install_phase() -> bool:
    """Install the clearance hub + verdict + notifier units.

    Soft-skip when ``terok_clearance`` isn't importable — headless
    servers don't need the desktop bridge, and the sandbox shield /
    vault / gate stack is perfectly functional without it.
    """
    try:
        from terok_clearance.runtime.installer import (
            HUB_UNIT_NAME,
            NOTIFIER_UNIT_NAME,
            VERDICT_UNIT_NAME,
            install_notifier_service,
            install_service,
        )
    except ImportError:
        _stage("Clearance", Marker.SKIP, "terok_clearance not installed")
        return True

    # Avoid ``shutil.which("terok-clearance-hub")``: a hostile PATH
    # could otherwise poison the ExecStart= baked into the persistent
    # user unit.  ``sys.executable`` isn't resolved through PATH, so
    # the pipx venv's own python is the one the unit invokes.
    hub_ok = _install_clearance_unit_pair(
        label="Clearance hub",
        install_fn=lambda: install_service([sys.executable, "-m", "terok_clearance.cli.main"]),
        units_to_enable=(HUB_UNIT_NAME, VERDICT_UNIT_NAME),
    )
    # Notifier failure is non-fatal — the hub is the critical path;
    # the notifier only enriches desktop popups.  Return value is
    # discarded so a notifier glitch (e.g. missing session bus on a
    # remote SSH install) doesn't flip the aggregator's exit code.
    _install_clearance_unit_pair(
        label="Clearance notifier",
        install_fn=lambda: install_notifier_service(
            [sys.executable, "-m", "terok_clearance.notifier.app"]
        ),
        units_to_enable=(NOTIFIER_UNIT_NAME,),
    )
    return hub_ok


# ── Service uninstall phases ──────────────────────────────────────────


def run_shield_uninstall_phase(*, root: bool) -> bool:
    """Remove shield OCI hooks — per-user or system-wide depending on *root*."""
    from .shield import run_uninstall

    scope = "system" if root else "user"
    try:
        run_uninstall(root=root, user=not root)
    except Exception as exc:  # noqa: BLE001 — aggregator uniform error surface
        _stage("Shield hooks", Marker.FAIL, str(exc))
        return False
    _stage("Shield hooks", Marker.OK, f"removed ({scope})")
    return True


def run_vault_uninstall_phase(cfg: SandboxConfig) -> bool:
    """Remove vault systemd units; WARN-skip without a systemd user session."""
    from .vault.lifecycle import VaultManager

    mgr = VaultManager(cfg)
    if not mgr.is_systemd_available():
        _stage("Vault", Marker.WARN, "systemd unavailable, skipping")
        return True
    try:
        mgr.uninstall_systemd_units()
    except SystemExit as exc:
        _stage("Vault", Marker.FAIL, str(exc))
        return False
    except Exception as exc:  # noqa: BLE001
        _stage("Vault", Marker.FAIL, f"uninstall: {exc}")
        return False
    _stage("Vault", Marker.OK, "removed")
    return True


def run_gate_uninstall_phase(cfg: SandboxConfig) -> bool:
    """Stop any stray gate daemon, then remove systemd units."""
    from .gate.lifecycle import GateServerManager

    mgr = GateServerManager(cfg)
    try:
        if mgr.get_status().mode == "daemon":
            mgr.stop_daemon()
        if mgr.is_systemd_available():
            mgr.uninstall_systemd_units()
    except SystemExit as exc:
        _stage("Gate server", Marker.FAIL, str(exc))
        return False
    except Exception as exc:  # noqa: BLE001
        _stage("Gate server", Marker.FAIL, f"uninstall: {exc}")
        return False
    _stage("Gate server", Marker.OK, "removed")
    return True


def run_clearance_uninstall_phase() -> bool:
    """Tear down the clearance hub + verdict + notifier units.

    Mirrors :func:`run_clearance_install_phase` — soft-skips when
    ``terok_clearance`` isn't importable so a headless host's teardown
    stays a one-liner rather than a crash.
    """
    try:
        from terok_clearance.runtime.installer import (
            uninstall_notifier_service,
            uninstall_service,
        )
    except ImportError:
        _stage("Clearance", Marker.SKIP, "terok_clearance not installed")
        return True
    try:
        uninstall_notifier_service()
        uninstall_service()
    except Exception as exc:  # noqa: BLE001
        _stage("Clearance", Marker.FAIL, str(exc))
        return False
    _stage("Clearance", Marker.OK, "removed")
    return True


# ── Shared service-install skeleton ───────────────────────────────────


def _reinstall_systemd_service(
    *,
    label: str,
    mgr,  # noqa: ANN001 — duck-typed manager; no shared base class across vault/gate
    reachable_exc: tuple[type[BaseException], ...] = (SystemExit,),
) -> bool:
    """Run the full stop → uninstall → install → verify cycle for one service.

    Shared between vault and gate because both sandbox-managed systemd
    services follow the same lifecycle contract: a ``stop_daemon`` +
    ``uninstall_systemd_units`` pair (best-effort), an
    ``install_systemd_units()`` call (authoritative), and an
    ``ensure_reachable()`` verify (fails with *reachable_exc*).  Shield
    and clearance are different shapes so they stay inline.
    """
    _stop_and_uninstall(mgr.stop_daemon, mgr.uninstall_systemd_units)
    try:
        mgr.install_systemd_units()
    except SystemExit as exc:
        _stage(label, Marker.FAIL, str(exc))
        return False
    except Exception as exc:  # noqa: BLE001
        _stage(label, Marker.FAIL, f"install: {exc}")
        return False
    try:
        mgr.ensure_reachable()
    except reachable_exc as exc:
        _stage(label, Marker.FAIL, f"installed but NOT reachable: {exc}")
        return False
    status = mgr.get_status()
    _stage(
        label,
        Marker.OK,
        f"{status.mode or 'systemd'}, {status.transport or 'tcp'}, reachable",
    )
    return True


def _install_clearance_unit_pair(
    *, label: str, install_fn: Callable[[], object], units_to_enable: Iterable[str]
) -> bool:
    """Render the unit file(s), then enable + start each, reporting one stage line.

    Batches ``daemon-reload`` once at the top so installing the
    hub/verdict pair doesn't pay three sequential ``daemon-reload``
    round-trips when it should only need one.
    """
    try:
        install_fn()
        _systemctl.run_best_effort("daemon-reload")
        for unit in units_to_enable:
            _enable_and_restart_user_unit(unit)
    except Exception as exc:  # noqa: BLE001 — aggregator uniform error surface
        _stage(label, Marker.FAIL, str(exc))
        return False
    _stage(label, Marker.OK, "installed + enabled")
    return True


# ── Lifecycle helpers ─────────────────────────────────────────────────


def _stop_and_uninstall(stop: Callable[[], None], uninstall: Callable[[], None]) -> None:
    """Best-effort stop + uninstall; lets ``install_systemd_units`` start fresh."""
    with contextlib.suppress(Exception):
        stop()
    with contextlib.suppress(Exception):
        uninstall()


def _enable_and_restart_user_unit(unit: str) -> None:
    """``systemctl --user enable`` + ``restart`` for *unit*.

    ``restart`` (not ``enable --now``) matters for pipx-upgrade
    scenarios: after ``pipx install --force terok-sandbox``, the venv
    has fresh code but the running daemon holds the old ExecStart's
    python process.  Restarting guarantees the new code is loaded.
    Caller is responsible for ``daemon-reload`` — batch it once per
    install instead of once per unit so a three-unit clearance install
    pays one reload, not three.
    """
    for verb in ("enable", "restart"):
        _systemctl.run_best_effort(verb, unit)


# ── Stage-line primitive ──────────────────────────────────────────────


def _stage(label: str, marker: Marker, detail: str = "") -> None:
    """Write one ``'  <label>  <marker> (<detail>)'`` line."""
    suffix = f" ({detail})" if detail else ""
    print(f"  {label:<{_STAGE_WIDTH}} {marker}{suffix}")
