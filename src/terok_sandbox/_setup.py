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

Stage-line output routes through `terok_sandbox._stage` (re-exported
via the package's public surface) so frontends (terok, terok-executor)
that mix their own stage lines in the same log share one renderer and
one colour palette.  Kept internal (underscore-prefixed module) because
every public entry point goes through `commands._handle_sandbox_setup`.
"""

from __future__ import annotations

import contextlib
import shutil
from collections.abc import Callable, Iterable
from typing import Any

from ._exit_codes import EXIT_MANUAL_STEP_NEEDED
from ._stage import stage_line as _stage_line
from ._util import _systemctl
from ._util._selinux import (
    SelinuxCheckResult,
    SelinuxStatus,
    check_status as check_selinux_status,
    install_command as selinux_install_command,
)

# Re-export so existing callers ``from ._setup import EXIT_MANUAL_STEP_NEEDED``
# keep working without reaching for the new foundation module.
__all__ = ["EXIT_MANUAL_STEP_NEEDED"]
from .config import SandboxConfig

_HOST_BINARIES: tuple[str, ...] = ("podman", "git", "ssh-keygen")


# ── Prereq reporting (host binaries, firewall binaries, SELinux) ─────


def run_prereq_report(cfg: SandboxConfig) -> SelinuxCheckResult:
    """Print host prerequisites and return the SELinux check result.

    The result lets the caller decide whether to fail the setup or
    re-surface the install hint at the end of output — sandbox#854's
    fix for the install command getting buried mid-output.  Purely
    informational for the binary checks; never blocks on those.
    """
    print("Prerequisites:")
    _report_host_binaries()
    _report_firewall_binaries()
    return _report_selinux(cfg)


def _report_host_binaries() -> None:
    for name in _HOST_BINARIES:
        with _stage_line(name) as s:
            path = shutil.which(name)
            if path:
                s.ok(path)
            else:
                s.missing("not on PATH")


def _report_firewall_binaries() -> None:
    """Delegate the nft / dnsmasq / dig probes to terok-shield's own list."""
    # Top-level ``terok_shield`` namespace import (not ``.prereqs``) so the
    # test patch ``terok_shield.check_firewall_binaries`` still hits the
    # name we look up.  The lazy ``__getattr__`` returns ``object`` to
    # type-checkers — ``[operator]`` silences the resulting callable
    # warning.
    from terok_shield import check_firewall_binaries

    for check in check_firewall_binaries():  # type: ignore[operator]
        with _stage_line(check.name) as s:
            if check.ok:
                s.ok(check.path)
            else:
                s.missing(check.purpose)


def _report_selinux(cfg: SandboxConfig) -> SelinuxCheckResult:
    """Print SELinux policy status and return the check result.

    Stays silent when the host doesn't need a policy (TCP mode or
    SELinux disabled/permissive); the return value still carries the
    structured outcome so the caller can re-surface the install
    command at the end of setup output (where it's not scrolled out
    of view).
    """
    result = check_selinux_status(services_mode=cfg.services_mode)
    if result.status in (
        SelinuxStatus.NOT_APPLICABLE_TCP_MODE,
        SelinuxStatus.NOT_APPLICABLE_PERMISSIVE,
    ):
        return result
    with _stage_line("SELinux policy") as s:
        match result.status:
            case SelinuxStatus.OK:
                s.ok("installed")
            case SelinuxStatus.POLICY_MISSING:
                s.missing(f"install: {selinux_install_command()}")
            case SelinuxStatus.LIBSELINUX_MISSING:
                s.missing("libselinux.so.1 not loadable")
    return result


def print_selinux_install_hint(result: SelinuxCheckResult) -> None:
    """Print the SELinux install command + TCP-mode alternative at end of setup output.

    No-op when the SELinux state doesn't require operator action
    (``OK``, ``NOT_APPLICABLE_*``).  Renders the two alternatives on
    their own lines so the operator can copy-paste either without
    surrounding output bleeding in.

    Called *after* all install phases finish so the hint is the last
    thing the operator sees — sandbox#854's complaint was that the
    install command landed mid-output and scrolled out of view by the
    time the install banner printed at the bottom.
    """
    if result.status is not SelinuxStatus.POLICY_MISSING:
        return
    print()
    print("─ SELinux policy required ─────────────────────────────────────")
    print("Socket-transport services need the terok_socket_t policy to be")
    print("loaded; without it, containers can't reach the host sockets.")
    print()
    print("Install the policy (recommended):")
    print()
    print(f"  {selinux_install_command()}")
    print()
    print("Or switch to TCP mode (no SELinux policy needed):")
    print()
    print("  yq -yi '.services.mode = \"tcp\"' ~/.config/terok/config.yml")
    print("  terok-sandbox setup")
    print()


# ── Service install phases ────────────────────────────────────────────


def run_shield_install_phase(*, root: bool) -> bool:
    """Install shield OCI hooks — per-user or system-wide depending on *root*."""
    from .shield import check_environment, run_setup

    with _stage_line("Shield hooks") as s:
        try:
            run_setup(root=root, user=not root)
        except Exception as exc:  # noqa: BLE001 — aggregator reports all failures uniformly
            s.fail(str(exc))
            return False

        env = check_environment()
        if env.health == "ok":
            s.ok("active")
            return True
        if env.health == "bypass":
            s.warn("bypass_firewall_no_protection is active")
            return True
        s.fail(f"installed but health: {env.health}")
        return False


def run_vault_install_phase(cfg: SandboxConfig) -> bool:
    """Install the vault — systemd units when available, managed daemon otherwise.

    On a systemd host this is the full stop → uninstall → install → verify
    cycle.  On a host without ``systemctl --user`` (containers, OpenRC,
    sysvinit) the same vault binary is started as a managed daemon — same
    code, same transport, just a PID file instead of a unit.  The fallback
    works because [`VaultManager.ensure_reachable`][terok_sandbox.vault.daemon.lifecycle.VaultManager.ensure_reachable] already accepts
    ``mode="daemon"`` as a healthy state.
    """
    from .vault.daemon.lifecycle import VaultManager, VaultUnreachableError

    mgr = VaultManager(cfg)
    if mgr.is_systemd_available():
        return _reinstall_systemd_service(
            label="Vault",
            mgr=mgr,
            reachable_exc=(VaultUnreachableError, SystemExit),
        )
    return _start_managed_daemon(
        label="Vault",
        mgr=mgr,
        reachable_exc=(VaultUnreachableError, SystemExit),
    )


def run_gate_install_phase(cfg: SandboxConfig) -> bool:
    """Install the gate's systemd units; warn-skip on hosts without user systemd.

    Unlike vault, the gate's inetd-style architecture has no managed-daemon
    fallback — the systemd socket unit accepts each connection and spawns
    a fresh process; without systemd the same role would need an inetd
    shim or a permanent listen-loop that doesn't yet exist.  The skip is
    therefore intentional: callers (`terok-executor.preflight`,
    `terok sickbay`) check
    [`GateServerManager.get_status`][terok_sandbox.gate.lifecycle.GateServerManager.get_status] and degrade gracefully —
    a container without a gate is just a container without the git push
    channel, which is exactly as secure as one with the gate.
    """
    from .gate.lifecycle import GateServerManager

    mgr = GateServerManager(cfg)
    if not mgr.is_systemd_available():
        with _stage_line("Gate server") as s:
            s.warn(
                "systemd unavailable — git gate disabled; "
                "containers will run without the git push channel"
            )
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
        with _stage_line("Clearance") as s:
            s.skip("terok_clearance not installed")
        return True

    # ``install_service()`` / ``install_notifier_service()`` default
    # their own argv internally — clearance owns the knowledge of
    # which module serves its hub / notifier, so sandbox doesn't have
    # to spell ``[sys.executable, "-m", "terok_clearance.cli.main"]``
    # and leak that layout across the package boundary.
    hub_ok = _install_clearance_unit_pair(
        label="Clearance hub",
        install_fn=install_service,
        units_to_enable=(HUB_UNIT_NAME, VERDICT_UNIT_NAME),
    )
    # Notifier failure is non-fatal — the hub is the critical path;
    # the notifier only enriches desktop popups.  Return value is
    # discarded so a notifier glitch (e.g. missing session bus on a
    # remote SSH install) doesn't flip the aggregator's exit code.
    _install_clearance_unit_pair(
        label="Clearance notifier",
        install_fn=install_notifier_service,
        units_to_enable=(NOTIFIER_UNIT_NAME,),
    )
    return hub_ok


# ── Service uninstall phases ──────────────────────────────────────────


def run_shield_uninstall_phase(*, root: bool) -> bool:
    """Remove shield OCI hooks — per-user or system-wide depending on *root*."""
    from .shield import run_uninstall

    scope = "system" if root else "user"
    with _stage_line("Shield hooks") as s:
        try:
            run_uninstall(root=root, user=not root)
        except Exception as exc:  # noqa: BLE001 — aggregator uniform error surface
            s.fail(str(exc))
            return False
        s.ok(f"removed ({scope})")
        return True


def run_vault_uninstall_phase(cfg: SandboxConfig) -> bool:
    """Remove vault systemd units; WARN-skip without a systemd user session."""
    from .vault.daemon.lifecycle import VaultManager

    mgr = VaultManager(cfg)
    with _stage_line("Vault") as s:
        if not mgr.is_systemd_available():
            s.warn("systemd unavailable, skipping")
            return True
        try:
            mgr.uninstall_systemd_units()
        except SystemExit as exc:
            s.fail(str(exc))
            return False
        except Exception as exc:  # noqa: BLE001
            s.fail(f"uninstall: {exc}")
            return False
        s.ok("removed")
        return True


def run_gate_uninstall_phase(cfg: SandboxConfig) -> bool:
    """Stop any stray gate daemon, then remove systemd units."""
    from .gate.lifecycle import GateServerManager

    mgr = GateServerManager(cfg)
    with _stage_line("Gate server") as s:
        try:
            if mgr.get_status().mode == "daemon":
                mgr.stop_daemon()
            if mgr.is_systemd_available():
                mgr.uninstall_systemd_units()
        except SystemExit as exc:
            s.fail(str(exc))
            return False
        except Exception as exc:  # noqa: BLE001
            s.fail(f"uninstall: {exc}")
            return False
        s.ok("removed")
        return True


def run_clearance_uninstall_phase() -> bool:
    """Tear down the clearance hub + verdict + notifier units.

    Mirrors `run_clearance_install_phase` — soft-skips when
    ``terok_clearance`` isn't importable so a headless host's teardown
    stays a one-liner rather than a crash.
    """
    try:
        from terok_clearance.runtime.installer import (
            uninstall_notifier_service,
            uninstall_service,
        )
    except ImportError:
        with _stage_line("Clearance") as s:
            s.skip("terok_clearance not installed")
        return True
    with _stage_line("Clearance") as s:
        try:
            uninstall_notifier_service()
            uninstall_service()
        except Exception as exc:  # noqa: BLE001
            s.fail(str(exc))
            return False
        s.ok("removed")
        return True


# ── Shared service-install skeleton ───────────────────────────────────


def _reinstall_systemd_service(
    *,
    label: str,
    mgr: Any,  # duck-typed manager; no shared base class across vault/gate
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
    with _stage_line(label) as s:
        _stop_and_uninstall(mgr.stop_daemon, mgr.uninstall_systemd_units)
        try:
            mgr.install_systemd_units()
        except SystemExit as exc:
            s.fail(str(exc))
            return False
        except Exception as exc:  # noqa: BLE001
            s.fail(f"install: {exc}")
            return False
        try:
            mgr.ensure_reachable()
        except reachable_exc as exc:
            s.fail(f"installed but NOT reachable: {exc}")
            return False
        status = mgr.get_status()
        s.ok(f"{status.mode or 'systemd'}, {status.transport or 'tcp'}, reachable")
        return True


def _start_managed_daemon(
    *,
    label: str,
    mgr: Any,  # duck-typed manager exposing stop_daemon / start_daemon / ensure_reachable
    reachable_exc: tuple[type[BaseException], ...] = (SystemExit,),
) -> bool:
    """Stop any existing daemon, start a fresh one, then verify reachability.

    The no-systemd counterpart of [`_reinstall_systemd_service`][terok_sandbox._setup._reinstall_systemd_service].  Same contract
    minus the unit-file phase: ``stop_daemon`` (best-effort), then a hard
    ``start_daemon()`` and ``ensure_reachable()`` verify.  Status line
    ends with "(no systemd)" so the operator can tell at a glance which
    init path produced this service, without having to ``ps`` for it.
    """
    with _stage_line(label) as s:
        with contextlib.suppress(Exception):
            mgr.stop_daemon()
        try:
            mgr.start_daemon()
        except SystemExit as exc:
            s.fail(str(exc))
            return False
        except Exception as exc:  # noqa: BLE001
            s.fail(f"daemon start: {exc}")
            return False
        try:
            mgr.ensure_reachable()
        except reachable_exc as exc:
            s.fail(f"started but NOT reachable: {exc}")
            return False
        status = mgr.get_status()
        s.ok(f"{status.mode}, {status.transport or 'tcp'}, reachable (no systemd)")
        return True


def _install_clearance_unit_pair(
    *, label: str, install_fn: Callable[[], object], units_to_enable: Iterable[str]
) -> bool:
    """Render the unit file(s), then enable + start each, reporting one stage line.

    Batches ``daemon-reload`` once at the top so installing the
    hub/verdict pair doesn't pay three sequential ``daemon-reload``
    round-trips when it should only need one.
    """
    with _stage_line(label) as s:
        try:
            install_fn()
            _systemctl.run_best_effort("daemon-reload")
            for unit in units_to_enable:
                _enable_and_restart_user_unit(unit)
        except Exception as exc:  # noqa: BLE001 — aggregator uniform error surface
            s.fail(str(exc))
            return False
        s.ok("installed + enabled")
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
