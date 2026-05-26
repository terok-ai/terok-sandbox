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
from pathlib import Path

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
from .integrations.shield import BinaryCheck

_HOST_BINARIES: tuple[str, ...] = ("podman", "git", "ssh-keygen")


# ── Prereq reporting (host binaries, firewall binaries, SELinux) ─────


def run_prereq_report(cfg: SandboxConfig) -> SelinuxCheckResult:
    """Print host prerequisites and return the SELinux check result.

    The result lets the caller decide whether to fail the setup or
    re-surface the install hint at the end of output — sandbox#854's
    fix for the install command getting buried mid-output.  Purely
    informational for the binary checks; never blocks on those.
    ``cfg.experimental`` gates the krun-only probes (currently ``ip``).
    """
    print("Prerequisites:")
    _report_host_binaries()
    _report_firewall_binaries()
    if cfg.experimental:
        _report_krun_binaries()
    return _report_selinux(cfg)


def _report_host_binaries() -> None:
    for name in _HOST_BINARIES:
        with _stage_line(name) as s:
            path = shutil.which(name)
            if path:
                s.ok(path)
            else:
                s.missing("not on PATH")


def _report_binary_checks(probe: Callable[[], Iterable[BinaryCheck]]) -> None:
    """Render one stage line per [`BinaryCheck`][terok_shield.BinaryCheck] *probe* returns.

    *probe* is one of shield's prereq probes (``check_firewall_binaries``,
    ``check_krun_binaries``); both are re-exported from
    [`terok_sandbox.integrations.shield`][terok_sandbox.integrations.shield] so the cross-package
    boundary contract holds.
    """
    for check in probe():
        with _stage_line(check.name) as s:
            if check.ok:
                s.ok(check.path)
            else:
                s.missing(check.purpose)


def _report_firewall_binaries() -> None:
    from .integrations.shield import check_firewall_binaries

    _report_binary_checks(check_firewall_binaries)


def _report_krun_binaries() -> None:
    from .integrations.shield import check_krun_binaries

    _report_binary_checks(check_krun_binaries)


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


def run_supervisor_install_phase() -> bool:
    """Install the OCI supervisor hook + wrapper under ``state_root()``.

    Lays down (with ``state_root()`` resolved from ``paths.root`` —
    the operator's single configured root):

    * ``<state_root>/hooks/supervisor_hook.py`` + ``_supervisor_state.py``
      — the OCI hook entrypoint and its stdlib-only ballast.
    * ``<state_root>/hooks/terok-sandbox-supervisor-<stage>.json`` — one
      OCI hook descriptor per stage (createRuntime + poststop).
      ``containers.conf`` is patched at install time to list
      ``state_root() / "hooks"`` in ``hooks_dir`` so podman scans the
      canonical terok-owned directory.
    * ``<state_root>/supervisor_wrapper.py`` — the restart-loop the
      hook spawns, with the ``terok-sandbox`` argv baked in at
      install time.

    Idempotent: re-running overwrites the installed files with the
    current package's copies.  Soft-fails on a missing
    ``terok-sandbox`` entry point (degraded install — operator hasn't
    sourced the venv yet).
    """
    from .supervisor.install import install_supervisor_hooks

    with _stage_line("Supervisor hooks") as s:
        try:
            install_supervisor_hooks()
        except Exception as exc:  # noqa: BLE001 — aggregator uniformity
            s.fail(str(exc))
            return False
        s.ok("installed (OCI hook + wrapper)")
        return True


def run_supervisor_uninstall_phase() -> bool:
    """Remove every file [`run_supervisor_install_phase`][terok_sandbox._setup.run_supervisor_install_phase] would write.

    Idempotent — missing files are tolerated.  Leaves any per-
    container PID files / log files alone; the operator can sweep
    those manually if a wrapper crashed in a way that left state
    behind (the wrapper's PID file is unlinked at poststop in the
    happy path).
    """
    from .supervisor.install import uninstall_supervisor_hooks

    with _stage_line("Supervisor hooks") as s:
        try:
            uninstall_supervisor_hooks()
        except Exception as exc:  # noqa: BLE001
            s.fail(str(exc))
            return False
        s.ok("removed")
        return True


def run_shield_install_phase() -> bool:
    """Install shield OCI hooks into the canonical terok-owned dir."""
    from .integrations.shield import ShieldHooks, check_environment

    with _stage_line("Shield hooks") as s:
        try:
            ShieldHooks.install()
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


# ── Service uninstall phases ──────────────────────────────────────────


def run_shield_uninstall_phase() -> bool:
    """Remove shield OCI hooks from the canonical terok-owned dir."""
    from .integrations.shield import ShieldHooks

    with _stage_line("Shield hooks") as s:
        try:
            ShieldHooks.uninstall()
        except Exception as exc:  # noqa: BLE001 — aggregator uniform error surface
            s.fail(str(exc))
            return False
        s.ok("removed")
        return True


#: Pre-supervisor systemd user units the cleanup phase sweeps.
_LEGACY_SYSTEMD_UNITS: tuple[str, ...] = (
    "terok-clearance-hub.service",
    "terok-clearance-verdict.service",
    "terok-clearance-notifier.service",
    "terok-vault.service",
    "terok-vault.socket",
    "terok-vault-socket.service",
    "terok-gate.socket",
    "terok-gate@.service",
    "terok-gate-socket.service",
)


def run_legacy_install_cleanup_phase() -> bool:
    """Sweep systemd units / sockets / install paths left by pre-supervisor versions.

    One-way cleanup.  Idempotent — every step soft-fails so a missing
    ``systemctl``, an absent unit, or a stale socket cannot abort the
    rest of the sweep.  Runs once during ``terok-sandbox setup``; the
    per-container supervisor lifecycle never invokes it.

    Sweeps:

    * the legacy clearance trio (``terok-clearance-hub.service``,
      ``terok-clearance-verdict.service``,
      ``terok-clearance-notifier.service``) from the W5 layout;
    * the legacy vault systemd units
      (``terok-vault.service`` / ``terok-vault.socket`` /
      ``terok-vault-socket.service``);
    * the legacy gate systemd units
      (``terok-gate.socket`` / ``terok-gate@.service`` /
      ``terok-gate-socket.service``) now that the gate lives in the
      per-container supervisor;
    * any ``terok-clearance-*.service`` / ``terok-vault-*`` /
      ``terok-gate*`` files lingering in the user's systemd unit
      directory (catches renamed variants from prior alphas);
    * the legacy global shield-events socket
      (``$XDG_RUNTIME_DIR/terok-shield-events.sock``) from the
      single-hub-socket era.

    Operators upgrading from a pre-supervisor install lose access to old
    tasks (per the hard rule: no state preservation across the
    refactor); the cleanup is purely about removing the *host-side*
    machinery that would fight a fresh setup for sockets / unit names.
    """
    with _stage_line("Legacy install cleanup") as s:
        _disable_legacy_units(_LEGACY_SYSTEMD_UNITS)
        _sweep_legacy_unit_files()
        _systemctl.run_best_effort("daemon-reload")
        _unlink_legacy_shield_events_socket()
        _unlink_legacy_runtime_sockets()
        _unlink_legacy_xdg_data_files()
        _unlink_legacy_shield_global_hooks()
        s.ok("swept (legacy units + sockets, if any)")
        return True


def _disable_legacy_units(units: tuple[str, ...]) -> None:
    """Disable + stop *units* via ``systemctl --user``, swallowing every error."""
    for unit in units:
        _systemctl.run_best_effort("disable", "--now", unit)


def _sweep_legacy_unit_files() -> None:
    """Unlink stray ``terok-clearance-*`` / ``terok-vault-*`` / ``terok-gate*`` user unit files.

    Catches renamed variants from prior alphas that the explicit
    [`_LEGACY_SYSTEMD_UNITS`][terok_sandbox._setup._LEGACY_SYSTEMD_UNITS]
    list doesn't enumerate.  Soft-fails on a missing systemd user unit
    dir (no XDG config home, fresh tmpfs, etc.) — the legacy install
    we're cleaning after necessarily lived in *some* unit dir, so a
    missing one means there's nothing to sweep.
    """
    try:
        from ._util import systemd_user_unit_dir
    except ImportError:
        return
    try:
        unit_dir = systemd_user_unit_dir()
    except SystemExit:
        return
    if not unit_dir.is_dir():
        return
    for pattern in (
        "terok-clearance-*.service",
        "terok-vault.service",
        "terok-vault.socket",
        "terok-vault-*.service",
        "terok-vault-*.socket",
        "terok-gate*.service",
        "terok-gate*.socket",
    ):
        for unit_file in unit_dir.glob(pattern):
            # Best-effort disable in case the unit is still active; then unlink.
            _systemctl.run_best_effort("disable", "--now", unit_file.name)
            try:
                unit_file.unlink(missing_ok=True)
            except OSError:
                pass


def _unlink_legacy_shield_events_socket() -> None:
    """Remove the pre-supervisor global shield-events socket if present.

    Pre-supervisor shield emitted events into a single host-global Unix
    socket; the per-container supervisor uses a per-container hub
    socket instead.  Soft-fails on missing ``$XDG_RUNTIME_DIR`` or a
    socket that has already been unlinked.
    """
    import os
    from pathlib import Path

    runtime_dir = os.environ.get("XDG_RUNTIME_DIR")
    if not runtime_dir:
        return
    sock = Path(runtime_dir) / "terok-shield-events.sock"
    try:
        sock.unlink(missing_ok=True)
    except OSError:
        pass


def _unlink_legacy_runtime_sockets() -> None:
    """Remove the pre-supervisor host-global runtime sockets.

    The previous architecture ran the vault broker, gate server, and
    SSH signer as long-lived host daemons that bound named sockets at
    ``<runtime_root>/{vault,gate-server,ssh-agent}.sock`` plus
    ``<runtime_root>/vault.passphrase``.  Per-container supervisors
    bind their own per-container paths now, so the leftover global
    socket files just confuse containers that mount the runtime dir
    wholesale (a stale ``vault.sock`` looks reachable but its AF_UNIX
    peer is gone).  Each ``unlink`` soft-fails so a partially-cleaned
    install still ends up clean.
    """
    from .paths import runtime_root

    try:
        rt = runtime_root()
    except OSError:
        return
    # ``vault.passphrase`` is intentionally NOT swept — it's a live
    # session-tier credential; wiping it would re-lock the vault on
    # every ``terok setup`` re-run.
    # Pre-supervisor host-global socket basenames.  The current
    # supervisor binds these names PER CONTAINER under
    # ``<rt>/run/<container>/`` instead, so any remaining file at
    # ``<rt>/<name>`` is left over from an older install — the gate
    # included now that it, too, is per-container.
    for name in ("vault.sock", "ssh-agent.sock", "gate-server.sock"):
        try:
            (rt / name).unlink(missing_ok=True)
        except OSError:
            pass


def _unlink_legacy_xdg_data_files() -> None:
    """Remove pre-paths.root shield script copies under ``$XDG_DATA_HOME``.

    Master-branch terok-shield wrote ``nflog-reader.py`` to
    ``$XDG_DATA_HOME/terok/shield/`` without honouring ``paths.root``.
    The current installer puts it under
    [`namespace_state_dir("shield")`][terok_util.paths.namespace_state_dir];
    this sweep removes the orphaned copy so a single
    ``terok-sandbox setup`` converges on the new layout.

    Only the supervisor's specific files are unlinked — anything else
    under those XDG dirs is left intact.
    """
    import os
    from pathlib import Path

    data_home = os.environ.get("XDG_DATA_HOME") or os.path.expanduser("~/.local/share")
    legacy_shield_root = Path(data_home) / "terok" / "shield"
    if legacy_shield_root.is_dir():
        for stale in ("nflog-reader.py",):
            try:
                (legacy_shield_root / stale).unlink(missing_ok=True)
            except OSError:
                pass
        with contextlib.suppress(OSError):
            legacy_shield_root.rmdir()
        parent = legacy_shield_root.parent
        if parent.name == "terok":
            with contextlib.suppress(OSError):
                parent.rmdir()


def _unlink_legacy_shield_global_hooks() -> None:
    """Remove the master-branch shield install at ``~/.local/share/containers/oci/hooks.d/``.

    Before the single-terok-owned-dir refactor, the user-scope shield
    install dropped both scripts and JSON descriptors into
    ``~/.local/share/containers/oci/hooks.d/`` (podman's default
    rootless scan path).  The current installer writes everything into
    ``namespace_state_dir("shield") / "hooks"`` and patches
    ``containers.conf`` to point podman there, so the old files become
    a stale duplicate set that podman would happily fire alongside the
    new ones.  This sweep removes them so re-running setup converges
    on the canonical layout.

    Operator-owned siblings (``.backup`` files, ``__pycache__/``) are
    deliberately left in place — they're not ours to delete.
    """
    legacy_dir = Path.home() / ".local" / "share" / "containers" / "oci" / "hooks.d"
    stale = (
        # Role scripts + shared ballast.
        "_oci_state.py",
        "terok-shield-hook",
        "terok-shield-bridge-hook",
        # JSON descriptors.
        "terok-shield-createRuntime.json",
        "terok-shield-poststop.json",
        "terok-shield-bridge-createRuntime.json",
        "terok-shield-bridge-poststop.json",
    )
    for name in stale:
        with contextlib.suppress(OSError):
            (legacy_dir / name).unlink(missing_ok=True)
