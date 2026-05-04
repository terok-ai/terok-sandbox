# SPDX-FileCopyrightText: 2026 Jiri Vyskocil
# SPDX-License-Identifier: Apache-2.0

"""Sandbox hardening — install + remove for the optional MAC layer.

Self-contained: knows its own SELinux modules, AppArmor profiles,
permissive domains, and systemd unit metadata; runs the install +
remove sequences as straight scripts (no helper layer) so each
function reads top-to-bottom in one screen.

Called from the top-level ``terok hardening install`` orchestrator.
The orchestrator just delegates here (and to the matching function in
terok-clearance) — it owns no per-package logic.

``sudo`` is invoked subprocess-by-subprocess only for the four
operations that genuinely need root: ``semodule -i`` (load),
``semodule -r`` (unload), ``semanage permissive -a/-d`` (soak
posture), and the AppArmor ``install`` + ``apparmor_parser`` pair.
Drop-ins under ``~/.config/systemd/user/`` and ``systemctl --user``
restarts run unprivileged because they're the calling user's own
unit tree.

Replaces the older ``resources/install_hardening.sh`` shell flow.
That script grew to ~600 lines (set -euo pipefail + sudo + bridged
user-systemctl + tempdir cleanup + anti-tamper + drift detection)
with several rounds of bug-fixes for shell-specific edge cases.
This file is the entire replacement.
"""

from __future__ import annotations

import shutil
import subprocess
import sys
import tempfile
from importlib.resources import files as _resource_files
from pathlib import Path

# ---------- Configuration ----------

_RES = _resource_files("terok_sandbox.resources")

SELINUX_MODULES: tuple[Path, ...] = (
    Path(str(_RES / "selinux/terok_socket.te")),
    Path(str(_RES / "selinux/terok_gate.te")),
    Path(str(_RES / "selinux/terok_vault.te")),
)
"""SELinux ``.te`` source files compiled + loaded by `install`.

``terok_socket`` is the legacy allow-rule (container_t connectto on
the per-service socket types) — always loaded, separate from the
optional confined-domain modules but bundled in the same install
sequence so users have one entry point."""

PERMISSIVE_DOMAINS: tuple[str, ...] = ("terok_gate_t", "terok_vault_t")
"""Marked permissive at install time so the soak window collects
denials without breaking services.  Flipped to enforcing in a
follow-up commit once the AVC trail is quiet."""

APPARMOR_PROFILES: tuple[Path, ...] = tuple(
    Path(str(_RES / f"apparmor/{p}")) for p in ("terok-gate", "terok-vault")
)
"""AppArmor profile files; copied to ``/etc/apparmor.d/`` and loaded
via ``apparmor_parser -r``.  Skipped cleanly when AppArmor isn't the
active LSM (Fedora / RHEL hosts)."""

SERVICE_UNITS: tuple[tuple[str, str, str], ...] = (
    ("terok-vault.service", "terok_vault_t", "terok-vault"),
    ("terok-vault-socket.service", "terok_vault_t", "terok-vault"),
    ("terok-gate@.service", "terok_gate_t", "terok-gate"),
    ("terok-gate-socket.service", "terok_gate_t", "terok-gate"),
)
"""Per-unit drop-in metadata: ``(unit, selinux_type, apparmor_profile)``.
Templates (``@.service``) are skipped at restart time — instances
are short-lived inetd children, the drop-in applies to new instances
on next launch."""

INSTALL_COMMAND = "terok hardening install"
"""Single source for the user-facing install invocation string used
by sickbay hints and setup tips."""


_SELINUX_ENFORCE = Path("/sys/fs/selinux/enforce")
_APPARMOR_ROOT = Path("/sys/kernel/security/apparmor")


# ---------- Install ----------


def install() -> None:
    """Load modules, write drop-ins, restart active units.

    Idempotent — re-running replaces loaded modules in place,
    overwrites drop-ins, restarts active units once.  Caller is
    responsible for caching sudo credentials up front (the orchestrator
    runs ``sudo -v`` so the user enters their password once).
    """
    if _SELINUX_ENFORCE.is_file():
        for tool in ("checkmodule", "semodule_package", "semodule", "semanage"):
            if not shutil.which(tool):
                sys.exit(
                    f"error: {tool} not found "
                    "(install: dnf install selinux-policy-devel "
                    "policycoreutils-python-utils)"
                )
        print("==> sandbox: loading SELinux modules")
        with tempfile.TemporaryDirectory(prefix="terok-sandbox-") as wd:
            for te in SELINUX_MODULES:
                mod = Path(wd) / f"{te.stem}.mod"
                pp = Path(wd) / f"{te.stem}.pp"
                subprocess.run(["checkmodule", "-M", "-m", "-o", str(mod), str(te)], check=True)
                subprocess.run(["semodule_package", "-o", str(pp), "-m", str(mod)], check=True)
                subprocess.run(["sudo", "semodule", "-i", str(pp)], check=True)
                print(f"    loaded {te.stem}")
        for dom in PERMISSIVE_DOMAINS:
            # ``-a`` is idempotent on modern semanage but emits "already
            # permissive" on stderr; capture so a re-install stays quiet.
            subprocess.run(
                ["sudo", "semanage", "permissive", "-a", dom],
                check=False,
                capture_output=True,
            )
            print(f"    {dom} permissive (soak)")

    if _APPARMOR_ROOT.is_dir() and shutil.which("apparmor_parser"):
        print("==> sandbox: loading AppArmor profiles")
        for p in APPARMOR_PROFILES:
            target = f"/etc/apparmor.d/{p.name}"
            subprocess.run(["sudo", "install", "-m", "0644", str(p), target], check=True)
            subprocess.run(["sudo", "apparmor_parser", "-r", target], check=True)
            print(f"    loaded {p.name}")

    _write_dropins()
    _restart_active_units()


# ---------- Remove ----------


def remove() -> None:
    """Tear down everything `install` set up.

    Removes drop-ins first (so the next restart drops the
    SELinuxContext attachment) then unloads modules.  Reverse order
    of install."""
    _remove_dropins()
    _restart_active_units()

    if _APPARMOR_ROOT.is_dir() and shutil.which("apparmor_parser"):
        print("==> sandbox: unloading AppArmor profiles")
        for p in APPARMOR_PROFILES:
            target = Path(f"/etc/apparmor.d/{p.name}")
            if target.exists():
                subprocess.run(["sudo", "apparmor_parser", "-R", str(target)], check=False)
                subprocess.run(["sudo", "rm", "-f", str(target)], check=False)
                print(f"    unloaded {p.name}")

    if _SELINUX_ENFORCE.is_file() and shutil.which("semodule"):
        print("==> sandbox: unloading SELinux modules")
        for dom in PERMISSIVE_DOMAINS:
            subprocess.run(
                ["sudo", "semanage", "permissive", "-d", dom],
                check=False,
                capture_output=True,
            )
        for te in SELINUX_MODULES:
            subprocess.run(
                ["sudo", "semodule", "-r", te.stem],
                check=False,
                capture_output=True,
            )
            print(f"    unloaded {te.stem}")


# ---------- Drop-in helpers ----------


def _write_dropins() -> None:
    selinux = _SELINUX_ENFORCE.is_file()
    apparmor = _APPARMOR_ROOT.is_dir()
    if not (selinux or apparmor):
        return
    unit_dir = Path.home() / ".config/systemd/user"
    if not unit_dir.is_dir():
        sys.exit(
            f"error: {unit_dir} missing — run `terok setup` as the same "
            "user, then re-run hardening install"
        )
    print("==> sandbox: writing systemd drop-ins")
    for unit, sel_type, aa_profile in SERVICE_UNITS:
        body: list[str] = []
        if selinux:
            body.append(f"SELinuxContext=-unconfined_u:unconfined_r:{sel_type}:s0")
        if apparmor:
            body.append(f"AppArmorProfile={aa_profile}")
        d = unit_dir / f"{unit}.d"
        d.mkdir(parents=True, exist_ok=True)
        f = d / "hardening-mac.conf"
        f.write_text(
            "# Installed by `terok hardening install`\n[Service]\n" + "\n".join(body) + "\n"
        )
        print(f"    wrote {f.relative_to(unit_dir)}")


def _remove_dropins() -> None:
    unit_dir = Path.home() / ".config/systemd/user"
    if not unit_dir.is_dir():
        return
    print("==> sandbox: removing systemd drop-ins")
    for unit, _, _ in SERVICE_UNITS:
        f = unit_dir / f"{unit}.d" / "hardening-mac.conf"
        if not f.exists():
            continue
        f.unlink()
        # Drop the .d/ directory if it's now empty (other drop-in layers
        # — auditd, namespacing — may live alongside ours; preserve them).
        try:
            f.parent.rmdir()
        except OSError:
            pass
        print(f"    removed {f.relative_to(unit_dir)}")


def _restart_active_units() -> None:
    print("==> sandbox: restarting active units")
    subprocess.run(["systemctl", "--user", "daemon-reload"], check=False)
    for unit, _, _ in SERVICE_UNITS:
        # Templates (``@.service``) have no concrete instance to restart;
        # the drop-in applies to new instances on next launch.
        if "@" in unit:
            continue
        active = subprocess.run(["systemctl", "--user", "is-active", "--quiet", unit])
        if active.returncode == 0:
            subprocess.run(["systemctl", "--user", "restart", unit], check=False)
            print(f"    restarted {unit}")
