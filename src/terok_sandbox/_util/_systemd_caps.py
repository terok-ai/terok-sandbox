# SPDX-FileCopyrightText: 2026 Jiri Vyskocil
# SPDX-License-Identifier: Apache-2.0

"""Capability-aware systemd-hardening directives for terok user units.

The base ``*.service`` templates ship a Ubuntu 24.04-safe set: a few
directives (``AmbientCapabilities=`` cleared, ``ProtectClock=``,
``ProtectKernelTunables=``, ``ProtectKernelLogs=``,
``ProtectKernelModules=``, ``PrivateDevices=``) all *implicitly*
modify ``CapabilityBoundingSet``, which fails in ``--user`` mode on
Ubuntu 24.04 with ``status=218/CAPABILITIES`` because the user
manager lacks ``CAP_SETPCAP``.  Fedora 43+ user managers have the
cap and accept the directives.

Rather than ship two flavours of unit, the install path renders the
canonical Ubuntu-safe unit and *additionally* drops in
``hardening-systemd.conf`` under ``<unit>.d/`` when the host's user
manager is known to support the strict block.  The drop-in stacks
cleanly with the MAC-side ``hardening-mac.conf`` (different filename,
written by a separate tool).  Distro packagers can ship the strict
block inlined in the canonical unit and skip the runtime drop-in
entirely.

The probe is intentionally crude (read ``/etc/os-release``,
allow-list a small set of distro/version pairs known to ship a
``CAP_SETPCAP``-capable user manager).  Soak data + reports from the
field will refine the table.
"""

# TODO(soak): Replace the os-release allow-list with a direct probe
# of the user-systemd manager's ``CapBnd`` (read ``CapBnd:`` from
# ``/proc/<systemd-user-pid>/status`` and check bit 8 / 0x100 = CAP_SETPCAP).
# The current allow-list will mis-fire in two ways:
#   * False negatives — Ubuntu / Debian configurations where the user
#     manager has been granted CAP_SETPCAP via a service-manager
#     override or distro patch will get the reduced block when they
#     could safely run the strict one.
#   * False positives — non-default Fedora 43/44 deployments where
#     the user manager has been stripped of CAP_SETPCAP (custom
#     security-hardening policies, certain immutable variants) will
#     try to apply the strict block and fail at unit start.
# Must be replaced before this lands on master.

from __future__ import annotations

from pathlib import Path

_OS_RELEASE = Path("/etc/os-release")

# Drop-in content — the six directives that master removed in #247 + #249.
# Lives as one block so the install/remove path is a single write/unlink.
STRICT_HARDENING_DROPIN = """\
# Installed by terok setup when the host's user-systemd manager
# supports capability-modifying hardening (currently a Fedora 43+
# allow-list — see terok_sandbox._util._systemd_caps).  These
# directives implicitly drop bits from CapabilityBoundingSet, which
# fails on Ubuntu 24.04 user managers (CAP_SETPCAP missing) — that
# fail mode is why they were lifted out of the canonical unit.
[Service]
AmbientCapabilities=
ProtectClock=yes
ProtectKernelTunables=yes
ProtectKernelLogs=yes
ProtectKernelModules=yes
PrivateDevices=yes
"""

DROPIN_FILENAME = "hardening-systemd.conf"
"""Symmetric with the MAC layer's ``hardening-mac.conf`` (distinct
filename, distinct concern, both stack under the same ``<unit>.d/``
directory)."""


def supports_strict_user_hardening() -> bool:
    """Return True iff the host's user-systemd manager is known to
    accept the strict directives in `STRICT_HARDENING_DROPIN`.

    Allow-list:
      * Fedora ≥43 (and Fedora-derivative distros via ``ID_LIKE``).

    Conservative default — anything we haven't explicitly cleared
    returns False.  Refining the list is a soak deliverable; the
    TODO at the top of the module spells out the right end state.
    """
    try:
        text = _OS_RELEASE.read_text(encoding="utf-8")
    except OSError:
        return False
    fields: dict[str, str] = {}
    for line in text.splitlines():
        if "=" not in line:
            continue
        k, _, v = line.partition("=")
        fields[k.strip()] = v.strip().strip('"')
    distro_id = fields.get("ID", "")
    id_like = fields.get("ID_LIKE", "").split()
    version_id = fields.get("VERSION_ID", "")
    if distro_id == "fedora" or "fedora" in id_like:
        # Fedora's VERSION_ID is just the major number; everything else
        # (rawhide, etc.) is "rawhide" so this comparison is safe.
        try:
            return int(version_id) >= 43
        except ValueError:
            # Rawhide / other non-numeric — assume capable.
            return True
    return False


def write_dropin(unit_dir: Path, unit_name: str) -> bool:
    """Write the strict-hardening drop-in for *unit_name* under *unit_dir*.

    No-op (returns False) when the host doesn't support the strict
    block.  Returns True when the drop-in was written.  Idempotent —
    overwrites an existing drop-in so re-running ``terok setup``
    after a probe refinement picks up the new content.
    """
    if not supports_strict_user_hardening():
        return False
    dropin_dir = unit_dir / f"{unit_name}.d"
    dropin_dir.mkdir(parents=True, exist_ok=True)
    (dropin_dir / DROPIN_FILENAME).write_text(STRICT_HARDENING_DROPIN, encoding="utf-8")
    return True


def remove_dropin(unit_dir: Path, unit_name: str) -> None:
    """Remove the strict-hardening drop-in for *unit_name*.

    Best-effort: missing drop-in or missing ``<unit>.d/`` directory
    is fine.  The directory is rmdir'd if it ends up empty so
    other drop-in layers (MAC) can still create their own without
    clashing on a stale empty dir.
    """
    dropin_file = unit_dir / f"{unit_name}.d" / DROPIN_FILENAME
    if dropin_file.exists():
        dropin_file.unlink()
    parent = dropin_file.parent
    if parent.is_dir():
        try:
            parent.rmdir()
        except OSError:
            pass  # other drop-ins still living here
