# SPDX-FileCopyrightText: 2026 Jiri Vyskocil
# SPDX-License-Identifier: Apache-2.0

"""AppArmor profile helpers for the per-container dnsmasq DNS tier.

terok-shield runs a per-container dnsmasq whose config/pid/log live under
the sandbox-live ``tasks/<project>/<task>/shield`` tree in the operator's
home.  Distros that ship an enforcing AppArmor profile for
``/usr/sbin/dnsmasq`` (Arch/Manjaro, the apparmor.d set) confine it to the
conventional server paths and deny that tree, so shield falls back to the
dig tier.  This module detects that confinement and points the operator at
the bundled installer that adds an addendum permitting the shield tree.

Detection is by file presence — unprivileged, no ``aa-status``/root: an
AppArmor-enabled host that has dnsmasq and a stock dnsmasq profile but no
terok addendum is ``PROFILE_MISSING``, and one whose addendum is an older
revision (marker present, current revision absent) is ``PROFILE_OUTDATED``
— both point the operator at the installer.  Install is delegated to
``resources/apparmor/install_profile.sh`` — a short, auditable script run
with ``sudo bash`` (no compilation, just ``apparmor_parser -r``).
"""

from __future__ import annotations

import shutil
from dataclasses import dataclass
from enum import Enum
from functools import lru_cache
from importlib.resources import files as _resource_files
from pathlib import Path

# Kernel sysfs node: "Y" when AppArmor is enabled.
_APPARMOR_ENABLED = Path("/sys/module/apparmor/parameters/enabled")

# Stock dnsmasq profile locations, by profile set.
_DNSMASQ_PROFILES = (
    Path("/etc/apparmor.d/usr.sbin.dnsmasq"),  # Debian/Ubuntu
    Path("/etc/apparmor.d/dnsmasq"),  # apparmor.d project / Arch
)

# Marker the installer writes into the local include (see install_profile.sh),
# and the revision suffix that identifies the CURRENT rule set.  Bump the
# revision whenever install_profile.sh's rules change: an older on-disk block
# still carries the base marker but not the current revision, so it reads as
# installed-but-outdated and the operator is prompted to reinstall — the rules
# live in the shell installer, so this revision is the single fact the two
# sides share.  ``r2`` = the one ``dnsmasq.*`` glob that replaced the per-file
# (conf/pid/log) rules, which broke DNS whenever shield added a file
# (terok-ai/terok#1246).
_ADDENDUM_MARKER = "terok-shield apparmor"
_ADDENDUM_REVISION = "r2"


def is_apparmor_enabled() -> bool:
    """Return ``True`` if the kernel has AppArmor enabled (sysfs ``Y``)."""
    try:
        return _APPARMOR_ENABLED.read_text().strip() == "Y"
    except OSError:
        return False


def _dnsmasq_profile() -> Path | None:
    """Return the stock dnsmasq AppArmor profile present on this host, if any."""
    return next((p for p in _DNSMASQ_PROFILES if p.is_file()), None)


def _local_include_text(profile: Path) -> str:
    """Return *profile*'s local-include text, or ``""`` if absent/unreadable."""
    try:
        return (profile.parent / "local" / profile.name).read_text()
    except OSError:
        return ""


class AppArmorStatus(Enum):
    """Outcome of [`check_status`][terok_sandbox._util._apparmor.check_status]."""

    NOT_APPLICABLE = "not_applicable"
    """No AppArmor, no dnsmasq, or no dnsmasq profile — nothing to do."""

    PROFILE_MISSING = "profile_missing"
    """dnsmasq is AppArmor-profiled but the terok addendum isn't installed."""

    PROFILE_OUTDATED = "profile_outdated"
    """The terok addendum is installed but at an older revision whose rules no
    longer cover what shield writes — dnsmasq stays confined and DNS silently
    rides the dig tier until the operator reinstalls (terok-ai/terok#1246)."""

    OK = "ok"
    """The terok addendum is installed at the current revision."""


@dataclass(frozen=True)
class AppArmorCheckResult:
    """Structured outcome of [`check_status`][terok_sandbox._util._apparmor.check_status]."""

    status: AppArmorStatus


def check_status() -> AppArmorCheckResult:
    """Evaluate whether the dnsmasq AppArmor addendum is needed, stale, or current.

    File-based and unprivileged.  An AppArmor-enabled host with dnsmasq and
    a stock dnsmasq profile is ``PROFILE_MISSING`` with no terok addendum,
    ``PROFILE_OUTDATED`` when an older-revision addendum is present (the
    marker but not the current revision), and ``OK`` at the current
    revision; anything else is ``NOT_APPLICABLE``.
    """
    if not is_apparmor_enabled() or shutil.which("dnsmasq") is None:
        return AppArmorCheckResult(AppArmorStatus.NOT_APPLICABLE)
    profile = _dnsmasq_profile()
    if profile is None:
        return AppArmorCheckResult(AppArmorStatus.NOT_APPLICABLE)
    addendum = _local_include_text(profile)
    if _ADDENDUM_MARKER not in addendum:
        return AppArmorCheckResult(AppArmorStatus.PROFILE_MISSING)
    # Match the revision as a whole token (trailing space), so a future ``r20``
    # is not read as the current ``r2`` and wrongly reported OK.
    if f"{_ADDENDUM_MARKER} {_ADDENDUM_REVISION} " not in addendum:
        return AppArmorCheckResult(AppArmorStatus.PROFILE_OUTDATED)
    return AppArmorCheckResult(AppArmorStatus.OK)


@lru_cache(maxsize=1)
def install_script_path() -> Path:
    """Return the path to the bundled ``install_profile.sh`` AppArmor installer.

    Installation is delegated to this short, inspectable shell script —
    run with ``sudo bash <path> <state_root>`` — so it can be ``cat``-ed
    and audited before the privilege escalation.
    """
    return Path(str(_resource_files("terok_sandbox.resources.apparmor") / "install_profile.sh"))


def install_command(state_root: Path) -> str:
    """Return the ``sudo bash <script> <state_root>`` installer invocation.

    *state_root* is the sandbox-live root whose ``tasks/*/*/shield`` tree
    the rendered profile must permit.  The caller supplies it because the
    script runs under ``sudo`` and cannot resolve the operator's home.
    """
    return f"sudo bash {install_script_path()} {state_root}"
