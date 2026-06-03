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
terok addendum is ``PROFILE_MISSING``.  Install is delegated to
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

# Marker the installer writes into the local include (see install_profile.sh).
_ADDENDUM_MARKER = "terok-shield apparmor"


def is_apparmor_enabled() -> bool:
    """Return ``True`` if the kernel has AppArmor enabled (sysfs ``Y``)."""
    try:
        return _APPARMOR_ENABLED.read_text().strip() == "Y"
    except OSError:
        return False


def _dnsmasq_profile() -> Path | None:
    """Return the stock dnsmasq AppArmor profile present on this host, if any."""
    return next((p for p in _DNSMASQ_PROFILES if p.is_file()), None)


def _addendum_installed(profile: Path) -> bool:
    """Return ``True`` if the terok addendum is present in *profile*'s local include."""
    local = profile.parent / "local" / profile.name
    try:
        return _ADDENDUM_MARKER in local.read_text()
    except OSError:
        return False


class AppArmorStatus(Enum):
    """Outcome of [`check_status`][terok_sandbox._util._apparmor.check_status]."""

    NOT_APPLICABLE = "not_applicable"
    """No AppArmor, no dnsmasq, or no dnsmasq profile — nothing to do."""

    PROFILE_MISSING = "profile_missing"
    """dnsmasq is AppArmor-profiled but the terok addendum isn't installed."""

    OK = "ok"
    """The terok addendum is installed."""


@dataclass(frozen=True)
class AppArmorCheckResult:
    """Structured outcome of [`check_status`][terok_sandbox._util._apparmor.check_status]."""

    status: AppArmorStatus


def check_status() -> AppArmorCheckResult:
    """Evaluate whether the dnsmasq AppArmor addendum is needed or installed.

    File-based and unprivileged: an AppArmor-enabled host with dnsmasq and
    a stock dnsmasq profile but no terok addendum is ``PROFILE_MISSING``;
    everything else is ``NOT_APPLICABLE`` or ``OK``.
    """
    if not is_apparmor_enabled() or shutil.which("dnsmasq") is None:
        return AppArmorCheckResult(AppArmorStatus.NOT_APPLICABLE)
    profile = _dnsmasq_profile()
    if profile is None:
        return AppArmorCheckResult(AppArmorStatus.NOT_APPLICABLE)
    if _addendum_installed(profile):
        return AppArmorCheckResult(AppArmorStatus.OK)
    return AppArmorCheckResult(AppArmorStatus.PROFILE_MISSING)


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
