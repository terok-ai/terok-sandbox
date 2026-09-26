# SPDX-FileCopyrightText: 2026 Jiri Vyskocil
# SPDX-License-Identifier: Apache-2.0

"""AppArmor profile helpers for the per-container dnsmasq DNS tier.

terok-shield runs a per-container dnsmasq whose config/pid/log live under
the sandbox-live ``tasks/<project>/<task>/shield`` tree in the operator's
home.  Distros that ship an enforcing AppArmor profile for
``/usr/sbin/dnsmasq`` (Arch/Manjaro, the apparmor.d set) confine it to the
conventional server paths and deny that tree, so shield falls back to the
lookup tier.  This module detects that confinement and points the operator at
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

from dataclasses import dataclass
from enum import Enum
from functools import lru_cache
from importlib.resources import files as _resource_files
from pathlib import Path

from terok_util import find_host_tool

# Kernel sysfs node: "Y" when AppArmor is enabled.
_APPARMOR_ENABLED = Path("/sys/module/apparmor/parameters/enabled")

# Stock dnsmasq profile locations, by profile set.
_DNSMASQ_PROFILES = (
    Path("/etc/apparmor.d/usr.sbin.dnsmasq"),  # Debian/Ubuntu
    Path("/etc/apparmor.d/dnsmasq"),  # apparmor.d project / Arch
)

# The rules, their revision, and the marker that frames them all live in
# ``resources/apparmor/dnsmasq_addendum.template`` — the single source of
# truth both ``install_profile.sh`` and
# [`render_addendum`][terok_sandbox._util._apparmor.render_addendum] render.
# Bump the revision in that template's header line when the rules change:
# an older on-disk block carries the base marker but a different header, so
# it reads as installed-but-outdated and the operator is prompted to
# reinstall.  ``r2`` = the one ``dnsmasq.*`` glob that replaced the per-file
# (conf/pid/log) rules, which broke DNS whenever shield added a file
# (terok-ai/terok#1246).
#
# The bare marker is the one fact that must NOT move on a bump: the
# installer's strip range and the staleness probe both key on it.
_ADDENDUM_MARKER = "terok-shield apparmor"


def is_apparmor_enabled() -> bool:
    """Return ``True`` if the kernel has AppArmor enabled (sysfs ``Y``)."""
    try:
        return _APPARMOR_ENABLED.read_text().strip() == "Y"
    except OSError:
        return False


def dnsmasq_profile() -> Path | None:
    """Return the stock dnsmasq AppArmor profile present on this host, if any."""
    return next((p for p in _DNSMASQ_PROFILES if p.is_file()), None)


def local_include_path(profile: Path) -> Path:
    """Return the ``local/`` include *profile* is extended through.

    AppArmor profiles include ``local/<profile name>`` for site changes;
    that file — not the distro profile — is where the managed block
    lands, and what the interactive flow names before installing.
    """
    return profile.parent / "local" / profile.name


def _local_include_text(profile: Path) -> str:
    """Return *profile*'s local-include text, or ``""`` if absent/unreadable."""
    try:
        return local_include_path(profile).read_text()
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
    rides the lookup tier until the operator reinstalls (terok-ai/terok#1246)."""

    OK = "ok"
    """The terok addendum is installed at the current revision."""

    @property
    def action_needed(self) -> bool:
        """Whether this status calls for an operator install."""
        return self in (AppArmorStatus.PROFILE_MISSING, AppArmorStatus.PROFILE_OUTDATED)


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
    if not is_apparmor_enabled() or find_host_tool("dnsmasq") is None:
        return AppArmorCheckResult(AppArmorStatus.NOT_APPLICABLE)
    profile = dnsmasq_profile()
    if profile is None:
        return AppArmorCheckResult(AppArmorStatus.NOT_APPLICABLE)
    addendum = _local_include_text(profile)
    if _ADDENDUM_MARKER not in addendum:
        return AppArmorCheckResult(AppArmorStatus.PROFILE_MISSING)
    # An install is a verbatim template render, so the template's own
    # header line is the current-revision test — and comparing whole
    # lines keeps a future ``r20`` from reading as today's ``r2``.
    if _addendum_header() not in addendum.splitlines():
        return AppArmorCheckResult(AppArmorStatus.PROFILE_OUTDATED)
    return AppArmorCheckResult(AppArmorStatus.OK)


def state_root() -> Path:
    """The sandbox-live root the addendum rules must permit.

    Full precedence (env, config, default) via
    [`sandbox_live_root`][terok_sandbox.paths.sandbox_live_root], so
    the rules name the tree the shield actually writes no matter which
    frontend asks.
    """
    from ..paths import sandbox_live_root

    return sandbox_live_root()


@lru_cache(maxsize=1)
def install_script_path() -> Path:
    """Return the path to the bundled ``install_profile.sh`` AppArmor installer.

    Installation is delegated to this short, inspectable shell script —
    run with ``sudo bash <path> <state_root>`` — so it can be ``cat``-ed
    and audited before the privilege escalation.
    """
    return Path(str(_resource_files("terok_sandbox.resources.apparmor") / "install_profile.sh"))


def render_addendum(state_root: Path) -> str:
    """Render the managed addendum block exactly as the installer writes it.

    Same template, same substitution as ``install_profile.sh`` (a lone
    ``@STATE_ROOT@`` token, trailing slash stripped) — rendered on the
    unprivileged side, where the operator's paths are known.  What the
    ``setup apparmor`` show option prints is therefore byte-identical to
    what a subsequent ``sudo bash`` install appends to the profile.
    """
    return _addendum_template().replace("@STATE_ROOT@", str(state_root).rstrip("/"))


@lru_cache(maxsize=1)
def _addendum_template() -> str:
    """The raw ``dnsmasq_addendum.template`` text (single ``@STATE_ROOT@`` token)."""
    return Path(
        str(_resource_files("terok_sandbox.resources.apparmor") / "dnsmasq_addendum.template")
    ).read_text()


def _addendum_header() -> str:
    """The template's marker line — the current revision, verbatim."""
    return _addendum_template().splitlines()[0]
