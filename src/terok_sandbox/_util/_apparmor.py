# SPDX-FileCopyrightText: 2026 Jiri Vyskocil
# SPDX-License-Identifier: Apache-2.0

"""AppArmor helpers — parallel of `_selinux` for the Debian/Ubuntu world.

terok ships optional confinement profiles for the host-side daemons
(gate, vault) so a compromise of either is bounded by policy rather
than the user's UID.  AppArmor is the second supported MAC backend
alongside SELinux; exactly one (or neither) is active per host, and
the install script (`install_command`) covers both.

The probes here are read-only and never raise on non-AppArmor systems
— they degrade to ``False`` / empty so callers can render a uniform
"not applicable" row without try/except plumbing at every call site.
"""

from __future__ import annotations

import shutil
from dataclasses import dataclass, field
from enum import Enum
from functools import lru_cache
from pathlib import Path

# ---------- Constants ----------

CONFINED_PROFILES: tuple[str, ...] = ("terok-gate", "terok-vault")
"""AppArmor profile names shipped under ``resources/apparmor/``.
Each is loaded as a *named* profile (no path attachment); the install
script wires the actual attachment via systemd
``AppArmorProfile=<name>`` in a unit drop-in."""

_SECFS_ROOT = Path("/sys/kernel/security/apparmor")
"""Root of the apparmor securityfs interface — its existence is the
authoritative 'AppArmor is loaded into this kernel' signal."""

_PROFILES_FILE = _SECFS_ROOT / "profiles"
"""Per-profile state listing — each line is ``<name> (<mode>)``."""


# ---------- Detection ----------


def is_apparmor_enabled() -> bool:
    """Return ``True`` if AppArmor is active in the running kernel.

    The presence of the ``/sys/kernel/security/apparmor`` directory is
    the canonical check — distros that ship AppArmor build but don't
    load the LSM (rare, but possible via ``apparmor=0`` on the kernel
    command line) won't have this directory.
    """
    return _SECFS_ROOT.is_dir()


def missing_policy_tools() -> list[str]:
    """Return names of AppArmor userspace tools not found on ``PATH``.

    ``apparmor_parser`` is the load/unload tool used by
    `install_command`'s install script; ``aa-complain`` / ``aa-enforce``
    flip the per-profile mode but are optional (the soak posture is
    declared via ``flags=(complain)`` in the profile header itself).
    """
    return [t for t in ("apparmor_parser",) if not shutil.which(t)]


# ---------- Profile-state probes ----------


def loaded_profiles() -> dict[str, str]:
    """Parse ``/sys/kernel/security/apparmor/profiles`` into ``{name: mode}``.

    The file is world-readable on every distro that ships AppArmor, so
    no privilege is needed.  Each line is ``<name> (<mode>)`` — modes
    are ``enforce``, ``complain``, ``kill``, or ``unconfined``.

    Returns an empty dict on non-AppArmor systems or if the file is
    absent for any reason.
    """
    try:
        text = _PROFILES_FILE.read_text()
    except (FileNotFoundError, PermissionError, OSError):
        return {}
    out: dict[str, str] = {}
    for line in text.splitlines():
        # ``profile-name (mode)`` — split on the last space-paren.
        if not line.endswith(")"):
            continue
        head, _, tail = line.rpartition(" (")
        if not head:
            continue
        out[head.strip()] = tail.rstrip(")").strip()
    return out


def is_profile_loaded(name: str) -> bool:
    """Return ``True`` if a profile named *name* is currently loaded."""
    return name in loaded_profiles()


def profile_mode(name: str) -> str | None:
    """Return the mode (``enforce`` / ``complain`` / …) of a loaded profile.

    Returns ``None`` if the profile is not loaded.
    """
    return loaded_profiles().get(name)


def loaded_confined_profiles() -> tuple[str, ...]:
    """Return the subset of `CONFINED_PROFILES` currently loaded.

    Mirrors `_selinux.loaded_confined_domains` so sickbay can render the
    two backends with the same code path.
    """
    loaded = loaded_profiles()
    return tuple(p for p in CONFINED_PROFILES if p in loaded)


# ---------- Aggregate status ----------


class ApparmorStatus(Enum):
    """Outcome of `check_status`.  Same shape as `_selinux.SelinuxStatus`
    so downstream renderers can branch uniformly across backends."""

    NOT_APPLICABLE = "not_applicable"
    """AppArmor is not active in the kernel.  This is the common case
    on Fedora / RHEL hosts where SELinux is the active MAC."""

    PROFILES_MISSING = "profiles_missing"
    """AppArmor active, but none of the terok profiles are loaded.
    User has not installed the optional hardening layer."""

    PROFILES_PARTIAL = "profiles_partial"
    """Some terok profiles loaded, others not.  Usually a botched
    install worth surfacing for repair."""

    OK_COMPLAIN = "ok_complain"
    """All terok profiles loaded, all in complain mode (initial soak
    posture).  Denials log but don't block."""

    OK_ENFORCE = "ok_enforce"
    """All terok profiles loaded, all in enforce mode — full hardening
    active."""


@dataclass(frozen=True)
class ApparmorCheckResult:
    """Structured outcome of `check_status`."""

    status: ApparmorStatus
    """Which branch of the decision tree fired."""

    loaded: tuple[str, ...] = field(default_factory=tuple)
    """Subset of `CONFINED_PROFILES` currently loaded (for diagnostics)."""

    missing_policy_tools: tuple[str, ...] = field(default_factory=tuple)
    """Names of missing userspace tools (only populated when relevant)."""


@lru_cache(maxsize=1)
def check_status() -> ApparmorCheckResult:
    """Evaluate AppArmor hardening readiness.

    Cached: profile state is loaded once per process — sickbay calls
    this from multiple rows and ``terok setup`` calls it for the
    summary line; recomputing the parse for each call is wasted work.
    """
    if not is_apparmor_enabled():
        return ApparmorCheckResult(ApparmorStatus.NOT_APPLICABLE)

    # Single parse of /sys/kernel/security/apparmor/profiles serves
    # both the loaded-set membership check and the per-profile mode
    # check below.
    modes = loaded_profiles()
    loaded = tuple(p for p in CONFINED_PROFILES if p in modes)
    if not loaded:
        return ApparmorCheckResult(
            ApparmorStatus.PROFILES_MISSING,
            missing_policy_tools=tuple(missing_policy_tools()),
        )
    if len(loaded) < len(CONFINED_PROFILES):
        return ApparmorCheckResult(ApparmorStatus.PROFILES_PARTIAL, loaded=loaded)

    all_enforcing = all(modes.get(p) == "enforce" for p in CONFINED_PROFILES)
    return ApparmorCheckResult(
        ApparmorStatus.OK_ENFORCE if all_enforcing else ApparmorStatus.OK_COMPLAIN,
        loaded=loaded,
    )
