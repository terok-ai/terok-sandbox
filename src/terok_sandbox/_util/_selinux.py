# SPDX-FileCopyrightText: 2026 Jiri Vyskocil
# SPDX-License-Identifier: Apache-2.0

"""SELinux helpers for socket labeling and policy management.

Terok services listen on Unix sockets that rootless Podman containers
must ``connect()`` to.  SELinux blocks this by default — the kernel's
``connectto`` check uses the **socket object's** SID (inherited from the
creating process, typically ``unconfined_t``), not the file inode's label.

To work around this without disabling confinement:

1. A custom policy module defines ``terok_socket_t`` and grants
   ``container_t → terok_socket_t:unix_stream_socket connectto``.
2. Services call `setsockcreatecon` *before* ``socket()`` so the
   kernel assigns ``terok_socket_t`` to the socket object.
3. After ``bind()``, the socket object carries ``terok_socket_t`` and
   containers can connect.

The ``sock_file { write }`` check (file-level access) is separately
handled by Podman's ``:z`` volume relabeling.

libselinux is loaded via [`ctypes`][ctypes] at call time, so this module has
no runtime dependency on the ``python3-libselinux`` distribution package
— ``libselinux.so.1`` from the base ``libselinux`` package is sufficient.
All functions degrade gracefully on non-SELinux systems.
"""

from __future__ import annotations

import ctypes
import shutil
from contextlib import contextmanager
from dataclasses import dataclass, field
from enum import Enum
from functools import lru_cache
from importlib.resources import files as _resource_files
from pathlib import Path
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from collections.abc import Iterator

# ---------- Constants ----------

SELINUX_SOCKET_TYPE = "terok_socket_t"
"""Legacy SELinux type applied to terok service sockets via
``setsockcreatecon()``.  Kept for back-compat; the per-service socket
types declared in ``terok_gate.te`` / ``terok_vault.te`` are the
forward path."""

_SELINUX_CONTEXT = f"system_u:object_r:{SELINUX_SOCKET_TYPE}:s0"
"""Full SELinux context string for socket creation."""

CONFINED_DOMAINS: tuple[str, ...] = ("terok_gate_t", "terok_vault_t")
"""Optional confined process domains shipped by the hardening modules.
Each is loaded by its own ``.te`` (``terok_gate.te``, ``terok_vault.te``)
and is independent of the legacy ``terok_socket`` allow-rule."""

_ENFORCE_PATH = Path("/sys/fs/selinux/enforce")
"""Kernel sysfs node indicating SELinux enforcement state."""


# ---------- Detection ----------


def is_selinux_enforcing() -> bool:
    """Return ``True`` if SELinux is in enforcing mode.

    Reads ``/sys/fs/selinux/enforce`` directly — no external commands.
    Returns ``False`` on non-SELinux systems or if the file is unreadable.
    """
    try:
        return _ENFORCE_PATH.read_text().strip() == "1"
    except (FileNotFoundError, PermissionError, OSError):
        return False


def is_selinux_enabled() -> bool:
    """Return ``True`` if SELinux is active (enforcing or permissive)."""
    return _ENFORCE_PATH.is_file()


def is_policy_installed() -> bool:
    """Return ``True`` if ``terok_socket_t`` is a valid type in the loaded policy.

    Uses ``libselinux``'s ``security_check_context()``, which succeeds
    iff the context (and therefore the custom type) is known to the
    currently loaded policy — a pure userspace query requiring no
    subprocess and no privileges.

    The previous ``semodule -l`` subprocess approach silently failed
    for non-root callers on Fedora, where ``/var/lib/selinux/.../active/``
    is root-readable only.  ``terok sickbay`` and ``terok setup``
    both run as the user, so they would always report the policy as
    missing even right after a successful install.
    """
    lib = _load_libselinux()
    if lib is None:
        return False
    return lib.security_check_context(_SELINUX_CONTEXT.encode()) == 0


def is_libselinux_available() -> bool:
    """Return ``True`` if ``libselinux.so.1`` can be loaded via ctypes.

    On SELinux-enforcing hosts, a ``False`` return is a silent-failure
    risk: service sockets would bind without ``terok_socket_t`` labeling,
    and container clients would be denied ``connectto`` even when the
    ``terok_socket`` policy module is installed.
    """
    return _load_libselinux() is not None


def missing_policy_tools() -> list[str]:
    """Return names of policy-compilation tools not found on ``PATH``.

    Each ``.te`` policy source is compiled at install time by
    `install_command`'s shell script, which requires all three of
    ``checkmodule``, ``semodule_package``, and ``semodule``.  An empty
    list means the install script will not abort for missing tools.
    Names are returned in invocation order so callers can surface the
    first one a user would hit.
    """
    return [t for t in ("checkmodule", "semodule_package", "semodule") if not shutil.which(t)]


# ---------- Policy installation ----------


def policy_source_path() -> Path:
    """Return the path to the bundled ``terok_socket.te`` policy source."""
    return Path(str(_resource_files("terok_sandbox.resources.selinux") / "terok_socket.te"))


def install_command() -> str:
    """Return the install command for the connectto-allow policy module.

    The hardening tool lives outside the daily user CLI — under
    ``terok_sandbox.tools.hardening``, a Python entrypoint a
    packager (deb/rpm postinst) or a manual operator runs to load
    the bundled SELinux + AppArmor assets.  Sickbay's ``SELinux
    policy`` row hints at this command when the legacy
    ``terok_socket_t`` allow-rule isn't loaded.
    """
    return "python -m terok_sandbox.tools.hardening install"


def is_domain_loaded(domain: str) -> bool:
    """Return ``True`` if *domain* is a valid process type in the loaded policy.

    Same userspace probe as `is_policy_installed` but for an arbitrary
    process domain — the constructed context
    ``system_u:system_r:<domain>:s0`` validates iff the type exists AND
    the role association from the module's ``role system_r types ...;``
    rule is in effect.

    Used by sickbay to render a per-domain status row separate from the
    legacy allow-rule check (``terok_socket_t`` is an *object* type;
    confined domains are *process* types and load independently).
    """
    lib = _load_libselinux()
    if lib is None:
        return False
    ctx = f"system_u:system_r:{domain}:s0".encode()
    return lib.security_check_context(ctx) == 0


def is_socket_type_loaded(socket_type: str) -> bool:
    """Return ``True`` if *socket_type* is a valid object type in loaded policy.

    Object-type counterpart of `is_domain_loaded` — validates a context
    of the form ``system_u:object_r:<socket_type>:s0`` against the
    currently loaded policy.  Used by `socket_selinux_context` to pick
    the first loadable type from a candidate list, so per-service
    socket types added by the optional hardening modules degrade
    gracefully to ``terok_socket_t`` on hosts where only the legacy
    allow-rule is installed.
    """
    lib = _load_libselinux()
    if lib is None:
        return False
    ctx = f"system_u:object_r:{socket_type}:s0".encode()
    return lib.security_check_context(ctx) == 0


def loaded_confined_domains() -> tuple[str, ...]:
    """Return the subset of `CONFINED_DOMAINS` whose modules are loaded.

    Convenience for sickbay: empty tuple means the optional hardening
    layer is not installed; full tuple means every domain is loaded;
    partial means a botched / partial install worth surfacing.
    """
    return tuple(d for d in CONFINED_DOMAINS if is_domain_loaded(d))


# ---------- Socket context labeling ----------


@contextmanager
def socket_selinux_context(*candidates: str) -> Iterator[None]:
    """Apply the first loadable type from *candidates* as the socket creation context.

    Any ``socket()`` call within the ``with`` body produces a socket
    whose kernel SID is the chosen type, enabling ``container_t``
    clients to ``connectto`` it once the matching policy is installed.
    The previous context is restored on exit.

    Variadic so service code can express "preferred type, with
    fallback" in one call — the per-service socket types from the
    optional confined-domain modules
    (``terok_gate_sock_t`` / ``terok_vault_sock_t`` /
    ``terok_vault_ssh_sock_t``) are loaded by ``terok hardening install``
    on top of the legacy allow rule, but services need to keep working
    on hosts where only the legacy ``terok_socket`` module is loaded
    (or none, on non-SELinux distros).  Usage::

        with socket_selinux_context("terok_gate_sock_t", SELINUX_SOCKET_TYPE):
            sock = socket.socket(AF_UNIX, SOCK_STREAM)
            sock.bind(str(path))

    Behavioural ladder, top-down:

    1. Non-SELinux host — yield without labelling (no-op).
    2. SELinux host, first candidate is a valid type — use it; the
       restrictive per-service connectto rule fires.
    3. SELinux host, first candidate not loaded but a fallback is —
       use the fallback (typically ``SELINUX_SOCKET_TYPE``); the
       broad legacy allow rule fires.
    4. SELinux host, none of the candidates loaded — yield without
       labelling; ``container_t connectto`` will be denied.  The
       sickbay row + setup WARN tell the user to install the policy.

    Calling with no arguments is equivalent to passing the legacy
    type (``SELINUX_SOCKET_TYPE``) — preserved for backward compat
    with pre-hardening callers.
    """
    if not candidates:
        candidates = (SELINUX_SOCKET_TYPE,)

    if not is_selinux_enabled():
        yield
        return

    chosen = next((t for t in candidates if is_socket_type_loaded(t)), None)
    if chosen is None:
        # No candidate is in the loaded policy — bind without
        # labelling.  Container connectto will be denied; the operator
        # surface (sickbay / terok setup) tells them to install the
        # policy module that declares the type.
        yield
        return

    context = f"system_u:object_r:{chosen}:s0"
    old = _try_getsockcreatecon()
    _try_setsockcreatecon(context)
    try:
        yield
    finally:
        _try_setsockcreatecon(old)


# ---------- libselinux ctypes bindings ----------


@lru_cache(maxsize=1)
def _load_libselinux() -> ctypes.CDLL | None:
    """Load ``libselinux.so.1`` and configure function prototypes (cached).

    Returns the ``CDLL`` handle, or ``None`` on systems without libselinux
    (non-SELinux hosts, or the rare SELinux host missing the base package).
    """
    try:
        lib = ctypes.CDLL("libselinux.so.1", use_errno=True)
    except OSError:
        return None
    lib.setsockcreatecon.argtypes = [ctypes.c_char_p]
    lib.setsockcreatecon.restype = ctypes.c_int
    lib.getsockcreatecon.argtypes = [ctypes.POINTER(ctypes.c_char_p)]
    lib.getsockcreatecon.restype = ctypes.c_int
    lib.freecon.argtypes = [ctypes.c_char_p]
    lib.freecon.restype = None
    lib.security_check_context.argtypes = [ctypes.c_char_p]
    lib.security_check_context.restype = ctypes.c_int
    return lib


def _try_setsockcreatecon(context: str | None) -> bool:
    """Attempt to set the socket creation context.  Pass ``None`` to clear.

    Returns ``True`` on success, ``False`` if libselinux is unavailable
    or the call returns a non-zero status.
    """
    lib = _load_libselinux()
    if lib is None:
        return False
    arg = context.encode() if context is not None else None
    return lib.setsockcreatecon(arg) == 0


def _try_getsockcreatecon() -> str | None:
    """Read the current socket creation context, or ``None`` if unset or unavailable."""
    lib = _load_libselinux()
    if lib is None:
        return None
    ptr = ctypes.c_char_p()
    if lib.getsockcreatecon(ctypes.byref(ptr)) != 0:
        return None
    if not ptr.value:
        return None
    result = ptr.value.decode()
    lib.freecon(ptr)
    return result


# ---------- Aggregate status ----------


class SelinuxStatus(Enum):
    """Outcome of `check_status` — the single decision tree behind
    both ``terok setup``'s prereq check and ``terok sickbay``'s health check.
    """

    NOT_APPLICABLE_TCP_MODE = "not_applicable_tcp_mode"
    """Transport is ``tcp``; the ``terok_socket_t`` policy is irrelevant."""

    NOT_APPLICABLE_PERMISSIVE = "not_applicable_permissive"
    """Socket transport, but SELinux is disabled or permissive."""

    POLICY_MISSING = "policy_missing"
    """Enforcing host, socket transport, but ``terok_socket`` module is not loaded."""

    LIBSELINUX_MISSING = "libselinux_missing"
    """Policy is loaded but ``libselinux.so.1`` cannot be dlopen'd — silent-
    failure case where sockets would bind as ``unconfined_t`` regardless."""

    OK = "ok"
    """Enforcing, policy installed, libselinux loadable — all good."""


@dataclass(frozen=True)
class SelinuxCheckResult:
    """Structured outcome of `check_status`.

    Callers decide how to present the result; this struct only carries
    the decision tree's output so that ``terok setup`` (printed multi-
    line warnings) and ``terok sickbay`` (tuple-based check result) can
    share one source of truth for the branching.
    """

    status: SelinuxStatus
    """Which branch of the decision tree fired."""

    missing_policy_tools: tuple[str, ...] = field(default_factory=tuple)
    """Names of missing compile tools (only populated for ``POLICY_MISSING``)."""


def check_status(*, services_mode: str) -> SelinuxCheckResult:
    """Evaluate SELinux readiness for socket-transport services.

    *services_mode* is the caller's configured transport (``tcp`` or
    ``socket``) — passed in rather than read from sandbox config so the
    helper stays free of cross-package config plumbing.  Consumers
    (``terok setup``, ``terok sickbay``) call
    [`terok_sandbox.config.services_mode`][terok_sandbox.config.services_mode] themselves.
    """
    if services_mode != "socket":
        return SelinuxCheckResult(SelinuxStatus.NOT_APPLICABLE_TCP_MODE)
    if not is_selinux_enforcing():
        return SelinuxCheckResult(SelinuxStatus.NOT_APPLICABLE_PERMISSIVE)
    if not is_policy_installed():
        return SelinuxCheckResult(
            SelinuxStatus.POLICY_MISSING,
            missing_policy_tools=tuple(missing_policy_tools()),
        )
    if not is_libselinux_available():
        return SelinuxCheckResult(SelinuxStatus.LIBSELINUX_MISSING)
    return SelinuxCheckResult(SelinuxStatus.OK)
