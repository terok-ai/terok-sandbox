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
2. Services call :func:`setsockcreatecon` *before* ``socket()`` so the
   kernel assigns ``terok_socket_t`` to the socket object.
3. After ``bind()``, the socket object carries ``terok_socket_t`` and
   containers can connect.

The ``sock_file { write }`` check (file-level access) is separately
handled by Podman's ``:z`` volume relabeling.

libselinux is loaded via :mod:`ctypes` at call time, so this module has
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
"""Custom SELinux type applied to terok service sockets."""

_SELINUX_CONTEXT = f"system_u:object_r:{SELINUX_SOCKET_TYPE}:s0"
"""Full SELinux context string for socket creation."""

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

    The ``terok_socket`` policy is compiled from its ``.te`` source at
    install time by :func:`install_policy`, which requires all three of
    ``checkmodule``, ``semodule_package``, and ``semodule``.  An empty
    list means ``install_policy()`` will not fail with ``SystemExit`` for
    missing tools.  Names are returned in invocation order so callers
    can surface the first one a user would hit.
    """
    return [t for t in ("checkmodule", "semodule_package", "semodule") if not shutil.which(t)]


# ---------- Policy installation ----------


def policy_source_path() -> Path:
    """Return the path to the bundled ``terok_socket.te`` policy source."""
    return Path(str(_resource_files("terok_sandbox.resources.selinux") / "terok_socket.te"))


@lru_cache(maxsize=1)
def install_script_path() -> Path:
    """Return the path to the bundled ``install_policy.sh`` installer.

    Installation is delegated to this short, inspectable shell script —
    which users run with ``sudo bash <path>`` — rather than a Python
    wrapper.  Running Python as root imports a large dependency graph;
    a dedicated shell script can be ``cat``-ed and audited in seconds
    before the privilege escalation.
    """
    return Path(str(_resource_files("terok_sandbox.resources.selinux") / "install_policy.sh"))


def install_command() -> str:
    """Return the full ``sudo bash <path>`` shell command for the installer.

    Single source for the command string so the setup hint, the sickbay
    check, and any future caller all render the same invocation.
    """
    return f"sudo bash {install_script_path()}"


# ---------- Socket context labeling ----------


@contextmanager
def socket_selinux_context(
    selinux_type: str = SELINUX_SOCKET_TYPE,
) -> Iterator[None]:
    """Apply *selinux_type* as the creation context for sockets bound in this block.

    Any ``socket()`` call within the ``with`` body produces a socket
    whose kernel SID is *selinux_type*, enabling ``container_t`` clients
    to ``connectto`` it once the matching policy is installed.  The
    previous context is restored on exit.

    No-op on non-SELinux systems or when ``libselinux.so.1`` is absent.

    Usage::

        with socket_selinux_context():
            sock = socket.socket(AF_UNIX, SOCK_STREAM)
            sock.bind(str(path))
        # socket object now carries terok_socket_t
    """
    if not is_selinux_enabled():
        yield
        return

    context = f"system_u:object_r:{selinux_type}:s0"
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
    """Outcome of :func:`check_status` — the single decision tree behind
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
    """Structured outcome of :func:`check_status`.

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
    :func:`terok_sandbox.config.services_mode` themselves.
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
