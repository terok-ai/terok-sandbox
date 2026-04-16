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
import subprocess
from contextlib import contextmanager
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

_POLICY_MODULE_NAME = "terok_socket"
"""Name of the SELinux policy module (matches the .te filename)."""

_ENFORCE_PATH = Path("/sys/fs/selinux/enforce")
"""Kernel sysfs node indicating SELinux enforcement state."""

_LIBSELINUX_SONAME = "libselinux.so.1"
"""Versioned SONAME of libselinux — stable across distro releases."""


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
    """Return ``True`` if the ``terok_socket`` policy module is loaded."""
    try:
        result = subprocess.run(
            ["semodule", "-l"],
            capture_output=True,
            text=True,
            timeout=10,
        )
        return any(
            (tokens := line.split()) and tokens[0] == _POLICY_MODULE_NAME
            for line in result.stdout.splitlines()
            if line.strip()
        )
    except (FileNotFoundError, subprocess.TimeoutExpired, OSError):
        return False


def is_libselinux_available() -> bool:
    """Return ``True`` if ``libselinux.so.1`` can be loaded via ctypes.

    On SELinux-enforcing hosts, a ``False`` return is a silent-failure
    risk: service sockets would bind without ``terok_socket_t`` labeling,
    and container clients would be denied ``connectto`` even when the
    ``terok_socket`` policy module is installed.
    """
    return _load_libselinux() is not None


# ---------- Policy installation ----------


def policy_source_path() -> Path:
    """Return the path to the bundled ``terok_socket.te`` policy source."""
    return Path(str(_resource_files("terok_sandbox.resources.selinux") / "terok_socket.te"))


def install_policy() -> None:
    """Compile and install the ``terok_socket`` SELinux policy module.

    Requires root privileges and ``checkmodule``, ``semodule_package``,
    ``semodule`` on ``PATH``.

    Raises :class:`SystemExit` on missing tools or compilation failure.

    Compilation artifacts (``.mod``, ``.pp``) are written next to the
    ``.te`` source when the directory is writable (editable install),
    otherwise a temporary directory is used (read-only site-packages).
    """
    import tempfile

    for tool in ("checkmodule", "semodule_package", "semodule"):
        if not shutil.which(tool):
            raise SystemExit(
                f"Required tool '{tool}' not found.\n"
                f"Install: sudo dnf install selinux-policy-devel policycoreutils"
            )

    te_path = policy_source_path()
    if not te_path.is_file():
        raise SystemExit(f"Policy source not found: {te_path}")

    try:
        mod_path = te_path.with_suffix(".mod")
        mod_path.touch()
        mod_path.unlink()
        artifact_dir = None
    except PermissionError:
        artifact_dir = tempfile.mkdtemp(prefix="terok-selinux-")
        mod_path = Path(artifact_dir) / te_path.with_suffix(".mod").name
    except OSError as exc:
        import errno

        if exc.errno == errno.EROFS:
            artifact_dir = tempfile.mkdtemp(prefix="terok-selinux-")
            mod_path = Path(artifact_dir) / te_path.with_suffix(".mod").name
        else:
            raise

    pp_path = mod_path.with_suffix(".pp")

    try:
        subprocess.run(
            ["checkmodule", "-M", "-m", "-o", str(mod_path), str(te_path)],
            check=True,
            timeout=30,
        )
        subprocess.run(
            ["semodule_package", "-o", str(pp_path), "-m", str(mod_path)],
            check=True,
            timeout=30,
        )
        subprocess.run(
            ["semodule", "-i", str(pp_path)],
            check=True,
            timeout=60,
        )
    finally:
        if artifact_dir:
            import shutil as _shutil

            _shutil.rmtree(artifact_dir, ignore_errors=True)
        else:
            for artifact in (mod_path, pp_path):
                artifact.unlink(missing_ok=True)


def uninstall_policy() -> None:
    """Remove the ``terok_socket`` SELinux policy module.

    No-op if the module is not installed.  Requires root privileges.
    """
    if is_policy_installed():
        subprocess.run(
            ["semodule", "-r", _POLICY_MODULE_NAME],
            check=True,
            timeout=60,
        )


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
        lib = ctypes.CDLL(_LIBSELINUX_SONAME, use_errno=True)
    except OSError:
        return None
    lib.setsockcreatecon.argtypes = [ctypes.c_char_p]
    lib.setsockcreatecon.restype = ctypes.c_int
    lib.getsockcreatecon.argtypes = [ctypes.POINTER(ctypes.c_char_p)]
    lib.getsockcreatecon.restype = ctypes.c_int
    lib.freecon.argtypes = [ctypes.c_char_p]
    lib.freecon.restype = None
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
