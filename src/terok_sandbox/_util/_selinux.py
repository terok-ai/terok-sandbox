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

All functions degrade gracefully on non-SELinux systems.
"""

from __future__ import annotations

import shutil
import subprocess
from contextlib import contextmanager
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
        return any(line.split()[0] == _POLICY_MODULE_NAME for line in result.stdout.splitlines())
    except (FileNotFoundError, subprocess.TimeoutExpired, OSError):
        return False


# ---------- Policy installation ----------


def policy_source_path() -> Path:
    """Return the path to the bundled ``terok_socket.te`` policy source."""
    return Path(str(_resource_files("terok_sandbox.resources.selinux") / "terok_socket.te"))


def install_policy() -> None:
    """Compile and install the ``terok_socket`` SELinux policy module.

    Requires root privileges and ``checkmodule``, ``semodule_package``,
    ``semodule`` on ``PATH``.

    Raises :class:`SystemExit` on missing tools or compilation failure.
    """
    for tool in ("checkmodule", "semodule_package", "semodule"):
        if not shutil.which(tool):
            raise SystemExit(
                f"Required tool '{tool}' not found.\n"
                f"Install: sudo dnf install selinux-policy-devel policycoreutils"
            )

    te_path = policy_source_path()
    if not te_path.is_file():
        raise SystemExit(f"Policy source not found: {te_path}")

    # Compile in a temporary directory next to the source
    mod_path = te_path.with_suffix(".mod")
    pp_path = te_path.with_suffix(".pp")

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

    # Clean up intermediate artifacts
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


# ---------- Socket context management ----------


def _try_setsockcreatecon(context: str | None) -> bool:
    """Attempt to set the socket creation context via libselinux.

    Returns ``True`` if successful, ``False`` if SELinux bindings are
    unavailable or the call fails.
    """
    try:
        import selinux  # type: ignore[import-untyped]  # system package

        selinux.setsockcreatecon(context)
    except (ImportError, OSError):
        return False
    return True


def _try_getsockcreatecon() -> str | None:
    """Read the current socket creation context, or ``None``."""
    try:
        import selinux  # type: ignore[import-untyped]

        _rc, ctx = selinux.getsockcreatecon()
        return ctx
    except (ImportError, OSError):
        return None


@contextmanager
def socket_selinux_context(
    selinux_type: str = SELINUX_SOCKET_TYPE,
) -> Iterator[None]:
    """Context manager that sets the SELinux socket creation context.

    Wraps ``setsockcreatecon()`` so that any ``socket()`` call within
    the block creates a socket with *selinux_type* as its kernel SID.
    Restores the previous context on exit.

    No-op on non-SELinux systems or when ``libselinux-python3`` is absent.

    Usage::

        with socket_selinux_context():
            sock = socket.socket(AF_UNIX, SOCK_STREAM)
            sock.bind(str(path))
        # socket object now carries terok_socket_t — containers can connectto
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
