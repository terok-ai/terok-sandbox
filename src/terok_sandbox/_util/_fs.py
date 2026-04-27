# SPDX-FileCopyrightText: 2025 Jiri Vyskocil
# SPDX-License-Identifier: Apache-2.0

"""Filesystem helpers for directory creation and writability checks."""

import os
from pathlib import Path


def ensure_dir(path: Path) -> None:
    """Create a directory (and parents) if it doesn't exist."""
    path.mkdir(parents=True, exist_ok=True)


def ensure_dir_writable(path: Path, label: str) -> None:
    """Create *path* if needed and verify it is writable, or exit with an error."""
    try:
        path.mkdir(parents=True, exist_ok=True)
    except Exception as e:
        raise SystemExit(f"{label} directory is not writable: {path} ({e})")
    if not path.is_dir():
        raise SystemExit(f"{label} path is not a directory: {path}")
    if not os.access(path, os.W_OK | os.X_OK):
        uid = os.getuid()
        gid = os.getgid()
        raise SystemExit(
            f"{label} directory is not writable: {path}\n"
            f"Fix permissions for the user running terok-sandbox (uid={uid}, gid={gid}). "
            f"Example: sudo chown -R {uid}:{gid} {path}"
        )


def systemd_user_unit_dir() -> Path:
    """Return the systemd user unit directory, validated against path traversal.

    Refuses to run as root (``euid == 0``) and resolves ``$XDG_CONFIG_HOME``
    to ensure the result stays beneath the user's home directory.

    Raises [`SystemExit`][SystemExit] on validation failure.
    """
    if os.geteuid() == 0:
        raise SystemExit(
            "Refusing to manage user systemd units as root. "
            "Run without sudo or use a system-level unit instead."
        )
    xdg = os.environ.get("XDG_CONFIG_HOME")
    config_home = Path(xdg).resolve() if xdg else Path.home() / ".config"
    home = Path.home().resolve()
    if not str(config_home).startswith(str(home)):
        raise SystemExit(
            f"XDG_CONFIG_HOME ({config_home}) resolves outside the home directory ({home}). "
            "Refusing to install systemd units to a potentially untrusted location."
        )
    return config_home / "systemd" / "user"


def write_sensitive_file(path: Path, content: str) -> bool:
    """Atomically create *path* with mode ``0o600`` and write *content*.

    Returns ``True`` if the file was created, ``False`` if it already existed.
    Parent directories are created with mode ``0o700``.

    Refuses to operate if *path.parent* is a symbolic link — chmod would
    otherwise follow the link target.  Opens the file with ``O_NOFOLLOW``
    so a planted symlink at the final path cannot redirect the write.
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.parent.is_symlink():
        raise RuntimeError(f"Refusing to use symlinked directory for sensitive file: {path.parent}")
    os.chmod(path.parent, 0o700)
    nofollow = getattr(os, "O_NOFOLLOW", 0)
    try:
        fd = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL | nofollow, 0o600)
    except FileExistsError:
        return False
    try:
        os.write(fd, content.encode())
    finally:
        os.close(fd)
    return True
