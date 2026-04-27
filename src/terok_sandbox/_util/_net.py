# SPDX-FileCopyrightText: 2026 Jiri Vyskocil
# SPDX-License-Identifier: Apache-2.0

"""Network utility helpers shared across lifecycle managers."""

from __future__ import annotations

import os
import socket
import stat
from pathlib import Path


def probe_unix_socket(path: Path, *, timeout: float = 2.0) -> bool:
    """Return ``True`` if the Unix socket at *path* accepts connections."""
    sock = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
    sock.settimeout(timeout)
    try:
        sock.connect(str(path))
        return True
    except (OSError, ConnectionRefusedError):
        return False
    finally:
        sock.close()


def prepare_socket_path(path: Path) -> None:
    """Ensure *path* is ready for ``bind()`` — remove stale sockets, create parents.

    Refuses to unlink non-socket files.  After ``bind()``, the caller should
    call `harden_socket` to restrict permissions.
    """
    try:
        if not stat.S_ISSOCK(path.lstat().st_mode):
            raise RuntimeError(f"Refusing to remove non-socket path: {path}")
        path.unlink()
    except FileNotFoundError:
        pass
    path.parent.mkdir(parents=True, exist_ok=True)


def harden_socket(path: Path) -> None:
    """Restrict a freshly bound socket to owner-only access."""
    os.chmod(path, 0o600)
