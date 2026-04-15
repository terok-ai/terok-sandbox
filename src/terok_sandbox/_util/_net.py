# SPDX-FileCopyrightText: 2026 Jiri Vyskocil
# SPDX-License-Identifier: Apache-2.0

"""Network utility helpers shared across lifecycle managers."""

from __future__ import annotations

import socket
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
