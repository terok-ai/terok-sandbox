# SPDX-FileCopyrightText: 2026 Jiri Vyskocil
# SPDX-License-Identifier: Apache-2.0

"""Select a desktop endpoint and restrict the proxy's compositor protocols.

Wayland grants protocol access, not merely permission to draw a window. The
filtered mode exposes only the globals listed here. Direct mode deliberately
retains the compositor's full client interface. Neither mode forwards audio,
X11, the desktop runtime directory, or inherited display file descriptors.
"""

from __future__ import annotations

import json
import logging
import os
import stat
import sys
from dataclasses import dataclass
from enum import StrEnum
from pathlib import Path
from typing import TYPE_CHECKING

from terok_util import find_host_tool, sanitize_tty

from ._stage import red
from .podman_args import CONTAINER_WAYLAND_SOCKET

if TYPE_CHECKING:
    from collections.abc import Callable

_logger = logging.getLogger(__name__)

WAYLAND_ENV_UNSET = ("WAYLAND_SOCKET", "DISPLAY", "XAUTHORITY")
"""Endpoints that must not override the selected Wayland socket."""

ALLOWED_GLOBALS = (
    "wl_compositor",
    "wl_subcompositor",
    "wl_shm",
    "wl_output",
    "wl_seat",
    "xdg_wm_base",
    "zxdg_decoration_manager_v1",
    "wp_viewporter",
    "wp_fractional_scale_manager_v1",
    "wp_presentation",
    "zwp_linux_dmabuf_v1",
    "zwp_linux_explicit_synchronization_v1",
    "wp_linux_drm_syncobj_manager_v1",
    "wp_single_pixel_buffer_manager_v1",
    "zwp_relative_pointer_manager_v1",
    "zwp_pointer_constraints_v1",
    "wp_cursor_shape_manager_v1",
)
"""App surfaces, focused input, and rendering; unknown globals stay denied.

Clipboard/data devices, capture, input injection, DRM leasing, and desktop
management are intentionally absent. FD transport remains wl-mitm's job.
"""

WAYLAND_PROBE = [
    "python3",
    "-c",
    "import socket, struct, sys\n"
    "with socket.socket(socket.AF_UNIX) as peer:\n"
    "    peer.settimeout(2)\n"
    "    peer.connect(sys.argv[1])\n"
    "    peer.sendall(struct.pack('=III', 1, 12 << 16, 2))\n"
    "    with peer.makefile('rb') as replies:\n"
    "        reply = replies.read(12)\n"
    "    if len(reply) != 12 or struct.unpack('=III', reply)[:2] != (2, 12 << 16):\n"
    "        raise RuntimeError('compositor did not acknowledge Wayland sync')\n",
    CONTAINER_WAYLAND_SOCKET,
]
"""Bounded wl_display.sync roundtrip from the container's security domain.

The proxy connects upstream only after accepting a client. A successful
connect alone cannot detect denied or unavailable compositor access.
"""


class WaylandMode(StrEnum):
    """The desktop access explicitly granted to a task."""

    NONE = "none"
    MITM = "wl-mitm"
    DIRECT = "wl-direct"


@dataclass(frozen=True)
class WaylandSession:
    """Host-selected endpoints captured before a supervisor crosses namespaces."""

    mode: WaylandMode = WaylandMode.NONE
    upstream: Path | None = None
    binary: Path | None = None
    error: str | None = None
    disabled_reason: str | None = None
    """Immutable isolation constraint: a restart must not add a mount creation refused."""
    direct_selinux: bool = False
    """Immutable selection of the direct-access process domain at container creation."""

    @property
    def proxied(self) -> bool:
        """Whether this start has enough host wiring to launch the proxy."""
        return self.mode == WaylandMode.MITM and bool(self.upstream and self.binary)

    def payload(self) -> dict[str, str | bool | None]:
        """Return the JSON representation stored in the private task sidecar."""
        return {
            "mode": self.mode.value,
            "upstream": str(self.upstream) if self.upstream else None,
            "binary": str(self.binary) if self.binary else None,
            "error": self.error,
            "disabled_reason": self.disabled_reason,
            "direct_selinux": self.direct_selinux,
        }

    @classmethod
    def from_payload(cls, value: object) -> WaylandSession:
        """Decode saved wiring, retaining malformed data as an explicit error."""
        if value is None:
            return cls()
        try:
            if not isinstance(value, dict):
                raise ValueError("expected an object")
            mode = WaylandMode(value["mode"])
            upstream, binary = (
                Path(value[key]) if value.get(key) else None for key in ("upstream", "binary")
            )
            if any(path is not None and not path.is_absolute() for path in (upstream, binary)):
                raise ValueError("paths must be absolute")
            direct_selinux = value.get("direct_selinux", False)
            if not isinstance(direct_selinux, bool):
                raise ValueError("direct_selinux must be a boolean")
            return cls(
                mode,
                upstream=upstream,
                binary=binary,
                error=value.get("error"),
                disabled_reason=value.get("disabled_reason"),
                direct_selinux=direct_selinux,
            )
        except (KeyError, TypeError, ValueError) as exc:
            return cls(error=f"invalid Wayland sidecar: {exc}")


def resolve_session(mode: WaylandMode, *, unavailable: str | None = None) -> WaylandSession:
    """Resolve the invoking user's current desktop; failure costs only GUI access."""
    if mode == WaylandMode.NONE:
        return WaylandSession()
    try:
        if unavailable:
            raise ValueError(unavailable)
        if os.geteuid() == 0:
            raise ValueError("requires rootless Podman, invoked as the desktop user")
        if any(os.environ.get(name) for name in ("CONTAINER_HOST", "CONTAINER_CONNECTION")):
            unavailable = "Wayland requires local Podman; remote connections are unsupported"
            raise ValueError(unavailable)
        upstream = _session_socket()
        binary = None
        if mode == WaylandMode.MITM:
            executable = find_host_tool("wl-mitm")
            if executable is None:
                raise ValueError("wl-mitm not found; install it on the host and restart the task")
            binary = Path(executable).resolve(strict=True)
        return WaylandSession(mode, upstream, binary)
    except (OSError, ValueError) as exc:
        return WaylandSession(mode, error=str(exc), disabled_reason=unavailable)


def proxy_config(upstream: Path, listener: Path) -> bytes:
    """Build stock wl-mitm TOML with no command hooks or permissive dry run."""
    return (
        f"[socket]\nlisten = {json.dumps(str(listener))}\n"
        f"upstream = {json.dumps(str(upstream))}\n"
        f"[filter]\nallowed_globals = {json.dumps(ALLOWED_GLOBALS)}\n"
        "requests = []\ndry_run = false\n"
    ).encode()


def warn_wayland(
    mode: WaylandMode,
    reason: str,
    *,
    log: Path | None = None,
    observer: Callable[[str], None] | None = None,
) -> None:
    """Make optional desktop failure visible without claiming its cause is SELinux."""
    emit_warning(
        f"WARNING: Wayland ({mode}) unavailable: {reason}. "
        "Task continues without GUI access."
        + (" No direct-socket fallback." if mode == WaylandMode.MITM else "")
        + (f" Supervisor log: {log}" if log else ""),
        observer=observer,
    )


def warn_desktop_access(
    mode: WaylandMode, *, observer: Callable[[str], None] | None = None
) -> None:
    """Keep the host-desktop attack surface explicit in both opt-in modes."""
    if mode == WaylandMode.NONE:
        return
    access = (
        "Direct mode exposes the compositor's full client interface without protocol filtering."
        if mode == WaylandMode.DIRECT
        else "Protocol filtering reduces exposure but does not isolate the host desktop."
    )
    emit_warning(
        f"WARNING: Wayland ({mode}) significantly increases host attack surface. {access}",
        observer=observer,
    )


def emit_warning(message: str, *, observer: Callable[[str], None] | None = None) -> None:
    """Print terminal-safe text; observers receive plain text for their own rendering."""
    message = sanitize_tty(message)
    rendered = message
    if message.startswith(f"WARNING: Wayland ({WaylandMode.DIRECT})"):
        prefix = red("WARNING:", enabled=sys.stderr.isatty() and "NO_COLOR" not in os.environ)
        rendered = message.replace("WARNING:", prefix, 1)
    print(rendered, file=sys.stderr)
    if observer is not None:
        try:
            observer(message)
        except Exception as exc:  # noqa: BLE001 - warning observers never control task startup
            _logger.warning("Wayland warning observer failed: %s", exc)


def _session_socket() -> Path:
    """Require a user-owned socket in a directory other users cannot modify."""
    display = os.environ.get("WAYLAND_DISPLAY")
    if not display:
        raise ValueError("WAYLAND_DISPLAY is unset; launch from your Wayland desktop session")
    path = Path(display)
    if not path.is_absolute():
        if path.name != display or display in {".", ".."}:
            raise ValueError("WAYLAND_DISPLAY must be a socket name or an absolute path")
        runtime_dir = Path(os.environ.get("XDG_RUNTIME_DIR", ""))
        if not runtime_dir.is_absolute():
            raise ValueError("XDG_RUNTIME_DIR must be absolute")
        path = runtime_dir / display
    parent = path.parent.resolve(strict=True)
    parent_info = parent.stat()
    path = parent / path.name
    info = path.lstat()
    if parent_info.st_uid != os.geteuid() or parent_info.st_mode & 0o022:
        raise ValueError("the socket directory must be owned and writable only by the desktop user")
    for ancestor in parent.parents:
        ancestor_info = ancestor.stat()
        if ancestor_info.st_uid not in {0, os.geteuid()} or (
            ancestor_info.st_mode & 0o022 and not ancestor_info.st_mode & stat.S_ISVTX
        ):
            raise ValueError(f"another user can replace a socket directory ancestor: {ancestor}")
    if not stat.S_ISSOCK(info.st_mode) or info.st_uid != os.geteuid():
        raise ValueError(f"not a socket owned by the desktop user: {path}")
    if not os.access(path, os.W_OK):
        raise PermissionError(f"cannot access the session socket: {path}")
    return path
