# SPDX-FileCopyrightText: 2026 Jiri Vyskocil
# SPDX-License-Identifier: Apache-2.0

"""Explain a rootfs traversal failure only when local storage evidence supports it."""

from __future__ import annotations

import errno
import json
import os
import re
import stat
import subprocess  # nosec B404 - read-only Podman diagnostics
from pathlib import Path

from terok_util import require_host_tool, sanitize_tty

_ROOTFS_DENIED = re.compile(
    r"error preparing rootfs: remount-propagation dst=(/[^\n]+), "
    r"flags=MS_PRIVATE: permission denied"
)
_PROBE_TIMEOUT = 3
_ACCESS_ACL = "system.posix_acl_access"


def storage_permission_hint(stderr: str, container_name: str) -> str:
    """Return a best-effort hint after a remapped-root storage traversal failure.

    Never launch a probe container or modify files. Require the matching OCI
    error, a local rootless store, remapped container root, and a private
    user-owned ancestor with an unmapped group and no access ACL. Unknown
    formats, ACLs and probe failures leave the original error alone.
    """
    match = _ROOTFS_DENIED.search(stderr)
    if match is None or os.geteuid() == 0:
        return ""
    try:
        info = _podman_json("info", "--format", "json")
        host = info["host"]
        if host["security"]["rootless"] is not True or host["serviceIsRemote"] is not False:
            return ""
        root = Path(info["store"]["graphRoot"])
        if not root.is_absolute():
            return ""
        root = root.resolve(strict=True)
        if root not in Path(match[1]).resolve().parents:
            return ""
        mappings = _podman_json(
            "inspect", "--format", "{{json .HostConfig.IDMappings}}", container_name
        )
        if not _root_is_remapped(mappings["UidMap"]):
            return ""
        groups = _host_ranges(host["idMappings"]["gidmap"])
        if not groups:
            return ""
        for directory in (root, *root.parents):
            if _private_unmapped_group(directory, groups):
                return (
                    "\n\nHint: a private ancestor of Podman's storage has a group outside its "
                    f"rootless mapping: {sanitize_tty(str(directory))}. "
                    "This can prevent remapped container root from traversing it. "
                    "Check this directory's intended group and ACLs locally. "
                    "Do not recursively chown/chmod container storage."
                )
    except (OSError, subprocess.SubprocessError, ValueError, KeyError, TypeError, RuntimeError):
        # Diagnostics must never replace the launch error with a probe error.
        return ""
    return ""


def _podman_json(*args: str) -> dict:
    """Read one bounded diagnostic response from the selected host Podman."""
    result = subprocess.run(  # nosec B603 - fixed verbs with a container name, no shell
        [require_host_tool("podman"), *args],
        capture_output=True,
        text=True,
        check=True,
        timeout=_PROBE_TIMEOUT,
    )
    return json.loads(result.stdout)


def _root_is_remapped(entries: list[str]) -> bool:
    """Require container UID zero to differ from the rootless namespace's owner."""
    if not isinstance(entries, list) or any(not isinstance(entry, str) for entry in entries):
        return False
    for entry in entries:
        container_id, parent_id, size = map(int, entry.split(":"))
        if container_id == 0 and size > 0:
            return parent_id > 0
    return False


def _host_ranges(entries: list[dict]) -> list[range]:
    """Read positive host-GID ranges, refusing incomplete mapping evidence."""
    ranges = []
    for entry in entries:
        start, size = entry["host_id"], entry["size"]
        if type(start) is not int or type(size) is not int or start < 0 or size <= 0:
            raise ValueError("Invalid Podman GID mapping")
        ranges.append(range(start, start + size))
    return ranges


def _private_unmapped_group(directory: Path, groups: list[range]) -> bool:
    """Recognize the reproduced mode/group case, not a general permission model."""
    metadata = directory.stat()
    if (
        not stat.S_ISDIR(metadata.st_mode)
        or metadata.st_uid != os.geteuid()
        or metadata.st_mode & stat.S_IXOTH
        or any(metadata.st_gid in group for group in groups)
    ):
        return False
    try:
        os.getxattr(directory, _ACCESS_ACL)
    except OSError as exc:
        return exc.errno == errno.ENODATA
    return False
