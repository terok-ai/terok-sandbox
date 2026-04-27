# SPDX-FileCopyrightText: 2026 Jiri Vyskocil
# SPDX-License-Identifier: Apache-2.0

"""Podman implementation of [`terok_clearance.ContainerInspector`][terok_clearance.ContainerInspector].

Exposes [`PodmanInspector`][terok_sandbox.podman.PodmanInspector] (ID → `ContainerInfo`, with
cache + bounded timeout + soft-fail) and [`create_container_inspector`][terok_sandbox.podman.create_container_inspector]
(runtime-neutral factory — picks an inspector that matches whatever
container runtime sandbox is configured for).

The data type `ContainerInfo` lives in terok-clearance because
every clearance consumer reads it; keeping it there avoids forcing
clearance to import back up from sandbox just to name the result of
an inspection.  Sandbox owns the runtime-aware *production* of those
values; clearance owns the *shape*.
"""

from __future__ import annotations

import json
import logging
import shutil
import subprocess  # nosec B404 — podman is a trusted host binary
from types import MappingProxyType
from typing import Any

from terok_clearance import ContainerInfo, ContainerInspector

_log = logging.getLogger(__name__)

#: Upper bound on each ``podman inspect`` invocation.  A slow local
#: podman (busy nft lock, throttled disk) mustn't pin the caller's
#: event loop waiting for metadata that's ultimately best-effort
#: cosmetic (the ID fallback is always good enough).
_INSPECT_TIMEOUT_S = 5


class PodmanInspector:
    """Cached ID → `ContainerInfo` lookup backed by ``podman inspect``.

    Callable: instances act as ``Callable[[str], ContainerInfo]``.  On
    a cache miss it shells out to ``podman inspect --format=json --``
    with a bounded timeout and stores the result.  Soft-fails on
    missing binary / container / malformed JSON by returning an empty
    `ContainerInfo`, so callers keep a usable fallback.

    Cache lifetime is "per instance".  Container names CAN change at
    runtime; callers that need live-rename visibility should call
    [`forget`][terok_sandbox.podman.PodmanInspector.forget] on ``container_exited`` (or rebuild the inspector)
    so the next lookup re-inspects.
    """

    def __init__(self) -> None:
        """Initialise with an empty cache."""
        self._cache: dict[str, ContainerInfo] = {}

    def __call__(self, container_id: str) -> ContainerInfo:
        """Return cached info for *container_id*, or inspect on miss."""
        if not container_id:
            return ContainerInfo()
        if (cached := self._cache.get(container_id)) is not None:
            return cached
        info = self._inspect(container_id)
        self._cache[container_id] = info
        return info

    def forget(self, container_id: str) -> None:
        """Drop *container_id* from the cache — call on container_exited."""
        self._cache.pop(container_id, None)

    @staticmethod
    def _inspect(container_id: str) -> ContainerInfo:
        """Shell out to ``podman inspect`` once; soft-fail on any error."""
        podman = shutil.which("podman")
        if not podman:
            _log.debug("podman not on PATH — inspector unavailable")
            return ContainerInfo()
        try:
            # ``--`` guards against a hostile container_id that starts
            # with a dash being interpreted as a podman flag.  IDs
            # don't naturally begin with one but the callable is a
            # public boundary; be defensive.
            result = subprocess.run(  # nosec B603 — fixed argv, no shell
                [podman, "inspect", "--format=json", "--", container_id],
                check=False,
                capture_output=True,
                text=True,
                timeout=_INSPECT_TIMEOUT_S,
            )
        except (OSError, subprocess.TimeoutExpired) as exc:
            _log.debug("podman inspect failed for %s: %s", container_id, exc)
            return ContainerInfo()
        if result.returncode != 0:
            _log.debug(
                "podman inspect %s returned %d: %s",
                container_id,
                result.returncode,
                result.stderr.strip(),
            )
            return ContainerInfo()
        try:
            records = json.loads(result.stdout)
        except json.JSONDecodeError as exc:
            _log.debug("podman inspect %s returned malformed JSON: %s", container_id, exc)
            return ContainerInfo()
        return _from_inspect(container_id, records)


def _str(obj: Any, key: str) -> str:
    """Return ``obj[key]`` as a string, or ``""`` for missing / non-string values."""
    if not isinstance(obj, dict):
        return ""
    value = obj.get(key)
    return value if isinstance(value, str) else ""


def _dict(obj: Any, key: str) -> dict:
    """Return ``obj[key]`` as a dict, or ``{}`` for missing / non-dict values."""
    if not isinstance(obj, dict):
        return {}
    value = obj.get(key)
    return value if isinstance(value, dict) else {}


def _from_inspect(container_id: str, records: Any) -> ContainerInfo:
    """Build a `ContainerInfo` from a ``podman inspect`` JSON payload."""
    if not isinstance(records, list) or not records:
        return ContainerInfo()
    head = records[0]
    if not isinstance(head, dict):
        return ContainerInfo()
    return ContainerInfo(
        container_id=container_id,
        # Podman prefixes names with '/'.
        name=_str(head, "Name").lstrip("/"),
        state=_str(_dict(head, "State"), "Status"),
        annotations=MappingProxyType(
            {
                k: v
                for k, v in _dict(_dict(head, "Config"), "Annotations").items()
                if isinstance(k, str) and isinstance(v, str)
            }
        ),
    )


def create_container_inspector() -> ContainerInspector:
    """Return a `ContainerInspector` matched to the active runtime.

    The runtime-neutral entry point for anyone (notifier, TUI, future
    diagnostic tools) who needs container introspection without
    knowing which backend sandbox is driving.  Today sandbox runs on
    podman and the factory hands back a [`PodmanInspector`][terok_sandbox.podman.PodmanInspector];
    when a second backend ships (krun, containerd, anything else),
    this is the single switch-site that grows to pick the right one.
    """
    return PodmanInspector()
