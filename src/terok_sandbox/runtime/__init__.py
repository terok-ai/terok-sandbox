# SPDX-FileCopyrightText: 2026 Jiri Vyskocil
# SPDX-License-Identifier: Apache-2.0

"""Container runtime surface — protocol + concrete backends.

Public imports live here; callers should never reach into the backend
modules directly.
"""

from __future__ import annotations

from .null import NullRuntime
from .podman import GpuConfigError, PodmanRuntime
from .protocol import (
    Container,
    ContainerRemoveResult,
    ContainerRuntime,
    ExecResult,
    Image,
    LogStream,
    PortReservation,
)

__all__ = [
    # Protocol surface
    "Container",
    "ContainerRemoveResult",
    "ContainerRuntime",
    "ExecResult",
    "Image",
    "LogStream",
    "PortReservation",
    # Backends
    "NullRuntime",
    "PodmanRuntime",
    # Error types that remain public
    "GpuConfigError",
]
