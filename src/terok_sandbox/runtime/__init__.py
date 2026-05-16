# SPDX-FileCopyrightText: 2026 Jiri Vyskocil
# SPDX-License-Identifier: Apache-2.0

"""Container runtime surface — protocol + concrete backends.

Public imports live here; callers should never reach into the backend
modules directly.
"""

from __future__ import annotations

from .krun import FakeKrunTransport, KrunRuntime, KrunTransport
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
    "KrunRuntime",
    "NullRuntime",
    "PodmanRuntime",
    # Krun transport seam (real impl ships separately)
    "FakeKrunTransport",
    "KrunTransport",
    # Error types that remain public
    "GpuConfigError",
]
