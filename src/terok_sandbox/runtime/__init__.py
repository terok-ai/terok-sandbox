# SPDX-FileCopyrightText: 2026 Jiri Vyskocil
# SPDX-License-Identifier: Apache-2.0

"""Container runtime surface — protocol + concrete backends.

Public imports live here; callers should never reach into the backend
modules directly.
"""

from __future__ import annotations

from .krun import FakeKrunTransport, KrunContainer, KrunRuntime, KrunTransport
from .krun_transport import (
    DEFAULT_PORT_ANNOTATION,
    DEFAULT_SSH_HOST,
    DEFAULT_SSH_USER,
    TcpEndpoint,
    TcpSSHTransport,
    port_annotation_resolver,
)
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
    # Krun container handle (krun-aware login_command override)
    "KrunContainer",
    # Krun transport — protocol, fake (test), real (TCP-via-passt SSH)
    "FakeKrunTransport",
    "KrunTransport",
    "TcpEndpoint",
    "TcpSSHTransport",
    "port_annotation_resolver",
    "DEFAULT_PORT_ANNOTATION",
    "DEFAULT_SSH_HOST",
    "DEFAULT_SSH_USER",
    # Error types that remain public
    "GpuConfigError",
]
