# SPDX-FileCopyrightText: 2026 Jiri Vyskocil
# SPDX-License-Identifier: Apache-2.0

"""Container runtime surface — protocol + concrete backends.

Public imports live here; callers should never reach into the backend
modules directly.
"""

from __future__ import annotations

from .gpu import (
    GPU_VENDORS,
    GpuConfigError,
    GpuSelector,
    GpuVendor,
    check_gpu_available,
    detect_gpu_vendors,
    normalize_gpus,
)
from .krun import FakeKrunTransport, KrunContainer, KrunRuntime, KrunTransport
from .krun_transport import (
    DEFAULT_GUEST_SSHD_PORT,
    DEFAULT_SSH_HOST,
    DEFAULT_SSH_USER,
    TcpEndpoint,
    TcpSSHTransport,
    podman_port_resolver,
)
from .null import NullRuntime
from .podman import (
    ContainerEvent,
    PodmanEventStream,
    PodmanRuntime,
)
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
    # Podman event stream (push-based container-state companion)
    "ContainerEvent",
    "PodmanEventStream",
    # Krun container handle (krun-aware login_command override)
    "KrunContainer",
    # Krun transport — protocol, fake (test), real (TCP-via-passt SSH)
    "FakeKrunTransport",
    "KrunTransport",
    "TcpEndpoint",
    "TcpSSHTransport",
    "podman_port_resolver",
    "DEFAULT_GUEST_SSHD_PORT",
    "DEFAULT_SSH_HOST",
    "DEFAULT_SSH_USER",
    # Error types that remain public
    "GpuConfigError",
    # GPU passthrough — selector types, probes, normalization
    "GPU_VENDORS",
    "GpuSelector",
    "GpuVendor",
    "check_gpu_available",
    "detect_gpu_vendors",
    "normalize_gpus",
]
