# SPDX-FileCopyrightText: 2026 Jiri Vyskocil
# SPDX-License-Identifier: Apache-2.0

"""Container runtime surface — protocol + concrete backends.

Public imports live here; callers should never reach into the backend
modules directly.
"""

from __future__ import annotations

from .krun import FakeKrunTransport, KrunRuntime, KrunTransport
from .krun_transport import (
    DEFAULT_CID_ANNOTATION,
    DEFAULT_SSH_USER,
    DEFAULT_VSOCK_SSHD_PORT,
    VsockEndpoint,
    VsockSSHTransport,
    podman_annotation_resolver,
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
    # Krun transport — protocol, fake (test), real (vsock-SSH)
    "FakeKrunTransport",
    "KrunTransport",
    "VsockEndpoint",
    "VsockSSHTransport",
    "podman_annotation_resolver",
    "DEFAULT_CID_ANNOTATION",
    "DEFAULT_SSH_USER",
    "DEFAULT_VSOCK_SSHD_PORT",
    # Error types that remain public
    "GpuConfigError",
]
