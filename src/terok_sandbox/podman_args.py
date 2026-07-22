# SPDX-FileCopyrightText: 2026 Jiri Vyskocil
# SPDX-License-Identifier: Apache-2.0

"""Podman argv policy: the flag surface sandbox owns, and its gates.

Sandbox emits a set of ``podman run`` flags itself (naming, networking,
capabilities, its socket mounts) — those are *managed*: caller-supplied
args must never carry them, or they would silently fight the launch
assembly.  This module is the single home for that policy: the
container-side mount-point constants the policy protects, the managed
flag/volume sets, and the validators every freeform-args entry point
runs (``terok-sandbox run`` trailing args, ``run.podman_args`` config).

A foundation leaf by design: [`config_schema`][terok_sandbox.config_schema]
validates ``run.podman_args`` at parse time and
[`launch`][terok_sandbox.launch] enforces the same policy at
launch time — both import from here, so the two can never disagree.
"""

from __future__ import annotations

from collections.abc import Sequence

# Container-side mount point for the host runtime directory (socket mode).
CONTAINER_RUNTIME_DIR = "/run/terok"

# Container-side path where bridge resources are bind-mounted (runtime
# pattern) or `COPY`ed into the image (build-time pattern).  The host
# source is always the package's ``resources/bridges/`` directory.
CONTAINER_BRIDGES_DIR = "/usr/local/share/terok-sandbox/bridges"

# Podman flags sandbox owns and rejects from user-supplied trailing args
# in `run`.  Mirrors terok-shield's set and extends it with ``--userns``
# (the bind-mounted socket UIDs depend on the host UID match) and ``-v``
# targets sandbox manages — those are checked separately because flag
# values, not names, are the collision surface.
SANDBOX_MANAGED_FLAGS = frozenset(
    {
        "--name",
        "--network",
        "--hooks-dir",
        "--annotation",
        "--cap-add",
        "--cap-drop",
        "--userns",
    }
)
_FLAG_ALIASES: dict[str, str] = {"--net": "--network"}

# Volume mount targets sandbox emits.  A user `-v ...:<target>` that
# overlaps any of these — or any path under ``CONTAINER_RUNTIME_DIR``
# itself — would shadow sandbox's sockets/mounts.
_MANAGED_VOLUME_TARGETS = frozenset(
    {
        CONTAINER_BRIDGES_DIR,
        CONTAINER_RUNTIME_DIR,
        f"{CONTAINER_RUNTIME_DIR}/vault.sock",
        f"{CONTAINER_RUNTIME_DIR}/ssh-agent.sock",
        f"{CONTAINER_RUNTIME_DIR}/gate-server.sock",
    }
)

PASSTHROUGH_DENIED_FLAGS = frozenset({"--privileged", "--security-opt"})
"""Flags additionally refused in freeform passthrough args.

Not sandbox-*managed* (sandbox doesn't always emit them) but
isolation-weakening: ``--privileged`` disables the seccomp/cap
confinement wholesale and ``--security-opt`` can drop SELinux labelling
or swap the seccomp profile.  Anything on this list needs a vetted typed
channel (like [`RunSpec.caps`][terok_sandbox.sandbox.RunSpec]), never a
freeform string.
"""


def reject_managed_flags(podman_args: list[str]) -> None:
    """Reject user-supplied flags that sandbox owns.

    Mirrors terok-shield's ``_reject_shield_managed_flags`` and adds
    sandbox-specific entries (e.g. ``--userns``).
    """
    conflicts: set[str] = set()
    for arg in podman_args:
        if not arg.startswith("--"):
            continue
        flag = arg.split("=", 1)[0]
        flag = _FLAG_ALIASES.get(flag, flag)
        if flag in SANDBOX_MANAGED_FLAGS:
            conflicts.add(flag)
    if conflicts:
        raise SystemExit(
            f"Flag(s) managed by terok-sandbox, cannot override: {', '.join(sorted(conflicts))}"
        )


def reject_managed_volumes(podman_args: list[str]) -> None:
    """Reject ``-v host:target`` whose target overlaps a sandbox mount."""
    conflicts: set[str] = set()
    iterator = iter(podman_args)
    for arg in iterator:
        spec: str | None = None
        if arg == "-v" or arg == "--volume":
            spec = next(iterator, None)
        elif arg.startswith("--volume="):
            spec = arg.split("=", 1)[1]
        if not spec:
            continue
        parts = spec.split(":")
        if len(parts) < 2:
            continue
        target = parts[1]
        # Block exact matches plus any path under ``CONTAINER_RUNTIME_DIR``
        # — sandbox owns that whole subtree, so a deeper user mount would
        # still hide a freshly-bound supervisor socket.
        if target in _MANAGED_VOLUME_TARGETS or target.startswith(f"{CONTAINER_RUNTIME_DIR}/"):
            conflicts.add(target)
    if conflicts:
        raise SystemExit(
            "Volume target(s) managed by terok-sandbox, cannot override: "
            f"{', '.join(sorted(conflicts))}"
        )


def validate_passthrough_args(podman_args: Sequence[str]) -> tuple[str, ...]:
    """Validate operator-supplied freeform ``podman run`` args.

    The one gate for config-sourced passthrough (``run.podman_args``):
    rejects sandbox-managed flags, mount targets that would shadow
    sandbox's sockets, and the isolation-weakening
    [`PASSTHROUGH_DENIED_FLAGS`][terok_sandbox.podman_args.PASSTHROUGH_DENIED_FLAGS].
    Raises [`SystemExit`][SystemExit] like its building blocks; returns
    the args as a tuple so callers can feed them straight to
    [`RunSpec.extra_args`][terok_sandbox.sandbox.RunSpec].
    """
    args = list(podman_args)
    reject_managed_flags(args)
    reject_managed_volumes(args)
    denied = {
        flag
        for arg in args
        if arg.startswith("--") and (flag := arg.split("=", 1)[0]) in PASSTHROUGH_DENIED_FLAGS
    }
    if denied:
        raise SystemExit(
            f"Flag(s) not allowed in passthrough args: {', '.join(sorted(denied))} — "
            "isolation-weakening flags need a typed, vetted channel"
        )
    return tuple(args)
