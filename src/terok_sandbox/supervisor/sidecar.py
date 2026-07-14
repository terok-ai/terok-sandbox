# SPDX-FileCopyrightText: 2026 Jiri Vyskocil
# SPDX-License-Identifier: Apache-2.0

"""Sidecar config + per-container path resolution for the supervisor.

The parent supervisor and every service child read the same per-container
sidecar JSON — written once at container-creation time by
[`write_sidecar`][terok_sandbox.launch.write_sidecar] and pinned by the
``terok.sandbox.sidecar`` annotation — so the config schema
([`SidecarConfig`][terok_sandbox.supervisor.sidecar.SidecarConfig]) and
the socket/log layout it resolves to
([`SupervisorPaths`][terok_sandbox.supervisor.sidecar.SupervisorPaths])
live here, imported by both the parent
([`main`][terok_sandbox.supervisor.main]) and the child runners
([`children`][terok_sandbox.supervisor.children]).  Keeping them in their
own module (rather than on ``main``) is what lets a child import the
config vocabulary without importing the parent's process-supervision
logic.
"""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass
from pathlib import Path

_logger = logging.getLogger("terok-supervisor")


@dataclass(frozen=True)
class SidecarConfig:
    """Per-container config the supervisor reads from the sidecar JSON.

    Written once at container-creation time by
    [`write_sidecar`][terok_sandbox.launch.write_sidecar] (terok-executor
    routes through the same writer) and read back by the OCI hook on
    every ``podman start``.  Keyed by container name
    (``<state>/sidecar/<name>.json``); the ``terok.sandbox.sidecar``
    annotation pins the absolute path.  The file persists across
    stop/start cycles — its ports and tokens must keep matching the
    container's immutable env — and is removed at real teardown by
    [`remove_container_state`][terok_sandbox.launch.remove_container_state].
    """

    container_name: str
    ipc_mode: str  # "socket" or "tcp"
    db_path: Path
    runtime_dir: Path
    """``/run/user/<host_uid>/terok/sandbox`` — pinned by the launch path
    because the supervisor cannot re-derive it from inside crun's
    rootless user namespace (its ``os.getuid()`` is 0 there, which
    misroutes generic resolvers to the root-only ``/run/terok``)."""
    scope_id: str | None = None
    project_id: str = ""
    task_id: str = ""
    tcp_port: int | None = None
    """Per-container TCP port for the vault proxy in TCP mode.
    ``None`` in socket mode (the path is derived from the
    container ID, not carried here)."""
    ssh_signer_port: int | None = None
    """Per-container TCP port for the SSH signer in TCP mode.
    ``None`` in socket mode."""
    gate_port: int | None = None
    """Per-container TCP port for the git gate in TCP mode.
    ``None`` in socket mode."""
    gate_base_path: Path | None = None
    """Directory holding the shared per-project bare mirrors
    (``<gate_base_path>/<project_id>.git``).  ``None`` when the gate is
    not wired for this container."""
    gate_token: str | None = None
    """The single token the gate validates.  Travels only via the
    sidecar; ``None`` when the gate is not wired."""
    dossier_path: Path | None = None
    allow_debugger: bool = False
    """Debug mode — the children leave themselves ptrace-able instead of
    clearing the dumpable flag, so a debugger can attach.  ``False`` (fully
    hardened) unless the launch path opted the task in."""


@dataclass(frozen=True)
class SupervisorPaths:
    """Resolved per-container socket / log / pid locations.

    Computed once at supervisor startup from the container ID and the
    runtime/state dirs the sidecar config doesn't carry directly.
    """

    container_id: str
    container_runtime_dir: Path
    """Per-container directory holding ``vault.sock`` and
    ``ssh-agent.sock``.  Keyed on ``container_name`` (which the
    launch path knows before ``podman run`` so it can pre-create
    the dir and bind-mount it as ``/run/terok/`` inside the
    container).  Different containers get different host dirs;
    the in-container view of these sockets is always /run/terok/."""
    vault_socket: Path
    ssh_signer_socket: Path
    gate_socket: Path
    """Per-container git-gate Unix socket inside ``container_runtime_dir``
    (= the in-container ``/run/terok``).  Used only in socket mode; in
    TCP mode the gate binds a loopback port instead."""
    clearance_socket: Path
    events_socket: Path
    """Per-container ingester socket the shield reader and ``shield
    up``/``down`` push raw line-JSON to.  Distinct from
    ``clearance_socket`` (the varlink subscriber socket operator UIs
    glob): the reader speaks line-JSON, not varlink, so the produce and
    subscribe roles need separate sockets."""
    verdict_socket: Path
    control_socket: Path
    log_path: Path

    @classmethod
    def for_container(
        cls,
        container_id: str,
        container_name: str,
        sidecar_path: Path,
        runtime_dir: Path,
    ) -> SupervisorPaths:
        """Build the per-container path bundle.

        Both anchors come from the launch path — neither is re-resolved
        inside the supervisor:

        * *runtime_dir* (``/run/user/<host_uid>/terok/sandbox``) for
          per-container sockets; carried in the sidecar because the
          rootless user namespace makes generic ``is_root``-based
          resolvers (``terok_util.namespace_runtime_dir``) misroute to
          ``/run/terok``.
        * *sidecar_path*'s grandparent for the persistent log file;
          honours whatever ``paths.root`` resolved to when the launch
          path wrote the sidecar.

        Sockets carry the **12-char short container ID** (podman's
        display convention) rather than the full UUID — AF_UNIX's
        ``sun_path`` is 108 bytes including the null terminator, and
        ``<terok-runtime>/clearance/<64-char-uuid>.sock`` lands at or
        past that limit.  Twelve characters of hex give 48 bits of
        entropy, well past the no-collisions-within-one-host bar.
        Logs keep the full UUID because they live on the filesystem
        with no AF_UNIX limit and the full UUID is easier to grep.

        Clearance / verdict / control sockets live at the
        cross-package ``<terok>/`` runtime root (parent of the
        sandbox-namespaced *runtime_dir*) because they're owned by
        terok-clearance semantically and consumed by every package
        that subscribes (terok-shield's NFLOG reader, terok-clearance
        TUI, …).  Sandbox-specific sockets (vault, ssh-agent) live in
        a per-container ``runtime_dir/run/<short_id>/`` directory the
        launch path bind-mounts at ``/run/terok/`` inside the
        container — keeping every container's sockets distinct on
        the host so concurrent containers don't collide.
        """
        short_id = container_id[:12]
        clearance_root = runtime_dir.parent  # <terok>/sandbox/  →  <terok>/
        state_anchor = sidecar_path.parent.parent  # <root>/sidecar/<name>.json → <root>
        # vault + ssh-agent are keyed on container_name (known at
        # launch time, before podman assigns the ID) so the launch
        # path can pre-create the dir and bind-mount it.  Clearance /
        # verdict / control use the 12-char container ID short prefix
        # because they're cross-package (shield's NFLOG reader keys
        # on it too).
        container_runtime_dir = runtime_dir / "run" / container_name
        return cls(
            container_id=container_id,
            container_runtime_dir=container_runtime_dir,
            vault_socket=container_runtime_dir / "vault.sock",
            ssh_signer_socket=container_runtime_dir / "ssh-agent.sock",
            gate_socket=container_runtime_dir / "gate-server.sock",
            clearance_socket=clearance_root / "clearance" / f"{short_id}.sock",
            events_socket=clearance_root / "events" / f"{short_id}.sock",
            verdict_socket=clearance_root / "verdict" / f"{short_id}.sock",
            control_socket=clearance_root / "control" / f"{short_id}.sock",
            log_path=state_anchor / "logs" / f"{container_id}.log",
        )


class _BadSidecar(Exception):
    """A sidecar field failed validation — the reason is already logged."""


def load_sidecar(sidecar_path: Path) -> SidecarConfig | None:
    """Read and parse the sidecar JSON at *sidecar_path*.

    The OCI hook pinned this exact path via the
    ``terok.sandbox.sidecar`` annotation, so the supervisor never
    guesses — it opens the named file directly.  Returns ``None`` on
    any I/O / schema failure; ``run_supervisor`` surfaces that as
    exit-code 2.  The per-field validation lives in helpers that raise
    [`_BadSidecar`][terok_sandbox.supervisor.sidecar._BadSidecar] on the
    first bad value, so this reader stays a flat read → validate → build.
    """
    try:
        with sidecar_path.open(encoding="utf-8") as fh:
            raw = json.load(fh)
    except (OSError, ValueError):
        _logger.exception("sidecar parse failure for %s", sidecar_path)
        return None
    if not isinstance(raw, dict):
        _logger.error("sidecar is not a JSON object: %s", sidecar_path)
        return None
    try:
        return _build_config(raw, sidecar_path)
    except _BadSidecar:
        return None
    except (KeyError, TypeError, ValueError):
        _logger.exception("sidecar schema error in %s", sidecar_path)
        return None


def _build_config(raw: dict, sidecar_path: Path) -> SidecarConfig:
    """Assemble a [`SidecarConfig`][terok_sandbox.supervisor.sidecar.SidecarConfig].

    Each field validator raises [`_BadSidecar`][terok_sandbox.supervisor.sidecar._BadSidecar]
    on a bad value (kwargs evaluate left-to-right, so the first failure
    wins, exactly as the old sequential guards did).
    """
    return SidecarConfig(
        container_name=_safe_container_name(raw, sidecar_path),
        ipc_mode=_checked_ipc_mode(raw, sidecar_path),
        db_path=_required_absolute_path(raw, "db_path", sidecar_path),
        runtime_dir=_required_absolute_path(raw, "runtime_dir", sidecar_path),
        scope_id=str(raw["scope_id"]) if raw.get("scope_id") else None,
        project_id=str(raw.get("project_id") or ""),
        task_id=str(raw.get("task_id") or ""),
        tcp_port=_optional_int(raw, "tcp_port"),
        ssh_signer_port=_optional_int(raw, "ssh_signer_port"),
        gate_port=_optional_int(raw, "gate_port"),
        gate_base_path=_optional_absolute_path(raw, "gate_base_path", sidecar_path),
        gate_token=str(raw["gate_token"]) if raw.get("gate_token") else None,
        dossier_path=_optional_absolute_path(raw, "dossier_path", sidecar_path),
        allow_debugger=bool(raw.get("allow_debugger")),
    )


def _safe_container_name(raw: dict, sidecar_path: Path) -> str:
    """The container name, rejected if empty or not a safe path component.

    It is interpolated into ``runtime_dir/run/<name>`` — a dir the
    supervisor mkdir's, chmod's, and rmtree's — so an absolute name or one
    carrying a separator / parent-dir reference could redirect those
    filesystem operations outside the runtime dir.
    """
    raw_name = raw.get("container_name")
    name = str(raw_name).strip() if raw_name is not None else ""
    if not name:
        _logger.error("sidecar missing required container_name: %s", sidecar_path)
        raise _BadSidecar
    if "/" in name or name in (".", ".."):
        _logger.error(
            "sidecar container_name is not a safe path component, got %r: %s", name, sidecar_path
        )
        raise _BadSidecar
    return name


def _checked_ipc_mode(raw: dict, sidecar_path: Path) -> str:
    """The transport mode, restricted to ``socket`` / ``tcp``."""
    raw_mode = raw.get("ipc_mode")
    mode = "socket" if raw_mode is None else str(raw_mode)
    if mode not in ("socket", "tcp"):
        _logger.error("sidecar ipc_mode must be 'socket' or 'tcp', got %r: %s", mode, sidecar_path)
        raise _BadSidecar
    return mode


def _optional_int(raw: dict, key: str) -> int | None:
    """``int(raw[key])`` when present, else ``None``."""
    return int(raw[key]) if raw.get(key) is not None else None


def _required_absolute_path(raw: dict, key: str, sidecar_path: Path) -> Path:
    """A required absolute path — raise [`_BadSidecar`][terok_sandbox.supervisor.sidecar._BadSidecar] otherwise.

    Refuse a relative path: the supervisor ``rmtree``s ``runtime_dir`` and
    binds sockets under it, so a relative value would resolve against
    whatever cwd the OCI hook spawned us with (typically ``/``).
    """
    path = _require_absolute_path(raw, key, sidecar_path)
    if path is None:
        raise _BadSidecar
    return path


def _optional_absolute_path(raw: dict, key: str, sidecar_path: Path) -> Path | None:
    """An absolute path when the key is present + truthy, else ``None``.

    Present-but-relative is a hard error (same reasoning as
    [`_required_absolute_path`][terok_sandbox.supervisor.sidecar._required_absolute_path]);
    an absent key is fine (``gate_base_path`` / ``dossier_path`` are optional).
    """
    if not raw.get(key):
        return None
    return _required_absolute_path(raw, key, sidecar_path)


def _require_absolute_path(raw: dict, key: str, sidecar_path: Path) -> Path | None:
    """Return ``Path(raw[key])`` if absolute, else log + ``None``."""
    value = Path(str(raw[key]))
    if not value.is_absolute():
        _logger.error(
            "sidecar %s must be an absolute path, got %r: %s", key, str(value), sidecar_path
        )
        return None
    return value
