# SPDX-FileCopyrightText: 2026 Jiri Vyskocil
# SPDX-License-Identifier: Apache-2.0

"""Sidecar config + per-container path resolution for the supervisor.

The parent supervisor and every service child read the same per-container
sidecar JSON — written at container-creation time by
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

The write side lives here too, beside the reader whose rules it has to
satisfy: [`store_sidecar`][terok_sandbox.supervisor.sidecar.store_sidecar]
is the one writer,
[`refresh_clearance`][terok_sandbox.supervisor.sidecar.refresh_clearance]
the one edit a sidecar takes after creation — its ``clearance`` field,
rewritten from current config before a restart — and
[`remove_sidecar`][terok_sandbox.supervisor.sidecar.remove_sidecar] the
one removal.  The three take turns under one lock per sidecar directory,
so none of them lands between another's check and its act.  All stay
stdlib-only like the rest of the module.
"""

from __future__ import annotations

import contextlib
import errno
import json
import logging
import os
import stat
from dataclasses import dataclass, field, replace
from pathlib import Path
from typing import TYPE_CHECKING

from .. import wayland
from ..podman_args import WAYLAND_RUNTIME_SUBDIR, WAYLAND_SOCKET_NAME
from ..vault.context import BrokerContext
from ..wayland import WaylandMode, WaylandSession

if TYPE_CHECKING:
    from collections.abc import Iterator, Mapping

    from ..config_schema import ClearanceMode

_logger = logging.getLogger("terok-supervisor")

#: Every supervisor service, in launch order: verdict before clearance
#: (the hub connects to it), gate before vault (the container clones
#: through the gate first), then vault, signer, and optional Wayland. The runner table in
#: [`children`][terok_sandbox.supervisor.children] keys on these names and
#: takes its launch order from them.
#:
#: The names live here, with the sidecar that decides which of them are
#: wired, because every reader of that decision — the parent supervisor,
#: the post-start check, the doctor — needs the vocabulary without
#: needing the runners.
SERVICE_NAMES: tuple[str, ...] = ("verdict", "clearance", "gate", "vault", "signer", "wayland")

#: The clearance arm: the hub that puts a refusal to the operator and the
#: verdict server that applies the answer.  ``services.clearance: off``
#: leaves both out; the shield refuses without them, and its audit log, kept
#: while ``shield.audit`` is on, is then a refusal's only record.
_CLEARANCE_ARM: frozenset[str] = frozenset({"verdict", "clearance"})

#: The values ``services.clearance`` takes.  Spelled here rather than read
#: off [`ClearanceMode`][terok_sandbox.config_schema.ClearanceMode], whose
#: module imports pydantic: this one stays stdlib-only because the parent
#: supervisor and every child import it.  A test pins the two together.
_CLEARANCE_MODES: tuple[ClearanceMode, ...] = ("prompt", "mute", "off")

#: Permission bits of every sidecar file: the payload can carry a live
#: gate token, so nobody but the owner may read it.
_SIDECAR_FILE_MODE = 0o600

#: The lock file every change to a sidecar holds — creation, the clearance
#: edit and removal alike — from its check through its act.  One per
#: sidecar directory, created on first use and never removed: a lock file
#: removed while a change waits on it would let two changes each hold "the"
#: lock.  Not a ``*.json`` name, so nothing takes it for a sidecar.
_LOCK_NAME = ".lock"


@dataclass(frozen=True)
class SidecarConfig:
    """Per-container config the supervisor reads from the sidecar JSON.

    Written at container-creation time by
    [`write_sidecar`][terok_sandbox.launch.write_sidecar] (terok-executor
    routes through the same writer) and read back by the OCI hook on
    every ``podman start``.  Keyed by container name
    (``<state>/sidecar/<name>.json``); the ``terok.sandbox.sidecar``
    annotation pins the absolute path.  The file persists across
    stop/start cycles — its ports and tokens must keep matching the
    container's immutable env — and is removed at real teardown by
    [`remove_container_state`][terok_sandbox.launch.remove_container_state].
    Restarts refresh ``clearance`` from current config and the selected
    Wayland session endpoint without changing the saved mount policy.
    """

    broker_context: BrokerContext
    container_name: str
    ipc_mode: str  # "socket" or "tcp"
    db_path: Path
    runtime_dir: Path
    """``/run/user/<host_uid>/terok/sandbox`` — pinned by the launch path
    because the supervisor cannot re-derive it from inside crun's
    rootless user namespace (its ``os.getuid()`` is 0 there, which
    misroutes generic resolvers to the root-only ``/run/terok``)."""
    routes_path: Path | None = None
    """Vault routing table captured by the launch process.

    ``None`` derives ``routes.json`` beside ``db_path`` for sidecars made
    outside the canonical writer.
    """
    vault_systemd_creds_file: Path | None = None
    """Machine-bound passphrase credential captured by the launch process.

    ``None`` derives ``vault.passphrase.cred`` beside ``db_path``.
    """
    credentials_use_desktop_keyring: bool = False
    """Whether the pre-confinement passphrase walk may consult the desktop keyring."""
    credentials_passphrase_command: str | None = None
    """Optional helper the child executes before installing Landlock."""
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
    clearance: ClearanceMode = "prompt"
    """Who is asked about a connection the shield refused (``services.clearance``).

    ``prompt`` runs the verdict and clearance children and puts each
    refusal to the operator; ``mute`` runs them with the hub muting this
    container from its start; ``off`` runs neither (see
    [`wired_services`][terok_sandbox.supervisor.sidecar.wired_services]).
    The shield refuses either way.  ``prompt`` also stands in for a
    sidecar written before the setting existed and for a value that
    cannot be read: a bad value costs the mute, never the supervisor.

    Recorded at creation and refreshed by a restart, so a changed setting
    reaches the supervisor at the task's next start; a running supervisor
    keeps the mode it started with."""
    _resolved_passphrase: str | None = field(default=None, repr=False, compare=False)
    """Ephemeral in-memory passphrase; never present in sidecar JSON."""

    wayland: WaylandSession = field(default_factory=WaylandSession)
    """Optional desktop wiring, refreshed from the invoking session on every start."""


@dataclass(frozen=True)
class SupervisorPaths:
    """Resolved per-container socket / log / pid locations.

    Computed once at supervisor startup from the container ID and the
    runtime/state dirs the sidecar config doesn't carry directly.
    """

    container_id: str
    container_runtime_dir: Path
    """Per-container directory holding isolated service subdirectories.

    Keyed on ``container_name`` (which the launch path knows before
    ``podman run`` so it can pre-create the dir and bind-mount it as
    ``/run/terok/`` inside the container).  Different containers get
    different host dirs.
    """
    vault_socket: Path
    ssh_signer_socket: Path
    gate_socket: Path
    """Per-container git-gate Unix socket in its service-only directory.

    Used only in socket mode; in TCP mode the gate binds a loopback port
    instead.
    """
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

    @property
    def wayland_socket(self) -> Path:
        """The proxy's listener, independent of the other services' transport."""
        return self.container_runtime_dir / WAYLAND_RUNTIME_SUBDIR / WAYLAND_SOCKET_NAME

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
        ``<terok-runtime>/clearance/<64-char-uuid>/hub.sock`` lands at
        or past that limit.  Twelve characters of hex give 48 bits of
        entropy, well past the no-collisions-within-one-host bar.
        Logs keep the full UUID because they live on the filesystem
        with no AF_UNIX limit and the full UUID is easier to grep.

        Clearance / verdict / control sockets live at the
        cross-package ``<terok>/`` runtime root (parent of the
        sandbox-namespaced *runtime_dir*) because they're owned by
        terok-clearance semantically and consumed by every package
        that subscribes (terok-shield's NFLOG reader, terok-clearance
        TUI, …).  Sandbox-specific sockets (vault, ssh-agent) live in
        a per-container ``runtime_dir/run/<container_name>/`` directory
        the launch path bind-mounts at ``/run/terok/`` inside the
        container.  Each service gets a child directory so its Landlock
        write grant cannot replace a sibling's socket.
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
            vault_socket=container_runtime_dir / "vault" / "vault.sock",
            ssh_signer_socket=container_runtime_dir / "signer" / "ssh-agent.sock",
            gate_socket=container_runtime_dir / "gate" / "gate-server.sock",
            clearance_socket=clearance_root / "clearance" / short_id / "hub.sock",
            events_socket=clearance_root / "events" / short_id / "ingester.sock",
            verdict_socket=clearance_root / "verdict" / f"{short_id}.sock",
            control_socket=clearance_root / "control" / f"{short_id}.sock",
            log_path=state_anchor / "logs" / f"{container_id}.log",
        )


def wired_services(cfg: SidecarConfig) -> tuple[str, ...]:
    """The services *cfg* wires, in launch order.

    Three parts of the set are conditional:

    * the gate — a sidecar without a mirror path and a token has no repo
      to serve;
    * the clearance arm, verdict and clearance together — left out when
      ``services.clearance`` is ``off``.  The shield still refuses every
      blocked connection, and its audit log — kept while ``shield.audit``
      is on — is then the only record of the refused destinations; nobody
      is asked about them, and no verdict path is left that could widen the
      firewall.  (``mute`` keeps both: the hub runs, it just starts with
      the container muted.)
    * Wayland — included only when the task requested a proxy and its host
      binary and desktop endpoint were resolved for this start.

    The parent launches no child for a service left out, and the doctor's
    children check expects none.  Vault and signer run for every container.
    """
    unwired: set[str] = set()
    if not (cfg.gate_base_path and cfg.gate_token):
        unwired.add("gate")
    if cfg.clearance == "off":
        unwired |= _CLEARANCE_ARM
    if not cfg.wayland.proxied:
        unwired.add("wayland")
    return tuple(name for name in SERVICE_NAMES if name not in unwired)


class _BadSidecar(Exception):
    """A sidecar field failed validation — the reason is already logged."""


def load_sidecar(sidecar_path: Path) -> SidecarConfig | None:
    """Read and parse the sidecar JSON at *sidecar_path*.

    The OCI hook pinned this exact path via the
    ``terok.sandbox.sidecar`` annotation, so the supervisor never
    guesses — it opens the named file directly.  Returns ``None`` on
    any I/O / schema failure; ``run_supervisor`` surfaces that as
    exit-code 2.  The per-field validation lives in helpers that raise
    ``_BadSidecar`` on the first bad value, so this reader stays a flat
    read → validate → build.
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

    Each field validator raises ``_BadSidecar`` on a bad value (kwargs
    evaluate left-to-right, so the first failure
    wins, exactly as the old sequential guards did).
    """
    return SidecarConfig(
        broker_context=_broker_context(raw),
        container_name=_safe_container_name(raw, sidecar_path),
        ipc_mode=_checked_ipc_mode(raw, sidecar_path),
        db_path=_required_absolute_path(raw, "db_path", sidecar_path),
        runtime_dir=_required_absolute_path(raw, "runtime_dir", sidecar_path),
        routes_path=_optional_absolute_path(raw, "routes_path", sidecar_path),
        vault_systemd_creds_file=_optional_absolute_path(
            raw, "vault_systemd_creds_file", sidecar_path
        ),
        credentials_use_desktop_keyring=_optional_bool(raw, "credentials_use_desktop_keyring"),
        credentials_passphrase_command=_optional_string(raw, "credentials_passphrase_command"),
        scope_id=str(raw["scope_id"]) if raw.get("scope_id") else None,
        project_id=str(raw.get("project_id") or ""),
        task_id=str(raw.get("task_id") or ""),
        tcp_port=_optional_int(raw, "tcp_port"),
        ssh_signer_port=_optional_int(raw, "ssh_signer_port"),
        gate_port=_optional_int(raw, "gate_port"),
        gate_base_path=_optional_absolute_path(raw, "gate_base_path", sidecar_path),
        gate_token=str(raw["gate_token"]) if raw.get("gate_token") else None,
        dossier_path=_optional_absolute_path(raw, "dossier_path", sidecar_path),
        allow_debugger=_optional_bool(raw, "allow_debugger"),
        clearance=_lenient_clearance(raw, sidecar_path),
        wayland=WaylandSession.from_payload(raw.get("wayland")),
    )


def _broker_context(raw: dict) -> BrokerContext:
    """Require the launcher's complete authorization identity before starting services."""
    try:
        context = raw["broker_context"]
        if not isinstance(context, dict):
            raise ValueError("context is not an object")
        return BrokerContext(**context)
    except (KeyError, TypeError, ValueError) as exc:
        _logger.error("Missing or invalid broker context; recreate this legacy task container")
        raise _BadSidecar from exc


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


def _lenient_clearance(raw: dict, sidecar_path: Path) -> ClearanceMode:
    """The clearance mode — ``prompt`` when absent, and when it cannot be read.

    Lenient on purpose, unlike the validated fields around it, and never
    ``_BadSidecar``: a rejected sidecar costs the container its whole
    supervisor — vault, gate and signer with it — and no clearance value
    is worth that.  An absent key is a sidecar written before the setting
    existed, which always prompted.  A value outside the vocabulary is
    logged — by type, like the other field errors, never echoed — and read
    as ``prompt``: the operator is asked rather than silenced by a value
    nobody chose.
    """
    value = raw.get("clearance")
    if value is None:
        return "prompt"
    for mode in _CLEARANCE_MODES:
        if value == mode:
            return mode
    _logger.warning(
        "sidecar clearance must be one of %s; got an unrecognised %s — prompting instead: %s",
        ", ".join(_CLEARANCE_MODES),
        type(value).__name__,
        sidecar_path,
    )
    return "prompt"


def _optional_int(raw: dict, key: str) -> int | None:
    """``int(raw[key])`` when present, else ``None``."""
    return int(raw[key]) if raw.get(key) is not None else None


def _optional_bool(raw: dict, key: str) -> bool:
    """Return a bool field, defaulting absent/null to ``False``."""
    value = raw.get(key)
    if value is None:
        return False
    if not isinstance(value, bool):
        _logger.error("sidecar %s must be a boolean, got %s", key, type(value).__name__)
        raise _BadSidecar
    return value


def _optional_string(raw: dict, key: str) -> str | None:
    """Return an optional string field, rejecting non-string values."""
    value = raw.get(key)
    if value is None:
        return None
    if not isinstance(value, str):
        _logger.error("sidecar %s must be a string, got %s", key, type(value).__name__)
        raise _BadSidecar
    return value or None


def _required_absolute_path(raw: dict, key: str, sidecar_path: Path) -> Path:
    """A required absolute path — raise ``_BadSidecar`` otherwise.

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


def store_sidecar(
    target: Path,
    payload: Mapping[str, object],
    *,
    replacing: os.stat_result | None = None,
) -> None:
    """Create the sidecar at *target*, or atomically replace it, with *payload*.

    The one writer behind both producers of a sidecar —
    [`write_sidecar`][terok_sandbox.launch.write_sidecar] at creation and
    [`refresh_clearance`][terok_sandbox.supervisor.sidecar.refresh_clearance]
    before a restart — so the file's rules are kept in one place:

    * **0600 from the first byte.**  The payload can carry a live gate
      token.  It goes to a temp file ``mkstemp`` creates ``0600`` beside
      *target* — ``O_EXCL`` under a random name, so nothing planted there
      is followed — and ``fchmod`` pins the mode against a restrictive
      umask.  The rename then drops whatever mode an older writer left.
    * **Atomic.**  The temp file is fsync'd, then renamed over *target*.
      The OCI hook and the supervisor re-read the file on every ``podman
      start``; they see the old sidecar or the new one, never a torn one,
      and a write that fails midway leaves the old file as it was.
    * **Never through a symlink.**  Whatever sits at *target* must be a
      regular file.  A symlink planted there (pointing at, say,
      ``~/.ssh/authorized_keys``) is refused rather than replaced.
    * **One change at a time.**  The check of what sits at *target* and the
      rename over it run under the sidecar directory's lock, which every
      change holds from its check through its act — creation, the clearance
      edit and
      [`remove_sidecar`][terok_sandbox.supervisor.sidecar.remove_sidecar]
      alike — so no other change lands in between.  Readers take no lock:
      the rename hands them a whole file either way.
    * **An edit stays an edit.**  *replacing* — the ``fstat`` of the file
      a caller read, and still holds open so its inode number cannot be
      reused — makes the rename land only while *target* is still that
      file, unchanged since: the same inode, with the size and modification
      time it had.  An edit never recreates a sidecar removed since it was
      read, nor overwrites one a relaunch has written since, and a rewrite
      in place by a writer that takes no lock shows in the size or the
      modification time.

    Raises:
        OSError: *target* is not a regular file, is no longer the file
            *replacing* describes, the directory's lock file cannot be
            opened, or the write failed.  *target* is left as it was, and
            no temp file is left behind.
    """
    with _sidecars_locked(target.parent):
        _replace_sidecar(target, payload, replacing)


def _replace_sidecar(
    target: Path, payload: Mapping[str, object], replacing: os.stat_result | None
) -> None:
    """Write *payload* beside *target* and rename it over, the directory's lock already held.

    ``store_sidecar``'s body, split out for ``refresh_clearance``: an edit
    holds the lock across its read as well, and taking it a second time,
    from a second descriptor, would wait on itself.
    """
    # Deferred: the supervisor and its children import this module and never write.
    import tempfile

    fd, tmp_name = tempfile.mkstemp(prefix=f".{target.name}.", suffix=".tmp", dir=target.parent)
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as fh:
            os.fchmod(fh.fileno(), _SIDECAR_FILE_MODE)
            json.dump(payload, fh, indent=2)
            fh.flush()
            os.fsync(fh.fileno())
        _check_replaceable(target, replacing)
        os.replace(tmp_name, target)
    except BaseException:
        with contextlib.suppress(OSError):
            os.unlink(tmp_name)
        raise


def refresh_clearance(sidecar_path: Path, clearance: ClearanceMode) -> None:
    """Rewrite an existing sidecar's ``clearance`` field, and nothing else.

    ``services.clearance`` reaches a supervisor only through the sidecar,
    and every ``podman start`` re-reads it; a restart that should honour
    the current setting rewrites this one field first.  The
    orchestrator-facing verb is
    [`Sandbox.clearance_refresh`][terok_sandbox.Sandbox.clearance_refresh].
    Every other field — the ports and tokens that must keep matching the
    container's immutable env, the paths, the task identity — is written
    back exactly as parsed: same keys, same order, same values.  A sidecar
    written before the setting existed gains the key.

    Strict where the reader is lenient.  The reader takes any stored value
    and falls back to ``prompt``, because a bad value must never cost the
    supervisor; here an unknown *clearance* is refused before anything is
    touched, so nothing outside the vocabulary is ever written.  The file
    is trusted only the way the OCI hook trusts it — a regular file this
    user owns, never reached through a symlink — and replaced under the
    rules of [`store_sidecar`][terok_sandbox.supervisor.sidecar.store_sidecar],
    the writer creation uses.

    The whole edit — read, check, rename — is one change: it holds the
    sidecar directory's lock throughout, so a teardown or a relaunch of the
    same container takes its turn before the edit or after it, never in
    the middle.  A sidecar removed first is refused as missing; a
    relaunch's fresh sidecar, written first, is the one edited.  A missing
    sidecar is refused, never created: the container has no supervisor to
    configure, and writing one would invent wiring it was not launched
    with.

    Raises:
        ValueError: *clearance* is not ``prompt``, ``mute`` or ``off``, or
            the sidecar is not a JSON object.
        FileNotFoundError: No sidecar exists at *sidecar_path*.
        OSError: The sidecar is a symlink, not a regular file, or owned by
            another user; it changed while being rewritten; the directory's
            lock file cannot be opened; or the open or the write failed.
            The sidecar is left as it was in every case.
    """
    if clearance not in _CLEARANCE_MODES:
        raise ValueError(
            f"unknown clearance mode {clearance!r}: expected one of {', '.join(_CLEARANCE_MODES)}"
        )
    with _own_sidecar(sidecar_path) as (payload, identity):
        payload["clearance"] = clearance
        _replace_sidecar(sidecar_path, payload, identity)


def record_wayland(sidecar_path: Path, session: WaylandSession) -> None:
    """Record desktop wiring only after its creation-time mounts were prepared."""
    with _own_sidecar(sidecar_path) as (payload, identity):
        payload["wayland"] = session.payload()
        _replace_sidecar(sidecar_path, payload, identity)


def refresh_wayland(sidecar_path: Path) -> WaylandSession:
    """Capture the current desktop without changing a container's saved mount policy.

    A restart keeps creation-time policy and refreshes only session endpoints.
    Podman cannot retarget a saved direct bind mount or process domain. A
    missing or changed source requires recreation instead of guessing another
    compositor; the creation-time domain choice remains unchanged.
    """
    with _own_sidecar(sidecar_path) as (payload, identity):
        previous = WaylandSession.from_payload(payload.get("wayland"))
        if previous.mode == WaylandMode.NONE and previous.error:
            raise WaylandRecreationRequired(
                f"Wayland recreation required: cannot verify saved desktop access: {previous.error}"
            )
        selected = previous.mode
        session = replace(
            wayland.resolve_session(selected, unavailable=previous.disabled_reason),
            direct_selinux=previous.direct_selinux,
        )
        if selected == WaylandMode.DIRECT and not previous.disabled_reason:
            if session.upstream is None or session.upstream != previous.upstream:
                raise WaylandRecreationRequired(
                    "wl-direct recreation required: the saved compositor socket is missing "
                    "or the current session selects another socket"
                )
        if selected != WaylandMode.NONE:
            payload["wayland"] = session.payload()
            _replace_sidecar(sidecar_path, payload, identity)
        return session


class WaylandRecreationRequired(RuntimeError):
    """A saved socket bind or process domain cannot change without recreation."""


@contextlib.contextmanager
def _own_sidecar(sidecar_path: Path) -> Iterator[tuple[dict[str, object], os.stat_result]]:
    """Open the sidecar at *sidecar_path* for an edit, trusting only a plain file this user owns.

    The sidecar directory's lock comes first and stays held until the edit
    is done, so the read, the check and the rename are one change (see
    ``_sidecars_locked``).  With no sidecar directory there is no lock to
    take and no sidecar either, and that is refused as a missing sidecar.

    Then the rules the OCI hook applies before it hands a sidecar to a
    supervisor, checked on the open descriptor so nothing can be swapped
    in between: never through a symlink (``O_NOFOLLOW``), a regular file
    (``O_NONBLOCK`` lets a planted FIFO be refused instead of waited on),
    owned by this process's effective uid.  Yields the parsed object and
    the ``fstat`` the rewrite must still find at *sidecar_path*.

    The descriptor stays open until the edit is done: an open file keeps
    its inode allocated, so no file swapped in meanwhile can carry the same
    inode number, and the identity check before the rename stays sound.
    """
    with contextlib.ExitStack() as held:
        try:
            held.enter_context(_sidecars_locked(sidecar_path.parent))
        except FileNotFoundError:
            raise _no_sidecar(sidecar_path) from None
        fd = _open_for_edit(sidecar_path)
        held.callback(os.close, fd)
        identity = os.fstat(fd)
        if not stat.S_ISREG(identity.st_mode):
            raise OSError(
                errno.EINVAL, "refusing a sidecar that is not a regular file", str(sidecar_path)
            )
        if identity.st_uid != os.geteuid():
            raise PermissionError(
                errno.EPERM,
                f"refusing a sidecar owned by uid {identity.st_uid}, not by this user",
                str(sidecar_path),
            )
        with open(fd, encoding="utf-8", closefd=False) as fh:
            try:
                raw = json.load(fh)
            except ValueError as exc:  # JSONDecodeError and UnicodeDecodeError alike
                raise ValueError(f"sidecar is not readable JSON ({exc}): {sidecar_path}") from exc
        if not isinstance(raw, dict):
            raise ValueError(f"sidecar is not a JSON object: {sidecar_path}")
        yield raw, identity


def _open_for_edit(sidecar_path: Path) -> int:
    """Open the sidecar read-only — never through a symlink, never waiting on a FIFO.

    A missing sidecar and a symlink get refusals that say so; any other
    failure (``EACCES``, ``ENAMETOOLONG``, …) is raised as the open raised
    it, never relabelled as one of those two.
    """
    flags = os.O_RDONLY | os.O_NOFOLLOW | os.O_NONBLOCK | os.O_CLOEXEC
    try:
        return os.open(sidecar_path, flags)
    except FileNotFoundError:
        raise _no_sidecar(sidecar_path) from None
    except OSError as exc:
        if exc.errno != errno.ELOOP:
            raise
        raise OSError(
            errno.ELOOP, "refusing a sidecar that is a symlink", str(sidecar_path)
        ) from None


def _no_sidecar(sidecar_path: Path) -> FileNotFoundError:
    """The refusal of an edit that finds no sidecar to edit."""
    return FileNotFoundError(
        errno.ENOENT,
        "no sidecar to refresh — one is written only when the container is created",
        str(sidecar_path),
    )


def remove_sidecar(sidecar_path: Path) -> None:
    """Remove the sidecar at *sidecar_path*, if there is one, once no other change is under way.

    The removal beside
    [`store_sidecar`][terok_sandbox.supervisor.sidecar.store_sidecar],
    called by
    [`remove_container_state`][terok_sandbox.launch.remove_container_state]
    at real teardown.  The unlink holds the sidecar directory's lock: an
    edit already under way finishes first and its file goes, and an edit
    that comes later finds no sidecar and is refused.  None can land its
    rename after the unlink and bring the sidecar back — gate token and all
    — for a container that is gone.

    Idempotent: a missing sidecar is a no-op, and so is a missing sidecar
    directory.

    Raises:
        OSError: The directory's lock file cannot be opened, or the unlink
            failed.
    """
    with contextlib.ExitStack() as held:
        try:
            held.enter_context(_sidecars_locked(sidecar_path.parent))
        except FileNotFoundError:
            return  # no sidecar directory, so no sidecar in it
        sidecar_path.unlink(missing_ok=True)


@contextlib.contextmanager
def _sidecars_locked(directory: Path) -> Iterator[None]:
    """Hold *directory*'s sidecar lock: one change to its sidecars at a time.

    Every change to a sidecar checks what sits at its path and then acts on
    it — renames over it, or unlinks it.  Two syscalls, and without the
    lock another change could land between them: a teardown's unlink,
    after which an edit's rename would bring the sidecar back, or a
    relaunch's fresh sidecar, which an edit would overwrite with the old
    container's gate token and ports.  Creation, the clearance edit and
    removal all hold this lock from their check through their act, and
    every producer of a sidecar — sandbox, terok-executor, terok — goes
    through them.  Readers (the OCI hook, the supervisor, the doctor) take
    no lock: the rename hands them the whole old file or the whole new one.

    An advisory ``flock`` on ``_LOCK_NAME``, opened read-write because an
    exclusive lock over NFS needs a writable descriptor, and never through
    a symlink.  Closing the descriptor releases the lock, so a holder that
    dies cannot strand it.

    Raises:
        FileNotFoundError: *directory* does not exist.
        OSError: The lock file cannot be opened — *directory* is not a
            directory, or a symlink sits at the lock file's name.
    """
    # Deferred like ``tempfile``: the supervisor and its children never write.
    import fcntl

    fd = os.open(
        directory / _LOCK_NAME,
        os.O_RDWR | os.O_CREAT | os.O_NOFOLLOW | os.O_CLOEXEC,
        _SIDECAR_FILE_MODE,
    )
    try:
        fcntl.flock(fd, fcntl.LOCK_EX)
        yield
    finally:
        os.close(fd)


def _check_replaceable(target: Path, replacing: os.stat_result | None) -> None:
    """Refuse to rename over anything but the regular file expected at *target*.

    Runs under the directory's lock, just before the rename and after the
    new content is safely on disk, so no change made through this module
    can land between this check and the swap.  The check catches what the
    lock cannot: a change made before the lock was taken, after a caller
    read *replacing* without holding it, and one made by a writer that
    takes no lock — an inode swapped in, or the same inode rewritten in
    place, which shows in its size or its modification time.  Only a
    same-size rewrite within one tick of the filesystem's timestamp clock
    could pass for the file that was read.
    """
    try:
        current: os.stat_result | None = os.lstat(target)
    except FileNotFoundError:
        current = None
    if current is not None and not stat.S_ISREG(current.st_mode):
        link = stat.S_ISLNK(current.st_mode)
        what = "a symlink" if link else "not a regular file"
        raise OSError(
            errno.ELOOP if link else errno.EINVAL,
            f"refusing to replace a sidecar that is {what}",
            str(target),
        )
    if replacing is None:
        return
    if current is None:
        raise FileNotFoundError(
            errno.ENOENT, "sidecar removed during the rewrite — not recreating it", str(target)
        )
    if (current.st_dev, current.st_ino) != (replacing.st_dev, replacing.st_ino):
        raise OSError(
            errno.ESTALE, "sidecar replaced during the rewrite — keeping the new one", str(target)
        )
    if (current.st_size, current.st_mtime_ns) != (replacing.st_size, replacing.st_mtime_ns):
        raise OSError(
            errno.ESTALE,
            "sidecar changed in place during the rewrite — keeping that version",
            str(target),
        )
