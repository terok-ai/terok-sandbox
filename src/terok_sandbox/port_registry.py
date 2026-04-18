# SPDX-FileCopyrightText: 2026 Jiri Vyskocil
# SPDX-License-Identifier: Apache-2.0

"""Shared port registry for multi-user isolation.

Every port allocation — infrastructure services and container web ports —
flows through :class:`PortRegistry`.  Claims are persisted as per-user
JSON files in a shared directory (default ``/tmp/terok-ports/``).  All
users' claim files are read at allocation time to avoid collisions;
socket bind tests verify the port is actually free.

A module-level singleton (:data:`_default`) provides the convenience API
(``claim_port``, ``release_port``, etc.) used by the rest of the stack.
"""

from __future__ import annotations

import json
import os
import pwd
import re
import socket
import stat
import tempfile
import threading
from dataclasses import dataclass
from pathlib import Path

from .paths import port_registry_dir, read_config_section

_DEFAULT_RANGE_START = 18700
_DEFAULT_RANGE_END = 32700

SERVICE_GATE = "gate"
SERVICE_PROXY = "proxy"
SERVICE_SSH_AGENT = "ssh_agent"

_LOCALHOST = "127.0.0.1"
_CLAIMS_FILENAME = "port-claims.json"

_MAX_CLAIM_FILE_BYTES = 16_384  # 16 KiB — plenty for port claims
_MAX_CLAIM_FILES = 256  # sane upper bound for user claim files


def _resolve_port_range() -> range:
    """Resolve the port allocation range from config or defaults.

    Reads ``network.port_range_start`` / ``network.port_range_end`` from
    the layered config.  Falls back to 18700–32700 (~14 000 ports,
    below the Linux ephemeral range at 32768).
    """
    try:
        net = read_config_section("network")
        start = max(1024, min(65535, int(net.get("port_range_start", _DEFAULT_RANGE_START))))
        end = max(start, min(65535, int(net.get("port_range_end", _DEFAULT_RANGE_END))))
    except (ValueError, TypeError):
        start, end = _DEFAULT_RANGE_START, _DEFAULT_RANGE_END
    return range(start, end + 1)


@dataclass(frozen=True)
class ServicePorts:
    """Resolved infrastructure service ports for one terok session."""

    gate: int
    proxy: int
    ssh_agent: int


class PortRegistry:
    """File-based shared port registry for multi-user isolation.

    Each instance manages its own in-memory claim set and shared directory.
    Use the module-level singleton (:data:`_default`) for production code;
    tests can construct isolated instances with a temporary directory.
    """

    def __init__(self, registry_dir: Path, port_range: range) -> None:
        self.registry_dir = registry_dir
        self.port_range = port_range
        self._held: dict[str, int] = {}
        self._service_ports: ServicePorts | None = None
        self._dir_ensured = False
        # Guards ``_held`` and ``_service_ports`` so concurrent callers
        # (e.g. textual worker threads each constructing their own
        # ``SandboxConfig``) cannot race between the "already held?"
        # check and the subsequent claim/write.  Reentrant because
        # ``resolve_service_ports`` calls ``claim`` under the lock.
        self._lock = threading.RLock()

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def resolve_service_ports(
        self,
        gate_pref: int | None,
        proxy_pref: int | None,
        ssh_pref: int | None,
        *,
        gate_explicit: bool = False,
        proxy_explicit: bool = False,
        ssh_explicit: bool = False,
        state_dir: Path | None = None,
    ) -> ServicePorts:
        """Resolve and claim infrastructure ports (cached after first call).

        Each *_pref* is a preferred starting port or ``None`` for auto-allocation.
        When ``*_explicit`` is True the port is a hard pin (``SystemExit`` if busy).

        When *state_dir* is provided, port assignments are persisted across
        restarts.  If a previously saved port cannot be reclaimed, the call
        fails with ``SystemExit`` so the user can resolve the conflict.
        """
        with self._lock:
            return self._resolve_service_ports_locked(
                gate_pref,
                proxy_pref,
                ssh_pref,
                gate_explicit=gate_explicit,
                proxy_explicit=proxy_explicit,
                ssh_explicit=ssh_explicit,
                state_dir=state_dir,
            )

    def _resolve_service_ports_locked(
        self,
        gate_pref: int | None,
        proxy_pref: int | None,
        ssh_pref: int | None,
        *,
        gate_explicit: bool,
        proxy_explicit: bool,
        ssh_explicit: bool,
        state_dir: Path | None,
    ) -> ServicePorts:
        """Lock-holder body of :meth:`resolve_service_ports`."""
        if self._service_ports is not None:
            return self._service_ports

        # Priority chain for port preferences (highest wins):
        #   1. Explicit config (user pinned via config file)
        #   2. Installed systemd units (ground truth for running services)
        #   3. Saved claims file (persisted from a previous session)
        #   4. Fresh auto-allocation
        installed = _read_installed_ports()
        saved = _load_saved_ports(state_dir) if state_dir else {}

        # Track which services got their preference from installed units
        # or saved claims so claim() can trust the port (our service IS
        # the listener).  Both sources are our own prior allocations —
        # installed units are authoritative, saved file is a fallback for
        # non-systemd hosts or cleaned-up units.
        gate_trusted = proxy_trusted = ssh_trusted = False
        if not gate_explicit and gate_pref is None:
            if SERVICE_GATE in installed:
                gate_pref = installed[SERVICE_GATE]
                gate_trusted = True
            elif SERVICE_GATE in saved:
                gate_pref = saved[SERVICE_GATE]
                gate_trusted = True
        if not proxy_explicit and proxy_pref is None:
            if SERVICE_PROXY in installed:
                proxy_pref = installed[SERVICE_PROXY]
                proxy_trusted = True
            elif SERVICE_PROXY in saved:
                proxy_pref = saved[SERVICE_PROXY]
                proxy_trusted = True
        if not ssh_explicit and ssh_pref is None:
            if SERVICE_SSH_AGENT in installed:
                ssh_pref = installed[SERVICE_SSH_AGENT]
                ssh_trusted = True
            elif SERVICE_SSH_AGENT in saved:
                ssh_pref = saved[SERVICE_SSH_AGENT]
                ssh_trusted = True

        claimed: list[str] = []
        try:
            gate = self.claim(SERVICE_GATE, gate_pref, explicit=gate_explicit, trusted=gate_trusted)
            claimed.append(SERVICE_GATE)
            proxy = self.claim(
                SERVICE_PROXY, proxy_pref, explicit=proxy_explicit, trusted=proxy_trusted
            )
            claimed.append(SERVICE_PROXY)
            ssh = self.claim(
                SERVICE_SSH_AGENT, ssh_pref, explicit=ssh_explicit, trusted=ssh_trusted
            )
            claimed.append(SERVICE_SSH_AGENT)
        except SystemExit:
            for key in claimed:
                self.release(key)
            raise

        # Fail-closed: if a saved/installed port was displaced (not
        # explicitly overridden), containers are broken — the user must
        # resolve the conflict.
        expected_ports = {**saved, **installed}  # installed wins over saved
        explicits = {
            SERVICE_GATE: gate_explicit,
            SERVICE_PROXY: proxy_explicit,
            SERVICE_SSH_AGENT: ssh_explicit,
        }
        for key, port in [(SERVICE_GATE, gate), (SERVICE_PROXY, proxy), (SERVICE_SSH_AGENT, ssh)]:
            expected = expected_ports.get(key)
            if expected is not None and port != expected and not explicits[key]:
                for k in (SERVICE_GATE, SERVICE_PROXY, SERVICE_SSH_AGENT):
                    self.release(k)
                source = "installed service" if key in installed else "previous session"
                raise SystemExit(
                    f"Port {expected} ({key}) was assigned by {source} but is now taken.\n"
                    f"Existing containers expect this port.\n\n"
                    f"Options:\n"
                    f"  - Resolve the conflict and retry\n"
                    f"  - Delete {state_dir / _CLAIMS_FILENAME} to force re-allocation\n"
                    f"    (existing containers will need re-creation)"
                )

        if state_dir:
            _save_ports(
                state_dir, {SERVICE_GATE: gate, SERVICE_PROXY: proxy, SERVICE_SSH_AGENT: ssh}
            )

        self._service_ports = ServicePorts(gate=gate, proxy=proxy, ssh_agent=ssh)
        return self._service_ports

    def claim(
        self,
        service_key: str,
        preferred: int | None = None,
        *,
        explicit: bool = False,
        trusted: bool = False,
    ) -> int:
        """Claim one port via the shared file-based registry.

        Reads all users' claim files to avoid collisions, then verifies
        via socket bind that the port is actually free.  The claim is
        persisted to the shared directory so other users can see it.

        When *trusted* is True the port originates from a prior
        allocation by this user — either an installed systemd unit file
        (see :func:`_read_installed_ports`) or a saved claims file
        (``port-claims.json``).  Our own service may be listening, so
        the ``_is_port_free`` bind check is skipped; other-user
        collision checks still apply.  The ``trusted`` flag is set by
        :meth:`resolve_service_ports` during the preference resolution
        phase.
        """
        with self._lock:
            return self._claim_locked(service_key, preferred, explicit=explicit, trusted=trusted)

    def _claim_locked(
        self,
        service_key: str,
        preferred: int | None,
        *,
        explicit: bool,
        trusted: bool,
    ) -> int:
        """Lock-holder body of :meth:`claim`."""
        if service_key in self._held:
            return self._held[service_key]

        self._ensure_dir()
        others = self._read_other_claims()
        own_ports = set(self._held.values())

        if (explicit or trusted) and preferred is not None:
            if not 1 <= preferred <= 65535:
                raise SystemExit(f"Port {preferred} for {service_key} is not a valid port number")
            if preferred in own_ports:
                # Same-key re-claim was caught by the early return above;
                # name the actual holder so the conflict is debuggable.
                holder = next(k for k, v in self._held.items() if v == preferred)
                raise SystemExit(
                    f"Port {preferred} for {service_key} is already held by "
                    f"{holder} in this process"
                )
            if preferred in others:
                raise SystemExit(f"Port {preferred} for {service_key} is claimed by another user")
            # Trusted ports (from installed units) skip the bind check:
            # our own service is the listener.  Explicit config pins
            # still require a genuinely free port.
            if explicit and not trusted and not _is_port_free(preferred):
                raise SystemExit(f"Port {preferred} for {service_key} is unavailable")
            self._held[service_key] = preferred
            self._write_shared_claims()
            return preferred

        # Auto-allocation: clamp preferred to port_range, scan with wrap-around.
        start = preferred if preferred in self.port_range else self.port_range.start
        for candidate in range(start, self.port_range.stop):
            if candidate not in others and candidate not in own_ports and _is_port_free(candidate):
                self._held[service_key] = candidate
                self._write_shared_claims()
                return candidate
        for candidate in range(self.port_range.start, start):
            if candidate not in others and candidate not in own_ports and _is_port_free(candidate):
                self._held[service_key] = candidate
                self._write_shared_claims()
                return candidate

        raise SystemExit(
            f"No free port for {service_key}"
            f" in range {self.port_range.start}–{self.port_range.stop - 1}"
        )

    def release(self, service_key: str) -> None:
        """Release a previously claimed port and update the shared claim file."""
        with self._lock:
            if self._held.pop(service_key, None) is not None:
                self._write_shared_claims(remove={service_key})

    def reset(self) -> None:
        """Clear all in-memory state (for testing)."""
        with self._lock:
            self._held.clear()
            self._service_ports = None
            self._dir_ensured = False

    # ------------------------------------------------------------------
    # Shared claim files (multi-user coordination)
    # ------------------------------------------------------------------

    def _read_other_claims(self) -> set[int]:
        """Read other users' claim files, returning the set of taken ports.

        Own claim file is skipped — same-user port stability is managed
        via the per-user backup (``port-claims.json``) and socket bind.

        Non-regular files (symlinks, FIFOs) and oversized files are skipped
        to defend against hostile entries in the shared directory.
        """
        taken: set[int] = set()
        own = f"{_username()}.json"
        scanned = 0
        try:
            for path in self.registry_dir.iterdir():
                if path.name == own or not path.name.endswith(".json"):
                    continue
                scanned += 1
                if scanned > _MAX_CLAIM_FILES:
                    break
                try:
                    st = path.lstat()
                    if not stat.S_ISREG(st.st_mode) or st.st_size > _MAX_CLAIM_FILE_BYTES:
                        continue
                    data = json.loads(path.read_text())
                    if isinstance(data, dict):
                        taken.update(v for v in data.values() if isinstance(v, int))
                except (OSError, ValueError, TypeError):
                    continue  # malformed or unreadable — skip silently
        except OSError:
            pass
        return taken

    def _write_shared_claims(self, *, remove: set[str] | None = None) -> None:
        """Merge current session claims into the user's shared claim file.

        Uses ``mkstemp`` + ``os.replace`` for atomic, symlink-safe writes.
        Keys in *remove* are deleted from the persisted file (used by release).
        """
        target = self.registry_dir / f"{_username()}.json"
        existing: dict[str, int] = {}
        try:
            if not target.is_symlink():
                existing = json.loads(target.read_text())
                if not isinstance(existing, dict):
                    existing = {}
        except (OSError, ValueError, TypeError):
            pass
        merged = {**existing, **self._held}
        for key in remove or ():
            merged.pop(key, None)
        try:
            fd, tmp_name = tempfile.mkstemp(dir=self.registry_dir, suffix=".tmp")
            try:
                with os.fdopen(fd, "w") as f:
                    json.dump(merged, f)
                os.replace(tmp_name, target)
            except BaseException:
                Path(tmp_name).unlink(missing_ok=True)
                raise
        except OSError:
            pass  # best-effort — worst case, another user may briefly collide

    def _ensure_dir(self) -> None:
        """Create the shared claims directory with sticky-bit permissions.

        Only applies ``chmod 1777`` to newly created directories — never
        widens permissions on pre-existing paths.  Re-validates the path
        after the create/exist race window to catch concurrent symlink
        or non-directory replacements.
        """
        if self._dir_ensured:
            return
        if self.registry_dir.is_symlink():
            raise SystemExit(f"Refusing to use symlinked port registry dir: {self.registry_dir}")
        created = False
        try:
            self.registry_dir.mkdir(parents=True, exist_ok=False)
            created = True
        except FileExistsError:
            # Re-check after race window: concurrent create could be a symlink or non-dir.
            if self.registry_dir.is_symlink():
                raise SystemExit(
                    f"Refusing to use symlinked port registry dir: {self.registry_dir}"
                )
            if not self.registry_dir.is_dir():
                raise SystemExit(
                    f"Port registry path exists but is not a directory: {self.registry_dir}"
                )
        if created:
            try:
                os.chmod(self.registry_dir, 0o1777)  # nosec B103 — shared multi-user dir
            except PermissionError:
                pass  # pre-provisioned by admin — already writable
        self._dir_ensured = True


# ---------------------------------------------------------------------------
# Per-user backup (survives shared dir cleanup / reboot)
# ---------------------------------------------------------------------------


def _load_saved_ports(state_dir: Path) -> dict[str, int]:
    """Load previously saved infra port claims, or empty dict on failure."""
    try:
        data = json.loads((state_dir / _CLAIMS_FILENAME).read_text())
        if not isinstance(data, dict):
            return {}
        return {k: v for k, v in data.items() if isinstance(k, str) and isinstance(v, int)}
    except (OSError, ValueError, TypeError):
        return {}


def _save_ports(state_dir: Path, ports: dict[str, int]) -> None:
    """Persist infra port claims to *state_dir* (best-effort, atomic).

    Only writes if *state_dir* already exists — avoids creating
    directories as a side effect (important for test isolation).
    """
    if not state_dir.is_dir():
        return
    target = state_dir / _CLAIMS_FILENAME
    tmp = target.with_suffix(".tmp")
    try:
        tmp.write_text(json.dumps(ports))
        tmp.rename(target)
    except OSError:
        pass  # non-critical — worst case, ports may change on next restart


# ---------------------------------------------------------------------------
# Installed service introspection
# ---------------------------------------------------------------------------

_LISTEN_RE = re.compile(r"^ListenStream=127\.0\.0\.1:(\d+)", re.MULTILINE)
_SSH_PORT_RE = re.compile(r"--ssh-signer-port[= ](\d+)")

_PROXY_SOCKET_UNIT = "terok-vault.socket"
_PROXY_SERVICE_UNIT = "terok-vault.service"
_GATE_SOCKET_UNIT = "terok-gate.socket"


def _read_installed_ports() -> dict[str, int]:
    """Read ports from installed systemd unit files (ground truth).

    Parses the **rendered** (installed) unit files — not templates — so
    the returned ports are exactly what the running services are
    configured to use.  This is authoritative: if a unit file says
    ``ListenStream=127.0.0.1:18701``, port 18701 belongs to us
    regardless of what ``_is_port_free()`` reports.

    Returns a dict mapping service keys to ports.  Missing or
    unreadable unit files are silently skipped — the caller falls back
    to the saved-claims file or fresh allocation.
    """
    try:
        from ._util import systemd_user_unit_dir

        unit_dir = systemd_user_unit_dir()
    except SystemExit:
        return {}  # running as root or XDG traversal — no user units

    ports: dict[str, int] = {}

    # Gate port: ListenStream in gate socket unit
    gate_socket = unit_dir / _GATE_SOCKET_UNIT
    if port := _parse_listen_port(gate_socket):
        ports[SERVICE_GATE] = port

    # Token broker port: ListenStream in vault socket unit
    proxy_socket = unit_dir / _PROXY_SOCKET_UNIT
    if port := _parse_listen_port(proxy_socket):
        ports[SERVICE_PROXY] = port

    # SSH signer port: --ssh-signer-port in vault service unit
    proxy_service = unit_dir / _PROXY_SERVICE_UNIT
    if port := _parse_ssh_signer_port(proxy_service):
        ports[SERVICE_SSH_AGENT] = port

    return ports


def _parse_listen_port(unit_path: Path) -> int | None:
    """Extract the TCP port from a ``ListenStream=127.0.0.1:PORT`` directive."""
    try:
        text = unit_path.read_text()
    except OSError:
        return None
    match = _LISTEN_RE.search(text)
    if match:
        port = int(match.group(1))
        return port if 1 <= port <= 65535 else None
    return None


def _parse_ssh_signer_port(unit_path: Path) -> int | None:
    """Extract the SSH signer port from the vault service ExecStart line."""
    try:
        text = unit_path.read_text()
    except OSError:
        return None
    match = _SSH_PORT_RE.search(text)
    if match:
        port = int(match.group(1))
        return port if 1 <= port <= 65535 else None
    return None


# ---------------------------------------------------------------------------
# Internals
# ---------------------------------------------------------------------------


def _username() -> str:
    """Return the current OS username from the effective UID (non-spoofable).

    Uses ``pwd.getpwuid`` rather than ``getpass.getuser`` so that the
    result cannot be influenced by ``$USER`` / ``$LOGNAME`` env vars.
    """
    return pwd.getpwuid(os.geteuid()).pw_name


def _is_port_free(port: int) -> bool:
    """Return True if *port* can be bound on localhost.

    .. note:: There is an inherent TOCTOU window between this check and
       the actual service bind.  The file-based claim prevents coordination
       races between terok users; the OS ``bind()`` is the true reservation.
       Holding the socket open is not feasible because the allocating CLI
       process exits before the service process binds.
    """
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        try:
            s.bind((_LOCALHOST, port))
        except OSError:
            return False
    return True


# ---------------------------------------------------------------------------
# Module-level singleton + convenience API
# ---------------------------------------------------------------------------

_default = PortRegistry(port_registry_dir(), _resolve_port_range())

PORT_RANGE = _default.port_range
"""Contiguous range for all auto-allocated ports (infra + web)."""


def resolve_service_ports(
    gate_pref: int | None,
    proxy_pref: int | None,
    ssh_pref: int | None,
    *,
    gate_explicit: bool = False,
    proxy_explicit: bool = False,
    ssh_explicit: bool = False,
    state_dir: Path | None = None,
) -> ServicePorts:
    """Resolve and claim infrastructure ports via the default registry."""
    return _default.resolve_service_ports(
        gate_pref,
        proxy_pref,
        ssh_pref,
        gate_explicit=gate_explicit,
        proxy_explicit=proxy_explicit,
        ssh_explicit=ssh_explicit,
        state_dir=state_dir,
    )


def claim_port(
    service_key: str,
    preferred: int | None = None,
    *,
    explicit: bool = False,
) -> int:
    """Claim one port via the default registry."""
    return _default.claim(service_key, preferred, explicit=explicit)


def release_port(service_key: str) -> None:
    """Release a previously claimed port via the default registry."""
    _default.release(service_key)


def reset_cache() -> None:
    """Clear all in-memory state on the default registry (for testing)."""
    _default.reset()
