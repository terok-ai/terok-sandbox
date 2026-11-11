# SPDX-FileCopyrightText: 2026 Jiri Vyskocil
# SPDX-License-Identifier: Apache-2.0

"""Post-start supervision check — did the per-container supervisor come up?

A container start deliberately *survives* a broken supervisor: the OCI
hook soft-fails so a spawn failure never blocks the container, and
terok-shield's egress firewall is fail-closed on its own hook.  What must
not happen is that the degradation stays **silent** — a container whose
vault-routed providers and git gate are all dead, behaving normally at the
shell, with nothing said (issue #458).

[`verify_supervision`][terok_sandbox.supervision.verify_supervision] closes
that gap on the API launch path.  After the container starts, it reads the
same sidecar the supervisor reads, and — when the sidecar declares
socket-mode services — polls briefly for the sockets the supervisor binds.
On timeout it returns a [`SupervisionStatus`][terok_sandbox.supervision.SupervisionStatus]
naming the container, the unbound socket(s), and the hook diary to read;
the caller shouts it but the launch still succeeds (soft-fail preserved),
and an orchestrator may escalate on the structured result.

Both transports are covered.  Socket-mode wiring binds sockets under
``/run/terok``; TCP-mode wiring binds per-container loopback ports, and
``/proc/net/tcp`` says which ports are listening — one read answers for
every service at once, and it contacts none of them (an accept-and-close
probe on each start would put a stray connection in three service logs to
learn what a file already says).  The pure-Python file poll never runs a
subprocess and never raises.

Every service the sidecar wires is checked, the SSH signer included.  A
dead signer costs the task its git access, and it is the child most likely
to die alone: it and the vault are the only two that must open the
credential store, so a passphrase the supervisor cannot resolve takes out
exactly those two and leaves the rest of the bundle looking healthy.

[`outdated_container_warning`][terok_sandbox.supervision.outdated_container_warning]
closes the mirror-image gap.  A container's environment is frozen when it is
created.  A container that outlives a change to the ``/run/terok`` layout
therefore keeps naming sockets this sandbox no longer binds.  The supervisor is
healthy and the bridges start; they connect to nothing.  The symptom arrives
half a minute later as an empty reply, not as a refused connection.  The check
reads the container's protocol stamp and reports the gap.  It never enforces —
an operator can keep using the parts of the container that still work, and
recreate it when that suits them.
"""

from __future__ import annotations

import ipaddress
import stat
import sys
import time
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING

from .supervisor.sidecar import SupervisorPaths, load_sidecar

if TYPE_CHECKING:
    from .config import SandboxConfig
    from .supervisor.sidecar import SidecarConfig

#: How long to wait for the supervisor to bind its sockets before declaring
#: it unresponsive.  The gate binds early and the vault right after, so a
#: healthy supervisor clears this in ~100 ms; the budget only elapses in
#: full when the supervisor genuinely never came up.
_DEFAULT_TIMEOUT_S = 5.0

#: Filesystem poll cadence while waiting for the sockets to appear.
_POLL_INTERVAL_S = 0.1

#: The kernel's TCP tables, one per address family.  Read to learn which
#: loopback ports have a listener in TCP mode.
_PROC_NET_TCP = (Path("/proc/net/tcp"), Path("/proc/net/tcp6"))

#: ``TCP_LISTEN`` in the ``st`` column of those tables.
_TCP_LISTEN = "0A"

MIN_RUNTIME_PROTOCOL = 3
"""Lowest ``TEROK_CONTAINER_PROTOCOL`` whose socket layout this sandbox still binds.

Each service's socket moved into its own ``/run/terok`` subdirectory, which
changed ``TEROK_VAULT_SOCKET``, ``TEROK_SSH_SIGNER_SOCKET`` and
``TEROK_GATE_SOCKET``.  A container stamped below this value predates that
move.  terok-executor stamps the containers, so this value tracks
[`CONTAINER_PROTOCOL`][terok_executor.container.env.CONTAINER_PROTOCOL] there.
"""


@dataclass(frozen=True)
class ServiceEndpoint:
    """One supervisor service and the address its child binds.

    Exactly one of *socket* and *port* is set — the sidecar's transport
    decides which — so the poll can test either kind and the warning can
    name the address the operator will look for.
    """

    service: str
    socket: Path | None = None
    port: int | None = None

    def __str__(self) -> str:
        """``service (address)``, for a warning line."""
        address = self.socket if self.socket is not None else f"127.0.0.1:{self.port}"
        return f"{self.service} ({address})"


@dataclass(frozen=True)
class SupervisionStatus:
    """The result of a post-start supervision check for one container.

    ``missing`` is the subset of ``checked`` endpoints still unbound when
    the poll gave up — empty on a healthy start.  ``skipped`` marks the
    cases with nothing to verify (no sidecar, or a host whose listening
    ports cannot be read), which is *not* a failure.
    """

    container_name: str
    checked: tuple[ServiceEndpoint, ...]
    missing: tuple[ServiceEndpoint, ...]
    hook_log: Path
    skipped: bool = False
    hook_fired: bool | None = None
    """Whether the hook diary has an entry for this container; ``None`` when unknown."""
    supervisor_log: Path | None = None
    """The per-container supervisor log, when the container id was known."""

    @property
    def ok(self) -> bool:
        """``True`` when every required endpoint was bound (or nothing needed checking)."""
        return not self.missing

    def warning(self) -> str:
        """A loud, multi-line operator warning naming the failure and where to look."""
        endpoints = "\n".join(f"warning:     {endpoint}" for endpoint in self.missing)
        return (
            f"warning: container {self.container_name!r} started but these supervisor "
            "services never bound\n"
            f"{endpoints}\n"
            "warning:   the container continues; the listed services are unavailable\n"
            f"warning:   {self._where_to_look()}"
        )

    def _where_to_look(self) -> str:
        """The one line that says what the hook diary knows about this container."""
        if self.hook_fired is None:
            return f"hook diary: {self.hook_log} (absent or empty ⇒ the OCI supervisor hook never fired)"
        if not self.hook_fired:
            return (
                f"hook diary {self.hook_log} has no entry for this container — the OCI "
                "supervisor hook never fired; rerun `terok-sandbox setup`"
            )
        return f"the hook fired; the supervisor log says which child died and why: {self.supervisor_log}"


def verify_supervision(
    cfg: SandboxConfig,
    container_name: str,
    *,
    find_container_id: Callable[[], str | None] | None = None,
    timeout: float = _DEFAULT_TIMEOUT_S,
) -> SupervisionStatus:
    """Poll for the supervisor's sockets after *container_name* has started.

    Reads ``<state>/sidecar/<container_name>.json`` — the same bundle the
    supervisor reads — and, in socket mode, waits up to *timeout* seconds
    for the vault socket (always bound) and the gate socket (when the
    sidecar wired a gate).  Returns a
    [`SupervisionStatus`][terok_sandbox.supervision.SupervisionStatus]; a
    missing socket means the supervisor is not up.  *find_container_id*
    is asked for the container's id only when something is missing; with
    it the status also says whether the hook diary saw this container, so
    the warning points at the log that has the answer.  Never raises and
    never blocks a healthy start beyond the time the sockets take to
    appear.
    """
    sidecar_path = cfg.state_dir / "sidecar" / f"{container_name}.json"
    # The install-global hook diary the OCI hook appends to (mirrors
    # ``ContainerDiagnostics.hook_log``, which is the host-facing SSOT — but
    # that lives in the surface layer, out of reach from here).
    hook_log = cfg.state_dir / "logs" / "hook.log"
    sidecar = load_sidecar(sidecar_path) if sidecar_path.exists() else None
    if sidecar is None:
        return SupervisionStatus(container_name, (), (), hook_log, skipped=True)

    paths = SupervisorPaths.for_container(
        container_id="",  # the service sockets key on the name, not the id
        container_name=container_name,
        sidecar_path=sidecar_path,
        runtime_dir=sidecar.runtime_dir,
    )
    expected = _expected_endpoints(sidecar, paths)
    if not expected:
        return SupervisionStatus(container_name, (), (), hook_log, skipped=True)

    missing = _poll_until_bound(expected, timeout)
    if missing is None:
        # This host will not say which ports listen, so the TCP-mode
        # answer is unknown rather than bad.  Reporting every service
        # missing would be a false alarm on every start.
        return SupervisionStatus(container_name, expected, (), hook_log, skipped=True)
    container_id = find_container_id() if missing and find_container_id else None
    if container_id is None:
        return SupervisionStatus(container_name, expected, missing, hook_log)
    return SupervisionStatus(
        container_name,
        expected,
        missing,
        hook_log,
        hook_fired=_diary_mentions(hook_log, container_id),
        supervisor_log=cfg.state_dir / "logs" / f"{container_id}.log",
    )


def _diary_mentions(hook_log: Path, container_id: str) -> bool | None:
    """Whether the hook diary carries a line tagged with this container.

    The hook tags every line with the container's short id (see
    ``_supervisor_state.set_log_context``), so the tag is the evidence
    that it fired at all.  An absent diary is a diary that never fired;
    one that cannot be read is an unknown, and says so.
    """
    if not hook_log.exists():
        return False
    try:
        return f"[{container_id[:12]}]" in hook_log.read_text(encoding="utf-8")
    except OSError:
        return None


def _expected_endpoints(
    sidecar: SidecarConfig, paths: SupervisorPaths
) -> tuple[ServiceEndpoint, ...]:
    """The addresses this container's supervisor children must bind.

    Vault and signer always; the gate only when the sidecar wired one.
    The clearance and verdict children bind at the cross-package runtime
    root shared by every container, so a bound socket there does not
    belong to this container and is not evidence about it.

    An endpoint whose port the sidecar never recorded is dropped rather
    than guessed at — a TCP-mode sidecar written before the port existed
    would otherwise report a service permanently missing.
    """
    endpoints: list[ServiceEndpoint] = []
    socket_mode = sidecar.ipc_mode == "socket"
    wiring: list[tuple[str, Path, int | None]] = [
        ("vault", paths.vault_socket, sidecar.tcp_port),
        ("signer", paths.ssh_signer_socket, sidecar.ssh_signer_port),
    ]
    if sidecar.gate_base_path and sidecar.gate_token:
        wiring.append(("gate", paths.gate_socket, sidecar.gate_port))
    for service, socket_path, port in wiring:
        if socket_mode:
            endpoints.append(ServiceEndpoint(service, socket=socket_path))
        elif port is not None:
            endpoints.append(ServiceEndpoint(service, port=port))
    if sidecar.wayland.proxied:
        endpoints.append(ServiceEndpoint("wayland", socket=paths.wayland_socket))
    return tuple(endpoints)


def warn_unsupervised(status: SupervisionStatus) -> None:
    """Print the loud warning for a failed check to stderr; no-op when healthy."""
    if status.missing:
        print(status.warning(), file=sys.stderr)


def outdated_container_warning(container_name: str, env: dict[str, str]) -> str | None:
    """Return the warning for a container older than the current socket layout.

    *env* is the environment recorded on the container at creation — see
    [`Container.env`][terok_sandbox.runtime.protocol.Container.env].  Returns
    ``None`` for a current container, and for one with no usable stamp.
    Sidecar tool containers are built from a minimal environment that never
    carried the stamp, so its absence says nothing about age.  Pure: no
    filesystem, no subprocess, never raises.
    """
    try:
        recorded = int(env["TEROK_CONTAINER_PROTOCOL"])
    except (KeyError, ValueError):
        return None
    if recorded >= MIN_RUNTIME_PROTOCOL:
        return None
    return (
        f"warning: container {container_name!r} predates the current /run/terok socket "
        f"layout (protocol {recorded}, this host binds {MIN_RUNTIME_PROTOCOL})\n"
        "warning:   its git gate and vault-routed providers connect nowhere\n"
        "warning:   the bridges still listen, so the symptom is a hang, then an empty reply\n"
        "warning:   recreate the container to pick up the current layout\n"
        "warning:   the rest of the container keeps working until you do"
    )


def _poll_until_bound(
    expected: tuple[ServiceEndpoint, ...], timeout: float
) -> tuple[ServiceEndpoint, ...] | None:
    """Return the endpoints from *expected* still unbound when *timeout* elapses.

    ``None`` when a TCP-mode endpoint cannot be answered at all because
    this host's listening ports are unreadable — an unknown, which the
    caller reports as skipped rather than as a failure.

    A healthy start returns on the first pass and waits for nothing; the
    budget elapses in full only when a service genuinely never binds.
    """
    deadline = time.monotonic() + timeout
    remaining = expected
    while True:
        listening = (
            _listening_ports() if any(e.port is not None for e in remaining) else frozenset()
        )
        if listening is None:
            return None
        remaining = tuple(e for e in remaining if not _is_bound(e, listening))
        if not remaining or time.monotonic() >= deadline:
            return remaining
        time.sleep(_POLL_INTERVAL_S)


def _is_bound(endpoint: ServiceEndpoint, listening: frozenset[int]) -> bool:
    """Is *endpoint* bound — its socket present, or its port listening?

    Neither half proves the binder is *ours*: a stolen port and a
    leftover socket file both read as bound.  The check answers "did
    anything come up here", which is the question a silent start leaves
    open; ownership is the supervisor log's to answer.
    """
    if endpoint.socket is not None:
        return _is_socket(endpoint.socket)
    return endpoint.port in listening


def _is_socket(path: Path) -> bool:
    """``True`` when *path* exists and is an AF_UNIX socket (best-effort)."""
    try:
        return stat.S_ISSOCK(path.stat().st_mode)
    except OSError:
        return False


def _listening_ports() -> frozenset[int] | None:
    """Local ports with a listening TCP socket in this network namespace.

    One read per address family answers for every service at once and
    touches none of them.  The supervisor binds on the host loopback and
    this runs on the host, so both see the same table.

    ``None`` when neither file could be read — the honest answer is that
    this host will not say, not that nothing is listening.
    """
    ports: set[int] = set()
    read_any = False
    for path in _PROC_NET_TCP:
        try:
            table = path.read_text(encoding="ascii", errors="replace")
        except OSError:
            continue
        read_any = True
        for line in table.splitlines()[1:]:
            fields = line.split()
            if len(fields) < 4 or fields[3] != _TCP_LISTEN:
                continue
            address, _, port_hex = fields[1].rpartition(":")
            if not _serves_loopback(address):
                continue
            try:
                ports.add(int(port_hex, 16))
            except ValueError:
                continue
    return frozenset(ports) if read_any else None


def _serves_loopback(address: str) -> bool:
    """Does a listener on this local address answer a container's host loopback?

    The port alone is not the endpoint.  A listener held on one interface
    of the host would otherwise report a supervisor service as bound,
    while the loopback address the container dials has nothing on it.

    Loopback and the wildcard both qualify: the supervisor binds
    ``127.0.0.1``, and a wildcard listener accepts there too.  Every
    other address is a different endpoint that happens to share a port.

    The kernel writes each 32-bit word of the address in host byte
    order, so the words are reversed before the address is read.
    """
    try:
        raw = bytes.fromhex(address)
    except ValueError:
        return False
    packed = b"".join(raw[word : word + 4][::-1] for word in range(0, len(raw), 4))
    try:
        parsed = ipaddress.ip_address(packed)
    except ValueError:
        return False
    if isinstance(parsed, ipaddress.IPv6Address) and parsed.ipv4_mapped:
        parsed = parsed.ipv4_mapped
    return parsed.is_loopback or parsed.is_unspecified


__all__ = [
    "MIN_RUNTIME_PROTOCOL",
    "ServiceEndpoint",
    "SupervisionStatus",
    "outdated_container_warning",
    "verify_supervision",
    "warn_unsupervised",
]
