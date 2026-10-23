# SPDX-FileCopyrightText: 2026 Jiri Vyskocil
# SPDX-License-Identifier: Apache-2.0

"""Container health check protocol and sandbox-level diagnostics.

Defines the shared [`DoctorCheck`][terok_sandbox.doctor.DoctorCheck] / [`CheckVerdict`][terok_sandbox.doctor.CheckVerdict] protocol
used across the terok package chain (sandbox → agent → terok).  Each
package contributes domain-specific checks; the top-level ``terok sickbay``
orchestrates execution inside containers via ``podman exec``.

Sandbox-level checks verify host-side service reachability from within a
container (vault token broker TCP, SSH signer TCP) and shield firewall state.
"""

from __future__ import annotations

import os
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path

# ---------------------------------------------------------------------------
# Shared protocol types — imported by terok-executor and terok
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class CheckVerdict:
    """Result of evaluating a single health check probe."""

    severity: str
    """``"ok"``, ``"warn"``, or ``"error"``."""

    detail: str
    """Human-readable explanation."""

    fixable: bool = False
    """Whether ``fix_cmd`` should be offered to the operator."""


@dataclass(frozen=True)
class DoctorCheck:
    """A single health check to run inside (or against) a container.

    The ``probe_cmd`` is executed via ``podman exec <cname> ...`` by the
    orchestrator.  The ``evaluate`` callable interprets the result.
    If ``fix_cmd`` is set, the orchestrator may offer it when the check
    fails with ``fixable=True``.

    **Dual execution modes:**

    - *Container mode* (``host_side=False``): the orchestrator runs
      ``probe_cmd`` via ``podman exec`` and passes the result to
      ``evaluate``.  The standalone ``doctor`` command runs the same
      ``probe_cmd`` directly via ``subprocess`` on the host.
    - *Host-side mode* (``host_side=True``): the orchestrator bypasses
      ``probe_cmd`` entirely and performs the check via Python APIs
      (e.g. ``ShieldManager``), then passes resolved state to ``evaluate``.
      The standalone ``doctor`` command calls ``evaluate(0, "", "")`` and
      the function performs the check itself or reports a neutral result.
    """

    category: str
    """Grouping key: ``"bridge"``, ``"env"``, ``"mount"``, ``"network"``,
    ``"shield"``, ``"git"``."""

    label: str
    """Human-readable check name shown in output."""

    probe_cmd: list[str]
    """Shell command to run inside the container via ``podman exec``."""

    evaluate: Callable[[int, str, str], CheckVerdict]
    """``(returncode, stdout, stderr) → CheckVerdict``."""

    fix_cmd: list[str] | None = None
    """Optional remediation command for ``podman exec``."""

    fix_description: str = ""
    """Shown to the operator before applying the fix."""

    host_side: bool = False
    """If ``True``, the check runs on the host (not via ``podman exec``).
    The orchestrator calls ``evaluate(0, "", "")`` and the evaluate
    function performs the host-side check itself."""


# ---------------------------------------------------------------------------
# Sandbox-level check assembly
# ---------------------------------------------------------------------------


def sandbox_doctor_checks(
    *,
    token_broker_port: int | None = None,
    ssh_signer_port: int | None = None,
    desired_shield_state: str | None = None,
    container_id: str | None = None,
    container_name: str | None = None,
) -> list[DoctorCheck]:
    """Return sandbox-level health checks for in-container diagnostics.

    Args:
        token_broker_port: Token broker TCP port (skip check if ``None``).
        ssh_signer_port: SSH signer TCP port (skip check if ``None``).
        desired_shield_state: Expected shield state from ``shield_desired_state``
            file (``"up"``, ``"down"``, ``"disengaged"``, or ``None`` to skip).
        container_id: Container ID, for the supervisor-children check.
            Skipped when ``None`` or when *container_name* is missing —
            the children are found by ID and the wiring is read from the
            sidecar, which is keyed by name.
        container_name: Container name, for the same check.

    Returns:
        List of [`DoctorCheck`][terok_sandbox.doctor.DoctorCheck] instances ready for orchestration.
    """
    checks: list[DoctorCheck] = [
        _make_vault_unlocked_check(),
    ]
    if container_id and container_name:
        checks.append(_make_supervisor_children_check(container_id, container_name))
    if token_broker_port is not None:
        checks.append(_make_token_broker_check(token_broker_port))
    if ssh_signer_port is not None:
        checks.append(_make_ssh_signer_check(ssh_signer_port))
    checks.append(_make_shield_check(desired_shield_state))
    return checks


def _make_supervisor_children_check(container_id: str, container_name: str) -> DoctorCheck:
    """Verify every service the sidecar wires has a live supervisor child.

    Host-side check, and the one that names this class of failure before
    its symptoms do.  The parent supervisor outlives its children, so a
    dead child surfaces only as whatever stops working — a refused
    connection to the vault port, an SSH agent that answers nothing —
    while the supervisor's own PID file still reads healthy.  Comparing
    the wired services against the running ones says which child is gone,
    and the supervisor log then says why it went.
    """

    def _eval(_rc: int, _stdout: str, _stderr: str) -> CheckVerdict:
        """Compare the sidecar's wiring against the live children."""
        from ._util._proc import service_children
        from .paths import state_root
        from .supervisor.sidecar import load_sidecar, wired_services

        sidecar_path = state_root() / "sidecar" / f"{container_name}.json"
        sidecar = load_sidecar(sidecar_path) if sidecar_path.is_file() else None
        if sidecar is None:
            return CheckVerdict("ok", "no sidecar — this container has no supervisor to check")
        expected = set(wired_services(sidecar))
        running = set(service_children(container_id))
        missing = sorted(expected - running)
        if not missing:
            return CheckVerdict("ok", f"{len(expected)} supervisor service(s) running")
        log = state_root() / "logs" / f"{container_id}.log"
        return CheckVerdict(
            "error",
            f"supervisor children not running: {', '.join(missing)}."
            f" What they serve is dead in this container — see {log} for why they exited",
        )

    return DoctorCheck(
        category="supervisor",
        label="Supervisor children",
        probe_cmd=[],
        evaluate=_eval,
        host_side=True,
    )


def _make_vault_unlocked_check() -> DoctorCheck:
    """Verify the passphrase resolves — and that the supervisor can resolve it too.

    Host-side check: walks the resolution chain (systemd-creds → desktop
    keyring → session cache → passphrase-command) and reports an
    actionable error when nothing yields.  The vault and signer children
    do not start without a passphrase, so this is the first check
    operators should see fail.

    Walking the chain here answers it for *this* process.  The cache
    backing follows the supervisor's placement
    ([`session_cache`][terok_sandbox.vault.store.session_cache]), so the
    children read the same backing this process reads; whether they
    actually came up is the supervisor-children check's question.
    """

    def _eval(_rc: int, _stdout: str, _stderr: str) -> CheckVerdict:
        """Walk the resolution chain locally; report the verdict."""
        from .config import SandboxConfig
        from .vault.store.encryption import WrongPassphraseError

        cfg = SandboxConfig()
        try:
            passphrase, tier = cfg.resolve_passphrase_with_source()
        except WrongPassphraseError as exc:
            return CheckVerdict("error", f"vault tier broken — {exc}")
        if passphrase is None:
            return CheckVerdict(
                "error",
                "vault is locked — no passphrase available."
                " Run `terok-sandbox vault unlock` (temporary session cache)"
                " or `terok-sandbox setup` to provision.",
            )
        source = f" via {tier.display_name}" if tier is not None else ""
        return CheckVerdict("ok", f"credentials-DB passphrase available{source}")

    return DoctorCheck(
        category="vault",
        label="Credentials DB passphrase",
        probe_cmd=[],
        evaluate=_eval,
        host_side=True,
        fix_description=(
            "Run `terok-sandbox vault unlock` to provision the passphrase for this session."
        ),
    )


def make_recovery_acknowledged_check() -> DoctorCheck:
    """Warn when the operator hasn't confirmed they saved the recovery key.

    Two severity bands depending on the resolved tier when the marker
    is absent: "unconfirmed AND volatile-only" is an ``error`` because
    the temporary cache is lost at reboot or earlier. Durable tiers
    (desktop keyring, systemd-creds, passphrase-command) get a ``warn``:
    an off-host copy is still needed for disaster recovery.

    Intentionally NOT bundled into
    [`sandbox_doctor_checks`][terok_sandbox.doctor.sandbox_doctor_checks]:
    that list is consumed per-container by terok's sickbay, and a
    host-bound recovery check would render once per task.  Top-level
    callers (the ``terok-sandbox doctor`` CLI, terok's host-level
    sickbay row) invoke this factory directly so the check renders
    exactly once.
    """

    def _eval(_rc: int, _stdout: str, _stderr: str) -> CheckVerdict:
        from ._stage import bold  # noqa: PLC0415
        from .vault.store.recovery import RecoveryStatus  # noqa: PLC0415

        status = RecoveryStatus.load()
        if status.acknowledged:
            return CheckVerdict("ok", "recovery key acknowledged")
        reveal = bold("terok-sandbox vault passphrase reveal")
        ack = bold("terok-sandbox vault passphrase acknowledge")
        if status.volatile_only:
            return CheckVerdict(
                "error",
                "vault recovery key UNCONFIRMED and the passphrase lives ONLY"
                " in the temporary cache (kernel keyring or tmpfs session file)"
                " — lost at reboot or earlier. Without a saved copy, cache loss"
                " makes your vault UNRECOVERABLE."
                f" Run {reveal} NOW and save the value off-host,"
                f" or {ack} if you already captured it.",
            )
        return CheckVerdict(
            "warn",
            "vault recovery key unconfirmed — every keystore tier is"
            " machine-bound, so a hardware failure strands the vault."
            f" Run {reveal} to view and save the value off-host,"
            f" or {ack} if you already captured it.",
        )

    return DoctorCheck(
        category="vault",
        label="Recovery key acknowledged",
        probe_cmd=[],
        evaluate=_eval,
        host_side=True,
        fix_description=(
            "Run `terok-sandbox vault passphrase reveal`, copy the value into"
            " an off-host store (password manager / paper safe), and confirm"
            " when prompted; or run `terok-sandbox vault passphrase acknowledge`"
            " after capturing the value via `--echo-passphrase`."
        ),
    )


#: Public docs page explaining the per-container kernel-keyring leak → EDQUOT.
_KERNEL_KEYRING_DOC_URL = "https://terok-ai.github.io/terok/kernel-keyring/"

#: Warn once the per-uid kernel keyring's key-count *or* byte quota is at
#: least this full.  A healthy host sits far below the edge, where a
#: warning would only be noise.
_KERNEL_KEYRING_QUOTA_WARN_AT = 0.95


def _kernel_keyring_quota() -> tuple[int, int, int, int] | None:
    """Return ``(keys_used, keys_max, bytes_used, bytes_max)`` for this uid.

    Read from ``/proc/key-users`` — the kernel's own per-uid quota
    accounting.  ``None`` when the file is absent (non-Linux, no
    ``CONFIG_KEYS``) or carries no line for the effective uid, so there
    is nothing to account here.
    """
    try:
        rows = Path("/proc/key-users").read_text(encoding="utf-8").splitlines()
    except OSError:
        return None
    prefix = f"{os.geteuid()}:"
    for row in rows:
        if row.lstrip().startswith(prefix):
            # <uid>:  <usage> <nkeys>/<nikeys> <qnkeys>/<maxkeys> <qnbytes>/<maxbytes>
            fields = row.split()
            try:
                keys_used, keys_max = (int(n) for n in fields[3].split("/"))
                bytes_used, bytes_max = (int(n) for n in fields[4].split("/"))
            except (IndexError, ValueError):
                return None
            return keys_used, keys_max, bytes_used, bytes_max
    return None


def make_kernel_keyring_quota_check() -> DoctorCheck:
    """Warn when the per-uid kernel keyring is nearly full.

    The OCI runtime creates a kernel session keyring per container and does not
    reliably reclaim it, so a host that cycles many agent containers
    drifts toward the per-uid key quota (200 keys by default) and then
    fails to launch new ones with a misleading "Disk quota exceeded".
    This surfaces the pressure a step before it bites — and only near
    the edge (``_KERNEL_KEYRING_QUOTA_WARN_AT``), since a healthy host sits far
    below it and a warning there is noise.

    Host-level like
    [`make_recovery_acknowledged_check`][terok_sandbox.doctor.make_recovery_acknowledged_check]:
    the quota is per-uid, not per-container, so it renders exactly once.
    """

    def _eval(_rc: int, _stdout: str, _stderr: str) -> CheckVerdict:
        quota = _kernel_keyring_quota()
        if quota is None:
            return CheckVerdict("ok", "per-uid kernel keyring quota not accounted on this host")
        keys_used, keys_max, bytes_used, bytes_max = quota
        key_frac = keys_used / keys_max if keys_max else 0.0
        byte_frac = bytes_used / bytes_max if bytes_max else 0.0
        if max(key_frac, byte_frac) >= _KERNEL_KEYRING_QUOTA_WARN_AT:
            return CheckVerdict(
                "warn",
                f"kernel keyring {round(max(key_frac, byte_frac) * 100)}% full"
                f" ({keys_used}/{keys_max} keys) — leaked per-container kernel keyrings can block"
                f" new containers with 'Disk quota exceeded'; see {_KERNEL_KEYRING_DOC_URL}",
            )
        return CheckVerdict("ok", f"{keys_used}/{keys_max} keys used (per-uid quota)")

    return DoctorCheck(
        category="host",
        label="Kernel keyring quota",
        probe_cmd=[],
        evaluate=_eval,
        host_side=True,
        fix_description=(
            "Restart the host to reclaim leaked container kernel keyrings, or set"
            " `[containers] keyring = false` in containers.conf to stop the leak"
            f" ({_KERNEL_KEYRING_DOC_URL})."
        ),
    )


# ---------------------------------------------------------------------------
# Check factories (in assembly order)
# ---------------------------------------------------------------------------


def _make_token_broker_check(token_broker_port: int) -> DoctorCheck:
    """Check that the token broker is reachable from inside the container."""
    from .vault.daemon import HEALTH_PATH

    url = f"http://host.containers.internal:{token_broker_port}{HEALTH_PATH}"

    def _eval(rc: int, stdout: str, stderr: str) -> CheckVerdict:
        """Evaluate wget probe exit code."""
        if rc == 0:
            return CheckVerdict("ok", f"token broker reachable at port {token_broker_port}")
        return CheckVerdict(
            "error",
            f"token broker unreachable at {url} — check host token broker status",
        )

    return DoctorCheck(
        category="network",
        label="Token broker (TCP)",
        probe_cmd=["wget", "-q", "--spider", "--timeout=3", url],
        evaluate=_eval,
        fix_description="Not fixable from container — host-side token broker must be running.",
    )


def _make_ssh_signer_check(ssh_signer_port: int) -> DoctorCheck:
    """Check that the SSH signer is reachable from inside the container."""

    def _eval(rc: int, stdout: str, stderr: str) -> CheckVerdict:
        """Evaluate nc probe exit code."""
        if rc == 0:
            return CheckVerdict("ok", f"SSH signer reachable at port {ssh_signer_port}")
        return CheckVerdict(
            "error",
            f"SSH signer unreachable at port {ssh_signer_port} — check host SSH signer",
        )

    return DoctorCheck(
        category="network",
        label="SSH signer (TCP)",
        probe_cmd=[
            "bash",
            "-c",
            f"echo | nc -w2 host.containers.internal {ssh_signer_port}",
        ],
        evaluate=_eval,
        fix_description="Not fixable from container — host-side SSH signer must be running.",
    )


def _make_shield_check(desired_state: str | None) -> DoctorCheck:
    """Check that shield firewall state matches operator intent.

    This is a host-side check — the evaluate function calls the shield
    Python API directly rather than probing via ``podman exec``.
    The ``desired_state`` is read from the ``shield_desired_state`` file.
    """
    # Stored as closure state; the orchestrator will call evaluate(0, "", "")
    # and the function performs the actual host-side check.
    _desired = desired_state

    def _eval(rc: int, stdout: str, stderr: str) -> CheckVerdict:
        """Compare actual shield state against desired.

        In orchestrated mode (terok's ``container_doctor``), the caller
        resolves the actual state via ``_check_shield_state()`` and passes
        it as *stdout*.  In standalone mode, *stdout* is empty and the
        function returns a neutral verdict when no desired state is set.
        """
        actual = stdout.strip() if stdout else ""
        if _desired is None:
            return CheckVerdict("ok", "no desired state configured — shield not managed")
        if actual == _desired:
            return CheckVerdict("ok", f"shield state matches desired ({_desired})")
        return CheckVerdict(
            "warn",
            f"shield state mismatch: actual={actual!r}, desired={_desired!r}",
            fixable=True,
        )

    return DoctorCheck(
        category="shield",
        label="Shield state",
        probe_cmd=[],  # host-side check — no podman exec needed
        evaluate=_eval,
        host_side=True,
        fix_cmd=[],  # fix is handled by the orchestrator via shield Python API
        fix_description="Restore shield to desired state (up/down).",
    )
