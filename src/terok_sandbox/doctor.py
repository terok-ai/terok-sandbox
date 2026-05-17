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

from collections.abc import Callable
from dataclasses import dataclass

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
      (e.g. ``make_shield``), then passes resolved state to ``evaluate``.
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
) -> list[DoctorCheck]:
    """Return sandbox-level health checks for in-container diagnostics.

    Args:
        token_broker_port: Token broker TCP port (skip check if ``None``).
        ssh_signer_port: SSH signer TCP port (skip check if ``None``).
        desired_shield_state: Expected shield state from ``shield_desired_state``
            file (``"up"``, ``"down"``, ``"disengaged"``, or ``None`` to skip).

    Returns:
        List of [`DoctorCheck`][terok_sandbox.doctor.DoctorCheck] instances ready for orchestration.
    """
    checks: list[DoctorCheck] = [
        _make_vault_unlocked_check(),
        _make_plaintext_passphrase_warning_check(),
    ]
    if token_broker_port is not None:
        checks.append(_make_token_broker_check(token_broker_port))
    if ssh_signer_port is not None:
        checks.append(_make_ssh_signer_check(ssh_signer_port))
    checks.append(_make_shield_check(desired_shield_state))
    return checks


def _make_vault_unlocked_check() -> DoctorCheck:
    """Verify the credentials-DB passphrase resolves through some tier.

    Host-side check: walks the resolution chain (session-unlock file →
    OS keyring → config-file fallback) and reports an actionable error
    when nothing yields.  The vault daemon won't start without a
    passphrase, so this is the first check operators should see fail.
    """

    def _eval(_rc: int, _stdout: str, _stderr: str) -> CheckVerdict:
        """Walk the resolution chain locally; report the verdict."""
        from .config import SandboxConfig
        from .vault.store.encryption import WrongPassphraseError

        try:
            passphrase = SandboxConfig().resolve_passphrase()
        except WrongPassphraseError as exc:
            return CheckVerdict("error", f"vault tier broken — {exc}")
        if passphrase is not None:
            return CheckVerdict("ok", "credentials-DB passphrase available")
        return CheckVerdict(
            "error",
            "vault is locked — no passphrase available."
            " Run `terok-sandbox vault unlock` (session-unlock)"
            " or `terok-sandbox setup` to provision.",
        )

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
    is absent — the session-file tier dies on the next reboot, so
    "unconfirmed AND session-only" is a genuine ``error`` (you are
    literally one reboot away from losing the vault), while every
    durable tier (keyring, systemd-creds, config) is "only" a ``warn``
    (machine-bound; needs an off-host copy for disaster recovery).

    Intentionally NOT bundled into
    [`sandbox_doctor_checks`][terok_sandbox.doctor.sandbox_doctor_checks]:
    that list is consumed per-container by terok's sickbay, and a
    host-bound recovery check would render once per task.  Top-level
    callers (the ``terok-sandbox doctor`` CLI, terok's host-level
    sickbay row) invoke this factory directly so the check renders
    exactly once.
    """

    def _eval(_rc: int, _stdout: str, _stderr: str) -> CheckVerdict:
        """Resolve marker + tier in one shot; severity escalates on session-only."""
        from terok_sandbox import recovery_status  # noqa: PLC0415

        status = recovery_status()
        if status.acknowledged:
            return CheckVerdict("ok", "recovery key acknowledged")
        if status.urgent:
            return CheckVerdict(
                "error",
                "vault recovery key UNCONFIRMED and the passphrase lives ONLY"
                " in the session-unlock tmpfs file — it will be wiped on the"
                " next reboot and your vault becomes UNRECOVERABLE then."
                " Run `terok-sandbox vault passphrase reveal` NOW and save"
                " the value off-host, or `terok-sandbox vault passphrase"
                " acknowledge` if you already captured it (CI / TUI flow).",
            )
        return CheckVerdict(
            "warn",
            "vault recovery key unconfirmed — every keystore tier is"
            " machine-bound, so a hardware failure strands the vault."
            " Run `terok-sandbox vault passphrase reveal` to view and save"
            " the value off-host, or `terok-sandbox vault passphrase"
            " acknowledge` if you already captured it (CI / TUI flow).",
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


def _make_plaintext_passphrase_warning_check() -> DoctorCheck:
    """Flag a plaintext ``credentials.passphrase`` field in any layered config.

    The visibility hook for sandbox#282: the field is one of the
    supported chain tiers, but it's the *only* tier whose security
    boundary is "operator accepted plaintext on disk".  Surface that
    decision permanently — both in
    [`_handle_vault_status`][terok_sandbox.commands.vault._handle_vault_status]
    and here in doctor — so its visibility doesn't depend on the
    operator running a specific verb.
    """

    def _eval(_rc: int, _stdout: str, _stderr: str) -> CheckVerdict:
        """Walk the config files for ``credentials.passphrase``; warn if found."""
        from .paths import plaintext_passphrase_config_path

        path = plaintext_passphrase_config_path()
        if path is None:
            return CheckVerdict("ok", "no plaintext passphrase configured")
        return CheckVerdict(
            "warn",
            f"vault passphrase stored in plaintext at {path};"
            " accept on-disk plaintext as your trust boundary,"
            " or migrate to keyring/systemd-creds.",
        )

    return DoctorCheck(
        category="vault",
        label="Plaintext passphrase",
        probe_cmd=[],
        evaluate=_eval,
        host_side=True,
        fix_description=(
            "Remove `credentials.passphrase` from config.yml; provision the same value via"
            " the session-unlock file (RAM-backed, cleared on reboot) or a sealed"
            " systemd-creds credential (TPM2 / host-key bound)."
        ),
    )


# ---------------------------------------------------------------------------
# Check factories (in assembly order)
# ---------------------------------------------------------------------------


def _make_token_broker_check(token_broker_port: int) -> DoctorCheck:
    """Check that the token broker is reachable from inside the container."""
    from .vault.daemon.constants import HEALTH_PATH

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
