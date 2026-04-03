# SPDX-FileCopyrightText: 2026 Jiri Vyskocil
# SPDX-License-Identifier: Apache-2.0

"""Container health check protocol and sandbox-level diagnostics.

Defines the shared :class:`DoctorCheck` / :class:`CheckVerdict` protocol
used across the terok package chain (sandbox → agent → terok).  Each
package contributes domain-specific checks; the top-level ``terok sickbay``
orchestrates execution inside containers via ``podman exec``.

Sandbox-level checks verify host-side service reachability from within a
container (credential proxy TCP, SSH agent TCP) and shield firewall state.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass

# ---------------------------------------------------------------------------
# Shared protocol types — imported by terok-agent and terok
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
# Sandbox-level checks
# ---------------------------------------------------------------------------


def _make_proxy_check(proxy_port: int) -> DoctorCheck:
    """Check that the credential proxy is reachable from inside the container."""
    url = f"http://host.containers.internal:{proxy_port}/"

    def _eval(rc: int, stdout: str, stderr: str) -> CheckVerdict:
        """Evaluate wget probe exit code."""
        if rc == 0:
            return CheckVerdict("ok", f"proxy reachable at port {proxy_port}")
        return CheckVerdict(
            "error",
            f"proxy unreachable at {url} — check host proxy status",
        )

    return DoctorCheck(
        category="network",
        label="Credential proxy (TCP)",
        probe_cmd=["wget", "-q", "--spider", "--timeout=3", url],
        evaluate=_eval,
        fix_description="Not fixable from container — host-side proxy must be running.",
    )


def _make_ssh_agent_check(ssh_agent_port: int) -> DoctorCheck:
    """Check that the SSH agent proxy is reachable from inside the container."""

    def _eval(rc: int, stdout: str, stderr: str) -> CheckVerdict:
        """Evaluate nc probe exit code."""
        if rc == 0:
            return CheckVerdict("ok", f"SSH agent reachable at port {ssh_agent_port}")
        return CheckVerdict(
            "error",
            f"SSH agent unreachable at port {ssh_agent_port} — check host proxy",
        )

    return DoctorCheck(
        category="network",
        label="SSH agent (TCP)",
        probe_cmd=[
            "bash",
            "-c",
            f"echo | nc -w2 host.containers.internal {ssh_agent_port}",
        ],
        evaluate=_eval,
        fix_description="Not fixable from container — host-side SSH agent must be running.",
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


def sandbox_doctor_checks(
    *,
    proxy_port: int | None = None,
    ssh_agent_port: int | None = None,
    desired_shield_state: str | None = None,
) -> list[DoctorCheck]:
    """Return sandbox-level health checks for in-container diagnostics.

    Args:
        proxy_port: Credential proxy TCP port (skip check if ``None``).
        ssh_agent_port: SSH agent TCP port (skip check if ``None``).
        desired_shield_state: Expected shield state from ``shield_desired_state``
            file (``"up"``, ``"down"``, ``"down_all"``, or ``None`` to skip).

    Returns:
        List of :class:`DoctorCheck` instances ready for orchestration.
    """
    checks: list[DoctorCheck] = []
    if proxy_port is not None:
        checks.append(_make_proxy_check(proxy_port))
    if ssh_agent_port is not None:
        checks.append(_make_ssh_agent_check(ssh_agent_port))
    checks.append(_make_shield_check(desired_shield_state))
    return checks
