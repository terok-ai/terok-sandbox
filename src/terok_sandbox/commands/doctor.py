# SPDX-FileCopyrightText: 2026 Jiri Vyskocil
# SPDX-License-Identifier: Apache-2.0

"""Doctor CLI verb — run sandbox-level host-side health checks."""

from __future__ import annotations

import subprocess  # nosec B404 — doctor probes shell out via subprocess.run — doctor probes (probe_cmd subprocess.run)
import sys
from typing import TYPE_CHECKING

from terok_util import LazyHandler

from ._types import CommandDef

if TYPE_CHECKING:
    from ..config import SandboxConfig


def _handle_doctor(*, cfg: SandboxConfig | None = None) -> None:
    """Run sandbox-level health checks and print results.

    Standalone host-side doctor — runs on the host, not inside a
    container.  ``host_side`` checks evaluate via Python APIs; the
    remaining checks shell out to ``probe_cmd``.  Exit code mirrors
    the worst verdict (warn = 1, error = 2).

    Vault / shield port probes are skipped at this layer: the vault is
    a per-container service composed by the supervisor, so there is no
    host-global port to probe and "no supervisor is running" is a
    valid resting state (no live containers).  Per-container probes
    run from inside the container.
    """
    from ..config import SandboxConfig
    from ..doctor import make_recovery_acknowledged_check, sandbox_doctor_checks
    from ..launch import make_stray_sidecar_check

    if cfg is None:
        cfg = SandboxConfig()
    # The recovery-acknowledged and stray-sidecar checks are host-level
    # concerns (one marker / one sweep per install, not per task), so
    # they live outside ``sandbox_doctor_checks`` to avoid duplicating
    # them per container when terok's sickbay loops over tasks.  The
    # standalone CLI is the natural caller that re-appends them.
    checks = [
        *sandbox_doctor_checks(
            token_broker_port=None,  # per-container — see docstring
            ssh_signer_port=None,
            desired_shield_state=None,  # standalone mode — no task context
        ),
        make_recovery_acknowledged_check(),
        make_stray_sidecar_check(cfg),
    ]
    worst = "ok"
    markers = {"ok": "ok", "warn": "WARN", "error": "ERROR"}
    for check in checks:
        if check.host_side:
            verdict = check.evaluate(0, "", "")
        elif check.probe_cmd:
            try:
                result = subprocess.run(  # noqa: S603  # nosec B603 — argv is a fixed list controlled by this module — argv is a fixed list controlled by this module
                    check.probe_cmd,
                    capture_output=True,
                    text=True,
                    timeout=5,
                )
                verdict = check.evaluate(result.returncode, result.stdout, result.stderr)
            except (FileNotFoundError, subprocess.TimeoutExpired):
                verdict = check.evaluate(1, "", "probe command unavailable or timed out")
        else:
            verdict = check.evaluate(0, "", "")
        tag = markers.get(verdict.severity, verdict.severity)
        print(f"  {check.label} .... {tag} ({verdict.detail})")
        if verdict.severity == "error" or worst == "error":
            worst = "error"
        elif verdict.severity == "warn" or worst == "warn":
            worst = "warn"

    if worst == "error":
        sys.exit(2)
    elif worst == "warn":
        sys.exit(1)


DOCTOR_COMMANDS: tuple[CommandDef, ...] = (
    CommandDef(
        name="doctor",
        help="Run sandbox health checks",
        handler=LazyHandler("terok_sandbox.commands.doctor:_handle_doctor"),
        group="doctor",
    ),
)

#: Per-verb lazy-dispatch entry point resolved by ``commands.COMMANDS``
#: via its ``source`` string (see that module).  Co-located with the
#: registry tuple above so the verb definition stays the single source.
DOCTOR: CommandDef = DOCTOR_COMMANDS[0]


__all__ = ["DOCTOR", "DOCTOR_COMMANDS"]
