# SPDX-FileCopyrightText: 2026 Jiri Vyskocil
# SPDX-License-Identifier: Apache-2.0

"""Doctor CLI verb — run sandbox-level host-side health checks."""

from __future__ import annotations

import subprocess
import sys

from ..config import SandboxConfig
from ._types import CommandDef


def _handle_doctor(*, cfg: SandboxConfig | None = None) -> None:
    """Run sandbox-level health checks and print results.

    Standalone host-side doctor — runs on the host, not inside a
    container.  ``host_side`` checks evaluate via Python APIs; the
    remaining checks shell out to ``probe_cmd``.  Exit code mirrors
    the worst verdict (warn = 1, error = 2).
    """
    from ..doctor import sandbox_doctor_checks
    from ..vault.lifecycle import VaultManager

    if cfg is None:
        cfg = SandboxConfig()
    mgr = VaultManager(cfg)
    checks = sandbox_doctor_checks(
        token_broker_port=mgr.token_broker_port,
        ssh_signer_port=mgr.ssh_signer_port,
        desired_shield_state=None,  # standalone mode — no task context
    )
    worst = "ok"
    markers = {"ok": "ok", "warn": "WARN", "error": "ERROR"}
    for check in checks:
        if check.host_side:
            verdict = check.evaluate(0, "", "")
        elif check.probe_cmd:
            try:
                result = subprocess.run(  # noqa: S603
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
        handler=_handle_doctor,
        group="doctor",
    ),
)


__all__ = ["DOCTOR_COMMANDS"]
