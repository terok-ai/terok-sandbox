# SPDX-FileCopyrightText: 2026 Jiri Vyskocil
# SPDX-License-Identifier: Apache-2.0

"""Sandbox-owned setup receipt and downward readiness checks."""

from __future__ import annotations

import hashlib
from importlib.metadata import version

from terok_util import (
    SetupCheck,
    SetupReceipt,
    SetupStatus,
    find_host_tool,
    host_tools_source,
    python_identity,
)

from .config import SandboxConfig
from .integrations.shield import ShieldHooks
from .supervisor.install import check_supervisor_hooks

_OWNER = "terok-sandbox"
_RECEIPT_NAME = "setup.json"
_SQLITE_HEADER = b"SQLite format 3\x00"


def check_setup(cfg: SandboxConfig | None = None, *, live: bool = False) -> tuple[SetupCheck, ...]:
    """Check this installation and its Shield dependency without changing state.

    Live checks also discover required host tools in the current environment.
    Parents compose these results, never reading another owner's receipt.
    """
    cfg = cfg or SandboxConfig()
    children = ShieldHooks.check_setup(live=live and not cfg.shield_disabled)
    if cfg.shield_disabled:
        children = tuple(check for check in children if check.status is SetupStatus.DOWNGRADE)
    return (*children, setup_receipt(cfg).check(), *check_artifacts(cfg, live=live))


def setup_receipt(cfg: SandboxConfig) -> SetupReceipt:
    """Describe only sandbox's installed version and persistent setup inputs."""
    return SetupReceipt(
        cfg.state_dir / _RECEIPT_NAME,
        _OWNER,
        version(_OWNER),
        {
            "python": python_identity(),
            "credentials": str(cfg.db_path),
            "host_tools": hashlib.sha256(host_tools_source().encode()).hexdigest(),
        },
    )


def check_artifacts(cfg: SandboxConfig, *, live: bool = False) -> tuple[SetupCheck, ...]:
    """Verify required owned artifacts independently of the receipt."""
    checks = list(check_supervisor_hooks(root=cfg.state_dir, live=live))
    try:
        with cfg.db_path.open("rb") as db:
            header = db.read(len(_SQLITE_HEADER))
        ready = len(header) == len(_SQLITE_HEADER) and header != _SQLITE_HEADER
    except OSError:
        ready = False
    checks.append(
        SetupCheck(
            _OWNER,
            "credentials",
            SetupStatus.READY if ready else SetupStatus.MISSING,
            "" if ready else "Encrypted credentials database missing; run setup",
        )
    )
    if live:
        checks.extend(check_host_tools())
    return tuple(checks)


def check_host_tools() -> tuple[SetupCheck, ...]:
    """Discover required host tools before setup or managed launch changes state."""
    return tuple(
        SetupCheck(
            _OWNER,
            name,
            SetupStatus.READY if find_host_tool(name) else SetupStatus.MISSING,
            f"{name} must be executable on PATH",
        )
        for name in ("podman", "git")
    )
