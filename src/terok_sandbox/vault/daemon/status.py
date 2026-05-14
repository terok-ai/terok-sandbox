# SPDX-FileCopyrightText: 2026 Jiri Vyskocil
# SPDX-License-Identifier: Apache-2.0

"""Vault status vocabulary — the dataclass and exception every renderer reads.

[`VaultStatus`][terok_sandbox.vault.daemon.status.VaultStatus] is the
snapshot every consumer (sandbox ``vault status``, executor status
table, terok TUI, doctor) builds against; [`VaultUnreachableError`][terok_sandbox.vault.daemon.status.VaultUnreachableError]
is what task creation raises when the daemon isn't responding.
Living in their own module keeps
[`lifecycle`][terok_sandbox.vault.daemon.lifecycle] focused on the
``VaultManager`` controller — and lets renderers depend on just the
vocabulary without dragging in systemd / subprocess machinery.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from ..store.encryption import PassphraseSource


@dataclass(frozen=True)
class VaultStatus:
    """Current state of the vault."""

    mode: str
    """``"systemd"``, ``"daemon"``, or ``"none"``."""

    running: bool
    """Whether the vault is active (systemd socket listening or daemon alive)."""

    healthy: bool
    """Whether the vault is healthy for its current activation mode.

    HTTP-probe based when the systemd service is active; socket-liveness
    based when the service is idle but the socket is listening.
    """

    socket_path: Path
    """Configured Unix socket path."""

    db_path: Path
    """Configured credential database path."""

    routes_path: Path
    """Configured routes JSON path."""

    routes_configured: int
    """Number of routes in routes.json (0 if missing or invalid)."""

    credentials_stored: tuple[str, ...]
    """Provider names with stored credentials."""

    transport: str | None = None
    """Detected transport: ``"tcp"``, ``"socket"``, or ``None`` if not running."""

    ssh_keys_stored: int = 0
    """Number of distinct SSH keypairs in the credential DB (0 when locked or absent)."""

    passphrase_source: PassphraseSource | None = None
    """Which tier of the resolution chain unlocked the DB this call.

    ``None`` when the DB couldn't be opened (locked, absent, or
    schema-corrupted).  See
    [`PassphraseSource`][terok_sandbox.vault.store.encryption.PassphraseSource]
    for the closed set of values.
    """

    locked: bool = False
    """``True`` when the DB exists but no passphrase tier could open it.

    Distinct from ``credentials_stored == ()`` (empty but unlocked) and
    from ``not db_path.is_file()`` (no DB at all) — set only when an
    open was attempted and failed for passphrase reasons.
    """

    plaintext_passphrase_path: Path | None = None
    """Config file that stores the SQLCipher passphrase as plaintext, or ``None``.

    Set whenever ``credentials.passphrase`` is configured *anywhere* in
    the layered config — independent of which tier actually unlocked
    the DB this call.  Renderers (sandbox ``vault status``, executor
    status table, terok TUI) surface this as a WARNING so the
    plaintext-on-disk trust boundary scales visibility with risk
    instead of disappearing into the resolver chain.
    """


class VaultUnreachableError(RuntimeError):
    """Raised when the vault is not reachable.

    Carries diagnostic paths so CLI layers can append their own
    remediation hints (specific command names vary by package).
    """

    def __init__(self, *, socket_path: Path, db_path: Path) -> None:
        self.socket_path = socket_path
        self.db_path = db_path
        super().__init__(
            "Vault is not reachable.\n"
            "\n"
            "The vault injects real API credentials into container\n"
            "requests without exposing secrets to the container filesystem.\n"
            "\n"
            "Start the vault (socket activation or manual daemon)\n"
            "before creating tasks.\n"
            "\n"
            f"Socket: {socket_path}\n"
            f"DB:     {db_path}"
        )


__all__ = ["VaultStatus", "VaultUnreachableError"]
