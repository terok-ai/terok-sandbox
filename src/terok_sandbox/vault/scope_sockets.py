# SPDX-FileCopyrightText: 2026 Jiri Vyskocil
# SPDX-License-Identifier: Apache-2.0

"""Per-scope SSH-agent sockets for host-local git operations.

The vault exposes one UID-gated Unix socket per project scope that has at
least one assigned SSH key.  Host-side gate operations (``terok gate-sync``)
point ``SSH_AUTH_SOCK`` at the corresponding path to reach that scope's
keys without a phantom-token handshake and without the daemon knowing or
caring which process is connecting — filesystem permissions (0600 on a
0700 parent) are the sole access control.

The [`ScopeSocketReconciler`][terok_sandbox.vault.scope_sockets.ScopeSocketReconciler] watches the DB's ``ssh_keys_version``
counter and binds / closes sockets as scopes gain and lose assignments.
It runs as an asyncio background task inside the vault daemon.
"""

from __future__ import annotations

import asyncio
import logging
import sqlite3
from pathlib import Path

from .ssh_signer import start_ssh_signer_local

_logger = logging.getLogger("terok-vault.scope-sockets")

_POLL_INTERVAL_SECONDS = 2.0
"""How often to check the DB version counter for changes."""


class ScopeSocketReconciler:
    """Keeps one local SSH-agent socket per scope in sync with DB assignments.

    Lifecycle:

    * [`start`][terok_sandbox.vault.scope_sockets.ScopeSocketReconciler.start] performs the initial bind pass and launches the polling
      task.
    * [`stop`][terok_sandbox.vault.scope_sockets.ScopeSocketReconciler.stop] cancels the poller, closes every bound server, and
      unlinks every socket file.
    """

    def __init__(self, *, db_path: str, runtime_dir: Path) -> None:
        self._db_path = db_path
        self._runtime_dir = runtime_dir
        self._servers: dict[str, asyncio.Server] = {}
        self._last_version: int = -1
        self._task: asyncio.Task | None = None

    async def start(self) -> None:
        """Bind sockets for the current assignment state, then begin polling."""
        self._runtime_dir.mkdir(parents=True, exist_ok=True)
        await self._reconcile()
        self._task = asyncio.create_task(self._poll(), name="scope-socket-reconciler")

    async def stop(self) -> None:
        """Cancel the poller and tear down every bound socket."""
        if self._task is not None:
            self._task.cancel()
            try:
                await self._task
            except asyncio.CancelledError:
                pass
            self._task = None
        for scope, server in list(self._servers.items()):
            server.close()
            try:
                await server.wait_closed()
            except Exception:  # noqa: BLE001
                pass
            self._socket_path(scope).unlink(missing_ok=True)
        self._servers.clear()

    def socket_path(self, scope: str) -> Path:
        """Return the per-scope socket path; public for callers that render it."""
        return self._socket_path(scope)

    async def _poll(self) -> None:
        """Reconcile on each version bump until cancelled."""
        try:
            while True:
                await asyncio.sleep(_POLL_INTERVAL_SECONDS)
                try:
                    await self._reconcile()
                except Exception:
                    _logger.exception("scope-socket reconciliation failed")
        except asyncio.CancelledError:
            raise

    async def _reconcile(self) -> None:
        """Bind missing sockets, close orphaned ones.

        ``_last_version`` only advances when the full reconciliation step
        converges on the DB's current state — a transient failure on one
        scope leaves the counter unchanged so the next poll retries
        instead of waiting for some unrelated DB write to bump the
        version again.
        """
        version, desired = self._snapshot()
        if version == self._last_version:
            return

        current = set(self._servers)
        converged = True
        for scope in desired - current:
            if not await self._bind_scope(scope):
                converged = False
        for scope in current - desired:
            if not await self._unbind_scope(scope):
                converged = False

        if converged:
            self._last_version = version

    async def _bind_scope(self, scope: str) -> bool:
        """Start a per-scope signer; return ``True`` on success."""
        path = self._socket_path(scope)
        try:
            server = await start_ssh_signer_local(
                scope=scope,
                socket_path=path,
                db_path=self._db_path,
            )
        except Exception:
            _logger.exception("Failed to bind scope socket for %r", scope)
            return False
        self._servers[scope] = server
        return True

    async def _unbind_scope(self, scope: str) -> bool:
        """Close *scope*'s server and unlink its socket; return ``True`` on success.

        Failures on close *or* unlink keep the scope in ``_servers`` so
        the next reconciliation pass retries the teardown — otherwise
        ``_last_version`` could advance past an incomplete cleanup and
        leave a stale socket path behind indefinitely.
        """
        server = self._servers.get(scope)
        if server is None:
            return True
        try:
            server.close()
            await server.wait_closed()
        except Exception:
            _logger.exception("Failed to close scope server for %r", scope)
            return False
        try:
            self._socket_path(scope).unlink(missing_ok=True)
        except OSError:
            _logger.exception("Failed to unlink scope socket path for %r", scope)
            return False
        # Both close and unlink succeeded — stop tracking the scope.
        self._servers.pop(scope, None)
        return True

    def _socket_path(self, scope: str) -> Path:
        """Return the canonical socket path for a scope."""
        return self._runtime_dir / f"ssh-agent-local-{scope}.sock"

    def _snapshot(self) -> tuple[int, set[str]]:
        """Return ``(version, scopes)`` from a short-lived sqlite3 read."""
        conn = sqlite3.connect(self._db_path)
        try:
            version_row = conn.execute(
                "SELECT version FROM ssh_keys_version WHERE id = 0",
            ).fetchone()
            version = version_row[0] if version_row else 0
            scopes = {
                r[0]
                for r in conn.execute(
                    "SELECT DISTINCT scope FROM ssh_key_assignments",
                ).fetchall()
            }
        finally:
            conn.close()
        return version, scopes
