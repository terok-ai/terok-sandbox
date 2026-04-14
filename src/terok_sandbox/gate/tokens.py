# SPDX-FileCopyrightText: 2026 Jiri Vyskocil
# SPDX-License-Identifier: Apache-2.0

"""Token CRUD for gate server per-task authentication.

Each task gets a prefixed random 128-bit hex token scoped to a credential scope.
Tokens are stored in ``state_root()/gate/tokens.json`` and read by the
standalone gate server process (which receives the file path via
``--token-file``).

Token format: ``terok-g-<32 hex chars>`` (e.g. ``terok-g-a1b2c3…``).

File format::

    {"terok-g-<hex>": {"scope": "<scope>", "task": "<task_id>"}}
"""

from __future__ import annotations

import contextlib
import fcntl
import json
import os
import secrets
import tempfile
from collections.abc import Iterator
from pathlib import Path

from ..config import SandboxConfig


class TokenStore:
    """Manages per-task authentication tokens for the gate server.

    Encapsulates the token file path, advisory locking, and atomic
    read-modify-write cycles.  The gate server receives the same file
    path via ``--token-file`` and reads tokens at request time.
    """

    def __init__(self, cfg: SandboxConfig | None = None) -> None:
        self._path = (cfg or SandboxConfig()).token_file_path

    @property
    def file_path(self) -> Path:
        """Return the path to the shared token file."""
        return self._path

    def create(self, scope: str, task_id: str) -> str:
        """Generate a 128-bit hex token, persist atomically, and return it.

        Uses ``secrets.token_hex(16)`` for cryptographic randomness.
        Atomic write via ``tempfile`` + ``os.replace()``.
        """
        token = f"terok-g-{secrets.token_hex(16)}"
        with self._lock():
            tokens = self._read()
            tokens[token] = {"scope": scope, "task": task_id}
            self._write(tokens)
        return token

    def revoke_for_task(self, scope: str, task_id: str) -> None:
        """Remove all tokens for the given scope+task pair.  Idempotent."""
        with self._lock():
            tokens = self._read()
            to_remove = [
                t
                for t, info in tokens.items()
                if info.get("scope") == scope and info.get("task") == task_id
            ]
            if not to_remove:
                return
            for t in to_remove:
                del tokens[t]
            self._write(tokens)

    def _read(self) -> dict[str, dict[str, str]]:
        """Load tokens.json.  Returns ``{}`` when the file does not exist.

        Raises on I/O or parse errors to prevent silent data loss — a
        subsequent ``_write()`` would overwrite all existing tokens with
        whatever the caller accumulated in memory.
        """
        if not self._path.is_file():
            return {}
        data = json.loads(self._path.read_text(encoding="utf-8"))
        if not isinstance(data, dict):
            raise ValueError(f"Token file is not a JSON object: {self._path}")
        return {
            tok: info
            for tok, info in data.items()
            if isinstance(tok, str)
            and isinstance(info, dict)
            and isinstance(info.get("scope"), str)
            and isinstance(info.get("task"), str)
        }

    def _write(self, tokens: dict[str, dict[str, str]]) -> None:
        """Atomic write: write to a temp file, then ``os.replace()`` over the original."""
        self._path.parent.mkdir(parents=True, exist_ok=True)
        fd, tmp = tempfile.mkstemp(dir=self._path.parent, suffix=".tmp")
        try:
            with os.fdopen(fd, "w", encoding="utf-8") as f:
                json.dump(tokens, f, indent=2)
                f.write("\n")
            os.replace(tmp, self._path)
        except BaseException:
            with contextlib.suppress(OSError):
                os.unlink(tmp)
            raise

    @contextlib.contextmanager
    def _lock(self) -> Iterator[None]:
        """Advisory file lock serializing token read-modify-write cycles."""
        lock_path = self._path.with_suffix(self._path.suffix + ".lock")
        lock_path.parent.mkdir(parents=True, exist_ok=True)
        with lock_path.open("a+", encoding="utf-8") as lockf:
            fcntl.flock(lockf.fileno(), fcntl.LOCK_EX)
            try:
                yield
            finally:
                fcntl.flock(lockf.fileno(), fcntl.LOCK_UN)
