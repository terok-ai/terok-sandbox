# SPDX-FileCopyrightText: 2026 Jiri Vyskocil
# SPDX-License-Identifier: Apache-2.0

"""Append-only JSONL audit log for credential-bearing broker requests.

Records one line per phantom-token-validated proxy call so the operator can
answer "what did this subject's credentials get used for?" after the fact.
The schema is deliberately small — caller-supplied labels (``scope``,
``subject``, ``credential_set``, ``provider``) plus request shape
(``method``, ``path``, ``status``, ``outcome``, ``duration_ms``).  No
request bodies, no response bodies; the broker is a transparent proxy and
audit's job is forensic context, not deep packet capture.

One JSONL file under the vault state dir, shared across every subject the
broker has ever served.  The sandbox makes no per-subject layout decisions
because it doesn't model "subject" as anything other than an opaque label —
read-side filtering is the orchestrator's job.

Soft-fail semantics throughout: a missing parent dir, a full disk, an
EACCES on the file all degrade gracefully to a single ``WARNING`` log
line.  The proxy's primary job is forwarding, not auditing; an audit
write must never block or kill a credential-bearing request.
"""

from __future__ import annotations

import asyncio
import json
import logging
import stat
from io import TextIOBase
from pathlib import Path
from typing import Any

_logger = logging.getLogger(__name__)

#: Filename of the credential-audit log under the vault state dir.  Single
#: file across every subject; readers filter in user space.
_AUDIT_BASENAME = "credential_audit.jsonl"


def credential_audit_log_path(vault_root: Path) -> Path:
    """Return the canonical credential-audit JSONL path under *vault_root*."""
    return vault_root / _AUDIT_BASENAME


class AuditWriter:
    """Append-only JSONL writer with one process-wide async lock per file.

    Holds a single line-buffered append handle for the lifetime of the
    broker — opening per request would dominate the sub-millisecond proxy
    hot path.  Concurrent aiohttp handlers serialise through an
    [`asyncio.Lock`][asyncio.Lock] so their JSONL bytes can't interleave.

    Args:
        path: Where to write.  Parent directories and the file itself are
            created lazily on first write so the broker can be started
            against a vault dir that doesn't exist yet (fresh installs,
            tests using ``tmp_path``).
    """

    def __init__(self, path: Path) -> None:
        """Bind the writer to *path* without touching the filesystem."""
        self._path = path
        self._lock = asyncio.Lock()
        self._fh: TextIOBase | None = None
        self._open_failed = False

    async def write(self, entry: dict[str, Any]) -> None:
        """Append one JSON-encoded line.  Soft-fail on every error path.

        Marshals *entry* with compact separators (no whitespace) and
        writes ``json + "\\n"`` under the per-writer lock.  Line-buffered
        text mode flushes after every newline, so a crash mid-broker
        loses at most the most recent partial line.
        """
        line = json.dumps(entry, separators=(",", ":")) + "\n"
        async with self._lock:
            if self._fh is None and not self._open_failed:
                self._fh = self._lazy_open()
            if self._fh is None:
                return
            try:
                self._fh.write(line)
            except OSError as exc:
                _logger.warning("Credential audit write failed (%s): %s", self._path, exc)

    def _lazy_open(self) -> TextIOBase | None:
        """Create parent dir, open the file, tighten perms.  ``None`` on failure.

        ``buffering=1`` selects line-buffered text mode — every ``write``
        ending in ``"\\n"`` flushes immediately, which is exactly the
        durability the audit log wants without paying for an explicit
        ``flush()`` per call.
        """
        try:
            self._path.parent.mkdir(parents=True, exist_ok=True)
            fh = self._path.open("a", encoding="utf-8", buffering=1)
            self._path.chmod(stat.S_IRUSR | stat.S_IWUSR)  # 0600
        except OSError as exc:
            _logger.warning("Credential audit log unavailable (%s): %s", self._path, exc)
            self._open_failed = True
            return None
        return fh

    async def close(self) -> None:
        """Close the underlying handle if open.  Idempotent."""
        async with self._lock:
            if self._fh is not None:
                try:
                    self._fh.close()
                finally:
                    self._fh = None
