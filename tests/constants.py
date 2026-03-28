# SPDX-FileCopyrightText: 2026 Jiri Vyskocil
# SPDX-License-Identifier: Apache-2.0

"""Shared test constants: filesystem paths, IPs, and ports.

Centralises magic literals so they can be changed in one place.
All mock paths live under ``MOCK_BASE`` to avoid polluting real directories.
"""

from pathlib import Path

# ── Placeholder directories ──────────────────────────────────────────────────

MOCK_BASE = Path("/tmp/terok-testing")
"""Root for synthetic filesystem paths used by mocked tests."""

MOCK_TASK_DIR = MOCK_BASE / "tasks" / "42"
"""Fake per-task directory used by shield adapter tests."""

MOCK_CONFIG_ROOT = Path("/home/user/.config/terok")
"""Fake XDG-style config root used by path-related tests."""

FAKE_GATE_DIR = MOCK_BASE / "gate"
"""Fake gate mirror path used by gate-server tests."""

FAKE_STATE_DIR = MOCK_BASE / "state"
"""Fake state root used by gate-server related tests."""

FAKE_TEROK_STATE_DIR = MOCK_BASE / "terok-state"
"""Fake state root used by token-file path tests."""

# ── Nonexistent / missing paths ──────────────────────────────────────────────

NONEXISTENT_DIR = Path("/nonexistent")
"""Guaranteed-missing absolute path used for missing-file behavior tests."""

NONEXISTENT_TOKENS_PATH = NONEXISTENT_DIR / "tokens.json"
"""Missing gate token store path used by token-store tests."""

MISSING_TOKENS_PATH = MOCK_BASE / "does-not-exist" / "tokens.json"
"""Absent token-store path with a writable parent used by token-lock tests."""

# ── Network constants ────────────────────────────────────────────────────────

LOCALHOST = "127.0.0.1"
"""Loopback address used for bind/connect in tests."""

GATE_PORT = 9418
"""Default gate server port."""

PROXY_PORT = 18731
"""Default credential proxy port."""

FAKE_PEER_PORT = 12345
"""Arbitrary port for fake client_address tuples."""

LOCALHOST_PEER = (LOCALHOST, FAKE_PEER_PORT)
"""Fake peer address for HTTP handler tests."""
