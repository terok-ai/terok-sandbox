# SPDX-FileCopyrightText: 2025 Jiri Vyskocil
# SPDX-License-Identifier: Apache-2.0

"""Tests for the gate's ``_git_env_with_ssh`` — three-branch policy.

The gate prefers the vault-managed per-scope socket by default; explicit
opt-in (``use_personal_ssh=True``) falls through to the user's ambient
SSH; with neither, [`GateAuthNotConfigured`][terok_sandbox.GateAuthNotConfigured] is raised.

The vault branch additionally pins OpenSSH so personal keys can never
leak in and no interactive prompt can ever leak out — asserted below as
``IdentityAgent=<sock>`` + ``-F /dev/null`` + ``BatchMode=yes``.
"""

from __future__ import annotations

import socket
import sqlite3
from contextlib import ExitStack
from pathlib import Path
from unittest.mock import patch

import pytest

from terok_sandbox.gate.mirror import GateAuthNotConfigured, _git_env_with_ssh


def _patched_socket_path(tmp_path: Path, scope: str) -> Path:
    """Return a tmp-path-based socket location used by the patched config."""
    return tmp_path / f"ssh-agent-local-{scope}.sock"


def _bind_socket(path: Path) -> socket.socket:
    """Bind a real Unix domain socket at *path* so ``S_ISSOCK`` is true."""
    s = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
    s.bind(str(path))
    return s


def _seed_assignments_db(tmp_path: Path, scopes: list[str]) -> Path:
    """Create a minimal ``ssh_key_assignments`` table seeded with *scopes*."""
    db_path = tmp_path / "credentials.db"
    conn = sqlite3.connect(str(db_path))
    try:
        conn.execute(
            "CREATE TABLE ssh_key_assignments (scope TEXT NOT NULL, key_id INTEGER NOT NULL)"
        )
        conn.executemany(
            "INSERT INTO ssh_key_assignments (scope, key_id) VALUES (?, ?)",
            [(s, 1) for s in scopes],
        )
        conn.commit()
    finally:
        conn.close()
    return db_path


def _patch_config(tmp_path: Path, *, scopes_with_keys: list[str] | None = None):
    """Patch ``SandboxConfig`` so the socket dir and DB both point into *tmp_path*.

    When *scopes_with_keys* is ``None`` the DB file doesn't exist, which
    forces `_db_has_keys_for_scope` down its ``sqlite3.Error`` path
    and skips the wait-for-bind window — matching the "no keys assigned"
    scenario.
    """
    db_path = (
        _seed_assignments_db(tmp_path, scopes_with_keys)
        if scopes_with_keys is not None
        else tmp_path / "nonexistent.db"
    )
    stack = ExitStack()
    stack.enter_context(
        patch(
            "terok_sandbox.config.SandboxConfig.ssh_signer_local_socket_path",
            new=lambda self, scope: _patched_socket_path(tmp_path, scope),
        )
    )
    stack.enter_context(
        patch(
            "terok_sandbox.config.SandboxConfig.db_path",
            new=property(lambda self: db_path),
        )
    )
    return stack


class TestVaultPath:
    """Verify the default vault-only branch."""

    def test_sets_env_when_socket_present(self, tmp_path: Path) -> None:
        """With a vault socket in place, env steers git at the vault."""
        sock_path = _patched_socket_path(tmp_path, "proj")
        sock = _bind_socket(sock_path)
        try:
            with _patch_config(tmp_path):
                env = _git_env_with_ssh(scope="proj")
        finally:
            sock.close()

        assert env["SSH_AUTH_SOCK"] == str(sock_path)
        cmd = env["GIT_SSH_COMMAND"]
        # Vault agent is the pinned identity source.
        assert f"IdentityAgent={sock_path}" in cmd
        assert "IdentityFile=none" in cmd
        # User's ~/.ssh/config must not influence auth.
        assert "-F /dev/null" in cmd
        # No passphrase / host-key / password prompts may leak to the caller.
        assert "BatchMode=yes" in cmd
        assert "StrictHostKeyChecking=no" in cmd
        # Agent must remain consulted — IdentitiesOnly with IdentityFile=none
        # would leave ssh with no identities at all.
        assert "IdentitiesOnly=yes" not in cmd

    def test_non_socket_file_is_rejected(self, tmp_path: Path) -> None:
        """A regular file at the socket path must not pass the guard."""
        _patched_socket_path(tmp_path, "proj").touch()
        with _patch_config(tmp_path), pytest.raises(GateAuthNotConfigured):
            _git_env_with_ssh(scope="proj")

    def test_raises_when_socket_missing_and_no_opt_in(self, tmp_path: Path) -> None:
        """Without a vault socket and without opt-in, refuse to run."""
        with _patch_config(tmp_path), pytest.raises(GateAuthNotConfigured) as excinfo:
            _git_env_with_ssh(scope="nowhere")
        assert "nowhere" in str(excinfo.value)
        assert "ssh-init" in str(excinfo.value)
        assert "use-personal-ssh" in str(excinfo.value)


class TestBindRaceWindow:
    """Verify the grace window for the reconciler to bind a fresh socket."""

    def test_waits_for_socket_when_db_says_scope_has_keys(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Socket missing but DB shows keys → poll until socket appears."""
        sock_path = _patched_socket_path(tmp_path, "proj")

        ticks: list[int] = []
        sockets: list[socket.socket] = []

        def fake_sleep(_delay: float) -> None:
            # On the third poll, the reconciler "binds" the socket.
            ticks.append(1)
            if len(ticks) == 3:
                sockets.append(_bind_socket(sock_path))

        monkeypatch.setattr("terok_sandbox.gate.mirror.time.sleep", fake_sleep)

        try:
            with _patch_config(tmp_path, scopes_with_keys=["proj"]):
                env = _git_env_with_ssh(scope="proj")
        finally:
            for s in sockets:
                s.close()

        assert env["SSH_AUTH_SOCK"] == str(sock_path)
        assert len(ticks) >= 3  # proved we actually waited

    def test_no_wait_when_db_has_no_keys(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Socket missing and DB empty → raise immediately, no waiting."""
        slept: list[float] = []
        monkeypatch.setattr("terok_sandbox.gate.mirror.time.sleep", slept.append)

        with _patch_config(tmp_path, scopes_with_keys=[]), pytest.raises(GateAuthNotConfigured):
            _git_env_with_ssh(scope="proj")

        assert slept == []  # no grace window when nothing's assigned


class TestPersonalOptIn:
    """Verify ``use_personal_ssh=True`` leaves the env alone."""

    def test_returns_untouched_env(self, tmp_path: Path) -> None:
        """Opt-in: no GIT_SSH_COMMAND injected, SSH_AUTH_SOCK not forced."""
        import os

        with _patch_config(tmp_path):
            env = _git_env_with_ssh(scope="anything", use_personal_ssh=True)

        assert env.get("SSH_AUTH_SOCK") == os.environ.get("SSH_AUTH_SOCK")
        assert "GIT_SSH_COMMAND" not in env or env["GIT_SSH_COMMAND"] == os.environ.get(
            "GIT_SSH_COMMAND", ""
        )

    def test_opt_in_wins_even_when_socket_exists(self, tmp_path: Path) -> None:
        """Opt-in bypasses the vault socket entirely."""
        sock = _bind_socket(_patched_socket_path(tmp_path, "proj"))
        try:
            with _patch_config(tmp_path):
                env = _git_env_with_ssh(scope="proj", use_personal_ssh=True)
        finally:
            sock.close()

        assert "GIT_SSH_COMMAND" not in env or "IdentityFile=none" not in env["GIT_SSH_COMMAND"]
