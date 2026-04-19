# SPDX-FileCopyrightText: 2025 Jiri Vyskocil
# SPDX-License-Identifier: Apache-2.0

"""Tests for the gate's ``_git_env_with_ssh`` — three-branch policy.

The gate prefers the vault-managed per-scope socket by default; explicit
opt-in (``use_personal_ssh=True``) falls through to the user's ambient
SSH; with neither, :class:`GateAuthNotConfigured` is raised.
"""

from __future__ import annotations

import socket
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


def _patch_config(tmp_path: Path):
    """Patch ``SandboxConfig().ssh_signer_local_socket_path`` to point into *tmp_path*."""
    return patch(
        "terok_sandbox.config.SandboxConfig.ssh_signer_local_socket_path",
        new=lambda self, scope: _patched_socket_path(tmp_path, scope),
    )


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
        assert "IdentityFile=none" in cmd
        assert "StrictHostKeyChecking=no" in cmd
        assert "IdentitiesOnly=yes" not in cmd  # agent must remain consulted

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
