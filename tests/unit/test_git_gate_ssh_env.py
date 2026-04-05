# SPDX-FileCopyrightText: 2025 Jiri Vyskocil
# SPDX-License-Identifier: Apache-2.0

"""Tests for _git_env_with_ssh — SSH env setup for host-side gate operations."""

from __future__ import annotations

from pathlib import Path

from terok_sandbox.git_gate import _git_env_with_ssh


class TestGitEnvWithSsh:
    """Verify GIT_SSH_COMMAND is built from the key file, not a config file."""

    def test_sets_git_ssh_command_when_key_exists(self, tmp_path: Path) -> None:
        """When the private key file exists, GIT_SSH_COMMAND points to it."""
        key = tmp_path / "id_ed25519_myproj"
        key.write_text("fake-key")

        env = _git_env_with_ssh(project_id="myproj", ssh_host_dir=tmp_path, ssh_key_name=None)

        assert "GIT_SSH_COMMAND" in env
        cmd = env["GIT_SSH_COMMAND"]
        assert str(key) in cmd
        assert "IdentitiesOnly=yes" in cmd
        assert "StrictHostKeyChecking=no" in cmd
        assert env["SSH_AUTH_SOCK"] == ""

    def test_no_git_ssh_command_when_key_missing(self, tmp_path: Path) -> None:
        """When no key file exists, env is returned unmodified (HTTPS fallback)."""
        env = _git_env_with_ssh(project_id="myproj", ssh_host_dir=tmp_path, ssh_key_name=None)

        assert "GIT_SSH_COMMAND" not in env

    def test_respects_explicit_key_name(self, tmp_path: Path) -> None:
        """An explicit ssh_key_name overrides the derived default."""
        key = tmp_path / "my_custom_key"
        key.write_text("fake-key")

        env = _git_env_with_ssh(
            project_id="myproj", ssh_host_dir=tmp_path, ssh_key_name="my_custom_key"
        )

        assert str(key) in env["GIT_SSH_COMMAND"]
