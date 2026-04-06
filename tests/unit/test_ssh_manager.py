# SPDX-FileCopyrightText: 2026 Jiri Vyskocil
# SPDX-License-Identifier: Apache-2.0

"""Tests for SSHManager constructor and key resolution."""

from __future__ import annotations

from pathlib import Path

import pytest

from terok_sandbox import SandboxConfig
from terok_sandbox.credentials.ssh import SSHManager, effective_ssh_key_name


class TestSSHManagerConstructor:
    """Verify SSHManager init stores parameters correctly."""

    def test_minimal_construction(self) -> None:
        """SSHManager requires only project_id."""
        mgr = SSHManager(project_id="demo")
        assert mgr._project_id == "demo"
        assert mgr._ssh_host_dir is None
        assert mgr._ssh_key_name is None
        assert mgr._ssh_config_template is None

    def test_all_params(self, tmp_path: Path) -> None:
        """SSHManager accepts all optional parameters."""
        host_dir = tmp_path / "ssh"
        template = tmp_path / "tpl"
        mgr = SSHManager(
            project_id="proj",
            ssh_host_dir=host_dir,
            ssh_key_name="my_key",
            ssh_config_template=template,
        )
        assert mgr._ssh_host_dir == host_dir
        assert mgr._ssh_key_name == "my_key"
        assert mgr._ssh_config_template == template

    def test_string_paths_converted(self) -> None:
        """String paths are converted to Path objects."""
        mgr = SSHManager(
            project_id="p",
            ssh_host_dir="/tmp/ssh",
            ssh_config_template="/tmp/tpl",
        )
        assert isinstance(mgr._ssh_host_dir, Path)
        assert isinstance(mgr._ssh_config_template, Path)

    def test_no_envs_base_dir_param(self) -> None:
        """SSHManager no longer accepts envs_base_dir."""
        with pytest.raises(TypeError, match="envs_base_dir"):
            SSHManager(project_id="p", envs_base_dir="/tmp")  # type: ignore[call-arg]


class TestKeyName:
    """Verify key_name property resolution."""

    def test_default_key_name(self) -> None:
        """Default key name derives from project_id."""
        mgr = SSHManager(project_id="myproj")
        assert mgr.key_name == "id_ed25519_myproj"

    def test_custom_key_name(self) -> None:
        """Explicit ssh_key_name overrides the default."""
        mgr = SSHManager(project_id="myproj", ssh_key_name="custom_key")
        assert mgr.key_name == "custom_key"


class TestEffectiveKeyName:
    """Verify the effective_ssh_key_name helper."""

    def test_default(self) -> None:
        """Default: id_ed25519_<project>."""
        assert effective_ssh_key_name("demo") == "id_ed25519_demo"

    def test_custom_name_passthrough(self) -> None:
        """Explicit name is returned as-is."""
        assert effective_ssh_key_name("demo", ssh_key_name="mykey") == "mykey"

    def test_custom_key_type(self) -> None:
        """Key type is reflected in the derived name."""
        assert effective_ssh_key_name("demo", key_type="rsa") == "id_rsa_demo"


class TestSandboxConfigNoEnvsDir:
    """Verify effective_envs_dir was removed from SandboxConfig."""

    def test_no_effective_envs_dir(self) -> None:
        """SandboxConfig no longer exposes effective_envs_dir."""
        assert not hasattr(SandboxConfig, "effective_envs_dir")
        cfg = SandboxConfig()
        assert not hasattr(cfg, "effective_envs_dir")

    def test_ssh_keys_dir_still_exists(self) -> None:
        """ssh_keys_dir property is preserved (SSH keys have their own dir)."""
        cfg = SandboxConfig()
        assert cfg.ssh_keys_dir == cfg.state_dir / "ssh-keys"


class TestInitFallbackResolution:
    """Verify init() target directory resolution."""

    def test_init_uses_ssh_host_dir_when_set(self, tmp_path: Path) -> None:
        """Explicit ssh_host_dir is used as the target directory."""
        ssh_dir = tmp_path / "custom-ssh"
        mgr = SSHManager(project_id="proj", ssh_host_dir=ssh_dir)
        result = mgr.init()
        assert Path(result["dir"]) == ssh_dir.resolve()

    def test_init_falls_back_to_ssh_keys_dir(
        self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
    ) -> None:
        """Without ssh_host_dir, init() resolves via SandboxConfig().ssh_keys_dir."""
        monkeypatch.setenv("TEROK_SANDBOX_STATE_DIR", str(tmp_path / "sandbox"))
        mgr = SSHManager(project_id="demo")
        result = mgr.init()
        expected = (tmp_path / "sandbox" / "ssh-keys" / "demo").resolve()
        assert Path(result["dir"]) == expected
