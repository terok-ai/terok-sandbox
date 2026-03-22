# SPDX-FileCopyrightText: 2026 Jiri Vyskocil
# SPDX-License-Identifier: Apache-2.0

"""Smoke tests verifying the package imports correctly."""


def test_import_package():
    """Package root is importable and exposes public API."""
    import terok_sandbox

    assert hasattr(terok_sandbox, "__version__")
    assert isinstance(terok_sandbox.__version__, str)
    assert hasattr(terok_sandbox, "SandboxConfig")
    assert hasattr(terok_sandbox, "SSHManager")
    assert hasattr(terok_sandbox, "GitGate")


def test_import_util():
    """Vendored _util subpackage is importable."""
    from terok_sandbox._util import dump, ensure_dir, ensure_dir_writable, load, render_template

    assert callable(ensure_dir)
    assert callable(ensure_dir_writable)
    assert callable(load)
    assert callable(dump)
    assert callable(render_template)


def test_import_gate():
    """Gate subpackage is importable."""
    from terok_sandbox.gate import server

    assert hasattr(server, "main")
    assert hasattr(server, "TokenStore")


def test_import_paths():
    """Path resolution module is importable and returns Path objects."""
    from pathlib import Path

    from terok_sandbox.paths import config_root, runtime_root, state_root

    assert isinstance(config_root(), Path)
    assert isinstance(state_root(), Path)
    assert isinstance(runtime_root(), Path)


def test_import_config():
    """SandboxConfig is constructible with defaults."""
    from pathlib import Path

    from terok_sandbox.config import SandboxConfig

    cfg = SandboxConfig()
    assert isinstance(cfg.state_dir, Path)
    assert isinstance(cfg.gate_port, int)
    assert cfg.gate_port == 9418


def test_import_shield():
    """Shield adapter module is importable."""
    from terok_sandbox import shield

    assert callable(shield.make_shield)
    assert callable(shield.pre_start)
    assert callable(shield.status)


def test_import_runtime():
    """Runtime module is importable."""
    from terok_sandbox import runtime

    assert callable(runtime.get_container_state)
    assert callable(runtime.is_container_running)
    assert callable(runtime.stop_task_containers)


def test_import_gate_server():
    """Gate server lifecycle module is importable."""
    from terok_sandbox import gate_server

    assert callable(gate_server.get_server_status)
    assert callable(gate_server.is_systemd_available)


def test_import_gate_tokens():
    """Gate tokens module is importable."""
    from terok_sandbox import gate_tokens

    assert callable(gate_tokens.create_token)
    assert callable(gate_tokens.revoke_token_for_task)


def test_import_ssh():
    """SSH module is importable."""
    from terok_sandbox.ssh import SSHManager, effective_ssh_key_name

    assert callable(SSHManager)
    assert callable(effective_ssh_key_name)
    assert effective_ssh_key_name("myproj") == "id_ed25519_myproj"


def test_import_git_gate():
    """GitGate class is importable."""
    from terok_sandbox.git_gate import GitGate

    assert callable(GitGate)


def test_cli_main_exits_cleanly():
    """CLI entry point prints version and exits cleanly."""
    import pytest

    from terok_sandbox.cli import main

    with pytest.raises(SystemExit, match="0"):
        main()
