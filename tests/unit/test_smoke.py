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
    from terok_sandbox._util import ensure_dir, ensure_dir_writable, render_template

    assert callable(ensure_dir)
    assert callable(ensure_dir_writable)
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
    """SandboxConfig is constructible with defaults; tcp mode auto-allocates ports."""
    from pathlib import Path

    from terok_sandbox.config import SandboxConfig

    cfg = SandboxConfig(services_mode="tcp")
    assert isinstance(cfg.state_dir, Path)
    assert isinstance(cfg.gate_port, int)
    from terok_sandbox.port_registry import PORT_RANGE

    assert cfg.gate_port in PORT_RANGE


def test_import_shield():
    """Shield adapter module is importable."""
    from terok_sandbox import shield

    assert callable(shield.make_shield)
    assert callable(shield.pre_start)
    assert callable(shield.status)


def test_import_runtime():
    """Runtime subpackage exposes the protocol and backend types."""
    from terok_sandbox import runtime

    assert callable(runtime.PodmanRuntime)
    assert callable(runtime.NullRuntime)
    # Protocol types are classes (Protocol-derived)
    assert isinstance(runtime.ContainerRuntime, type)
    assert isinstance(runtime.Container, type)
    assert isinstance(runtime.Image, type)


def test_import_gate_lifecycle():
    """Gate lifecycle module is importable."""
    from terok_sandbox.gate.lifecycle import GateServerManager

    mgr = GateServerManager()
    assert callable(mgr.get_status)
    assert callable(mgr.is_systemd_available)


def test_import_gate_tokens():
    """Gate tokens module is importable."""
    from terok_sandbox.gate.tokens import TokenStore

    store = TokenStore()
    assert callable(store.create)
    assert callable(store.revoke_for_task)


def test_import_ssh():
    """SSH modules are importable."""
    from terok_sandbox.credentials.ssh import SSHManager
    from terok_sandbox.credentials.ssh_keypair import (
        export_ssh_keypair,
        generate_keypair,
        import_ssh_keypair,
    )

    assert callable(SSHManager)
    assert callable(generate_keypair)
    assert callable(import_ssh_keypair)
    assert callable(export_ssh_keypair)


def test_import_git_gate():
    """GitGate class is importable."""
    from terok_sandbox.gate.mirror import GitGate

    assert callable(GitGate)


def test_cli_main_shows_help_without_args():
    """CLI entry point shows help and exits with code 1 when no subcommand given."""
    from unittest.mock import patch

    import pytest

    from terok_sandbox.cli import main

    with patch("sys.argv", ["terok-sandbox"]), pytest.raises(SystemExit) as exc_info:
        main()
    assert exc_info.value.code == 1


def test_reserve_port_returns_valid_port():
    """PodmanRuntime.reserve_port yields a port in the ephemeral range."""
    from terok_sandbox import PodmanRuntime

    with PodmanRuntime().reserve_port() as reservation:
        assert 1024 <= reservation.port <= 65535


def test_reserve_port_unique():
    """Consecutive reservations return different ports while held."""
    import contextlib

    from terok_sandbox import PodmanRuntime

    runtime = PodmanRuntime()
    with contextlib.ExitStack() as stack:
        reservations = []
        for _ in range(10):
            reservation = runtime.reserve_port()
            reservations.append(reservation)
            stack.callback(reservation.close)
        ports = {r.port for r in reservations}
        assert len(ports) == 10
