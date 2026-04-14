# SPDX-FileCopyrightText: 2026 Jiri Vyskocil
# SPDX-License-Identifier: Apache-2.0

"""Tests for the file-based shared port registry."""

from __future__ import annotations

import json
from pathlib import Path
from unittest.mock import patch

import pytest

from terok_sandbox import port_registry as reg
from terok_sandbox.port_registry import PortRegistry, _is_port_free, _save_ports

# Capture real _save_ports before conftest patches it away.
_real_save_ports = _save_ports


@pytest.fixture(autouse=True)
def _isolated_registry(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """Redirect the default registry to a tmp dir and restore _save_ports.

    The global conftest suppresses ``_save_ports`` to prevent FS leaks,
    but persistence tests in this module need the real implementation.

    ``_is_port_free`` is stubbed to always return True so tests are
    deterministic regardless of host port availability.  Tests that
    need real socket behaviour override this with their own patch.
    """
    registry = tmp_path / "terok-ports"
    registry.mkdir(exist_ok=True)
    monkeypatch.setattr(reg._default, "registry_dir", registry)
    monkeypatch.setattr(reg, "_save_ports", _real_save_ports)
    monkeypatch.setattr(reg, "_is_port_free", lambda _port: True)
    reg.reset_cache()


@pytest.fixture()
def fresh_registry(tmp_path: Path) -> PortRegistry:
    """Return an isolated PortRegistry instance with its own directory."""
    d = tmp_path / "fresh-ports"
    d.mkdir()
    return PortRegistry(d, reg.PORT_RANGE)


def test_claim_preferred_free() -> None:
    """Preferred port is returned when available."""
    assert reg.claim_port("proxy", preferred=18700) == 18700


def test_claim_defaults_to_range_start() -> None:
    """No preference → starts from PORT_RANGE.start."""
    assert reg.claim_port("gate") == reg.PORT_RANGE.start


def test_claim_busy_increments() -> None:
    """When preferred port is busy, scan upward for the next free one."""
    busy = {18700, 18701}

    def mock_free(p: int) -> bool:
        return p not in busy and _is_port_free(p)

    with patch.object(reg, "_is_port_free", side_effect=mock_free):
        assert reg.claim_port("proxy", preferred=18700) == 18702


def test_self_collision_avoidance() -> None:
    """Consecutive claims from the same process don't collide."""
    a = reg.claim_port("gate")
    b = reg.claim_port("proxy")
    c = reg.claim_port("ssh_agent")
    assert len({a, b, c}) == 3


def test_claim_skips_other_users_ports() -> None:
    """Ports claimed by another user's claim file are skipped."""
    (reg._default.registry_dir / "bob.json").write_text(json.dumps({"gate": 18700}))
    port = reg.claim_port("gate", preferred=18700)
    assert port != 18700


def test_claim_writes_shared_file(monkeypatch: pytest.MonkeyPatch) -> None:
    """Claiming a port writes the user's claim file to the shared dir."""
    monkeypatch.setattr(reg, "_username", lambda: "alice")
    port = reg.claim_port("gate", preferred=18700)
    data = json.loads((reg._default.registry_dir / "alice.json").read_text())
    assert data["gate"] == port


def test_release_removes_from_shared_file(monkeypatch: pytest.MonkeyPatch) -> None:
    """Releasing a port removes it from the shared claim file."""
    monkeypatch.setattr(reg, "_username", lambda: "alice")
    reg.claim_port("web:proj/task-1", preferred=18710)
    reg.release_port("web:proj/task-1")
    data = json.loads((reg._default.registry_dir / "alice.json").read_text())
    assert "web:proj/task-1" not in data


def test_claims_merge_across_sessions(monkeypatch: pytest.MonkeyPatch) -> None:
    """New claims merge with existing ones from a previous session."""
    monkeypatch.setattr(reg, "_username", lambda: "alice")
    # Pre-populate with a previous session's web port
    (reg._default.registry_dir / "alice.json").write_text(json.dumps({"web:proj/old-task": 18710}))
    reg.claim_port("gate", preferred=18700)
    data = json.loads((reg._default.registry_dir / "alice.json").read_text())
    assert data["gate"] == 18700
    assert data["web:proj/old-task"] == 18710


def test_explicit_rejects_other_users_port() -> None:
    """Explicit pin on a port claimed by another user → SystemExit."""
    (reg._default.registry_dir / "bob.json").write_text(json.dumps({"gate": 19000}))
    with pytest.raises(SystemExit, match="claimed by another user"):
        reg.claim_port("proxy", preferred=19000, explicit=True)


def test_explicit_rejects_self_collision() -> None:
    """Explicit pin on a port already held in this process → SystemExit."""
    reg.claim_port("gate", preferred=19000)
    with pytest.raises(SystemExit, match="already claimed in this process"):
        reg.claim_port("proxy", preferred=19000, explicit=True)


def test_stale_other_user_file_ignored() -> None:
    """Malformed JSON in another user's file is silently skipped."""
    (reg._default.registry_dir / "bob.json").write_text("not json at all!!!")
    port = reg.claim_port("gate", preferred=18700)
    assert port == 18700


def test_explicit_busy_fails() -> None:
    """Explicit pin + busy port → SystemExit."""
    with (
        patch.object(reg, "_is_port_free", return_value=False),
        pytest.raises(SystemExit, match="unavailable"),
    ):
        reg.claim_port("proxy", preferred=19000, explicit=True)


def test_explicit_invalid_port_number() -> None:
    """Explicit pin with out-of-range port number → SystemExit."""
    with pytest.raises(SystemExit, match="not a valid port number"):
        reg.claim_port("proxy", preferred=0, explicit=True)
    with pytest.raises(SystemExit, match="not a valid port number"):
        reg.claim_port("proxy", preferred=70000, explicit=True)


def test_repeated_claim_returns_same_port() -> None:
    """Claiming the same service_key again returns the cached port."""
    a = reg.claim_port("gate", preferred=18700)
    b = reg.claim_port("gate", preferred=18800)  # different preferred, ignored
    assert a == b == 18700


def test_release_frees_port() -> None:
    """After release, the port can be claimed by a new key."""
    reg.claim_port("web:proj/task-1", preferred=18710)
    reg.release_port("web:proj/task-1")
    assert reg.claim_port("web:proj/task-2", preferred=18710) == 18710


def test_release_nonexistent_is_noop() -> None:
    """Releasing an unclaimed key does not raise."""
    reg.release_port("never-claimed")


def test_resolve_auto_allocates_distinct() -> None:
    """Auto-resolve assigns three distinct ports."""
    ports = reg.resolve_service_ports(None, None, None)
    assert len({ports.gate, ports.proxy, ports.ssh_agent}) == 3


def test_resolve_cached() -> None:
    """Second call returns the same cached result."""
    assert reg.resolve_service_ports(None, None, None) == reg.resolve_service_ports(
        None, None, None
    )


def test_resolve_explicit() -> None:
    """Explicit ports are passed through."""
    ports = reg.resolve_service_ports(
        19100,
        19200,
        19300,
        gate_explicit=True,
        proxy_explicit=True,
        ssh_explicit=True,
    )
    assert (ports.gate, ports.proxy, ports.ssh_agent) == (19100, 19200, 19300)


def test_resolve_persists_to_state_dir(tmp_path: Path) -> None:
    """resolve_service_ports writes a claims file to state_dir."""
    state = tmp_path / "state"
    state.mkdir()
    ports = reg.resolve_service_ports(None, None, None, state_dir=state)
    claims = json.loads((state / reg._CLAIMS_FILENAME).read_text())
    assert claims == {"gate": ports.gate, "proxy": ports.proxy, "ssh_agent": ports.ssh_agent}


def test_resolve_prefers_saved_ports(tmp_path: Path) -> None:
    """Saved ports are reclaimed when free."""
    state = tmp_path / "state"
    state.mkdir()
    (state / reg._CLAIMS_FILENAME).write_text(
        json.dumps({"gate": 18750, "proxy": 18751, "ssh_agent": 18752})
    )
    ports = reg.resolve_service_ports(None, None, None, state_dir=state)
    assert (ports.gate, ports.proxy, ports.ssh_agent) == (18750, 18751, 18752)


def test_resolve_fails_when_saved_taken(tmp_path: Path) -> None:
    """SystemExit when a saved port is taken by another user."""
    state = tmp_path / "state"
    state.mkdir()
    (state / reg._CLAIMS_FILENAME).write_text(
        json.dumps({"gate": 18750, "proxy": 18751, "ssh_agent": 18752})
    )
    (reg._default.registry_dir / "bob.json").write_text(json.dumps({"gate": 18750}))
    with pytest.raises(SystemExit, match="previously assigned"):
        reg.resolve_service_ports(None, None, None, state_dir=state)


def test_resolve_explicit_overrides_saved(tmp_path: Path) -> None:
    """Explicit config port wins over saved claim."""
    state = tmp_path / "state"
    state.mkdir()
    (state / reg._CLAIMS_FILENAME).write_text(
        json.dumps({"gate": 18750, "proxy": 18751, "ssh_agent": 18752})
    )
    ports = reg.resolve_service_ports(
        19100,
        None,
        None,
        gate_explicit=True,
        state_dir=state,
    )
    assert ports.gate == 19100
    assert ports.proxy == 18751


def test_corrupt_claims_file_ignored(tmp_path: Path) -> None:
    """Garbage claims file → auto-allocate normally."""
    state = tmp_path / "state"
    state.mkdir()
    (state / reg._CLAIMS_FILENAME).write_text("not json at all!!!")
    ports = reg.resolve_service_ports(None, None, None, state_dir=state)
    assert len({ports.gate, ports.proxy, ports.ssh_agent}) == 3


def test_fresh_instance_independent(fresh_registry: PortRegistry) -> None:
    """An isolated PortRegistry instance does not share state with _default."""
    port = fresh_registry.claim("gate", preferred=18700)
    assert port == 18700
    # _default has no record of this
    assert "gate" not in reg._default._held


def test_symlink_registry_dir_rejected(tmp_path: Path) -> None:
    """Registry dir that is a symlink → SystemExit."""
    real = tmp_path / "real-dir"
    real.mkdir()
    link = tmp_path / "symlinked-ports"
    link.symlink_to(real)
    registry = PortRegistry(link, reg.PORT_RANGE)
    with pytest.raises(SystemExit, match="symlinked"):
        registry.claim("gate")


def test_non_dir_registry_path_rejected(tmp_path: Path) -> None:
    """Registry path that is a regular file (not a dir) → SystemExit."""
    path = tmp_path / "not-a-dir"
    path.write_text("")
    registry = PortRegistry(path, reg.PORT_RANGE)
    with pytest.raises(SystemExit, match="not a directory"):
        registry.claim("gate")


def test_symlink_claim_file_skipped(tmp_path: Path) -> None:
    """Symlink posing as another user's claim file is silently skipped."""
    # Target lives outside the registry dir to isolate the symlink test
    target = tmp_path / "decoy.json"
    target.write_text(json.dumps({"gate": 18700}))
    link = reg._default.registry_dir / "bob.json"
    link.symlink_to(target)
    # Symlink is not a regular file → 18700 is NOT treated as taken
    port = reg.claim_port("gate", preferred=18700)
    assert port == 18700


def test_oversized_claim_file_skipped() -> None:
    """Oversized claim file in shared dir is silently skipped."""
    big = reg._default.registry_dir / "bob.json"
    big.write_text("x" * 20_000)  # > 16 KiB limit
    port = reg.claim_port("gate", preferred=18700)
    assert port == 18700


def test_auto_clamps_out_of_range_preferred() -> None:
    """Auto-allocation ignores a preferred port outside PORT_RANGE."""
    port = reg.claim_port("gate", preferred=50000)
    assert port in reg.PORT_RANGE


def test_sandbox_config_auto_resolves(tmp_path: Path) -> None:
    """SandboxConfig with default (None) ports auto-resolves and persists."""
    from terok_sandbox import SandboxConfig

    state = tmp_path / "sandbox-state"
    state.mkdir()
    cfg = SandboxConfig(state_dir=state)
    assert isinstance(cfg.gate_port, int)
    assert len({cfg.gate_port, cfg.proxy_port, cfg.ssh_agent_port}) == 3
    assert (state / reg._CLAIMS_FILENAME).exists()


def test_sandbox_config_explicit_passthrough() -> None:
    """SandboxConfig with explicit ports does not auto-resolve."""
    from terok_sandbox import SandboxConfig

    cfg = SandboxConfig(gate_port=9418, proxy_port=18731, ssh_agent_port=18732)
    assert (cfg.gate_port, cfg.proxy_port, cfg.ssh_agent_port) == (9418, 18731, 18732)
