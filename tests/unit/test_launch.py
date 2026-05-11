# SPDX-FileCopyrightText: 2026 Jiri Vyskocil
# SPDX-License-Identifier: Apache-2.0

"""Tests for ``terok-sandbox prepare / run / cleanup``."""

from __future__ import annotations

import inspect
import json
from pathlib import Path
from unittest.mock import patch

import pytest

from terok_sandbox.commands import (
    LAUNCH_COMMANDS,
    _handle_cleanup,
    _handle_prepare,
)
from terok_sandbox.config import SandboxConfig
from terok_sandbox.launch import (
    CONTAINER_BRIDGES_DIR,
    LOOPBACK_VAULT_PORT,
    WiringPlan,
    cleanup,
    compose,
    format_args,
    reject_managed_flags,
    reject_managed_volumes,
    run_state_dir,
)


def _make_cfg(tmp_path: Path, services_mode: str = "socket") -> SandboxConfig:
    """Build a SandboxConfig rooted in *tmp_path* for hermetic tests."""
    return SandboxConfig(
        state_dir=tmp_path / "state",
        runtime_dir=tmp_path / "run",
        config_dir=tmp_path / "config",
        vault_dir=tmp_path / "vault",
        gate_port=18000 if services_mode == "tcp" else None,
        token_broker_port=18001 if services_mode == "tcp" else None,
        ssh_signer_port=18002 if services_mode == "tcp" else None,
        services_mode=services_mode,  # type: ignore[arg-type]
    )


# ---------------------------------------------------------------------------
# Compose — happy paths per flag combination
# ---------------------------------------------------------------------------


class TestCompose:
    """Verify ``compose`` emits the right podman args per flag combination."""

    def test_shield_only_when_scope_omitted(self, tmp_path: Path) -> None:
        """With no --scope, gate/broker/ssh skip silently; shield still applies."""
        cfg = _make_cfg(tmp_path)
        with patch("terok_sandbox.shield.pre_start", return_value=["--annotation=t-s=1"]):
            args, plan = compose("myc", cfg=cfg, shield=True, gate=True, broker=True, scope=None)
        assert plan.shield is True
        assert plan.gate is False  # silently skipped
        assert plan.broker is False
        assert plan.ssh is False
        assert "--annotation=t-s=1" in args
        assert "--name" in args and "myc" in args
        # No token-related env vars
        assert not any("TEROK_SSH_SIGNER_TOKEN" in a for a in args)
        assert not any("TEROK_GATE_TOKEN" in a for a in args)

    def test_full_wiring_with_scope_socket_mode(self, tmp_path: Path) -> None:
        """All subsystems active in socket mode → sockets bind-mounted, tokens minted."""
        cfg = _make_cfg(tmp_path, services_mode="socket")
        with (
            patch("terok_sandbox.shield.pre_start", return_value=["--annotation=t-s=1"]),
            patch("terok_sandbox.gate.tokens.TokenStore.create", return_value="terok-g-abc"),
            patch(
                "terok_sandbox.credentials.db.CredentialDB.create_token",
                return_value="terok-p-xyz",
            ),
        ):
            args, plan = compose("myc", cfg=cfg, shield=True, gate=True, broker=True, scope="proj")
        joined = " ".join(args)
        assert plan.gate and plan.broker and plan.ssh
        # Bridge resources mount
        assert CONTAINER_BRIDGES_DIR in joined
        # Vault socket mount
        assert "/run/terok/vault.sock" in joined
        assert f"TEROK_VAULT_LOOPBACK_PORT={LOOPBACK_VAULT_PORT}" in joined
        # Gate socket + token
        assert "/run/terok/gate-server.sock" in joined
        assert "TEROK_GATE_TOKEN=terok-g-abc" in joined
        # SSH signer socket + token
        assert "/run/terok/ssh-agent.sock" in joined
        assert "TEROK_SSH_SIGNER_TOKEN=terok-p-xyz" in joined
        # Name flag
        assert args[-2:] == ["--name", "myc"]

    def test_full_wiring_with_scope_tcp_mode(self, tmp_path: Path) -> None:
        """TCP mode emits port env vars instead of socket mounts."""
        cfg = _make_cfg(tmp_path, services_mode="tcp")
        with (
            patch("terok_sandbox.shield.pre_start", return_value=[]),
            patch("terok_sandbox.gate.tokens.TokenStore.create", return_value="terok-g-abc"),
            patch(
                "terok_sandbox.credentials.db.CredentialDB.create_token",
                return_value="terok-p-xyz",
            ),
        ):
            args, _ = compose("myc", cfg=cfg, shield=False, gate=True, broker=True, scope="proj")
        joined = " ".join(args)
        assert "TEROK_TOKEN_BROKER_PORT=18001" in joined
        assert "TEROK_SSH_SIGNER_PORT=18002" in joined
        assert "TEROK_GATE_PORT=18000" in joined
        assert "/run/terok/vault.sock" not in joined

    def test_no_shield_skips_shield_pre_start(self, tmp_path: Path) -> None:
        """--no-shield path doesn't call shield.pre_start."""
        cfg = _make_cfg(tmp_path)
        with patch("terok_sandbox.shield.pre_start") as mocked:
            compose("myc", cfg=cfg, shield=False, gate=False, broker=False, scope=None)
            mocked.assert_not_called()

    def test_scope_required_for_gate_broker_notes_stderr(
        self, tmp_path: Path, capsys: pytest.CaptureFixture
    ) -> None:
        """When --gate/--broker requested without --scope, stderr notes it."""
        cfg = _make_cfg(tmp_path)
        with patch("terok_sandbox.shield.pre_start", return_value=[]):
            compose("myc", cfg=cfg, shield=False, gate=True, broker=True, scope=None)
        err = capsys.readouterr().err
        assert "--gate requires --scope" in err
        assert "--broker requires --scope" in err


# ---------------------------------------------------------------------------
# Meta persistence + cleanup
# ---------------------------------------------------------------------------


class TestMetaAndCleanup:
    """Verify per-container state is persisted at prepare and torn down at cleanup."""

    def test_meta_written_on_compose(self, tmp_path: Path) -> None:
        cfg = _make_cfg(tmp_path)
        with patch("terok_sandbox.shield.pre_start", return_value=[]):
            compose("myc", cfg=cfg, shield=True, gate=False, broker=False, scope=None)
        meta = run_state_dir(cfg, "myc") / "meta.json"
        assert meta.is_file()
        data = json.loads(meta.read_text())
        assert data["shield"] is True
        assert data["gate"] is False
        assert data["scope"] is None

    def test_cleanup_no_state_returns_false(self, tmp_path: Path) -> None:
        cfg = _make_cfg(tmp_path)
        assert cleanup("never-prepared", cfg=cfg) is False

    def test_cleanup_idempotent(self, tmp_path: Path) -> None:
        """Second cleanup is a no-op."""
        cfg = _make_cfg(tmp_path)
        with (
            patch("terok_sandbox.shield.pre_start", return_value=[]),
            patch("terok_sandbox.shield.down") as down,
            patch("terok_sandbox.gate.tokens.TokenStore.create", return_value="terok-g-abc"),
            patch(
                "terok_sandbox.credentials.db.CredentialDB.create_token",
                return_value="terok-p-xyz",
            ),
            patch("terok_sandbox.credentials.db.CredentialDB.revoke_tokens", return_value=2),
            patch("terok_sandbox.gate.tokens.TokenStore.revoke_for_task", return_value=None),
        ):
            compose("myc", cfg=cfg, shield=True, gate=True, broker=True, scope="proj")
            assert cleanup("myc", cfg=cfg) is True
            assert cleanup("myc", cfg=cfg) is False
            down.assert_called_once()  # only the first cleanup invokes shield.down

    def test_cleanup_revokes_tokens_for_container_as_subject(self, tmp_path: Path) -> None:
        """Cleanup uses container name as the subject when revoking."""
        cfg = _make_cfg(tmp_path)
        with (
            patch("terok_sandbox.shield.pre_start", return_value=[]),
            patch("terok_sandbox.shield.down"),
            patch("terok_sandbox.gate.tokens.TokenStore.create", return_value="terok-g-abc"),
            patch(
                "terok_sandbox.credentials.db.CredentialDB.create_token",
                return_value="terok-p-xyz",
            ),
            patch(
                "terok_sandbox.credentials.db.CredentialDB.revoke_tokens", return_value=2
            ) as revoke_db,
            patch(
                "terok_sandbox.gate.tokens.TokenStore.revoke_for_task", return_value=None
            ) as revoke_gate,
        ):
            compose("myc", cfg=cfg, shield=True, gate=True, broker=True, scope="proj")
            cleanup("myc", cfg=cfg)
        revoke_db.assert_called_with("proj", "myc")
        revoke_gate.assert_called_with("proj", "myc")


# ---------------------------------------------------------------------------
# Collision rejection
# ---------------------------------------------------------------------------


class TestRejectManagedFlags:
    """Verify ``run`` rejects collisions with sandbox-managed flags/volumes."""

    @pytest.mark.parametrize(
        "flag", ["--name", "--network", "--hooks-dir", "--annotation", "--userns"]
    )
    def test_rejects_each_managed_flag(self, flag: str) -> None:
        with pytest.raises(SystemExit) as exc_info:
            reject_managed_flags([flag, "foo"])
        assert flag in str(exc_info.value)

    def test_accepts_user_flags(self) -> None:
        reject_managed_flags(["-it", "--rm", "-e", "FOO=bar"])

    def test_rejects_net_alias(self) -> None:
        with pytest.raises(SystemExit) as exc_info:
            reject_managed_flags(["--net=host"])
        assert "--network" in str(exc_info.value)

    def test_rejects_managed_volume_target(self) -> None:
        with pytest.raises(SystemExit) as exc_info:
            reject_managed_volumes(["-v", f"/foo:{CONTAINER_BRIDGES_DIR}:ro"])
        assert CONTAINER_BRIDGES_DIR in str(exc_info.value)

    def test_volume_aliases_recognised(self) -> None:
        with pytest.raises(SystemExit):
            reject_managed_volumes(["--volume=/foo:/run/terok/vault.sock:Z"])

    def test_user_volume_unrelated_target_passes(self) -> None:
        reject_managed_volumes(["-v", "/workspace:/workspace:Z"])


# ---------------------------------------------------------------------------
# Output formatting
# ---------------------------------------------------------------------------


class TestFormatArgs:
    """Verify ``format_args`` shell-quotes and JSON-serialises correctly."""

    def test_shell_quoted(self) -> None:
        out = format_args(["--name", "my container"], output_json=False)
        assert "'my container'" in out

    def test_json(self) -> None:
        out = format_args(["--name", "myc"], output_json=True)
        assert json.loads(out) == ["--name", "myc"]


# ---------------------------------------------------------------------------
# WiringPlan round-trip
# ---------------------------------------------------------------------------


class TestWiringPlan:
    """Plan serialises round-trip; needs_bridges reflects subsystem flags."""

    def test_round_trip(self) -> None:
        plan = WiringPlan(scope="proj", shield=True, gate=True, broker=False, ssh=True)
        assert WiringPlan.from_dict(plan.to_dict()) == plan

    def test_needs_bridges_true_when_any_subsystem_active(self) -> None:
        for plan in (
            WiringPlan(scope="p", shield=False, gate=True, broker=False, ssh=False),
            WiringPlan(scope="p", shield=False, gate=False, broker=True, ssh=False),
            WiringPlan(scope="p", shield=False, gate=False, broker=False, ssh=True),
        ):
            assert plan.needs_bridges()

    def test_needs_bridges_false_when_only_shield(self) -> None:
        plan = WiringPlan(scope=None, shield=True, gate=False, broker=False, ssh=False)
        assert plan.needs_bridges() is False


# ---------------------------------------------------------------------------
# CLI registration
# ---------------------------------------------------------------------------


class TestRegistration:
    """Verify the three commands are wired and have consistent signatures."""

    def test_three_commands_registered(self) -> None:
        names = {c.name for c in LAUNCH_COMMANDS}
        assert names == {"prepare", "run", "cleanup"}

    def test_handlers_accept_cfg(self) -> None:
        for cmd in LAUNCH_COMMANDS:
            sig = inspect.signature(cmd.handler)
            assert "cfg" in sig.parameters, f"{cmd.handler.__name__} missing cfg"

    def test_prepare_help_mentions_both_delivery_patterns(self) -> None:
        prep = next(c for c in LAUNCH_COMMANDS if c.name == "prepare")
        assert "Build-time" in prep.epilog and "Runtime" in prep.epilog


# ---------------------------------------------------------------------------
# Handler integration — prepare prints args, cleanup tears down
# ---------------------------------------------------------------------------


class TestHandlers:
    """Smoke tests for the dispatchable handlers."""

    def test_prepare_prints_shielded_args(
        self, tmp_path: Path, capsys: pytest.CaptureFixture
    ) -> None:
        cfg = _make_cfg(tmp_path)
        with patch("terok_sandbox.shield.pre_start", return_value=["--annotation=x"]):
            _handle_prepare("myc", cfg=cfg)
        out = capsys.readouterr().out
        assert "--annotation=x" in out
        assert "--name myc" in out

    def test_prepare_json_output(self, tmp_path: Path, capsys: pytest.CaptureFixture) -> None:
        cfg = _make_cfg(tmp_path)
        with patch("terok_sandbox.shield.pre_start", return_value=["--annotation=x"]):
            _handle_prepare("myc", output_json=True, cfg=cfg)
        out = capsys.readouterr().out.strip()
        parsed = json.loads(out)
        assert parsed[-2:] == ["--name", "myc"]

    def test_cleanup_handler_idempotent(
        self, tmp_path: Path, capsys: pytest.CaptureFixture
    ) -> None:
        cfg = _make_cfg(tmp_path)
        _handle_cleanup("never-prepared", cfg=cfg)
        out = capsys.readouterr().out
        assert "Nothing to clean up" in out or "No sandbox state found" in out
