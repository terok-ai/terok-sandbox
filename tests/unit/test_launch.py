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
    _handle_run,
)
from terok_sandbox.config import SandboxConfig
from terok_sandbox.launch import (
    CONTAINER_BRIDGES_DIR,
    LOOPBACK_VAULT_PORT,
    WiringPlan,
    _find_podman,
    _read_meta,
    bridges_resource_dir,
    cleanup,
    compose,
    exec_podman,
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

    def test_cleanup_warns_when_locked_vault_skips_revocation(
        self, tmp_path: Path, capsys: pytest.CaptureFixture[str]
    ) -> None:
        """A locked vault during cleanup MUST warn so operator knows tokens linger."""
        from terok_sandbox.credentials.db import NoPassphraseError

        cfg = _make_cfg(tmp_path)
        with (
            patch("terok_sandbox.shield.pre_start", return_value=[]),
            patch("terok_sandbox.shield.down"),
            patch("terok_sandbox.gate.tokens.TokenStore.create", return_value="terok-g-abc"),
            patch(
                "terok_sandbox.credentials.db.CredentialDB.create_token",
                return_value="terok-p-xyz",
            ),
            patch("terok_sandbox.gate.tokens.TokenStore.revoke_for_task", return_value=None),
        ):
            compose("myc", cfg=cfg, shield=True, gate=True, broker=True, scope="proj")
            with patch(
                "terok_sandbox.config.SandboxConfig.open_credential_db",
                side_effect=NoPassphraseError("locked"),
            ):
                cleanup("myc", cfg=cfg)
        err = capsys.readouterr().err
        assert "couldn't revoke broker/SSH tokens for proj/myc" in err
        assert "NoPassphraseError" in err
        assert "vault unlock" in err

    def test_ssh_mint_failure_rolls_back_gate_token(self, tmp_path: Path) -> None:
        """A locked vault during SSH-token minting must not leak the gate token.

        Pre-fix: ``compose`` raised before ``_write_meta``; cleanup() had
        no meta to read and the gate token stayed in ``TokenStore``.
        """
        from terok_sandbox.credentials.db import NoPassphraseError

        cfg = _make_cfg(tmp_path)
        with (
            patch("terok_sandbox.shield.pre_start", return_value=[]),
            patch(
                "terok_sandbox.gate.tokens.TokenStore.create", return_value="terok-g-abc"
            ) as gate_create,
            patch(
                "terok_sandbox.config.SandboxConfig.open_credential_db",
                side_effect=NoPassphraseError("locked"),
            ),
            patch(
                "terok_sandbox.gate.tokens.TokenStore.revoke_for_task", return_value=None
            ) as revoke_gate,
        ):
            with pytest.raises(NoPassphraseError):
                compose("myc", cfg=cfg, shield=False, gate=True, broker=False, scope="proj")
            gate_create.assert_called_once_with("proj", "myc")
            revoke_gate.assert_called_once_with("proj", "myc")


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

    def test_cleanup_handler_reports_cleanup_when_state_present(
        self, tmp_path: Path, capsys: pytest.CaptureFixture
    ) -> None:
        cfg = _make_cfg(tmp_path)
        with patch("terok_sandbox.shield.pre_start", return_value=[]):
            compose("myc", cfg=cfg, shield=True, gate=False, broker=False, scope=None)
        with patch("terok_sandbox.shield.down"):
            _handle_cleanup("myc", cfg=cfg)
        out = capsys.readouterr().out
        assert "Cleaned up sandbox state for myc" in out

    def test_run_handler_exec_into_podman(self, tmp_path: Path) -> None:
        """`_handle_run` composes args and `os.execv`s into podman."""
        cfg = _make_cfg(tmp_path)
        with (
            patch("terok_sandbox.shield.pre_start", return_value=["--annotation=x"]),
            patch("terok_sandbox.launch.shutil.which", return_value="/usr/bin/podman"),
            patch("terok_sandbox.launch.Path.resolve", return_value=Path("/usr/bin/podman")),
            patch("terok_sandbox.launch.Path.is_file", return_value=True),
            patch("terok_sandbox.launch.os.access", return_value=True),
            patch("terok_sandbox.launch.os.execv") as execv,
        ):
            _handle_run("myc", cfg=cfg, podman_args=["ubuntu:24.04", "bash"])
        execv.assert_called_once()
        argv = execv.call_args[0][1]
        assert argv[0] == "/usr/bin/podman"
        assert argv[1] == "run"
        assert "--name" in argv and "myc" in argv
        assert argv[-2:] == ["ubuntu:24.04", "bash"]


# ---------------------------------------------------------------------------
# exec_podman + _find_podman edge cases
# ---------------------------------------------------------------------------


class TestExecPodman:
    """Verify `exec_podman` and `_find_podman` error paths."""

    def test_exec_podman_no_args_raises(self) -> None:
        with pytest.raises(SystemExit, match="No image specified"):
            exec_podman(["--name", "myc"], [])

    def test_find_podman_missing_raises(self) -> None:
        with patch("terok_sandbox.launch.shutil.which", return_value=None):
            with pytest.raises(SystemExit, match="podman binary not found"):
                _find_podman()

    def test_exec_podman_rejects_collisions_before_execv(self) -> None:
        """Collision check fires before os.execv — execv never reached."""
        with (
            patch("terok_sandbox.launch.shutil.which", return_value="/usr/bin/podman"),
            patch("terok_sandbox.launch.os.execv") as execv,
            pytest.raises(SystemExit, match="--name"),
        ):
            exec_podman(["--name", "myc"], ["--name", "evil", "ubuntu"])
        execv.assert_not_called()


# ---------------------------------------------------------------------------
# Profile override + corrupted meta + bridges resource path
# ---------------------------------------------------------------------------


class TestEdgeCases:
    """Cover branches the happy-path tests don't exercise."""

    def test_profiles_override_shield_profiles(self, tmp_path: Path) -> None:
        """`--profiles` reaches shield via a `dataclasses.replace`d cfg."""
        cfg = _make_cfg(tmp_path)
        captured: list[SandboxConfig] = []

        def fake_pre_start(container: str, task_dir: Path, c: SandboxConfig) -> list[str]:
            captured.append(c)
            return []

        with patch("terok_sandbox.shield.pre_start", side_effect=fake_pre_start):
            compose(
                "myc",
                cfg=cfg,
                shield=True,
                gate=False,
                broker=False,
                scope=None,
                profiles=("alt-strict",),
            )
        assert captured[0].shield_profiles == ("alt-strict",)

    def test_read_meta_corrupted_returns_none(self, tmp_path: Path) -> None:
        """`_read_meta` swallows JSONDecodeError and treats it as no-state."""
        cfg = _make_cfg(tmp_path)
        state = run_state_dir(cfg, "myc")
        state.mkdir(parents=True)
        (state / "meta.json").write_text("{ not json")
        assert _read_meta(state) is None

    def test_cleanup_swallows_shield_down_failure(self, tmp_path: Path) -> None:
        """`cleanup` is best-effort against shield.down errors."""
        cfg = _make_cfg(tmp_path)
        with patch("terok_sandbox.shield.pre_start", return_value=[]):
            compose("myc", cfg=cfg, shield=True, gate=False, broker=False, scope=None)
        with patch("terok_sandbox.shield.down", side_effect=OSError("nft gone")):
            assert cleanup("myc", cfg=cfg) is True
        # State dir gone even though shield.down failed.
        assert not run_state_dir(cfg, "myc").exists()

    def test_bridges_resource_dir_returns_existing_path(self) -> None:
        """The bundled bridge scripts are reachable via the resolver."""
        d = bridges_resource_dir()
        assert (d / "ensure-bridges.sh").is_file()
        assert (d / "ssh-agent-bridge.sh").is_file()

    def test_reject_managed_volumes_skips_no_target(self) -> None:
        """`-v hostpath` (no colon) is skipped, not flagged."""
        reject_managed_volumes(["-v", "/just/a/path"])

    def test_reject_managed_volumes_skips_empty_after_v(self) -> None:
        """Trailing `-v` with no value is skipped."""
        reject_managed_volumes(["-v"])

    def test_cleanup_db_missing_does_not_crash(self, tmp_path: Path) -> None:
        """A missing credential DB at cleanup time is treated as already-revoked."""
        cfg = _make_cfg(tmp_path)
        with (
            patch("terok_sandbox.shield.pre_start", return_value=[]),
            patch("terok_sandbox.gate.tokens.TokenStore.create", return_value="t"),
            patch("terok_sandbox.credentials.db.CredentialDB.create_token", return_value="p"),
        ):
            compose("myc", cfg=cfg, shield=True, gate=True, broker=True, scope="proj")
        # Now make CredentialDB construction fail at cleanup.
        with (
            patch("terok_sandbox.shield.down"),
            patch(
                "terok_sandbox.config.SandboxConfig.open_credential_db",
                side_effect=OSError("db file vanished"),
            ),
            patch("terok_sandbox.gate.tokens.TokenStore.revoke_for_task"),
        ):
            assert cleanup("myc", cfg=cfg) is True

    def test_handlers_construct_default_cfg(self, tmp_path: Path) -> None:
        """Handlers fall back to `SandboxConfig()` when *cfg* is omitted."""
        from terok_sandbox import commands as cmd_mod

        fake_cfg = _make_cfg(tmp_path)
        with (
            patch.object(cmd_mod, "SandboxConfig", return_value=fake_cfg),
            patch("terok_sandbox.shield.pre_start", return_value=[]),
            patch("terok_sandbox.launch.shutil.which", return_value="/usr/bin/podman"),
            patch("terok_sandbox.launch.Path.resolve", return_value=Path("/usr/bin/podman")),
            patch("terok_sandbox.launch.Path.is_file", return_value=True),
            patch("terok_sandbox.launch.os.access", return_value=True),
            patch("terok_sandbox.launch.os.execv"),
        ):
            _handle_prepare("a")
            _handle_run("b", podman_args=["ubuntu"])
            _handle_cleanup("a")
