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
    PerContainerResources,
    WiringPlan,
    _find_podman,
    _read_meta,
    _resolve_container_id,
    _rollback_compose_state,
    _validate_container_name,
    _write_sidecar,  # noqa: PLC2701 — internal under test
    allocate_per_container_resources,
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
        with patch(
            "terok_sandbox.integrations.shield.ShieldManager.pre_start",
            return_value=["--annotation=t-s=1"],
        ):
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
            patch(
                "terok_sandbox.integrations.shield.ShieldManager.pre_start",
                return_value=["--annotation=t-s=1"],
            ),
            patch("terok_sandbox.launch.mint_gate_token", return_value="terok-g-abc"),
            patch(
                "terok_sandbox.vault.store.db.CredentialDB.create_token",
                return_value="terok-p-xyz",
            ),
        ):
            args, plan = compose("myc", cfg=cfg, shield=True, gate=True, broker=True, scope="proj")
        joined = " ".join(args)
        assert plan.gate and plan.broker and plan.ssh
        # Bridge resources mount
        assert CONTAINER_BRIDGES_DIR in joined
        # Per-container runtime dir mounted at /run/terok/ (the supervisor
        # binds vault.sock + ssh-agent.sock + gate-server.sock inside it).
        assert ":/run/terok" in joined
        assert "/run/myc" in joined  # host-side dir name = container name
        assert f"TEROK_VAULT_LOOPBACK_PORT={LOOPBACK_VAULT_PORT}" in joined
        # Gate socket env + token (the socket is bound by the supervisor
        # inside the per-container dir mount — no separate -v sub-mount).
        assert "TEROK_GATE_SOCKET=/run/terok/gate-server.sock" in joined
        assert "TEROK_GATE_TOKEN=terok-g-abc" in joined
        # The gate socket is NOT a -v sub-mount anymore.
        assert not any(a == "-v" and "gate-server.sock" in args[i + 1] for i, a in enumerate(args))
        # SSH signer env var (the socket itself is bound by the supervisor
        # inside the per-container dir mount above).
        assert "TEROK_SSH_SIGNER_SOCKET=/run/terok/ssh-agent.sock" in joined
        assert "TEROK_SSH_SIGNER_TOKEN=terok-p-xyz" in joined
        # Name flag (followed by the supervisor sidecar annotation)
        assert "--name" in args and "myc" in args
        annotation_value = next(a for a in args if a.startswith("terok.sandbox.sidecar="))
        assert "/sidecar/myc.json" in annotation_value

    def test_full_wiring_with_scope_tcp_mode(self, tmp_path: Path) -> None:
        """TCP mode emits per-container port env vars instead of socket mounts."""
        cfg = _make_cfg(tmp_path, services_mode="tcp")
        with (
            patch("terok_sandbox.integrations.shield.ShieldManager.pre_start", return_value=[]),
            patch("terok_sandbox.launch.mint_gate_token", return_value="terok-g-abc"),
            patch(
                "terok_sandbox.vault.store.db.CredentialDB.create_token",
                return_value="terok-p-xyz",
            ),
        ):
            args, _ = compose("myc", cfg=cfg, shield=False, gate=True, broker=True, scope="proj")
        joined = " ".join(args)
        # Broker/signer/gate ports are now per-container (via bind(0)), not
        # the singleton cfg.* ports.  Just check the envs are present with
        # SOME numeric value.
        import re

        assert re.search(r"TEROK_TOKEN_BROKER_PORT=\d+", joined)
        assert re.search(r"TEROK_SSH_SIGNER_PORT=\d+", joined)
        assert re.search(r"TEROK_GATE_PORT=\d+", joined)
        assert "TEROK_GATE_TOKEN=terok-g-abc" in joined
        assert "/run/terok/vault.sock" not in joined

    def test_no_shield_skips_shield_pre_start(self, tmp_path: Path) -> None:
        """--no-shield path doesn't call shield.pre_start."""
        cfg = _make_cfg(tmp_path)
        with patch("terok_sandbox.integrations.shield.ShieldManager.pre_start") as mocked:
            compose("myc", cfg=cfg, shield=False, gate=False, broker=False, scope=None)
            mocked.assert_not_called()

    def test_scope_required_for_gate_broker_notes_stderr(
        self, tmp_path: Path, capsys: pytest.CaptureFixture
    ) -> None:
        """When --gate/--broker requested without --scope, stderr notes it."""
        cfg = _make_cfg(tmp_path)
        with patch("terok_sandbox.integrations.shield.ShieldManager.pre_start", return_value=[]):
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
        with patch("terok_sandbox.integrations.shield.ShieldManager.pre_start", return_value=[]):
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
            patch("terok_sandbox.integrations.shield.ShieldManager.pre_start", return_value=[]),
            patch("terok_sandbox.integrations.shield.ShieldManager.down") as down,
            patch("terok_sandbox.launch._resolve_container_id", return_value="ctr-uuid"),
            patch("terok_sandbox.launch.mint_gate_token", return_value="terok-g-abc"),
            patch(
                "terok_sandbox.vault.store.db.CredentialDB.create_token",
                return_value="terok-p-xyz",
            ),
            patch("terok_sandbox.vault.store.db.CredentialDB.revoke_tokens", return_value=2),
        ):
            compose("myc", cfg=cfg, shield=True, gate=True, broker=True, scope="proj")
            assert cleanup("myc", cfg=cfg) is True
            assert cleanup("myc", cfg=cfg) is False
            down.assert_called_once_with("myc", "ctr-uuid")  # container UUID is plumbed through

    def test_cleanup_revokes_vault_tokens_for_container_as_subject(self, tmp_path: Path) -> None:
        """Cleanup uses container name as the subject when revoking vault tokens.

        The gate token has no on-disk store anymore (revocation = supervisor
        death), so cleanup only revokes the vault broker/SSH tokens.
        """
        cfg = _make_cfg(tmp_path)
        with (
            patch("terok_sandbox.integrations.shield.ShieldManager.pre_start", return_value=[]),
            patch("terok_sandbox.integrations.shield.ShieldManager.down"),
            patch("terok_sandbox.launch.mint_gate_token", return_value="terok-g-abc"),
            patch(
                "terok_sandbox.vault.store.db.CredentialDB.create_token",
                return_value="terok-p-xyz",
            ),
            patch(
                "terok_sandbox.vault.store.db.CredentialDB.revoke_tokens", return_value=2
            ) as revoke_db,
        ):
            compose("myc", cfg=cfg, shield=True, gate=True, broker=True, scope="proj")
            cleanup("myc", cfg=cfg)
        revoke_db.assert_called_with("proj", "myc")

    def test_cleanup_warns_when_locked_vault_skips_revocation(
        self, tmp_path: Path, capsys: pytest.CaptureFixture[str]
    ) -> None:
        """A locked vault during cleanup MUST warn so operator knows tokens linger."""
        from terok_sandbox.vault.store.db import NoPassphraseError

        cfg = _make_cfg(tmp_path)
        with (
            patch("terok_sandbox.integrations.shield.ShieldManager.pre_start", return_value=[]),
            patch("terok_sandbox.integrations.shield.ShieldManager.down"),
            patch("terok_sandbox.launch.mint_gate_token", return_value="terok-g-abc"),
            patch(
                "terok_sandbox.vault.store.db.CredentialDB.create_token",
                return_value="terok-p-xyz",
            ),
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

    def test_ssh_mint_failure_does_not_write_sidecar(self, tmp_path: Path) -> None:
        """A locked vault during SSH-token minting aborts compose before the sidecar.

        The gate token now lives only in-process (no on-disk store), so a
        failed launch leaks nothing to revoke — compose simply raises and
        ``_write_meta`` / ``_write_sidecar`` never run.
        """
        from terok_sandbox.vault.store.db import NoPassphraseError

        cfg = _make_cfg(tmp_path)
        with (
            patch("terok_sandbox.integrations.shield.ShieldManager.pre_start", return_value=[]),
            patch("terok_sandbox.launch.mint_gate_token", return_value="terok-g-abc") as gate_mint,
            patch(
                "terok_sandbox.config.SandboxConfig.open_credential_db",
                side_effect=NoPassphraseError("locked"),
            ),
        ):
            with pytest.raises(NoPassphraseError):
                compose("myc", cfg=cfg, shield=False, gate=True, broker=False, scope="proj")
            gate_mint.assert_called_once_with()
        # No sidecar / meta was written because compose raised first.
        assert not (run_state_dir(cfg, "myc") / "meta.json").exists()
        assert not (cfg.state_dir / "sidecar" / "myc.json").exists()


# ---------------------------------------------------------------------------
# Sidecar write — success payload + best-effort failure handling
# ---------------------------------------------------------------------------


def _socket_resources(tmp_path: Path) -> PerContainerResources:
    """A socket-mode resource bundle (no ports) rooted under *tmp_path*."""
    return PerContainerResources(
        container_runtime_dir=tmp_path / "run" / "myc",
        token_broker_port=None,
        ssh_signer_port=None,
        gate_port=None,
    )


class TestWriteSidecar:
    """``_write_sidecar`` persists the supervisor's per-container config.

    Returns the absolute path on success; logs to stderr and returns
    ``None`` on any filesystem failure (the caller fails the launch
    closed rather than starting a container with dead endpoints).
    """

    def test_writes_socket_mode_payload(self, tmp_path: Path) -> None:
        """Socket mode: no port keys; gate keys only when a token is wired."""
        cfg = _make_cfg(tmp_path, services_mode="socket")
        plan = WiringPlan(scope="proj", shield=False, gate=True, broker=True, ssh=True)
        path = _write_sidecar(cfg, "myc", plan, _socket_resources(tmp_path), "terok-g-abc")
        assert path == cfg.state_dir / "sidecar" / "myc.json"
        payload = json.loads(path.read_text())
        assert payload["container_name"] == "myc"
        assert payload["ipc_mode"] == "socket"
        assert payload["scope_id"] == "proj"
        assert payload["project_id"] == "proj"  # set because gate_token + scope present
        assert payload["gate_token"] == "terok-g-abc"
        assert payload["gate_base_path"] == str(cfg.gate_base_path)
        # Socket mode carries no TCP ports.
        assert "tcp_port" not in payload
        assert "gate_port" not in payload

    def test_tcp_mode_carries_ports(self, tmp_path: Path) -> None:
        """TCP mode records the per-container broker / signer / gate ports."""
        cfg = _make_cfg(tmp_path, services_mode="tcp")
        plan = WiringPlan(scope="proj", shield=False, gate=True, broker=True, ssh=True)
        res = PerContainerResources(
            container_runtime_dir=tmp_path / "run" / "myc",
            token_broker_port=21001,
            ssh_signer_port=21002,
            gate_port=21003,
        )
        path = _write_sidecar(cfg, "myc", plan, res, "terok-g-abc")
        assert path is not None
        payload = json.loads(path.read_text())
        assert payload["tcp_port"] == 21001
        assert payload["ssh_signer_port"] == 21002
        assert payload["gate_port"] == 21003

    def test_no_gate_token_omits_gate_keys_and_project(self, tmp_path: Path) -> None:
        """Without a gate token, gate keys are dropped and ``project_id`` is blank."""
        cfg = _make_cfg(tmp_path, services_mode="socket")
        plan = WiringPlan(scope="proj", shield=False, gate=False, broker=True, ssh=True)
        path = _write_sidecar(cfg, "myc", plan, _socket_resources(tmp_path), None)
        assert path is not None
        payload = json.loads(path.read_text())
        assert payload["project_id"] == ""
        assert "gate_token" not in payload
        assert "gate_base_path" not in payload

    def test_mkdir_failure_returns_none(
        self, tmp_path: Path, capsys: pytest.CaptureFixture
    ) -> None:
        """A sidecar-dir mkdir OSError soft-fails to ``None`` with a stderr warning."""
        cfg = _make_cfg(tmp_path, services_mode="socket")
        plan = WiringPlan(scope=None, shield=True, gate=False, broker=False, ssh=False)
        with patch("pathlib.Path.mkdir", side_effect=OSError("read-only fs")):
            result = _write_sidecar(cfg, "myc", plan, _socket_resources(tmp_path), None)
        assert result is None
        assert "sidecar dir setup failed" in capsys.readouterr().err

    def test_write_failure_returns_none(
        self, tmp_path: Path, capsys: pytest.CaptureFixture
    ) -> None:
        """A json.dump OSError (e.g. disk full) soft-fails to ``None`` with a warning."""
        cfg = _make_cfg(tmp_path, services_mode="socket")
        plan = WiringPlan(scope=None, shield=True, gate=False, broker=False, ssh=False)
        with patch("pathlib.Path.open", side_effect=OSError("no space left")):
            result = _write_sidecar(cfg, "myc", plan, _socket_resources(tmp_path), None)
        assert result is None
        assert "sidecar write failed" in capsys.readouterr().err


class TestComposeAbortsOnSidecarFailure:
    """A failed sidecar write rolls back compose state and aborts the launch."""

    def test_sidecar_write_failure_rolls_back_and_exits(self, tmp_path: Path) -> None:
        """``compose`` fails closed: rollback runs, then ``SystemExit`` with a hint.

        A launch with no sidecar means the supervisor never starts, so the
        container would hit dead vault/SSH/gate endpoints — better to abort
        and unwind the minted tokens + state dirs than orphan them.
        """
        cfg = _make_cfg(tmp_path, services_mode="socket")
        with (
            patch("terok_sandbox.integrations.shield.ShieldManager.pre_start", return_value=[]),
            patch("terok_sandbox.launch.mint_gate_token", return_value="terok-g-abc"),
            patch(
                "terok_sandbox.vault.store.db.CredentialDB.create_token",
                return_value="terok-p-xyz",
            ),
            patch("terok_sandbox.launch._write_sidecar", return_value=None),
            patch("terok_sandbox.launch._rollback_compose_state") as rollback,
            pytest.raises(SystemExit) as exc,
        ):
            compose("myc", cfg=cfg, shield=False, gate=True, broker=True, scope="proj")
        assert "sidecar write failed" in str(exc.value)
        rollback.assert_called_once()


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

    def test_profiles_flag_takes_comma_separated_value(self) -> None:
        """``--profiles base,extra myc`` parses without eating the container.

        Regression for #606 — argparse's greedy ``nargs="+"`` used to
        slurp the following positional, so ``--profiles base extra myc``
        bound ``profiles=["base","extra","myc"]`` and errored
        "container required".  The fix swaps to a single comma-separated
        value (matches podman's ``--cap-add=A,B``), split by a small
        ``_csv_list`` helper so the parsed shape stays ``list[str]``
        for downstream consumers.
        """
        import argparse

        from terok_sandbox.commands._types import CommandTree

        parser = argparse.ArgumentParser()
        CommandTree(LAUNCH_COMMANDS).wire(parser)

        # New comma-separated form parses cleanly.
        args = parser.parse_args(["prepare", "--profiles", "base,extra", "myc"])
        assert args.profiles == ["base", "extra"]
        assert args.container == "myc"

        # Whitespace around items is tolerated.
        args = parser.parse_args(["prepare", "--profiles", "base, extra", "myc"])
        assert args.profiles == ["base", "extra"]

        # Old greedy form is now a hard parse error rather than a
        # silent positional hijack.
        with pytest.raises(SystemExit):
            parser.parse_args(["prepare", "--profiles", "base", "extra", "myc"])


# ---------------------------------------------------------------------------
# Handler integration — prepare prints args, cleanup tears down
# ---------------------------------------------------------------------------


class TestHandlers:
    """Smoke tests for the dispatchable handlers."""

    def test_prepare_prints_shielded_args(
        self, tmp_path: Path, capsys: pytest.CaptureFixture
    ) -> None:
        cfg = _make_cfg(tmp_path)
        with patch(
            "terok_sandbox.integrations.shield.ShieldManager.pre_start",
            return_value=["--annotation=x"],
        ):
            _handle_prepare("myc", cfg=cfg)
        out = capsys.readouterr().out
        assert "--annotation=x" in out
        assert "--name myc" in out

    def test_prepare_json_output(self, tmp_path: Path, capsys: pytest.CaptureFixture) -> None:
        cfg = _make_cfg(tmp_path)
        with patch(
            "terok_sandbox.integrations.shield.ShieldManager.pre_start",
            return_value=["--annotation=x"],
        ):
            _handle_prepare("myc", output_json=True, cfg=cfg)
        out = capsys.readouterr().out.strip()
        parsed = json.loads(out)
        assert "--name" in parsed and "myc" in parsed
        assert any(a.startswith("terok.sandbox.sidecar=") for a in parsed)

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
        with patch("terok_sandbox.integrations.shield.ShieldManager.pre_start", return_value=[]):
            compose("myc", cfg=cfg, shield=True, gate=False, broker=False, scope=None)
        with patch("terok_sandbox.integrations.shield.ShieldManager.down"):
            _handle_cleanup("myc", cfg=cfg)
        out = capsys.readouterr().out
        assert "Cleaned up sandbox state for myc" in out

    def test_run_handler_exec_into_podman(self, tmp_path: Path) -> None:
        """`_handle_run` composes args and `os.execv`s into podman."""
        cfg = _make_cfg(tmp_path)
        with (
            patch(
                "terok_sandbox.integrations.shield.ShieldManager.pre_start",
                return_value=["--annotation=x"],
            ),
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
        from terok_sandbox.integrations.shield import ShieldManager

        cfg = _make_cfg(tmp_path)
        captured: list[SandboxConfig] = []
        real_init = ShieldManager.__init__

        def capturing_init(
            self,  # noqa: ANN001 — bound instance
            task_dir: Path,
            c: SandboxConfig | None = None,
            **kwargs: object,
        ) -> None:
            captured.append(c)
            real_init(self, task_dir, c, **kwargs)

        with (
            patch.object(ShieldManager, "__init__", capturing_init),
            patch.object(ShieldManager, "pre_start", return_value=[]),
        ):
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
        with patch("terok_sandbox.integrations.shield.ShieldManager.pre_start", return_value=[]):
            compose("myc", cfg=cfg, shield=True, gate=False, broker=False, scope=None)
        with (
            patch(
                "terok_sandbox.integrations.shield.ShieldManager.down",
                side_effect=OSError("nft gone"),
            ),
            patch("terok_sandbox.launch._resolve_container_id", return_value="ctr-uuid"),
        ):
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

    def test_reject_managed_volumes_blocks_runtime_dir_and_subpaths(self) -> None:
        """Mounts over ``CONTAINER_RUNTIME_DIR`` (or any descendant) are rejected."""
        from terok_sandbox.config import CONTAINER_RUNTIME_DIR

        for target in (CONTAINER_RUNTIME_DIR, f"{CONTAINER_RUNTIME_DIR}/anything"):
            with pytest.raises(SystemExit, match="managed by terok-sandbox"):
                reject_managed_volumes(["-v", f"/tmp/x:{target}"])

    def test_cleanup_db_missing_does_not_crash(self, tmp_path: Path) -> None:
        """A missing credential DB at cleanup time is treated as already-revoked."""
        cfg = _make_cfg(tmp_path)
        with (
            patch("terok_sandbox.integrations.shield.ShieldManager.pre_start", return_value=[]),
            patch("terok_sandbox.launch.mint_gate_token", return_value="t"),
            patch("terok_sandbox.vault.store.db.CredentialDB.create_token", return_value="p"),
        ):
            compose("myc", cfg=cfg, shield=True, gate=True, broker=True, scope="proj")
        # Now make CredentialDB construction fail at cleanup.
        with (
            patch("terok_sandbox.integrations.shield.ShieldManager.down"),
            patch(
                "terok_sandbox.config.SandboxConfig.open_credential_db",
                side_effect=OSError("db file vanished"),
            ),
        ):
            assert cleanup("myc", cfg=cfg) is True

    def test_handlers_construct_default_cfg(self, tmp_path: Path) -> None:
        """Handlers fall back to `SandboxConfig()` when *cfg* is omitted."""
        from terok_sandbox.commands import launch as launch_cmds

        fake_cfg = _make_cfg(tmp_path)
        with (
            patch.object(launch_cmds, "SandboxConfig", return_value=fake_cfg),
            patch("terok_sandbox.integrations.shield.ShieldManager.pre_start", return_value=[]),
            patch("terok_sandbox.launch.shutil.which", return_value="/usr/bin/podman"),
            patch("terok_sandbox.launch.Path.resolve", return_value=Path("/usr/bin/podman")),
            patch("terok_sandbox.launch.Path.is_file", return_value=True),
            patch("terok_sandbox.launch.os.access", return_value=True),
            patch("terok_sandbox.launch.os.execv"),
        ):
            _handle_prepare("a")
            _handle_run("b", podman_args=["ubuntu"])
            _handle_cleanup("a")


# ---------------------------------------------------------------------------
# _validate_container_name — the launch-side path-component guard
# ---------------------------------------------------------------------------


class TestValidateContainerName:
    """The name is interpolated into state + runtime dirs that get rmtree'd,
    so it must be a single safe path component."""

    def test_accepts_plain_name(self) -> None:
        """A simple alphanumeric name passes silently."""
        _validate_container_name("my-task-42")  # must not raise

    @pytest.mark.parametrize("bad", ["", "a/b", "/abs", "..", ".", "../evil"])
    def test_rejects_unsafe_names(self, bad: str) -> None:
        """Empty, separator-bearing, or parent-ref names are a hard SystemExit.

        Mirrors the supervisor-side ``load_sidecar`` guard — both ends
        refuse a name that could redirect filesystem operations."""
        with pytest.raises(SystemExit, match="unsafe container name"):
            _validate_container_name(bad)


# ---------------------------------------------------------------------------
# allocate_per_container_resources — per-container dir + (TCP) ports
# ---------------------------------------------------------------------------


class TestAllocatePerContainerResources:
    """Both modes get a 0700 per-container runtime dir; only TCP gets ports."""

    def test_socket_mode_makes_dir_and_no_ports(self, tmp_path: Path) -> None:
        """Socket mode: directory created mode 0700, all ports ``None``."""
        cfg = _make_cfg(tmp_path, services_mode="socket")
        res = allocate_per_container_resources(cfg, "myc")

        assert res.container_runtime_dir == cfg.runtime_dir / "run" / "myc"
        assert res.container_runtime_dir.is_dir()
        assert (res.container_runtime_dir.stat().st_mode & 0o777) == 0o700
        assert res.token_broker_port is None
        assert res.ssh_signer_port is None
        assert res.gate_port is None

    def test_tcp_mode_allocates_three_distinct_ports(self, tmp_path: Path) -> None:
        """TCP mode: three distinct free ports come back alongside the dir."""
        cfg = _make_cfg(tmp_path, services_mode="tcp")
        res = allocate_per_container_resources(cfg, "myc")

        assert res.container_runtime_dir.is_dir()
        ports = {res.token_broker_port, res.ssh_signer_port, res.gate_port}
        assert None not in ports
        # Allocated simultaneously against open sockets, so all three differ.
        assert len(ports) == 3


# ---------------------------------------------------------------------------
# _rollback_compose_state — best-effort teardown of a half-started launch
# ---------------------------------------------------------------------------


class TestRollbackComposeState:
    """When a launch aborts after minting tokens + creating dirs, the
    rollback revokes the phantom tokens and removes the durable state."""

    def test_revokes_tokens_and_removes_dirs(self, tmp_path: Path) -> None:
        """A plan with a scope revokes the container's tokens, then rmtrees
        both the per-container runtime dir and the state dir."""
        from unittest.mock import MagicMock

        cfg = _make_cfg(tmp_path)
        runtime_dir = tmp_path / "run" / "myc"
        runtime_dir.mkdir(parents=True)
        state_dir = tmp_path / "state" / "myc"
        state_dir.mkdir(parents=True)
        per = PerContainerResources(
            container_runtime_dir=runtime_dir,
            token_broker_port=None,
            ssh_signer_port=None,
            gate_port=None,
        )
        plan = WiringPlan(scope="proj", shield=True, gate=True, broker=True, ssh=True)
        db = MagicMock()

        with patch.object(SandboxConfig, "open_credential_db", return_value=db):
            _rollback_compose_state(cfg, "myc", plan, per, state_dir)

        db.revoke_tokens.assert_called_once_with("proj", "myc")
        db.close.assert_called_once()
        assert not runtime_dir.exists()
        assert not state_dir.exists()

    def test_no_scope_skips_token_revocation(self, tmp_path: Path) -> None:
        """A shield-only plan (no scope) has no tokens to revoke — the DB is
        never opened, but the dirs are still removed."""
        cfg = _make_cfg(tmp_path)
        runtime_dir = tmp_path / "run" / "myc"
        runtime_dir.mkdir(parents=True)
        state_dir = tmp_path / "state" / "myc"
        state_dir.mkdir(parents=True)
        per = PerContainerResources(
            container_runtime_dir=runtime_dir,
            token_broker_port=None,
            ssh_signer_port=None,
            gate_port=None,
        )
        plan = WiringPlan(scope=None, shield=True, gate=False, broker=False, ssh=False)

        with patch.object(SandboxConfig, "open_credential_db") as open_db:
            _rollback_compose_state(cfg, "myc", plan, per, state_dir)

        open_db.assert_not_called()
        assert not runtime_dir.exists()
        assert not state_dir.exists()

    def test_locked_vault_does_not_crash_rollback(self, tmp_path: Path) -> None:
        """If the vault won't open, rollback still removes the dirs (best-effort)."""
        cfg = _make_cfg(tmp_path)
        runtime_dir = tmp_path / "run" / "myc"
        runtime_dir.mkdir(parents=True)
        state_dir = tmp_path / "state" / "myc"
        state_dir.mkdir(parents=True)
        per = PerContainerResources(
            container_runtime_dir=runtime_dir,
            token_broker_port=None,
            ssh_signer_port=None,
            gate_port=None,
        )
        plan = WiringPlan(scope="proj", shield=True, gate=True, broker=False, ssh=False)

        with patch.object(SandboxConfig, "open_credential_db", side_effect=RuntimeError("locked")):
            _rollback_compose_state(cfg, "myc", plan, per, state_dir)

        assert not runtime_dir.exists()
        assert not state_dir.exists()


# ---------------------------------------------------------------------------
# _resolve_container_id — name → full podman UUID at cleanup time
# ---------------------------------------------------------------------------


class TestResolveContainerId:
    """Resolves the operator-facing name to podman's full UUID, soft-failing
    to ``None`` whenever podman can't answer."""

    def test_returns_full_uuid_on_success(self) -> None:
        """A successful ``podman inspect`` yields the stripped UUID."""
        from unittest.mock import MagicMock

        result = MagicMock(returncode=0, stdout="abc123def456\n")
        with (
            patch("terok_sandbox.launch.shutil.which", return_value="/usr/bin/podman"),
            patch("terok_sandbox.launch.subprocess.run", return_value=result),
        ):
            assert _resolve_container_id("myc") == "abc123def456"

    def test_returns_none_when_podman_missing(self) -> None:
        """No podman on PATH ⇒ nothing to resolve."""
        with patch("terok_sandbox.launch.shutil.which", return_value=None):
            assert _resolve_container_id("myc") is None

    def test_returns_none_on_subprocess_error(self) -> None:
        """A subprocess OSError/timeout collapses to ``None``."""
        with (
            patch("terok_sandbox.launch.shutil.which", return_value="/usr/bin/podman"),
            patch("terok_sandbox.launch.subprocess.run", side_effect=OSError("boom")),
        ):
            assert _resolve_container_id("myc") is None

    def test_returns_none_on_nonzero_exit(self) -> None:
        """A non-zero ``podman inspect`` (container already pruned) ⇒ ``None``."""
        from unittest.mock import MagicMock

        result = MagicMock(returncode=125, stdout="")
        with (
            patch("terok_sandbox.launch.shutil.which", return_value="/usr/bin/podman"),
            patch("terok_sandbox.launch.subprocess.run", return_value=result),
        ):
            assert _resolve_container_id("myc") is None

    def test_returns_none_on_empty_stdout(self) -> None:
        """A zero exit but empty stdout (unexpected) still soft-fails to ``None``."""
        from unittest.mock import MagicMock

        result = MagicMock(returncode=0, stdout="  \n")
        with (
            patch("terok_sandbox.launch.shutil.which", return_value="/usr/bin/podman"),
            patch("terok_sandbox.launch.subprocess.run", return_value=result),
        ):
            assert _resolve_container_id("myc") is None
