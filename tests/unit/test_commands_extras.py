# SPDX-FileCopyrightText: 2026 Jiri Vyskocil
# SPDX-License-Identifier: Apache-2.0

"""Tests for command handlers: gate-stop branches, vault install/uninstall,
shield-status setup hint, doctor, and SSH key removal helpers.
"""

from __future__ import annotations

import json
import subprocess
from collections.abc import Iterator
from contextlib import contextmanager
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from terok_sandbox.commands import (
    KeyRow,
    _build_key_rows,
    _delete_key_files,
    _handle_doctor,
    _handle_gate_stop,
    _handle_shield_status,
    _handle_vault_install,
    _handle_vault_uninstall,
    _prompt_file_action,
    _remove_keys_from_json,
    _validate_scope_exists,
)
from terok_sandbox.config import SandboxConfig
from terok_sandbox.doctor import CheckVerdict, DoctorCheck
from terok_sandbox.gate.lifecycle import GateServerManager, GateServerStatus
from terok_sandbox.vault.lifecycle import VaultManager

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _gate_status(mode: str) -> GateServerStatus:
    """Build a minimal GateServerStatus with the given mode."""
    return GateServerStatus(mode=mode, running=False, port=9418)


def _make_cfg(tmp_path: Path) -> SandboxConfig:
    """Build a SandboxConfig rooted at *tmp_path* (TCP mode → ports auto-allocated)."""
    return SandboxConfig(
        state_dir=tmp_path / "state",
        runtime_dir=tmp_path / "run",
        config_dir=tmp_path / "config",
        vault_dir=tmp_path / "vault",
    )


# ---------------------------------------------------------------------------
# _handle_gate_stop "not running" branch (line 106)
# ---------------------------------------------------------------------------


class TestHandleGateStopNotRunning:
    """When status.mode is neither 'systemd' nor 'daemon', print the idle message."""

    def test_prints_not_running(self, capsys: pytest.CaptureFixture[str]) -> None:
        with patch.object(GateServerManager, "get_status", return_value=_gate_status("none")):
            _handle_gate_stop()
        out = capsys.readouterr().out
        assert "not running" in out


# ---------------------------------------------------------------------------
# _handle_shield_status setup-hint branch (line 155)
# ---------------------------------------------------------------------------


class TestHandleShieldStatusHint:
    """When shield needs setup, the hint is written to stderr."""

    def test_setup_hint_emitted_on_stderr(self, capsys: pytest.CaptureFixture[str]) -> None:
        env = MagicMock(hooks="missing", health="degraded", needs_setup=True, setup_hint="Run X.")
        cfg = {"mode": "hook", "profiles": ["dev-standard"], "audit_enabled": True}
        # The handler imports check_environment + status from .shield at runtime,
        # so patching the shield module is what takes effect.
        with (
            patch("terok_sandbox.shield.check_environment", return_value=env),
            patch("terok_sandbox.shield.status", return_value=cfg),
        ):
            _handle_shield_status()
        captured = capsys.readouterr()
        assert "Run X." in captured.err

    def test_no_hint_when_not_needed(self, capsys: pytest.CaptureFixture[str]) -> None:
        env = MagicMock(hooks="ok", health="ok", needs_setup=False, setup_hint="should not show")
        cfg = {"mode": "hook", "profiles": [], "audit_enabled": False}
        with (
            patch("terok_sandbox.shield.check_environment", return_value=env),
            patch("terok_sandbox.shield.status", return_value=cfg),
        ):
            _handle_shield_status()
        assert "should not show" not in capsys.readouterr().err


# ---------------------------------------------------------------------------
# _handle_vault_install / _handle_vault_uninstall — systemd-unavailable branch
# ---------------------------------------------------------------------------


class TestHandleVaultInstall:
    """Installer fails loudly when the systemd user session is unavailable."""

    def test_install_systemd_unavailable_exits_1(self, capsys: pytest.CaptureFixture[str]) -> None:
        with patch.object(VaultManager, "is_systemd_available", return_value=False):
            with pytest.raises(SystemExit) as exc:
                _handle_vault_install()
        assert exc.value.code == 1
        assert "systemd" in capsys.readouterr().out

    def test_install_systemd_available_calls_install(
        self, capsys: pytest.CaptureFixture[str]
    ) -> None:
        with (
            patch.object(VaultManager, "is_systemd_available", return_value=True),
            patch.object(VaultManager, "install_systemd_units") as install,
        ):
            _handle_vault_install()
        install.assert_called_once()
        assert "installed" in capsys.readouterr().out.lower()


class TestHandleVaultUninstall:
    """Uninstaller mirrors install: fail loudly when systemd missing."""

    def test_uninstall_systemd_unavailable_exits_1(
        self, capsys: pytest.CaptureFixture[str]
    ) -> None:
        with patch.object(VaultManager, "is_systemd_available", return_value=False):
            with pytest.raises(SystemExit) as exc:
                _handle_vault_uninstall()
        assert exc.value.code == 1
        assert "Nothing to uninstall" in capsys.readouterr().out

    def test_uninstall_systemd_available_calls_uninstall(
        self, capsys: pytest.CaptureFixture[str]
    ) -> None:
        with (
            patch.object(VaultManager, "is_systemd_available", return_value=True),
            patch.object(VaultManager, "uninstall_systemd_units") as un,
        ):
            _handle_vault_uninstall()
        un.assert_called_once()
        assert "removed" in capsys.readouterr().out.lower()


# ---------------------------------------------------------------------------
# _validate_scope_exists
# ---------------------------------------------------------------------------


class TestValidateScopeExists:
    """Reject unknown scopes; tolerate missing or malformed JSON."""

    def test_known_scope_passes(self, tmp_path: Path) -> None:
        cfg = _make_cfg(tmp_path)
        cfg.ssh_keys_json_path.parent.mkdir(parents=True, exist_ok=True)
        cfg.ssh_keys_json_path.write_text(json.dumps({"scope-a": [{"private_key": "k"}]}))
        # Must not raise
        _validate_scope_exists("scope-a", create_scope=False, cfg=cfg)

    def test_unknown_scope_raises_with_known_list(self, tmp_path: Path) -> None:
        cfg = _make_cfg(tmp_path)
        cfg.ssh_keys_json_path.parent.mkdir(parents=True, exist_ok=True)
        cfg.ssh_keys_json_path.write_text(json.dumps({"a": [], "b": []}))
        with pytest.raises(SystemExit) as exc:
            _validate_scope_exists("unknown", create_scope=False, cfg=cfg)
        assert "Unknown scope" in str(exc.value)
        assert "a, b" in str(exc.value)

    def test_create_scope_bypasses_check(self, tmp_path: Path) -> None:
        cfg = _make_cfg(tmp_path)
        # File doesn't exist → empty existing — but create_scope=True bypasses
        _validate_scope_exists("anything", create_scope=True, cfg=cfg)

    def test_malformed_json_swallowed(self, tmp_path: Path) -> None:
        """A corrupt ssh-keys.json should not crash validation —
        the empty-existing branch produces an Unknown-scope error."""
        cfg = _make_cfg(tmp_path)
        cfg.ssh_keys_json_path.parent.mkdir(parents=True, exist_ok=True)
        cfg.ssh_keys_json_path.write_text("not json")
        with pytest.raises(SystemExit, match="Unknown scope"):
            _validate_scope_exists("foo", create_scope=False, cfg=cfg)


# ---------------------------------------------------------------------------
# _build_key_rows error branches
# ---------------------------------------------------------------------------


class TestBuildKeyRowsErrors:
    """Tolerant of missing files; loud on corrupt input."""

    def test_missing_file_returns_empty(self, tmp_path: Path) -> None:
        assert _build_key_rows(_make_cfg(tmp_path)) == []

    def test_corrupt_json_raises_systemexit(self, tmp_path: Path) -> None:
        cfg = _make_cfg(tmp_path)
        cfg.ssh_keys_json_path.parent.mkdir(parents=True, exist_ok=True)
        cfg.ssh_keys_json_path.write_text("{not valid json")
        with pytest.raises(SystemExit, match="Cannot read"):
            _build_key_rows(cfg)

    def test_non_dict_raises_systemexit(self, tmp_path: Path) -> None:
        cfg = _make_cfg(tmp_path)
        cfg.ssh_keys_json_path.parent.mkdir(parents=True, exist_ok=True)
        cfg.ssh_keys_json_path.write_text("[1, 2, 3]")
        with pytest.raises(SystemExit, match="expected top-level JSON object"):
            _build_key_rows(cfg)

    def test_skips_non_list_scope_entries(self, tmp_path: Path) -> None:
        cfg = _make_cfg(tmp_path)
        cfg.ssh_keys_json_path.parent.mkdir(parents=True, exist_ok=True)
        cfg.ssh_keys_json_path.write_text(
            json.dumps({"a": "wrong-type", "b": [{"private_key": "/x", "public_key": "/y"}]})
        )
        rows = _build_key_rows(cfg)
        # Only b is processed; a is skipped silently.
        assert all(r.scope == "b" for r in rows)

    def test_pub_missing_yields_placeholder(self, tmp_path: Path) -> None:
        cfg = _make_cfg(tmp_path)
        cfg.ssh_keys_json_path.parent.mkdir(parents=True, exist_ok=True)
        cfg.ssh_keys_json_path.write_text(
            json.dumps({"a": [{"private_key": "/k", "public_key": "/missing.pub"}]})
        )
        rows = _build_key_rows(cfg)
        assert len(rows) == 1
        assert rows[0].fingerprint == "(pub missing)"


# ---------------------------------------------------------------------------
# _remove_keys_from_json
# ---------------------------------------------------------------------------


class TestRemoveKeysFromJson:
    """Mutates ssh-keys.json under flock; deletes empty scopes."""

    def test_removes_matching_entry_and_keeps_others(self, tmp_path: Path) -> None:
        path = tmp_path / "ssh-keys.json"
        path.write_text(
            json.dumps(
                {
                    "a": [
                        {"private_key": "/keep", "public_key": "/keep.pub"},
                        {"private_key": "/drop", "public_key": "/drop.pub"},
                    ],
                    "b": [{"private_key": "/drop2", "public_key": "/drop2.pub"}],
                }
            )
        )
        rows = [KeyRow("a", "c", "ed25519", "fp", "/drop", "/drop.pub")]
        _remove_keys_from_json(path, rows)
        result = json.loads(path.read_text())
        assert {"private_key": "/keep", "public_key": "/keep.pub"} in result["a"]
        assert all(e.get("private_key") != "/drop" for e in result["a"])
        assert "b" in result  # untouched

    def test_empty_scope_is_removed(self, tmp_path: Path) -> None:
        path = tmp_path / "ssh-keys.json"
        path.write_text(json.dumps({"only": [{"private_key": "/x", "public_key": "/x.pub"}]}))
        rows = [KeyRow("only", "c", "ed25519", "fp", "/x", "/x.pub")]
        _remove_keys_from_json(path, rows)
        assert "only" not in json.loads(path.read_text())

    def test_corrupt_json_raises_systemexit(self, tmp_path: Path) -> None:
        path = tmp_path / "ssh-keys.json"
        path.write_text("{nope")
        with pytest.raises(SystemExit, match="Cannot read"):
            _remove_keys_from_json(path, removals=[])

    def test_non_dict_raises_systemexit(self, tmp_path: Path) -> None:
        path = tmp_path / "ssh-keys.json"
        path.write_text("[1]")
        with pytest.raises(SystemExit, match="expected top-level JSON object"):
            _remove_keys_from_json(path, removals=[])

    def test_empty_file_treated_as_empty_dict(self, tmp_path: Path) -> None:
        path = tmp_path / "ssh-keys.json"
        path.touch()
        # Should not raise — empty file becomes {}; nothing to remove
        _remove_keys_from_json(path, removals=[])
        assert json.loads(path.read_text()) == {}


# ---------------------------------------------------------------------------
# _delete_key_files
# ---------------------------------------------------------------------------


class TestDeleteKeyFiles:
    """File deletions are scoped to the managed directory; failures collected."""

    def test_deletes_files_in_managed_dir(self, tmp_path: Path) -> None:
        cfg = _make_cfg(tmp_path)
        keys_dir = cfg.ssh_keys_dir / "scope"
        keys_dir.mkdir(parents=True)
        priv = keys_dir / "id"
        pub = keys_dir / "id.pub"
        priv.write_text("PRIV")
        pub.write_text("PUB")
        rows = [KeyRow("scope", "c", "ed25519", "fp", str(priv), str(pub))]

        deleted, errors = _delete_key_files(rows, cfg)
        assert deleted == 2
        assert errors == []
        assert not priv.exists()
        assert not pub.exists()

    def test_refuses_paths_outside_managed_dir(self, tmp_path: Path) -> None:
        cfg = _make_cfg(tmp_path)
        cfg.ssh_keys_dir.mkdir(parents=True, exist_ok=True)
        outside = tmp_path / "elsewhere"
        outside.mkdir()
        rogue_priv = outside / "id"
        rogue_pub = outside / "id.pub"
        rogue_priv.write_text("PRIV")
        rogue_pub.write_text("PUB")

        rows = [KeyRow("s", "c", "ed25519", "fp", str(rogue_priv), str(rogue_pub))]
        deleted, errors = _delete_key_files(rows, cfg)
        assert deleted == 0
        assert len(errors) == 2
        assert all("Refusing to delete" in e for e in errors)
        # Files survived
        assert rogue_priv.exists()
        assert rogue_pub.exists()

    def test_missing_files_silently_skipped(self, tmp_path: Path) -> None:
        cfg = _make_cfg(tmp_path)
        rows = [KeyRow("s", "c", "ed25519", "fp", "/nope/priv", "/nope/pub")]
        deleted, errors = _delete_key_files(rows, cfg)
        assert deleted == 0
        assert errors == []

    def test_unlink_failure_recorded_as_error(self, tmp_path: Path) -> None:
        """OSError during unlink is caught and reported."""
        cfg = _make_cfg(tmp_path)
        keys_dir = cfg.ssh_keys_dir / "s"
        keys_dir.mkdir(parents=True)
        priv = keys_dir / "id"
        priv.write_text("x")
        rows = [KeyRow("s", "c", "ed25519", "fp", str(priv), str(priv))]

        # Patch Path.unlink at the source
        from pathlib import Path as PPath

        with patch.object(PPath, "unlink", side_effect=OSError("perm denied")):
            deleted, errors = _delete_key_files(rows, cfg)
        assert deleted == 0
        assert errors and "perm denied" in errors[0]


# ---------------------------------------------------------------------------
# _prompt_file_action
# ---------------------------------------------------------------------------


class TestPromptFileAction:
    """Resolves the keep-vs-delete decision from flags or interactive prompt."""

    def test_both_flags_raises(self) -> None:
        with pytest.raises(SystemExit, match="both"):
            _prompt_file_action(delete_files=True, keep_files=True)

    def test_delete_flag_returns_true(self) -> None:
        assert _prompt_file_action(delete_files=True, keep_files=False) is True

    def test_keep_flag_returns_false(self) -> None:
        assert _prompt_file_action(delete_files=False, keep_files=True) is False

    def test_yes_without_explicit_flag_returns_false(self) -> None:
        # --yes without --delete-files defaults to keeping files (safer default).
        assert _prompt_file_action(delete_files=False, keep_files=False, yes=True) is False

    def test_prompt_yes_returns_true(self) -> None:
        with patch("builtins.input", return_value="y"):
            assert _prompt_file_action(delete_files=False, keep_files=False) is True

    def test_prompt_no_returns_false(self) -> None:
        with patch("builtins.input", return_value="N"):
            assert _prompt_file_action(delete_files=False, keep_files=False) is False

    def test_eof_returns_false(self) -> None:
        with patch("builtins.input", side_effect=EOFError):
            assert _prompt_file_action(delete_files=False, keep_files=False) is False

    def test_keyboard_interrupt_returns_false(self) -> None:
        with patch("builtins.input", side_effect=KeyboardInterrupt):
            assert _prompt_file_action(delete_files=False, keep_files=False) is False


# ---------------------------------------------------------------------------
# _handle_doctor — the largest uncovered chunk (lines 975-1023)
# ---------------------------------------------------------------------------


def _make_check(
    *,
    label: str = "test",
    severity: str,
    detail: str = "",
    host_side: bool = False,
    probe_cmd: list[str] | None = None,
) -> DoctorCheck:
    """Build a DoctorCheck whose evaluate returns a fixed verdict."""
    return DoctorCheck(
        category="test",
        label=label,
        probe_cmd=probe_cmd or ["true"],
        evaluate=lambda rc, out, err: CheckVerdict(severity=severity, detail=detail),
        host_side=host_side,
    )


@contextmanager
def _doctor_patches(
    checks: list[DoctorCheck],
    *,
    subprocess_side_effect: object = None,
) -> Iterator[MagicMock]:
    """Patch ``sandbox_doctor_checks``, ``subprocess.run``, and VaultManager ports.

    Yields the ``subprocess.run`` mock so tests can inspect calls or override
    return_value.  By default the mock returns a successful CompletedProcess.
    Pass *subprocess_side_effect* (e.g. ``FileNotFoundError``, a
    ``TimeoutExpired`` instance) to simulate probe failures.
    """
    with (
        patch("terok_sandbox.doctor.sandbox_doctor_checks", return_value=checks),
        patch("subprocess.run") as run,
        patch.object(VaultManager, "token_broker_port", new=1),
        patch.object(VaultManager, "ssh_signer_port", new=2),
    ):
        if subprocess_side_effect is not None:
            run.side_effect = subprocess_side_effect
        else:
            run.return_value = subprocess.CompletedProcess(
                args=[], returncode=0, stdout="", stderr=""
            )
        yield run


class TestHandleDoctor:
    """The standalone doctor runs each check and exits per worst severity."""

    def test_all_ok_exits_normally(self, capsys: pytest.CaptureFixture[str]) -> None:
        checks = [_make_check(label="A", severity="ok", detail="fine")]
        with _doctor_patches(checks):
            _handle_doctor()  # must not raise
        out = capsys.readouterr().out
        assert "A" in out and "ok" in out

    def test_warn_exits_1(self, capsys: pytest.CaptureFixture[str]) -> None:
        checks = [_make_check(label="W", severity="warn", detail="be careful")]
        with _doctor_patches(checks), pytest.raises(SystemExit) as exc:
            _handle_doctor()
        assert exc.value.code == 1
        assert "WARN" in capsys.readouterr().out

    def test_error_exits_2(self, capsys: pytest.CaptureFixture[str]) -> None:
        checks = [
            _make_check(label="A", severity="ok"),
            _make_check(label="B", severity="error", detail="boom"),
        ]
        with _doctor_patches(checks), pytest.raises(SystemExit) as exc:
            _handle_doctor()
        assert exc.value.code == 2
        assert "ERROR" in capsys.readouterr().out

    def test_host_side_check_skips_subprocess(self) -> None:
        """host_side=True checks call evaluate(0,'','') directly, no subprocess."""
        checks = [_make_check(label="H", severity="ok", host_side=True)]
        with _doctor_patches(checks) as run:
            _handle_doctor()
        run.assert_not_called()

    def test_probe_unavailable_yields_unavailable_verdict(self) -> None:
        """FileNotFoundError from probe → evaluate is called with rc=1 and 'unavailable'."""
        captured: dict = {}

        def evaluate(rc: int, out: str, err: str) -> CheckVerdict:
            captured["rc"], captured["err"] = rc, err
            return CheckVerdict(severity="warn", detail="probe missing")

        check = DoctorCheck(
            category="net", label="P", probe_cmd=["nonexistent-binary"], evaluate=evaluate
        )
        with (
            _doctor_patches([check], subprocess_side_effect=FileNotFoundError),
            pytest.raises(SystemExit),
        ):
            _handle_doctor()
        assert captured["rc"] == 1
        assert "unavailable" in captured["err"]

    def test_probe_timeout_yields_unavailable_verdict(self) -> None:
        """TimeoutExpired from probe → evaluate is called with rc=1 and 'unavailable'."""
        captured: dict = {}

        def evaluate(rc: int, out: str, err: str) -> CheckVerdict:
            captured["rc"], captured["err"] = rc, err
            return CheckVerdict(severity="warn", detail="t/o")

        check = DoctorCheck(category="net", label="P", probe_cmd=["sleep", "9"], evaluate=evaluate)
        timeout = subprocess.TimeoutExpired("sleep", 5)
        with _doctor_patches([check], subprocess_side_effect=timeout), pytest.raises(SystemExit):
            _handle_doctor()
        assert captured["rc"] == 1
        assert "unavailable" in captured["err"]

    def test_check_without_probe_cmd_calls_evaluate_directly(self) -> None:
        """A check with no probe_cmd and host_side=False still gets evaluate(0,'','')."""
        called: list[tuple] = []

        def evaluate(rc: int, out: str, err: str) -> CheckVerdict:
            called.append((rc, out, err))
            return CheckVerdict(severity="ok", detail="")

        check = DoctorCheck(category="x", label="L", probe_cmd=[], evaluate=evaluate)
        with _doctor_patches([check]) as run:
            _handle_doctor()
        run.assert_not_called()
        assert called == [(0, "", "")]
