# SPDX-FileCopyrightText: 2026 Jiri Vyskocil
# SPDX-License-Identifier: Apache-2.0

"""Tests for the ``vault status`` CLI verb and its chain probe.

``vault status`` is a read-only diagnostic.  It walks the passphrase
resolution chain *without short-circuiting* (so a session file
shadowing a durable systemd-creds / keyring tier is visible), reports
the lock state, re-states the plaintext-on-disk and unconfirmed-recovery
warnings, and lists stored credential providers on a best-effort DB
open.  The probe ([`probe_passphrase_chain`][terok_sandbox.vault.store.encryption.probe_passphrase_chain])
is pure and exercised directly; the handler is driven through a mock
``SandboxConfig`` with the recovery / plaintext seams patched.
"""

from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import pytest

from terok_sandbox.commands.vault import _handle_vault_status
from terok_sandbox.vault.store import encryption
from terok_sandbox.vault.store.encryption import probe_passphrase_chain

# Captured at import time — before conftest's autouse ``_isolate_credential_keyring``
# stubs ``load_passphrase_from_file`` to ``None`` — so file-tier tests can restore
# the real reader.  Same idiom as ``test_credential_encryption.py``.
from terok_sandbox.vault.store.encryption import (  # noqa: E402  isort: skip
    load_passphrase_from_file as _real_load_file,
)


@pytest.fixture
def real_file_tier(monkeypatch: pytest.MonkeyPatch) -> None:
    """Undo the conftest stub so the session-file tier reads real files."""
    monkeypatch.setattr(encryption, "load_passphrase_from_file", _real_load_file)


class TestProbePassphraseChain:
    """``probe_passphrase_chain`` reports per-tier presence in resolution order."""

    def test_empty_chain_all_absent(self) -> None:
        chain = probe_passphrase_chain()
        assert [t.source for t in chain] == [
            "session-file",
            "systemd-creds",
            "keyring",
            "passphrase-command",
            "config",
        ]
        assert all(not t.present for t in chain)

    def test_session_file_present_when_nonempty(self, tmp_path: Path, real_file_tier: None) -> None:
        session = tmp_path / "vault.passphrase"
        session.write_text("hunter2\n")
        chain = probe_passphrase_chain(passphrase_file=session)
        assert chain[0].source == "session-file"
        assert chain[0].present is True

    def test_empty_session_file_is_absent(self, tmp_path: Path, real_file_tier: None) -> None:
        session = tmp_path / "vault.passphrase"
        session.write_text("")  # SQLCipher no-encryption sentinel — treat as absent
        chain = probe_passphrase_chain(passphrase_file=session)
        assert chain[0].present is False

    def test_systemd_creds_present_when_sealed_file_exists(self, tmp_path: Path) -> None:
        sealed = tmp_path / "vault.passphrase.cred"
        sealed.write_text("sealed-blob")
        chain = probe_passphrase_chain(systemd_creds_file=sealed)
        assert chain[1].source == "systemd-creds"
        assert chain[1].present is True

    def test_systemd_creds_not_unsealed(self, tmp_path: Path) -> None:
        """Presence is file existence — the probe must never call unseal()."""
        sealed = tmp_path / "vault.passphrase.cred"
        sealed.write_text("sealed-blob")
        with patch.object(encryption, "_systemd_creds") as creds:
            probe_passphrase_chain(systemd_creds_file=sealed)
        creds.unseal.assert_not_called()

    def test_keyring_only_probed_when_enabled(self) -> None:
        with patch.object(encryption, "load_passphrase_from_keyring", return_value="k") as load:
            on = probe_passphrase_chain(use_keyring=True)
            assert on[2].present is True
            off = probe_passphrase_chain(use_keyring=False)
            assert off[2].present is False
        # one lookup for the enabled probe, none for the disabled one
        assert load.call_count == 1

    def test_passphrase_command_present_but_not_executed(self) -> None:
        chain = probe_passphrase_chain(passphrase_command="pass show vault")
        assert chain[3].source == "passphrase-command"
        assert chain[3].present is True
        assert "not executed" in chain[3].detail

    def test_config_fallback_present(self) -> None:
        chain = probe_passphrase_chain(config_fallback="plaintext-secret")
        assert chain[4].source == "config"
        assert chain[4].present is True


def _status_cfg(
    *,
    session: Path | None = None,
    sealed: Path | None = None,
    use_keyring: bool = False,
    passphrase_command: str | None = None,
    config_passphrase: str | None = None,
    db: MagicMock | None = None,
    db_error: Exception | None = None,
    marker: Path | None = None,
) -> MagicMock:
    """A mock ``SandboxConfig`` exposing exactly the knobs ``status`` reads."""
    cfg = MagicMock()
    cfg.vault_passphrase_file = session or Path("/nonexistent/session")
    cfg.vault_systemd_creds_file = sealed or Path("/nonexistent/sealed")
    cfg.credentials_use_keyring = use_keyring
    cfg.credentials_passphrase_command = passphrase_command
    cfg.credentials_passphrase = config_passphrase
    cfg.db_path = Path("/var/lib/terok/vault/credentials.db")
    cfg.vault_recovery_marker_file = marker or Path("/nonexistent/marker")
    if db_error is not None:
        cfg.open_credential_db.side_effect = db_error
    else:
        cfg.open_credential_db.return_value = db or MagicMock(
            list_credential_sets=MagicMock(return_value=[]),
        )
    return cfg


def _run_status(
    cfg: MagicMock,
    *,
    acknowledged: bool = False,
    plaintext: Path | None = None,
    as_json: bool = False,
) -> None:
    """Drive the handler with the recovery / plaintext seams pinned."""
    with (
        patch(
            "terok_sandbox.vault.store.recovery.RecoveryStatus.load",
            return_value=SimpleNamespace(acknowledged=acknowledged, source=None),
        ),
        patch("terok_sandbox.paths.plaintext_passphrase_config_path", return_value=plaintext),
    ):
        _handle_vault_status(cfg=cfg, as_json=as_json)


class TestHandleVaultStatusText:
    """Human-readable rendering of the lock state and chain."""

    def test_locked_when_no_tier_present(self, capsys: pytest.CaptureFixture[str]) -> None:
        cfg = _status_cfg(db_error=RuntimeError("locked"))
        _run_status(cfg)
        out = capsys.readouterr().out
        assert "Vault: LOCKED" in out
        assert "terok vault unlock" in out
        assert "Credentials: vault locked" in out

    def test_default_cfg_branch(self, capsys: pytest.CaptureFixture[str]) -> None:
        """``cfg=None`` constructs a default ``SandboxConfig`` rather than crashing."""
        cfg = _status_cfg(db_error=RuntimeError("locked"))
        with (
            patch("terok_sandbox.commands.vault.SandboxConfig", return_value=cfg) as ctor,
            patch(
                "terok_sandbox.vault.store.recovery.RecoveryStatus.load",
                return_value=SimpleNamespace(acknowledged=False, source=None),
            ),
            patch("terok_sandbox.paths.plaintext_passphrase_config_path", return_value=None),
        ):
            _handle_vault_status()  # cfg omitted → default-construction branch
        ctor.assert_called_once_with()
        assert "Vault: LOCKED" in capsys.readouterr().out

    def test_unlocked_names_active_tier(
        self, tmp_path: Path, capsys: pytest.CaptureFixture[str]
    ) -> None:
        sealed = tmp_path / "sealed.cred"
        sealed.write_text("blob")
        cfg = _status_cfg(sealed=sealed)
        _run_status(cfg)
        out = capsys.readouterr().out
        assert "passphrase via systemd-creds" in out
        assert "systemd-creds       active" in out

    def test_session_file_shadows_durable_tier(
        self, tmp_path: Path, capsys: pytest.CaptureFixture[str], real_file_tier: None
    ) -> None:
        session = tmp_path / "session"
        session.write_text("pw\n")
        sealed = tmp_path / "sealed.cred"
        sealed.write_text("blob")
        cfg = _status_cfg(session=session, sealed=sealed)
        _run_status(cfg)
        out = capsys.readouterr().out
        assert "session-file        active" in out
        assert "systemd-creds       shadowed" in out
        assert "shadowing a durable tier (systemd-creds)" in out

    def test_unacknowledged_recovery_warns(self, capsys: pytest.CaptureFixture[str]) -> None:
        cfg = _status_cfg(db_error=RuntimeError("locked"))
        _run_status(cfg, acknowledged=False)
        assert "Recovery key: NOT acknowledged" in capsys.readouterr().out

    def test_plaintext_passphrase_warning(
        self, tmp_path: Path, capsys: pytest.CaptureFixture[str]
    ) -> None:
        cfg = _status_cfg(config_passphrase="secret")
        _run_status(cfg, plaintext=Path("/etc/terok/config.yml"))
        out = capsys.readouterr().out
        assert "plaintext at /etc/terok/config.yml" in out

    def test_credentials_listed_when_open(self, capsys: pytest.CaptureFixture[str]) -> None:
        db = MagicMock()
        db.list_credential_sets.return_value = ["default"]
        db.list_credentials.return_value = ["github", "openai"]
        cfg = _status_cfg(config_passphrase="secret", db=db)
        _run_status(cfg)
        out = capsys.readouterr().out
        assert "Credentials: 2 stored (github, openai)" in out


class TestHandleVaultStatusJson:
    """``--json`` carries the same facts in a machine-readable shape."""

    def test_json_shape(
        self, tmp_path: Path, capsys: pytest.CaptureFixture[str], real_file_tier: None
    ) -> None:
        session = tmp_path / "session"
        session.write_text("pw\n")
        sealed = tmp_path / "sealed.cred"
        sealed.write_text("blob")
        cfg = _status_cfg(session=session, sealed=sealed, db_error=RuntimeError("x"))
        _run_status(cfg, acknowledged=True, as_json=True)
        data = json.loads(capsys.readouterr().out)
        assert data["locked"] is False
        assert data["passphrase_source"] == "session-file"
        assert data["shadowed_tiers"] == ["systemd-creds"]
        assert data["recovery_acknowledged"] is True
        assert data["credentials"] is None  # DB wouldn't open
        assert [c["source"] for c in data["chain"]][0] == "session-file"
