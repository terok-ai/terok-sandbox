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
        assert "exists but unreadable or empty" in chain[0].detail

    def test_unreadable_session_file_is_flagged(
        self, tmp_path: Path, real_file_tier: None, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """A blocked read (EACCES / SELinux) must not masquerade as a locked vault."""
        session = tmp_path / "vault.passphrase"
        session.write_text("pw\n")
        real_read_text = Path.read_text

        def _denied(self: Path, *args: object, **kwargs: object) -> str:
            if self == session:
                raise PermissionError(13, "Permission denied")
            return real_read_text(self, *args, **kwargs)  # type: ignore[arg-type]

        monkeypatch.setattr(Path, "read_text", _denied)
        chain = probe_passphrase_chain(passphrase_file=session)
        assert chain[0].present is False
        assert "exists but unreadable or empty" in chain[0].detail

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

    def test_keyring_empty_string_is_absent(self) -> None:
        """An empty keyring value is the resolver's no-passphrase sentinel — treat as absent."""
        with patch.object(encryption, "load_passphrase_from_keyring", return_value=""):
            chain = probe_passphrase_chain(use_keyring=True)
        assert chain[2].present is False

    def test_passphrase_command_present_but_not_executed(self) -> None:
        chain = probe_passphrase_chain(passphrase_command="pass show vault")
        assert chain[3].source == "passphrase-command"
        assert chain[3].present is True
        assert "not executed" in chain[3].detail

    def test_config_fallback_present(self) -> None:
        chain = probe_passphrase_chain(config_fallback="plaintext-secret")
        assert chain[4].source == "config"
        assert chain[4].present is True


def _recovery(source: str | None = None, resolve_error: str | None = None) -> SimpleNamespace:
    """A ``RecoveryStatus`` stand-in with just the fields the classifier reads."""
    return SimpleNamespace(acknowledged=False, source=source, resolve_error=resolve_error)


class TestClassifyVaultAccess:
    """``_classify_vault_access`` separates the three operator problems 'locked' hides."""

    def test_broken_tier_reports_resolve_error(self) -> None:
        """A fail-closed resolver (broken seal / dead helper) is named, not just 'locked'."""
        from terok_sandbox.commands.vault import _classify_vault_access

        cfg = MagicMock()
        reason, providers, db_error = _classify_vault_access(
            cfg, _recovery(resolve_error="sealed credential present but could not be unsealed")
        )
        assert reason is not None and "unreadable" in reason
        assert "could not be unsealed" in reason
        assert providers is None and db_error is None
        cfg.open_credential_db.assert_not_called()  # nothing to try — resolution already failed

    def test_no_passphrase_anywhere(self) -> None:
        from terok_sandbox.commands.vault import _classify_vault_access

        reason, providers, db_error = _classify_vault_access(MagicMock(), _recovery())
        assert reason == "no passphrase in any tier"
        assert providers is None and db_error is None

    def test_wrong_passphrase_names_the_tier(self) -> None:
        """A resolved value the DB rejects points at the tier carrying the bad key."""
        from terok_sandbox.commands.vault import _classify_vault_access
        from terok_sandbox.vault.store.encryption import WrongPassphraseError

        cfg = MagicMock()
        cfg.open_credential_db.side_effect = WrongPassphraseError("could not decrypt")
        reason, providers, db_error = _classify_vault_access(cfg, _recovery(source="keyring"))
        assert reason is not None and "via keyring does not open the DB" in reason
        assert providers is None and db_error is None

    def test_open_no_passphrase_race_is_plain_lock(self) -> None:
        """A tier that vanishes between the resolve and the open is a plain lock."""
        from terok_sandbox.commands.vault import _classify_vault_access
        from terok_sandbox.vault.store.encryption import NoPassphraseError

        cfg = MagicMock()
        cfg.open_credential_db.side_effect = NoPassphraseError("tier gone")
        reason, providers, db_error = _classify_vault_access(cfg, _recovery(source="config"))
        assert reason == "no passphrase in any tier"
        assert providers is None and db_error is None

    def test_system_exit_propagates(self) -> None:
        """An explicit exit from a lower layer must not be stringified into status."""
        from terok_sandbox.commands.vault import _classify_vault_access

        cfg = MagicMock()
        cfg.open_credential_db.side_effect = SystemExit(3)
        with pytest.raises(SystemExit):
            _classify_vault_access(cfg, _recovery(source="config"))

    def test_open_ok_lists_providers(self) -> None:
        from terok_sandbox.commands.vault import _classify_vault_access

        db = MagicMock()
        db.list_credential_sets.return_value = ["default"]
        db.list_credentials.return_value = ["github", "openai"]
        cfg = MagicMock()
        cfg.open_credential_db.return_value = db
        reason, providers, db_error = _classify_vault_access(cfg, _recovery(source="config"))
        assert reason is None and db_error is None
        assert providers == ["github", "openai"]
        db.close.assert_called_once()

    def test_mid_read_failure_is_db_error(self) -> None:
        """A DB that opens but fails mid-read is a DB fault, not a lock; close still runs."""
        from terok_sandbox.commands.vault import _classify_vault_access

        db = MagicMock()
        db.list_credential_sets.side_effect = RuntimeError("corrupt page")
        cfg = MagicMock()
        cfg.open_credential_db.return_value = db
        reason, providers, db_error = _classify_vault_access(cfg, _recovery(source="config"))
        assert reason is None and providers is None
        assert db_error is not None and "corrupt page" in db_error
        db.close.assert_called_once()


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
    source: str | None = None,
    resolve_error: str | None = None,
) -> None:
    """Drive the handler with the recovery / plaintext seams pinned.

    *source* / *resolve_error* shape the stubbed ``RecoveryStatus`` —
    the lock classification reads them, so tests state the resolution
    outcome explicitly instead of inheriting a hardwired ``None``.
    """
    with (
        patch(
            "terok_sandbox.vault.store.recovery.RecoveryStatus.load",
            return_value=SimpleNamespace(
                acknowledged=acknowledged, source=source, resolve_error=resolve_error
            ),
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
        assert "Vault: LOCKED — no passphrase in any tier" in out
        assert "terok-sandbox vault unlock" in out
        assert "Credentials: vault locked" in out

    def test_locked_header_names_wrong_passphrase(self, capsys: pytest.CaptureFixture[str]) -> None:
        """A rejected key reads differently from a missing one — the remedy differs."""
        from terok_sandbox.vault.store.encryption import WrongPassphraseError

        cfg = _status_cfg(db_error=WrongPassphraseError("could not decrypt"))
        _run_status(cfg, source="session-file")
        out = capsys.readouterr().out
        assert "LOCKED — the passphrase via session-file does not open the DB" in out

    def test_locked_header_names_broken_tier(self, capsys: pytest.CaptureFixture[str]) -> None:
        """A fail-closed tier (broken seal) is surfaced verbatim, not as a plain lock."""
        cfg = _status_cfg()
        _run_status(cfg, resolve_error="sealed credential present but could not be unsealed")
        out = capsys.readouterr().out
        assert "LOCKED — a configured tier is unreadable" in out
        assert "could not be unsealed" in out

    def test_db_error_header_renders_error_not_locked(
        self, capsys: pytest.CaptureFixture[str]
    ) -> None:
        """A non-passphrase DB failure renders as ERROR with the message, not LOCKED."""
        cfg = _status_cfg(db_error=RuntimeError("schema drift"))
        _run_status(cfg, source="config")
        out = capsys.readouterr().out
        assert "Vault: ERROR — schema drift" in out
        assert "LOCKED" not in out
        assert "Credentials: DB unreadable — see the error above" in out

    def test_default_cfg_branch(self, capsys: pytest.CaptureFixture[str]) -> None:
        """``cfg=None`` constructs a default ``SandboxConfig`` rather than crashing."""
        cfg = _status_cfg(db_error=RuntimeError("locked"))
        with (
            patch("terok_sandbox.commands.vault.SandboxConfig", return_value=cfg) as ctor,
            patch(
                "terok_sandbox.vault.store.recovery.RecoveryStatus.load",
                return_value=_recovery(),
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
        _run_status(cfg, source="systemd-creds")
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
        _run_status(cfg, source="session-file")
        out = capsys.readouterr().out
        assert "session-file        active" in out
        assert "systemd-creds       shadowed" in out
        assert "shadowing a durable tier (systemd-creds)" in out

    def test_durable_active_tier_does_not_report_shadow(
        self, tmp_path: Path, capsys: pytest.CaptureFixture[str]
    ) -> None:
        """A durable active tier outranking a lower durable tier is not 'shadowing'."""
        sealed = tmp_path / "sealed.cred"
        sealed.write_text("blob")
        # systemd-creds (durable) active, config (durable) present below it.
        cfg = _status_cfg(sealed=sealed, config_passphrase="from-config")
        _run_status(cfg, source="systemd-creds")
        out = capsys.readouterr().out
        assert "passphrase via systemd-creds" in out
        assert "shadowed" not in out
        assert "shadowing a durable tier" not in out

    def test_unacknowledged_recovery_warns(self, capsys: pytest.CaptureFixture[str]) -> None:
        cfg = _status_cfg(db_error=RuntimeError("locked"))
        _run_status(cfg, acknowledged=False)
        assert "Recovery key: NOT acknowledged" in capsys.readouterr().out

    def test_plaintext_passphrase_warning(
        self, tmp_path: Path, capsys: pytest.CaptureFixture[str]
    ) -> None:
        cfg = _status_cfg(config_passphrase="secret")
        _run_status(cfg, plaintext=Path("/etc/terok/config.yml"), source="config")
        out = capsys.readouterr().out
        assert "plaintext at /etc/terok/config.yml" in out

    def test_credentials_listed_when_open(self, capsys: pytest.CaptureFixture[str]) -> None:
        db = MagicMock()
        db.list_credential_sets.return_value = ["default"]
        db.list_credentials.return_value = ["github", "openai"]
        cfg = _status_cfg(config_passphrase="secret", db=db)
        _run_status(cfg, source="config")
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
        _run_status(cfg, acknowledged=True, as_json=True, source="session-file")
        data = json.loads(capsys.readouterr().out)
        # The open failed for a non-passphrase reason — that's a DB error,
        # not a lock; the chain still reports what's on hand.
        assert data["locked"] is False
        assert data["lock_reason"] is None
        assert data["db_error"] == "x"
        assert data["passphrase_source"] == "session-file"
        assert data["shadowed_tiers"] == ["systemd-creds"]
        assert data["recovery_acknowledged"] is True
        assert data["credentials"] is None  # DB wouldn't open
        assert [c["source"] for c in data["chain"]][0] == "session-file"

    def test_json_lock_reasons(self, capsys: pytest.CaptureFixture[str]) -> None:
        """The three lock states are distinguishable in machine output."""
        from terok_sandbox.vault.store.encryption import WrongPassphraseError

        # (a) no passphrase anywhere
        _run_status(_status_cfg(), as_json=True)
        data = json.loads(capsys.readouterr().out)
        assert data["locked"] is True
        assert data["lock_reason"] == "no passphrase in any tier"

        # (b) resolved value rejected by the DB
        cfg = _status_cfg(db_error=WrongPassphraseError("could not decrypt"))
        _run_status(cfg, as_json=True, source="keyring")
        data = json.loads(capsys.readouterr().out)
        assert data["locked"] is True
        assert "via keyring does not open the DB" in data["lock_reason"]

        # (c) a configured tier failed closed at resolve time
        _run_status(_status_cfg(), as_json=True, resolve_error="could not be unsealed")
        data = json.loads(capsys.readouterr().out)
        assert data["locked"] is True
        assert "unreadable" in data["lock_reason"]
        assert "could not be unsealed" in data["lock_reason"]
