# SPDX-FileCopyrightText: 2026 Jiri Vyskocil
# SPDX-License-Identifier: Apache-2.0

"""Tests for the ``vault status`` CLI verb and its chain probe.

``vault status`` is a read-only diagnostic.  It walks the passphrase
resolution chain *without short-circuiting* (so a session file
shadowing a durable systemd-creds / keyring tier is visible), reports
the lock state, re-states the shared warning catalog (recovery-key and
session-shadow warnings), and lists stored credential providers on a
best-effort DB open.  The probe ([`probe_passphrase_chain`][terok_sandbox.vault.store.encryption.probe_passphrase_chain])
is pure and exercised directly; the handler is driven through a mock
``SandboxConfig`` with the recovery / shadow seams patched.  The
snapshot the handler renders ([`VaultStatus`][terok_sandbox.vault.store.status.VaultStatus])
has its own tests in ``test_vault_state_classifier.py``.
"""

from __future__ import annotations

import json
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from terok_sandbox.commands.vault import _handle_vault_status
from terok_sandbox.vault.store import encryption
from terok_sandbox.vault.store.encryption import probe_passphrase_chain
from terok_sandbox.vault.store.recovery import RecoveryStatus
from terok_sandbox.vault.store.status import SessionShadow, _classify_db_access
from terok_sandbox.vault.store.tiers import PassphraseTier
from tests.constants import MOCK_BASE

# Captured at import time — before conftest's autouse ``_isolate_credential_keyring``
# stubs ``load_passphrase_from_file`` to ``None`` — so file-tier tests can restore
# the real reader.  Same idiom as ``test_credential_encryption.py``.
from terok_sandbox.vault.store.encryption import (  # noqa: E402  isort: skip
    load_passphrase_from_file as _real_load_file,
)

MOCK_DB_PATH = MOCK_BASE / "vault" / "credentials.db"


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

    def test_systemd_creds_unconfigured_says_not_configured(self) -> None:
        """No path wired at all reads like the other absent tiers, not a blank."""
        chain = probe_passphrase_chain()
        assert chain[1].detail == "not configured"

    def test_systemd_creds_absent_file_says_not_sealed(self, tmp_path: Path) -> None:
        """A configured path with nothing sealed must not masquerade as a live tier."""
        cred = tmp_path / "vault.passphrase.cred"  # never created
        with patch.object(encryption._systemd_creds, "unavailable_reason", return_value=None):
            chain = probe_passphrase_chain(systemd_creds_file=cred)
        assert chain[1].present is False
        assert "not sealed" in chain[1].detail
        assert str(cred) in chain[1].detail

    def test_systemd_creds_unusable_reason_surfaced(self, tmp_path: Path) -> None:
        """When the tier can't run here (e.g. systemd 255), status says why."""
        cred = tmp_path / "vault.passphrase.cred"
        reason = "needs systemd ≥ 257 for non-root --user mode (host has 255)"
        with patch.object(encryption._systemd_creds, "unavailable_reason", return_value=reason):
            chain = probe_passphrase_chain(systemd_creds_file=cred)
        assert "unusable here" in chain[1].detail
        assert "host has 255" in chain[1].detail

    def test_systemd_creds_sealed_and_usable_shows_bare_path(self, tmp_path: Path) -> None:
        """Sealed + tier available → detail is just the path, no noise appended."""
        cred = tmp_path / "vault.passphrase.cred"
        cred.write_text("sealed-blob")
        with patch.object(encryption._systemd_creds, "unavailable_reason", return_value=None):
            chain = probe_passphrase_chain(systemd_creds_file=cred)
        assert chain[1].present is True
        assert chain[1].detail == str(cred)

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


def _recovery(
    source: PassphraseTier | str | None = None,
    resolve_error: str | None = None,
    *,
    acknowledged: bool = False,
) -> RecoveryStatus:
    """A real ``RecoveryStatus`` with just the fields the classifier reads pinned."""
    return RecoveryStatus(
        acknowledged=acknowledged,
        source=PassphraseTier(source) if source is not None else None,
        resolve_error=resolve_error,
    )


class TestClassifyDbAccess:
    """``_classify_db_access`` separates the three operator problems 'locked' hides."""

    def test_broken_tier_reports_resolve_error(self) -> None:
        """A fail-closed resolver (broken seal / dead helper) is named, not just 'locked'."""
        cfg = MagicMock()
        access = _classify_db_access(
            cfg,
            _recovery(resolve_error="sealed credential present but could not be unsealed"),
            db_exists=True,
        )
        assert access.lock_reason is not None and "unreadable" in access.lock_reason
        assert "could not be unsealed" in access.lock_reason
        assert access.providers is None and access.db_error is None
        cfg.open_credential_db.assert_not_called()  # nothing to try — resolution already failed

    def test_no_passphrase_anywhere(self) -> None:
        access = _classify_db_access(MagicMock(), _recovery(), db_exists=True)
        assert access.lock_reason == "no passphrase in any tier"
        assert access.providers is None and access.db_error is None

    def test_missing_db_with_ready_tier_never_opens(self) -> None:
        """A fresh install with a resolving tier is 'unlocked' *without* touching SQLite.

        Opening would *create* the DB as a side effect — a status read
        must never be the write that defines the vault's encryption key.
        """
        cfg = MagicMock()
        access = _classify_db_access(cfg, _recovery(source="keyring"), db_exists=False)
        assert access.lock_reason is None and access.db_error is None
        assert access.providers == ()
        assert access.ssh_keys == 0 and dict(access.credential_types or {}) == {}
        cfg.open_credential_db.assert_not_called()

    def test_wrong_passphrase_names_the_tier(self) -> None:
        """A resolved value the DB rejects points at the tier carrying the bad key."""
        from terok_sandbox.vault.store.encryption import WrongPassphraseError

        cfg = MagicMock()
        cfg.open_credential_db.side_effect = WrongPassphraseError("could not decrypt")
        access = _classify_db_access(cfg, _recovery(source="keyring"), db_exists=True)
        assert access.lock_reason is not None
        assert "via keyring does not open the DB" in access.lock_reason
        assert access.providers is None and access.db_error is None

    def test_open_no_passphrase_race_is_plain_lock(self) -> None:
        """A tier that vanishes between the resolve and the open is a plain lock."""
        from terok_sandbox.vault.store.encryption import NoPassphraseError

        cfg = MagicMock()
        cfg.open_credential_db.side_effect = NoPassphraseError("tier gone")
        access = _classify_db_access(cfg, _recovery(source="keyring"), db_exists=True)
        assert access.lock_reason == "no passphrase in any tier"
        assert access.providers is None and access.db_error is None

    def test_system_exit_propagates(self) -> None:
        """An explicit exit from a lower layer must not be stringified into status."""
        cfg = MagicMock()
        cfg.open_credential_db.side_effect = SystemExit(3)
        with pytest.raises(SystemExit):
            _classify_db_access(cfg, _recovery(source="keyring"), db_exists=True)

    def test_open_ok_lists_providers(self) -> None:
        db = MagicMock()
        db.list_credential_sets.return_value = ["default"]
        db.list_credentials.return_value = ["github", "openai"]
        db.load_credential.side_effect = lambda _cs, provider: {"type": f"{provider}-type"}
        db.count_ssh_keys.return_value = 3
        cfg = MagicMock()
        cfg.open_credential_db.return_value = db
        access = _classify_db_access(cfg, _recovery(source="keyring"), db_exists=True)
        assert access.lock_reason is None and access.db_error is None
        assert access.providers == ("github", "openai")
        assert dict(access.credential_types or {}) == {
            "github": "github-type",
            "openai": "openai-type",
        }
        assert access.ssh_keys == 3
        db.close.assert_called_once()

    def test_mid_read_failure_is_db_error(self) -> None:
        """A DB that opens but fails mid-read is a DB fault, not a lock; close still runs."""
        db = MagicMock()
        db.list_credential_sets.side_effect = RuntimeError("corrupt page")
        cfg = MagicMock()
        cfg.open_credential_db.return_value = db
        access = _classify_db_access(cfg, _recovery(source="keyring"), db_exists=True)
        assert access.lock_reason is None and access.providers is None
        assert access.db_error is not None and "corrupt page" in access.db_error
        db.close.assert_called_once()


def _status_cfg(
    *,
    session: Path | None = None,
    sealed: Path | None = None,
    use_keyring: bool = False,
    passphrase_command: str | None = None,
    db: MagicMock | None = None,
    db_error: Exception | None = None,
    db_path: Path | None = None,
    marker: Path | None = None,
) -> MagicMock:
    """A mock ``SandboxConfig`` exposing exactly the knobs ``status`` reads.

    ``db_path`` defaults to a never-existing mock path — pass a real
    (created) file to exercise the DB-open branches; the classifier
    refuses to open a DB that doesn't exist.
    """
    cfg = MagicMock()
    cfg.vault_passphrase_file = session or MOCK_BASE / "absent" / "session"
    cfg.vault_systemd_creds_file = sealed or MOCK_BASE / "absent" / "sealed"
    cfg.credentials_use_keyring = use_keyring
    cfg.credentials_passphrase_command = passphrase_command
    cfg.db_path = db_path or MOCK_DB_PATH
    cfg.vault_recovery_marker_file = marker or MOCK_BASE / "absent" / "marker"
    if db_error is not None:
        cfg.open_credential_db.side_effect = db_error
    else:
        cfg.open_credential_db.return_value = db or MagicMock(
            list_credential_sets=MagicMock(return_value=[]),
        )
    return cfg


def _existing_db(tmp_path: Path) -> Path:
    """An on-disk stand-in for an already-provisioned credentials DB."""
    db_file = tmp_path / "credentials.db"
    db_file.write_bytes(b"stand-in")
    return db_file


def _run_status(
    cfg: MagicMock,
    *,
    acknowledged: bool = False,
    as_json: bool = False,
    source: str | None = None,
    resolve_error: str | None = None,
    shadow: SessionShadow | None = None,
) -> None:
    """Drive the handler with the recovery / session-shadow seams pinned.

    *source* / *resolve_error* shape the stubbed ``RecoveryStatus`` —
    the lock classification reads them, so tests state the resolution
    outcome explicitly instead of inheriting a hardwired ``None``.
    *shadow* is what ``session_shadow_state`` reports; the comparison
    logic has its own tests, so no real unseal is paid here.
    """
    with (
        patch(
            "terok_sandbox.vault.store.recovery.RecoveryStatus.load",
            return_value=_recovery(source, resolve_error, acknowledged=acknowledged),
        ),
        patch("terok_sandbox.vault.store.status.session_shadow_state", return_value=shadow),
    ):
        _handle_vault_status(cfg=cfg, as_json=as_json)


class TestHandleVaultStatusText:
    """Human-readable rendering of the lock state and chain."""

    def test_locked_when_no_tier_present(
        self, tmp_path: Path, capsys: pytest.CaptureFixture[str]
    ) -> None:
        cfg = _status_cfg(db_path=_existing_db(tmp_path))
        _run_status(cfg)
        out = capsys.readouterr().out
        assert "Vault: LOCKED — no passphrase in any tier" in out
        assert "terok-sandbox vault unlock" in out
        assert "Credentials: vault locked" in out

    def test_unprovisioned_fresh_install(self, capsys: pytest.CaptureFixture[str]) -> None:
        """No DB and no tier → provisioning guidance, not an unlock prompt."""
        cfg = _status_cfg()  # default db_path never exists
        _run_status(cfg)
        out = capsys.readouterr().out
        assert "Vault: UNPROVISIONED — no credentials DB and no stored passphrase yet" in out
        assert "(created encrypted on first use)" in out
        assert "run setup (or the TUI) to provision a vault passphrase" in out
        assert "terok-sandbox vault unlock" not in out

    def test_locked_header_names_wrong_passphrase(
        self, tmp_path: Path, capsys: pytest.CaptureFixture[str]
    ) -> None:
        """A rejected key reads differently from a missing one — the remedy differs."""
        from terok_sandbox.vault.store.encryption import WrongPassphraseError

        cfg = _status_cfg(
            db_error=WrongPassphraseError("could not decrypt"), db_path=_existing_db(tmp_path)
        )
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
        self, tmp_path: Path, capsys: pytest.CaptureFixture[str]
    ) -> None:
        """A non-passphrase DB failure renders as ERROR with the message, not LOCKED."""
        cfg = _status_cfg(db_error=RuntimeError("schema drift"), db_path=_existing_db(tmp_path))
        _run_status(cfg, source="keyring")
        out = capsys.readouterr().out
        assert "Vault: ERROR — schema drift" in out
        assert "LOCKED" not in out
        assert "Credentials: DB unreadable — see the error above" in out

    def test_default_cfg_branch(self, capsys: pytest.CaptureFixture[str]) -> None:
        """``cfg=None`` constructs a default ``SandboxConfig`` rather than crashing."""
        cfg = _status_cfg()
        with (
            patch("terok_sandbox.config.SandboxConfig", return_value=cfg) as ctor,
            patch(
                "terok_sandbox.vault.store.recovery.RecoveryStatus.load",
                return_value=_recovery(),
            ),
        ):
            _handle_vault_status()  # cfg omitted → default-construction branch
        ctor.assert_called_once_with()
        assert "Vault: UNPROVISIONED" in capsys.readouterr().out

    def test_unlocked_names_active_tier(
        self, tmp_path: Path, capsys: pytest.CaptureFixture[str]
    ) -> None:
        sealed = tmp_path / "sealed.cred"
        sealed.write_text("blob")
        cfg = _status_cfg(sealed=sealed, db_path=_existing_db(tmp_path))
        _run_status(cfg, source="systemd-creds")
        out = capsys.readouterr().out
        assert "passphrase via systemd-creds" in out
        assert "systemd-creds       active" in out

    def test_session_file_shadows_durable_tier(
        self, tmp_path: Path, capsys: pytest.CaptureFixture[str], real_file_tier: None
    ) -> None:
        """The chain table marks the shadow; the warning names same-vs-different."""
        session = tmp_path / "session"
        session.write_text("pw\n")
        sealed = tmp_path / "sealed.cred"
        sealed.write_text("blob")
        cfg = _status_cfg(session=session, sealed=sealed)
        _run_status(
            cfg,
            source="session-file",
            shadow=SessionShadow(PassphraseTier.SYSTEMD_CREDS, redundant=False),
        )
        out = capsys.readouterr().out
        assert "session-file        active" in out
        assert "systemd-creds       shadowed" in out
        assert "shadows the durable systemd-creds tier with a DIFFERENT passphrase" in out

    def test_redundant_shadow_renders_as_note(
        self, tmp_path: Path, capsys: pytest.CaptureFixture[str], real_file_tier: None
    ) -> None:
        session = tmp_path / "session"
        session.write_text("pw\n")
        sealed = tmp_path / "sealed.cred"
        sealed.write_text("blob")
        cfg = _status_cfg(session=session, sealed=sealed)
        _run_status(
            cfg,
            source="session-file",
            shadow=SessionShadow(PassphraseTier.SYSTEMD_CREDS, redundant=True),
        )
        out = capsys.readouterr().out
        assert "duplicates the durable systemd-creds tier (same passphrase)" in out
        assert "redundant residue" in out
        assert "note:" in out

    def test_unverifiable_shadow_renders_as_warning(
        self, tmp_path: Path, capsys: pytest.CaptureFixture[str], real_file_tier: None
    ) -> None:
        session = tmp_path / "session"
        session.write_text("pw\n")
        sealed = tmp_path / "sealed.cred"
        sealed.write_text("blob")
        cfg = _status_cfg(session=session, sealed=sealed)
        _run_status(
            cfg,
            source="session-file",
            shadow=SessionShadow(PassphraseTier.SYSTEMD_CREDS, redundant=None),
        )
        out = capsys.readouterr().out
        assert "shadows systemd-creds" in out
        assert "could not be read to compare" in out
        assert "warning:" in out

    def test_durable_active_tier_does_not_report_shadow(
        self, tmp_path: Path, capsys: pytest.CaptureFixture[str]
    ) -> None:
        """A durable active tier outranking a lower durable tier is not 'shadowing'."""
        sealed = tmp_path / "sealed.cred"
        sealed.write_text("blob")
        # systemd-creds (durable) active, passphrase-command (durable) present below it.
        cfg = _status_cfg(
            sealed=sealed, passphrase_command="pass show vault", db_path=_existing_db(tmp_path)
        )
        _run_status(cfg, source="systemd-creds")
        out = capsys.readouterr().out
        assert "passphrase via systemd-creds" in out
        assert "shadowed" not in out
        assert "shadowing a durable tier" not in out

    def test_unacknowledged_recovery_warns(self, capsys: pytest.CaptureFixture[str]) -> None:
        cfg = _status_cfg()
        _run_status(cfg, acknowledged=False)
        assert "Recovery key: NOT acknowledged" in capsys.readouterr().out

    def test_unconfirmed_recovery_warning_for_durable_tier(
        self, tmp_path: Path, capsys: pytest.CaptureFixture[str]
    ) -> None:
        """A resolving durable tier without an off-host copy gets the catalog warning."""
        cfg = _status_cfg(db_path=_existing_db(tmp_path))
        _run_status(cfg, source="keyring", acknowledged=False)
        out = capsys.readouterr().out
        assert "warning: the vault passphrase is not confirmed saved off-host" in out

    def test_urgent_recovery_warning_for_session_only(
        self, tmp_path: Path, capsys: pytest.CaptureFixture[str]
    ) -> None:
        """Session-file-only + unacknowledged escalates to the reboot-loss error."""
        cfg = _status_cfg(db_path=_existing_db(tmp_path))
        _run_status(cfg, source="session-file", acknowledged=False)
        out = capsys.readouterr().out
        assert "error: the only copy of the vault passphrase is the session file" in out
        assert "not confirmed saved off-host" not in out  # the urgent variant replaces it

    def test_credentials_listed_when_open(
        self, tmp_path: Path, capsys: pytest.CaptureFixture[str]
    ) -> None:
        db = MagicMock()
        db.list_credential_sets.return_value = ["default"]
        db.list_credentials.return_value = ["github", "openai"]
        db.load_credential.side_effect = lambda _cs, provider: {"type": f"{provider}-type"}
        db.count_ssh_keys.return_value = 3
        cfg = _status_cfg(db=db, db_path=_existing_db(tmp_path))
        _run_status(cfg, source="keyring", acknowledged=True)
        out = capsys.readouterr().out
        assert "Credentials: 2 stored (github (github-type), openai (openai-type))" in out
        assert "SSH keys:    3 stored" in out


class TestHandleVaultStatusJson:
    """``--json`` carries the same facts in a machine-readable shape."""

    def test_json_shape(
        self, tmp_path: Path, capsys: pytest.CaptureFixture[str], real_file_tier: None
    ) -> None:
        session = tmp_path / "session"
        session.write_text("pw\n")
        sealed = tmp_path / "sealed.cred"
        sealed.write_text("blob")
        cfg = _status_cfg(
            session=session,
            sealed=sealed,
            db_error=RuntimeError("x"),
            db_path=_existing_db(tmp_path),
        )
        _run_status(cfg, acknowledged=True, as_json=True, source="session-file")
        data = json.loads(capsys.readouterr().out)
        # The open failed for a non-passphrase reason — that's a DB error,
        # not a lock; the chain still reports what's on hand.
        assert data["state"] == "error"
        assert data["locked"] is True  # anything non-unlocked counts as locked
        assert data["lock_reason"] is None
        assert data["db_error"] == "x"
        assert data["passphrase_source"] == "session-file"
        assert data["shadowed_tiers"] == ["systemd-creds"]
        assert data["recovery_acknowledged"] is True
        assert data["credentials"] is None  # DB wouldn't open
        assert [c["source"] for c in data["chain"]][0] == "session-file"
        assert len(data["chain"]) == 4
        assert isinstance(data["warnings"], list)
        assert "plaintext_passphrase_path" not in data

    def test_json_unprovisioned(self, capsys: pytest.CaptureFixture[str]) -> None:
        """A fresh install is a distinct machine-readable state, not a plain lock."""
        _run_status(_status_cfg(), as_json=True)
        data = json.loads(capsys.readouterr().out)
        assert data["state"] == "unprovisioned"
        assert data["locked"] is True
        assert data["credentials"] is None
        assert data["warnings"] == []

    def test_json_lock_reasons(self, tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
        """The three lock states are distinguishable in machine output."""
        from terok_sandbox.vault.store.encryption import WrongPassphraseError

        # (a) no passphrase anywhere (with a DB on disk — otherwise unprovisioned)
        _run_status(_status_cfg(db_path=_existing_db(tmp_path)), as_json=True)
        data = json.loads(capsys.readouterr().out)
        assert data["state"] == "locked"
        assert data["locked"] is True
        assert data["lock_reason"] == "no passphrase in any tier"

        # (b) resolved value rejected by the DB
        cfg = _status_cfg(
            db_error=WrongPassphraseError("could not decrypt"), db_path=_existing_db(tmp_path)
        )
        _run_status(cfg, as_json=True, source="keyring")
        data = json.loads(capsys.readouterr().out)
        assert data["state"] == "locked"
        assert data["locked"] is True
        assert "via keyring does not open the DB" in data["lock_reason"]

        # (c) a configured tier failed closed at resolve time
        _run_status(_status_cfg(), as_json=True, resolve_error="could not be unsealed")
        data = json.loads(capsys.readouterr().out)
        assert data["state"] == "locked"
        assert data["locked"] is True
        assert "unreadable" in data["lock_reason"]
        assert "could not be unsealed" in data["lock_reason"]

    def test_json_session_shadow(
        self, tmp_path: Path, capsys: pytest.CaptureFixture[str], real_file_tier: None
    ) -> None:
        """``session_shadow`` carries the durable source + redundancy verdict."""
        session = tmp_path / "session"
        session.write_text("pw\n")
        sealed = tmp_path / "sealed.cred"
        sealed.write_text("blob")
        cfg = _status_cfg(session=session, sealed=sealed)
        _run_status(
            cfg,
            as_json=True,
            source="session-file",
            shadow=SessionShadow(PassphraseTier.SYSTEMD_CREDS, redundant=True),
        )
        data = json.loads(capsys.readouterr().out)
        assert data["session_shadow"] == {"durable_source": "systemd-creds", "redundant": True}

    def test_json_session_shadow_absent(self, capsys: pytest.CaptureFixture[str]) -> None:
        """No shadow → the field is explicitly null, never an unseal on the common path."""
        _run_status(_status_cfg(), as_json=True, source="keyring")
        assert json.loads(capsys.readouterr().out)["session_shadow"] is None


class TestSessionShadowState:
    """``session_shadow_state`` / ``clear_redundant_session_file`` over real tiers.

    Uses the keyring tier as the durable one — the conftest keyring stub
    is repinned per test, so the same/different-key comparison is
    exercised end to end without a TPM.  ``status`` calls the file
    reader through the ``encryption`` namespace, so the conftest's
    blanket file-tier stub applies — the autouse fixture below restores
    the real reader for this class.
    """

    @pytest.fixture(autouse=True)
    def _file_tier(self, real_file_tier: None) -> None:
        """Every test in this class reads a real session file."""

    @pytest.fixture
    def keyring_value(self, monkeypatch: pytest.MonkeyPatch) -> str:
        """Pin the (conftest-stubbed) keyring tier to a known durable passphrase."""
        monkeypatch.setattr(encryption, "load_passphrase_from_keyring", lambda: "K")
        return "K"

    def test_no_session_file_is_no_shadow(self, keyring_value: str) -> None:
        from terok_sandbox.vault.store.status import session_shadow_state

        cfg = _status_cfg(use_keyring=True)  # durable present, but no session file
        assert session_shadow_state(cfg) is None

    def test_no_durable_tier_is_no_shadow(self, tmp_path: Path) -> None:
        from terok_sandbox.vault.store.status import session_shadow_state

        session = tmp_path / "session"
        session.write_text("only-tier\n")
        cfg = _status_cfg(session=session)  # session present, nothing durable under it
        assert session_shadow_state(cfg) is None

    def test_same_key_is_redundant(self, tmp_path: Path, keyring_value: str) -> None:
        from terok_sandbox.vault.store.status import session_shadow_state

        session = tmp_path / "session"
        session.write_text(f"{keyring_value}\n")
        cfg = _status_cfg(session=session, use_keyring=True)
        shadow = session_shadow_state(cfg)
        assert shadow is not None
        assert shadow.durable_source is PassphraseTier.KEYRING
        assert shadow.redundant is True

    def test_different_key_is_not_redundant(self, tmp_path: Path, keyring_value: str) -> None:
        from terok_sandbox.vault.store.status import session_shadow_state

        session = tmp_path / "session"
        session.write_text("session-key\n")
        cfg = _status_cfg(session=session, use_keyring=True)
        shadow = session_shadow_state(cfg)
        assert shadow is not None and shadow.redundant is False

    def test_clear_removes_only_redundant(self, tmp_path: Path, keyring_value: str) -> None:
        from terok_sandbox.vault.store.status import clear_redundant_session_file

        session = tmp_path / "session"
        session.write_text(f"{keyring_value}\n")
        cfg = _status_cfg(session=session, use_keyring=True)
        assert clear_redundant_session_file(cfg) is PassphraseTier.KEYRING
        assert not session.exists()

    def test_clear_keeps_a_different_key_override(self, tmp_path: Path, keyring_value: str) -> None:
        from terok_sandbox.vault.store.status import clear_redundant_session_file

        session = tmp_path / "session"
        session.write_text("override\n")
        cfg = _status_cfg(session=session, use_keyring=True)
        assert clear_redundant_session_file(cfg) is None
        assert session.exists()  # a deliberate override is never auto-removed

    def test_unreadable_durable_tier_is_an_unverifiable_shadow(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """A present-but-unsealable durable tier can't be compared → ``redundant=None``.

        The session file may be doing real work in that state, so the
        shadow is reported without a verdict and never auto-removed.
        """
        from terok_sandbox.vault.store import systemd_creds
        from terok_sandbox.vault.store.status import session_shadow_state

        session = tmp_path / "session"
        session.write_text("session-key\n")
        sealed = tmp_path / "sealed.cred"
        sealed.write_bytes(b"sealed-on-another-boot")
        monkeypatch.setattr(systemd_creds, "unseal", lambda _path: None)
        cfg = _status_cfg(session=session, sealed=sealed)

        shadow = session_shadow_state(cfg)

        assert shadow is not None
        assert shadow.durable_source is PassphraseTier.SYSTEMD_CREDS
        assert shadow.redundant is None


class TestResolveCfg:
    """The lazy default-config seam every status entry point shares."""

    def test_none_builds_a_default_config(self) -> None:
        """``cfg=None`` constructs a real default config (isolated HOME in tests)."""
        from terok_sandbox import SandboxConfig
        from terok_sandbox.vault.store.status import _resolve_cfg

        assert isinstance(_resolve_cfg(None), SandboxConfig)

    def test_explicit_cfg_passes_through_unchanged(self) -> None:
        """A caller-supplied config is returned as-is, never rebuilt."""
        from terok_sandbox.vault.store.status import _resolve_cfg

        cfg = _status_cfg()
        assert _resolve_cfg(cfg) is cfg
