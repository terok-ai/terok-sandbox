# SPDX-FileCopyrightText: 2026 Jiri Vyskocil
# SPDX-License-Identifier: Apache-2.0

"""Tests for the ``vault status`` CLI verb and its chain probe.

``vault status`` is a read-only diagnostic.  It walks the passphrase
resolution chain *without short-circuiting* (so every tier that holds
material is visible), reports the lock state, re-states the shared
warning catalog (recovery-key warnings), and lists stored credential
providers on a best-effort DB open.  The probe ([`probe_passphrase_chain`][terok_sandbox.vault.store.encryption.probe_passphrase_chain])
is pure and exercised directly; the handler is driven through a mock
``SandboxConfig`` with the recovery seam patched.  The snapshot the
handler renders ([`VaultStatus`][terok_sandbox.vault.store.status.VaultStatus])
has its own tests in ``test_vault_state_classifier.py``.
"""

from __future__ import annotations

import json
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

import terok_sandbox.vault.store.kernel_keyring as _kk
from terok_sandbox.commands.vault import _handle_vault_status
from terok_sandbox.vault.store import encryption
from terok_sandbox.vault.store.encryption import probe_passphrase_chain
from terok_sandbox.vault.store.recovery import RecoveryStatus
from terok_sandbox.vault.store.status import _classify_db_access
from terok_sandbox.vault.store.tiers import PassphraseTier
from tests.constants import MOCK_BASE

MOCK_DB_PATH = MOCK_BASE / "vault" / "credentials.db"


class TestProbePassphraseChain:
    """``probe_passphrase_chain`` reports per-tier presence in resolution order."""

    def test_empty_chain_all_absent(self) -> None:
        chain = probe_passphrase_chain()
        assert [t.source for t in chain] == [
            "systemd-creds",
            "keyring",
            "kernel-keyring",
            "passphrase-command",
        ]
        assert all(not t.present for t in chain)

    def test_kernel_keyring_present_when_cached(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setattr(_kk, "is_cached", lambda: True)
        chain = probe_passphrase_chain()
        assert chain[2].source == "kernel-keyring"
        assert chain[2].present is True
        assert "cached in the user keyring" in chain[2].detail

    def test_kernel_keyring_absent_when_empty(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setattr(_kk, "is_cached", lambda: False)
        chain = probe_passphrase_chain()
        assert chain[2].present is False
        assert "no passphrase cached" in chain[2].detail

    def test_chain_probe_never_reads_the_cached_passphrase(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Status reports that a tier holds material, never its value."""

        def _explode() -> str:
            raise AssertionError("probe_passphrase_chain must not read the passphrase")

        monkeypatch.setattr(_kk, "is_cached", lambda: True)
        monkeypatch.setattr(_kk, "load", _explode)
        assert probe_passphrase_chain()[2].present is True

    def test_systemd_creds_present_when_sealed_file_exists(self, tmp_path: Path) -> None:
        sealed = tmp_path / "vault.passphrase.cred"
        sealed.write_text("sealed-blob")
        chain = probe_passphrase_chain(systemd_creds_file=sealed)
        assert chain[0].source == "systemd-creds"
        assert chain[0].present is True

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
        assert chain[0].detail == "not configured"

    def test_systemd_creds_absent_file_says_not_sealed(self, tmp_path: Path) -> None:
        """A configured path with nothing sealed must not masquerade as a live tier."""
        cred = tmp_path / "vault.passphrase.cred"  # never created
        with patch.object(encryption._systemd_creds, "unavailable_reason", return_value=None):
            chain = probe_passphrase_chain(systemd_creds_file=cred)
        assert chain[0].present is False
        assert "not sealed" in chain[0].detail
        assert str(cred) in chain[0].detail

    def test_systemd_creds_unusable_reason_surfaced(self, tmp_path: Path) -> None:
        """When the tier can't run here (e.g. systemd 255), status says why."""
        cred = tmp_path / "vault.passphrase.cred"
        reason = "needs systemd ≥ 257 for non-root --user mode (host has 255)"
        with patch.object(encryption._systemd_creds, "unavailable_reason", return_value=reason):
            chain = probe_passphrase_chain(systemd_creds_file=cred)
        assert "unusable here" in chain[0].detail
        assert "host has 255" in chain[0].detail

    def test_systemd_creds_sealed_and_usable_shows_bare_path(self, tmp_path: Path) -> None:
        """Sealed + tier available → detail is just the path, no noise appended."""
        cred = tmp_path / "vault.passphrase.cred"
        cred.write_text("sealed-blob")
        with patch.object(encryption._systemd_creds, "unavailable_reason", return_value=None):
            chain = probe_passphrase_chain(systemd_creds_file=cred)
        assert chain[0].present is True
        assert chain[0].detail == str(cred)

    def test_keyring_only_probed_when_enabled(self) -> None:
        with patch.object(encryption, "load_passphrase_from_keyring", return_value="k") as load:
            on = probe_passphrase_chain(use_keyring=True)
            assert on[1].present is True
            off = probe_passphrase_chain(use_keyring=False)
            assert off[1].present is False
        # one lookup for the enabled probe, none for the disabled one
        assert load.call_count == 1

    def test_keyring_empty_string_is_absent(self) -> None:
        """An empty keyring value is the resolver's no-passphrase sentinel — treat as absent."""
        with patch.object(encryption, "load_passphrase_from_keyring", return_value=""):
            chain = probe_passphrase_chain(use_keyring=True)
        assert chain[1].present is False

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
) -> None:
    """Drive the handler with the recovery seam pinned.

    *source* / *resolve_error* shape the stubbed ``RecoveryStatus`` —
    the lock classification reads them, so tests state the resolution
    outcome explicitly instead of inheriting a hardwired ``None``.
    """
    with patch(
        "terok_sandbox.vault.store.recovery.RecoveryStatus.load",
        return_value=_recovery(source, resolve_error, acknowledged=acknowledged),
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
        _run_status(cfg, source="kernel-keyring")
        out = capsys.readouterr().out
        assert "LOCKED — the passphrase via kernel-keyring does not open the DB" in out

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

    def test_present_but_inactive_tier_marked_present(
        self, tmp_path: Path, capsys: pytest.CaptureFixture[str]
    ) -> None:
        """A durable active tier outranking a lower present tier marks the latter 'present'."""
        sealed = tmp_path / "sealed.cred"
        sealed.write_text("blob")
        # systemd-creds (durable) active, passphrase-command present below it.
        cfg = _status_cfg(
            sealed=sealed, passphrase_command="pass show vault", db_path=_existing_db(tmp_path)
        )
        _run_status(cfg, source="systemd-creds")
        out = capsys.readouterr().out
        assert "passphrase via systemd-creds" in out
        assert "systemd-creds       active" in out
        assert "passphrase-command  present" in out
        assert "shadowed" not in out

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

    def test_urgent_recovery_warning_for_volatile_only(
        self, tmp_path: Path, capsys: pytest.CaptureFixture[str]
    ) -> None:
        """Kernel-keyring-only + unacknowledged escalates to the logout-loss error."""
        cfg = _status_cfg(db_path=_existing_db(tmp_path))
        _run_status(cfg, source="kernel-keyring", acknowledged=False)
        out = capsys.readouterr().out
        assert "error: the only copy of the vault passphrase is the kernel-keyring cache" in out
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

    def test_json_shape(self, tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
        sealed = tmp_path / "sealed.cred"
        sealed.write_text("blob")
        cfg = _status_cfg(
            sealed=sealed,
            db_error=RuntimeError("x"),
            db_path=_existing_db(tmp_path),
        )
        _run_status(cfg, acknowledged=True, as_json=True, source="kernel-keyring")
        data = json.loads(capsys.readouterr().out)
        # The open failed for a non-passphrase reason — that's a DB error,
        # not a lock; the chain still reports what's on hand.
        assert data["state"] == "error"
        assert data["locked"] is True  # anything non-unlocked counts as locked
        assert data["lock_reason"] is None
        assert data["db_error"] == "x"
        assert data["passphrase_source"] == "kernel-keyring"
        assert data["recovery_acknowledged"] is True
        assert data["credentials"] is None  # DB wouldn't open
        assert [c["source"] for c in data["chain"]][0] == "systemd-creds"
        assert len(data["chain"]) == 4
        assert isinstance(data["warnings"], list)
        assert "plaintext_passphrase_path" not in data
        assert "shadowed_tiers" not in data
        assert "session_shadow" not in data

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
