# SPDX-FileCopyrightText: 2026 Jiri Vyskocil
# SPDX-License-Identifier: Apache-2.0

"""Change-passphrase coverage — registry, rekey primitive, orchestration, CLI.

Exercises the full stack bottom-up: the tier registry's derived sets,
``rekey_in_place`` against a real SQLCipher DB, the ``change_passphrase``
orchestration (verify → rekey → tier fan-out → marker drop), and the
``vault passphrase change`` handler's piped-stdin contract.
"""

from __future__ import annotations

import io
import sqlite3
import sys
from pathlib import Path

import pytest

from terok_sandbox import PassphraseTier, SandboxConfig, change_passphrase
from terok_sandbox.commands import vault as vault_cmd
from terok_sandbox.commands.credentials import plan_provisioning
from terok_sandbox.commands.vault import (
    _change_or_exit,
    _collect_current_passphrase,
    _collect_new_passphrase,
    _handle_vault_passphrase_change,
    _rewrite_tier,
)
from terok_sandbox.vault.store import encryption, systemd_creds
from terok_sandbox.vault.store.db import CredentialDB
from terok_sandbox.vault.store.encryption import (
    NoPassphraseError,
    WrongPassphraseError,
    load_passphrase_from_file as _real_load_file,
    probe_passphrase_chain,
    rekey_in_place,
)
from terok_sandbox.vault.store.recovery import acknowledge, acknowledged
from terok_sandbox.vault.store.tiers import (
    _TRAITS,
    CHOOSER_TIERS,
    DURABLE_TIERS,
    PROVISIONABLE_TIERS,
)

OLD = "old-passphrase"
NEW = "new-passphrase"


@pytest.fixture(autouse=True)
def _restore_file_tier(monkeypatch: pytest.MonkeyPatch) -> None:
    """Undo conftest's blanket file-tier stub — this module exercises the real session file."""
    monkeypatch.setattr(encryption, "load_passphrase_from_file", _real_load_file)


def _cfg(tmp_path: Path, *, use_keyring: bool = False) -> SandboxConfig:
    """Sandbox config rooted under *tmp_path*, keyring tier off unless asked."""
    return SandboxConfig(
        state_dir=tmp_path / "state",
        runtime_dir=tmp_path / "rt",
        config_dir=tmp_path / "cfg",
        vault_dir=tmp_path / "vault",
        services_mode="socket",
        credentials_use_keyring=use_keyring,
    )


def _seed_db(cfg: SandboxConfig, passphrase: str) -> None:
    """Create an encrypted credentials DB holding one credential row."""
    db = CredentialDB(cfg.db_path, passphrase=passphrase)
    db.store_credential("personal", "blablador", {"type": "api_key", "api_key": "k-123"})
    db.close()


def _write_session(cfg: SandboxConfig, value: str) -> None:
    """Land *value* on the session-file tier."""
    cfg.vault_passphrase_file.parent.mkdir(parents=True, exist_ok=True)
    cfg.vault_passphrase_file.write_text(value + "\n", encoding="utf-8")


def _opens_with(cfg: SandboxConfig, passphrase: str) -> bool:
    """Whether the DB opens (and reads) under *passphrase*."""
    try:
        CredentialDB(cfg.db_path, passphrase=passphrase).close()
    except WrongPassphraseError:
        return False
    return True


class TestTierRegistry:
    """The registry's derived subsets — every consumer keys off these."""

    def test_every_member_has_a_traits_row(self) -> None:
        """A new tier without a traits row must fail here, not at a call site."""
        for tier in PassphraseTier:
            assert tier in _TRAITS
            assert isinstance(tier.durable, bool)
            assert isinstance(tier.provisionable, bool)
            assert isinstance(tier.chooser_offered, bool)

    def test_derived_sets(self) -> None:
        """The subsets encode the design decisions the modules rely on."""
        expected_durable = {
            PassphraseTier.SYSTEMD_CREDS,
            PassphraseTier.KEYRING,
            PassphraseTier.PASSPHRASE_COMMAND,
        }
        expected_provisionable = {
            PassphraseTier.SESSION_FILE,
            PassphraseTier.SYSTEMD_CREDS,
            PassphraseTier.KEYRING,
        }
        assert expected_durable == DURABLE_TIERS
        assert expected_provisionable == PROVISIONABLE_TIERS
        assert CHOOSER_TIERS == (PassphraseTier.SESSION_FILE, PassphraseTier.KEYRING)

    def test_members_are_their_string_values(self) -> None:
        """StrEnum contract — status JSON and CLI args need plain strings."""
        assert PassphraseTier.SESSION_FILE == "session-file"
        assert f"{PassphraseTier.KEYRING}" == "keyring"

    def test_probe_order_matches_declaration_order(self, tmp_path: Path) -> None:
        """The enum's declaration order is the resolution-chain order."""
        cfg = _cfg(tmp_path)
        probed = [
            row.source
            for row in probe_passphrase_chain(
                passphrase_file=cfg.vault_passphrase_file,
                systemd_creds_file=cfg.vault_systemd_creds_file,
                use_keyring=False,
                passphrase_command=None,
            )
        ]
        storing = [tier for tier in PassphraseTier if tier is not PassphraseTier.PROMPT]
        assert probed == storing


class TestRekeyInPlace:
    """The SQLCipher ``PRAGMA rekey`` primitive."""

    def test_roundtrip_preserves_data_and_retires_old_key(self, tmp_path: Path) -> None:
        cfg = _cfg(tmp_path)
        _seed_db(cfg, OLD)

        rekey_in_place(cfg.db_path, OLD, NEW)

        db = CredentialDB(cfg.db_path, passphrase=NEW)
        assert db.load_credential("personal", "blablador")["api_key"] == "k-123"
        db.close()
        assert not _opens_with(cfg, OLD)

    def test_wrong_old_key_raises_and_changes_nothing(self, tmp_path: Path) -> None:
        cfg = _cfg(tmp_path)
        _seed_db(cfg, OLD)

        with pytest.raises(WrongPassphraseError):
            rekey_in_place(cfg.db_path, "not-the-key", NEW)
        assert _opens_with(cfg, OLD)

    def test_empty_new_key_is_rejected(self, tmp_path: Path) -> None:
        """An empty passphrase is SQLCipher's no-encryption sentinel."""
        cfg = _cfg(tmp_path)
        _seed_db(cfg, OLD)

        with pytest.raises(ValueError, match="empty passphrase"):
            rekey_in_place(cfg.db_path, OLD, "")

    def test_refuses_while_another_connection_holds_the_db(self, tmp_path: Path) -> None:
        """A live reader (a supervisor stand-in) must abort the rekey before any change."""
        from terok_sandbox.vault.store.encryption import open_sqlcipher

        cfg = _cfg(tmp_path)
        _seed_db(cfg, OLD)
        holder = open_sqlcipher(cfg.db_path, OLD)
        holder.execute("BEGIN")
        holder.execute("SELECT count(*) FROM sqlite_master")
        try:
            # Contention surfaces either as SQLite's own raise or as the
            # verified-result-row guard — both say "database is locked",
            # which is what the CLI/TUI handlers key their hint on.
            with pytest.raises(Exception, match="database is locked"):
                rekey_in_place(cfg.db_path, OLD, NEW)
        finally:
            holder.close()
        assert _opens_with(cfg, OLD)

    def test_no_old_key_sidecars_survive(self, tmp_path: Path) -> None:
        """Leftover WAL/journal frames under the old key would poison the next open."""
        cfg = _cfg(tmp_path)
        _seed_db(cfg, OLD)

        rekey_in_place(cfg.db_path, OLD, NEW)

        for suffix in ("-wal", "-shm", "-journal"):
            assert not Path(str(cfg.db_path) + suffix).exists()


class _FakeRow:
    """A one-row cursor stand-in serving a crafted ``fetchone()`` result."""

    def __init__(self, row: tuple[object, ...]) -> None:
        self._row = row

    def fetchone(self) -> tuple[object, ...]:
        """Return the scripted row."""
        return self._row


class _FakeConn:
    """A sqlcipher connection double with scriptable result rows per statement.

    ``wal_checkpoint`` and ``journal_mode`` report contention through
    their *result rows* rather than by raising, so a real second
    connection can't deterministically steer both guards — a scripted
    double can.
    """

    def __init__(
        self,
        *,
        select_error: Exception | None = None,
        checkpoint_busy: int = 0,
        journal_mode: str = "delete",
    ) -> None:
        self._select_error = select_error
        self._checkpoint_busy = checkpoint_busy
        self._journal_mode = journal_mode
        self.closed = False
        self.rekeyed = False

    def execute(self, sql: str) -> _FakeRow:
        """Dispatch on the statement the way the real connection would."""
        if sql.startswith("SELECT"):
            if self._select_error is not None:
                raise self._select_error
            return _FakeRow((0,))
        if "wal_checkpoint" in sql:
            return _FakeRow((self._checkpoint_busy, 0, 0))
        if "journal_mode=DELETE" in sql:
            return _FakeRow((self._journal_mode,))
        if sql.startswith("PRAGMA rekey"):
            self.rekeyed = True
        return _FakeRow((None,))

    def close(self) -> None:
        """Record the mandatory cleanup."""
        self.closed = True


class TestRekeyInPlaceContentionGuards:
    """The result-row contention guards ``rekey_in_place`` verifies explicitly."""

    def _rekey_with(self, monkeypatch: pytest.MonkeyPatch, conn: _FakeConn) -> None:
        """Run ``rekey_in_place`` over the scripted connection double."""
        monkeypatch.setattr(encryption, "open_sqlcipher", lambda *_a, **_kw: conn)
        rekey_in_place(Path("unused.db"), OLD, NEW)

    def test_non_key_database_error_propagates_untranslated(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """A real fault on the probe read must not be mislabelled a key mismatch."""
        import sqlcipher3

        conn = _FakeConn(select_error=sqlcipher3.DatabaseError("database is locked"))
        with pytest.raises(sqlcipher3.DatabaseError, match="database is locked") as excinfo:
            self._rekey_with(monkeypatch, conn)
        assert not isinstance(excinfo.value, WrongPassphraseError)
        assert not conn.rekeyed
        assert conn.closed

    def test_busy_wal_checkpoint_refuses_before_rekey(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """A busy checkpoint means old-key WAL frames would survive — abort."""
        conn = _FakeConn(checkpoint_busy=1)
        with pytest.raises(RuntimeError, match="database is locked.*drain the WAL"):
            self._rekey_with(monkeypatch, conn)
        assert not conn.rekeyed
        assert conn.closed

    def test_refused_journal_mode_switch_refuses_before_rekey(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """``journal_mode`` echoing the old mode means no exclusive hold — abort."""
        conn = _FakeConn(journal_mode="wal")
        with pytest.raises(RuntimeError, match="database is locked.*exclusive hold"):
            self._rekey_with(monkeypatch, conn)
        assert not conn.rekeyed
        assert conn.closed


class TestChangePassphrase:
    """The prompt-free orchestration shared by CLI and TUI."""

    def test_happy_path_over_the_session_tier(self, tmp_path: Path) -> None:
        cfg = _cfg(tmp_path)
        _seed_db(cfg, OLD)
        _write_session(cfg, OLD)
        acknowledge(cfg.vault_recovery_marker_file)

        result = change_passphrase(cfg, new=NEW)

        assert result.rekeyed and not result.generated and result.passphrase == NEW
        assert [(r.tier, r.ok) for r in result.rewrites] == [(PassphraseTier.SESSION_FILE, True)]
        assert cfg.vault_passphrase_file.read_text(encoding="utf-8").strip() == NEW
        assert _opens_with(cfg, NEW) and not _opens_with(cfg, OLD)
        # The confirmed-saved marker referred to the old passphrase.
        assert not acknowledged(cfg.vault_recovery_marker_file)
        # The rekey stamp lets health surfaces flag pre-rekey supervisors.
        assert cfg.vault_rekey_stamp_file.exists()
        # A tier holds the new value, so the crash-recovery escrow is gone.
        assert not cfg.vault_pending_passphrase_file.exists()

    def test_minted_when_new_is_omitted(self, tmp_path: Path) -> None:
        cfg = _cfg(tmp_path)
        _seed_db(cfg, OLD)
        _write_session(cfg, OLD)

        result = change_passphrase(cfg)

        assert result.generated and len(result.passphrase) > 20
        assert _opens_with(cfg, result.passphrase)

    def test_explicit_old_outranks_a_stale_tier(self, tmp_path: Path) -> None:
        """A session file left holding a stale value must not block the change."""
        cfg = _cfg(tmp_path)
        _seed_db(cfg, OLD)
        _write_session(cfg, "stale-earlier-value")

        result = change_passphrase(cfg, old=OLD, new=NEW)

        assert result.rekeyed
        assert cfg.vault_passphrase_file.read_text(encoding="utf-8").strip() == NEW

    def test_locked_vault_with_supplied_old_lands_the_session_tier(self, tmp_path: Path) -> None:
        """No tier holds material → the new value must land somewhere reachable."""
        cfg = _cfg(tmp_path)
        _seed_db(cfg, OLD)

        result = change_passphrase(cfg, old=OLD, new=NEW)

        assert [(r.tier, r.ok) for r in result.rewrites] == [(PassphraseTier.SESSION_FILE, True)]
        assert cfg.vault_passphrase_file.read_text(encoding="utf-8").strip() == NEW

    def test_tier_only_change_without_a_db(self, tmp_path: Path) -> None:
        """Pre-first-use: nothing to rekey, but the tier value still rotates."""
        cfg = _cfg(tmp_path)
        _write_session(cfg, OLD)

        result = change_passphrase(cfg, new=NEW)

        assert not result.rekeyed
        assert cfg.vault_passphrase_file.read_text(encoding="utf-8").strip() == NEW

    def test_keyring_write_failure_is_reported_not_raised(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """After the rekey a failing tier is purged + reported, never aborted on."""
        cfg = _cfg(tmp_path, use_keyring=True)
        _seed_db(cfg, OLD)
        monkeypatch.setattr(encryption, "load_passphrase_from_keyring", lambda: OLD)
        monkeypatch.setattr(encryption, "store_passphrase_in_keyring", lambda _v: False)
        monkeypatch.setattr(encryption, "forget_passphrase_in_keyring", lambda: True)

        result = change_passphrase(cfg, new=NEW)

        assert _opens_with(cfg, NEW)
        (problem,) = result.problems
        assert problem.tier is PassphraseTier.KEYRING
        assert "stale entry removed" in problem.detail

    def test_refuses_while_passphrase_command_is_configured(self, tmp_path: Path) -> None:
        """The external store's copy can't be rewritten from here — fail up front."""
        cfg = SandboxConfig(
            state_dir=tmp_path / "state",
            runtime_dir=tmp_path / "rt",
            config_dir=tmp_path / "cfg",
            vault_dir=tmp_path / "vault",
            services_mode="socket",
            credentials_use_keyring=False,
            credentials_passphrase_command="pass show terok/vault",
        )

        with pytest.raises(RuntimeError, match="external secret store"):
            change_passphrase(cfg, new=NEW)

    def test_empty_new_is_rejected(self, tmp_path: Path) -> None:
        cfg = _cfg(tmp_path)
        with pytest.raises(ValueError, match="empty passphrase"):
            change_passphrase(cfg, new="")

    def test_identical_new_is_rejected(self, tmp_path: Path) -> None:
        cfg = _cfg(tmp_path)
        _seed_db(cfg, OLD)
        _write_session(cfg, OLD)

        with pytest.raises(ValueError, match="identical"):
            change_passphrase(cfg, new=OLD)
        assert _opens_with(cfg, OLD)

    def test_locked_vault_without_old_raises(self, tmp_path: Path) -> None:
        cfg = _cfg(tmp_path)
        _seed_db(cfg, OLD)

        with pytest.raises(NoPassphraseError, match="locked"):
            change_passphrase(cfg, new=NEW)

    def test_escrow_survives_when_every_tier_rewrite_fails(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """If no tier took the new value, the pending file is its only on-host copy."""
        cfg = _cfg(tmp_path, use_keyring=True)
        _seed_db(cfg, OLD)
        monkeypatch.setattr(encryption, "load_passphrase_from_keyring", lambda: OLD)
        monkeypatch.setattr(encryption, "store_passphrase_in_keyring", lambda _v: False)
        monkeypatch.setattr(encryption, "forget_passphrase_in_keyring", lambda: False)

        result = change_passphrase(cfg, new=NEW)

        assert result.problems and not any(r.ok for r in result.rewrites)
        assert _opens_with(cfg, NEW)
        escrowed = cfg.vault_pending_passphrase_file.read_text(encoding="utf-8").strip()
        assert escrowed == NEW

    def test_unprovisioned_vault_raises(self, tmp_path: Path) -> None:
        cfg = _cfg(tmp_path)

        with pytest.raises(NoPassphraseError, match="provision"):
            change_passphrase(cfg, new=NEW)

    def test_wrong_old_raises_and_changes_nothing(self, tmp_path: Path) -> None:
        cfg = _cfg(tmp_path)
        _seed_db(cfg, OLD)
        _write_session(cfg, OLD)
        acknowledge(cfg.vault_recovery_marker_file)

        with pytest.raises(WrongPassphraseError):
            change_passphrase(cfg, old="not-the-key", new=NEW)
        assert _opens_with(cfg, OLD)
        assert acknowledged(cfg.vault_recovery_marker_file)
        # Nothing changed, so no escrow debris may remain either.
        assert not cfg.vault_pending_passphrase_file.exists()

    def test_refuses_a_legacy_plaintext_db(self, tmp_path: Path) -> None:
        """A plaintext DB has no key to rotate — route to the encrypt-db migration."""
        cfg = _cfg(tmp_path)
        cfg.db_path.parent.mkdir(parents=True, exist_ok=True)
        conn = sqlite3.connect(cfg.db_path)
        conn.execute("CREATE TABLE legacy (x)")
        conn.commit()
        conn.close()

        with pytest.raises(RuntimeError, match="legacy plaintext.*encrypt-db"):
            change_passphrase(cfg, new=NEW)


class TestRewriteTier:
    """Per-tier fan-out after the rekey — always reports, never raises."""

    def test_systemd_creds_reseals_when_available(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """A capable host re-seals the credential file under the new value."""
        cfg = _cfg(tmp_path)
        sealed: list[tuple[str, Path, str]] = []
        monkeypatch.setattr(systemd_creds, "is_available", lambda: True)
        monkeypatch.setattr(
            systemd_creds,
            "seal",
            lambda pw, path, key_mode: sealed.append((pw, path, key_mode)),
        )

        rewrite = _rewrite_tier(cfg, PassphraseTier.SYSTEMD_CREDS, NEW)

        assert rewrite.ok and "re-sealed" in rewrite.detail
        assert sealed == [(NEW, cfg.vault_systemd_creds_file, "auto")]

    def test_systemd_creds_unavailable_purges_the_stale_seal(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """An unsealable host must not keep a credential sealed under the OLD key."""
        cfg = _cfg(tmp_path)
        cfg.vault_systemd_creds_file.parent.mkdir(parents=True, exist_ok=True)
        cfg.vault_systemd_creds_file.write_bytes(b"sealed-under-old")
        monkeypatch.setattr(systemd_creds, "is_available", lambda: False)
        monkeypatch.setattr(systemd_creds, "unavailable_reason", lambda: "systemd too old")

        rewrite = _rewrite_tier(cfg, PassphraseTier.SYSTEMD_CREDS, NEW)

        assert not rewrite.ok
        assert "systemd too old" in rewrite.detail
        assert "vault passphrase seal" in rewrite.detail
        assert not cfg.vault_systemd_creds_file.exists()

    def test_keyring_rewrite_success(self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
        """A keyring backend that takes the write reports a plain success."""
        cfg = _cfg(tmp_path, use_keyring=True)
        stored: list[str] = []
        monkeypatch.setattr(
            encryption, "store_passphrase_in_keyring", lambda pw: stored.append(pw) or True
        )

        rewrite = _rewrite_tier(cfg, PassphraseTier.KEYRING, NEW)

        assert rewrite.ok and rewrite.detail == "keyring entry rewritten"
        assert stored == [NEW]

    def test_unwritable_tier_is_reported(self, tmp_path: Path) -> None:
        """passphrase_command points at an external store terok cannot write."""
        cfg = _cfg(tmp_path)

        rewrite = _rewrite_tier(cfg, PassphraseTier.PASSPHRASE_COMMAND, NEW)

        assert not rewrite.ok
        assert "cannot be rewritten programmatically" in rewrite.detail

    def test_tier_exception_becomes_a_report(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """The DB is already rekeyed at this point — a raising tier must not abort."""
        cfg = _cfg(tmp_path)
        monkeypatch.setattr(systemd_creds, "is_available", lambda: True)

        def _boom(*_a: object, **_kw: object) -> None:
            raise OSError("disk full")

        monkeypatch.setattr(systemd_creds, "seal", _boom)

        rewrite = _rewrite_tier(cfg, PassphraseTier.SYSTEMD_CREDS, NEW)

        assert not rewrite.ok and rewrite.detail == "disk full"


class TestCollectCurrentPassphrase:
    """The current-passphrase seam: chain first, prompt only when locked."""

    def test_broken_durable_tier_exits_with_direction(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """A present-but-unsealable tier fails closed and names the fix."""
        cfg = _cfg(tmp_path)
        cfg.vault_systemd_creds_file.parent.mkdir(parents=True, exist_ok=True)
        cfg.vault_systemd_creds_file.write_bytes(b"sealed-elsewhere")
        monkeypatch.setattr(systemd_creds, "unseal", lambda _path: None)

        with pytest.raises(SystemExit, match="Fix or remove the broken tier"):
            _collect_current_passphrase(cfg)

    def test_unprovisioned_vault_exits_towards_setup(self, tmp_path: Path) -> None:
        """No DB and no tier: there is nothing to change — setup is the answer."""
        cfg = _cfg(tmp_path)

        with pytest.raises(SystemExit, match="run setup to provision"):
            _collect_current_passphrase(cfg)

    def test_locked_vault_prompts_for_the_current_value(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
    ) -> None:
        """A DB with an empty chain is locked — the prompt fills in the value."""
        cfg = _cfg(tmp_path)
        cfg.db_path.parent.mkdir(parents=True, exist_ok=True)
        cfg.db_path.write_bytes(b"stand-in")
        monkeypatch.setattr(encryption, "prompt_passphrase", lambda: OLD)

        assert _collect_current_passphrase(cfg) == OLD
        assert "vault is locked" in capsys.readouterr().out

    def test_cancelled_prompt_exits_cleanly(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """An empty prompt entry maps to a nothing-was-changed exit, not a traceback."""
        cfg = _cfg(tmp_path)
        cfg.db_path.parent.mkdir(parents=True, exist_ok=True)
        cfg.db_path.write_bytes(b"stand-in")

        def _empty() -> str:
            raise ValueError("empty passphrase")

        monkeypatch.setattr(encryption, "prompt_passphrase", _empty)

        with pytest.raises(SystemExit, match="nothing was changed: empty passphrase"):
            _collect_current_passphrase(cfg)


class TestCollectNewPassphraseTTY:
    """The new-passphrase seam's interactive (TTY) half."""

    def test_typed_and_confirmed_value_is_returned(
        self, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
    ) -> None:
        """A typed entry rides the masked-echo prompt straight through."""
        monkeypatch.setattr("sys.stdin.isatty", lambda: True)
        monkeypatch.setattr(encryption, "prompt_new_passphrase", lambda: NEW)

        assert _collect_new_passphrase() == NEW
        assert "NEW passphrase" in capsys.readouterr().out

    def test_mismatched_confirmation_exits_cleanly(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """A confirmation mismatch maps to a nothing-was-changed exit."""
        monkeypatch.setattr("sys.stdin.isatty", lambda: True)

        def _mismatch() -> str:
            raise ValueError("passphrases do not match")

        monkeypatch.setattr(encryption, "prompt_new_passphrase", _mismatch)

        with pytest.raises(SystemExit, match="nothing was changed: passphrases do not match"):
            _collect_new_passphrase()


class TestChangeOrExit:
    """The refusal→exit mapping the CLI handler relies on."""

    def test_wrong_passphrase_names_the_db(self, tmp_path: Path) -> None:
        """The WrongPassphraseError map points at the DB that refused the key."""
        cfg = _cfg(tmp_path)
        _seed_db(cfg, OLD)

        with pytest.raises(SystemExit, match="does not open .*credentials.db"):
            _change_or_exit(cfg, old="not-the-key", new=NEW)
        assert _opens_with(cfg, OLD)

    def test_refusals_map_to_nothing_was_changed(self, tmp_path: Path) -> None:
        """ValueError refusals (here: identical new) exit with the refusal text."""
        cfg = _cfg(tmp_path)
        _seed_db(cfg, OLD)
        _write_session(cfg, OLD)

        with pytest.raises(SystemExit, match="nothing was changed: .*identical"):
            _change_or_exit(cfg, old=OLD, new=OLD)

    def test_locked_db_exit_carries_the_fuser_hint(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """The in-use case tells the operator how to find the holding supervisor."""
        cfg = _cfg(tmp_path)

        def _locked(*_a: object, **_kw: object) -> None:
            raise Exception("database is locked")  # noqa: TRY002 — sqlite's own shape

        monkeypatch.setattr(vault_cmd, "change_passphrase", _locked)

        with pytest.raises(SystemExit, match="fuser -v") as excinfo:
            _change_or_exit(cfg, old=OLD, new=NEW)
        assert str(cfg.db_path) in str(excinfo.value)

    def test_unrelated_exception_propagates(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Anything that isn't a refusal or lock is a real bug — never swallowed."""
        cfg = _cfg(tmp_path)

        def _boom(*_a: object, **_kw: object) -> None:
            raise OSError("disk on fire")

        monkeypatch.setattr(vault_cmd, "change_passphrase", _boom)

        with pytest.raises(OSError, match="disk on fire"):
            _change_or_exit(cfg, old=OLD, new=NEW)


class TestChangeHandlerPiped:
    """The CLI handler's non-TTY (piped stdin) contract."""

    def test_piped_new_passphrase_changes_the_vault(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
    ) -> None:
        cfg = _cfg(tmp_path)
        _seed_db(cfg, OLD)
        _write_session(cfg, OLD)
        monkeypatch.setattr(sys, "stdin", io.StringIO(NEW + "\n"))

        _handle_vault_passphrase_change(cfg=cfg)

        assert _opens_with(cfg, NEW)
        out = capsys.readouterr().out
        assert "re-encrypted" in out
        assert "session file rewritten" in out

    def test_tier_only_change_prints_no_rekey_line(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
    ) -> None:
        """Pre-first-use (no DB): the tier rotates, and no re-encryption is claimed."""
        cfg = _cfg(tmp_path)
        _write_session(cfg, OLD)
        monkeypatch.setattr(sys, "stdin", io.StringIO(NEW + "\n"))

        _handle_vault_passphrase_change(cfg=cfg)

        out = capsys.readouterr().out
        assert "re-encrypted" not in out
        assert "session file rewritten" in out
        assert cfg.vault_passphrase_file.read_text(encoding="utf-8").strip() == NEW

    def test_failed_tier_rewrites_exit_nonzero(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
    ) -> None:
        """The fail-loud contract: a tier left without the new value cannot scroll past."""
        cfg = _cfg(tmp_path, use_keyring=True)
        _seed_db(cfg, OLD)
        monkeypatch.setattr(encryption, "load_passphrase_from_keyring", lambda: OLD)
        monkeypatch.setattr(encryption, "store_passphrase_in_keyring", lambda _v: False)
        monkeypatch.setattr(encryption, "forget_passphrase_in_keyring", lambda: False)
        monkeypatch.setattr(sys, "stdin", io.StringIO(NEW + "\n"))

        with pytest.raises(SystemExit, match="could not be rewritten"):
            _handle_vault_passphrase_change(cfg=cfg)

        out = capsys.readouterr().out
        assert "✗ keyring" in out
        # The change itself succeeded — only the tier fan-out is incomplete.
        assert _opens_with(cfg, NEW)

    def test_piped_mint_refuses_before_changing_anything(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """A minted value needs a TTY to be displayed on — refuse up front."""
        cfg = _cfg(tmp_path)
        _seed_db(cfg, OLD)
        _write_session(cfg, OLD)
        acknowledge(cfg.vault_recovery_marker_file)
        monkeypatch.setattr(sys, "stdin", io.StringIO("\n"))

        with pytest.raises(SystemExit, match="needs a terminal"):
            _handle_vault_passphrase_change(cfg=cfg)
        assert _opens_with(cfg, OLD)
        assert acknowledged(cfg.vault_recovery_marker_file)


class TestChangeHandlerTTY:
    """The CLI handler's interactive (TTY) contract."""

    def test_minted_passphrase_is_announced_after_the_rekey(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
    ) -> None:
        """Empty entry mints a value and announces it only once the vault holds it."""
        from terok_sandbox.commands import credentials as credentials_cmd

        cfg = _cfg(tmp_path)
        _seed_db(cfg, OLD)
        _write_session(cfg, OLD)
        monkeypatch.setattr("sys.stdin.isatty", lambda: True)
        monkeypatch.setattr(encryption, "prompt_new_passphrase", lambda: None)
        announced: list[str] = []
        monkeypatch.setattr(
            credentials_cmd,
            "_announce_generated_passphrase",
            lambda pw, **_kw: announced.append(pw),
        )
        monkeypatch.setattr(
            credentials_cmd, "_maybe_acknowledge_recovery", lambda _cfg, **_kw: None
        )

        _handle_vault_passphrase_change(cfg=cfg)

        (minted,) = announced
        assert _opens_with(cfg, minted) and not _opens_with(cfg, OLD)
        assert "re-encrypted" in capsys.readouterr().out


class TestPlanProvisioning:
    """The shared decision core both frontends render."""

    def test_fresh_host_offers_the_chooser(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        from terok_sandbox.vault.store import systemd_creds

        monkeypatch.setattr(systemd_creds, "is_available", lambda: False)
        plan = plan_provisioning(_cfg(tmp_path))

        assert not plan.provisioned
        assert plan.auto_tier is None
        assert plan.choices == CHOOSER_TIERS
        assert isinstance(plan.keyring_available, bool)

    def test_systemd_creds_auto_selects(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        from terok_sandbox.vault.store import systemd_creds

        monkeypatch.setattr(systemd_creds, "is_available", lambda: True)
        plan = plan_provisioning(_cfg(tmp_path))

        assert plan.auto_tier is PassphraseTier.SYSTEMD_CREDS

    def test_keyring_choice_survives_a_missing_user_config(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """No user-scope config file → the mode persist is a silent no-op, not a crash."""
        from terok_sandbox.commands import credentials as credentials_mod

        monkeypatch.setattr(
            "terok_sandbox.paths.config_file_paths",
            lambda: [("system", tmp_path / "system.yml")],
        )
        credentials_mod._persist_mode_choice(PassphraseTier.KEYRING)  # must not raise
        assert not (tmp_path / "system.yml").exists()

    def test_existing_tier_short_circuits(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        from terok_sandbox.vault.store import systemd_creds

        monkeypatch.setattr(systemd_creds, "is_available", lambda: False)
        cfg = _cfg(tmp_path)
        _write_session(cfg, OLD)
        plan = plan_provisioning(cfg)

        assert plan.provisioned
        assert plan.choices == ()

    def test_default_config_is_built_lazily(self) -> None:
        """``cfg=None`` builds a default config — isolated HOME, stubbed keyring tier."""
        plan = plan_provisioning()

        # The conftest keyring stub resolves "test", so the default
        # config counts as provisioned; the point here is that the
        # cfg=None path built a real SandboxConfig and ran the probe.
        assert plan.provisioned
        assert plan.choices == ()
        assert isinstance(plan.keyring_available, bool)
