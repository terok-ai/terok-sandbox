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
import sys
from pathlib import Path

import pytest

from terok_sandbox import PassphraseTier, SandboxConfig, change_passphrase
from terok_sandbox.commands.credentials import plan_provisioning
from terok_sandbox.commands.vault import _handle_vault_passphrase_change
from terok_sandbox.vault.store import encryption
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

    def test_no_old_key_sidecars_survive(self, tmp_path: Path) -> None:
        """Leftover WAL/journal frames under the old key would poison the next open."""
        cfg = _cfg(tmp_path)
        _seed_db(cfg, OLD)

        rekey_in_place(cfg.db_path, OLD, NEW)

        for suffix in ("-wal", "-shm", "-journal"):
            assert not Path(str(cfg.db_path) + suffix).exists()


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
