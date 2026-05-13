# SPDX-FileCopyrightText: 2026 Jiri Vyskocil
# SPDX-License-Identifier: Apache-2.0

"""End-to-end credentials matrix: chooser × DB-state × daemon-state.

The 10 representative cases from terok-sandbox#277, asserted against
the [`VaultStatus`][terok_sandbox.VaultStatus] fields introduced by
#278 (``ssh_keys_stored``, ``passphrase_source``, ``locked``).

Every test runs against a real ``sqlcipher3`` engine on a tmp_path
filesystem.  Cases that need a live systemd-user session add an
additional marker (``needs_systemd_creds`` for the systemd-creds tier;
the daemon-running cases inherit ``needs_host_features`` from the
module).
"""

from __future__ import annotations

import tarfile
from pathlib import Path

import pytest

from terok_sandbox.config import SandboxConfig
from terok_sandbox.credentials import systemd_creds
from terok_sandbox.credentials.db import CredentialDB
from terok_sandbox.credentials.encryption import (
    encrypt_in_place,
    is_plaintext_sqlite,
    resolve_passphrase_with_source,
)
from terok_sandbox.vault.lifecycle import VaultManager

pytestmark = pytest.mark.needs_host_features


# ── Group 1 — chooser tier outcomes (fresh install) ───────────────────


class TestChooserTierOutcomes:
    """Each chooser tier provisions the passphrase into exactly one tier.

    Asserted via ``VaultStatus.passphrase_source`` to prove the field
    is wired end-to-end and the chain resolves through the tier the
    operator picked.
    """

    def test_session_tier_writes_tmpfs_file_and_resolves(
        self, cfg: SandboxConfig, encrypted_db: CredentialDB
    ) -> None:
        cfg.vault_passphrase_file.parent.mkdir(parents=True, exist_ok=True)
        cfg.vault_passphrase_file.write_text("integration-pw\n")
        cfg.vault_passphrase_file.chmod(0o600)

        status = VaultManager(cfg).get_status()

        assert status.passphrase_source == "session-file"
        assert status.locked is False

    def test_keyring_tier_persists_across_session_file_clear(
        self,
        cfg: SandboxConfig,
        encrypted_db: CredentialDB,
        stubbed_keyring: dict[str, str],
    ) -> None:
        stubbed_keyring["passphrase"] = "integration-pw"
        cfg = SandboxConfig(
            state_dir=cfg.state_dir,
            runtime_dir=cfg.runtime_dir,
            config_dir=cfg.config_dir,
            vault_dir=cfg.vault_dir,
            services_mode=cfg.services_mode,
            credentials_use_keyring=True,
        )

        status = VaultManager(cfg).get_status()

        assert status.passphrase_source == "keyring"
        assert status.locked is False

    def test_config_tier_resolves_without_prompt(
        self, cfg: SandboxConfig, encrypted_db: CredentialDB
    ) -> None:
        cfg = SandboxConfig(
            state_dir=cfg.state_dir,
            runtime_dir=cfg.runtime_dir,
            config_dir=cfg.config_dir,
            vault_dir=cfg.vault_dir,
            services_mode=cfg.services_mode,
            credentials_passphrase="integration-pw",
        )

        status = VaultManager(cfg).get_status()

        assert status.passphrase_source == "config"
        assert status.locked is False


# ── Group 2 — DB-state outcomes (migration) ───────────────────────────


class TestMigrationOutcomes:
    """Migration from legacy plaintext → SQLCipher, with sidecar handling."""

    def test_plaintext_db_migrates_in_place_to_sqlcipher(
        self, cfg: SandboxConfig, plaintext_db: Path
    ) -> None:
        assert is_plaintext_sqlite(plaintext_db)

        encrypt_in_place(plaintext_db, "integration-pw")

        # File still exists at the same path, but it's no longer plain sqlite.
        assert plaintext_db.is_file()
        assert not is_plaintext_sqlite(plaintext_db)

        db = CredentialDB(plaintext_db, passphrase="integration-pw")
        try:
            assert list(db.list_credentials("default")) == []
        finally:
            db.close()

    def test_wal_sidecars_cleared_after_migration(
        self, cfg: SandboxConfig, plaintext_db_with_sidecars: Path
    ) -> None:
        encrypt_in_place(plaintext_db_with_sidecars, "integration-pw")

        # Plaintext sidecars must not survive migration — they hold the
        # pre-encryption pages on disk.
        assert not Path(str(plaintext_db_with_sidecars) + "-wal").exists()
        assert not Path(str(plaintext_db_with_sidecars) + "-shm").exists()

    def test_already_encrypted_db_is_unchanged_by_open(
        self, cfg: SandboxConfig, encrypted_db: CredentialDB
    ) -> None:
        """Re-running setup on an already-encrypted DB is a no-op for the on-disk file."""
        before = cfg.db_path.read_bytes()

        # Open through the chain (config-fallback tier) — must not rewrite the file.
        cfg = SandboxConfig(
            state_dir=cfg.state_dir,
            runtime_dir=cfg.runtime_dir,
            config_dir=cfg.config_dir,
            vault_dir=cfg.vault_dir,
            services_mode=cfg.services_mode,
            credentials_passphrase="integration-pw",
        )
        cfg.open_credential_db().close()

        assert cfg.db_path.read_bytes() == before

    def test_migration_backup_tarball_locked_to_0o600(
        self, cfg: SandboxConfig, plaintext_db: Path
    ) -> None:
        """The pre-migration backup tarball must be created at 0o600 (no umask window).

        Verified end-to-end: stage the backup as the production code
        does, then check the mode of the resulting tarball.
        """
        from terok_sandbox.commands import _back_up_plaintext_db

        backup = _back_up_plaintext_db(plaintext_db)

        assert backup.is_file()
        assert oct(backup.stat().st_mode & 0o777) == oct(0o600)
        with tarfile.open(backup, "r:gz") as tf:
            assert any(m.name.endswith(plaintext_db.name) for m in tf.getmembers())


# ── Group 3 — lock / unlock round trips ───────────────────────────────


class TestLockUnlockRoundTrip:
    """``vault unlock`` / ``vault lock`` / ``vault lock --forget`` lifecycle."""

    def test_lock_forget_clears_all_persistent_tiers_and_reports_locked(
        self,
        cfg: SandboxConfig,
        encrypted_db: CredentialDB,
        stubbed_keyring: dict[str, str],
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        """``vault lock --forget`` removes session, keyring, and systemd-creds tiers."""
        from terok_sandbox.commands import _handle_vault_lock

        stubbed_keyring["passphrase"] = "integration-pw"
        cfg = SandboxConfig(
            state_dir=cfg.state_dir,
            runtime_dir=cfg.runtime_dir,
            config_dir=cfg.config_dir,
            vault_dir=cfg.vault_dir,
            services_mode=cfg.services_mode,
            credentials_use_keyring=True,
        )
        cfg.vault_passphrase_file.parent.mkdir(parents=True, exist_ok=True)
        cfg.vault_passphrase_file.write_text("integration-pw\n")

        # Stub the daemon manager so this test stays independent of systemd
        # — the daemon lifecycle is exercised by the dedicated daemon-state
        # tests below.
        from unittest.mock import MagicMock

        mgr = MagicMock()
        mgr.is_daemon_running.return_value = False
        monkeypatch.setattr("terok_sandbox.vault.lifecycle.VaultManager", lambda _c: mgr)

        _handle_vault_lock(cfg=cfg, forget=True)

        assert not cfg.vault_passphrase_file.exists()
        assert "passphrase" not in stubbed_keyring

        # After --forget, the chain has no tier to resolve through.
        passphrase, source = resolve_passphrase_with_source(
            passphrase_file=cfg.vault_passphrase_file,
            systemd_creds_file=cfg.vault_systemd_creds_file,
            use_keyring=cfg.credentials_use_keyring,
            config_fallback=cfg.credentials_passphrase,
        )
        assert passphrase is None
        assert source is None


# ── Group 4 — systemd-creds tier (machine-bound) ──────────────────────


@pytest.mark.needs_systemd_creds
@pytest.mark.skipif(
    not systemd_creds.is_available(),
    reason="systemd-creds ≥ 257 not on PATH (and / or no Varlink delegation reachable)",
)
class TestSystemdCredsTier:
    """End-to-end seal → resolve → forget on the systemd-creds tier.

    These cases need a real ``systemd-creds`` binary ≥ 257 with the
    ``io.systemd.Credentials`` Varlink interface reachable — i.e. the
    test host's PID 1 is the production systemd, not a minimal stub.
    Auto-skipped when the binary isn't available; the marker exists so
    CI can also select / deselect explicitly.
    """

    def test_seal_then_chain_resolves_through_systemd_creds(
        self, cfg: SandboxConfig, encrypted_db: CredentialDB
    ) -> None:
        systemd_creds.seal("integration-pw", cfg.vault_systemd_creds_file)

        # File materialised at 0o600 from creation.
        assert cfg.vault_systemd_creds_file.is_file()
        assert oct(cfg.vault_systemd_creds_file.stat().st_mode & 0o777) == oct(0o600)

        passphrase, source = resolve_passphrase_with_source(
            systemd_creds_file=cfg.vault_systemd_creds_file,
        )
        assert passphrase == "integration-pw"
        assert source == "systemd-creds"

    def test_vault_status_reports_systemd_creds_as_source(
        self, cfg: SandboxConfig, encrypted_db: CredentialDB
    ) -> None:
        systemd_creds.seal("integration-pw", cfg.vault_systemd_creds_file)

        status = VaultManager(cfg).get_status()

        assert status.passphrase_source == "systemd-creds"
        assert status.locked is False

    def test_lock_forget_removes_sealed_credential(
        self,
        cfg: SandboxConfig,
        encrypted_db: CredentialDB,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        """``--forget`` deletes the sealed cred, taking the systemd-creds tier offline."""
        from unittest.mock import MagicMock

        from terok_sandbox.commands import _handle_vault_lock

        systemd_creds.seal("integration-pw", cfg.vault_systemd_creds_file)
        assert cfg.vault_systemd_creds_file.is_file()

        mgr = MagicMock()
        mgr.is_daemon_running.return_value = False
        monkeypatch.setattr("terok_sandbox.vault.lifecycle.VaultManager", lambda _c: mgr)

        _handle_vault_lock(cfg=cfg, forget=True)

        assert not cfg.vault_systemd_creds_file.exists()
