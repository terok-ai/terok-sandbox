# SPDX-FileCopyrightText: 2026 Jiri Vyskocil
# SPDX-License-Identifier: Apache-2.0

"""Shared fixtures for the credentials integration matrix.

The matrix covers ``chooser × DB-state × daemon-state`` end-to-end on a
real filesystem with the real ``sqlcipher3`` engine — every test in
this directory operates under ``needs_host_features``.  Tests that
exercise the systemd-creds tier add the ``needs_systemd_creds`` marker
to skip on hosts without a recent-enough systemd.

Fixtures here intentionally avoid running the full ``terok setup``
phase pipeline — that's a sandbox concern, not a credentials concern.
Each test exercises the credentials surface (chooser handler,
migration, lock/unlock, vault seal) directly against a ``tmp_path``-
rooted ``SandboxConfig`` so behaviour matches what the integrating
frontend sees, without coupling to the full setup state machine.
"""

from __future__ import annotations

import sqlite3
from collections.abc import Iterator
from pathlib import Path

import pytest

from terok_sandbox.config import SandboxConfig
from terok_sandbox.vault.store.db import CredentialDB


@pytest.fixture()
def cfg(tmp_path: Path) -> SandboxConfig:
    """Return a tmp_path-rooted ``SandboxConfig`` with deterministic dirs.

    Every tier path is under ``tmp_path``:

    - ``vault_passphrase_file`` (session-unlock) → ``runtime_dir/``
    - ``vault_systemd_creds_file`` → ``vault_dir/``
    - ``db_path`` (credentials.db) → ``vault_dir/``

    No host state is touched.  ``services_mode="socket"`` because the
    matrix doesn't allocate TCP ports.
    """
    return SandboxConfig(
        state_dir=tmp_path / "state",
        runtime_dir=tmp_path / "rt",
        config_dir=tmp_path / "cfg",
        vault_dir=tmp_path / "vault",
        services_mode="socket",
    )


@pytest.fixture()
def encrypted_db(cfg: SandboxConfig) -> CredentialDB:
    """Materialise a fresh SQLCipher DB under ``cfg.db_path`` with a known passphrase.

    Returns a *closed* handle — tests typically only need the file to
    exist at-rest.  Tests that need an open connection should construct
    their own ``CredentialDB(cfg.db_path, passphrase="integration-pw")``.
    """
    db = CredentialDB(cfg.db_path, passphrase="integration-pw")
    db.close()
    return db


@pytest.fixture()
def plaintext_db(cfg: SandboxConfig) -> Path:
    """Seed a legacy plaintext sqlite DB at ``cfg.db_path``.

    Used to drive the migration path.  Plain stdlib ``sqlite3`` (not
    sqlcipher) so the format matches what pre-encryption installs left
    on disk; the migration probe distinguishes the two by the header
    bytes.
    """
    cfg.db_path.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(str(cfg.db_path))
    conn.execute("CREATE TABLE legacy (id INTEGER PRIMARY KEY, blob TEXT)")
    conn.execute("INSERT INTO legacy (blob) VALUES ('marker')")
    conn.commit()
    conn.close()
    return cfg.db_path


@pytest.fixture()
def plaintext_db_with_sidecars(plaintext_db: Path) -> Path:
    """Plaintext DB plus planted ``-wal`` and ``-shm`` sidecars.

    Reproduces what an interrupted WAL-mode daemon would leave behind.
    The migration is responsible for sweeping these into the backup
    tarball and removing them from the live tree.
    """
    Path(str(plaintext_db) + "-wal").write_bytes(b"wal-pages-here")
    Path(str(plaintext_db) + "-shm").write_bytes(b"shm-state-here")
    return plaintext_db


@pytest.fixture()
def stubbed_keyring(monkeypatch: pytest.MonkeyPatch) -> Iterator[dict[str, str]]:
    """In-memory keyring backed by a shared dict — no D-Bus / Secret Service.

    The dict is yielded so tests can pre-populate it (simulating a
    keyring with a stale entry) and assert on it after the handler
    runs.  Both load and store paths in ``credentials.encryption`` are
    redirected here.
    """
    store: dict[str, str] = {}

    def _load() -> str | None:
        return store.get("passphrase")

    def _store(passphrase: str) -> bool:
        store["passphrase"] = passphrase
        return True

    def _forget() -> bool:
        return store.pop("passphrase", None) is not None

    monkeypatch.setattr("terok_sandbox.vault.store.encryption.load_passphrase_from_keyring", _load)
    monkeypatch.setattr("terok_sandbox.vault.store.encryption.store_passphrase_in_keyring", _store)
    monkeypatch.setattr(
        "terok_sandbox.vault.store.encryption.forget_passphrase_in_keyring", _forget
    )
    yield store
