# SPDX-FileCopyrightText: 2026 Jiri Vyskocil
# SPDX-License-Identifier: Apache-2.0

"""Tests for at-rest encryption of the credentials DB."""

from __future__ import annotations

import sqlite3
from pathlib import Path

import pytest

from terok_sandbox.credentials.db import (
    CredentialDB,
    NoPassphraseError,
    PlaintextDBFoundError,
    WrongPassphraseError,
    open_credential_db,
)
from terok_sandbox.credentials.encryption import (
    encrypt_in_place,
    forget_passphrase_in_keyring,
    generate_passphrase,
    is_plaintext_sqlite,
    load_passphrase_from_file,
    load_passphrase_from_keyring,
    open_sqlcipher,
    open_sqlcipher_via_chain,
    prompt_passphrase,
    resolve_passphrase,
    store_passphrase_in_keyring,
)

_PASSPHRASE = "correct-horse-battery-staple"


def _scripted_tty_prompt(monkeypatch: pytest.MonkeyPatch, *responses: str) -> None:
    """Coerce :func:`prompt_passphrase` onto the TTY branch with scripted answers.

    The production code routes through ``prompt_toolkit.prompt`` on a real
    terminal and ``sys.stdin.readline`` otherwise; pytest captures stdin,
    so the non-TTY branch is unreachable.  This helper fixes all three
    knobs in one call: ``isatty()`` reports ``True``, ``prompt_toolkit.prompt``
    returns the next scripted answer per invocation, and any ad-hoc
    ``sys.stdin.readline`` (used by the setup chooser) returns ``""`` so
    callers default through their fallback branch.
    """
    answers = iter(responses)
    monkeypatch.setattr("sys.stdin.isatty", lambda: True)
    monkeypatch.setattr("prompt_toolkit.prompt", lambda *_a, **_kw: next(answers))
    monkeypatch.setattr("sys.stdin.readline", lambda: "")


def _make_cfg(tmp_path: Path, *, use_keyring: bool = False, passphrase: str | None = None):
    """Return a SandboxConfig rooted under tmp_path with deterministic credential knobs."""
    from terok_sandbox.config import SandboxConfig

    return SandboxConfig(
        state_dir=tmp_path / "state",
        runtime_dir=tmp_path / "rt",
        config_dir=tmp_path / "cfg",
        vault_dir=tmp_path / "vault",
        services_mode="socket",
        credentials_passphrase=passphrase,
        credentials_use_keyring=use_keyring,
    )


class TestPlaintextProbe:
    """The setup-time probe distinguishes legacy plaintext sqlite from SQLCipher."""

    def test_missing_file(self, tmp_path: Path) -> None:
        """An absent file is not a stale plaintext DB."""
        assert is_plaintext_sqlite(tmp_path / "absent.db") is False

    def test_empty_file(self, tmp_path: Path) -> None:
        """A zero-byte file is not a stale plaintext DB."""
        path = tmp_path / "empty.db"
        path.touch()
        assert is_plaintext_sqlite(path) is False

    def test_fresh_plaintext_db(self, tmp_path: Path) -> None:
        """A populated plaintext sqlite file is identified as such."""
        path = tmp_path / "plain.db"
        conn = sqlite3.connect(path)
        conn.execute("CREATE TABLE t (x INTEGER)")
        conn.close()
        assert is_plaintext_sqlite(path) is True

    def test_sqlcipher_db(self, tmp_path: Path) -> None:
        """A SQLCipher file fails stdlib quick_check and reports as non-plaintext."""
        path = tmp_path / "encrypted.db"
        conn = open_sqlcipher(path, _PASSPHRASE)
        conn.execute("CREATE TABLE t (x INTEGER)")
        conn.close()
        assert is_plaintext_sqlite(path) is False


class TestLoadPassphraseFromFile:
    """The session-unlock tmpfs tier reads cleanly under varied edge cases."""

    def test_missing_file(self, tmp_path: Path) -> None:
        """Absent file returns None — caller falls through to next tier."""
        assert load_passphrase_from_file(tmp_path / "absent") is None

    def test_with_trailing_newline(self, tmp_path: Path) -> None:
        """Standard editors add a trailing newline; we strip exactly one."""
        path = tmp_path / "p"
        path.write_text(_PASSPHRASE + "\n")
        assert load_passphrase_from_file(path) == _PASSPHRASE

    def test_without_trailing_newline(self, tmp_path: Path) -> None:
        """Atomic writes via ``write_text(s)`` without trailing newline also work."""
        path = tmp_path / "p"
        path.write_text(_PASSPHRASE)
        assert load_passphrase_from_file(path) == _PASSPHRASE

    def test_empty_file_returns_none(self, tmp_path: Path) -> None:
        """An empty file is not a valid passphrase — fall through."""
        path = tmp_path / "p"
        path.touch()
        assert load_passphrase_from_file(path) is None


class TestResolvePassphrase:
    """Walk the resolution chain: file → keyring (opt-in) → config fallback → prompt."""

    def test_file_tier_wins(self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
        """A session-unlock file pre-empts every other tier."""
        from terok_sandbox.credentials import encryption as enc

        monkeypatch.setattr(enc, "load_passphrase_from_file", load_passphrase_from_file)
        monkeypatch.setattr(enc, "load_passphrase_from_keyring", lambda: "from-keyring")
        path = tmp_path / "p"
        path.write_text("from-file\n")
        assert resolve_passphrase(passphrase_file=path, use_keyring=True) == "from-file"

    def test_keyring_only_consulted_when_opted_in(
        self,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        """``use_keyring=False`` short-circuits the keyring tier entirely."""
        from terok_sandbox.credentials import encryption as enc

        called = {"keyring": 0}

        def _boobytrap() -> str | None:
            called["keyring"] += 1
            return "from-keyring"

        monkeypatch.setattr(enc, "load_passphrase_from_keyring", _boobytrap)
        assert resolve_passphrase(use_keyring=False, config_fallback="from-config") == "from-config"
        assert called["keyring"] == 0

    def test_keyring_used_when_opted_in(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """``use_keyring=True`` consults the keyring (Linux Secret Service / Keychain)."""
        from terok_sandbox.credentials import encryption as enc

        monkeypatch.setattr(enc, "load_passphrase_from_keyring", lambda: "from-keyring")
        assert resolve_passphrase(use_keyring=True, config_fallback="from-config") == "from-keyring"

    def test_config_fallback_when_higher_tiers_empty(
        self,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        """File absent + keyring empty (or opted out) → config fallback."""
        from terok_sandbox.credentials import encryption as enc

        monkeypatch.setattr(enc, "load_passphrase_from_keyring", lambda: None)
        assert resolve_passphrase(use_keyring=True, config_fallback="from-config") == "from-config"

    def test_returns_none_when_nothing_resolves(
        self,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        """Every tier empty → caller's job to surface a clear setup error."""
        from terok_sandbox.credentials import encryption as enc

        monkeypatch.setattr(enc, "load_passphrase_from_keyring", lambda: None)
        assert resolve_passphrase() is None

    def test_prompt_on_tty_fires_when_chain_is_empty(
        self,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        """Last-resort prompt fires only with prompt_on_tty=True AND a TTY."""
        from terok_sandbox.credentials import encryption as enc

        monkeypatch.setattr(enc, "load_passphrase_from_keyring", lambda: None)
        _scripted_tty_prompt(monkeypatch, "from-prompt")
        assert resolve_passphrase(prompt_on_tty=True) == "from-prompt"

    def test_prompt_skipped_when_not_tty(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """No TTY → prompt_on_tty has no effect; chain returns None as usual."""
        from terok_sandbox.credentials import encryption as enc

        monkeypatch.setattr(enc, "load_passphrase_from_keyring", lambda: None)
        monkeypatch.setattr("sys.stdin.isatty", lambda: False)
        assert resolve_passphrase(prompt_on_tty=True) is None

    def test_prompt_off_by_default(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """Default behaviour does not prompt even when a TTY is attached."""
        from terok_sandbox.credentials import encryption as enc

        monkeypatch.setattr(enc, "load_passphrase_from_keyring", lambda: None)
        monkeypatch.setattr("sys.stdin.isatty", lambda: True)
        # No mock on prompt_toolkit — would block if called; assertion proves it isn't.
        assert resolve_passphrase() is None


class TestEncryptInPlace:
    """One-shot setup migration from legacy plaintext to SQLCipher."""

    def test_round_trip_preserves_data(self, tmp_path: Path) -> None:
        """Encrypt a populated plaintext DB; reopen encrypted and find the rows."""
        path = tmp_path / "rt.db"
        conn = sqlite3.connect(path)
        conn.execute("CREATE TABLE t (k TEXT, v TEXT)")
        conn.execute("INSERT INTO t VALUES (?, ?)", ("k1", "preserved"))
        conn.commit()
        conn.close()

        encrypt_in_place(path, _PASSPHRASE)
        assert not is_plaintext_sqlite(path)

        reopened = open_sqlcipher(path, _PASSPHRASE)
        try:
            row = reopened.execute("SELECT v FROM t WHERE k = ?", ("k1",)).fetchone()
            assert row == ("preserved",)
        finally:
            reopened.close()

    def test_rejects_empty_passphrase(self, tmp_path: Path) -> None:
        """An empty passphrase would produce a plaintext output via SQLCipher's sentinel."""
        path = tmp_path / "fresh.db"
        sqlite3.connect(path).close()
        with pytest.raises(ValueError, match="empty passphrase"):
            encrypt_in_place(path, "")

    def test_missing_file_raises(self, tmp_path: Path) -> None:
        """Encrypting a non-existent file fails fast."""
        with pytest.raises(FileNotFoundError):
            encrypt_in_place(tmp_path / "no-such-file.db", _PASSPHRASE)

    def test_wal_sidecars_cleaned_after_migration(self, tmp_path: Path) -> None:
        """Plaintext ``-wal``/``-shm`` next to the legacy DB must not survive migration."""
        import os as _os

        path = tmp_path / "wal.db"
        conn = sqlite3.connect(path)
        conn.execute("CREATE TABLE secrets (k TEXT, v TEXT)")
        conn.execute("INSERT INTO secrets VALUES (?, ?)", ("api_key", "plaintext"))
        conn.commit()
        conn.close()
        # SQLite auto-checkpoints WAL on a clean close, so we simulate
        # the "daemon died mid-write" shape by planting the sidecars
        # ourselves — that's exactly the artifact we want migration to
        # mop up regardless of how it got there.
        for suffix in ("-wal", "-shm", "-journal"):
            (tmp_path / f"wal.db{suffix}").write_bytes(b"leftover plaintext")

        encrypt_in_place(path, _PASSPHRASE)

        for suffix in ("-wal", "-shm", "-journal"):
            assert not (tmp_path / f"wal.db{suffix}").exists(), (
                f"plaintext sidecar wal.db{suffix} survived encryption"
            )
        # The encrypted file itself stays 0o600 — no umask leakage.
        assert _os.stat(path).st_mode & 0o777 == 0o600


class TestCredentialDBEncryption:
    """End-to-end open + read round-trip through ``CredentialDB``."""

    def test_open_with_passphrase(self, tmp_path: Path) -> None:
        """Construction with a valid passphrase yields a working DB."""
        path = tmp_path / "ok.db"
        db = CredentialDB(path, passphrase=_PASSPHRASE)
        try:
            db.store_credential("default", "claude", {"token": "secret"})
        finally:
            db.close()

        reopened = CredentialDB(path, passphrase=_PASSPHRASE)
        try:
            assert reopened.load_credential("default", "claude") == {"token": "secret"}
        finally:
            reopened.close()

    def test_empty_passphrase_raises(self, tmp_path: Path) -> None:
        """An empty passphrase is rejected before any file work happens."""
        with pytest.raises(NoPassphraseError, match="no SQLCipher passphrase"):
            CredentialDB(tmp_path / "bad.db", passphrase="")

    def test_plaintext_db_translates_to_clear_error(self, tmp_path: Path) -> None:
        """A stale plaintext DB at the path surfaces as PlaintextDBFoundError."""
        path = tmp_path / "legacy.db"
        conn = sqlite3.connect(path)
        conn.execute("CREATE TABLE x (y INTEGER)")
        conn.commit()
        conn.close()
        assert is_plaintext_sqlite(path)
        with pytest.raises(PlaintextDBFoundError, match="credentials encrypt-db"):
            CredentialDB(path, passphrase=_PASSPHRASE)

    def test_wrong_passphrase_translates_to_clear_error(self, tmp_path: Path) -> None:
        """Opening with the wrong passphrase surfaces WrongPassphraseError, not raw sqlcipher."""
        path = tmp_path / "wrong.db"
        CredentialDB(path, passphrase=_PASSPHRASE).close()
        with pytest.raises(WrongPassphraseError, match="wrong passphrase"):
            CredentialDB(path, passphrase="not-the-right-one")

    def test_open_credential_db_uses_resolution_chain(
        self,
        tmp_path: Path,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        """The configured-open wrapper threads passphrase_file + config_fallback through."""
        path = tmp_path / "wrap.db"
        passphrase_file = tmp_path / "session.passphrase"
        passphrase_file.write_text(_PASSPHRASE + "\n")
        from terok_sandbox.credentials import encryption as enc

        monkeypatch.setattr(enc, "load_passphrase_from_file", load_passphrase_from_file)
        monkeypatch.setattr(enc, "load_passphrase_from_keyring", lambda: None)
        db = open_credential_db(path, passphrase_file=passphrase_file)
        try:
            assert db.load_credential("default", "missing") is None
        finally:
            db.close()

    def test_open_credential_db_no_passphrase_raises(
        self,
        tmp_path: Path,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        """Nothing in any tier → NoPassphraseError pointing at vault unlock or setup."""
        from terok_sandbox.credentials import encryption as enc

        monkeypatch.setattr(enc, "load_passphrase_from_keyring", lambda: None)
        with pytest.raises(NoPassphraseError, match="vault unlock"):
            open_credential_db(tmp_path / "no-key.db")

    def test_open_credential_db_prompt_on_tty(
        self,
        tmp_path: Path,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        """CLI wrapper with prompt_on_tty=True falls through to the interactive prompt."""
        from terok_sandbox.credentials import encryption as enc

        monkeypatch.setattr(enc, "load_passphrase_from_keyring", lambda: None)
        _scripted_tty_prompt(monkeypatch, _PASSPHRASE)
        db = open_credential_db(tmp_path / "p.db", prompt_on_tty=True)
        try:
            assert db.load_credential("default", "missing") is None
        finally:
            db.close()


class TestOpenSqlcipherViaChain:
    """The helper that consolidates resolve+raise+open in one call."""

    def test_opens_when_file_tier_has_passphrase(
        self,
        tmp_path: Path,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        """Session-unlock file → connection opens, no error raised."""
        from terok_sandbox.credentials import encryption as enc

        monkeypatch.setattr(enc, "load_passphrase_from_file", load_passphrase_from_file)
        passphrase_file = tmp_path / "p"
        passphrase_file.write_text(_PASSPHRASE)
        path = tmp_path / "via.db"
        conn = open_sqlcipher_via_chain(path, passphrase_file=passphrase_file)
        try:
            conn.execute("CREATE TABLE t (x INTEGER)")
        finally:
            conn.close()

    def test_opens_when_keyring_has_passphrase(
        self,
        tmp_path: Path,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        """Keyring hit (with opt-in) → connection opens."""
        from terok_sandbox.credentials import encryption as enc

        monkeypatch.setattr(enc, "load_passphrase_from_keyring", lambda: _PASSPHRASE)
        path = tmp_path / "via.db"
        conn = open_sqlcipher_via_chain(path, use_keyring=True)
        try:
            conn.execute("CREATE TABLE t (x INTEGER)")
        finally:
            conn.close()

    def test_raises_when_nothing_resolves(
        self,
        tmp_path: Path,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        """Empty chain → NoPassphraseError with the actionable hint."""
        from terok_sandbox.credentials import encryption as enc

        monkeypatch.setattr(enc, "load_passphrase_from_keyring", lambda: None)
        with pytest.raises(NoPassphraseError, match="vault unlock"):
            open_sqlcipher_via_chain(tmp_path / "empty.db")


class TestPromptPassphrase:
    """The interactive entry point rejects mistakes before they corrupt anything."""

    def test_rejects_empty(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """An empty entry is SQLCipher's no-encryption sentinel; reject it explicitly."""
        _scripted_tty_prompt(monkeypatch, "")
        with pytest.raises(ValueError, match="empty"):
            prompt_passphrase()

    def test_confirm_rejects_mismatch(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """Confirm mode rejects typos so they don't lock the operator out."""
        _scripted_tty_prompt(monkeypatch, "one", "two")
        with pytest.raises(ValueError, match="do not match"):
            prompt_passphrase(confirm=True)

    def test_confirm_accepts_match(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """Two matching entries pass through and return the passphrase."""
        _scripted_tty_prompt(monkeypatch, _PASSPHRASE, _PASSPHRASE)
        assert prompt_passphrase(confirm=True) == _PASSPHRASE

    def test_empty_confirm_generates(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """Setup-path empty entry mints a fresh passphrase — UX affordance, not error."""
        _scripted_tty_prompt(monkeypatch, "")
        pw = prompt_passphrase(confirm=True)
        assert len(pw) >= 40
        assert all(c.isalnum() or c in "-_" for c in pw)

    def test_generated_passphrase_goes_to_stderr_not_stdout(
        self, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
    ) -> None:
        """Pipe-fed automation captures stdout; the generated secret must NOT land there."""
        _scripted_tty_prompt(monkeypatch, "")
        pw = prompt_passphrase(confirm=True)
        captured = capsys.readouterr()
        assert pw not in captured.out
        assert pw in captured.err


class TestKeyringHelpers:
    """Storage-tier helpers fail closed on backend errors."""

    def test_store_returns_true_on_success(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """A working keyring backend persists the secret and reports ``True``."""
        import sys

        calls: dict[str, tuple[str, str, str]] = {}

        def _set_password(svc: str, user: str, pw: str) -> None:
            calls["args"] = (svc, user, pw)

        fake = type("FakeKeyring", (), {"set_password": staticmethod(_set_password)})
        monkeypatch.setitem(sys.modules, "keyring", fake)
        assert store_passphrase_in_keyring("hunter2") is True
        assert calls["args"] == ("terok-sandbox", "credentials-db", "hunter2")

    def test_store_returns_false_when_backend_raises(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """``False`` (not exception) so the chain can fall through to config tier."""
        import sys

        def _boom(*_a: object, **_kw: object) -> None:
            raise RuntimeError("denied")

        fake = type("FakeKeyring", (), {"set_password": staticmethod(_boom)})
        monkeypatch.setitem(sys.modules, "keyring", fake)
        assert store_passphrase_in_keyring("hunter2") is False

    def test_forget_returns_true_on_success(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """``forget`` clears the keyring entry and reports success."""
        import sys

        deleted: dict[str, tuple[str, str]] = {}

        def _delete_password(svc: str, user: str) -> None:
            deleted["args"] = (svc, user)

        fake = type("FakeKeyring", (), {"delete_password": staticmethod(_delete_password)})
        monkeypatch.setitem(sys.modules, "keyring", fake)
        assert forget_passphrase_in_keyring() is True
        assert deleted["args"] == ("terok-sandbox", "credentials-db")

    def test_forget_returns_false_when_backend_raises(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """A denied delete surfaces as ``False`` so callers can keep going."""
        import sys

        def _boom(*_a: object, **_kw: object) -> None:
            raise RuntimeError("denied")

        fake = type("FakeKeyring", (), {"delete_password": staticmethod(_boom)})
        monkeypatch.setitem(sys.modules, "keyring", fake)
        assert forget_passphrase_in_keyring() is False


class TestPromptPassphraseNonTTY:
    """Non-TTY (piped stdin) path: read one line, no prompt_toolkit involvement."""

    def test_reads_from_stdin_when_not_tty(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """``echo s3cret | terok-sandbox …`` lands here — pipe-fed automation works."""
        monkeypatch.setattr("sys.stdin.isatty", lambda: False)
        monkeypatch.setattr("sys.stdin.readline", lambda: "s3cret\n")
        assert prompt_passphrase() == "s3cret"

    def test_rejects_empty_pipe_input(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """A blank pipe payload is still empty; non-TTY path enforces the same guard."""
        monkeypatch.setattr("sys.stdin.isatty", lambda: False)
        monkeypatch.setattr("sys.stdin.readline", lambda: "\n")
        with pytest.raises(ValueError, match="empty"):
            prompt_passphrase()

    def test_ctrl_c_translates_to_systemexit(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """``Ctrl+C`` at the prompt surfaces as a clean ``SystemExit``, not a traceback."""

        def _interrupt(*_a: object, **_kw: object) -> str:
            raise KeyboardInterrupt

        monkeypatch.setattr("sys.stdin.isatty", lambda: True)
        monkeypatch.setattr("prompt_toolkit.prompt", _interrupt)
        with pytest.raises(SystemExit, match="cancelled"):
            prompt_passphrase()


class TestEncryptInPlaceErrors:
    """Edge cases on the migration's error path — keep tmp DB from leaking through."""

    def test_nonzero_export_result_raises_and_cleans_up(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """A non-zero ``sqlcipher_export`` result must surface and clean its tmp file."""
        path = tmp_path / "src.db"
        sqlite3.connect(path).close()

        import sqlcipher3

        original_connect = sqlcipher3.connect

        class _ConnWithBadExport:
            """Wraps the real connection but lies on the export return value."""

            def __init__(self, real: object) -> None:
                self._real = real

            def __getattr__(self, name: str) -> object:
                return getattr(self._real, name)

            def execute(self, sql: str, *args: object) -> object:
                cursor = self._real.execute(sql, *args)
                if "sqlcipher_export" in sql:

                    class _Cur:
                        def fetchone(_self) -> tuple[int]:
                            return (1,)

                    return _Cur()
                return cursor

        def _fake_connect(*args: object, **kwargs: object) -> _ConnWithBadExport:
            return _ConnWithBadExport(original_connect(*args, **kwargs))

        monkeypatch.setattr("sqlcipher3.connect", _fake_connect)
        with pytest.raises(RuntimeError, match="sqlcipher_export returned 1"):
            encrypt_in_place(path, _PASSPHRASE)
        # ``src.db.encrypting`` must be gone — no orphan tmp DB.
        assert not (tmp_path / "src.db.encrypting").exists()


class TestEmptyPassphraseGuards:
    """Defensive guards against ``""`` accidentally becoming a working key."""

    def test_open_sqlcipher_rejects_empty(self, tmp_path: Path) -> None:
        """The lowest-level open path refuses empty — bypassing ``CredentialDB``."""
        with pytest.raises(ValueError, match="empty passphrase"):
            open_sqlcipher(tmp_path / "x.db", "")

    def test_store_in_keyring_rejects_empty(self) -> None:
        """``store_passphrase_in_keyring("")`` raises rather than persisting a blank."""
        # Top-level import captured the real callable before the
        # autouse stub took hold, so calling it directly bypasses the
        # module-attribute patch.
        with pytest.raises(ValueError, match="empty"):
            store_passphrase_in_keyring("")

    def test_resolve_treats_empty_keyring_as_missing(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """A blank keyring entry must not shadow lower tiers in the chain."""
        from terok_sandbox.credentials import encryption as enc

        monkeypatch.setattr(enc, "load_passphrase_from_keyring", lambda: "")
        assert resolve_passphrase(use_keyring=True, config_fallback="from-config") == "from-config"


class TestGeneratePassphrase:
    """Auto-generated passphrases are url-safe and high-entropy."""

    def test_default_length_is_reasonable(self) -> None:
        """A token_urlsafe of 32 bytes yields ~43 url-safe characters."""
        pw = generate_passphrase()
        assert len(pw) >= 40
        assert all(c.isalnum() or c in "-_" for c in pw)

    def test_each_call_is_distinct(self) -> None:
        """Sequential calls must not return the same value."""
        assert generate_passphrase() != generate_passphrase()


class TestProvisionPassphrase:
    """Setup-time passphrase provisioning across the three chooser modes."""

    def test_session_mode_creates_new_tmpfs_file(
        self,
        tmp_path: Path,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        """Session mode prompts (empty → generate) and writes the tmpfs file with 0600."""
        from terok_sandbox.commands import _provision_passphrase

        cfg = _make_cfg(tmp_path)
        # Empty entry takes the generate-and-echo affordance — same UX
        # as ``vault unlock`` against a fresh DB.
        _scripted_tty_prompt(monkeypatch, "")
        pw, source = _provision_passphrase(cfg, mode="session")
        assert source == "session-file"
        assert cfg.vault_passphrase_file.read_text().rstrip("\n") == pw
        # Mode 0600 enforced — same protection as the encrypted DB itself.
        assert (cfg.vault_passphrase_file.stat().st_mode & 0o777) == 0o600

    def test_session_mode_reuses_existing_file(
        self,
        tmp_path: Path,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        """An existing session-unlock file is reused; no fresh generation."""
        from terok_sandbox.commands import _provision_passphrase
        from terok_sandbox.credentials import encryption as enc

        monkeypatch.setattr(enc, "load_passphrase_from_file", load_passphrase_from_file)
        cfg = _make_cfg(tmp_path)
        cfg.vault_passphrase_file.parent.mkdir(parents=True, exist_ok=True, mode=0o700)
        cfg.vault_passphrase_file.write_text(_PASSPHRASE + "\n")
        pw, source = _provision_passphrase(cfg, mode="session")
        assert pw == _PASSPHRASE
        assert source == "session-file"

    def test_keyring_mode_uses_existing_keyring_entry(
        self,
        tmp_path: Path,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        """Keyring mode returns the stored entry verbatim."""
        from terok_sandbox.commands import _provision_passphrase
        from terok_sandbox.credentials import encryption as enc

        monkeypatch.setattr(enc, "load_passphrase_from_keyring", lambda: _PASSPHRASE)
        pw, source = _provision_passphrase(_make_cfg(tmp_path), mode="keyring")
        assert pw == _PASSPHRASE
        assert source == "keyring"

    def test_keyring_mode_generates_when_keyring_empty(
        self,
        tmp_path: Path,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        """Empty keyring + working backend → generate + store, source 'keyring'."""
        from terok_sandbox.commands import _provision_passphrase
        from terok_sandbox.credentials import encryption as enc

        stored: dict[str, str] = {}
        monkeypatch.setattr(enc, "load_passphrase_from_keyring", lambda: None)
        monkeypatch.setattr(
            enc, "store_passphrase_in_keyring", lambda pw: stored.__setitem__("pw", pw) or True
        )
        pw, source = _provision_passphrase(_make_cfg(tmp_path), mode="keyring")
        assert source == "keyring"
        assert pw == stored["pw"]

    def test_keyring_mode_raises_when_backend_denies(
        self,
        tmp_path: Path,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        """No backend / user-denied → RuntimeError with actionable suggestion."""
        from terok_sandbox.commands import _provision_passphrase
        from terok_sandbox.credentials import encryption as enc

        monkeypatch.setattr(enc, "load_passphrase_from_keyring", lambda: None)
        monkeypatch.setattr(enc, "store_passphrase_in_keyring", lambda _pw: False)
        with pytest.raises(RuntimeError, match="different storage mode"):
            _provision_passphrase(_make_cfg(tmp_path), mode="keyring")

    def test_config_mode_uses_existing_config_passphrase(self, tmp_path: Path) -> None:
        """Config mode honours an existing credentials.passphrase value."""
        from terok_sandbox.commands import _provision_passphrase

        pw, source = _provision_passphrase(
            _make_cfg(tmp_path, passphrase=_PASSPHRASE), mode="config"
        )
        assert pw == _PASSPHRASE
        assert source == "config"

    def test_config_mode_prompts_when_unset(
        self,
        tmp_path: Path,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        """Config mode without a stored passphrase prompts on TTY; source 'prompt'."""
        from terok_sandbox.commands import _provision_passphrase

        _scripted_tty_prompt(monkeypatch, _PASSPHRASE, _PASSPHRASE)
        pw, source = _provision_passphrase(_make_cfg(tmp_path), mode="config")
        assert pw == _PASSPHRASE
        assert source == "prompt"

    def test_unknown_mode_raises(self, tmp_path: Path) -> None:
        """An unrecognised mode is a programmer error, not a soft fall-through."""
        from terok_sandbox.commands import _provision_passphrase

        with pytest.raises(ValueError, match="unknown mode"):
            _provision_passphrase(_make_cfg(tmp_path), mode="bogus")


class TestChooserAndEncryptHandler:
    """End-to-end setup chooser + migration handler covering common postures."""

    def test_fresh_install_session_mode_creates_passphrase(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Setup with no existing DB writes the session-unlock file."""
        from terok_sandbox.commands import _handle_credentials_encrypt_db

        cfg = _make_cfg(tmp_path)
        assert not cfg.db_path.exists()
        assert not cfg.vault_passphrase_file.exists()
        _scripted_tty_prompt(monkeypatch, "")
        _handle_credentials_encrypt_db(cfg=cfg)
        assert cfg.vault_passphrase_file.exists()

    def test_existing_plaintext_db_migrates(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Plaintext DB found + session mode → encrypted in place with the new key.

        Migration must leave a tarred plaintext backup next to the DB so
        a failed re-key still has a recovery path.  The operator is
        warned (loudly, in red on a TTY) to delete it.
        """
        import tarfile

        from terok_sandbox.commands import _handle_credentials_encrypt_db

        cfg = _make_cfg(tmp_path)
        cfg.db_path.parent.mkdir(parents=True, exist_ok=True)
        plaintext = sqlite3.connect(cfg.db_path)
        plaintext.execute("CREATE TABLE t (x INTEGER)")
        plaintext.execute("INSERT INTO t VALUES (?)", (42,))
        plaintext.commit()
        plaintext.close()
        assert is_plaintext_sqlite(cfg.db_path)

        _scripted_tty_prompt(monkeypatch, "")
        _handle_credentials_encrypt_db(cfg=cfg)
        assert not is_plaintext_sqlite(cfg.db_path)
        passphrase = cfg.vault_passphrase_file.read_text().rstrip("\n")
        reopened = open_sqlcipher(cfg.db_path, passphrase)
        try:
            (val,) = reopened.execute("SELECT x FROM t").fetchone()
            assert val == 42
        finally:
            reopened.close()

        # Plaintext backup tarball exists, holds the original DB, mode 0600.
        backups = list(cfg.db_path.parent.glob(f"{cfg.db_path.name}.plaintext-backup-*.tar.gz"))
        assert len(backups) == 1, f"expected 1 backup, got {len(backups)}: {backups}"
        backup = backups[0]
        assert backup.stat().st_mode & 0o777 == 0o600
        with tarfile.open(backup, "r:gz") as tar:
            assert cfg.db_path.name in tar.getnames()

    def test_already_encrypted_db_is_noop_on_migration(self, tmp_path: Path) -> None:
        """Encrypted DB short-circuits before any passphrase is provisioned."""
        from terok_sandbox.commands import _handle_credentials_encrypt_db

        cfg = _make_cfg(tmp_path)
        CredentialDB(cfg.db_path, passphrase=_PASSPHRASE).close()
        before_mtime = cfg.db_path.stat().st_mtime_ns
        _handle_credentials_encrypt_db(cfg=cfg)
        assert cfg.db_path.stat().st_mtime_ns == before_mtime  # DB untouched
        # And no new passphrase was minted — the existing one stays canonical.
        assert not cfg.vault_passphrase_file.exists()


class TestUserCancelsKeyring:
    """User clicks Cancel on the OS keyring dialog — must fall through cleanly.

    Pins the Signal-cascade-avoidance contract: a failed keyring access
    never deletes or corrupts the encrypted DB.  Signal Desktop's bug
    is to interpret 'denied' as 'corrupted, must wipe' — we MUST NOT
    do that, however indirectly.
    """

    def test_load_failure_returns_none(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """A keyring backend raising on read surfaces as 'no entry' to the chain."""
        import sys

        def _denied(_svc: str, _user: str) -> str:
            raise RuntimeError("user cancelled")

        # ``load_passphrase_from_keyring`` does ``import keyring`` per
        # call, so swapping the entry in ``sys.modules`` is enough — no
        # module reload needed.  The top-level import captured the real
        # callable before the autouse stub took hold, so calling it
        # directly bypasses the module-attribute patch.
        fake = type("FakeKeyring", (), {"get_password": staticmethod(_denied)})
        monkeypatch.setitem(sys.modules, "keyring", fake)
        assert load_passphrase_from_keyring() is None

    def test_store_failure_falls_through(
        self,
        tmp_path: Path,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        """A denied keyring write surfaces as `False`, never partial state."""
        from terok_sandbox.credentials import encryption as enc

        monkeypatch.setattr(enc, "store_passphrase_in_keyring", lambda _pw: False)
        assert enc.store_passphrase_in_keyring("anything") is False

    def test_failed_keyring_does_not_touch_db(
        self,
        tmp_path: Path,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        """The encrypted DB file is byte-identical after a denied keyring op."""
        import sys

        path = tmp_path / "creds.db"
        CredentialDB(path, passphrase=_PASSPHRASE).close()
        before_bytes = path.read_bytes()

        def _denied(_svc: str, _user: str) -> str:
            raise RuntimeError("denied")

        fake = type("FakeKeyring", (), {"get_password": staticmethod(_denied)})
        monkeypatch.setitem(sys.modules, "keyring", fake)
        assert load_passphrase_from_keyring() is None
        assert path.read_bytes() == before_bytes


class TestPlaintextBackupTarball:
    """The pre-migration plaintext snapshot includes WAL/SHM sidecars too."""

    def test_sidecars_included_in_backup(self, tmp_path: Path) -> None:
        """Leftover ``-wal`` / ``-shm`` files at backup time end up in the tarball.

        Why: those files can hold plaintext pages the daemon hadn't yet
        checkpointed.  If the migration goes sideways, the operator
        needs the *whole* plaintext picture in the backup, not just the
        main DB file.

        Tests the backup helper directly — the full handler flow runs
        ``is_plaintext_sqlite`` first, which has the side effect of
        scrubbing invalid sidecars (sqlite's own recovery behaviour),
        so we exercise the snapshot primitive in isolation.
        """
        import tarfile

        from terok_sandbox.commands import _back_up_plaintext_db

        db = tmp_path / "credentials.db"
        sqlite3.connect(db).close()
        (tmp_path / "credentials.db-wal").write_bytes(b"wal-bytes")
        (tmp_path / "credentials.db-shm").write_bytes(b"shm-bytes")
        (tmp_path / "credentials.db-journal").write_bytes(b"journal-bytes")

        backup_path = _back_up_plaintext_db(db)

        assert backup_path.stat().st_mode & 0o777 == 0o600
        with tarfile.open(backup_path, "r:gz") as tar:
            names = set(tar.getnames())
        assert names == {
            "credentials.db",
            "credentials.db-wal",
            "credentials.db-shm",
            "credentials.db-journal",
        }

    def test_backup_perms_locked_under_permissive_umask(self, tmp_path: Path) -> None:
        """Backup file is 0o600 from creation, NOT chmod'd after a permissive-umask write.

        Pre-fix: ``tarfile.open(path, "w:gz")`` honored the process
        umask, leaving the file world-readable during the write
        window.  Force an open umask and assert no observable mode
        other than 0o600 ever lands on the path.
        """
        import os

        from terok_sandbox.commands import _back_up_plaintext_db

        db = tmp_path / "creds.db"
        sqlite3.connect(db).close()
        prev_umask = os.umask(0o022)
        try:
            backup_path = _back_up_plaintext_db(db)
        finally:
            os.umask(prev_umask)
        # The file exists and is mode 0o600 *with* a permissive
        # umask in effect — proves the O_CREAT|O_EXCL,0o600 path is
        # taken (umask would otherwise make it 0o644).
        assert backup_path.stat().st_mode & 0o777 == 0o600


class TestAskPassphraseMode:
    """Setup chooser defaults to ``session`` on non-TTY runs (CI, piped install)."""

    def test_non_tty_defaults_to_session(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """``terok setup < /dev/null`` lands here — must not block on stdin."""
        from terok_sandbox.commands import _ask_passphrase_mode

        monkeypatch.setattr("sys.stdin.isatty", lambda: False)
        assert _ask_passphrase_mode() == "session"


class TestVaultUnlockLock:
    """``terok-sandbox vault unlock`` / ``vault lock`` CLI handlers."""

    def test_unlock_writes_passphrase_and_restarts_running_daemon(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Happy path: prompt, write tmpfs file, bounce the daemon."""
        from unittest.mock import MagicMock

        from terok_sandbox.commands import _handle_vault_unlock

        cfg = _make_cfg(tmp_path)
        _scripted_tty_prompt(monkeypatch, "freshly-typed")
        mgr = MagicMock()
        mgr.is_daemon_running.return_value = True
        monkeypatch.setattr("terok_sandbox.vault.lifecycle.VaultManager", lambda _cfg: mgr)

        _handle_vault_unlock(cfg=cfg)

        assert cfg.vault_passphrase_file.read_text().rstrip("\n") == "freshly-typed"
        assert (cfg.vault_passphrase_file.stat().st_mode & 0o777) == 0o600
        mgr.stop_daemon.assert_called_once()
        mgr.start_daemon.assert_called_once()

    def test_unlock_skips_restart_when_daemon_not_running(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """No daemon running → file is written, restart is not attempted, message printed."""
        from unittest.mock import MagicMock

        from terok_sandbox.commands import _handle_vault_unlock

        cfg = _make_cfg(tmp_path)
        _scripted_tty_prompt(monkeypatch, "freshly-typed")
        mgr = MagicMock()
        mgr.is_daemon_running.return_value = False
        monkeypatch.setattr("terok_sandbox.vault.lifecycle.VaultManager", lambda _cfg: mgr)

        _handle_vault_unlock(cfg=cfg)

        mgr.stop_daemon.assert_not_called()
        mgr.start_daemon.assert_not_called()

    def test_lock_removes_file_and_stops_daemon(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Lock path: delete tmpfs file, stop daemon — symmetric to unlock."""
        from unittest.mock import MagicMock

        from terok_sandbox.commands import _handle_vault_lock

        cfg = _make_cfg(tmp_path)
        cfg.vault_passphrase_file.parent.mkdir(parents=True, exist_ok=True)
        cfg.vault_passphrase_file.write_text("stale\n")
        mgr = MagicMock()
        mgr.is_daemon_running.return_value = True
        monkeypatch.setattr("terok_sandbox.vault.lifecycle.VaultManager", lambda _cfg: mgr)

        _handle_vault_lock(cfg=cfg)

        assert not cfg.vault_passphrase_file.exists()
        mgr.stop_daemon.assert_called_once()

    def test_lock_is_idempotent_when_already_locked(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Already-locked state (no file, no daemon) is a clean no-op."""
        from unittest.mock import MagicMock

        from terok_sandbox.commands import _handle_vault_lock

        cfg = _make_cfg(tmp_path)
        mgr = MagicMock()
        mgr.is_daemon_running.return_value = False
        monkeypatch.setattr("terok_sandbox.vault.lifecycle.VaultManager", lambda _cfg: mgr)

        _handle_vault_lock(cfg=cfg)  # must not raise

        mgr.stop_daemon.assert_not_called()

    def test_lock_warns_when_non_session_tiers_active(
        self,
        tmp_path: Path,
        monkeypatch: pytest.MonkeyPatch,
        capsys: pytest.CaptureFixture[str],
    ) -> None:
        """Plain ``lock`` with keyring/config tiers warns about silent auto-unlock."""
        from unittest.mock import MagicMock

        from terok_sandbox.commands import _handle_vault_lock

        cfg = _make_cfg(tmp_path, use_keyring=True, passphrase="from-config")
        mgr = MagicMock()
        mgr.is_daemon_running.return_value = False
        monkeypatch.setattr("terok_sandbox.vault.lifecycle.VaultManager", lambda _cfg: mgr)

        _handle_vault_lock(cfg=cfg)

        err = capsys.readouterr().err
        assert "non-session passphrase tiers still active" in err
        assert "keyring" in err
        assert "config.yml" in err
        assert "--forget" in err

    def test_lock_forget_clears_keyring_and_config(
        self,
        tmp_path: Path,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        """``lock --forget`` removes the keyring entry and clears config.yml."""
        from unittest.mock import MagicMock

        from terok_sandbox import config as _config
        from terok_sandbox.commands import _handle_vault_lock

        cfg = _make_cfg(tmp_path, use_keyring=True, passphrase="from-config")
        mgr = MagicMock()
        mgr.is_daemon_running.return_value = False
        monkeypatch.setattr("terok_sandbox.vault.lifecycle.VaultManager", lambda _cfg: mgr)

        forget_calls: dict[str, int] = {"n": 0}

        def _forget() -> bool:
            forget_calls["n"] += 1
            return True

        monkeypatch.setattr(
            "terok_sandbox.credentials.encryption.forget_passphrase_in_keyring", _forget
        )

        user_config = tmp_path / "config.yml"
        user_config.write_text("credentials:\n  use_keyring: true\n  passphrase: from-config\n")
        monkeypatch.setattr(
            "terok_sandbox.paths._config_file_paths", lambda: [("user", user_config)]
        )
        _config._credentials_section.cache_clear()

        _handle_vault_lock(cfg=cfg, forget=True)

        assert forget_calls["n"] == 1
        assert "passphrase: from-config" not in user_config.read_text()


class TestCredentialsSetupPhaseDaemonHandling:
    """The credentials phase must stop a live daemon so migration doesn't race."""

    def test_daemon_is_unconditionally_quiesced_before_migration(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """``stop_daemon`` runs every time — not gated by the PID-file probe.

        ``is_daemon_running`` is PID-file only and reports False for
        systemd-managed daemons; gating on it (as we initially did)
        silently skipped the load-bearing stop on real installs.
        """
        from unittest.mock import MagicMock

        from terok_sandbox.commands import _run_credentials_setup_phase

        cfg = _make_cfg(tmp_path)
        mgr = MagicMock()
        # Mimic the systemd shape: no PID file → is_daemon_running False
        # — and we still expect stop_daemon to fire.
        mgr.is_daemon_running.return_value = False
        monkeypatch.setattr("terok_sandbox.vault.lifecycle.VaultManager", lambda _cfg: mgr)
        _scripted_tty_prompt(monkeypatch, "")

        assert _run_credentials_setup_phase(cfg) is True
        mgr.stop_daemon.assert_called_once()
        mgr.uninstall_systemd_units.assert_not_called()  # not needed on happy path

    def test_locked_db_triggers_auto_recovery_and_succeeds(
        self,
        tmp_path: Path,
        monkeypatch: pytest.MonkeyPatch,
        capsys: pytest.CaptureFixture[str],
    ) -> None:
        """Socket-activation respawn race → uninstall units + retry → success."""
        from unittest.mock import MagicMock

        from terok_sandbox import commands

        cfg = _make_cfg(tmp_path)
        attempts = {"n": 0}

        def _migrate(**_kw: object) -> None:
            attempts["n"] += 1
            if attempts["n"] == 1:
                raise RuntimeError("database is locked")
            # Second call (after auto-uninstall) succeeds.

        monkeypatch.setattr(commands, "_handle_credentials_encrypt_db", _migrate)
        mgr = MagicMock()
        monkeypatch.setattr("terok_sandbox.vault.lifecycle.VaultManager", lambda _cfg: mgr)

        assert commands._run_credentials_setup_phase(cfg) is True
        assert attempts["n"] == 2
        mgr.uninstall_systemd_units.assert_called_once()
        assert "auto-recovering" in capsys.readouterr().out

    def test_locked_db_after_recovery_bails_with_fuser_hint(
        self,
        tmp_path: Path,
        monkeypatch: pytest.MonkeyPatch,
        capsys: pytest.CaptureFixture[str],
    ) -> None:
        """If the lock persists after uninstall, surface the ``fuser`` diagnostic."""
        from unittest.mock import MagicMock

        from terok_sandbox import commands

        def _always_locked(**_kw: object) -> None:
            raise RuntimeError("database is locked")

        monkeypatch.setattr(commands, "_handle_credentials_encrypt_db", _always_locked)
        monkeypatch.setattr("terok_sandbox.vault.lifecycle.VaultManager", lambda _cfg: MagicMock())

        cfg = _make_cfg(tmp_path)
        assert commands._run_credentials_setup_phase(cfg) is False
        out = capsys.readouterr().out
        assert "recovery FAILED" in out
        assert f"fuser -v {cfg.db_path}" in out

    def test_non_lock_error_bails_immediately_no_recovery(
        self,
        tmp_path: Path,
        monkeypatch: pytest.MonkeyPatch,
        capsys: pytest.CaptureFixture[str],
    ) -> None:
        """Unrelated migration errors must NOT trigger auto-uninstall."""
        from unittest.mock import MagicMock

        from terok_sandbox import commands

        def _boom(**_kw: object) -> None:
            raise RuntimeError("disk is on fire")

        monkeypatch.setattr(commands, "_handle_credentials_encrypt_db", _boom)
        mgr = MagicMock()
        monkeypatch.setattr("terok_sandbox.vault.lifecycle.VaultManager", lambda _cfg: mgr)

        assert commands._run_credentials_setup_phase(_make_cfg(tmp_path)) is False
        mgr.uninstall_systemd_units.assert_not_called()
        assert "disk is on fire" in capsys.readouterr().out


class TestPersistModeChoice:
    """Persisting the chooser's decision writes both fields and invalidates caches."""

    def test_keyring_mode_writes_use_keyring_and_clears_passphrase(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Switching to keyring tier removes the inline passphrase so the chain doesn't fork."""
        from terok_sandbox import config as _config
        from terok_sandbox.commands import _persist_mode_choice

        user_config = tmp_path / "config.yml"
        user_config.write_text("credentials:\n  passphrase: leftover\n  use_keyring: false\n")
        monkeypatch.setattr(
            "terok_sandbox.paths._config_file_paths", lambda: [("user", user_config)]
        )
        _config._credentials_section.cache_clear()
        _persist_mode_choice("keyring", "ignored")
        text = user_config.read_text()
        assert "use_keyring: true" in text
        assert "leftover" not in text

    def test_config_mode_writes_passphrase_and_clears_use_keyring(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Switching to config tier inlines the new passphrase and clears the keyring opt-in."""
        from terok_sandbox import config as _config
        from terok_sandbox.commands import _persist_mode_choice

        user_config = tmp_path / "config.yml"
        user_config.write_text("credentials:\n  use_keyring: true\n")
        monkeypatch.setattr(
            "terok_sandbox.paths._config_file_paths", lambda: [("user", user_config)]
        )
        _config._credentials_section.cache_clear()
        _persist_mode_choice("config", "new-secret")
        text = user_config.read_text()
        assert "use_keyring: false" in text
        assert "new-secret" in text

    def test_session_mode_is_noop(self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
        """Session mode doesn't touch config.yml — the tmpfs file is self-describing."""
        from terok_sandbox.commands import _persist_mode_choice

        user_config = tmp_path / "config.yml"
        monkeypatch.setattr(
            "terok_sandbox.paths._config_file_paths", lambda: [("user", user_config)]
        )
        _persist_mode_choice("session", "irrelevant")
        assert not user_config.exists()


class TestCredentialsSetupPhase:
    """The thin wrapper that frames ``_handle_credentials_encrypt_db`` for setup output."""

    def test_returns_true_on_success(self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
        """Happy path: handler runs without raising → True."""
        from terok_sandbox.commands import _run_credentials_setup_phase

        cfg = _make_cfg(tmp_path)
        _scripted_tty_prompt(monkeypatch, "")
        assert _run_credentials_setup_phase(cfg) is True

    def test_returns_false_when_handler_raises(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Any exception → caller can mark ``failed`` and keep going through other phases."""
        from terok_sandbox import commands

        def _boom(**_kw: object) -> None:
            raise RuntimeError("simulated failure")

        monkeypatch.setattr(commands, "_handle_credentials_encrypt_db", _boom)
        assert commands._run_credentials_setup_phase(_make_cfg(tmp_path)) is False
