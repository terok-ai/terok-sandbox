# SPDX-FileCopyrightText: 2026 Jiri Vyskocil
# SPDX-License-Identifier: Apache-2.0

"""Tests for at-rest encryption of the credentials DB."""

from __future__ import annotations

import sqlite3
from pathlib import Path

import pytest

from terok_sandbox.vault.store.db import (
    CredentialDB,
    NoPassphraseError,
    PlaintextDBFoundError,
    WrongPassphraseError,
    open_credential_db,
    open_credential_db_with_source,
)
from terok_sandbox.vault.store.encryption import (
    encrypt_in_place,
    forget_passphrase_in_keyring,
    generate_passphrase,
    is_plaintext_sqlite,
    load_passphrase_from_command,
    load_passphrase_from_file,
    load_passphrase_from_keyring,
    open_sqlcipher,
    open_sqlcipher_via_chain,
    prompt_passphrase,
    resolve_passphrase,
    resolve_passphrase_with_source,
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


def _disable_systemd_creds(monkeypatch: pytest.MonkeyPatch) -> None:
    """Force the chooser path by pretending systemd-creds isn't available.

    ``_handle_credentials_encrypt_db`` auto-detects systemd-creds and
    bypasses the chooser when present.  Tests that exercise the
    chooser path need to disable that detection explicitly so the
    test outcome doesn't depend on the host's systemd version.
    """
    monkeypatch.setattr("terok_sandbox.vault.store.systemd_creds.is_available", lambda: False)


class _TtyCapture:
    """Records writes to ``/dev/tty`` and serves scripted reads back.

    The production code uses ``/dev/tty`` for two directions: the
    auto-mint announcement writes the passphrase there, and the ack
    flow reads the operator's "SAVED" confirmation from the same
    file.  The capture supports both — *response_lines* feeds the
    reader, *value* accumulates the writes.
    """

    def __init__(self, response_lines: tuple[str, ...] = ()) -> None:
        self.value = ""
        self._responses = list(response_lines)

    def __enter__(self) -> _TtyCapture:
        return self

    def __exit__(self, *_exc: object) -> None:
        return None

    def write(self, s: str) -> int:
        self.value += s
        return len(s)

    def flush(self) -> None:
        """No-op flush so ``_read_from_controlling_tty`` can pump the prompt."""

    def readline(self) -> str:
        """Pop the next scripted response, or return ``""`` (= "no ack typed")."""
        if not self._responses:
            return ""
        return self._responses.pop(0) + "\n"


def _patch_dev_tty(
    monkeypatch: pytest.MonkeyPatch,
    *,
    responses: tuple[str, ...] = (),
) -> _TtyCapture:
    """Divert ``Path("/dev/tty").open(...)`` into an in-memory capture.

    The production helpers write the generated passphrase to ``/dev/tty``
    explicitly so stdout redirects can't capture the recovery key.
    Tests patch ``Path.open`` selectively (only for the ``/dev/tty``
    target) so other ``Path`` reads keep working and we can inspect
    what the helper would have written to the operator's terminal.

    *responses* feeds the ack-flow reader; the default ``()`` returns
    ``""`` for every read, simulating an operator who walked away
    without typing SAVED.
    """
    from pathlib import Path as _Path

    capture = _TtyCapture(response_lines=responses)
    real_open = _Path.open

    def _selective_open(self, *args, **kwargs):
        if str(self) == "/dev/tty":
            return capture
        return real_open(self, *args, **kwargs)

    monkeypatch.setattr(_Path, "open", _selective_open)
    return capture


def _make_cfg(
    tmp_path: Path,
    *,
    use_keyring: bool = False,
    passphrase: str | None = None,
    passphrase_command: str | None = None,
):
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
        credentials_passphrase_command=passphrase_command,
    )


def _ack_recovery(cfg) -> None:
    """Mark the recovery key as saved so the escrow-before-destroy / -enable gates pass."""
    from terok_sandbox.vault.store.recovery import acknowledge

    acknowledge(cfg.vault_recovery_marker_file)


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

    def test_blocked_read_returns_none_and_warns(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture
    ) -> None:
        """EACCES / SELinux denials degrade to None but leave a trace in the log.

        Silent degradation made a blocked read indistinguishable from a
        locked vault on every surface — the warning is the only breadcrumb.
        """
        path = tmp_path / "p"
        path.write_text(_PASSPHRASE)
        real_read_text = Path.read_text

        def _denied(self: Path, *args: object, **kwargs: object) -> str:
            if self == path:
                raise PermissionError(13, "Permission denied")
            return real_read_text(self, *args, **kwargs)  # type: ignore[arg-type]

        monkeypatch.setattr(Path, "read_text", _denied)
        with caplog.at_level("WARNING", logger="terok_sandbox.vault.store.encryption"):
            assert load_passphrase_from_file(path) is None
        assert "exists but is unreadable" in caplog.text

    def test_missing_file_does_not_warn(
        self, tmp_path: Path, caplog: pytest.LogCaptureFixture
    ) -> None:
        """An absent file is the normal locked state — no log noise."""
        with caplog.at_level("WARNING", logger="terok_sandbox.vault.store.encryption"):
            assert load_passphrase_from_file(tmp_path / "absent") is None
        assert "unreadable" not in caplog.text


class TestResolvePassphrase:
    """Walk the resolution chain: file → systemd-creds → keyring → passphrase_command → config → prompt."""

    def test_file_tier_wins(self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
        """A session-unlock file pre-empts every other tier."""
        from terok_sandbox.vault.store import encryption as enc

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
        from terok_sandbox.vault.store import encryption as enc

        called = {"keyring": 0}

        def _boobytrap() -> str | None:
            called["keyring"] += 1
            return "from-keyring"

        monkeypatch.setattr(enc, "load_passphrase_from_keyring", _boobytrap)
        assert resolve_passphrase(use_keyring=False, config_fallback="from-config") == "from-config"
        assert called["keyring"] == 0

    def test_keyring_used_when_opted_in(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """``use_keyring=True`` consults the keyring (Linux Secret Service / Keychain)."""
        from terok_sandbox.vault.store import encryption as enc

        monkeypatch.setattr(enc, "load_passphrase_from_keyring", lambda: "from-keyring")
        assert resolve_passphrase(use_keyring=True, config_fallback="from-config") == "from-keyring"

    def test_config_fallback_when_higher_tiers_empty(
        self,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        """File absent + keyring empty (or opted out) → config fallback."""
        from terok_sandbox.vault.store import encryption as enc

        monkeypatch.setattr(enc, "load_passphrase_from_keyring", lambda: None)
        assert resolve_passphrase(use_keyring=True, config_fallback="from-config") == "from-config"

    def test_returns_none_when_nothing_resolves(
        self,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        """Every tier empty → caller's job to surface a clear setup error."""
        from terok_sandbox.vault.store import encryption as enc

        monkeypatch.setattr(enc, "load_passphrase_from_keyring", lambda: None)
        assert resolve_passphrase() is None

    def test_prompt_on_tty_fires_when_chain_is_empty(
        self,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        """Last-resort prompt fires only with prompt_on_tty=True AND a TTY."""
        from terok_sandbox.vault.store import encryption as enc

        monkeypatch.setattr(enc, "load_passphrase_from_keyring", lambda: None)
        _scripted_tty_prompt(monkeypatch, "from-prompt")
        assert resolve_passphrase(prompt_on_tty=True) == "from-prompt"

    def test_prompt_skipped_when_not_tty(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """No TTY → prompt_on_tty has no effect; chain returns None as usual."""
        from terok_sandbox.vault.store import encryption as enc

        monkeypatch.setattr(enc, "load_passphrase_from_keyring", lambda: None)
        monkeypatch.setattr("sys.stdin.isatty", lambda: False)
        assert resolve_passphrase(prompt_on_tty=True) is None

    def test_prompt_off_by_default(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """Default behaviour does not prompt even when a TTY is attached."""
        from terok_sandbox.vault.store import encryption as enc

        monkeypatch.setattr(enc, "load_passphrase_from_keyring", lambda: None)
        monkeypatch.setattr("sys.stdin.isatty", lambda: True)
        # No mock on prompt_toolkit — would block if called; assertion proves it isn't.
        assert resolve_passphrase() is None


class TestResolvePassphraseWithSource:
    """The source-tracking variant labels each tier as it hits."""

    def test_session_file_source(self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
        from terok_sandbox.vault.store import encryption as enc

        monkeypatch.setattr(enc, "load_passphrase_from_file", load_passphrase_from_file)
        path = tmp_path / "p"
        path.write_text("file-pw\n")
        assert resolve_passphrase_with_source(passphrase_file=path) == ("file-pw", "session-file")

    def test_systemd_creds_source(self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
        """systemd-creds tier slots between session-file and keyring."""
        from terok_sandbox.vault.store import encryption as enc, systemd_creds as sc

        monkeypatch.setattr(enc, "load_passphrase_from_keyring", lambda: "ring-pw")
        monkeypatch.setattr(sc, "unseal", lambda _p: "sealed-pw")
        cred = tmp_path / "v.cred"
        cred.write_bytes(b"sealed-blob")
        # Both systemd-creds and keyring would succeed; systemd-creds wins because it sits above.
        assert resolve_passphrase_with_source(systemd_creds_file=cred, use_keyring=True) == (
            "sealed-pw",
            "systemd-creds",
        )

    def test_systemd_creds_present_but_unsealable_fails_closed(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """A locked / corrupt / wrong-machine credential raises rather than silently downgrading.

        Falling through to keyring / config would change the security
        posture (machine-bound → keyring or plaintext-on-disk) without
        the operator's knowledge — a classic auth-chain downgrade.
        """
        from terok_sandbox.vault.store import encryption as enc, systemd_creds as sc

        monkeypatch.setattr(enc, "load_passphrase_from_keyring", lambda: "ring-pw")
        monkeypatch.setattr(sc, "unseal", lambda _p: None)
        cred = tmp_path / "v.cred"
        cred.write_bytes(b"sealed-blob")
        with pytest.raises(WrongPassphraseError, match="could not be unsealed"):
            resolve_passphrase_with_source(systemd_creds_file=cred, use_keyring=True)

    def test_systemd_creds_absent_falls_through(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """A missing credential file is "tier not configured" — fall through cleanly."""
        from unittest.mock import MagicMock

        from terok_sandbox.vault.store import encryption as enc, systemd_creds as sc

        monkeypatch.setattr(enc, "load_passphrase_from_keyring", lambda: "ring-pw")
        unseal = MagicMock()
        monkeypatch.setattr(sc, "unseal", unseal)
        absent = tmp_path / "never-created.cred"
        assert resolve_passphrase_with_source(systemd_creds_file=absent, use_keyring=True) == (
            "ring-pw",
            "keyring",
        )
        unseal.assert_not_called()

    def test_session_file_pre_empts_systemd_creds(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Explicit session-unlock outranks the machine-bound tier — operator intent wins."""
        from unittest.mock import MagicMock

        from terok_sandbox.vault.store import encryption as enc, systemd_creds as sc

        monkeypatch.setattr(enc, "load_passphrase_from_file", load_passphrase_from_file)
        unseal = MagicMock(return_value="sealed-pw")
        monkeypatch.setattr(sc, "unseal", unseal)
        session = tmp_path / "session"
        session.write_text("session-pw")
        cred = tmp_path / "v.cred"
        cred.write_bytes(b"sealed-blob")
        result = resolve_passphrase_with_source(passphrase_file=session, systemd_creds_file=cred)
        assert result == ("session-pw", "session-file")
        unseal.assert_not_called()  # tier skipped entirely

    def test_keyring_source(self, monkeypatch: pytest.MonkeyPatch) -> None:
        from terok_sandbox.vault.store import encryption as enc

        monkeypatch.setattr(enc, "load_passphrase_from_keyring", lambda: "ring-pw")
        assert resolve_passphrase_with_source(use_keyring=True) == ("ring-pw", "keyring")

    def test_passphrase_command_source(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """A helper command sits between keyring and config in the chain."""
        from terok_sandbox.vault.store import encryption as enc

        monkeypatch.setattr(enc, "load_passphrase_from_keyring", lambda: None)
        # Both helper and config would resolve; helper wins because it sits above.
        result = resolve_passphrase_with_source(
            passphrase_command="/bin/echo helper-pw",
            config_fallback="cfg-pw",
        )
        assert result == ("helper-pw", "passphrase-command")

    def test_keyring_pre_empts_passphrase_command(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """Keyring outranks the helper — explicit OS-level storage wins over delegation."""
        from unittest.mock import MagicMock

        from terok_sandbox.vault.store import encryption as enc

        monkeypatch.setattr(enc, "load_passphrase_from_keyring", lambda: "ring-pw")
        spy = MagicMock()
        monkeypatch.setattr(enc, "load_passphrase_from_command", spy)
        result = resolve_passphrase_with_source(
            use_keyring=True,
            passphrase_command="/bin/echo never-runs",
        )
        assert result == ("ring-pw", "keyring")
        spy.assert_not_called()

    def test_passphrase_command_broken_fails_closed(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """A configured-but-empty helper raises rather than silently downgrading to config."""
        from terok_sandbox.vault.store import encryption as enc

        monkeypatch.setattr(enc, "load_passphrase_from_keyring", lambda: None)
        with pytest.raises(WrongPassphraseError, match="passphrase_command produced no passphrase"):
            resolve_passphrase_with_source(
                passphrase_command="/bin/false",
                config_fallback="should-not-be-used",
            )

    def test_empty_passphrase_command_falls_through(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """An unset / empty-string command is "tier not configured" — fall through cleanly."""
        from terok_sandbox.vault.store import encryption as enc

        monkeypatch.setattr(enc, "load_passphrase_from_keyring", lambda: None)
        assert resolve_passphrase_with_source(
            passphrase_command="",
            config_fallback="cfg-pw",
        ) == ("cfg-pw", "config")

    def test_config_source(self, monkeypatch: pytest.MonkeyPatch) -> None:
        from terok_sandbox.vault.store import encryption as enc

        monkeypatch.setattr(enc, "load_passphrase_from_keyring", lambda: None)
        assert resolve_passphrase_with_source(config_fallback="cfg-pw") == ("cfg-pw", "config")

    def test_prompt_source(self, monkeypatch: pytest.MonkeyPatch) -> None:
        from terok_sandbox.vault.store import encryption as enc

        monkeypatch.setattr(enc, "load_passphrase_from_keyring", lambda: None)
        _scripted_tty_prompt(monkeypatch, "tty-pw")
        assert resolve_passphrase_with_source(prompt_on_tty=True) == ("tty-pw", "prompt")

    def test_none_when_empty(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """Every tier empty → (None, None) so VaultStatus.locked stays derivable."""
        from terok_sandbox.vault.store import encryption as enc

        monkeypatch.setattr(enc, "load_passphrase_from_keyring", lambda: None)
        assert resolve_passphrase_with_source() == (None, None)


class TestLoadPassphraseFromCommand:
    """The `passphrase_command` tier delegates retrieval to an operator-supplied helper.

    Mirrors the silent-on-failure shape of the other tier primitives
    so the resolver can decide whether ``None`` means "skip this tier"
    or "fail closed".
    """

    def test_stdout_is_returned_stripped(self) -> None:
        """Trailing newline from ``echo`` (and any helper) is trimmed."""
        assert load_passphrase_from_command("/bin/echo hunter2") == "hunter2"

    def test_quoted_argv_is_shlex_split(self) -> None:
        """``shlex`` handles quoted arguments so YAML strings need no special escaping."""
        # /bin/sh -c 'printf "a b"' — one quoted arg through shell, no trailing newline
        assert load_passphrase_from_command('/bin/sh -c "printf abc"') == "abc"

    def test_blank_command_returns_none(self) -> None:
        """A whitespace-only command shlex-splits to nothing — fall through, don't exec."""
        assert load_passphrase_from_command("   ") is None

    def test_non_zero_exit_returns_none(self, caplog: pytest.LogCaptureFixture) -> None:
        """A failed helper logs the exit code + stderr at WARNING and returns ``None``."""
        with caplog.at_level("WARNING", logger="terok_sandbox.vault.store.encryption"):
            assert load_passphrase_from_command("/bin/false") is None
        assert "exited 1" in caplog.text

    def test_missing_binary_returns_none(self, caplog: pytest.LogCaptureFixture) -> None:
        """ENOENT (helper not installed) is silent-with-WARNING, not a crash."""
        with caplog.at_level("WARNING", logger="terok_sandbox.vault.store.encryption"):
            assert load_passphrase_from_command("/nonexistent/binary") is None
        assert "failed to spawn" in caplog.text

    def test_empty_stdout_returns_none(self) -> None:
        """A helper that exits 0 with no output is treated as "nothing to give"."""
        assert load_passphrase_from_command("/bin/true") is None

    def test_unbalanced_quotes_return_none(self, caplog: pytest.LogCaptureFixture) -> None:
        """``shlex.split`` rejects unbalanced quotes — log + None, no exception bubbles."""
        with caplog.at_level("WARNING", logger="terok_sandbox.vault.store.encryption"):
            assert load_passphrase_from_command('pass "show terok') is None
        assert "shlex parse failed" in caplog.text

    def test_timeout_returns_none(self, caplog: pytest.LogCaptureFixture) -> None:
        """A wedged helper hits the budget and falls through with a WARNING."""
        # Sub-second timeout so the test stays fast.
        with caplog.at_level("WARNING", logger="terok_sandbox.vault.store.encryption"):
            assert load_passphrase_from_command("/bin/sleep 5", timeout=0.1) is None
        assert "timed out" in caplog.text


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
        from terok_sandbox.vault.store import encryption as enc

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
        """Nothing in any tier → diagnostic NoPassphraseError naming the DB path."""
        from terok_sandbox.vault.store import encryption as enc

        monkeypatch.setattr(enc, "load_passphrase_from_keyring", lambda: None)
        with pytest.raises(NoPassphraseError, match="no SQLCipher passphrase"):
            open_credential_db(tmp_path / "no-key.db")

    def test_open_credential_db_prompt_on_tty(
        self,
        tmp_path: Path,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        """CLI wrapper with prompt_on_tty=True falls through to the interactive prompt."""
        from terok_sandbox.vault.store import encryption as enc

        monkeypatch.setattr(enc, "load_passphrase_from_keyring", lambda: None)
        _scripted_tty_prompt(monkeypatch, _PASSPHRASE)
        db = open_credential_db(tmp_path / "p.db", prompt_on_tty=True)
        try:
            assert db.load_credential("default", "missing") is None
        finally:
            db.close()

    def test_open_credential_db_with_source_reports_tier(
        self,
        tmp_path: Path,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        """The source-aware opener returns which tier of the chain hit."""
        from terok_sandbox.vault.store import encryption as enc

        monkeypatch.setattr(enc, "load_passphrase_from_keyring", lambda: _PASSPHRASE)
        db, source = open_credential_db_with_source(tmp_path / "src.db", use_keyring=True)
        try:
            assert source == "keyring"
        finally:
            db.close()

    def test_open_credential_db_with_source_raises_on_empty_chain(
        self,
        tmp_path: Path,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        """Empty chain → diagnostic NoPassphraseError, same shape as the non-source variant."""
        from terok_sandbox.vault.store import encryption as enc

        monkeypatch.setattr(enc, "load_passphrase_from_keyring", lambda: None)
        with pytest.raises(NoPassphraseError, match="no SQLCipher passphrase"):
            open_credential_db_with_source(tmp_path / "src.db")


class TestOpenSqlcipherViaChain:
    """The helper that consolidates resolve+raise+open in one call."""

    def test_opens_when_file_tier_has_passphrase(
        self,
        tmp_path: Path,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        """Session-unlock file → connection opens, no error raised."""
        from terok_sandbox.vault.store import encryption as enc

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
        from terok_sandbox.vault.store import encryption as enc

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
        """Empty chain → diagnostic NoPassphraseError naming the DB path."""
        from terok_sandbox.vault.store import encryption as enc

        monkeypatch.setattr(enc, "load_passphrase_from_keyring", lambda: None)
        with pytest.raises(NoPassphraseError, match="no SQLCipher passphrase"):
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
        _patch_dev_tty(monkeypatch)
        _scripted_tty_prompt(monkeypatch, "")
        pw = prompt_passphrase(confirm=True)
        assert len(pw) >= 40
        assert all(c.isalnum() or c in "-_" for c in pw)

    def test_generated_passphrase_announced_to_controlling_tty(
        self, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
    ) -> None:
        """Auto-mint at setup writes to ``/dev/tty`` so a redirected stdout can't capture it.

        Earlier versions printed to stdout on the theory that the
        operator was reading the terminal.  Aisle review (CWE-532)
        flagged that stdout is exactly the surface
        ``terok-sandbox setup > install.log`` (CI pipelines, Ansible,
        cloud-init) does capture, so we route via the controlling
        TTY now.  ``/dev/tty`` reaches the operator's screen even
        through a stdout redirect; missing TTY (fully automated
        install) raises ``SystemExit`` rather than silently dropping
        the recovery key.
        """
        tty_text = _patch_dev_tty(monkeypatch)
        _scripted_tty_prompt(monkeypatch, "")
        pw = prompt_passphrase(confirm=True)
        assert pw in tty_text.value
        assert "Write this down" in tty_text.value
        # Stdout + stderr stay free of the secret — that's the whole
        # point of the /dev/tty routing.
        capture = capsys.readouterr()
        assert pw not in capture.out
        assert pw not in capture.err

    def test_generated_passphrase_refused_without_controlling_tty(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """No ``/dev/tty`` (CI, fully detached process) raises rather than dropping the value silently."""
        from pathlib import Path as _Path

        real_open = _Path.open

        def _selective_open(self, *args, **kwargs):
            if str(self) == "/dev/tty":
                raise OSError("no such device or address")
            return real_open(self, *args, **kwargs)

        monkeypatch.setattr(_Path, "open", _selective_open)
        _scripted_tty_prompt(monkeypatch, "")
        with pytest.raises(SystemExit) as exc:
            prompt_passphrase(confirm=True)
        assert "no controlling TTY" in str(exc.value)


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
        from terok_sandbox.vault.store import encryption as enc

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
        first, second = generate_passphrase(), generate_passphrase()
        assert first != second


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
        _patch_dev_tty(monkeypatch)
        _scripted_tty_prompt(monkeypatch, "")
        pw, source, _ = _provision_passphrase(cfg, mode="session-file")
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
        from terok_sandbox.vault.store import encryption as enc

        monkeypatch.setattr(enc, "load_passphrase_from_file", load_passphrase_from_file)
        cfg = _make_cfg(tmp_path)
        cfg.vault_passphrase_file.parent.mkdir(parents=True, exist_ok=True, mode=0o700)
        cfg.vault_passphrase_file.write_text(_PASSPHRASE + "\n")
        pw, source, _ = _provision_passphrase(cfg, mode="session-file")
        assert pw == _PASSPHRASE
        assert source == "session-file"

    def test_keyring_mode_uses_existing_keyring_entry(
        self,
        tmp_path: Path,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        """Keyring mode returns the stored entry verbatim."""
        from terok_sandbox.commands import _provision_passphrase
        from terok_sandbox.vault.store import encryption as enc

        monkeypatch.setattr(enc, "load_passphrase_from_keyring", lambda: _PASSPHRASE)
        pw, source, _ = _provision_passphrase(_make_cfg(tmp_path), mode="keyring")
        assert pw == _PASSPHRASE
        assert source == "keyring"

    def test_keyring_mode_generates_when_keyring_empty(
        self,
        tmp_path: Path,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        """Empty keyring + working backend → generate + store, source 'keyring'."""
        from terok_sandbox.commands import _provision_passphrase
        from terok_sandbox.vault.store import encryption as enc

        stored: dict[str, str] = {}
        _patch_dev_tty(monkeypatch)
        monkeypatch.setattr(enc, "load_passphrase_from_keyring", lambda: None)
        monkeypatch.setattr(
            enc, "store_passphrase_in_keyring", lambda pw: stored.__setitem__("pw", pw) or True
        )
        pw, source, _ = _provision_passphrase(_make_cfg(tmp_path), mode="keyring")
        assert source == "keyring"
        assert pw == stored["pw"]

    def test_keyring_mode_raises_when_backend_denies(
        self,
        tmp_path: Path,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        """No backend / user-denied → RuntimeError with actionable suggestion."""
        from terok_sandbox.commands import _provision_passphrase
        from terok_sandbox.vault.store import encryption as enc

        monkeypatch.setattr(enc, "load_passphrase_from_keyring", lambda: None)
        monkeypatch.setattr(enc, "store_passphrase_in_keyring", lambda _pw: False)
        with pytest.raises(RuntimeError, match="different storage mode"):
            _provision_passphrase(_make_cfg(tmp_path), mode="keyring")

    def test_config_mode_uses_existing_config_passphrase(self, tmp_path: Path) -> None:
        """Config mode honours an existing credentials.passphrase value."""
        from terok_sandbox.commands import _provision_passphrase

        pw, source, _ = _provision_passphrase(
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
        pw, source, _ = _provision_passphrase(_make_cfg(tmp_path), mode="config")
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
        """Setup with no existing DB and explicit ``[s]`` writes the session-unlock file."""
        from terok_sandbox.commands import _handle_credentials_encrypt_db

        cfg = _make_cfg(tmp_path)
        assert not cfg.db_path.exists()
        assert not cfg.vault_passphrase_file.exists()
        _disable_systemd_creds(monkeypatch)
        _patch_dev_tty(monkeypatch)
        # Pick ``[s]`` explicitly: the chooser default is now keyring, so
        # an empty answer would take the keyring tier instead of session.
        monkeypatch.setattr("sys.stdin.isatty", lambda: True)
        responses = iter(["s\n"])
        monkeypatch.setattr("sys.stdin.readline", lambda: next(responses))
        monkeypatch.setattr("prompt_toolkit.prompt", lambda *_a, **_kw: "")
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

        _disable_systemd_creds(monkeypatch)
        _patch_dev_tty(monkeypatch)
        # Pick ``[s]`` so the test still verifies the session-file path.
        monkeypatch.setattr("sys.stdin.isatty", lambda: True)
        responses = iter(["s\n"])
        monkeypatch.setattr("sys.stdin.readline", lambda: next(responses))
        monkeypatch.setattr("prompt_toolkit.prompt", lambda *_a, **_kw: "")
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
        from terok_sandbox.vault.store import encryption as enc

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


class TestAnnounceGeneratedPassphrase:
    """``_announce_generated_passphrase`` writes to TTY and (optionally) stdout.

    Pins the behaviour that ``--echo-passphrase`` (``echo_to_stdout=True``)
    makes the ``/dev/tty`` write *best-effort* — a CI run without a
    controlling TTY must still get the value on stdout instead of
    crashing in [`_write_to_controlling_tty`][terok_sandbox.vault.store.encryption._write_to_controlling_tty].
    """

    def test_tty_required_when_no_echo(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """Without ``echo_to_stdout``, an unreachable /dev/tty must hard-fail."""
        from pathlib import Path as _Path

        from terok_sandbox.commands.credentials import _announce_generated_passphrase

        real_open = _Path.open

        def _no_tty(self, *args, **kwargs):
            if str(self) == "/dev/tty":
                raise OSError("no /dev/tty in test")
            return real_open(self, *args, **kwargs)

        monkeypatch.setattr(_Path, "open", _no_tty)
        with pytest.raises(SystemExit, match="no controlling TTY"):
            _announce_generated_passphrase(_PASSPHRASE)

    def test_echo_to_stdout_degrades_tty_to_best_effort(
        self,
        monkeypatch: pytest.MonkeyPatch,
        capsys: pytest.CaptureFixture[str],
    ) -> None:
        """``echo_to_stdout=True`` makes the TTY write best-effort; stdout still gets it."""
        from pathlib import Path as _Path

        from terok_sandbox.commands.credentials import _announce_generated_passphrase

        real_open = _Path.open

        def _no_tty(self, *args, **kwargs):
            if str(self) == "/dev/tty":
                raise OSError("no /dev/tty in test")
            return real_open(self, *args, **kwargs)

        monkeypatch.setattr(_Path, "open", _no_tty)
        # Must not raise — the documented escape hatch.
        _announce_generated_passphrase(_PASSPHRASE, echo_to_stdout=True)
        assert _PASSPHRASE in capsys.readouterr().out


class TestPostSetupRecoveryHint:
    """End-of-setup reminder that surfaces only when the marker is absent."""

    def test_prints_reminder_when_marker_missing(
        self,
        tmp_path: Path,
        capsys: pytest.CaptureFixture[str],
    ) -> None:
        """Fresh install / unacked re-run → trailing block names both verbs."""
        from terok_sandbox.commands.credentials import _post_setup_recovery_hint

        cfg = _make_cfg(tmp_path)
        _post_setup_recovery_hint(cfg)
        out = capsys.readouterr().out
        assert "Recovery key" in out
        # Both remediation verbs surface, no "(CI / TUI flow)" parenthetical.
        assert "terok vault passphrase reveal" in out
        assert "terok vault passphrase acknowledge" in out
        assert "CI / TUI flow" not in out

    def test_silent_when_already_acked(
        self,
        tmp_path: Path,
        capsys: pytest.CaptureFixture[str],
    ) -> None:
        """Re-run on an acked host stays quiet — no need to nudge again."""
        from terok_sandbox.commands.credentials import _post_setup_recovery_hint
        from terok_sandbox.vault.store.recovery import acknowledge

        cfg = _make_cfg(tmp_path)
        acknowledge(cfg.vault_recovery_marker_file)
        _post_setup_recovery_hint(cfg)
        assert capsys.readouterr().out == ""


class TestMaybeAcknowledgeRecovery:
    """The interactive SAVED prompt that fires after an auto-mint."""

    def test_saved_response_writes_marker(
        self,
        tmp_path: Path,
        monkeypatch: pytest.MonkeyPatch,
        capsys: pytest.CaptureFixture[str],
    ) -> None:
        """Operator types SAVED → marker file lands, success message prints."""
        from terok_sandbox.commands.credentials import _maybe_acknowledge_recovery
        from terok_sandbox.vault.store.recovery import acknowledged

        cfg = _make_cfg(tmp_path)
        # ``_read_from_controlling_tty`` returns the next scripted line.
        _patch_dev_tty(monkeypatch, responses=("SAVED",))
        _maybe_acknowledge_recovery(cfg, echo_to_stdout=False)
        assert acknowledged(cfg.vault_recovery_marker_file)
        assert "marked as saved" in capsys.readouterr().out


class TestAskPassphraseMode:
    """Setup chooser refuses non-TTY without an explicit --passphrase-tier."""

    def test_non_tty_refuses_with_actionable_hint(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """``terok setup < /dev/null`` without ``--passphrase-tier`` must fail closed.

        Earlier releases silently picked ``session-file`` here — fine
        for `terok setup` running under the TUI's no-TTY worker, broken
        for the operator who lost the auto-generated key on the next
        reboot.  The non-interactive path now hard-fails with a hint
        pointing at ``--passphrase-tier``.
        """
        from terok_sandbox.commands import _ask_passphrase_mode

        monkeypatch.setattr("sys.stdin.isatty", lambda: False)
        with pytest.raises(SystemExit, match="--passphrase-tier"):
            _ask_passphrase_mode()

    def test_session_keyring_pass_through(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """``s`` / ``k`` choices skip the config-tier confirmation entirely."""
        from terok_sandbox.commands import _ask_passphrase_mode

        monkeypatch.setattr("sys.stdin.isatty", lambda: True)
        for letter, expected in (("s", "session-file"), ("k", "keyring")):
            monkeypatch.setattr("sys.stdin.readline", lambda _letter=letter: _letter + "\n")
            assert _ask_passphrase_mode() == expected

    def test_config_choice_requires_yes(
        self, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
    ) -> None:
        """``c`` then ``yes`` accepts the plaintext-on-disk trust boundary."""
        from terok_sandbox.commands import _ask_passphrase_mode

        responses = iter(["c\n", "yes\n"])
        monkeypatch.setattr("sys.stdin.isatty", lambda: True)
        monkeypatch.setattr("sys.stdin.readline", lambda: next(responses))
        assert _ask_passphrase_mode() == "config"
        # The stern explanation must surface so the operator sees what
        # they're confirming.
        out = capsys.readouterr().out
        assert "plaintext" in out
        assert "trust boundary" in out or "filesystem" in out

    def test_config_choice_declined_reprompts(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """``c`` then anything-but-``yes`` re-prompts; eventual ``s`` lands."""
        from terok_sandbox.commands import _ask_passphrase_mode

        responses = iter(["c\n", "no\n", "s\n"])
        monkeypatch.setattr("sys.stdin.isatty", lambda: True)
        monkeypatch.setattr("sys.stdin.readline", lambda: next(responses))
        assert _ask_passphrase_mode() == "session-file"

    def test_empty_choice_defaults_to_keyring(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """Pressing Enter at the chooser takes the recommended default (keyring)."""
        from terok_sandbox.commands import _ask_passphrase_mode

        monkeypatch.setattr("sys.stdin.isatty", lambda: True)
        monkeypatch.setattr("sys.stdin.readline", lambda: "\n")
        assert _ask_passphrase_mode() == "keyring"

    def test_chooser_prompt_lists_keyring_first_and_hints_systemd_creds(
        self, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
    ) -> None:
        """Prompt names keyring as recommended and points operators at the systemd upgrade."""
        from terok_sandbox.commands import _ask_passphrase_mode

        monkeypatch.setattr("sys.stdin.isatty", lambda: True)
        monkeypatch.setattr("sys.stdin.readline", lambda: "\n")
        _ask_passphrase_mode()
        out = capsys.readouterr().out
        assert "[k] keyring" in out
        assert "recommended" in out.lower()
        assert "systemd" in out and "≥ 257" in out


class TestAutoSystemdCredsBranch:
    """When systemd-creds is available, ``_handle_credentials_encrypt_db`` skips the chooser.

    The strongest available tier is unambiguous on hosts where the
    Varlink service is up — asking would only slow the operator down.
    """

    def test_uses_systemd_creds_without_chooser(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
    ) -> None:
        """Fresh install on a TPM-capable host seals; announce lands on ``/dev/tty``."""
        from terok_sandbox.commands import _handle_credentials_encrypt_db

        cfg = _make_cfg(tmp_path)
        seal_calls: list[tuple] = []

        def _fake_seal(passphrase: str, path: Path, *, key_mode: str = "auto") -> None:
            seal_calls.append((passphrase, path, key_mode))

        tty_text = _patch_dev_tty(monkeypatch)
        monkeypatch.setattr("terok_sandbox.vault.store.systemd_creds.is_available", lambda: True)
        monkeypatch.setattr("terok_sandbox.vault.store.systemd_creds.seal", _fake_seal)
        # If the chooser fired we'd hang on readline; failing fast here
        # tells us the auto-branch short-circuited correctly.
        monkeypatch.setattr(
            "sys.stdin.readline",
            lambda: pytest.fail("chooser should not be invoked on systemd-creds-available hosts"),
        )

        _handle_credentials_encrypt_db(cfg=cfg)

        assert len(seal_calls) == 1
        sealed_passphrase, sealed_path, key_mode = seal_calls[0]
        # Passphrase was minted (non-empty) and sealed under the systemd
        # credential name with ``--with-key=auto`` (TPM2 + host on
        # equipped hosts, host alone otherwise).
        assert sealed_passphrase
        assert sealed_path == cfg.vault_systemd_creds_file
        assert key_mode == "auto"
        # Token-mint framing reaches the controlling TTY (not stdout)
        # so a redirected install — ``setup > install.log``, CI — can't
        # capture the recovery key.
        assert sealed_passphrase in tty_text.value
        assert "Write this down" in tty_text.value
        # Source label still lands on stdout so ``vault status`` readers
        # see ``passphrase source: systemd-creds`` in their pipe.
        out = capsys.readouterr()
        assert "systemd-creds" in out.out
        # And the secret stays OFF stdout/stderr.
        assert sealed_passphrase not in out.out
        assert sealed_passphrase not in out.err

    def test_echo_passphrase_also_writes_to_stdout(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
    ) -> None:
        """``echo_passphrase=True`` puts the recovery key on stdout for non-interactive bootstraps."""
        from terok_sandbox.commands import _handle_credentials_encrypt_db

        cfg = _make_cfg(tmp_path)
        sealed: list[str] = []

        def _fake_seal(passphrase: str, _path: Path, *, key_mode: str = "auto") -> None:  # noqa: ARG001
            sealed.append(passphrase)

        _patch_dev_tty(monkeypatch)
        monkeypatch.setattr("terok_sandbox.vault.store.systemd_creds.is_available", lambda: True)
        monkeypatch.setattr("terok_sandbox.vault.store.systemd_creds.seal", _fake_seal)

        _handle_credentials_encrypt_db(cfg=cfg, echo_passphrase=True)

        assert sealed
        out = capsys.readouterr().out
        # The opt-in surfaces the value to stdout so an Ansible / CI driver
        # can capture it; without the flag the secret would only reach
        # /dev/tty and a no-TTY run would drop it silently.
        assert sealed[0] in out


class TestProvisionSessionPassphrase:
    """``provision_session_passphrase`` — the validated single writer of the session tier."""

    @staticmethod
    def _make_encrypted_db(cfg) -> None:
        """Create a minimal SQLCipher DB at ``cfg.db_path`` keyed with ``_PASSPHRASE``."""
        cfg.db_path.parent.mkdir(parents=True, exist_ok=True)
        conn = open_sqlcipher(cfg.db_path, _PASSPHRASE)
        conn.execute("CREATE TABLE t (x INTEGER)")
        conn.commit()
        conn.close()

    def test_validates_against_existing_db_and_writes(self, tmp_path: Path) -> None:
        from terok_sandbox.commands import provision_session_passphrase

        cfg = _make_cfg(tmp_path)
        self._make_encrypted_db(cfg)
        result = provision_session_passphrase(cfg, _PASSPHRASE)
        assert result.written is True and result.validated is True
        assert cfg.vault_passphrase_file.read_text().rstrip("\n") == _PASSPHRASE

    def test_rejects_wrong_passphrase_and_writes_nothing(self, tmp_path: Path) -> None:
        from terok_sandbox.commands import provision_session_passphrase
        from terok_sandbox.vault.store.encryption import WrongPassphraseError

        cfg = _make_cfg(tmp_path)
        self._make_encrypted_db(cfg)
        with pytest.raises(WrongPassphraseError):
            provision_session_passphrase(cfg, "wrong-guess")
        # The whole point: a rejected value must never land on the
        # highest-priority tier where it would shadow working state.
        assert not cfg.vault_passphrase_file.exists()

    def test_skips_validation_when_db_missing(self, tmp_path: Path) -> None:
        """No DB yet → nothing to validate against; the value becomes the key on first use."""
        from terok_sandbox.commands import provision_session_passphrase

        cfg = _make_cfg(tmp_path)
        result = provision_session_passphrase(cfg, "brand-new-key")
        assert result.written is True and result.validated is False
        assert cfg.vault_passphrase_file.exists()
        # Validation must not create the DB as a side effect.
        assert not cfg.db_path.exists()

    def test_skips_validation_on_legacy_plaintext_db(self, tmp_path: Path) -> None:
        """A plaintext DB has no key to validate against — unlock-then-migrate is the documented flow."""
        from terok_sandbox.commands import provision_session_passphrase

        cfg = _make_cfg(tmp_path)
        cfg.db_path.parent.mkdir(parents=True, exist_ok=True)
        conn = sqlite3.connect(str(cfg.db_path))
        conn.execute("CREATE TABLE t (x INTEGER)")
        conn.commit()
        conn.close()
        result = provision_session_passphrase(cfg, "future-key")
        assert result.written is True and result.validated is False
        assert cfg.vault_passphrase_file.exists()

    def test_refuses_to_shadow_a_durable_tier(self, tmp_path: Path) -> None:
        """A durable tier already resolving → refuse the write, name the tier, no validation."""
        from terok_sandbox.commands import provision_session_passphrase

        # ``config`` plaintext tier is durable and present.
        cfg = _make_cfg(tmp_path, passphrase="durable-key")
        self._make_encrypted_db(cfg)  # exists, but must not be touched
        result = provision_session_passphrase(cfg, "anything")
        assert result.written is False
        assert result.shadowed_durable == "config"
        assert not cfg.vault_passphrase_file.exists()

    def test_force_overrides_the_shadow_guard(self, tmp_path: Path) -> None:
        """``force=True`` writes even over a durable tier (re-key / deliberate override)."""
        from terok_sandbox.commands import provision_session_passphrase

        cfg = _make_cfg(tmp_path, passphrase=_PASSPHRASE)
        self._make_encrypted_db(cfg)  # keyed _PASSPHRASE, so validation passes
        result = provision_session_passphrase(cfg, _PASSPHRASE, force=True)
        assert result.written is True
        assert cfg.vault_passphrase_file.read_text().rstrip("\n") == _PASSPHRASE


class TestVaultUnlockLock:
    """``terok-sandbox vault unlock`` / ``vault lock`` CLI handlers."""

    def test_unlock_rejects_wrong_passphrase(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """A typed value that doesn't open the DB exits with an error; nothing is written."""

        from terok_sandbox.commands import _handle_vault_unlock

        cfg = _make_cfg(tmp_path)
        TestProvisionSessionPassphrase._make_encrypted_db(cfg)
        _scripted_tty_prompt(monkeypatch, "wrong-guess")
        with pytest.raises(SystemExit, match="does not open"):
            _handle_vault_unlock(cfg=cfg)
        assert not cfg.vault_passphrase_file.exists()

    def test_unlock_surfaces_plaintext_db_as_clean_exit(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """A legacy plaintext DB surfaces its migration message as a clean CLI exit.

        Defensive seam: ``provision_session_passphrase`` gates on
        ``is_plaintext_sqlite`` before validating, but ``CredentialDB``
        runs its own plaintext probe — if the two ever disagree the
        handler must still exit with the actionable message, not a
        traceback.
        """
        from unittest.mock import MagicMock

        from terok_sandbox.commands import _handle_vault_unlock
        from terok_sandbox.vault.store.db import PlaintextDBFoundError

        cfg = _make_cfg(tmp_path)
        _scripted_tty_prompt(monkeypatch, "whatever")
        monkeypatch.setattr(
            "terok_sandbox.commands.vault.provision_session_passphrase",
            MagicMock(side_effect=PlaintextDBFoundError("legacy plaintext sqlite DB — migrate")),
        )
        with pytest.raises(SystemExit, match="legacy plaintext"):
            _handle_vault_unlock(cfg=cfg)

    def test_unlock_reports_validation(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
    ) -> None:
        """A value that opens the DB is written and reported as verified."""

        from terok_sandbox.commands import _handle_vault_unlock

        cfg = _make_cfg(tmp_path)
        TestProvisionSessionPassphrase._make_encrypted_db(cfg)
        _scripted_tty_prompt(monkeypatch, _PASSPHRASE)
        _handle_vault_unlock(cfg=cfg)
        assert cfg.vault_passphrase_file.read_text().rstrip("\n") == _PASSPHRASE
        assert "verified: the value opens the credentials DB" in capsys.readouterr().out

    def test_unlock_writes_passphrase_and_restarts_running_daemon(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Happy path: prompt, write tmpfs file, bounce the daemon."""

        from terok_sandbox.commands import _handle_vault_unlock

        cfg = _make_cfg(tmp_path)
        _scripted_tty_prompt(monkeypatch, "freshly-typed")
        _handle_vault_unlock(cfg=cfg)

        assert cfg.vault_passphrase_file.read_text().rstrip("\n") == "freshly-typed"
        assert (cfg.vault_passphrase_file.stat().st_mode & 0o777) == 0o600

    def test_unlock_skips_restart_when_daemon_not_running(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """No daemon running → file is written, restart is not attempted, message printed."""

        from terok_sandbox.commands import _handle_vault_unlock

        cfg = _make_cfg(tmp_path)
        _scripted_tty_prompt(monkeypatch, "freshly-typed")
        _handle_vault_unlock(cfg=cfg)

    def test_unlock_skips_when_durable_tier_present(
        self, tmp_path: Path, capsys: pytest.CaptureFixture[str]
    ) -> None:
        """``unlock`` won't shadow a durable tier — a sealed credential blocks the session write."""

        from terok_sandbox.commands import _handle_vault_unlock

        cfg = _make_cfg(tmp_path)
        cfg.vault_systemd_creds_file.parent.mkdir(parents=True, exist_ok=True)
        cfg.vault_systemd_creds_file.write_bytes(b"sealed-blob")
        _handle_vault_unlock(cfg=cfg)

        assert not cfg.vault_passphrase_file.exists()
        assert "already auto-unlocks via systemd-creds" in capsys.readouterr().out

    def test_unlock_force_overrides_durable_shadow(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """``--force`` writes the session file even when a durable tier resolves."""

        from terok_sandbox.commands import _handle_vault_unlock

        cfg = _make_cfg(tmp_path)
        cfg.vault_systemd_creds_file.parent.mkdir(parents=True, exist_ok=True)
        cfg.vault_systemd_creds_file.write_bytes(b"sealed-blob")
        _scripted_tty_prompt(monkeypatch, "freshly-typed")
        _handle_vault_unlock(cfg=cfg, force=True)

        assert cfg.vault_passphrase_file.read_text().rstrip("\n") == "freshly-typed"

    def test_unlock_default_cfg_branch(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
    ) -> None:
        """``cfg=None`` constructs a default config; a durable tier short-circuits the prompt."""
        from terok_sandbox.commands import _handle_vault_unlock

        cfg = _make_cfg(tmp_path)
        cfg.vault_systemd_creds_file.parent.mkdir(parents=True, exist_ok=True)
        cfg.vault_systemd_creds_file.write_bytes(b"sealed-blob")
        monkeypatch.setattr("terok_sandbox.config.SandboxConfig", lambda: cfg)
        _handle_vault_unlock()  # cfg omitted → default-construction branch
        assert "already auto-unlocks" in capsys.readouterr().out

    # ── lock = clear every stored copy (absorbs the old `destroy`) ──

    def test_lock_aborts_when_unacknowledged(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """An unconfirmed vault refuses to lock without a typed SAVED — nothing is purged."""

        from terok_sandbox.commands import _handle_vault_lock

        cfg = _make_cfg(tmp_path)
        cfg.vault_passphrase_file.parent.mkdir(parents=True, exist_ok=True)
        cfg.vault_passphrase_file.write_text("stale\n")
        # Headless: the confirm channel yields nothing → fail closed.
        monkeypatch.setattr(
            "terok_sandbox.vault.store.encryption._read_from_controlling_tty",
            lambda _prompt: None,
        )

        with pytest.raises(SystemExit, match="lock aborted"):
            _handle_vault_lock(cfg=cfg)
        assert cfg.vault_passphrase_file.exists()  # untouched

    def test_lock_confirm_proceeds_on_saved(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Typing SAVED on an unconfirmed vault lets the lock through."""

        from terok_sandbox.commands import _handle_vault_lock

        cfg = _make_cfg(tmp_path)
        cfg.vault_passphrase_file.parent.mkdir(parents=True, exist_ok=True)
        cfg.vault_passphrase_file.write_text("stale\n")
        monkeypatch.setattr(
            "terok_sandbox.vault.store.encryption._read_from_controlling_tty",
            lambda _prompt: "SAVED",
        )
        _handle_vault_lock(cfg=cfg)

        assert not cfg.vault_passphrase_file.exists()

    def test_lock_force_skips_confirm(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """``--force`` bypasses the confirmation even when unacknowledged."""

        from terok_sandbox.commands import _handle_vault_lock

        cfg = _make_cfg(tmp_path)
        cfg.vault_passphrase_file.parent.mkdir(parents=True, exist_ok=True)
        cfg.vault_passphrase_file.write_text("stale\n")
        # Confirm channel must never be consulted.
        monkeypatch.setattr(
            "terok_sandbox.vault.store.encryption._read_from_controlling_tty",
            lambda _prompt: (_ for _ in ()).throw(AssertionError("should not prompt")),
        )
        _handle_vault_lock(cfg=cfg, force=True)

        assert not cfg.vault_passphrase_file.exists()

    def test_lock_clears_session_file_when_acknowledged(self, tmp_path: Path) -> None:
        """An acknowledged vault locks without prompting — session file removed."""

        from terok_sandbox.commands import _handle_vault_lock

        cfg = _make_cfg(tmp_path)
        cfg.vault_passphrase_file.parent.mkdir(parents=True, exist_ok=True)
        cfg.vault_passphrase_file.write_text("stale\n")
        _ack_recovery(cfg)
        _handle_vault_lock(cfg=cfg)

        assert not cfg.vault_passphrase_file.exists()

    def test_lock_is_idempotent_when_already_locked(self, tmp_path: Path) -> None:
        """Already-locked state (nothing stored) is a clean no-op."""

        from terok_sandbox.commands import _handle_vault_lock

        cfg = _make_cfg(tmp_path)
        _ack_recovery(cfg)
        _handle_vault_lock(cfg=cfg)  # must not raise

    def test_lock_clears_keyring_and_config(
        self,
        tmp_path: Path,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        """``lock`` removes the keyring entry and clears config.yml."""

        from terok_sandbox import config as _config
        from terok_sandbox.commands import _handle_vault_lock

        cfg = _make_cfg(tmp_path, use_keyring=True, passphrase="from-config")
        _ack_recovery(cfg)
        forget_calls: dict[str, int] = {"n": 0}

        def _forget() -> bool:
            forget_calls["n"] += 1
            return True

        monkeypatch.setattr(
            "terok_sandbox.vault.store.encryption.forget_passphrase_in_keyring", _forget
        )

        user_config = tmp_path / "config.yml"
        user_config.write_text("credentials:\n  use_keyring: true\n  passphrase: from-config\n")
        monkeypatch.setattr(
            "terok_sandbox.paths.config_file_paths", lambda: [("user", user_config)]
        )
        _config._credentials_section.cache_clear()

        _handle_vault_lock(cfg=cfg)

        assert forget_calls["n"] == 1
        assert "passphrase: from-config" not in user_config.read_text()

    def test_lock_clears_passphrase_command(
        self,
        tmp_path: Path,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        """``lock`` removes ``credentials.passphrase_command`` so the next start can't auto-resolve via the helper."""

        from terok_sandbox import config as _config
        from terok_sandbox.commands import _handle_vault_lock

        cfg = _make_cfg(tmp_path, passphrase_command="pass show terok-sandbox/vault")
        _ack_recovery(cfg)
        user_config = tmp_path / "config.yml"
        user_config.write_text(
            "credentials:\n  passphrase_command: pass show terok-sandbox/vault\n"
        )
        monkeypatch.setattr(
            "terok_sandbox.paths.config_file_paths", lambda: [("user", user_config)]
        )
        _config._credentials_section.cache_clear()

        _handle_vault_lock(cfg=cfg)

        # Key may remain with a null value depending on YAML serializer behaviour;
        # the recipe string itself must be gone so the next start can't auto-resolve.
        assert "pass show terok-sandbox/vault" not in user_config.read_text()

    def test_lock_treats_absent_keyring_entry_as_idempotent(
        self,
        tmp_path: Path,
        monkeypatch: pytest.MonkeyPatch,
        capsys: pytest.CaptureFixture[str],
    ) -> None:
        """Locking a configured-but-empty keyring is success (not a hard failure)."""

        from terok_sandbox.commands import _handle_vault_lock

        cfg = _make_cfg(tmp_path, use_keyring=True)
        _ack_recovery(cfg)
        # Helper returns False on missing entry (keyring.delete_password raises);
        # the readback then confirms the entry really is absent.
        monkeypatch.setattr(
            "terok_sandbox.vault.store.encryption.forget_passphrase_in_keyring",
            lambda: False,
        )
        monkeypatch.setattr(
            "terok_sandbox.vault.store.encryption.load_passphrase_from_keyring",
            lambda: None,
        )

        _handle_vault_lock(cfg=cfg)  # must not raise

        assert "already absent" in capsys.readouterr().out

    def test_lock_raises_when_keyring_backend_rejects_delete(
        self,
        tmp_path: Path,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        """A residual entry after a failed delete means lock couldn't honour its contract."""

        from terok_sandbox.commands import _handle_vault_lock

        cfg = _make_cfg(tmp_path, use_keyring=True)
        _ack_recovery(cfg)
        # Helper claimed failure AND the entry is still there: real backend rejection.
        monkeypatch.setattr(
            "terok_sandbox.vault.store.encryption.forget_passphrase_in_keyring",
            lambda: False,
        )
        monkeypatch.setattr(
            "terok_sandbox.vault.store.encryption.load_passphrase_from_keyring",
            lambda: "still-there",
        )

        with pytest.raises(SystemExit, match="failed to clear keyring entry"):
            _handle_vault_lock(cfg=cfg)

    def test_lock_removes_sealed_credential(
        self,
        tmp_path: Path,
    ) -> None:
        """``lock`` deletes the systemd-creds blob so the next daemon start can't auto-unlock from it."""

        from terok_sandbox.commands import _handle_vault_lock

        cfg = _make_cfg(tmp_path)
        _ack_recovery(cfg)
        cfg.vault_systemd_creds_file.parent.mkdir(parents=True, exist_ok=True)
        cfg.vault_systemd_creds_file.write_bytes(b"sealed-blob")
        _handle_vault_lock(cfg=cfg)

        assert not cfg.vault_systemd_creds_file.exists()

    def test_lock_surfaces_unlink_failure_as_systemexit(
        self,
        tmp_path: Path,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        """A permissions / IO error on the sealed credential bubbles to SystemExit, not a traceback."""

        from terok_sandbox.commands import _handle_vault_lock

        cfg = _make_cfg(tmp_path)
        _ack_recovery(cfg)
        cfg.vault_systemd_creds_file.parent.mkdir(parents=True, exist_ok=True)
        cfg.vault_systemd_creds_file.write_bytes(b"sealed-blob")
        # Target only the sealed-credential unlink so any other Path.unlink
        # call inside the handler keeps its real behaviour and the test stays
        # robust against future changes to ``purge_passphrase_tiers``.
        original_unlink = Path.unlink

        def _conditional_unlink(self: Path, missing_ok: bool = False) -> None:
            if self == cfg.vault_systemd_creds_file:
                raise OSError("permission denied")
            return original_unlink(self, missing_ok=missing_ok)

        monkeypatch.setattr("pathlib.Path.unlink", _conditional_unlink)

        with pytest.raises(SystemExit, match="failed to remove sealed credential"):
            _handle_vault_lock(cfg=cfg)

    def test_lock_default_cfg_branch(self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
        """``cfg=None`` constructs a default config; ``--force`` carries it through the purge."""
        from terok_sandbox.commands import _handle_vault_lock

        cfg = _make_cfg(tmp_path)  # use_keyring=False → purge skips every tier cleanly
        monkeypatch.setattr("terok_sandbox.config.SandboxConfig", lambda: cfg)
        _handle_vault_lock(force=True)  # cfg omitted → default-construction branch


class TestVaultSeal:
    """``terok-sandbox vault seal`` CLI handler.

    The handler is intentionally thin: validate the ``--key`` vocabulary,
    resolve a passphrase from another tier, hand off to
    ``systemd_creds.seal`` with a ``KeyMode`` that maps 1:1 onto
    systemd's own ``--with-key=`` values.  Tests stub ``sc.seal`` so the
    actual subprocess is unit-test-irrelevant.
    """

    @pytest.mark.parametrize(
        ("cli_key", "expected_mode"),
        [
            ("auto", "auto"),
            ("tpm", "tpm2"),
            ("host", "host"),
            ("tpm+host", "host+tpm2"),
        ],
    )
    def test_key_argument_maps_to_systemd_key_mode(
        self,
        tmp_path: Path,
        monkeypatch: pytest.MonkeyPatch,
        cli_key: str,
        expected_mode: str,
    ) -> None:
        """Each ``--key=…`` value funnels through to the systemd ``--with-key=`` vocabulary.

        Notably ``--key=auto`` maps to ``--with-key=auto`` — we hand the
        host-vs-TPM choice to systemd, which picks ``host+tpm2`` on
        TPM-equipped hosts (defense in depth) rather than the weaker
        TPM-only default the wrapper used to imply.
        """
        from terok_sandbox.commands import handle_vault_seal

        cfg = self._seed_cfg(tmp_path)
        _, seal = self._stub_seal_ready(monkeypatch)

        handle_vault_seal(cfg=cfg, key=cli_key)

        seal.assert_called_once_with(
            "current-pw", cfg.vault_systemd_creds_file, key_mode=expected_mode
        )

    def test_seal_propagates_systemd_creds_failure_as_systemexit(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """A ``--key=tpm`` request on a TPM-less host bubbles up as a SystemExit.

        The wrapper itself can't know whether the host has a TPM — systemd-creds
        is the authority — so the handler trusts ``seal()`` to fail loudly and
        translates the resulting RuntimeError into a CLI-friendly SystemExit.
        """
        from terok_sandbox.commands import handle_vault_seal

        cfg = self._seed_cfg(tmp_path)
        _, seal = self._stub_seal_ready(monkeypatch)
        seal.side_effect = RuntimeError("systemd-creds encrypt failed (exit 1): no TPM2 device")

        with pytest.raises(SystemExit, match="no TPM2 device"):
            handle_vault_seal(cfg=cfg, key="tpm")

    def test_seal_unknown_key_value_rejected(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """A typo'd --key value fails loudly rather than silently picking a default."""
        from terok_sandbox.commands import handle_vault_seal
        from terok_sandbox.vault.store import systemd_creds as sc

        monkeypatch.setattr(sc, "is_available", lambda: True)

        with pytest.raises(SystemExit, match="unknown --key value"):
            handle_vault_seal(cfg=_make_cfg(tmp_path), key="bogus")

    def test_seal_refuses_when_binary_missing(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """``systemd-creds`` absent or too old → exit with an actionable hint."""
        from terok_sandbox.commands import handle_vault_seal
        from terok_sandbox.vault.store import systemd_creds as sc

        monkeypatch.setattr(sc, "is_available", lambda: False)

        with pytest.raises(SystemExit, match="needs systemd"):
            handle_vault_seal(cfg=_make_cfg(tmp_path))

    def test_seal_refuses_when_no_current_passphrase(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Without an existing tier to seal from, the command fails loudly."""
        from terok_sandbox.commands import handle_vault_seal
        from terok_sandbox.vault.store import systemd_creds as sc

        cfg = _make_cfg(tmp_path)
        monkeypatch.setattr(sc, "is_available", lambda: True)
        monkeypatch.setattr(
            "terok_sandbox.vault.store.encryption.load_passphrase_from_keyring",
            lambda: None,
        )

        with pytest.raises(SystemExit, match="no current passphrase"):
            handle_vault_seal(cfg=cfg)

    def test_seal_converts_broken_tier_to_systemexit(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """A fail-closed resolver error (e.g. broken `passphrase_command`) surfaces as a clean CLI exit, not a traceback."""
        from terok_sandbox.commands import handle_vault_seal
        from terok_sandbox.vault.store import systemd_creds as sc

        cfg = _make_cfg(tmp_path, passphrase_command="/bin/false")
        monkeypatch.setattr(sc, "is_available", lambda: True)
        monkeypatch.setattr(
            "terok_sandbox.vault.store.encryption.load_passphrase_from_keyring",
            lambda: None,
        )

        with pytest.raises(
            SystemExit, match="cannot seal: passphrase_command produced no passphrase"
        ):
            handle_vault_seal(cfg=cfg)

    def test_seal_never_prompts_for_passphrase(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Even with a TTY, seal must not invite a fresh passphrase entry.

        Sealing whatever the operator types would happily encrypt a value
        that *doesn't* open the existing DB; the failure mode lands at
        the next reboot.  The handler routes through
        ``resolve_passphrase(..., prompt_on_tty=False)`` so the prompt
        branch is unreachable from this code path.
        """
        from unittest.mock import MagicMock

        from terok_sandbox.commands import handle_vault_seal
        from terok_sandbox.vault.store import systemd_creds as sc

        cfg = _make_cfg(tmp_path)
        monkeypatch.setattr(sc, "is_available", lambda: True)
        monkeypatch.setattr(
            "terok_sandbox.vault.store.encryption.load_passphrase_from_keyring",
            lambda: None,
        )
        monkeypatch.setattr("sys.stdin.isatty", lambda: True)
        prompt = MagicMock()
        monkeypatch.setattr("prompt_toolkit.prompt", prompt)

        with pytest.raises(SystemExit, match="no current passphrase"):
            handle_vault_seal(cfg=cfg)
        prompt.assert_not_called()

    # ── Lifecycle helpers ──────────────────────────────────────────

    @staticmethod
    def _seed_cfg(tmp_path: Path) -> object:
        """Return a cfg with a session-unlock file populated and recovery acknowledged.

        Recovery is pre-acknowledged so the escrow-before-enable gate
        passes — the tests below exercise the sealing mechanics, not the
        gate (which has its own test).
        """
        cfg = _make_cfg(tmp_path)
        cfg.vault_passphrase_file.parent.mkdir(parents=True, exist_ok=True)
        cfg.vault_passphrase_file.write_text("current-pw\n")
        _ack_recovery(cfg)
        return cfg

    def test_seal_refuses_until_recovery_acknowledged(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Escrow-before-enable: sealing is blocked until the operator confirms a saved copy."""
        from terok_sandbox.commands import handle_vault_seal

        cfg = _make_cfg(tmp_path)
        cfg.vault_passphrase_file.parent.mkdir(parents=True, exist_ok=True)
        cfg.vault_passphrase_file.write_text("current-pw\n")  # passphrase resolvable…
        self._stub_seal_ready(monkeypatch)  # …and systemd-creds available
        # …but recovery is NOT acknowledged.
        with pytest.raises(SystemExit, match="recovery passphrase isn't marked as saved"):
            handle_vault_seal(cfg=cfg)

    @staticmethod
    def _stub_seal_ready(
        monkeypatch: pytest.MonkeyPatch,
    ) -> tuple[object, object]:
        """Stub ``sc.is_available`` true and capture ``sc.seal`` invocations."""
        from unittest.mock import MagicMock

        from terok_sandbox.vault.store import systemd_creds as sc

        monkeypatch.setattr(sc, "is_available", lambda: True)
        monkeypatch.setattr(
            "terok_sandbox.vault.store.encryption.load_passphrase_from_file",
            load_passphrase_from_file,
        )
        seal = MagicMock()
        monkeypatch.setattr(sc, "seal", seal)
        return sc, seal

    def test_seal_default_cfg_branch(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """``cfg=None`` constructs a default config before the systemd-creds availability gate."""
        from terok_sandbox.commands import handle_vault_seal
        from terok_sandbox.vault.store import systemd_creds as sc

        monkeypatch.setattr(sc, "is_available", lambda: False)
        with pytest.raises(SystemExit, match="systemd-creds unavailable"):
            handle_vault_seal()  # cfg omitted → default-construction branch, then the gate

    def test_seal_removes_session_shadow(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """After sealing, the redundant session file is dropped so it can't shadow the seal."""
        from terok_sandbox.commands import handle_vault_seal

        cfg = self._seed_cfg(tmp_path)  # writes a session file + acks recovery
        self._stub_seal_ready(monkeypatch)
        assert cfg.vault_passphrase_file.exists()

        handle_vault_seal(cfg=cfg)

        assert not cfg.vault_passphrase_file.exists()


class TestVaultToKeyring:
    """``terok-sandbox vault passphrase to-keyring`` — relocate the passphrase to the OS keyring."""

    def test_writes_to_keyring_and_flips_config(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """A session-tier passphrase moves into the keyring, config flips on, session file removed."""
        from unittest.mock import MagicMock

        from terok_sandbox import config as _config
        from terok_sandbox.commands import handle_vault_to_keyring

        cfg = _make_cfg(tmp_path)
        cfg.vault_passphrase_file.parent.mkdir(parents=True, exist_ok=True)
        cfg.vault_passphrase_file.write_text("current-pw\n")
        _ack_recovery(cfg)
        # Undo the autouse blank of the session-file tier so this test
        # actually exercises a session → keyring relocation.
        monkeypatch.setattr(
            "terok_sandbox.vault.store.encryption.load_passphrase_from_file",
            load_passphrase_from_file,
        )
        user_config = tmp_path / "config.yml"
        user_config.write_text("credentials: {}\n")
        monkeypatch.setattr(
            "terok_sandbox.paths.config_file_paths", lambda: [("user", user_config)]
        )
        _config._credentials_section.cache_clear()

        store = MagicMock(return_value=True)
        monkeypatch.setattr(
            "terok_sandbox.vault.store.encryption.store_passphrase_in_keyring", store
        )
        handle_vault_to_keyring(cfg=cfg)

        store.assert_called_once_with("current-pw")
        assert not cfg.vault_passphrase_file.exists()
        assert "use_keyring: true" in user_config.read_text()

    def test_noop_when_already_in_keyring(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """If the chain already hits keyring, the verb is idempotent — no write, no restart."""
        from unittest.mock import MagicMock

        from terok_sandbox.commands import handle_vault_to_keyring

        cfg = _make_cfg(tmp_path, use_keyring=True)
        monkeypatch.setattr(
            "terok_sandbox.vault.store.encryption.load_passphrase_from_keyring",
            lambda: "current-pw",
        )
        store = MagicMock()
        monkeypatch.setattr(
            "terok_sandbox.vault.store.encryption.store_passphrase_in_keyring", store
        )

        handle_vault_to_keyring(cfg=cfg)

        store.assert_not_called()

    def test_refuses_when_no_passphrase_resolvable(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """No tier hits and no TTY → fail loudly rather than write nothing silently."""
        from terok_sandbox.commands import handle_vault_to_keyring

        # ``use_keyring=False`` blocks the autouse keyring stub from
        # claiming the chain; file tier is already blanked by autouse.
        cfg = _make_cfg(tmp_path)
        monkeypatch.setattr("sys.stdin.isatty", lambda: False)

        with pytest.raises(SystemExit, match="no current passphrase"):
            handle_vault_to_keyring(cfg=cfg)

    def test_aborts_when_keyring_write_fails(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """A keyring backend rejection leaves the source tier untouched."""
        from unittest.mock import MagicMock

        from terok_sandbox.commands import handle_vault_to_keyring

        cfg = _make_cfg(tmp_path)
        cfg.vault_passphrase_file.parent.mkdir(parents=True, exist_ok=True)
        cfg.vault_passphrase_file.write_text("current-pw\n")
        _ack_recovery(cfg)
        monkeypatch.setattr(
            "terok_sandbox.vault.store.encryption.load_passphrase_from_file",
            load_passphrase_from_file,
        )
        monkeypatch.setattr(
            "terok_sandbox.vault.store.encryption.store_passphrase_in_keyring",
            MagicMock(return_value=False),
        )

        with pytest.raises(SystemExit, match="OS keyring is unreachable"):
            handle_vault_to_keyring(cfg=cfg)
        # Source tier is preserved on failure — no half-done migration.
        assert cfg.vault_passphrase_file.read_text() == "current-pw\n"

    def test_restarts_daemon_when_running(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """A running daemon is stopped + restarted so it picks up the new tier on this boot."""
        from unittest.mock import MagicMock

        from terok_sandbox.commands import handle_vault_to_keyring

        cfg = _make_cfg(tmp_path)
        cfg.vault_passphrase_file.parent.mkdir(parents=True, exist_ok=True)
        cfg.vault_passphrase_file.write_text("current-pw\n")
        _ack_recovery(cfg)
        user_config = tmp_path / "config.yml"
        user_config.write_text("credentials: {}\n")
        monkeypatch.setattr(
            "terok_sandbox.paths.config_file_paths", lambda: [("user", user_config)]
        )
        monkeypatch.setattr(
            "terok_sandbox.vault.store.encryption.load_passphrase_from_file",
            load_passphrase_from_file,
        )
        monkeypatch.setattr(
            "terok_sandbox.vault.store.encryption.store_passphrase_in_keyring",
            MagicMock(return_value=True),
        )
        handle_vault_to_keyring(cfg=cfg)

    def test_propagates_wrong_passphrase_as_systemexit(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """A fail-closed resolver error (e.g. broken sealed credential) surfaces as a CLI-friendly SystemExit."""
        from terok_sandbox.commands import handle_vault_to_keyring
        from terok_sandbox.vault.store.encryption import WrongPassphraseError

        cfg = _make_cfg(tmp_path)

        def _boom(**_kwargs: object) -> tuple[None, None]:
            raise WrongPassphraseError("sealed credential present but could not be unsealed")

        monkeypatch.setattr(
            "terok_sandbox.vault.store.encryption.resolve_passphrase_with_source", _boom
        )

        with pytest.raises(SystemExit, match="cannot move to keyring: sealed credential"):
            handle_vault_to_keyring(cfg=cfg)

    def test_default_cfg_branch(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """``cfg=None`` constructs a default ``SandboxConfig``; the no-passphrase exit path is reached."""
        from terok_sandbox.commands import handle_vault_to_keyring

        # No passphrase resolvable + no TTY ⇒ deterministic SystemExit.
        # The autouse fixtures stub the keyring tier to return "test"
        # and ``credentials_use_keyring`` to True — undo both so the
        # chain genuinely yields nothing and we test the default-cfg
        # construction without falling through to a real config write.
        monkeypatch.setattr("sys.stdin.isatty", lambda: False)
        monkeypatch.setattr("terok_sandbox.config.credentials_use_keyring", lambda: False)
        monkeypatch.setattr(
            "terok_sandbox.vault.store.encryption.load_passphrase_from_keyring",
            lambda: None,
        )

        with pytest.raises(SystemExit, match="no current passphrase"):
            handle_vault_to_keyring()

    def test_to_keyring_refuses_until_recovery_acknowledged(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Escrow-before-enable: moving to the keyring is blocked until recovery is acknowledged."""
        from unittest.mock import MagicMock

        from terok_sandbox.commands import handle_vault_to_keyring

        cfg = _make_cfg(tmp_path)
        cfg.vault_passphrase_file.parent.mkdir(parents=True, exist_ok=True)
        cfg.vault_passphrase_file.write_text("current-pw\n")  # passphrase resolvable…
        monkeypatch.setattr(
            "terok_sandbox.vault.store.encryption.load_passphrase_from_file",
            load_passphrase_from_file,
        )
        store = MagicMock(return_value=True)
        monkeypatch.setattr(
            "terok_sandbox.vault.store.encryption.store_passphrase_in_keyring", store
        )
        # …but recovery is NOT acknowledged → refuse before any keyring write.
        with pytest.raises(SystemExit, match="recovery passphrase isn't marked as saved"):
            handle_vault_to_keyring(cfg=cfg)
        store.assert_not_called()


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
            "terok_sandbox.paths.config_file_paths", lambda: [("user", user_config)]
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
            "terok_sandbox.paths.config_file_paths", lambda: [("user", user_config)]
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
            "terok_sandbox.paths.config_file_paths", lambda: [("user", user_config)]
        )
        _persist_mode_choice("session-file", "irrelevant")
        assert not user_config.exists()


class TestCredentialsSetupPhase:
    """The thin wrapper that frames ``_handle_credentials_encrypt_db`` for setup output."""

    def test_returns_true_on_success(self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
        """Happy path: handler runs without raising → True."""
        from terok_sandbox.commands import _run_credentials_setup_phase

        cfg = _make_cfg(tmp_path)
        # Drive the chooser path explicitly (default is keyring; tests on
        # CI shouldn't depend on which tier the host actually has).
        _disable_systemd_creds(monkeypatch)
        _patch_dev_tty(monkeypatch)
        monkeypatch.setattr("sys.stdin.isatty", lambda: True)
        responses = iter(["s\n"])
        monkeypatch.setattr("sys.stdin.readline", lambda: next(responses))
        monkeypatch.setattr("prompt_toolkit.prompt", lambda *_a, **_kw: "")
        assert _run_credentials_setup_phase(cfg) is True

    def test_returns_false_when_handler_raises(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Any exception → caller can mark ``failed`` and keep going through other phases."""
        from terok_sandbox.commands import credentials as commands

        def _boom(**_kw: object) -> None:
            raise RuntimeError("simulated failure")

        monkeypatch.setattr(commands, "_handle_credentials_encrypt_db", _boom)
        assert commands._run_credentials_setup_phase(_make_cfg(tmp_path)) is False

    def test_database_locked_emits_fuser_hint(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
    ) -> None:
        """A ``database is locked`` failure points the operator at ``fuser``.

        In the per-container supervisor world a lock means a live
        supervisor still holds the DB; the hint names the exact
        ``fuser -v <db_path>`` to find it and tells the operator to delete
        the matching task before re-running setup.  The DB path is
        ``shlex.quote``-d so a path with spaces stays one shell token.
        """
        from terok_sandbox.commands import credentials as commands

        def _boom(**_kw: object) -> None:
            raise RuntimeError("attempt to write: database is locked")

        monkeypatch.setattr(commands, "_handle_credentials_encrypt_db", _boom)
        cfg = _make_cfg(tmp_path)
        assert commands._run_credentials_setup_phase(cfg) is False

        out = capsys.readouterr().out
        assert "FAILED" in out
        assert "per-container supervisor still holds the DB" in out
        assert f"fuser -v {cfg.db_path}" in out

    def test_non_lock_failure_omits_fuser_hint(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
    ) -> None:
        """A failure that isn't a DB lock reports FAILED without the ``fuser`` hint."""
        from terok_sandbox.commands import credentials as commands

        def _boom(**_kw: object) -> None:
            raise RuntimeError("no passphrase resolvable")

        monkeypatch.setattr(commands, "_handle_credentials_encrypt_db", _boom)
        assert commands._run_credentials_setup_phase(_make_cfg(tmp_path)) is False

        out = capsys.readouterr().out
        assert "FAILED" in out
        assert "fuser" not in out


class TestCredentialsCommandCoverageGaps:
    """Branches the main credential-encryption tests don't reach."""

    def test_encrypt_db_defaults_cfg(self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
        """``credentials encrypt-db`` without ``cfg=`` constructs a default SandboxConfig.

        Short-circuits as soon as the DB existence check sees the path
        is absent — that's enough to cover the default-cfg branch.
        """
        from terok_sandbox.commands import _handle_credentials_encrypt_db, credentials as cred_cmds
        from terok_sandbox.config import SandboxConfig

        # Build the fallback config with the real class *before* patching,
        # so the default-factory (which imports SandboxConfig lazily) hands
        # back this instance without recursing into the stub.
        fake_cfg = SandboxConfig(
            state_dir=tmp_path / "state",
            runtime_dir=tmp_path / "rt",
            config_dir=tmp_path / "cfg",
            vault_dir=tmp_path / "vault-absent",
            services_mode="socket",
        )
        monkeypatch.setattr("terok_sandbox.config.SandboxConfig", lambda: fake_cfg)
        # systemd-creds path is the fast-track that needs least mocking.
        monkeypatch.setattr("terok_sandbox.vault.store.systemd_creds.is_available", lambda: True)
        monkeypatch.setattr(
            cred_cmds,
            "_provision_systemd_creds_tier",
            lambda _cfg, **_: ("pw", "systemd-creds"),
        )
        _handle_credentials_encrypt_db()  # cfg= omitted — exercises the default-factory line

    def test_back_up_plaintext_db_unlinks_on_failure(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """A mid-tar exception removes the half-written tarball before re-raising.

        Streaming the tar into an O_EXCL fd means a crash would otherwise
        leave a partial tarball containing partial cleartext, which is
        worse than no backup at all.
        """
        from terok_sandbox.commands import _back_up_plaintext_db

        db_path = tmp_path / "vault" / "credentials.db"
        db_path.parent.mkdir(parents=True)
        db_path.write_bytes(b"plain")

        import tarfile

        real_open = tarfile.open

        def _boom(*args, **kwargs):  # noqa: ANN002, ANN003
            real = real_open(*args, **kwargs)
            real.close()  # so the fd doesn't leak
            raise RuntimeError("simulated mid-tar failure")

        monkeypatch.setattr(tarfile, "open", _boom)
        with pytest.raises(RuntimeError, match="simulated mid-tar failure"):
            _back_up_plaintext_db(db_path)
        # The cleanup branch must have unlinked the partial tarball.
        backups = list(db_path.parent.glob("*.plaintext-backup-*.tar.gz"))
        assert backups == []


class TestProvisionPassphraseTier:
    """``provision_passphrase_tier`` — the no-terminal provisioning API for TUI frontends."""

    @staticmethod
    def _make_encrypted_db(cfg) -> None:
        """Create a minimal SQLCipher DB at ``cfg.db_path`` keyed with ``_PASSPHRASE``."""
        cfg.db_path.parent.mkdir(parents=True, exist_ok=True)
        conn = open_sqlcipher(cfg.db_path, _PASSPHRASE)
        conn.execute("CREATE TABLE t (x INTEGER)")
        conn.commit()
        conn.close()

    def test_unknown_tier_fails_fast(self, tmp_path: Path) -> None:
        """The plaintext ``config`` tier (and typos) are rejected before anything runs."""
        from terok_sandbox.commands import provision_passphrase_tier

        with pytest.raises(ValueError, match="cannot provision tier 'config'"):
            provision_passphrase_tier(_make_cfg(tmp_path), tier="config")

    def test_session_file_mints_when_no_passphrase_given(self, tmp_path: Path) -> None:
        """``passphrase=None`` mints; the caller gets the value back for its reveal surface."""
        from terok_sandbox.commands import provision_passphrase_tier

        cfg = _make_cfg(tmp_path)
        result = provision_passphrase_tier(cfg, tier="session-file")
        assert result.generated is True
        assert result.source == "session-file"
        assert cfg.vault_passphrase_file.read_text().rstrip("\n") == result.passphrase
        # A mint is ~256 bits of token_urlsafe, never something short.
        assert len(result.passphrase) > 20

    def test_session_file_accepts_caller_supplied_value(self, tmp_path: Path) -> None:
        """A typed (twice-confirmed by the frontend) value lands verbatim, generated=False."""
        from terok_sandbox.commands import provision_passphrase_tier

        cfg = _make_cfg(tmp_path)
        result = provision_passphrase_tier(cfg, tier="session-file", passphrase=_PASSPHRASE)
        assert result.generated is False
        assert cfg.vault_passphrase_file.read_text().rstrip("\n") == _PASSPHRASE

    def test_keyring_stores_and_persists_mode_choice(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Keyring tier stores the value and flips ``use_keyring`` in config.yml."""
        from terok_sandbox.commands import credentials as cred_cmds
        from terok_sandbox.commands.credentials import provision_passphrase_tier

        stored: dict[str, str] = {}
        monkeypatch.setattr(
            "terok_sandbox.vault.store.encryption.store_passphrase_in_keyring",
            lambda pw: stored.update(pw=pw) or True,
        )
        persisted: dict[str, str] = {}
        monkeypatch.setattr(
            cred_cmds,
            "_persist_mode_choice",
            lambda mode, pw: persisted.update(mode=mode),
        )
        result = provision_passphrase_tier(_make_cfg(tmp_path), tier="keyring")
        assert result.source == "keyring" and result.generated is True
        assert stored["pw"] == result.passphrase
        assert persisted["mode"] == "keyring"

    def test_keyring_unreachable_raises(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """A dead Secret Service raises instead of silently losing the value."""
        from terok_sandbox.commands import provision_passphrase_tier

        monkeypatch.setattr(
            "terok_sandbox.vault.store.encryption.store_passphrase_in_keyring",
            lambda _pw: False,
        )
        with pytest.raises(RuntimeError, match="keyring is unreachable"):
            provision_passphrase_tier(_make_cfg(tmp_path), tier="keyring")

    def test_systemd_creds_unavailable_raises(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Explicitly asking for systemd-creds on an old host refuses loudly."""
        from terok_sandbox.commands import provision_passphrase_tier

        _disable_systemd_creds(monkeypatch)
        with pytest.raises(RuntimeError, match="systemd-creds is unavailable"):
            provision_passphrase_tier(_make_cfg(tmp_path), tier="systemd-creds")

    def test_systemd_creds_seals_the_mint(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Available systemd-creds → mint is sealed to the configured credential file."""
        from terok_sandbox.commands import provision_passphrase_tier

        sealed: dict[str, object] = {}
        monkeypatch.setattr("terok_sandbox.vault.store.systemd_creds.is_available", lambda: True)
        monkeypatch.setattr(
            "terok_sandbox.vault.store.systemd_creds.seal",
            lambda pw, path, key_mode: sealed.update(pw=pw, path=path, key_mode=key_mode),
        )
        cfg = _make_cfg(tmp_path)
        result = provision_passphrase_tier(cfg, tier="systemd-creds")
        assert result.source == "systemd-creds" and result.generated is True
        assert sealed["pw"] == result.passphrase
        assert sealed["path"] == cfg.vault_systemd_creds_file

    def test_encrypted_db_refuses_a_mint(self, tmp_path: Path) -> None:
        """No fresh mint can ever open an existing DB — fail before anything lands."""
        from terok_sandbox.commands import provision_passphrase_tier

        cfg = _make_cfg(tmp_path)
        self._make_encrypted_db(cfg)
        with pytest.raises(NoPassphraseError, match="already encrypted"):
            provision_passphrase_tier(cfg, tier="session-file")
        assert not cfg.vault_passphrase_file.exists()

    def test_encrypted_db_rejects_a_wrong_value(self, tmp_path: Path) -> None:
        """The fresh-install trapdoor, closed: a mismatch never lands on any tier."""
        from terok_sandbox.commands import provision_passphrase_tier

        cfg = _make_cfg(tmp_path)
        self._make_encrypted_db(cfg)
        with pytest.raises(WrongPassphraseError):
            provision_passphrase_tier(cfg, tier="session-file", passphrase="wrong-guess")
        assert not cfg.vault_passphrase_file.exists()

    def test_encrypted_db_accepts_the_matching_value(self, tmp_path: Path) -> None:
        """The real key validates against the DB, then lands on the chosen tier."""
        from terok_sandbox.commands import provision_passphrase_tier

        cfg = _make_cfg(tmp_path)
        self._make_encrypted_db(cfg)
        result = provision_passphrase_tier(cfg, tier="session-file", passphrase=_PASSPHRASE)
        assert result.generated is False
        assert cfg.vault_passphrase_file.read_text().rstrip("\n") == _PASSPHRASE


class TestCredentialsProvisioned:
    """``credentials_provisioned`` — the frontends' pre-flight probe before dispatching setup."""

    def test_false_on_a_fresh_host(self, tmp_path: Path) -> None:
        """No DB, no tier → a non-TTY setup would fail closed, so the probe says False."""
        from terok_sandbox.commands import credentials_provisioned

        assert credentials_provisioned(_make_cfg(tmp_path)) is False

    def test_true_when_db_already_encrypted(self, tmp_path: Path) -> None:
        """An encrypted DB short-circuits the whole credentials phase."""
        from terok_sandbox.commands import credentials_provisioned

        cfg = _make_cfg(tmp_path)
        TestProvisionPassphraseTier._make_encrypted_db(cfg)
        assert credentials_provisioned(cfg) is True

    def test_true_when_a_tier_resolves(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """A provisioned session file counts even before any DB exists."""
        import terok_sandbox.vault.store.encryption as _enc
        from terok_sandbox.commands import credentials_provisioned, provision_passphrase_tier

        # The autouse chain-isolation fixture blanks the file tier;
        # restore the real reader — this test *is about* the file tier.
        monkeypatch.setattr(_enc, "load_passphrase_from_file", load_passphrase_from_file)
        cfg = _make_cfg(tmp_path)
        provision_passphrase_tier(cfg, tier="session-file")
        assert credentials_provisioned(cfg) is True


class TestSelectAndProvisionReusesExistingTier:
    """Setup's tier selection reuses whatever already resolves instead of re-asking."""

    def test_session_file_reused_without_chooser(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Pre-provisioned tier + non-TTY setup → no chooser, no fail-closed refusal."""
        import terok_sandbox.vault.store.encryption as _enc
        from terok_sandbox.commands.credentials import (
            _select_and_provision,
            provision_passphrase_tier,
        )

        # Restore the file-tier reader blanked by the chain-isolation fixture.
        monkeypatch.setattr(_enc, "load_passphrase_from_file", load_passphrase_from_file)
        cfg = _make_cfg(tmp_path)
        provisioned = provision_passphrase_tier(cfg, tier="session-file")
        _disable_systemd_creds(monkeypatch)
        monkeypatch.setattr("sys.stdin.isatty", lambda: False)  # chooser would SystemExit
        passphrase, source, auto_generated = _select_and_provision(
            cfg, passphrase_tier=None, echo_passphrase=False
        )
        assert (passphrase, source, auto_generated) == (
            provisioned.passphrase,
            "session-file",
            False,
        )

    def test_config_tier_reused_without_chooser(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """A configured plaintext fallback also pre-empts the chooser on re-runs."""
        from terok_sandbox.commands.credentials import _select_and_provision

        cfg = _make_cfg(tmp_path, passphrase="configured-key")
        _disable_systemd_creds(monkeypatch)
        monkeypatch.setattr("sys.stdin.isatty", lambda: False)
        passphrase, source, auto_generated = _select_and_provision(
            cfg, passphrase_tier=None, echo_passphrase=False
        )
        assert (passphrase, source, auto_generated) == ("configured-key", "config", False)


class TestCredentialsSetupPhaseSystemExit:
    """The stage line must terminate cleanly when the handler refuses via SystemExit."""

    def test_system_exit_hint_prints_on_its_own_lines(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
    ) -> None:
        """The chooser's multi-line refusal must not glue onto the ``→ credentials`` line."""
        from terok_sandbox.commands import credentials as commands

        hint = "setup: running non-interactively but no passphrase tier was chosen.\n  pick one"

        def _refuse(**_kw: object) -> None:
            raise SystemExit(hint)

        monkeypatch.setattr(commands, "_handle_credentials_encrypt_db", _refuse)
        assert commands._run_credentials_setup_phase(_make_cfg(tmp_path)) is False

        out = capsys.readouterr().out
        assert "→ credentials — FAILED\n" in out
        assert hint in out

    def test_integer_exit_code_is_reported(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
    ) -> None:
        """A bare numeric SystemExit still terminates the line and names the code."""
        from terok_sandbox.commands import credentials as commands

        def _refuse(**_kw: object) -> None:
            raise SystemExit(3)

        monkeypatch.setattr(commands, "_handle_credentials_encrypt_db", _refuse)
        assert commands._run_credentials_setup_phase(_make_cfg(tmp_path)) is False

        out = capsys.readouterr().out
        assert "→ credentials — FAILED\n" in out
        assert "exit code 3" in out


class TestKeyringBackendAvailable:
    """``keyring_backend_available`` — the chooser's should-we-offer-keyring probe."""

    def test_fail_backend_reports_unavailable(self, monkeypatch: pytest.MonkeyPatch) -> None:
        from keyring.backends import fail

        from terok_sandbox.vault.store.encryption import keyring_backend_available

        monkeypatch.setattr("keyring.get_keyring", lambda: fail.Keyring())
        assert keyring_backend_available() is False

    def test_null_backend_reports_unavailable(self, monkeypatch: pytest.MonkeyPatch) -> None:
        from keyring.backends import null

        from terok_sandbox.vault.store.encryption import keyring_backend_available

        monkeypatch.setattr("keyring.get_keyring", lambda: null.Keyring())
        assert keyring_backend_available() is False

    def test_real_backend_reports_available(self, monkeypatch: pytest.MonkeyPatch) -> None:
        import keyring.backend

        from terok_sandbox.vault.store.encryption import keyring_backend_available

        class _InMemory(keyring.backend.KeyringBackend):
            priority = 1

            def get_password(self, service, username):  # noqa: D102
                return None

            def set_password(self, service, username, password):  # noqa: D102
                return None

            def delete_password(self, service, username):  # noqa: D102
                return None

        monkeypatch.setattr("keyring.get_keyring", lambda: _InMemory())
        assert keyring_backend_available() is True

    def test_import_failure_degrades_to_unavailable(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """No keyring package at all → probe answers False instead of raising."""
        from terok_sandbox.vault.store.encryption import keyring_backend_available

        def _boom() -> None:
            raise RuntimeError("no backend")

        monkeypatch.setattr("keyring.get_keyring", _boom)
        assert keyring_backend_available() is False
