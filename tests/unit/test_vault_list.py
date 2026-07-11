# SPDX-FileCopyrightText: 2026 Jiri Vyskocil
# SPDX-License-Identifier: Apache-2.0

"""Tests for the ``vault list`` CLI verb and its render helpers.

``_handle_vault_list`` is read-only inventory: it opens the credential
DB, walks every credential set / provider, and renders either a
fixed-width table or a JSON blob.  Secret values never leave the DB —
only field *names* surface for credential payloads and an 8-char prefix
for proxy tokens.  A locked / unreachable vault collapses to a friendly
one-line error and ``SystemExit(2)``.

The DB is mocked at the ``cfg.open_credential_db`` boundary (same style
as ``test_gate_mirror_extras.py``); the render helpers
(``_mask_token`` / ``_print_credentials_table`` / ``_print_tokens_table``)
are pure and exercised directly.
"""

from __future__ import annotations

import json
from unittest.mock import MagicMock

import pytest

from terok_sandbox.commands.vault import (
    _handle_vault_list,
    _mask_token,
    _print_credentials_table,
    _print_tokens_table,
)


def _db_with(
    *,
    credential_sets: dict[str, dict[str, dict]] | None = None,
    tokens: list[dict] | None = None,
) -> MagicMock:
    """Build a mock CredentialDB whose list/load methods return *credential_sets*.

    *credential_sets* maps ``set_name -> {provider -> payload}``.  The
    mock mirrors the real DB's ``list_credential_sets`` /
    ``list_credentials`` / ``load_credential`` / ``list_tokens`` /
    ``close`` surface so ``_handle_vault_list`` walks it unchanged.
    """
    credential_sets = credential_sets or {}
    db = MagicMock()
    db.list_credential_sets.return_value = list(credential_sets)
    db.list_credentials.side_effect = lambda cs: list(credential_sets.get(cs, {}))
    db.load_credential.side_effect = lambda cs, prov: credential_sets.get(cs, {}).get(prov)
    db.list_tokens.return_value = tokens or []
    return db


def _cfg_returning(db: MagicMock) -> MagicMock:
    """A mock SandboxConfig whose ``open_credential_db`` yields *db*."""
    cfg = MagicMock()
    cfg.open_credential_db.return_value = db
    return cfg


class TestHandleVaultListErrors:
    """A vault that won't open surfaces a friendly hint + ``SystemExit(2)``."""

    def test_open_failure_exits_2_with_hint(self, capsys: pytest.CaptureFixture[str]) -> None:
        cfg = MagicMock()
        cfg.open_credential_db.side_effect = RuntimeError("vault locked")

        with pytest.raises(SystemExit) as exc:
            _handle_vault_list(cfg=cfg)
        assert exc.value.code == 2

        err = capsys.readouterr().err
        assert "vault unreachable" in err
        assert "RuntimeError" in err
        assert "vault unlock" in err

    def test_systemexit_from_open_is_recaught_as_2(
        self, capsys: pytest.CaptureFixture[str]
    ) -> None:
        """A ``SystemExit`` raised by the passphrase prompt is re-framed, not propagated raw.

        ``open_credential_db(prompt_on_tty=True)`` can ``raise SystemExit``
        on a non-interactive terminal; the handler catches that too so the
        operator always gets the same actionable message + code 2.
        """
        cfg = MagicMock()
        cfg.open_credential_db.side_effect = SystemExit("no passphrase on tty")

        with pytest.raises(SystemExit) as exc:
            _handle_vault_list(cfg=cfg)
        assert exc.value.code == 2
        assert "vault unreachable" in capsys.readouterr().err

    def test_defaults_to_sandbox_config_when_cfg_is_none(
        self, capsys: pytest.CaptureFixture[str], monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Called with no ``cfg`` the handler builds a default ``SandboxConfig``."""
        from unittest.mock import patch

        default_cfg = MagicMock()
        default_cfg.open_credential_db.side_effect = RuntimeError("vault locked")
        with patch("terok_sandbox.config.SandboxConfig", return_value=default_cfg):
            with pytest.raises(SystemExit) as exc:
                _handle_vault_list()
        assert exc.value.code == 2
        assert "vault unreachable" in capsys.readouterr().err


class TestHandleVaultListTable:
    """Table rendering walks the DB and always closes the handle."""

    def test_empty_vault_renders_placeholder_and_closes(
        self, capsys: pytest.CaptureFixture[str]
    ) -> None:
        db = _db_with(credential_sets={})
        _handle_vault_list(cfg=_cfg_returning(db))
        out = capsys.readouterr().out
        assert "(no credentials stored)" in out
        db.close.assert_called_once()

    def test_renders_credential_rows_with_field_names_only(
        self, capsys: pytest.CaptureFixture[str]
    ) -> None:
        db = _db_with(
            credential_sets={
                "default": {
                    "claude": {"type": "oauth", "access_token": "SECRET", "refresh_token": "S2"},
                }
            }
        )
        _handle_vault_list(cfg=_cfg_returning(db))
        out = capsys.readouterr().out
        # Set / provider / type appear; secret *values* never do.
        assert "default" in out and "claude" in out and "oauth" in out
        assert "access_token" in out and "refresh_token" in out
        assert "SECRET" not in out and "S2" not in out
        db.close.assert_called_once()

    def test_credential_with_no_extra_fields_renders_dash(
        self, capsys: pytest.CaptureFixture[str]
    ) -> None:
        """A payload that's only ``type`` (no other keys) shows ``—`` for fields."""
        db = _db_with(credential_sets={"default": {"claude": {"type": "api_key"}}})
        _handle_vault_list(cfg=_cfg_returning(db))
        out = capsys.readouterr().out
        assert "api_key" in out
        assert "—" in out

    def test_missing_payload_uses_dash_type(self, capsys: pytest.CaptureFixture[str]) -> None:
        """``load_credential`` returning ``None`` doesn't crash — type falls back to ``—``."""
        db = _db_with(credential_sets={"default": {"claude": None}})  # type: ignore[dict-item]
        _handle_vault_list(cfg=_cfg_returning(db))
        out = capsys.readouterr().out
        assert "claude" in out
        db.close.assert_called_once()

    def test_include_tokens_prints_token_table(self, capsys: pytest.CaptureFixture[str]) -> None:
        db = _db_with(
            credential_sets={"default": {"claude": {"type": "oauth"}}},
            tokens=[
                {
                    "token": "terok-p-abcdef0123456789",
                    "scope": "proj-a",
                    "subject": "task-1",
                    "credential_set": "default",
                    "provider": "claude",
                }
            ],
        )
        _handle_vault_list(cfg=_cfg_returning(db), include_tokens=True)
        out = capsys.readouterr().out
        assert "proxy tokens" in out
        # Masked prefix is shown; full token value is not.
        assert "terok-p-abcdef01" in out
        assert "terok-p-abcdef0123456789" not in out
        db.list_tokens.assert_called_once()

    def test_tokens_not_listed_without_flag(self) -> None:
        """``list_tokens`` is never queried when ``include_tokens`` is False."""
        db = _db_with(credential_sets={"default": {"claude": {"type": "oauth"}}})
        _handle_vault_list(cfg=_cfg_returning(db), include_tokens=False)
        db.list_tokens.assert_not_called()

    def test_db_closed_even_when_listing_raises(self) -> None:
        """A query failure mid-walk still closes the handle (the ``finally``)."""
        db = _db_with(credential_sets={"default": {"claude": {"type": "oauth"}}})
        db.list_credentials.side_effect = RuntimeError("schema drift")
        with pytest.raises(RuntimeError):
            _handle_vault_list(cfg=_cfg_returning(db))
        db.close.assert_called_once()


class TestHandleVaultListJson:
    """``as_json`` emits a machine-readable blob; secrets stay masked."""

    def test_json_credentials_only(self, capsys: pytest.CaptureFixture[str]) -> None:
        db = _db_with(
            credential_sets={
                "default": {"claude": {"type": "oauth", "access_token": "X", "client_id": "Y"}}
            }
        )
        _handle_vault_list(cfg=_cfg_returning(db), as_json=True)
        payload = json.loads(capsys.readouterr().out)
        assert "tokens" not in payload
        cred = payload["credentials"][0]
        assert cred["credential_set"] == "default"
        assert cred["provider"] == "claude"
        assert cred["type"] == "oauth"
        # Fields are sorted names, never the secret values.
        assert cred["fields"] == ["access_token", "client_id"]

    def test_json_with_tokens_masks_token_values(self, capsys: pytest.CaptureFixture[str]) -> None:
        db = _db_with(
            credential_sets={},
            tokens=[
                {
                    "token": "terok-p-deadbeefcafef00d",
                    "scope": "proj-b",
                    "subject": "task-9",
                    "credential_set": "work",
                    "provider": "codex",
                }
            ],
        )
        _handle_vault_list(cfg=_cfg_returning(db), include_tokens=True, as_json=True)
        payload = json.loads(capsys.readouterr().out)
        token = payload["tokens"][0]
        assert token["token_prefix"] == "terok-p-deadbeef…"
        assert token["scope"] == "proj-b"
        assert token["subject"] == "task-9"
        assert token["credential_set"] == "work"
        assert token["provider"] == "codex"


class TestMaskToken:
    """``_mask_token`` keeps the ``terok-p-`` namespace + 8 random chars."""

    def test_namespaced_token_keeps_prefix_plus_8(self) -> None:
        assert _mask_token("terok-p-0123456789abcdef") == "terok-p-01234567…"

    def test_foreign_token_truncated_to_8(self) -> None:
        assert _mask_token("ghp_secretsecretsecret") == "ghp_secr…"


class TestPrintTables:
    """Render helpers are pure stdout formatters with sanitised cells."""

    def test_credentials_table_placeholder_when_empty(
        self, capsys: pytest.CaptureFixture[str]
    ) -> None:
        _print_credentials_table([])
        assert "(no credentials stored)" in capsys.readouterr().out

    def test_credentials_table_renders_header_and_rows(
        self, capsys: pytest.CaptureFixture[str]
    ) -> None:
        _print_credentials_table(
            [
                {
                    "credential_set": "default",
                    "provider": "claude",
                    "type": "oauth",
                    "fields": ["access_token", "refresh_token"],
                }
            ]
        )
        out = capsys.readouterr().out
        assert "credential_set" in out and "provider" in out
        assert "default" in out and "claude" in out
        assert "access_token, refresh_token" in out

    def test_credentials_table_empty_fields_shows_dash(
        self, capsys: pytest.CaptureFixture[str]
    ) -> None:
        _print_credentials_table(
            [{"credential_set": "s", "provider": "p", "type": "t", "fields": []}]
        )
        assert "—" in capsys.readouterr().out

    def test_credentials_table_sanitises_hostile_cells(
        self, capsys: pytest.CaptureFixture[str]
    ) -> None:
        """A credential-set name with an ANSI escape is neutralised before display."""
        _print_credentials_table(
            [
                {
                    "credential_set": "evil\x1b[31mset",
                    "provider": "p",
                    "type": "t",
                    "fields": ["k"],
                }
            ]
        )
        out = capsys.readouterr().out
        assert "\x1b" not in out

    def test_tokens_table_placeholder_when_empty(self, capsys: pytest.CaptureFixture[str]) -> None:
        _print_tokens_table([])
        assert "(no proxy tokens issued)" in capsys.readouterr().out

    def test_tokens_table_masks_and_renders(self, capsys: pytest.CaptureFixture[str]) -> None:
        _print_tokens_table(
            [
                {
                    "token": "terok-p-0123456789abcdef",
                    "scope": "proj-a",
                    "subject": "task-1",
                    "credential_set": "default",
                    "provider": "claude",
                }
            ]
        )
        out = capsys.readouterr().out
        assert "proxy tokens" in out
        assert "terok-p-01234567…" in out
        assert "0123456789abcdef" not in out
        assert "proj-a" in out and "task-1" in out
