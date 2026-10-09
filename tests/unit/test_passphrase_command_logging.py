# SPDX-FileCopyrightText: 2026 Jiri Vyskocil
# SPDX-License-Identifier: Apache-2.0

"""Passphrase helpers never disclose command text or output in diagnostics."""

import subprocess
from unittest.mock import Mock

import pytest

from terok_sandbox.vault.store import encryption

_SECRET = "dummy-secret-that-must-not-be-logged"


def test_failed_helper_logs_only_exit_code(
    monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture
) -> None:
    """Neither stdout, stderr nor even the executable token is safe to log."""
    result = subprocess.CompletedProcess([_SECRET], 7, stdout=_SECRET, stderr=_SECRET)
    monkeypatch.setattr(encryption.subprocess, "run", Mock(return_value=result))

    assert encryption.load_passphrase_from_command(_SECRET) is None
    assert caplog.messages == ["passphrase_command exited 7"]
    assert _SECRET not in caplog.text


@pytest.mark.parametrize(
    ("failure", "diagnostic"),
    [
        (OSError(_SECRET), "failed to spawn (OSError)"),
        (
            subprocess.TimeoutExpired(_SECRET, 1, output=_SECRET, stderr=_SECRET),
            "timed out after 1s",
        ),
        (
            UnicodeDecodeError("utf-8", _SECRET.encode(), 0, 1, _SECRET),
            "text conversion failed (UnicodeDecodeError)",
        ),
    ],
)
def test_execution_errors_do_not_expose_secret(
    monkeypatch: pytest.MonkeyPatch,
    caplog: pytest.LogCaptureFixture,
    failure: Exception,
    diagnostic: str,
) -> None:
    """Helper failures keep their category without copying secret-bearing exceptions."""
    monkeypatch.setattr(encryption.subprocess, "run", Mock(side_effect=failure))

    assert encryption.load_passphrase_from_command(_SECRET, timeout=1) is None
    assert caplog.messages == [f"passphrase_command {diagnostic}"]
    assert _SECRET not in caplog.text


def test_parse_error_does_not_expose_command_or_exception(
    monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture
) -> None:
    """Parse errors are safe even when the parser's diagnostic contains the secret."""
    monkeypatch.setattr(encryption.shlex, "split", Mock(side_effect=ValueError(_SECRET)))
    run = Mock()
    monkeypatch.setattr(encryption.subprocess, "run", run)

    assert encryption.load_passphrase_from_command(_SECRET) is None
    assert caplog.messages == ["passphrase_command shlex parse failed (ValueError)"]
    assert _SECRET not in caplog.text
    run.assert_not_called()


def test_success_does_not_log_stderr_and_preserves_passphrase_whitespace(
    monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture
) -> None:
    """Only the helper's line ending is removed; diagnostics never echo either stream."""
    passphrase = f"  {_SECRET}  "
    result = subprocess.CompletedProcess(["helper"], 0, stdout=f"{passphrase}\n", stderr=_SECRET)
    monkeypatch.setattr(encryption.subprocess, "run", Mock(return_value=result))

    assert encryption.load_passphrase_from_command("helper") == passphrase
    assert caplog.messages == []
