# SPDX-FileCopyrightText: 2026 Jiri Vyskocil
# SPDX-License-Identifier: Apache-2.0

"""TTY passphrase entry preserves the same exact text as the TUI and helpers."""

from unittest.mock import Mock

import pytest

from terok_sandbox.vault.store.encryption import prompt_new_passphrase, prompt_passphrase

_PADDED_VALUES = ("  dummy-passphrase  ", "\u2003 pässphrase\t", " \t\u2003 ")


@pytest.mark.parametrize("passphrase", _PADDED_VALUES)
def test_unlock_preserves_exact_text(monkeypatch: pytest.MonkeyPatch, passphrase: str) -> None:
    """Leading, trailing, Unicode and whitespace-only text remain key material."""
    monkeypatch.setattr("sys.stdin.isatty", lambda: True)
    monkeypatch.setattr("prompt_toolkit.prompt", Mock(return_value=passphrase))

    assert prompt_passphrase() == passphrase


@pytest.mark.parametrize("passphrase", _PADDED_VALUES)
@pytest.mark.parametrize("setup", [False, True])
def test_new_and_confirm_preserve_exact_text(
    monkeypatch: pytest.MonkeyPatch, passphrase: str, setup: bool
) -> None:
    """Both setup and rotation confirm and return the original untrimmed text."""
    monkeypatch.setattr("sys.stdin.isatty", lambda: True)
    prompt = Mock(side_effect=[passphrase, passphrase])
    monkeypatch.setattr("prompt_toolkit.prompt", prompt)
    monkeypatch.setattr(
        "terok_sandbox.vault.store.encryption.generate_passphrase",
        Mock(side_effect=AssertionError("nonempty text must never request generation")),
    )

    actual = prompt_passphrase(confirm=True) if setup else prompt_new_passphrase()
    assert actual == passphrase
    assert prompt.call_count == 2


@pytest.mark.parametrize(
    ("first", "second"),
    [
        (" dummy-passphrase", "dummy-passphrase"),
        ("dummy-passphrase ", "dummy-passphrase"),
        ("\u2003pässphrase\u2003", "pässphrase"),
        (" ", "  "),
    ],
)
def test_confirmation_rejects_different_edges(
    monkeypatch: pytest.MonkeyPatch, first: str, second: str
) -> None:
    """Entries differing only in surrounding whitespace are different passphrases."""
    monkeypatch.setattr("prompt_toolkit.prompt", Mock(side_effect=[first, second]))

    with pytest.raises(ValueError, match="passphrases do not match"):
        prompt_new_passphrase()


def test_only_empty_new_entry_requests_generation(monkeypatch: pytest.MonkeyPatch) -> None:
    """An actual empty string still returns the established generation sentinel."""
    prompt = Mock(return_value="")
    monkeypatch.setattr("prompt_toolkit.prompt", prompt)

    assert prompt_new_passphrase() is None
    prompt.assert_called_once()
