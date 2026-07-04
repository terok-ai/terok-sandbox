# SPDX-FileCopyrightText: 2026 Jiri Vyskocil
# SPDX-License-Identifier: Apache-2.0

"""Tests for operator-facing invocation identity in re-run hints."""

from __future__ import annotations

import pytest

from terok_sandbox.commands.credentials import _NON_TTY_TIER_HINT
from terok_sandbox.operator_cli import SETUP_INVOCATION_ENV, setup_invocation


class TestSetupInvocation:
    """Hints name the invocation the consuming front-end declared."""

    def test_frontend_declaration_wins(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """A front-end's declared spelling is used verbatim."""
        monkeypatch.setenv(SETUP_INVOCATION_ENV, "terok setup")
        assert setup_invocation() == "terok setup"

    def test_undeclared_falls_back_to_sandbox(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """No declaration → sandbox's own always-valid spelling."""
        monkeypatch.delenv(SETUP_INVOCATION_ENV, raising=False)
        assert setup_invocation() == "terok-sandbox setup"

    def test_empty_declaration_falls_back(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """An empty declaration is treated as absent, not as a spelling."""
        monkeypatch.setenv(SETUP_INVOCATION_ENV, "")
        assert setup_invocation() == "terok-sandbox setup"

    def test_non_tty_hint_renders_the_declared_invocation(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """The non-TTY tier hint speaks the front-end's language end to end."""
        monkeypatch.setenv(SETUP_INVOCATION_ENV, "terok setup")
        rendered = _NON_TTY_TIER_HINT.format(setup=setup_invocation())
        assert rendered.startswith("terok setup: running non-interactively")
        assert "--passphrase-tier keyring" in rendered
        assert "re-run `terok setup`" in rendered
