# SPDX-FileCopyrightText: 2026 Jiri Vyskocil
# SPDX-License-Identifier: Apache-2.0

"""Tests for the public stage-line renderer."""

from __future__ import annotations

from unittest.mock import patch

import pytest

import terok_sandbox._stage as _stage_module
from terok_sandbox._stage import (
    STAGE_WIDTH,
    Marker,
    bold,
    red,
    stage,
    stage_begin,
    stage_end,
    yellow,
)


@pytest.fixture
def plain() -> None:
    """Force ``_COLOUR_ON=False`` so assertions check raw strings without ANSI."""
    with patch.object(_stage_module, "_COLOUR_ON", False):
        yield


@pytest.fixture
def coloured() -> None:
    """Force ``_COLOUR_ON=True`` so assertions check that SGR codes round-trip."""
    with patch.object(_stage_module, "_COLOUR_ON", True):
        yield


class TestStage:
    """``stage`` writes one complete ``'  <label>  <marker>[ (<detail>)]'`` line."""

    def test_includes_label_marker_and_detail(
        self, plain: None, capsys: pytest.CaptureFixture[str]
    ) -> None:
        stage("Vault", Marker.OK, "systemd, tcp, reachable")
        out = capsys.readouterr().out
        assert "Vault" in out
        assert " ok " in out
        assert "(systemd, tcp, reachable)" in out

    def test_blank_detail_emits_no_parens(
        self, plain: None, capsys: pytest.CaptureFixture[str]
    ) -> None:
        stage("Shield hooks", Marker.OK)
        assert "()" not in capsys.readouterr().out

    def test_label_padded_to_consistent_column(
        self, plain: None, capsys: pytest.CaptureFixture[str]
    ) -> None:
        stage("x", Marker.OK, "a")
        stage("a_longer_label", Marker.OK, "b")
        lines = capsys.readouterr().out.splitlines()
        assert lines[0].index(" ok ") == lines[1].index(" ok ")

    def test_marker_gets_ansi_colour_when_enabled(
        self, coloured: None, capsys: pytest.CaptureFixture[str]
    ) -> None:
        stage("Vault", Marker.OK)
        out = capsys.readouterr().out
        assert "\x1b[32m" in out
        assert "\x1b[0m" in out

    def test_skip_marker_stays_uncoloured_even_when_enabled(
        self, coloured: None, capsys: pytest.CaptureFixture[str]
    ) -> None:
        """``skip`` is a soft, user-chosen state — neither success nor failure."""
        stage("Clearance", Marker.SKIP, "terok_clearance not installed")
        out = capsys.readouterr().out
        assert "\x1b[" not in out

    def test_fail_marker_is_red_when_enabled(
        self, coloured: None, capsys: pytest.CaptureFixture[str]
    ) -> None:
        stage("Vault", Marker.FAIL, "connection refused")
        assert "\x1b[31m" in capsys.readouterr().out


class TestStageBeginEnd:
    """Progressive pair — label up front, marker on completion."""

    def test_begin_writes_padded_label_without_newline(
        self, plain: None, capsys: pytest.CaptureFixture[str]
    ) -> None:
        stage_begin("Desktop entry")
        out = capsys.readouterr().out
        assert "Desktop entry" in out
        assert "\n" not in out

    def test_end_writes_marker_and_newline(
        self, plain: None, capsys: pytest.CaptureFixture[str]
    ) -> None:
        stage_end(Marker.OK, "installed")
        out = capsys.readouterr().out
        assert "ok" in out
        assert "(installed)" in out
        assert out.endswith("\n")

    def test_begin_end_compose_to_same_line_as_stage(
        self, plain: None, capsys: pytest.CaptureFixture[str]
    ) -> None:
        """``stage_begin + stage_end`` renders byte-for-byte like ``stage``."""
        stage_begin("Vault")
        stage_end(Marker.OK, "reachable")
        combined = capsys.readouterr().out

        stage("Vault", Marker.OK, "reachable")
        joined = capsys.readouterr().out

        assert combined == joined


class TestBannerColour:
    """``bold`` / ``red`` / ``yellow`` wrap banner text with the same gate as stage markers."""

    def test_bold_passthrough_when_disabled(self, plain: None) -> None:
        assert bold("Setup complete.") == "Setup complete."

    def test_bold_wraps_when_enabled(self, coloured: None) -> None:
        wrapped = bold("Setup complete.")
        assert wrapped.startswith("\x1b[1m")
        assert wrapped.endswith("\x1b[0m")

    def test_red_passthrough_when_disabled(self, plain: None) -> None:
        assert red("Setup failed.") == "Setup failed."

    def test_red_wraps_when_enabled(self, coloured: None) -> None:
        assert red("Setup failed.").startswith("\x1b[31m")

    def test_yellow_passthrough_when_disabled(self, plain: None) -> None:
        assert yellow("Warning.") == "Warning."

    def test_yellow_wraps_when_enabled(self, coloured: None) -> None:
        assert yellow("Warning.").startswith("\x1b[33m")


def test_stage_width_fits_widest_shipped_label() -> None:
    """Sanity: ``"Clearance notifier"`` (18 chars) fits the configured gutter."""
    assert len("Clearance notifier") <= STAGE_WIDTH
