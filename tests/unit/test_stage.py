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
    _detect_colour,
    bold,
    red,
    stage,
    stage_begin,
    stage_end,
    stage_line,
    supports_color,
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


class TestColourDetection:
    """Verify the NO_COLOR / FORCE_COLOR / isatty precedence in :func:`_detect_colour`."""

    def test_no_color_env_disables_colour(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """``NO_COLOR`` always wins — per the no-color.org contract."""
        monkeypatch.setenv("NO_COLOR", "1")
        monkeypatch.setenv("FORCE_COLOR", "1")  # ignored in NO_COLOR's presence
        assert _detect_colour() is False

    def test_force_color_opts_back_in_even_without_tty(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """``FORCE_COLOR`` beats the isatty probe when set to anything non-``"0"``."""
        monkeypatch.delenv("NO_COLOR", raising=False)
        monkeypatch.setenv("FORCE_COLOR", "1")
        assert _detect_colour() is True

    def test_force_color_zero_is_ignored(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """``FORCE_COLOR=0`` is the documented opt-out; isatty decides."""
        monkeypatch.delenv("NO_COLOR", raising=False)
        monkeypatch.setenv("FORCE_COLOR", "0")
        # sys.stdout under pytest is a capturing stream — not a TTY.
        assert _detect_colour() is False


def test_supports_color_returns_cached_module_value() -> None:
    """``supports_color()`` returns the snapshot captured at import time."""
    assert supports_color() is _stage_module._COLOUR_ON


class TestStageLine:
    """Context manager: progressive rendering coupled to one call site."""

    def test_label_flushes_before_body_runs(
        self, plain: None, capsys: pytest.CaptureFixture[str]
    ) -> None:
        """The label reaches stdout as soon as the ``with`` block begins."""
        with stage_line("Vault") as s:
            # Before the body completes, only the label should be on stdout.
            mid = capsys.readouterr().out
            assert "Vault" in mid
            assert "\n" not in mid
            s.ok("reachable")
        final = capsys.readouterr().out
        assert "ok" in final and "(reachable)" in final

    def test_ok_emits_single_line_matching_stage(
        self, plain: None, capsys: pytest.CaptureFixture[str]
    ) -> None:
        """``with stage_line(L) as s: s.ok(D)`` ≡ ``stage(L, Marker.OK, D)``."""
        with stage_line("Vault") as s:
            s.ok("reachable")
        via_ctx = capsys.readouterr().out

        stage("Vault", Marker.OK, "reachable")
        via_call = capsys.readouterr().out

        assert via_ctx == via_call

    @pytest.mark.parametrize(
        ("method", "expected_token"),
        [("fail", "FAIL"), ("warn", "WARN"), ("missing", "MISSING"), ("skip", "skip")],
    )
    def test_setter_methods_pick_matching_marker(
        self,
        plain: None,
        capsys: pytest.CaptureFixture[str],
        method: str,
        expected_token: str,
    ) -> None:
        """Each setter writes its marker's string token."""
        with stage_line("X") as s:
            getattr(s, method)("detail")
        assert f" {expected_token} " in " " + capsys.readouterr().out + " "

    def test_last_setter_wins_on_same_line(
        self, plain: None, capsys: pytest.CaptureFixture[str]
    ) -> None:
        """A tentative ``ok`` overridden by a later ``fail`` renders only the fail."""
        with stage_line("X") as s:
            s.ok("optimistic")
            s.fail("actual error")
        out = capsys.readouterr().out
        assert "ok" not in out and "(optimistic)" not in out
        assert "FAIL" in out and "(actual error)" in out

    def test_unhandled_exception_auto_fails_and_propagates(
        self, plain: None, capsys: pytest.CaptureFixture[str]
    ) -> None:
        """An exception escaping the block completes the line FAIL and re-raises."""
        with pytest.raises(RuntimeError, match="boom"):
            with stage_line("X"):
                raise RuntimeError("boom")
        out = capsys.readouterr().out
        assert "FAIL" in out and "(boom)" in out

    def test_uncaught_exception_overrides_earlier_ok(
        self, plain: None, capsys: pytest.CaptureFixture[str]
    ) -> None:
        """An optimistic ``.ok()`` set before a raise is replaced by the exception's FAIL.

        Stops the "looks-good-then-raises" bug from rendering a
        misleading ok line — the exception's detail wins because the
        optimism turned out to be wrong.
        """
        with pytest.raises(RuntimeError, match="actually bad"):
            with stage_line("X") as s:
                s.ok("looks good")
                raise RuntimeError("actually bad")
        out = capsys.readouterr().out
        assert "ok" not in out and "(looks good)" not in out
        assert "FAIL" in out and "(actually bad)" in out

    def test_caller_fail_without_raise_keeps_caller_message(
        self, plain: None, capsys: pytest.CaptureFixture[str]
    ) -> None:
        """When the caller catches + ``.fail()``s + returns, the caller message wins.

        Complement to :meth:`test_uncaught_exception_overrides_earlier_ok` —
        showing the only way to put a caller-authored message in the
        log is to keep the exception out of the ``with`` block.
        """
        with stage_line("X") as s:
            try:
                raise RuntimeError("noisy stack")
            except RuntimeError as exc:
                s.fail(f"install: {exc}")
        out = capsys.readouterr().out
        assert "(install: noisy stack)" in out

    def test_no_marker_no_exception_is_a_loud_fail(
        self, plain: None, capsys: pytest.CaptureFixture[str]
    ) -> None:
        """A block that forgets to set a marker completes with a diagnostic FAIL."""
        with stage_line("X"):
            pass
        out = capsys.readouterr().out
        assert "FAIL" in out and "no marker set" in out

    def test_early_return_still_emits_the_line(
        self, plain: None, capsys: pytest.CaptureFixture[str]
    ) -> None:
        """``return`` from inside the block triggers ``__exit__`` with the stored marker."""

        def do() -> bool:
            with stage_line("X") as s:
                s.ok("done")
                return True
            return False  # type: ignore[unreachable]  # kept for assertion clarity

        assert do() is True
        out = capsys.readouterr().out
        assert "ok" in out and "(done)" in out
