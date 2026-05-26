# SPDX-FileCopyrightText: 2026 Jiri Vyskocil
# SPDX-License-Identifier: Apache-2.0

"""Tests for the restart-loop wrapper.

The wrapper is shipped as a stdlib-only Python script with the
``terok-sandbox`` argv baked in at install time.  These tests render
the template with a known argv and import the rendered module to
exercise ``main()`` directly without spawning a real supervisor.
"""

from __future__ import annotations

import importlib.util
from pathlib import Path
from unittest.mock import patch

import pytest


def _render_wrapper(tmp_path: Path, sandbox_argv: list[str]) -> object:
    """Render the wrapper template with *sandbox_argv* baked in and import it."""
    import json

    src_template = (
        Path(__file__).resolve().parents[2]
        / "src"
        / "terok_sandbox"
        / "resources"
        / "supervisor_wrapper.py"
    )
    rendered = src_template.read_text().replace(
        '["__TEROK_SANDBOX_BIN__"]', json.dumps(sandbox_argv)
    )
    dst = tmp_path / "supervisor_wrapper.py"
    dst.write_text(rendered)
    spec = importlib.util.spec_from_file_location("supervisor_wrapper", dst)
    assert spec is not None
    assert spec.loader is not None
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


class TestRestartLoop:
    """Backoff + capped retry contract on supervisor exit codes."""

    def test_clean_exit_short_circuits(
        self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
    ) -> None:
        """``rc == 0`` from the first call stops the loop immediately."""
        mod = _render_wrapper(tmp_path, ["/bin/terok-sandbox"])
        with patch.object(mod.subprocess, "call", return_value=0) as call_mock:
            monkeypatch.setattr(
                mod.sys, "argv", ["supervisor_wrapper.py", "abc123", "/sidecar.json"]
            )
            assert mod.main() == 0
        call_mock.assert_called_once_with(
            ["/bin/terok-sandbox", "supervisor", "abc123", "/sidecar.json"]
        )

    def test_retries_then_gives_up(self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
        """Five non-zero exits return the last rc and stop retrying."""
        mod = _render_wrapper(tmp_path, ["/bin/terok-sandbox"])
        with (
            patch.object(mod.subprocess, "call", return_value=2) as call_mock,
            patch.object(mod.time, "sleep") as sleep_mock,
        ):
            monkeypatch.setattr(
                mod.sys, "argv", ["supervisor_wrapper.py", "abc123", "/sidecar.json"]
            )
            assert mod.main() == 2
        assert call_mock.call_count == 5
        # Backoff was applied between attempts.
        assert sleep_mock.call_count == 5

    def test_missing_argv_returns_error(
        self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
    ) -> None:
        """Invocation without container_id + sidecar_path is a hard fail."""
        mod = _render_wrapper(tmp_path, ["/bin/terok-sandbox"])
        monkeypatch.setattr(mod.sys, "argv", ["supervisor_wrapper.py", "abc"])
        assert mod.main() == 2

    def test_unrendered_template_refuses_to_run(
        self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
    ) -> None:
        """Running the unrendered template (placeholder intact) is a hard fail."""
        # Load the raw template (no replacement) — the placeholder must
        # trip the guard in main() before any subprocess.call happens.
        src_template = (
            Path(__file__).resolve().parents[2]
            / "src"
            / "terok_sandbox"
            / "resources"
            / "supervisor_wrapper.py"
        )
        spec = importlib.util.spec_from_file_location("supervisor_wrapper_raw", src_template)
        assert spec is not None
        assert spec.loader is not None
        mod = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(mod)
        monkeypatch.setattr(mod.sys, "argv", ["supervisor_wrapper.py", "abc", "/sidecar.json"])
        assert mod.main() == 2
