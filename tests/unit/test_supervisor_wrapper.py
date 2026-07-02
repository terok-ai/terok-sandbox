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
        # Backoff is applied between attempts but NOT after the final
        # failure — sleeping there just delays the inevitable exit.
        assert sleep_mock.call_count == 4

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


class TestArgvValidation:
    """Untrusted argv is gated before it reaches the supervisor CLI.

    The wrapper trusts whatever argv it is handed, so a container_id or
    sidecar_path that starts with ``-`` (which the ``supervisor`` verb's
    argparse would read as an option — CWE-88 argument injection) or a
    relative sidecar path is refused before any ``subprocess.call``.
    """

    @pytest.mark.parametrize(
        "bad_id",
        ["-x", "--config=/evil", "", "a b", "a;b", "../x"],
        ids=["dash", "long-opt", "empty", "space", "semicolon", "traversal"],
    )
    def test_unsafe_container_id_refused(
        self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path, bad_id: str
    ) -> None:
        """Anything that is not a safe single token is refused, never spawned."""
        mod = _render_wrapper(tmp_path, ["/bin/terok-sandbox"])
        with patch.object(mod.subprocess, "call") as call_mock:
            monkeypatch.setattr(mod.sys, "argv", ["supervisor_wrapper.py", bad_id, "/sidecar.json"])
            assert mod.main() == 2
        call_mock.assert_not_called()

    def test_relative_sidecar_path_refused(
        self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
    ) -> None:
        """A non-absolute sidecar path is refused before any spawn."""
        mod = _render_wrapper(tmp_path, ["/bin/terok-sandbox"])
        with patch.object(mod.subprocess, "call") as call_mock:
            monkeypatch.setattr(
                mod.sys, "argv", ["supervisor_wrapper.py", "abc123", "relative/sidecar.json"]
            )
            assert mod.main() == 2
        call_mock.assert_not_called()

    def test_valid_id_and_absolute_path_reach_cli_unchanged(
        self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
    ) -> None:
        """A normal podman id + absolute path pass straight through (no ``--``).

        The CLI reserves ``--`` for ``run``, so the wrapper must hand the
        positionals bare — validation, not a separator, is what keeps them
        from being misparsed.
        """
        mod = _render_wrapper(tmp_path, ["/bin/terok-sandbox"])
        with patch.object(mod.subprocess, "call", return_value=0) as call_mock:
            monkeypatch.setattr(
                mod.sys,
                "argv",
                ["supervisor_wrapper.py", "9f8e7d6c5b4a", "/state/sidecar/x.json"],
            )
            assert mod.main() == 0
        call_mock.assert_called_once_with(
            ["/bin/terok-sandbox", "supervisor", "9f8e7d6c5b4a", "/state/sidecar/x.json"]
        )
