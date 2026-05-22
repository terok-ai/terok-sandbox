# SPDX-FileCopyrightText: 2026 Jiri Vyskocil
# SPDX-License-Identifier: Apache-2.0

"""Tests for sandbox-specific security hardening.

The strict ``render_template`` validation and the
``write_sensitive_file`` 0o600 behaviour are exercised in
``terok-util``'s own test suite — they live there now.  The
``systemd_user_unit_dir`` helper stays in sandbox because the
user-systemd story is a sandbox concern.
"""

from __future__ import annotations

import os
from pathlib import Path
from unittest.mock import patch

import pytest

from terok_sandbox._util._fs import systemd_user_unit_dir


class TestSystemdUserUnitDir:
    """Issue #152: _systemd_unit_dir must validate XDG_CONFIG_HOME and refuse root."""

    def test_refuses_root(self) -> None:
        """Raises SystemExit when running as root."""
        with (
            patch("os.geteuid", return_value=0),
            pytest.raises(SystemExit, match="root"),
        ):
            systemd_user_unit_dir()

    def test_rejects_path_outside_home(self) -> None:
        """Raises SystemExit when XDG_CONFIG_HOME is outside $HOME."""
        with (
            patch("os.geteuid", return_value=1000),
            patch.dict(os.environ, {"XDG_CONFIG_HOME": "/etc/evil"}),
            pytest.raises(SystemExit, match="outside the home directory"),
        ):
            systemd_user_unit_dir()

    def test_default_path(self) -> None:
        """Falls back to ~/.config/systemd/user when XDG_CONFIG_HOME is unset."""
        env = {k: v for k, v in os.environ.items() if k != "XDG_CONFIG_HOME"}
        with (
            patch("os.geteuid", return_value=1000),
            patch.dict(os.environ, env, clear=True),
        ):
            result = systemd_user_unit_dir()
        assert result == Path.home() / ".config" / "systemd" / "user"

    def test_valid_xdg_under_home(self, tmp_path: Path) -> None:
        """Accepts XDG_CONFIG_HOME that resolves under $HOME."""
        xdg = tmp_path / "my-config"
        xdg.mkdir()
        with (
            patch("os.geteuid", return_value=1000),
            patch.dict(os.environ, {"XDG_CONFIG_HOME": str(xdg)}),
            patch("pathlib.Path.home", return_value=tmp_path),
        ):
            result = systemd_user_unit_dir()
        assert result == xdg / "systemd" / "user"
