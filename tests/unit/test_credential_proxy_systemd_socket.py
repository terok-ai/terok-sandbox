# SPDX-FileCopyrightText: 2026 Jiri Vyskocil
# SPDX-License-Identifier: Apache-2.0

"""Tests for the systemd socket activation helper in the proxy server."""

from __future__ import annotations

import os
from unittest.mock import patch

from terok_sandbox.credential_proxy.server import _systemd_socket


class TestSystemdSocket:
    """Verify _systemd_socket() sd_listen_fds protocol."""

    def test_returns_none_without_env(self) -> None:
        """Returns None when LISTEN_FDS is not set."""
        with patch.dict(os.environ, {}, clear=False):
            os.environ.pop("LISTEN_FDS", None)
            os.environ.pop("LISTEN_PID", None)
            assert _systemd_socket() is None

    def test_returns_none_wrong_pid(self) -> None:
        """Returns None when LISTEN_PID doesn't match."""
        with patch.dict(os.environ, {"LISTEN_FDS": "1", "LISTEN_PID": "99999999"}):
            assert _systemd_socket() is None

    def test_returns_none_wrong_fd_count(self) -> None:
        """Returns None when LISTEN_FDS is not '1'."""
        with patch.dict(os.environ, {"LISTEN_FDS": "0", "LISTEN_PID": str(os.getpid())}):
            assert _systemd_socket() is None
