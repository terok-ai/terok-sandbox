# SPDX-FileCopyrightText: 2026 Jiri Vyskocil
# SPDX-License-Identifier: Apache-2.0

"""Tests for the systemd socket activation helper in the token broker server."""

from __future__ import annotations

import os
import socket
from unittest.mock import patch

from terok_sandbox.vault.daemon.token_broker import _systemd_sockets


class TestSystemdSockets:
    """Verify _systemd_sockets() sd_listen_fds protocol."""

    def test_returns_none_pair_without_env(self) -> None:
        """Returns (None, None) when LISTEN_FDS is not set."""
        with patch.dict(os.environ, {}, clear=False):
            os.environ.pop("LISTEN_FDS", None)
            os.environ.pop("LISTEN_PID", None)
            assert _systemd_sockets() == (None, None)

    def test_returns_none_pair_wrong_pid(self) -> None:
        """Returns (None, None) when LISTEN_PID doesn't match."""
        with patch.dict(os.environ, {"LISTEN_FDS": "2", "LISTEN_PID": "99999999"}):
            assert _systemd_sockets() == (None, None)

    def test_returns_none_pair_zero_fds(self) -> None:
        """Returns (None, None) when LISTEN_FDS is '0'."""
        with patch.dict(os.environ, {"LISTEN_FDS": "0", "LISTEN_PID": str(os.getpid())}):
            assert _systemd_sockets() == (None, None)

    def test_returns_tcp_only_on_listen_fds_1(self) -> None:
        """Returns (None, tcp_sock) when the TCP-mode socket unit passes one TCP FD.

        Current ``terok-vault.socket`` declares a single ``ListenStream=
        127.0.0.1:<port>``, so ``LISTEN_FDS=1`` is the production shape.
        """
        saved: int | None = None
        try:
            saved = os.dup(3)
        except OSError:
            pass

        tcp_s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        try:
            os.dup2(tcp_s.fileno(), 3)
            with patch.dict(os.environ, {"LISTEN_FDS": "1", "LISTEN_PID": str(os.getpid())}):
                sd_unix, sd_tcp = _systemd_sockets()
            assert sd_unix is None
            assert sd_tcp is not None
            assert sd_tcp.fileno() == 3
            assert sd_tcp.family == socket.AF_INET
            assert sd_tcp.getblocking() is False
            sd_tcp.detach()
        finally:
            os.close(3)
            tcp_s.close()
            if saved is not None:
                os.dup2(saved, 3)
                os.close(saved)

    def test_returns_both_sockets_on_listen_fds_2(self) -> None:
        """Legacy 2-FD case (Unix + TCP) keeps working for in-flight rendered units.

        Until a host re-runs ``terok setup`` after upgrading sandbox, the
        installed unit may still emit the legacy dual ``ListenStream`` pair.
        The daemon classifies each inherited FD by family so the unit shape
        and the broker's expectations don't have to be wire-locked across
        the release.
        """
        # Create real sockets on FDs 3 and 4 so socket.socket(fileno=N) works.
        # Save whatever currently lives on FDs 3/4 to restore afterwards.
        saved: dict[int, int] = {}
        for fd in (3, 4):
            try:
                saved[fd] = os.dup(fd)
            except OSError:
                pass  # FD not open — nothing to save

        unix_s = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
        tcp_s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        try:
            os.dup2(unix_s.fileno(), 3)
            os.dup2(tcp_s.fileno(), 4)
            with patch.dict(os.environ, {"LISTEN_FDS": "2", "LISTEN_PID": str(os.getpid())}):
                sd_unix, sd_tcp = _systemd_sockets()
            assert sd_unix is not None
            assert sd_tcp is not None
            assert sd_unix.fileno() == 3
            assert sd_tcp.fileno() == 4
            assert sd_unix.family == socket.AF_UNIX
            assert sd_tcp.family == socket.AF_INET
            assert sd_unix.getblocking() is False
            assert sd_tcp.getblocking() is False
            # Detach so close() below doesn't double-close FD 3/4
            sd_unix.detach()
            sd_tcp.detach()
        finally:
            os.close(3)
            os.close(4)
            unix_s.close()
            tcp_s.close()
            for fd, dup_fd in saved.items():
                os.dup2(dup_fd, fd)
                os.close(dup_fd)
