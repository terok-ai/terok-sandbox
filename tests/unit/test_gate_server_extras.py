# SPDX-FileCopyrightText: 2026 Jiri Vyskocil
# SPDX-License-Identifier: Apache-2.0

"""Tests for gate server helpers: TokenStore, basic-auth parsing, CGI plumbing."""

from __future__ import annotations

import base64
import io
import logging
import subprocess
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from terok_sandbox.gate.server import (
    _ADMIN_WILDCARD,  # noqa: PLC2701
    TokenStore,
    _build_cgi_env,  # noqa: PLC2701
    _configure_logging,  # noqa: PLC2701
    _extract_basic_auth_token,  # noqa: PLC2701
    _parse_cgi_headers,  # noqa: PLC2701
    _parse_content_length,  # noqa: PLC2701
    _stream_request_body,  # noqa: PLC2701
    _stream_response_body,  # noqa: PLC2701
    _validate_token_data,  # noqa: PLC2701
    main,
)

# ---------------------------------------------------------------------------
# _validate_token_data
# ---------------------------------------------------------------------------


class TestValidateTokenData:
    """_validate_token_data accepts only well-formed {token: {scope, task}} maps."""

    def test_valid_returns_unchanged(self) -> None:
        data = {"tok-1": {"scope": "myproj", "task": "t1"}}
        assert _validate_token_data(data) == data

    def test_non_dict_returns_empty(self) -> None:
        assert _validate_token_data([1, 2, 3]) == {}
        assert _validate_token_data(None) == {}
        assert _validate_token_data("not-a-dict") == {}

    def test_filters_invalid_scope_or_task(self) -> None:
        data = {
            "ok": {"scope": "p", "task": "t"},
            "no-scope": {"task": "t"},
            "non-string-scope": {"scope": 1, "task": "t"},
            "missing-task": {"scope": "p"},
        }
        result = _validate_token_data(data)
        assert "ok" in result
        assert "no-scope" not in result
        assert "non-string-scope" not in result
        assert "missing-task" not in result


# ---------------------------------------------------------------------------
# TokenStore
# ---------------------------------------------------------------------------


class TestTokenStore:
    """TokenStore validates tokens against a JSON-backed registry."""

    def _write(self, tmp_path: Path, content: str) -> Path:
        p = tmp_path / "tokens.json"
        p.write_text(content)
        return p

    def test_admin_token_returns_wildcard(self, tmp_path: Path) -> None:
        store = TokenStore(self._write(tmp_path, "{}"), admin_token="root-token")
        assert store.validate("root-token") == _ADMIN_WILDCARD

    def test_unknown_token_returns_none(self, tmp_path: Path) -> None:
        store = TokenStore(self._write(tmp_path, '{"good": {"scope": "s", "task": "t"}}'))
        assert store.validate("bogus") is None

    def test_known_token_returns_scope(self, tmp_path: Path) -> None:
        store = TokenStore(self._write(tmp_path, '{"good": {"scope": "myscope", "task": "t1"}}'))
        assert store.validate("good") == "myscope"

    def test_missing_file_validates_to_none(self, tmp_path: Path) -> None:
        store = TokenStore(tmp_path / "missing.json")
        assert store.validate("any") is None

    def test_corrupt_json_validates_to_none(self, tmp_path: Path) -> None:
        store = TokenStore(self._write(tmp_path, "{not json"))
        assert store.validate("any") is None

    def test_reload_picks_up_new_token_after_mtime_change(self, tmp_path: Path) -> None:
        """File modified after first read → reload triggers on next validate."""
        path = self._write(tmp_path, '{"a": {"scope": "x", "task": "t"}}')
        store = TokenStore(path)
        assert store.validate("a") == "x"
        # Rewrite the file with a different token; bump mtime.
        path.write_text('{"b": {"scope": "y", "task": "t"}}')
        import os

        os.utime(path, (1_700_000_000, 1_700_000_000))
        assert store.validate("a") is None
        assert store.validate("b") == "y"


# ---------------------------------------------------------------------------
# _extract_basic_auth_token
# ---------------------------------------------------------------------------


class TestExtractBasicAuthToken:
    """Parse Basic auth username out of an Authorization header."""

    def test_valid_header(self) -> None:
        b64 = base64.b64encode(b"my-token:ignored").decode()
        assert _extract_basic_auth_token(f"Basic {b64}") == "my-token"

    def test_none_header_returns_none(self) -> None:
        assert _extract_basic_auth_token(None) is None

    def test_non_basic_header_returns_none(self) -> None:
        assert _extract_basic_auth_token("Bearer abc") is None

    def test_invalid_base64_returns_none(self) -> None:
        assert _extract_basic_auth_token("Basic !!!not-base64!!!") is None

    def test_missing_colon_returns_none(self) -> None:
        b64 = base64.b64encode(b"no-colon-here").decode()
        assert _extract_basic_auth_token(f"Basic {b64}") is None

    def test_empty_username_returns_none(self) -> None:
        b64 = base64.b64encode(b":only-password").decode()
        assert _extract_basic_auth_token(f"Basic {b64}") is None


# ---------------------------------------------------------------------------
# _parse_content_length
# ---------------------------------------------------------------------------


class TestParseContentLength:
    """Validate Content-Length header values; return (length, error)."""

    def test_none_returns_zero(self) -> None:
        assert _parse_content_length(None) == (0, None)

    def test_empty_returns_zero(self) -> None:
        assert _parse_content_length("") == (0, None)

    def test_valid_returns_int(self) -> None:
        assert _parse_content_length("42") == (42, None)

    def test_negative_returns_error(self) -> None:
        length, err = _parse_content_length("-1")
        assert length == 0
        assert err and "Invalid" in err

    def test_non_integer_returns_error(self) -> None:
        length, err = _parse_content_length("not-a-number")
        assert length == 0
        assert err and "Invalid" in err


# ---------------------------------------------------------------------------
# _build_cgi_env
# ---------------------------------------------------------------------------


class TestBuildCgiEnv:
    """_build_cgi_env composes a minimal CGI environment for git http-backend."""

    def test_basic_env(self, tmp_path: Path) -> None:
        env = _build_cgi_env(
            base_path=tmp_path,
            path_info="/repo.git/info/refs",
            query_string="service=git-upload-pack",
            method="GET",
            content_type="",
            protocol="HTTP/1.1",
            content_length=0,
        )
        assert env["GIT_PROJECT_ROOT"] == str(tmp_path)
        assert env["PATH_INFO"] == "/repo.git/info/refs"
        assert env["QUERY_STRING"] == "service=git-upload-pack"
        assert env["REQUEST_METHOD"] == "GET"
        assert env["GIT_HTTP_EXPORT_ALL"] == "1"
        assert env["REMOTE_USER"] == "token"
        # Hooks defense-in-depth
        assert env["GIT_CONFIG_KEY_0"] == "core.hooksPath"
        assert env["GIT_CONFIG_VALUE_0"] == "/dev/null"
        # No CONTENT_LENGTH when 0
        assert "CONTENT_LENGTH" not in env

    def test_inherits_path_and_home(self, tmp_path: Path) -> None:
        with patch.dict(
            "os.environ",
            {"PATH": "/custom/bin", "HOME": "/custom/home", "GIT_EXEC_PATH": "/git/exec"},
        ):
            env = _build_cgi_env(tmp_path, "/", "", "GET", "", "HTTP/1.1", 0)
        assert env["PATH"] == "/custom/bin"
        assert env["HOME"] == "/custom/home"
        assert env["GIT_EXEC_PATH"] == "/git/exec"

    def test_content_length_added_when_truthy(self, tmp_path: Path) -> None:
        env = _build_cgi_env(tmp_path, "/", "", "POST", "", "HTTP/1.1", 7)
        assert env["CONTENT_LENGTH"] == "7"

    def test_http_headers_forwarded_when_non_empty(self, tmp_path: Path) -> None:
        env = _build_cgi_env(
            tmp_path,
            "/",
            "",
            "POST",
            "",
            "HTTP/1.1",
            0,
            http_headers={"HTTP_GIT_PROTOCOL": "version=2", "HTTP_CONTENT_ENCODING": ""},
        )
        assert env["HTTP_GIT_PROTOCOL"] == "version=2"
        # Empty values are skipped, not forwarded.
        assert "HTTP_CONTENT_ENCODING" not in env


# ---------------------------------------------------------------------------
# _stream_request_body
# ---------------------------------------------------------------------------


class TestStreamRequestBody:
    """Stream up to N bytes from rfile to stdin; tolerate broken pipes."""

    def test_zero_bytes_is_noop(self) -> None:
        rfile = MagicMock()
        stdin = MagicMock()
        _stream_request_body(rfile, stdin, 0)
        rfile.read.assert_not_called()
        stdin.write.assert_not_called()

    def test_streams_chunks(self) -> None:
        rfile = io.BytesIO(b"x" * 20000)  # > one 8k chunk
        stdin = io.BytesIO()
        _stream_request_body(rfile, stdin, 20000)
        assert stdin.getvalue() == b"x" * 20000

    def test_short_read_breaks_loop(self) -> None:
        """When rfile returns empty bytes early, streaming halts."""
        rfile = io.BytesIO(b"abc")  # only 3 bytes available
        stdin = io.BytesIO()
        _stream_request_body(rfile, stdin, 100)  # asked for 100, got 3
        assert stdin.getvalue() == b"abc"

    def test_broken_pipe_tolerated(self) -> None:
        """If the CGI process closes stdin early, BrokenPipeError is swallowed."""
        rfile = io.BytesIO(b"data")
        stdin = MagicMock()
        stdin.write.side_effect = BrokenPipeError()
        # Must not raise
        _stream_request_body(rfile, stdin, 4)


# ---------------------------------------------------------------------------
# _parse_cgi_headers
# ---------------------------------------------------------------------------


class TestParseCgiHeaders:
    """Parse CGI response headers up to the blank-line separator."""

    def test_default_status_is_200(self) -> None:
        stdout = io.BytesIO(b"Content-Type: text/plain\r\n\r\nbody")
        status, headers = _parse_cgi_headers(stdout)
        assert status == 200
        assert ("Content-Type", "text/plain") in headers

    def test_explicit_status_header(self) -> None:
        stdout = io.BytesIO(b"Status: 404 Not Found\r\nContent-Type: text/plain\r\n\r\n")
        status, _ = _parse_cgi_headers(stdout)
        assert status == 404

    def test_malformed_status_falls_back_to_200(self) -> None:
        """A non-numeric Status line is tolerated; status defaults to 200."""
        stdout = io.BytesIO(b"Status: gibberish\r\n\r\n")
        status, _ = _parse_cgi_headers(stdout)
        assert status == 200

    def test_empty_status_line_falls_back_to_200(self) -> None:
        stdout = io.BytesIO(b"Status:\r\n\r\n")
        status, _ = _parse_cgi_headers(stdout)
        assert status == 200

    def test_lf_only_separator(self) -> None:
        stdout = io.BytesIO(b"Content-Type: text/plain\n\nbody")
        status, headers = _parse_cgi_headers(stdout)
        assert status == 200
        assert ("Content-Type", "text/plain") in headers

    def test_lines_without_colon_are_skipped(self) -> None:
        stdout = io.BytesIO(b"Content-Type: text/plain\r\nNo-Colon-Line\r\n\r\n")
        _, headers = _parse_cgi_headers(stdout)
        assert ("Content-Type", "text/plain") in headers
        # The malformed line must not have been parsed as a header at all.
        assert all(name != "No-Colon-Line" for name, _val in headers)


# ---------------------------------------------------------------------------
# _stream_response_body
# ---------------------------------------------------------------------------


class TestStreamResponseBody:
    """Stream chunks from CGI stdout to wfile until EOF."""

    def test_streams_until_eof(self) -> None:
        stdout = io.BytesIO(b"hello" * 10000)
        wfile = io.BytesIO()
        _stream_response_body(stdout, wfile)
        assert wfile.getvalue() == b"hello" * 10000

    def test_empty_stdout_is_noop(self) -> None:
        wfile = io.BytesIO()
        _stream_response_body(io.BytesIO(), wfile)
        assert wfile.getvalue() == b""


# ---------------------------------------------------------------------------
# _configure_logging
# ---------------------------------------------------------------------------


class TestConfigureLogging:
    """_configure_logging picks the right handler for daemon vs interactive use."""

    def teardown_method(self) -> None:
        # Drain handlers so tests don't pollute each other or other suites.
        from terok_sandbox.gate.server import _logger as gate_logger

        gate_logger.handlers.clear()

    def test_non_daemon_uses_streamhandler(self) -> None:
        from terok_sandbox.gate.server import _logger as gate_logger

        _configure_logging(daemon=False)
        assert any(isinstance(h, logging.StreamHandler) for h in gate_logger.handlers)
        assert gate_logger.level == logging.WARNING

    def test_daemon_uses_syslog_handler(self) -> None:
        # Patch SysLogHandler so we don't open /dev/log on the test host.
        from terok_sandbox.gate.server import _logger as gate_logger

        with patch("logging.handlers.SysLogHandler") as syslog_cls:
            syslog_cls.return_value = logging.NullHandler()
            _configure_logging(daemon=True)
            syslog_cls.assert_called_once_with(address="/dev/log")
        # The NullHandler we substituted in is what got attached.
        assert gate_logger.handlers


# ---------------------------------------------------------------------------
# main() — argparse + mode dispatch
# ---------------------------------------------------------------------------


def _make_token_file(tmp_path: Path) -> Path:
    """Create an empty tokens.json under *tmp_path* and return its path."""
    token_file = tmp_path / "tokens.json"
    token_file.write_text("{}")
    return token_file


class TestMainCliDispatch:
    """main() validates flag combinations and dispatches to the right serve fn."""

    def teardown_method(self) -> None:
        from terok_sandbox.gate.server import _logger as gate_logger

        gate_logger.handlers.clear()

    def test_inetd_dispatches_to_serve_inetd(self, tmp_path: Path) -> None:
        token_file = _make_token_file(tmp_path)
        with (
            patch(
                "sys.argv",
                [
                    "terok-gate",
                    "--base-path",
                    str(tmp_path),
                    "--token-file",
                    str(token_file),
                    "--inetd",
                ],
            ),
            patch("terok_sandbox.gate.server._serve_inetd") as inetd,
            patch("terok_sandbox.gate.server._configure_logging"),
        ):
            main()
        inetd.assert_called_once()

    def test_socket_path_dispatches_to_serve_foreground(self, tmp_path: Path) -> None:
        token_file = _make_token_file(tmp_path)
        sock_path = tmp_path / "g.sock"
        with (
            patch(
                "sys.argv",
                [
                    "terok-gate",
                    "--base-path",
                    str(tmp_path),
                    "--token-file",
                    str(token_file),
                    "--socket-path",
                    str(sock_path),
                    "--port",
                    "9418",
                    "--pid-file",
                    str(tmp_path / "g.pid"),
                ],
            ),
            patch("terok_sandbox.gate.server._serve_foreground") as fore,
            patch("terok_sandbox.gate.server._configure_logging"),
        ):
            main()
        fore.assert_called_once()
        kwargs = fore.call_args.kwargs
        assert kwargs["socket_path"] == sock_path
        assert kwargs["port"] == 9418

    def test_detach_dispatches_to_serve_daemon(self, tmp_path: Path) -> None:
        token_file = _make_token_file(tmp_path)
        with (
            patch(
                "sys.argv",
                [
                    "terok-gate",
                    "--base-path",
                    str(tmp_path),
                    "--token-file",
                    str(token_file),
                    "--detach",
                ],
            ),
            patch("terok_sandbox.gate.server._serve_daemon") as daemon,
            patch("terok_sandbox.gate.server._configure_logging"),
        ):
            main()
        daemon.assert_called_once()

    def test_no_mode_flag_errors_out(self, tmp_path: Path) -> None:
        """Without --inetd, --detach, or --socket-path, argparse errors."""
        token_file = _make_token_file(tmp_path)
        with (
            patch(
                "sys.argv",
                ["terok-gate", "--base-path", str(tmp_path), "--token-file", str(token_file)],
            ),
            patch("terok_sandbox.gate.server._configure_logging"),
        ):
            with pytest.raises(SystemExit):
                main()

    def test_mutually_exclusive_modes_error_out(self, tmp_path: Path) -> None:
        token_file = _make_token_file(tmp_path)
        with (
            patch(
                "sys.argv",
                [
                    "terok-gate",
                    "--base-path",
                    str(tmp_path),
                    "--token-file",
                    str(token_file),
                    "--inetd",
                    "--detach",
                ],
            ),
            patch("terok_sandbox.gate.server._configure_logging"),
        ):
            with pytest.raises(SystemExit):
                main()

    def test_admin_token_via_env_var(self, tmp_path: Path) -> None:
        """admin token can come from TEROK_GATE_ADMIN_TOKEN env var as fallback."""
        token_file = _make_token_file(tmp_path)
        with (
            patch(
                "sys.argv",
                [
                    "terok-gate",
                    "--base-path",
                    str(tmp_path),
                    "--token-file",
                    str(token_file),
                    "--inetd",
                ],
            ),
            patch.dict("os.environ", {"TEROK_GATE_ADMIN_TOKEN": "env-admin"}),
            patch("terok_sandbox.gate.server._serve_inetd"),
            patch("terok_sandbox.gate.server._configure_logging"),
            patch("terok_sandbox.gate.server.TokenStore") as store_cls,
        ):
            main()
        # TokenStore should have been built with the admin token from env
        assert store_cls.call_args.kwargs["admin_token"] == "env-admin"


# ---------------------------------------------------------------------------
# _run_cgi error paths via mocked Popen — regression for line 354-359, 375-377
# ---------------------------------------------------------------------------


class TestRunCgiErrors:
    """_run_cgi handles git-missing, OSError, and CGI timeout via mocked subprocess."""

    def _build_handler(self, tmp_path: Path):
        """Construct a GateRequestHandler instance bound to mocked I/O."""
        from terok_sandbox.gate.server import _make_handler_class

        token_file = tmp_path / "tokens.json"
        token_file.write_text('{"good-token": {"scope": "myrepo", "task": "t1"}}')
        store = TokenStore(token_file)

        cls = _make_handler_class(tmp_path, store)
        handler = cls.__new__(cls)
        handler.headers = {}
        handler.command = "GET"
        handler.request_version = "HTTP/1.1"
        handler.path = "/myrepo.git/info/refs?service=git-upload-pack"
        handler.rfile = io.BytesIO(b"")
        handler.wfile = io.BytesIO()

        # Monkeypatch BaseHTTPRequestHandler methods we don't want exercising real I/O.
        handler.send_error = MagicMock()
        handler.send_response = MagicMock()
        handler.send_header = MagicMock()
        handler.end_headers = MagicMock()
        return handler

    def test_git_not_found_emits_500(self, tmp_path: Path) -> None:
        handler = self._build_handler(tmp_path)
        with patch("subprocess.Popen", side_effect=FileNotFoundError):
            handler._run_cgi("/myrepo.git/info/refs", "")
        handler.send_error.assert_called_once_with(500, "git not found")

    def test_oserror_emits_500(self, tmp_path: Path) -> None:
        handler = self._build_handler(tmp_path)
        with patch("subprocess.Popen", side_effect=OSError("ENOMEM")):
            handler._run_cgi("/myrepo.git/info/refs", "")
        handler.send_error.assert_called_once_with(500, "git http-backend unavailable")

    def test_invalid_content_length_emits_400(self, tmp_path: Path) -> None:
        handler = self._build_handler(tmp_path)
        handler.headers = {"Content-Length": "garbage"}
        # Popen should not be called because we error before launching
        with patch("subprocess.Popen") as popen:
            handler._run_cgi("/myrepo.git/info/refs", "")
        handler.send_error.assert_called_once_with(400, "Invalid Content-Length")
        popen.assert_not_called()

    def test_cgi_wait_timeout_kills_process(self, tmp_path: Path) -> None:
        handler = self._build_handler(tmp_path)
        # Popen returns a fake proc whose wait() always times out.
        proc = MagicMock()
        proc.stdin = MagicMock()
        proc.stdout = io.BytesIO(b"\r\nbody")  # no headers, just blank line
        proc.stderr = io.BytesIO(b"")
        proc.wait.side_effect = [subprocess.TimeoutExpired("git", 30), None]
        with patch("subprocess.Popen", return_value=proc):
            handler._run_cgi("/myrepo.git/info/refs", "")
        proc.kill.assert_called_once()
