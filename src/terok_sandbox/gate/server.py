# SPDX-FileCopyrightText: 2026 Jiri Vyskocil
# SPDX-License-Identifier: Apache-2.0

"""HTTP gate component wrapping ``git http-backend`` with token auth.

Composed by the per-container supervisor as one of its services.  The
gate serves a single task's repo out of the shared per-project bare
mirror, gated on a single minted token.

Token validation:
    Each request must carry HTTP Basic Auth with the token as the username
    (password is ignored).  The supervisor minted exactly one token for the
    task this container serves; the requested repo must match the token's
    scope.

Transport:
    The supervisor binds the gate on a per-container Unix socket inside
    ``container_runtime_dir`` (= the in-container ``/run/terok``) in socket
    mode, or on a per-container ``127.0.0.1`` TCP port in TCP mode.
"""

from __future__ import annotations

import base64
import logging
import os
import re
import socket
import stat
import subprocess  # nosec B404 — spawning git http-backend (CGI)
import threading
from http.server import BaseHTTPRequestHandler, HTTPServer
from pathlib import Path
from socketserver import ThreadingMixIn
from typing import IO, Any, cast

from .._util._selinux import socket_selinux_context

_logger = logging.getLogger("terok-gate")

# ---------------------------------------------------------------------------
# Token store — single-token validation, no terok imports
# ---------------------------------------------------------------------------

_ROUTE = re.compile(
    r"^/(?P<repo>[A-Za-z0-9._-]+\.git)"
    r"(?P<path>/info/refs|/git-upload-pack|/git-receive-pack|/HEAD)$"
)

_CGI_WAIT_TIMEOUT = 30


_ADMIN_WILDCARD = "*"
"""Sentinel a token store may return to grant access to **all** repos.

The per-container gate never mints an admin token, but the request
handler still honours the sentinel so the audited routing logic stays
uniform across token stores.
"""


class _SingleTokenStore:
    """Validate requests against the one token the supervisor minted.

    The per-container gate serves a single task, so there is no
    ``tokens.json`` file to read: the supervisor passes the minted token
    and its scope directly.  ``validate`` returns the scope iff the
    presented token matches, mirroring the file-backed store's contract
    so [`_make_handler_class`][terok_sandbox.gate.server._make_handler_class]
    needs no changes.
    """

    def __init__(self, token: str, scope: str) -> None:
        """Bind the store to the single *token* and its *scope*."""
        self._token = token
        self._scope = scope

    def validate(self, token: str) -> str | None:
        """Return the scope if *token* matches the minted token, else ``None``."""
        return self._scope if token == self._token else None


# ---------------------------------------------------------------------------
# Module-level helpers — extracted to reduce handler cognitive complexity
# ---------------------------------------------------------------------------


def _extract_basic_auth_token(auth_header: str | None) -> str | None:
    """Parse ``Authorization: Basic`` header, return username (token)."""
    if not auth_header or not auth_header.startswith("Basic "):
        return None
    try:
        decoded = base64.b64decode(auth_header[6:]).decode("utf-8")
    except Exception:
        return None
    if ":" not in decoded:
        return None
    username, _password = decoded.split(":", 1)
    return username or None


def _parse_content_length(header: str | None) -> tuple[int, str | None]:
    """Validate a Content-Length header value.

    Returns ``(length, None)`` on success or ``(0, error_message)`` on failure.
    """
    if not header:
        return 0, None
    try:
        length = int(header)
        if length < 0:
            raise ValueError("negative")
    except ValueError:
        return 0, "Invalid Content-Length"
    return length, None


def _build_cgi_env(
    base_path: Path,
    path_info: str,
    query_string: str,
    method: str,
    content_type: str,
    protocol: str,
    content_length: int,
    hooks_path: Path | None,
    http_headers: dict[str, str] | None = None,
) -> dict[str, str]:
    """Build the CGI environment for ``git http-backend``.

    Inherits ``PATH`` and ``HOME`` from the parent process so that
    ``git http-backend`` can locate git sub-commands (e.g. ``git-upload-pack``)
    and read user config.

    ``core.hooksPath`` is pinned to *hooks_path* — the sandbox-owned
    directory outside every gate repo (see
    [`install_hooks`][terok_sandbox.gate.hooks.install_hooks]) — or to
    ``/dev/null`` when the composer supplied none.  Either way the
    defense-in-depth property holds: hooks can never originate from repo
    content.

    *http_headers* maps CGI variable names (e.g. ``HTTP_CONTENT_ENCODING``) to
    their values.  Only non-empty values are included.
    """
    env: dict[str, str] = {}
    # Inherit essential system variables
    for key in ("PATH", "HOME", "GIT_EXEC_PATH"):
        val = os.environ.get(key)
        if val is not None:
            env[key] = val
    env.update(
        {
            "GIT_PROJECT_ROOT": str(base_path),
            "GIT_HTTP_EXPORT_ALL": "1",
            "PATH_INFO": path_info,
            "QUERY_STRING": query_string,
            "REQUEST_METHOD": method,
            "CONTENT_TYPE": content_type,
            "SERVER_PROTOCOL": protocol,
            "REMOTE_USER": "token",
            # Defense in depth: hooks come only from the sandbox-owned dir,
            # never from repo content.
            "GIT_CONFIG_KEY_0": "core.hooksPath",
            "GIT_CONFIG_VALUE_0": str(hooks_path) if hooks_path else "/dev/null",
            "GIT_CONFIG_COUNT": "1",
        }
    )
    if content_length:
        env["CONTENT_LENGTH"] = str(content_length)
    if http_headers:
        for cgi_key, val in http_headers.items():
            if val:
                env[cgi_key] = val
    return env


def _stream_request_body(rfile: Any, stdin: IO[bytes], remaining: int) -> None:
    """Stream *remaining* bytes from *rfile* to CGI *stdin*."""
    if remaining <= 0:
        return
    try:
        while remaining > 0:
            chunk = rfile.read(min(remaining, 8192))
            if not chunk:
                break
            stdin.write(chunk)
            remaining -= len(chunk)
    except BrokenPipeError:
        pass  # CGI process closed stdin early


def _parse_cgi_headers(stdout: IO[bytes]) -> tuple[int, list[tuple[str, str]]]:
    """Read CGI response headers from *stdout*.

    Returns ``(status_code, [(header_name, header_value), ...])``.
    """
    status_code = 200
    headers: list[tuple[str, str]] = []
    while True:
        line = stdout.readline()
        if not line or line in (b"\r\n", b"\n"):
            break
        header_line = line.decode("utf-8", errors="replace").rstrip("\r\n")
        if header_line.startswith("Status:"):
            try:
                status_code = int(header_line.split(":", 1)[1].strip().split()[0])
            except (ValueError, IndexError):
                pass
        elif ":" in header_line:
            key, val = header_line.split(":", 1)
            headers.append((key.strip(), val.strip()))
    return status_code, headers


def _stream_response_body(stdout: IO[bytes], wfile: Any) -> None:
    """Stream CGI response body from *stdout* to *wfile*."""
    while True:
        chunk = stdout.read(8192)
        if not chunk:
            break
        wfile.write(chunk)


# ---------------------------------------------------------------------------
# HTTP request handler
# ---------------------------------------------------------------------------


def _make_handler_class(
    base_path: Path, token_store: _SingleTokenStore, hooks_path: Path | None = None
) -> type[BaseHTTPRequestHandler]:
    """Create a request handler class bound to the given base_path and token_store."""

    class GateRequestHandler(BaseHTTPRequestHandler):
        """Handle smart-HTTP git requests with token authentication."""

        server_version = "terok-gate/1.0"

        def do_GET(self) -> None:
            """Handle GET requests (info/refs discovery)."""
            self._handle()

        def do_POST(self) -> None:
            """Handle POST requests (upload-pack, receive-pack)."""
            self._handle()

        def _handle(self) -> None:
            """Route, authenticate, and delegate to CGI."""
            path, query_string = self._split_path()

            m = _ROUTE.match(path)
            if not m:
                self.send_error(404, "Not Found")
                return

            repo = m.group("repo")
            path_info = f"/{repo}{m.group('path')}"

            token = _extract_basic_auth_token(self.headers.get("Authorization"))
            if token is None:
                self._send_auth_required()
                return

            scope = token_store.validate(token)
            if scope is None:
                self.send_error(403, "Forbidden")
                return
            if scope != _ADMIN_WILDCARD and repo != f"{scope}.git":
                self.send_error(403, "Forbidden")
                return

            self._run_cgi(path_info, query_string)

        def _split_path(self) -> tuple[str, str]:
            """Split request path into path and query string."""
            if "?" in self.path:
                path, query = self.path.split("?", 1)
                return path, query
            return self.path, ""

        def _send_auth_required(self) -> None:
            """Send a 401 response with WWW-Authenticate header."""
            self.send_response(401)
            self.send_header("WWW-Authenticate", 'Basic realm="terok gate"')
            self.send_header("Content-Type", "text/plain")
            self.end_headers()
            self.wfile.write(b"Authentication required\n")

        def _run_cgi(self, path_info: str, query_string: str) -> None:
            """Execute ``git http-backend`` and stream the response."""
            content_length, err = _parse_content_length(self.headers.get("Content-Length"))
            if err:
                self.send_error(400, err)
                return

            # Forward HTTP headers that git http-backend needs as CGI variables.
            http_headers: dict[str, str] = {}
            content_encoding = self.headers.get("Content-Encoding")
            if content_encoding:
                http_headers["HTTP_CONTENT_ENCODING"] = content_encoding
            git_protocol = self.headers.get("Git-Protocol")
            if git_protocol:
                http_headers["HTTP_GIT_PROTOCOL"] = git_protocol

            cgi_env = _build_cgi_env(
                base_path,
                path_info,
                query_string,
                self.command,
                self.headers.get("Content-Type", ""),
                self.request_version,
                content_length,
                hooks_path,
                http_headers=http_headers,
            )

            try:
                proc = subprocess.Popen(  # nosec B603 B607 — argv built from fixed verbs + repo-relative paths — binary PATH lookup is the cross-distro contract
                    ["git", "http-backend"],
                    stdin=subprocess.PIPE,
                    stdout=subprocess.PIPE,
                    stderr=subprocess.PIPE,
                    env=cgi_env,
                )
            except FileNotFoundError:
                self.send_error(500, "git not found")
                return
            except OSError:
                self.send_error(500, "git http-backend unavailable")
                return

            # All three streams are guaranteed by ``subprocess.PIPE`` above.
            stdin = cast(IO[bytes], proc.stdin)
            stdout = cast(IO[bytes], proc.stdout)
            stderr = cast(IO[bytes], proc.stderr)

            _stream_request_body(self.rfile, stdin, content_length)
            stdin.close()

            status_code, headers = _parse_cgi_headers(stdout)
            self.send_response(status_code)
            for key, val in headers:
                self.send_header(key, val)
            self.end_headers()

            _stream_response_body(stdout, self.wfile)

            stderr_output = stderr.read()
            try:
                proc.wait(timeout=_CGI_WAIT_TIMEOUT)
            except subprocess.TimeoutExpired:
                proc.kill()
                proc.wait()
            if stderr_output:
                _logger.warning(
                    "git http-backend: %s",
                    stderr_output.decode("utf-8", errors="replace").rstrip(),
                )

        def log_message(self, _format: str, *args: object) -> None:
            """Log 4xx/5xx responses at WARNING level; suppress everything else."""
            try:
                if args and isinstance(args[0], str) and len(args) >= 2:
                    code = int(str(args[1]).split(None, 1)[0])
                    if code >= 400:
                        _logger.warning(_format, *args)
            except (ValueError, IndexError, TypeError):
                pass

    return GateRequestHandler


# ---------------------------------------------------------------------------
# Threading HTTP server
# ---------------------------------------------------------------------------


class _ThreadingHTTPServer(ThreadingMixIn, HTTPServer):
    """HTTP server that handles each request in a new thread."""

    daemon_threads = True
    allow_reuse_address = True


class _UnixThreadingHTTPServer(_ThreadingHTTPServer):
    """ThreadingHTTPServer pre-configured for AF_UNIX listeners.

    ``socketserver.TCPServer.__init__`` always creates ``socket.socket(
    self.address_family, ...)`` regardless of ``bind_and_activate``; the flag
    only skips ``bind`` / ``listen``.  Overriding ``address_family`` ensures
    that throwaway socket is AF_UNIX too, so the daemon survives a
    ``RestrictAddressFamilies=AF_UNIX`` sandbox.
    """

    address_family = socket.AF_UNIX


# ---------------------------------------------------------------------------
# Unix socket server factory
# ---------------------------------------------------------------------------


def _create_unix_server(
    handler_class: type[BaseHTTPRequestHandler],
    socket_path: Path,
) -> _ThreadingHTTPServer:
    """Create an HTTPServer bound to a Unix domain socket.

    Stale socket files are removed if they are actual sockets (not regular
    files that happen to share the path).  The socket is labeled
    ``terok_socket_t`` via `socket_selinux_context` so that
    rootless Podman containers (``container_t``) can ``connectto`` it.
    """
    try:
        if not stat.S_ISSOCK(socket_path.lstat().st_mode):
            raise RuntimeError(f"Refusing to remove non-socket path: {socket_path}")
        socket_path.unlink()
    except FileNotFoundError:
        pass
    socket_path.parent.mkdir(parents=True, exist_ok=True)

    with socket_selinux_context():
        sock = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
        sock.bind(str(socket_path))
        os.chmod(socket_path, 0o600)
        sock.listen(5)

    # bind_and_activate=False — we replace ``server.socket`` below, so the
    # constructor's address tuple is unused.  ``HTTPServer`` types the
    # first arg as a tuple, but the unix-socket path string is what makes
    # debugging easier (HTTPServer never looks at it after this point).
    server = _UnixThreadingHTTPServer(
        str(socket_path),  # type: ignore[arg-type]
        handler_class,
        bind_and_activate=False,
    )
    server.socket.close()
    server.socket = sock
    return server


# ---------------------------------------------------------------------------
# GateServer component
# ---------------------------------------------------------------------------


class GateServer:
    """Per-container git gate, composed by the supervisor alongside the vault.

    Serves the task's repo out of the shared per-project bare mirror at
    *mirror_root*, gated on the single *token* (scoped to *scope*).
    Binds either a per-container Unix socket (*socket_path*) or a
    per-container ``127.0.0.1`` TCP port (*host* + *port*); exactly one
    transport must be supplied.

    Stateless and self-contained — the only terok dependency is the
    SELinux socket-labelling helper the Unix listener needs.
    """

    def __init__(
        self,
        *,
        mirror_root: Path,
        token: str,
        scope: str,
        socket_path: Path | None = None,
        host: str | None = None,
        port: int | None = None,
        hooks_path: Path | None = None,
    ) -> None:
        """Bind the gate's configuration; ``start`` brings the listener up.

        *hooks_path* is the sandbox-owned hooks directory the composer
        prepared (see [`install_hooks`][terok_sandbox.gate.hooks.install_hooks]);
        ``None`` serves with hooks disabled (``core.hooksPath=/dev/null``).
        Plain value by design — this component stays free of gate-model
        imports.
        """
        self._mirror_root = mirror_root
        self._hooks_path = hooks_path
        self._token = token
        self._scope = scope
        self._socket_path = socket_path
        self._host = host
        self._port = port
        self._server: HTTPServer | None = None
        self._thread: threading.Thread | None = None

    async def start(self) -> None:
        """Bind the listener and serve it on a daemon thread."""
        import asyncio

        handler = _make_handler_class(
            self._mirror_root, _SingleTokenStore(self._token, self._scope), self._hooks_path
        )
        if self._socket_path is not None:
            server: HTTPServer = await asyncio.get_running_loop().run_in_executor(
                None, _create_unix_server, handler, self._socket_path
            )
        elif self._host and self._port:
            server = _ThreadingHTTPServer((self._host, self._port), handler)
        else:
            raise ValueError("GateServer needs either socket_path or host+port")
        self._server = server
        self._thread = threading.Thread(target=server.serve_forever, daemon=True, name="terok-gate")
        self._thread.start()

    async def stop(self) -> None:
        """Stop the listener and join the serving thread.

        ``shutdown()`` blocks until the accept loop exits, so it runs in
        an executor rather than inline on the event loop — calling it on
        the loop thread would deadlock.
        """
        import asyncio

        if self._server is None:
            return
        loop = asyncio.get_running_loop()
        await loop.run_in_executor(None, self._server.shutdown)
        self._server.server_close()
        if self._thread is not None:
            self._thread.join(timeout=2.0)
        self._server = None
        self._thread = None
