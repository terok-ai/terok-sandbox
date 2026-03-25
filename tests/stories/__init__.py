# SPDX-FileCopyrightText: 2026 Jiri Vyskocil
# SPDX-License-Identifier: Apache-2.0

"""End-to-end story tests for the credential proxy.

These tests verify user-facing behaviour: credential storage, phantom
token creation, proxy startup, request forwarding with real credential
injection, and token revocation — all against real sqlite databases and
a real aiohttp server (no mocks).

Intended for disposable CI containers via the test-matrix runner.
They create temporary files, bind Unix sockets, and start/stop
server processes within each test's ``tmp_path``.
"""
