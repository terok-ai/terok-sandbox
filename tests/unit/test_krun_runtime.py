# SPDX-FileCopyrightText: 2026 Jiri Vyskocil
# SPDX-License-Identifier: Apache-2.0

"""Tests for `KrunRuntime` and `FakeKrunTransport`.

Lifecycle methods are thin pass-throughs to a held
[`PodmanRuntime`][terok_sandbox.runtime.podman.PodmanRuntime]; the assertion
shape is "the call lands on the held delegate".  Exec routes through
the injected transport, asserted with a recording fake.
"""

from __future__ import annotations

import io
from unittest.mock import MagicMock

import pytest

from terok_sandbox.runtime import (
    ContainerRuntime,
    ExecResult,
    FakeKrunTransport,
    KrunRuntime,
    KrunTransport,
    NullRuntime,
)


class _StubContainer:
    """Minimal container handle — only ``name`` is read by the transport."""

    def __init__(self, name: str) -> None:
        self.name = name


class TestFakeKrunTransport:
    """`FakeKrunTransport` records calls and replays pre-registered results."""

    def test_default_exec_is_empty_success(self) -> None:
        """Unregistered commands return exit 0 with empty streams."""
        transport = FakeKrunTransport()
        result = transport.exec(_StubContainer("ctr"), ["true"])
        assert result == ExecResult(exit_code=0, stdout="", stderr="")

    def test_exec_returns_registered_result(self) -> None:
        """A pre-registered (container, cmd) replays its scripted result."""
        transport = FakeKrunTransport()
        scripted = ExecResult(exit_code=2, stdout="hello\n", stderr="oops\n")
        transport.set_result("ctr", ("ls", "/etc"), scripted)
        assert transport.exec(_StubContainer("ctr"), ["ls", "/etc"]) == scripted

    def test_exec_records_calls(self) -> None:
        """Every exec call is appended to `exec_calls` for assertion."""
        transport = FakeKrunTransport()
        transport.exec(_StubContainer("ctr"), ["echo", "hi"])
        transport.exec(_StubContainer("ctr"), ["true"])
        assert transport.exec_calls == [
            ("ctr", ("echo", "hi")),
            ("ctr", ("true",)),
        ]

    def test_exec_stdio_records_with_env(self) -> None:
        """exec_stdio records container, cmd, and env without moving I/O."""
        transport = FakeKrunTransport()
        transport.exec_stdio(
            _StubContainer("ctr"),
            ["sh"],
            stdin=io.BytesIO(),
            stdout=io.BytesIO(),
            env={"FOO": "1"},
        )
        assert transport.exec_stdio_calls == [("ctr", ("sh",), {"FOO": "1"})]

    def test_implements_transport_protocol(self) -> None:
        """`FakeKrunTransport` is structurally a `KrunTransport`."""
        assert isinstance(FakeKrunTransport(), KrunTransport)


class TestKrunRuntime:
    """`KrunRuntime` composes podman + transport with no surprises."""

    def test_implements_container_runtime_protocol(self) -> None:
        """`KrunRuntime` satisfies `ContainerRuntime` structurally."""
        rt = KrunRuntime(transport=FakeKrunTransport(), podman=NullRuntime())
        assert isinstance(rt, ContainerRuntime)

    def test_exec_routes_to_transport(self) -> None:
        """`exec` delegates to the injected transport, not to podman."""
        transport = FakeKrunTransport()
        scripted = ExecResult(exit_code=7, stdout="ok\n", stderr="")
        transport.set_result("ctr", ("uname", "-r"), scripted)

        rt = KrunRuntime(transport=transport, podman=NullRuntime())
        result = rt.exec(rt.container("ctr"), ["uname", "-r"])

        assert result == scripted
        assert transport.exec_calls == [("ctr", ("uname", "-r"))]

    def test_exec_empty_cmd_rejected(self) -> None:
        """Empty argv is a programmer error — refuse before reaching transport."""
        transport = FakeKrunTransport()
        rt = KrunRuntime(transport=transport, podman=NullRuntime())
        with pytest.raises(ValueError):
            rt.exec(rt.container("ctr"), [])
        assert transport.exec_calls == []

    def test_exec_stdio_routes_to_transport(self) -> None:
        """`exec_stdio` delegates to the injected transport."""
        transport = FakeKrunTransport()
        rt = KrunRuntime(transport=transport, podman=NullRuntime())
        rt.exec_stdio(
            rt.container("ctr"),
            ["sh"],
            stdin=io.BytesIO(),
            stdout=io.BytesIO(),
            env={"X": "y"},
        )
        assert transport.exec_stdio_calls == [("ctr", ("sh",), {"X": "y"})]

    def test_exec_stdio_empty_cmd_rejected(self) -> None:
        """Empty argv on exec_stdio path is also rejected up front."""
        transport = FakeKrunTransport()
        rt = KrunRuntime(transport=transport, podman=NullRuntime())
        with pytest.raises(ValueError):
            rt.exec_stdio(
                rt.container("ctr"),
                [],
                stdin=io.BytesIO(),
                stdout=io.BytesIO(),
            )

    def test_container_factory_delegates_to_podman(self) -> None:
        """`container()` returns whatever the held podman runtime returns."""
        podman = MagicMock()
        sentinel = object()
        podman.container.return_value = sentinel
        rt = KrunRuntime(transport=FakeKrunTransport(), podman=podman)
        assert rt.container("ctr") is sentinel
        podman.container.assert_called_once_with("ctr")

    def test_containers_with_prefix_delegates(self) -> None:
        """Prefix lookup is podman's concern."""
        podman = MagicMock()
        podman.containers_with_prefix.return_value = []
        rt = KrunRuntime(transport=FakeKrunTransport(), podman=podman)
        assert rt.containers_with_prefix("task") == []
        podman.containers_with_prefix.assert_called_once_with("task")

    def test_image_factory_delegates_to_podman(self) -> None:
        """`image()` returns whatever the held podman runtime returns."""
        podman = MagicMock()
        sentinel = object()
        podman.image.return_value = sentinel
        rt = KrunRuntime(transport=FakeKrunTransport(), podman=podman)
        assert rt.image("alpine:3") is sentinel
        podman.image.assert_called_once_with("alpine:3")

    def test_images_passes_dangling_filter(self) -> None:
        """`images(dangling_only=True)` propagates the flag."""
        podman = MagicMock()
        podman.images.return_value = []
        rt = KrunRuntime(transport=FakeKrunTransport(), podman=podman)
        rt.images(dangling_only=True)
        podman.images.assert_called_once_with(dangling_only=True)

    def test_force_remove_delegates_to_podman(self) -> None:
        """Force-remove is podman's job — never touches the transport."""
        podman = MagicMock()
        podman.force_remove.return_value = []
        transport = FakeKrunTransport()
        rt = KrunRuntime(transport=transport, podman=podman)
        rt.force_remove([rt.container("a"), rt.container("b")])
        podman.force_remove.assert_called_once()
        assert transport.exec_calls == []

    def test_reserve_port_delegates_to_podman(self) -> None:
        """Port reservation is host-side — runs through podman."""
        podman = MagicMock()
        podman.reserve_port.return_value = "RES"
        rt = KrunRuntime(transport=FakeKrunTransport(), podman=podman)
        assert rt.reserve_port("127.0.0.2") == "RES"
        podman.reserve_port.assert_called_once_with("127.0.0.2")

    def test_transport_property_exposes_injection(self) -> None:
        """`runtime.transport` returns whatever was injected at construction."""
        transport = FakeKrunTransport()
        rt = KrunRuntime(transport=transport, podman=NullRuntime())
        assert rt.transport is transport
