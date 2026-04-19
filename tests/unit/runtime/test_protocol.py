# SPDX-FileCopyrightText: 2026 Jiri Vyskocil
# SPDX-License-Identifier: Apache-2.0

"""Protocol-level tests — behaviour common to every backend."""

from __future__ import annotations

import pytest

from terok_sandbox import (
    Container,
    ContainerRemoveResult,
    ContainerRuntime,
    ExecResult,
    Image,
    NullRuntime,
    PodmanRuntime,
)


class TestExecResult:
    """The ``ExecResult`` value type — backend-neutral exec outcome."""

    def test_ok_is_true_on_zero(self) -> None:
        """Zero exit code → ``ok == True``."""
        assert ExecResult(exit_code=0, stdout="", stderr="").ok

    def test_ok_is_false_on_nonzero(self) -> None:
        """Non-zero exit code → ``ok == False``."""
        assert not ExecResult(exit_code=1, stdout="", stderr="").ok

    def test_frozen(self) -> None:
        """Dataclass is frozen."""
        r = ExecResult(exit_code=0, stdout="", stderr="")
        with pytest.raises(AttributeError):
            r.exit_code = 1  # type: ignore[misc]


class TestContainerRemoveResult:
    """The ``ContainerRemoveResult`` value type."""

    def test_default_error_is_none(self) -> None:
        """Successful removal carries no error."""
        assert ContainerRemoveResult(name="c", removed=True).error is None


@pytest.fixture(params=["null", "podman"])
def runtime(request):
    """Parametrise tests across every concrete runtime."""
    return NullRuntime() if request.param == "null" else PodmanRuntime()


class TestRuntimeProtocolConformance:
    """Every concrete runtime is a structural ``ContainerRuntime``."""

    def test_is_instance(self, runtime: ContainerRuntime) -> None:
        """Concrete runtime is a ``ContainerRuntime``."""
        assert isinstance(runtime, ContainerRuntime)

    def test_container_handles_conform(self, runtime: ContainerRuntime) -> None:
        """``runtime.container(...)`` yields a ``Container``."""
        assert isinstance(runtime.container("c"), Container)

    def test_image_handles_conform(self, runtime: ContainerRuntime) -> None:
        """``runtime.image(...)`` yields an ``Image``."""
        assert isinstance(runtime.image("ref"), Image)


class TestHandleIdentity:
    """Handle equality is backend-consistent."""

    def test_container_equality_by_name(self, runtime: ContainerRuntime) -> None:
        """Two handles with the same name compare equal and share a hash."""
        a = runtime.container("same")
        b = runtime.container("same")
        assert a == b
        assert hash(a) == hash(b)

    def test_container_inequality(self, runtime: ContainerRuntime) -> None:
        """Handles with different names compare unequal."""
        assert runtime.container("a") != runtime.container("b")

    def test_image_equality_by_ref(self, runtime: ContainerRuntime) -> None:
        """Two image handles with the same ref compare equal."""
        a = runtime.image("ref")
        b = runtime.image("ref")
        assert a == b
        assert hash(a) == hash(b)


class TestPortReservationProtocol:
    """Every backend exposes a usable PortReservation."""

    def test_reserve_port_returns_valid_port(self, runtime: ContainerRuntime) -> None:
        """Reserved port is in the dynamic range."""
        with runtime.reserve_port() as reservation:
            assert 1024 <= reservation.port <= 65535

    def test_reserve_port_close_is_idempotent(self, runtime: ContainerRuntime) -> None:
        """``close`` can be called multiple times without error."""
        reservation = runtime.reserve_port()
        reservation.close()
        reservation.close()
