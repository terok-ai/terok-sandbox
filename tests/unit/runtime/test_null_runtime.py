# SPDX-FileCopyrightText: 2026 Jiri Vyskocil
# SPDX-License-Identifier: Apache-2.0

"""Tests for :class:`NullRuntime` — the in-memory backend used in tests."""

from __future__ import annotations

from pathlib import Path

from terok_sandbox import (
    Container,
    ContainerRuntime,
    ExecResult,
    Image,
    NullRuntime,
)


class TestNullRuntimeProtocolConformance:
    """NullRuntime satisfies the runtime_checkable protocols."""

    def test_runtime_is_a_container_runtime(self) -> None:
        """NullRuntime is a structural match for ContainerRuntime."""
        assert isinstance(NullRuntime(), ContainerRuntime)

    def test_null_container_is_a_container(self) -> None:
        """NullContainer conforms to the Container protocol."""
        assert isinstance(NullRuntime().container("c"), Container)

    def test_null_image_is_an_image(self) -> None:
        """NullImage conforms to the Image protocol."""
        assert isinstance(NullRuntime().image("r"), Image)


class TestNullContainerDefaults:
    """Unfixed handles return safe defaults."""

    def test_state_is_none_without_fixture(self) -> None:
        """Unknown container yields ``state == None``."""
        assert NullRuntime().container("nobody").state is None

    def test_running_is_false_without_fixture(self) -> None:
        """Unknown container yields ``running == False``."""
        assert NullRuntime().container("nobody").running is False

    def test_image_is_none_without_fixture(self) -> None:
        """Unknown container yields ``image == None``."""
        assert NullRuntime().container("nobody").image is None

    def test_rw_size_is_none_without_fixture(self) -> None:
        """Unknown container yields ``rw_size == None``."""
        assert NullRuntime().container("nobody").rw_size is None

    def test_wait_defaults_to_zero(self) -> None:
        """Unknown container's wait returns 0."""
        assert NullRuntime().container("nobody").wait() == 0

    def test_stream_initial_logs_defaults_true(self) -> None:
        """``stream_initial_logs`` defaults to True (ready immediately)."""
        rt = NullRuntime()
        assert rt.container("c").stream_initial_logs(lambda _line: False, None) is True


class TestNullContainerFixtures:
    """Fixture setters influence subsequent handle reads."""

    def test_state_fixture(self) -> None:
        """``set_container_state`` is reflected by ``Container.state``."""
        rt = NullRuntime()
        rt.set_container_state("c", "running")
        assert rt.container("c").state == "running"
        assert rt.container("c").running is True

    def test_image_fixture(self) -> None:
        """``set_container_image`` connects container to image handle."""
        rt = NullRuntime()
        rt.add_image("img-ref", repository="r", tag="v1")
        rt.set_container_image("c", "img-ref")
        image = rt.container("c").image
        assert image is not None
        assert image.ref == "img-ref"

    def test_start_flips_state_to_running(self) -> None:
        """``start()`` flips the fixture state."""
        rt = NullRuntime()
        c = rt.container("c")
        c.start()
        assert c.state == "running"

    def test_stop_flips_state_to_exited(self) -> None:
        """``stop()`` flips the fixture state."""
        rt = NullRuntime()
        c = rt.container("c")
        c.start()
        c.stop()
        assert c.state == "exited"

    def test_copy_in_is_recorded(self, tmp_path: Path) -> None:
        """``copy_in`` appends to the runtime's recording list."""
        rt = NullRuntime()
        src = tmp_path / "x"
        src.write_text("y")
        rt.container("c").copy_in(src, "/dest")
        assert rt._copy_in_calls == [("c", src, "/dest")]

    def test_exit_code_fixture(self) -> None:
        """``set_exit_code`` is returned by ``wait``."""
        rt = NullRuntime()
        rt.set_exit_code("c", 7)
        assert rt.container("c").wait() == 7


class TestNullImageFixtures:
    """Image fixture behaviour."""

    def test_exists_and_fields(self) -> None:
        """``add_image`` populates ``exists`` and the descriptive fields."""
        rt = NullRuntime()
        rt.add_image(
            "ref1",
            repository="docker.io/x",
            tag="v2",
            size="100MB",
            created="yesterday",
        )
        img = rt.image("ref1")
        assert img.exists()
        assert img.repository == "docker.io/x"
        assert img.tag == "v2"
        assert img.size == "100MB"
        assert img.created == "yesterday"

    def test_labels_and_history(self) -> None:
        """Label and history fixtures are returned as provided."""
        rt = NullRuntime()
        rt.add_image(
            "ref",
            labels={"a": "b"},
            history=("FROM scratch", "COPY . /"),
        )
        assert rt.image("ref").labels() == {"a": "b"}
        assert rt.image("ref").history() == ["FROM scratch", "COPY . /"]

    def test_remove_clears_fixture(self) -> None:
        """``remove`` deletes the fixture and reports success once."""
        rt = NullRuntime()
        rt.add_image("ref", labels={"k": "v"})
        assert rt.image("ref").remove() is True
        assert rt.image("ref").remove() is False
        assert rt.image("ref").exists() is False

    def test_id_reflects_existence(self) -> None:
        """``id`` returns the ref when the image exists, else ``None``."""
        rt = NullRuntime()
        rt.add_image("ref")
        assert rt.image("ref").id == "ref"
        assert rt.image("gone").id is None


class TestNullRuntimeOperations:
    """Protocol-level runtime operations."""

    def test_containers_with_prefix(self) -> None:
        """Only containers with fixtures matching the prefix are returned."""
        rt = NullRuntime()
        rt.set_container_state("task-a", "running")
        rt.set_container_state("task-b", "exited")
        rt.set_container_state("other-x", "running")
        names = {c.name for c in rt.containers_with_prefix("task")}
        assert names == {"task-a", "task-b"}

    def test_images_returns_all(self) -> None:
        """``images()`` enumerates every fixture image."""
        rt = NullRuntime()
        rt.add_image("a", tag="v1")
        rt.add_image("b", tag="<none>")
        refs = {img.ref for img in rt.images()}
        assert refs == {"a", "b"}

    def test_images_dangling_only(self) -> None:
        """``dangling_only`` narrows to ``tag == "<none>"`` fixtures."""
        rt = NullRuntime()
        rt.add_image("a", tag="v1")
        rt.add_image("b", tag="<none>")
        assert [img.ref for img in rt.images(dangling_only=True)] == ["b"]

    def test_exec_returns_registered_result(self) -> None:
        """``exec`` returns the pre-registered :class:`ExecResult`."""
        rt = NullRuntime()
        rt.set_exec_result(
            "c",
            ("echo", "hi"),
            ExecResult(exit_code=0, stdout="hi\n", stderr=""),
        )
        result = rt.exec(rt.container("c"), ["echo", "hi"])
        assert result.ok
        assert result.stdout == "hi\n"

    def test_exec_default_is_empty_success(self) -> None:
        """Without a fixture, ``exec`` returns an empty-success result."""
        rt = NullRuntime()
        result = rt.exec(rt.container("c"), ["anything"])
        assert result.ok
        assert result.stdout == ""

    def test_force_remove_clears_state_and_records_call(self) -> None:
        """``force_remove`` clears state fixtures and records the names."""
        rt = NullRuntime()
        rt.set_container_state("a", "running")
        rt.set_container_state("b", "exited")
        results = rt.force_remove([rt.container("a"), rt.container("b")])
        assert [r.name for r in results] == ["a", "b"]
        assert all(r.removed for r in results)
        assert rt._force_remove_calls == [["a", "b"]]
        assert rt.container("a").state is None

    def test_reserve_port_yields_valid_port(self) -> None:
        """``reserve_port`` returns an actually-free port."""
        with NullRuntime().reserve_port() as reservation:
            assert 1024 <= reservation.port <= 65535
