# SPDX-FileCopyrightText: 2026 Jiri Vyskocil
# SPDX-License-Identifier: Apache-2.0

"""Tests for the podman container-event stream.

``podman events`` is a long-lived subscription, so the live ``Popen`` path needs
a daemon and isn't unit-tested here.  What we can pin down without one is the
pure line decoder — the filter that turns a raw JSON event into a
[`ContainerEvent`][terok_sandbox.runtime.podman.ContainerEvent] or rejects it as
noise.  That decoder is where every correctness decision lives.
"""

from __future__ import annotations

import io
import json
from typing import Any

import terok_sandbox.runtime.podman as podman
from terok_sandbox import ContainerEvent, PodmanEventStream
from terok_sandbox.runtime.podman import _parse_event

PREFIX = "proj"


def _line(**fields: object) -> str:
    """Render one ``podman events --format '{{json .}}'`` output line."""
    return json.dumps(fields)


class TestKeeps:
    """Lifecycle events on a matching container survive the filter."""

    def test_start_event_for_matching_container(self) -> None:
        line = _line(Type="container", Name="proj-cli-abc", Status="start")
        assert _parse_event(line, PREFIX) == ContainerEvent(name="proj-cli-abc", status="start")

    def test_died_event(self) -> None:
        line = _line(Type="container", Name="proj-run-xyz", Status="died")
        assert _parse_event(line, PREFIX) == ContainerEvent(name="proj-run-xyz", status="died")

    def test_lowercase_field_spellings(self) -> None:
        """Podman has shipped both ``Status`` and ``status`` — accept either."""
        line = _line(type="container", name="proj-cli-abc", status="stop")
        assert _parse_event(line, PREFIX) == ContainerEvent(name="proj-cli-abc", status="stop")

    def test_status_is_normalised_to_lowercase(self) -> None:
        line = _line(Type="container", Name="proj-cli-abc", Status="START")
        event = _parse_event(line, PREFIX)
        assert event is not None and event.status == "start"


class TestRejects:
    """Noise is dropped so it never wakes a reconcile."""

    def test_other_project(self) -> None:
        line = _line(Type="container", Name="other-cli-abc", Status="start")
        assert _parse_event(line, PREFIX) is None

    def test_prefix_must_end_on_a_dash_boundary(self) -> None:
        """``proj2-…`` must not match the ``proj`` prefix."""
        line = _line(Type="container", Name="proj2-cli-abc", Status="start")
        assert _parse_event(line, PREFIX) is None

    def test_non_container_type(self) -> None:
        line = _line(Type="image", Name="proj-cli-abc", Status="pull")
        assert _parse_event(line, PREFIX) is None

    def test_exec_attach_noise(self) -> None:
        line = _line(Type="container", Name="proj-cli-abc", Status="exec")
        assert _parse_event(line, PREFIX) is None

    def test_malformed_json(self) -> None:
        assert _parse_event("{not json", PREFIX) is None

    def test_missing_name(self) -> None:
        line = _line(Type="container", Status="start")
        assert _parse_event(line, PREFIX) is None


class _FakeProc:
    """A stand-in for the ``podman events`` child — replays canned stdout lines."""

    def __init__(self, lines: list[bytes]) -> None:
        self.stdout = io.BytesIO(b"".join(lines))
        self._returncode: int | None = None
        self.terminated = False

    def poll(self) -> int | None:
        return self._returncode

    def terminate(self) -> None:
        self.terminated = True
        self._returncode = 0

    def wait(self, timeout: float | None = None) -> int:
        return 0

    def kill(self) -> None:
        self._returncode = -9


class TestStream:
    """The Popen-backed iterator yields matching events and closes cleanly."""

    def _stream(self, monkeypatch: Any, lines: list[bytes]) -> tuple[PodmanEventStream, _FakeProc]:
        fake = _FakeProc(lines)
        monkeypatch.setattr(podman.subprocess, "Popen", lambda *a, **k: fake)
        return PodmanEventStream(PREFIX), fake

    def test_yields_matching_events_then_stops_at_eof(self, monkeypatch: Any) -> None:
        lines = [
            (
                json.dumps({"Type": "container", "Name": "proj-cli-1", "Status": "start"}) + "\n"
            ).encode(),
            (
                json.dumps({"Type": "container", "Name": "other-x", "Status": "start"}) + "\n"
            ).encode(),
            (
                json.dumps({"Type": "container", "Name": "proj-run-2", "Status": "died"}) + "\n"
            ).encode(),
        ]
        stream, _ = self._stream(monkeypatch, lines)
        assert list(stream) == [
            ContainerEvent("proj-cli-1", "start"),
            ContainerEvent("proj-run-2", "died"),
        ]

    def test_close_terminates_the_child_and_is_idempotent(self, monkeypatch: Any) -> None:
        stream, fake = self._stream(monkeypatch, [])
        stream.close()
        assert fake.terminated
        stream.close()  # second call is a no-op

    def test_context_manager_closes(self, monkeypatch: Any) -> None:
        lines = [
            (
                json.dumps({"Type": "container", "Name": "proj-cli-1", "Status": "start"}) + "\n"
            ).encode()
        ]
        with self._stream(monkeypatch, lines)[0] as stream:
            assert next(iter(stream)) == ContainerEvent("proj-cli-1", "start")


class _TimeoutProc(_FakeProc):
    """A child whose first ``wait`` times out, forcing the ``kill`` fallback."""

    def __init__(self) -> None:
        super().__init__([])
        self._returncode = None
        self._waits = 0

    def wait(self, timeout: float | None = None) -> int:
        import subprocess

        self._waits += 1
        if self._waits == 1:
            raise subprocess.TimeoutExpired("podman", timeout)
        return -9


class TestRuntimeEvents:
    """``PodmanRuntime.events`` hands back a live stream; teardown is robust."""

    def test_events_returns_a_stream(self, monkeypatch: Any) -> None:
        from terok_sandbox import PodmanRuntime

        monkeypatch.setattr(podman.subprocess, "Popen", lambda *a, **k: _FakeProc([]))
        stream = PodmanRuntime().events(PREFIX)
        assert isinstance(stream, PodmanEventStream)
        stream.close()

    def test_close_kills_when_terminate_times_out(self, monkeypatch: Any) -> None:
        fake = _TimeoutProc()
        monkeypatch.setattr(podman.subprocess, "Popen", lambda *a, **k: fake)
        stream = PodmanEventStream(PREFIX)
        stream.close()  # terminate → wait times out → kill → wait
        assert fake._waits == 2
