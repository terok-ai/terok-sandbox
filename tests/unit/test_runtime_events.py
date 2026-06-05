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

import json

from terok_sandbox import ContainerEvent
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
