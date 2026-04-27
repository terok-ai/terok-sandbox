# SPDX-FileCopyrightText: 2026 Jiri Vyskocil
# SPDX-License-Identifier: Apache-2.0

"""Tests for [`PodmanInspector`][terok_sandbox.PodmanInspector] + [`create_container_inspector`][terok_sandbox.create_container_inspector]."""

from __future__ import annotations

import subprocess
from unittest.mock import MagicMock, patch

import pytest
from terok_clearance import ContainerInfo, ContainerInspector

from terok_sandbox.podman import PodmanInspector, _from_inspect, create_container_inspector

_CONTAINER_ID = "fa0905d97a1c"


def _mock_completed(stdout: str, returncode: int = 0, stderr: str = "") -> MagicMock:
    """Minimal ``CompletedProcess`` stand-in for the ``subprocess.run`` patch."""
    proc = MagicMock()
    proc.stdout = stdout
    proc.returncode = returncode
    proc.stderr = stderr
    return proc


_FULL_JSON = """[
    {
        "Id": "fa0905d97a1c",
        "Name": "/sandbox-alpha",
        "State": {"Status": "running"},
        "Config": {
            "Annotations": {
                "ai.terok.project": "warp-core",
                "ai.terok.task": "t42",
                "io.kubernetes.cri-o.Unrelated": "ignored"
            }
        }
    }
]"""


class TestFromInspect:
    """Pure-function parse of the ``podman inspect`` JSON payload."""

    def test_happy_path_unpacks_name_state_annotations(self) -> None:
        import json

        info = _from_inspect(_CONTAINER_ID, json.loads(_FULL_JSON))
        assert info.container_id == _CONTAINER_ID
        # Name is stripped of podman's leading '/'.
        assert info.name == "sandbox-alpha"
        assert info.state == "running"
        # All string-valued annotations are preserved — sandbox doesn't
        # filter by key, that's the caller's (terok's) job.
        assert info.annotations == {
            "ai.terok.project": "warp-core",
            "ai.terok.task": "t42",
            "io.kubernetes.cri-o.Unrelated": "ignored",
        }

    def test_empty_list_returns_empty_info(self) -> None:
        assert _from_inspect(_CONTAINER_ID, []) == ContainerInfo()

    def test_non_list_returns_empty_info(self) -> None:
        """Guards against `podman inspect` future wire changes."""
        assert _from_inspect(_CONTAINER_ID, {}) == ContainerInfo()

    def test_missing_state_stanza_defaults_to_empty_string(self) -> None:
        info = _from_inspect(_CONTAINER_ID, [{"Name": "/c"}])
        assert info.state == ""
        assert info.name == "c"

    def test_non_string_annotations_skipped(self) -> None:
        """Robust to podman handing back non-string annotation values."""
        info = _from_inspect(
            _CONTAINER_ID,
            [{"Config": {"Annotations": {"good": "ok", "bad": 123, 99: "key-type"}}}],
        )
        assert info.annotations == {"good": "ok"}


class TestPodmanInspector:
    """End-to-end inspect + caching, with ``subprocess.run`` patched."""

    def test_empty_container_id_returns_empty(self) -> None:
        """No subprocess ever runs for an empty input."""
        with patch("terok_sandbox.podman.shutil.which") as which:
            result = PodmanInspector()("")
        which.assert_not_called()
        assert result == ContainerInfo()

    def test_missing_podman_binary_returns_empty(self) -> None:
        """Hosts without podman installed soft-fail to empty, no raise."""
        with patch("terok_sandbox.podman.shutil.which", return_value=None):
            result = PodmanInspector()(_CONTAINER_ID)
        assert result == ContainerInfo()

    def test_successful_inspect_populates_info(self) -> None:
        with (
            patch("terok_sandbox.podman.shutil.which", return_value="/usr/bin/podman"),
            patch(
                "terok_sandbox.podman.subprocess.run",
                return_value=_mock_completed(_FULL_JSON, returncode=0),
            ),
        ):
            info = PodmanInspector()(_CONTAINER_ID)
        assert info.name == "sandbox-alpha"
        assert info.annotations["ai.terok.project"] == "warp-core"

    def test_non_zero_exit_returns_empty(self) -> None:
        """`podman inspect` on a missing container exits non-zero; stay silent."""
        with (
            patch("terok_sandbox.podman.shutil.which", return_value="/usr/bin/podman"),
            patch(
                "terok_sandbox.podman.subprocess.run",
                return_value=_mock_completed("", returncode=125, stderr="no such container"),
            ),
        ):
            info = PodmanInspector()(_CONTAINER_ID)
        assert info == ContainerInfo()

    def test_malformed_json_returns_empty(self) -> None:
        with (
            patch("terok_sandbox.podman.shutil.which", return_value="/usr/bin/podman"),
            patch(
                "terok_sandbox.podman.subprocess.run",
                return_value=_mock_completed("not json"),
            ),
        ):
            info = PodmanInspector()(_CONTAINER_ID)
        assert info == ContainerInfo()

    def test_os_error_returns_empty(self) -> None:
        with (
            patch("terok_sandbox.podman.shutil.which", return_value="/usr/bin/podman"),
            patch("terok_sandbox.podman.subprocess.run", side_effect=OSError("exec fail")),
        ):
            info = PodmanInspector()(_CONTAINER_ID)
        assert info == ContainerInfo()

    def test_timeout_returns_empty(self) -> None:
        with (
            patch("terok_sandbox.podman.shutil.which", return_value="/usr/bin/podman"),
            patch(
                "terok_sandbox.podman.subprocess.run",
                side_effect=subprocess.TimeoutExpired(cmd=["podman"], timeout=5),
            ),
        ):
            info = PodmanInspector()(_CONTAINER_ID)
        assert info == ContainerInfo()

    def test_second_lookup_hits_cache(self) -> None:
        """Only one subprocess call for repeated hits on the same container ID."""
        with (
            patch("terok_sandbox.podman.shutil.which", return_value="/usr/bin/podman"),
            patch(
                "terok_sandbox.podman.subprocess.run",
                return_value=_mock_completed(_FULL_JSON),
            ) as run_mock,
        ):
            inspector = PodmanInspector()
            inspector(_CONTAINER_ID)
            inspector(_CONTAINER_ID)
            inspector(_CONTAINER_ID)
        assert run_mock.call_count == 1

    def test_inspect_uses_dash_dash_separator(self) -> None:
        """The `--` guard prevents a leading-dash container_id from being read as a flag."""
        with (
            patch("terok_sandbox.podman.shutil.which", return_value="/usr/bin/podman"),
            patch(
                "terok_sandbox.podman.subprocess.run",
                return_value=_mock_completed(_FULL_JSON),
            ) as run_mock,
        ):
            PodmanInspector()(_CONTAINER_ID)
        argv = run_mock.call_args.args[0]
        assert "--" in argv
        assert argv[argv.index("--") + 1] == _CONTAINER_ID

    def test_annotations_are_read_only(self) -> None:
        """Cached ``ContainerInfo`` instances are shared, so mutation must raise."""
        with (
            patch("terok_sandbox.podman.shutil.which", return_value="/usr/bin/podman"),
            patch(
                "terok_sandbox.podman.subprocess.run",
                return_value=_mock_completed(_FULL_JSON),
            ),
        ):
            info = PodmanInspector()(_CONTAINER_ID)
        with pytest.raises(TypeError):
            info.annotations["ai.terok.project"] = "hijacked"  # type: ignore[index]


class TestCreateContainerInspector:
    """Runtime-neutral factory — clearance's notifier reaches through it."""

    def test_returns_podman_inspector_today(self) -> None:
        """Only ``PodmanInspector`` is wired in today; the factory reflects that."""
        assert isinstance(create_container_inspector(), PodmanInspector)

    def test_satisfies_clearance_protocol(self) -> None:
        """Return value ducks-in as a [`terok_clearance.ContainerInspector`][terok_clearance.ContainerInspector]."""
        assert isinstance(create_container_inspector(), ContainerInspector)

    def test_returns_fresh_instances(self) -> None:
        """Each call hands back a fresh inspector so per-instance caches don't leak."""
        assert create_container_inspector() is not create_container_inspector()
