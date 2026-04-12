# SPDX-FileCopyrightText: 2026 Jiri Vyskocil
# SPDX-License-Identifier: Apache-2.0

"""Tests for container writable-layer size queries.

The runtime module answers questions about containers by asking podman.
These tests mock the subprocess boundary so we never need a real daemon.
"""

from __future__ import annotations

import subprocess
from unittest.mock import patch

from terok_sandbox.runtime import (
    _parse_human_size,
    get_container_rw_size,
    get_container_rw_sizes,
)

# -- _parse_human_size: the translator between podman's prose and our ints --


class TestParseHumanSize:
    """Podman speaks in human-readable sizes; we need bytes."""

    def test_simple_megabytes(self):
        assert _parse_human_size("12.5MB") == 12_500_000

    def test_gigabytes(self):
        assert _parse_human_size("1.23GB") == 1_230_000_000

    def test_bytes_bare(self):
        assert _parse_human_size("1024B") == 1024

    def test_kilobytes(self):
        assert _parse_human_size("512KB") == 512_000

    def test_binary_units(self):
        assert _parse_human_size("1MiB") == 1 << 20

    def test_podman_virtual_format(self):
        """Podman emits ``'12.5MB (virtual 1.23GB)'`` — we want only the first number."""
        assert _parse_human_size("12.5MB (virtual 1.23GB)") == 12_500_000

    def test_empty_string(self):
        assert _parse_human_size("") is None

    def test_garbage(self):
        assert _parse_human_size("not-a-size") is None


# -- get_container_rw_size: asking about one container --


class TestGetContainerRwSize:
    """Single-container inspect path — precise but per-container."""

    def test_returns_bytes(self):
        with patch("terok_sandbox.runtime.subprocess.check_output", return_value="4096\n"):
            assert get_container_rw_size("myproj-cli-abc123") == 4096

    def test_empty_output_yields_none(self):
        with patch("terok_sandbox.runtime.subprocess.check_output", return_value=""):
            assert get_container_rw_size("gone") is None

    def test_podman_missing_yields_none(self):
        with patch("terok_sandbox.runtime.subprocess.check_output", side_effect=FileNotFoundError):
            assert get_container_rw_size("any") is None

    def test_container_not_found_yields_none(self):
        with patch(
            "terok_sandbox.runtime.subprocess.check_output",
            side_effect=subprocess.CalledProcessError(125, "podman"),
        ):
            assert get_container_rw_size("nosuch") is None

    def test_non_numeric_output_yields_none(self):
        with patch("terok_sandbox.runtime.subprocess.check_output", return_value="<nil>\n"):
            assert get_container_rw_size("broken") is None


# -- get_container_rw_sizes: the bulk query — one podman call for all containers --


class TestGetContainerRwSizes:
    """Bulk size query: trade precision for speed with ``podman ps --size``."""

    SAMPLE_OUTPUT = (
        "myproj-cli-aaa\t12.5MB (virtual 1.23GB)\nmyproj-cli-bbb\t3.4kB (virtual 800MB)\n"
    )

    def test_parses_multiple_containers(self):
        with patch(
            "terok_sandbox.runtime.subprocess.check_output", return_value=self.SAMPLE_OUTPUT
        ):
            sizes = get_container_rw_sizes("myproj")
            assert sizes == {
                "myproj-cli-aaa": 12_500_000,
                "myproj-cli-bbb": 3_400,
            }

    def test_empty_output(self):
        with patch("terok_sandbox.runtime.subprocess.check_output", return_value=""):
            assert get_container_rw_sizes("empty") == {}

    def test_podman_missing(self):
        with patch("terok_sandbox.runtime.subprocess.check_output", side_effect=FileNotFoundError):
            assert get_container_rw_sizes("any") == {}

    def test_malformed_lines_skipped(self):
        output = "good-container\t12MB\nbad-line-no-tab\n"
        with patch("terok_sandbox.runtime.subprocess.check_output", return_value=output):
            sizes = get_container_rw_sizes("good")
            assert sizes == {"good-container": 12_000_000}

    def test_unparseable_size_skipped(self):
        output = "container\tnot-a-number\n"
        with patch("terok_sandbox.runtime.subprocess.check_output", return_value=output):
            assert get_container_rw_sizes("container") == {}
