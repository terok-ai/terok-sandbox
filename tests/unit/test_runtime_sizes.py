# SPDX-FileCopyrightText: 2026 Jiri Vyskocil
# SPDX-License-Identifier: Apache-2.0

"""Tests for container writable-layer size queries.

The runtime asks podman about container sizes via ``podman container inspect
--size`` (per-container) and ``podman ps --size`` (bulk).  These tests mock
the subprocess boundary so we never need a real daemon.
"""

from __future__ import annotations

import subprocess
from unittest.mock import patch

from terok_sandbox import PodmanRuntime
from terok_sandbox.runtime.podman import _parse_human_size

# -- _parse_human_size: the translator between podman's prose and our ints --


class TestParseHumanSize:
    """Podman speaks in human-readable sizes; we need bytes."""

    def test_simple_megabytes(self):
        """MB suffix is decimal megabytes (10^6)."""
        assert _parse_human_size("12.5MB") == 12_500_000

    def test_gigabytes(self):
        """GB suffix is decimal gigabytes (10^9)."""
        assert _parse_human_size("1.23GB") == 1_230_000_000

    def test_bytes_bare(self):
        """B suffix is raw bytes."""
        assert _parse_human_size("1024B") == 1024

    def test_kilobytes(self):
        """KB suffix is decimal kilobytes (10^3)."""
        assert _parse_human_size("512KB") == 512_000

    def test_binary_units(self):
        """MiB suffix is binary megabytes (2^20)."""
        assert _parse_human_size("1MiB") == 1 << 20

    def test_podman_virtual_format(self):
        """Podman emits ``'12.5MB (virtual 1.23GB)'`` — first number wins."""
        assert _parse_human_size("12.5MB (virtual 1.23GB)") == 12_500_000

    def test_empty_string(self):
        """Empty input yields ``None``."""
        assert _parse_human_size("") is None

    def test_garbage(self):
        """Unparseable input yields ``None``."""
        assert _parse_human_size("not-a-size") is None


# -- Container.rw_size: per-container inspect --


class TestContainerRwSize:
    """Per-container inspect path via ``Container.rw_size``."""

    def test_returns_bytes(self):
        """Numeric output is parsed as an int."""
        with patch(
            "terok_sandbox.runtime.podman.subprocess.check_output",
            return_value="4096\n",
        ):
            assert PodmanRuntime().container("myproj-cli-abc123").rw_size == 4096

    def test_empty_output_yields_none(self):
        """Empty podman output → ``None``."""
        with patch("terok_sandbox.runtime.podman.subprocess.check_output", return_value=""):
            assert PodmanRuntime().container("gone").rw_size is None

    def test_podman_missing_yields_none(self):
        """Missing podman binary → ``None``."""
        with patch(
            "terok_sandbox.runtime.podman.subprocess.check_output",
            side_effect=FileNotFoundError,
        ):
            assert PodmanRuntime().container("any").rw_size is None

    def test_container_not_found_yields_none(self):
        """CalledProcessError → ``None``."""
        with patch(
            "terok_sandbox.runtime.podman.subprocess.check_output",
            side_effect=subprocess.CalledProcessError(125, "podman"),
        ):
            assert PodmanRuntime().container("nosuch").rw_size is None

    def test_non_numeric_output_yields_none(self):
        """Non-parseable output → ``None``."""
        with patch(
            "terok_sandbox.runtime.podman.subprocess.check_output",
            return_value="<nil>\n",
        ):
            assert PodmanRuntime().container("broken").rw_size is None


# -- PodmanRuntime.container_rw_sizes: batch path --


class TestContainerRwSizesBatch:
    """Bulk size query: trade precision for speed with ``podman ps --size``."""

    SAMPLE_OUTPUT = (
        "myproj-cli-aaa\t12.5MB (virtual 1.23GB)\nmyproj-cli-bbb\t3.4kB (virtual 800MB)\n"
    )

    def test_parses_multiple_containers(self):
        """Each well-formed line becomes one dict entry."""
        with patch(
            "terok_sandbox.runtime.podman.subprocess.check_output",
            return_value=self.SAMPLE_OUTPUT,
        ):
            sizes = PodmanRuntime().container_rw_sizes("myproj")
            assert sizes == {
                "myproj-cli-aaa": 12_500_000,
                "myproj-cli-bbb": 3_400,
            }

    def test_empty_output(self):
        """Empty output yields an empty dict."""
        with patch("terok_sandbox.runtime.podman.subprocess.check_output", return_value=""):
            assert PodmanRuntime().container_rw_sizes("empty") == {}

    def test_podman_missing(self):
        """Missing podman yields an empty dict."""
        with patch(
            "terok_sandbox.runtime.podman.subprocess.check_output",
            side_effect=FileNotFoundError,
        ):
            assert PodmanRuntime().container_rw_sizes("any") == {}

    def test_malformed_lines_skipped(self):
        """Lines without a tab are skipped rather than crashing."""
        output = "good-container\t12MB\nbad-line-no-tab\n"
        with patch("terok_sandbox.runtime.podman.subprocess.check_output", return_value=output):
            sizes = PodmanRuntime().container_rw_sizes("good")
            assert sizes == {"good-container": 12_000_000}

    def test_unparseable_size_skipped(self):
        """Unparseable size strings are skipped."""
        output = "container\tnot-a-number\n"
        with patch("terok_sandbox.runtime.podman.subprocess.check_output", return_value=output):
            assert PodmanRuntime().container_rw_sizes("container") == {}
