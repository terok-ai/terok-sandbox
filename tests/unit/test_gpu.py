# SPDX-FileCopyrightText: 2026 Jiri Vyskocil
# SPDX-License-Identifier: Apache-2.0

"""Tests for vendor-aware GPU passthrough (``runtime/gpu.py``)."""

from __future__ import annotations

import subprocess
from collections.abc import Iterator
from contextlib import contextmanager
from pathlib import Path
from unittest.mock import patch

import pytest

from terok_sandbox import GpuConfigError, check_gpu_available
from terok_sandbox.runtime.gpu import (
    check_gpu_error,
    detect_gpu_vendors,
    gpu_run_args,
    normalize_gpus,
)

# ── Selector normalization ─────────────────────────────────────────────────


class TestNormalizeGpus:
    """``normalize_gpus`` maps every config/CLI shape onto the selector."""

    @pytest.mark.parametrize(
        ("value", "expected"),
        [
            (None, None),
            (False, None),
            ("", None),
            ([], None),
            (True, "all"),
            ("all", "all"),
            ("ALL", "all"),
            (["amd", "all"], "all"),
            ("nvidia", ("nvidia",)),
            (" AMD ", ("amd",)),
            ("nvidia,intel", ("nvidia", "intel")),
            (["intel", "amd"], ("amd", "intel")),
            (["amd", "amd"], ("amd",)),
        ],
        ids=repr,
    )
    def test_shapes(self, value: object, expected: object) -> None:
        """Booleans, tokens, comma-strings, and lists all normalize."""
        assert normalize_gpus(value) == expected  # type: ignore[arg-type]

    def test_vendor_order_is_canonical(self) -> None:
        """Emission order is fixed regardless of the input order."""
        assert normalize_gpus(["intel", "nvidia", "amd"]) == ("nvidia", "amd", "intel")

    def test_unknown_vendor_raises(self) -> None:
        """A typo'd vendor fails at parse time with the token in the message."""
        with pytest.raises(ValueError, match="matrox"):
            normalize_gpus("matrox")


# ── Podman args per vendor ─────────────────────────────────────────────────


@contextmanager
def _host(
    *,
    cdi_kinds: frozenset[str] = frozenset(),
    kfd: bool = False,
    intel_node: bool = False,
    tmp_path: Path,
) -> Iterator[None]:
    """Patch the host probes to a synthetic GPU inventory."""
    kfd_path = tmp_path / "kfd"
    if kfd:
        kfd_path.touch()
    drm = tmp_path / "drm"
    if intel_node:
        vendor = drm / "renderD128" / "device" / "vendor"
        vendor.parent.mkdir(parents=True)
        vendor.write_text("0x8086\n")
    with (
        patch("terok_sandbox.runtime.gpu._declared_cdi_kinds", return_value=cdi_kinds),
        patch("terok_sandbox.runtime.gpu._KFD_DEVICE", kfd_path),
        patch("terok_sandbox.runtime.gpu._DRM_SYSFS", drm),
    ):
        yield


def _pairs(args: list[str]) -> list[tuple[str, str]]:
    """View flat podman args as ``(flag, value)`` pairs."""
    return list(zip(args[::2], args[1::2], strict=True))


class TestGpuRunArgs:
    """``gpu_run_args`` builds per-vendor podman args from the selector."""

    def test_off_is_empty(self, tmp_path: Path) -> None:
        """``None`` (and the empty tuple) emit nothing and probe nothing."""
        assert gpu_run_args(None) == []
        assert gpu_run_args(()) == []

    def test_nvidia_cdi(self, tmp_path: Path) -> None:
        """NVIDIA with a CDI spec emits the CDI device plus visibility env."""
        with patch(
            "terok_sandbox.runtime.gpu._declared_cdi_kinds",
            return_value=frozenset({"nvidia.com/gpu"}),
        ):
            args = gpu_run_args(("nvidia",))
        assert ("--device", "nvidia.com/gpu=all") in _pairs(args)
        assert ("-e", "NVIDIA_VISIBLE_DEVICES=all") in _pairs(args)
        assert ("-e", "NVIDIA_DRIVER_CAPABILITIES=all") in _pairs(args)

    def test_nvidia_without_cdi_raises(self, tmp_path: Path) -> None:
        """Explicit nvidia without a CDI spec fails before launch."""
        with patch("terok_sandbox.runtime.gpu._declared_cdi_kinds", return_value=frozenset()):
            with pytest.raises(GpuConfigError, match="nvidia-ctk"):
                gpu_run_args(("nvidia",))

    def test_amd_prefers_cdi(self, tmp_path: Path) -> None:
        """AMD uses ``amd.com/gpu`` when amd-ctk generated a spec."""
        with _host(cdi_kinds=frozenset({"amd.com/gpu"}), kfd=True, tmp_path=tmp_path):
            args = gpu_run_args(("amd",))
        assert ("--device", "amd.com/gpu=all") in _pairs(args)
        assert ("--group-add", "keep-groups") in _pairs(args)
        assert not any("kfd" in a for a in args)

    def test_amd_raw_fallback(self, tmp_path: Path) -> None:
        """No AMD CDI spec falls back to the ROCm-documented device pair."""
        with _host(kfd=True, tmp_path=tmp_path):
            args = gpu_run_args(("amd",))
        pairs = _pairs(args)
        assert ("--device", str(tmp_path / "kfd")) in pairs
        assert ("--device", "/dev/dri") in pairs
        assert ("--group-add", "keep-groups") in pairs

    def test_amd_unusable_raises(self, tmp_path: Path) -> None:
        """Explicit amd without CDI or /dev/kfd fails with the driver hint."""
        with _host(tmp_path=tmp_path):
            with pytest.raises(GpuConfigError, match="amdgpu"):
                gpu_run_args(("amd",))

    def test_intel_prefers_cdi(self, tmp_path: Path) -> None:
        """Intel uses ``intel.com/gpu`` when a CDI spec declares it."""
        with _host(cdi_kinds=frozenset({"intel.com/gpu"}), tmp_path=tmp_path):
            args = gpu_run_args(("intel",))
        assert ("--device", "intel.com/gpu=all") in _pairs(args)

    def test_intel_raw_fallback(self, tmp_path: Path) -> None:
        """An Intel render node enables the plain ``/dev/dri`` recipe."""
        with _host(intel_node=True, tmp_path=tmp_path):
            args = gpu_run_args(("intel",))
        pairs = _pairs(args)
        assert ("--device", "/dev/dri") in pairs
        assert ("--group-add", "keep-groups") in pairs

    def test_intel_unusable_raises(self, tmp_path: Path) -> None:
        """Explicit intel with no render node fails with the driver hint."""
        with _host(tmp_path=tmp_path):
            with pytest.raises(GpuConfigError, match="i915"):
                gpu_run_args(("intel",))

    def test_amd_plus_intel_dedups_shared_args(self, tmp_path: Path) -> None:
        """Vendors sharing ``/dev/dri`` and keep-groups emit them once."""
        with _host(kfd=True, intel_node=True, tmp_path=tmp_path):
            args = gpu_run_args(("amd", "intel"))
        pairs = _pairs(args)
        assert pairs.count(("--device", "/dev/dri")) == 1
        assert pairs.count(("--group-add", "keep-groups")) == 1

    def test_all_resolves_to_detected_vendors(self, tmp_path: Path) -> None:
        """``"all"`` passes through what the host has — and only that."""
        with _host(kfd=True, tmp_path=tmp_path):
            args = gpu_run_args("all")
        assert ("--device", "/dev/dri") in _pairs(args)
        assert not any("nvidia" in a for a in args)

    def test_all_with_no_gpus_raises(self, tmp_path: Path) -> None:
        """``"all"`` on a GPU-less host fails loudly, never silently no-GPU."""
        with _host(tmp_path=tmp_path):
            with pytest.raises(GpuConfigError, match="no usable GPU"):
                gpu_run_args("all")


# ── Host detection ─────────────────────────────────────────────────────────


class TestDetectGpuVendors:
    """``detect_gpu_vendors`` reports per-vendor host support."""

    def test_mixed_host(self, tmp_path: Path) -> None:
        """CDI kinds and raw device nodes both count as detection."""
        with _host(
            cdi_kinds=frozenset({"nvidia.com/gpu"}),
            kfd=True,
            intel_node=True,
            tmp_path=tmp_path,
        ):
            assert detect_gpu_vendors() == frozenset({"nvidia", "amd", "intel"})

    def test_bare_host(self, tmp_path: Path) -> None:
        """Nothing detected → empty set, no exception."""
        with _host(tmp_path=tmp_path):
            assert detect_gpu_vendors() == frozenset()

    def test_non_intel_render_node_does_not_count(self, tmp_path: Path) -> None:
        """A render node from another vendor doesn't enable intel."""
        vendor = tmp_path / "drm" / "renderD129" / "device" / "vendor"
        vendor.parent.mkdir(parents=True)
        vendor.write_text("0x1002\n")
        with (
            patch("terok_sandbox.runtime.gpu._declared_cdi_kinds", return_value=frozenset()),
            patch("terok_sandbox.runtime.gpu._KFD_DEVICE", tmp_path / "kfd"),
            patch("terok_sandbox.runtime.gpu._DRM_SYSFS", tmp_path / "drm"),
        ):
            assert detect_gpu_vendors() == frozenset()


# ── CDI spec probing (moved from the podman backend) ──────────────────────


def _fake_podman_info(stdout: str) -> subprocess.CompletedProcess[str]:
    """Mint a ``podman info`` ``CompletedProcess`` carrying *stdout*."""
    return subprocess.CompletedProcess(
        args=["podman", "info"], returncode=0, stdout=stdout, stderr=""
    )


class TestCheckGpuAvailable:
    """``check_gpu_available`` reads CDI spec files podman points to."""

    def test_returns_true_when_nvidia_spec_listed(self, tmp_path: Path) -> None:
        """A spec file declaring ``nvidia.com/gpu`` flips the probe to true."""
        spec = tmp_path / "nvidia.yaml"
        spec.write_text("cdiVersion: 0.6.0\nkind: nvidia.com/gpu\n")
        with patch(
            "terok_sandbox.runtime.gpu.subprocess.run",
            return_value=_fake_podman_info(f'["{spec}"]'),
        ):
            assert check_gpu_available() is True

    def test_returns_false_when_specs_dont_mention_nvidia(self, tmp_path: Path) -> None:
        """Other vendors' specs don't enable the NVIDIA option."""
        spec = tmp_path / "amd.yaml"
        spec.write_text("cdiVersion: 0.6.0\nkind: amd.com/gpu\n")
        with patch(
            "terok_sandbox.runtime.gpu.subprocess.run",
            return_value=_fake_podman_info(f'["{spec}"]'),
        ):
            assert check_gpu_available() is False

    def test_returns_false_when_podman_missing(self) -> None:
        """No podman → no GPU; never raises."""
        with (
            patch(
                "terok_sandbox.runtime.gpu.subprocess.run",
                side_effect=FileNotFoundError("podman"),
            ),
            patch(
                "terok_sandbox.runtime.gpu._CDI_DEFAULT_DIRS",
                (Path("/nonexistent-cdi-a"), Path("/nonexistent-cdi-b")),
            ),
        ):
            assert check_gpu_available() is False

    def test_returns_false_when_podman_errors(self) -> None:
        """A non-zero ``podman info`` falls back to scanning default dirs."""
        err = subprocess.CalledProcessError(returncode=1, cmd=["podman", "info"], stderr=b"boom")
        with (
            patch("terok_sandbox.runtime.gpu.subprocess.run", side_effect=err),
            patch(
                "terok_sandbox.runtime.gpu._CDI_DEFAULT_DIRS",
                (Path("/nonexistent-cdi-a"),),
            ),
        ):
            assert check_gpu_available() is False

    def test_falls_back_to_default_dirs(self, tmp_path: Path) -> None:
        """An empty ``Host.CDISpecs`` triggers a default-directory scan."""
        cdi_dir = tmp_path / "etc-cdi"
        cdi_dir.mkdir()
        (cdi_dir / "nvidia.yaml").write_text("kind: nvidia.com/gpu\n")
        with (
            patch(
                "terok_sandbox.runtime.gpu.subprocess.run",
                return_value=_fake_podman_info("null"),
            ),
            patch("terok_sandbox.runtime.gpu._CDI_DEFAULT_DIRS", (cdi_dir,)),
        ):
            assert check_gpu_available() is True

    def test_unreadable_spec_does_not_raise(self, tmp_path: Path) -> None:
        """A spec path pointing at a directory (unreadable) collapses to false, not an exception."""
        bogus = tmp_path / "subdir"
        bogus.mkdir()
        with patch(
            "terok_sandbox.runtime.gpu.subprocess.run",
            return_value=_fake_podman_info(f'["{bogus}"]'),
        ):
            assert check_gpu_available() is False


# ── Launch-failure translation (moved from the podman backend) ────────────


class TestCheckGpuError:
    """``check_gpu_error`` raises ``GpuConfigError`` only on GPU patterns."""

    def test_cdi_pattern_raises(self) -> None:
        """Stderr matching a CDI pattern triggers ``GpuConfigError``."""
        exc = subprocess.CalledProcessError(
            returncode=125,
            cmd=["podman", "run"],
            stderr=b"Error: CDI device nvidia.com/gpu=all not registered",
        )
        with pytest.raises(GpuConfigError) as excinfo:
            check_gpu_error(exc)
        assert "GPU misconfiguration" in str(excinfo.value)
        assert excinfo.value.hint
        assert excinfo.value.__cause__ is exc

    def test_amd_cdi_pattern_raises(self) -> None:
        """AMD CDI failures are recognised the same way as NVIDIA ones."""
        exc = subprocess.CalledProcessError(
            returncode=125,
            cmd=["podman", "run"],
            stderr=b"Error: amd.com/gpu=all: unresolvable CDI device",
        )
        with pytest.raises(GpuConfigError):
            check_gpu_error(exc)

    def test_keep_groups_pattern_hints_at_crun(self) -> None:
        """The runc keep-groups rejection maps to the crun hint."""
        exc = subprocess.CalledProcessError(
            returncode=125,
            cmd=["podman", "run"],
            stderr=b"Error: option --group-add keep-groups is not supported by runc",
        )
        with pytest.raises(GpuConfigError) as excinfo:
            check_gpu_error(exc)
        assert "crun" in excinfo.value.hint

    def test_unrelated_error_does_not_raise(self) -> None:
        """Non-GPU stderr passes through silently."""
        exc = subprocess.CalledProcessError(
            returncode=125,
            cmd=["podman", "run"],
            stderr=b"Error: image not found",
        )
        check_gpu_error(exc)

    def test_no_stderr_does_not_raise(self) -> None:
        """Missing stderr is treated as no GPU match."""
        exc = subprocess.CalledProcessError(returncode=125, cmd=["podman"], stderr=None)
        check_gpu_error(exc)

    def test_text_stderr_is_handled(self) -> None:
        """``str`` stderr (``text=True`` callers) does not trip ``.decode``."""
        exc = subprocess.CalledProcessError(
            returncode=125,
            cmd=["podman", "run"],
            stderr="Error: CDI device nvidia.com/gpu=all not registered",
        )
        with pytest.raises(GpuConfigError):
            check_gpu_error(exc)
