# SPDX-FileCopyrightText: 2026 Jiri Vyskocil
# SPDX-License-Identifier: Apache-2.0

"""Cross-package convention: ``# terok-{vault,gate}-version:`` on line 1.

Sandbox's orphan-sweep code (``GateServerManager._sweep_orphan_units`` /
``VaultManager._sweep_orphan_units``) reads ``splitlines()[0]`` as a strict
ownership check against globbed legacy/renamed unit files.  A future PR
that pushes the marker below the SPDX header (a tempting reformat) would
silently break orphan sweep on hosts that still have a renamed unit on
disk.  Asserting line-1 here makes the contract machine-checked and
mirrors the equivalent test in terok-clearance.
"""

from __future__ import annotations

from pathlib import Path

import pytest

import terok_sandbox.gate as _gate_pkg
import terok_sandbox.vault as _vault_pkg


def _template_path(package, name: str) -> Path:
    """Return the on-disk path of a shipped unit template."""
    return Path(package.__file__).resolve().parent / "resources" / "systemd" / name


@pytest.mark.parametrize(
    ("package", "unit_name", "marker_prefix"),
    [
        (_vault_pkg, "terok-vault.service", "# terok-vault-version:"),
        (_vault_pkg, "terok-vault-socket.service", "# terok-vault-version:"),
        (_vault_pkg, "terok-vault.socket", "# terok-vault-version:"),
        (_gate_pkg, "terok-gate@.service", "# terok-gate-version:"),
        (_gate_pkg, "terok-gate-socket.service", "# terok-gate-version:"),
        (_gate_pkg, "terok-gate.socket", "# terok-gate-version:"),
    ],
)
def test_version_marker_is_on_line_1(package, unit_name: str, marker_prefix: str) -> None:
    first_line = _template_path(package, unit_name).read_text().splitlines()[0]
    assert first_line.startswith(marker_prefix), (
        f"{unit_name}: expected line 1 to start with {marker_prefix!r}, got {first_line!r}"
    )
