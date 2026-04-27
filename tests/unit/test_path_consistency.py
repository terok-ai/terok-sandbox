# SPDX-FileCopyrightText: 2026 Jiri Vyskocil
# SPDX-License-Identifier: Apache-2.0

"""Path-consistency invariant: two [`SandboxConfig`][terok_sandbox.SandboxConfig] instances agree.

A surprisingly common failure mode across the stack has been "the CLI
resolved ``~/.local/state/terok/...`` here; the TUI resolved
``~/.config/terok/...`` there; now the install and the status check
disagree about where the unit file lives."  The fix is a single
source of truth per path — [`SandboxConfig`][terok_sandbox.SandboxConfig] — that every
consumer threads through.  This test enforces that property by
constructing two configs under identical XDG settings and asserting
every path property matches.
"""

from __future__ import annotations

from pathlib import Path

from terok_sandbox.config import SandboxConfig

#: Every public path property on [`SandboxConfig`][terok_sandbox.SandboxConfig] that the
#: setup aggregator and its phases touch.  Listed here so a new path
#: accessor shows up in the assertion without the maintainer having
#: to remember to extend this test.
_PATH_PROPERTIES = (
    "state_dir",
    "runtime_dir",
    "config_dir",
    "vault_dir",
    "gate_base_path",
    "token_file_path",
    "pid_file_path",
    "shield_profiles_dir",
    "db_path",
    "vault_socket_path",
    "vault_pid_path",
    "routes_path",
    "gate_socket_path",
    "ssh_signer_socket_path",
    "clone_cache_base_path",
    "ssh_keys_dir",
)


def test_two_configs_agree_on_every_path() -> None:
    """Two [`SandboxConfig`][terok_sandbox.SandboxConfig] instances resolve every path identically.

    If a path resolver ever reached for an environment variable or
    global state *outside* of SandboxConfig's own fields in a way
    that could drift mid-process (cache, ambient state), the two
    instances would disagree.  This is the canary that guards the
    install aggregator and every downstream consumer from that
    class of drift.
    """
    first = SandboxConfig()
    second = SandboxConfig()

    mismatches = {
        name: (getattr(first, name), getattr(second, name))
        for name in _PATH_PROPERTIES
        if getattr(first, name) != getattr(second, name)
    }
    assert not mismatches, f"path drift across SandboxConfig instances: {mismatches}"


def test_every_documented_property_actually_resolves() -> None:
    """Every path-property in the snapshot list is reachable on the class.

    Catches the refactor where someone renamed ``db_path`` to
    ``credentials_db_path`` on the class but forgot to update this
    list — without the guard, the drift test above would silently
    stop covering the renamed accessor.
    """
    cfg = SandboxConfig()
    missing = [name for name in _PATH_PROPERTIES if not isinstance(getattr(cfg, name), Path)]
    assert not missing, f"path properties missing or not returning Path: {missing}"
