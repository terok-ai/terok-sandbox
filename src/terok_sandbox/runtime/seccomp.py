# SPDX-FileCopyrightText: 2026 Jiri Vyskocil
# SPDX-License-Identifier: Apache-2.0

"""Narrow ASLR control without replacing the host's syscall policy.

ThreadSanitizer queries the current personality, adds ``ADDR_NO_RANDOMIZE``,
and re-executes when its shadow-memory layout conflicts with ASLR. Grant only
that flag on the Linux personalities supported by the containers/common default.
Never permit arbitrary personality bits such as ``READ_IMPLIES_EXEC``.
"""

import json
import tempfile
from pathlib import Path

from .podman import default_seccomp_profile

_ADDR_NO_RANDOMIZE = 0x40000
_PERSONALITY_QUERY = 0xFFFFFFFF
_LINUX_PERSONALITIES = (0, 8, 0x20000, 0x20008)
"""PER_LINUX/PER_LINUX32, with or without ADDR_COMPAT_LAYOUT."""
_PROFILE_NAME = "seccomp-aslr.json"


def aslr_control_args(state_dir: Path) -> list[str]:
    """Write a per-container profile and return its Podman security option.

    Preserve the host profile, including architecture and capability conditions.
    Explicit personality denials are not overridden: combining them with allow
    rules has ambiguous/conflicting semantics, so refuse that custom policy.
    The file stays in host-side state for container restarts, outside its mounts.
    """
    source = default_seccomp_profile()
    try:
        profile = json.loads(source.read_text())
        if profile["defaultAction"] != "SCMP_ACT_ERRNO":
            raise ValueError("expected a default-deny SCMP_ACT_ERRNO profile")
        rules = profile["syscalls"]
        if any(
            "personality" in rule["names"] and rule["action"] != "SCMP_ACT_ALLOW" for rule in rules
        ):
            raise ValueError("the host profile explicitly denies personality; review it first")
        rules.extend(
            {
                "names": ["personality"],
                "action": "SCMP_ACT_ALLOW",
                "args": [{"index": 0, "value": value, "op": "SCMP_CMP_EQ"}],
            }
            for value in (
                _PERSONALITY_QUERY,
                *(base | _ADDR_NO_RANDOMIZE for base in _LINUX_PERSONALITIES),
            )
        )
        state_dir.mkdir(parents=True, exist_ok=True, mode=0o700)
        destination = state_dir / _PROFILE_NAME
        with tempfile.TemporaryDirectory(dir=state_dir) as staging:
            staged = Path(staging) / _PROFILE_NAME
            staged.write_text(json.dumps(profile, indent=2) + "\n")
            staged.replace(destination)
    except (OSError, ValueError, KeyError, TypeError) as exc:
        raise SystemExit(
            f"run.aslr_control: cannot extend seccomp profile {source}: {exc}"
        ) from exc
    return ["--security-opt", f"seccomp={destination}"]
