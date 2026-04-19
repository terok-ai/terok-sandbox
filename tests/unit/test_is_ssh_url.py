# SPDX-FileCopyrightText: 2026 Jiri Vyskocil
# SPDX-License-Identifier: Apache-2.0

"""Covers the SSH-URL predicate shared by gate and callers.

``is_ssh_url`` gates the deploy-key / fallback UX in both terok-main and
the gate's env builder, so misclassifying an scp-style remote silently
steers users onto the wrong path.  The cases below pin down every
form git itself accepts as SSH.
"""

from __future__ import annotations

import pytest

from terok_sandbox.gate.mirror import is_ssh_url


@pytest.mark.parametrize(
    "url",
    [
        "git@github.com:sliwowitz/terok.git",
        "deploy@host:repo.git",
        "user@host.example.com:/abs/path/repo.git",
        "github.com:org/repo.git",  # scp-style without a user part
        "ssh://git@github.com/sliwowitz/terok.git",
        "SSH://user@host/repo.git",  # case-insensitive scheme
    ],
)
def test_recognized_as_ssh(url: str) -> None:
    """Every git-accepted SSH form returns ``True``."""
    assert is_ssh_url(url) is True


@pytest.mark.parametrize(
    "url",
    [
        None,
        "",
        "https://github.com/sliwowitz/terok.git",
        "http://host/repo.git",
        "http://example.com:80/path",  # colon-port must not read as host:path
        "file:///tmp/repo.git",
        "/tmp/local/repo.git",
        "C:\\Users\\dev\\repo.git",  # Windows path must not match scp-style
    ],
)
def test_not_ssh(url: str | None) -> None:
    """Non-SSH inputs return ``False``."""
    assert is_ssh_url(url) is False
