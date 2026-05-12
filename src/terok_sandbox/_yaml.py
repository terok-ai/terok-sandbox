# SPDX-FileCopyrightText: 2026 Jiri Vyskocil
# SPDX-License-Identifier: Apache-2.0

"""Round-trip YAML write-back for the few sandbox sites that mutate config.yml.

Mirrors the round-trip facade in ``terok.lib.util.yaml`` — sandbox
can't import from terok, so the two stay duplicated; keep in sync if
the parser race-fix comment ever changes.

A fresh ``YAML()`` per call: the class carries parser state, so a
module-level singleton races under concurrent use from TUI worker
threads.
"""

from __future__ import annotations

import os
import tempfile
from io import StringIO
from pathlib import Path
from typing import Any

from ruamel.yaml import YAML

__all__ = ["update_section", "write_secret_text"]


def update_section(path: Path, section: str, updates: dict[str, Any]) -> None:
    """Merge *updates* into ``data[section]`` at *path*, preserving comments."""
    yaml = YAML(typ="rt")
    yaml.preserve_quotes = True
    if path.exists():
        existing = yaml.load(path.read_text(encoding="utf-8")) or {}
    else:
        path.parent.mkdir(parents=True, exist_ok=True, mode=0o700)
        existing = {}
    if not isinstance(existing, dict):
        raise ValueError(
            f"{path} top-level is {type(existing).__name__}, expected a mapping;"
            " refusing to silently overwrite — fix or move aside the file by hand"
        )
    # ``setdefault`` returns whatever sits at the key — a stale scalar
    # written by a previous schema version would explode on ``.update``.
    if not isinstance(existing.get(section), dict):
        existing[section] = {}
    existing[section].update(updates)
    buf = StringIO()
    yaml.dump(existing, buf)
    write_secret_text(path, buf.getvalue())


def write_secret_text(path: Path, text: str) -> None:
    """Atomically create-or-replace *path* with mode 0o600.

    Writes a temp file in the same directory, fsyncs, then
    ``os.replace``s it onto *path*.  A crash mid-write leaves either
    the old file intact or no file at all — never a truncated one.
    ``mkstemp`` opens with mode 0o600 from the kernel, so a passphrase
    never has a window of world-readable visibility.
    """
    path.parent.mkdir(parents=True, exist_ok=True, mode=0o700)
    fd, tmp_name = tempfile.mkstemp(prefix=path.name + ".", dir=path.parent)
    tmp_path = Path(tmp_name)
    try:
        # POSIX ``write`` is allowed to return a short count; loop until
        # the full payload is committed.  A truncated config.yml or
        # vault.passphrase would lock the operator out of the vault.
        data = memoryview(text.encode("utf-8"))
        while data:
            data = data[os.write(fd, data) :]
        os.fsync(fd)
        os.close(fd)
        fd = -1
        os.replace(tmp_path, path)
    except BaseException:
        if fd >= 0:
            os.close(fd)
        tmp_path.unlink(missing_ok=True)
        raise
