# SPDX-FileCopyrightText: 2026 Jiri Vyskocil
# SPDX-License-Identifier: Apache-2.0

"""Sandbox-owned git hooks the gate server injects into every agent push.

The gate's HTTP server historically pointed ``core.hooksPath`` at
``/dev/null`` so that no hook inside a gate repo could ever run — repo
content must never become host-side code.  This module keeps that
property while inverting the mechanism: hooks now live in a directory the
sandbox owns *outside* every gate repo, rendered from the constants below
at server start.  Repo content still can't inject code; the operator side
gains exactly one enforcement point.

One hook is installed, ``post-receive``, and it does two things:

- **Backs up destructive agent updates.**  For every pushed ref under
  ``refs/heads/`` that was force-moved or deleted, the old tip is saved
  as ``refs/terok/backup/<branch>/<stamp>-<sha12>`` — the same scheme the
  sync model uses, so ``list_backups()`` / ``prune_backups()`` cover both
  sides with one retention policy.  A backup is a ref, not a copy: the
  objects it pins arrived with the original push and are already
  reflog-retained, so the net-new storage is the ref file itself.
- **Writes the push marker.**  Every push overwrites
  ``$GIT_DIR/terok-push-marker`` with a timestamp and the updated refs,
  so the host can watch one file's mtime instead of polling refs.

Why ``post-receive`` and not ``update``: git quarantines pushed objects
while ``pre-receive``/``update`` run and refuses all ref updates during
that window, so the backup ref cannot be written there.  By
``post-receive`` the quarantine is over — and the old tips being backed
up predate the push anyway.  The cost is that a failed backup can only
warn (the push is already accepted), not reject; the warning goes to the
agent's push output and into the marker file, and the always-on reflog
remains the last-resort trail.

Identity needs no plumbing at all: only agent pushes traverse the HTTP
server, so everything these hooks see is agent-side by construction.
Operator pushes from the host use the gate repo's own (empty) hooks dir
and behave as before.
"""

from __future__ import annotations

from pathlib import Path

from terok_sandbox.gate.mirror import (
    _BACKUP_PREFIX,
    _BACKUP_STAMP_FORMAT,
    _HEADS_PREFIX,
    _ZERO_SHA,
)

#: Directory under the mirror root holding the injected hooks.  The dot
#: prefix keeps it out of every ``<repo>.git`` route the server accepts.
HOOKS_DIRNAME = ".terok-hooks"

#: File the ``post-receive`` hook (over)writes inside the bare gate repo on
#: every agent push — watch its mtime to react to pushes without polling.
PUSH_MARKER_FILENAME = "terok-push-marker"

_POST_RECEIVE = f'''#!/usr/bin/env python3
# SPDX-FileCopyrightText: 2026 Jiri Vyskocil
# SPDX-License-Identifier: Apache-2.0
"""Gate post-receive hook: back up destructive agent updates, mark the push.

Rendered by terok-sandbox (gate/hooks.py) — do not edit in place.
"""
import os
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path

ZERO = "{_ZERO_SHA}"
HEADS = "{_HEADS_PREFIX}"
BACKUPS = "{_BACKUP_PREFIX}"
STAMP = "{_BACKUP_STAMP_FORMAT}"
MARKER = "{PUSH_MARKER_FILENAME}"


def is_ancestor(old: str, new: str) -> bool:
    """True when *old*..*new* is a fast-forward."""
    res = subprocess.run(
        ["git", "merge-base", "--is-ancestor", old, new],
        capture_output=True,
    )
    return res.returncode == 0


def back_up(branch: str, old: str) -> str | None:
    """Pin *old* under a timestamped backup ref; return its name or None."""
    stamp = datetime.now(timezone.utc).strftime(STAMP)
    ref = BACKUPS + branch + "/" + stamp + "-" + old[:12]
    res = subprocess.run(["git", "update-ref", ref, old], capture_output=True)
    return ref if res.returncode == 0 else None


def main() -> int:
    """Process the pushed-ref lines on stdin; never block the push."""
    lines = sys.stdin.read().splitlines()
    notes = []
    for line in lines:
        old, new, refname = line.split(" ", 2)
        if not refname.startswith(HEADS) or old == ZERO:
            continue
        if new != ZERO and is_ancestor(old, new):
            continue
        branch = refname[len(HEADS):]
        if ref := back_up(branch, old):
            print("gate: saved " + ref, file=sys.stderr)
            notes.append("backed-up " + ref)
        else:
            print(
                "gate: WARNING could not back up " + branch + " (" + old[:12] + ") "
                "- old tip remains in the reflog only",
                file=sys.stderr,
            )
            notes.append("backup-FAILED " + branch + " " + old)
    marker = Path(os.environ.get("GIT_DIR", ".")) / MARKER
    marker.write_text(
        "\\n".join([datetime.now(timezone.utc).isoformat(), *lines, *notes]) + "\\n",
        encoding="utf-8",
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
'''


def hooks_dir_for(mirror_root: Path) -> Path:
    """Return the sandbox-owned hooks directory for *mirror_root*."""
    return mirror_root / HOOKS_DIRNAME


def install_hooks(hooks_dir: Path) -> None:
    """Idempotently render the hook scripts into *hooks_dir*.

    Writes are atomic (tmp file + rename) because several per-container
    gate servers may share one mirror root and race here; content-equal
    installs are skipped so repeated server starts never churn mtimes.
    """
    hooks_dir.mkdir(parents=True, exist_ok=True)
    for name, content in (("post-receive", _POST_RECEIVE),):
        target = hooks_dir / name
        try:
            if target.read_text(encoding="utf-8") == content:
                continue
        except (FileNotFoundError, UnicodeDecodeError):
            pass
        tmp = hooks_dir / f".{name}.tmp"
        tmp.write_text(content, encoding="utf-8")
        tmp.chmod(0o755)
        tmp.rename(target)
