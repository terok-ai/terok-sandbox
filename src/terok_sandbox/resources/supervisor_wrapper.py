#!/usr/bin/env python3

# SPDX-FileCopyrightText: 2026 Jiri Vyskocil
# SPDX-License-Identifier: Apache-2.0
"""Restart loop for the terok per-container supervisor.

Invoked by the OCI hook with ``argv = [container_id, sidecar_path]``.
Spawns ``terok-sandbox supervisor <id> <sidecar_path>``; on clean
exit (rc 0) we stop; on non-zero exit we retry with exponential
backoff capped at five attempts.

Stdlib-only by design (matches the OCI hook ballast): runs under the
operator's system ``/usr/bin/python3`` rather than inside any
virtualenv.  The absolute path to the ``terok-sandbox`` entry point
is baked into ``_SANDBOX_BIN_ARGV`` at install time by
[`install_supervisor_hooks`][terok_sandbox.supervisor.install.install_supervisor_hooks].
"""

import subprocess  # nosec B404
import sys
import time

#: Resolved entry-point argv for ``terok-sandbox``, rendered at install
#: time.  May be a single absolute path (when a console script is on
#: PATH) or ``[python3, -m, terok_sandbox]`` (editable installs).  The
#: ``["__TEROK_SANDBOX_BIN__"]`` list literal is the unique marker the
#: installer replaces; the standalone sentinel string below is used by
#: ``main()`` to detect a wrapper that was never rendered.
_SANDBOX_BIN_ARGV: list[str] = ["__TEROK_SANDBOX_BIN__"]
_UNRENDERED_SENTINEL = "__" + "TEROK_SANDBOX_BIN" + "__"  # not a replace target

_MAX_ATTEMPTS = 5
_BACKOFF_CAP_SECONDS = 60


def main() -> int:
    """Drive the restart loop; return the supervisor's last exit code."""
    if len(sys.argv) < 3:
        print(
            "usage: supervisor_wrapper.py <container_id> <sidecar_path>",
            file=sys.stderr,
        )
        return 2
    if _UNRENDERED_SENTINEL in _SANDBOX_BIN_ARGV:
        print(
            "supervisor_wrapper: unrendered template — rerun `terok-sandbox setup`",
            file=sys.stderr,
        )
        return 2
    container_id, sidecar_path = sys.argv[1], sys.argv[2]

    argv = [*_SANDBOX_BIN_ARGV, "supervisor", container_id, sidecar_path]
    attempts = 0
    rc = 1
    while attempts < _MAX_ATTEMPTS:
        rc = subprocess.call(argv)  # noqa: S603  # nosec B603 — argv is a fixed list
        if rc == 0:
            return 0
        attempts += 1
        time.sleep(min(2**attempts, _BACKOFF_CAP_SECONDS))
    return rc


if __name__ == "__main__":  # pragma: no cover
    sys.exit(main())
