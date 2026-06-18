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

import re
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

#: A podman container id (or, when an operator debugs by hand, a name):
#: one leading alphanumeric then the podman id/name charset.  The anchored
#: allow-list is what lets ``main()`` hand ``container_id`` to the
#: ``supervisor`` verb safely — a value starting with ``-`` would otherwise
#: be read as an option flag (CWE-88 argument injection), and ``--`` can't
#: rescue it because the CLI reserves that separator for ``run`` alone.
_SAFE_CONTAINER_ID = re.compile(r"\A[0-9A-Za-z][0-9A-Za-z_.-]*\Z")


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

    # Validate both positionals before they reach the CLI.  The OCI hook
    # only ever passes podman's hex id and an absolute, pre-validated
    # sidecar path, but the wrapper trusts whatever argv it is handed — so
    # gate it here: an id or path starting with ``-`` would be parsed as an
    # option by the ``supervisor`` verb's argparse rather than a positional
    # (argument injection), and a relative sidecar path would resolve
    # against the hook's cwd instead of the intended file.
    if not _SAFE_CONTAINER_ID.match(container_id):
        print(
            f"supervisor_wrapper: refusing unsafe container_id: {container_id!r}",
            file=sys.stderr,
        )
        return 2
    if not sidecar_path.startswith("/"):
        print(
            f"supervisor_wrapper: sidecar_path must be absolute: {sidecar_path!r}",
            file=sys.stderr,
        )
        return 2

    argv = [*_SANDBOX_BIN_ARGV, "supervisor", container_id, sidecar_path]
    attempts = 0
    rc = 1
    while attempts < _MAX_ATTEMPTS:
        try:
            rc = subprocess.call(argv)  # noqa: S603  # nosec B603 — fixed verb list, positionals validated above
        except FileNotFoundError:
            # Baked entry point vanished after install — fail controlled
            # rather than letting the wrapper crash mid-restart-loop.
            print(
                f"supervisor_wrapper: entry point not found: {_SANDBOX_BIN_ARGV[0]}",
                file=sys.stderr,
            )
            return 127
        if rc == 0:
            return 0
        attempts += 1
        if attempts >= _MAX_ATTEMPTS:
            # No more retries — don't sleep before returning.
            break
        time.sleep(min(2**attempts, _BACKOFF_CAP_SECONDS))
    return rc


if __name__ == "__main__":  # pragma: no cover
    sys.exit(main())
