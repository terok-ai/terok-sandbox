# SPDX-FileCopyrightText: 2026 Jiri Vyskocil
# SPDX-License-Identifier: Apache-2.0

"""Direct tests for the shared PID-file safety helpers.

The vault and gate lifecycles both reach for these helpers; their
integration tests cover the call-site contract, but the helper's own
failure-path branches (invalid contents, OSError variants) are
testable in isolation here without spinning up either manager.
"""

from __future__ import annotations

from pathlib import Path
from unittest.mock import patch

from terok_sandbox._util import read_pidfile_safely, unlink_pidfile_safely


class TestReadPidfileSafely:
    """Failure-path branches that ``stop_daemon`` / ``is_daemon_running`` exercise transitively."""

    def test_invalid_pid_content_returns_none(self, tmp_path: Path) -> None:
        """A pidfile that doesn't parse to an int returns ``None`` rather than crashing."""
        pidfile = tmp_path / "vault.pid"
        pidfile.write_text("not-a-pid\n")
        assert read_pidfile_safely(pidfile) is None

    def test_missing_file_returns_none(self, tmp_path: Path) -> None:
        """A missing pidfile path returns ``None`` (the common idle-state case)."""
        assert read_pidfile_safely(tmp_path / "does-not-exist.pid") is None

    def test_oserror_on_open_returns_none(self, tmp_path: Path) -> None:
        """Any other OSError on open (e.g. permission denied, ELOOP via O_NOFOLLOW)."""
        pidfile = tmp_path / "vault.pid"
        with patch("os.open", side_effect=PermissionError("denied")):
            assert read_pidfile_safely(pidfile) is None

    def test_symlink_returns_none(self, tmp_path: Path) -> None:
        """A symlinked pidfile is refused, not followed (CWE-59 guard).

        The target file contains a parseable PID so a regression that
        followed the symlink would return ``42`` instead of ``None`` —
        proves the ``O_NOFOLLOW`` open is doing what its docstring says.
        """
        target = tmp_path / "innocent"
        target.write_text("42")
        pidfile = tmp_path / "vault.pid"
        pidfile.symlink_to(target)
        assert read_pidfile_safely(pidfile) is None

    def test_non_regular_returns_none(self, tmp_path: Path) -> None:
        """A directory at the pidfile path is refused as non-regular."""
        pidfile = tmp_path / "vault.pid"
        pidfile.mkdir()
        assert read_pidfile_safely(pidfile) is None

    def test_valid_pid_returns_int(self, tmp_path: Path) -> None:
        """The happy path: a regular file with an integer content returns the int."""
        pidfile = tmp_path / "vault.pid"
        pidfile.write_text("12345\n")
        assert read_pidfile_safely(pidfile) == 12345


class TestUnlinkPidfileSafely:
    """`unlink_pidfile_safely` refuses symlinks; covers every error branch directly."""

    def test_regular_file_is_unlinked(self, tmp_path: Path) -> None:
        """The happy path: a regular file is removed."""
        pidfile = tmp_path / "vault.pid"
        pidfile.write_text("42")
        unlink_pidfile_safely(pidfile)
        assert not pidfile.exists()

    def test_missing_file_is_no_op(self, tmp_path: Path) -> None:
        """A missing pidfile is silently accepted — this is cleanup, not a critical path."""
        unlink_pidfile_safely(tmp_path / "does-not-exist.pid")  # must not raise

    def test_lstat_oserror_is_no_op(self, tmp_path: Path) -> None:
        """An OSError from lstat (e.g. permission on the parent dir) is silently accepted."""
        pidfile = tmp_path / "vault.pid"
        pidfile.write_text("42")
        with patch("os.lstat", side_effect=PermissionError("denied")):
            unlink_pidfile_safely(pidfile)  # must not raise
        # The file still exists because the lstat check short-circuited.
        assert pidfile.exists()

    def test_symlink_is_refused(self, tmp_path: Path) -> None:
        """A symlinked pidfile is left untouched (CWE-59 guard)."""
        target = tmp_path / "innocent"
        target.write_text("payload")
        pidfile = tmp_path / "vault.pid"
        pidfile.symlink_to(target)

        unlink_pidfile_safely(pidfile)
        assert pidfile.is_symlink()
        assert target.exists()
        assert target.read_text() == "payload"

    def test_non_regular_file_is_refused(self, tmp_path: Path) -> None:
        """A directory (or other non-regular) at the pidfile path is refused."""
        pidfile = tmp_path / "vault.pid"
        pidfile.mkdir()
        unlink_pidfile_safely(pidfile)  # must not raise (or rmdir!)
        assert pidfile.is_dir()

    def test_toctou_unlink_failure_is_swallowed(self, tmp_path: Path) -> None:
        """The file vanishes (or denies unlink) between lstat and unlink — accepted silently."""
        pidfile = tmp_path / "vault.pid"
        pidfile.write_text("42")
        with patch("os.unlink", side_effect=FileNotFoundError):
            unlink_pidfile_safely(pidfile)  # must not raise

    def test_unlink_oserror_is_swallowed(self, tmp_path: Path) -> None:
        """A non-FileNotFoundError OSError on unlink (e.g. EBUSY) is also swallowed."""
        pidfile = tmp_path / "vault.pid"
        pidfile.write_text("42")
        with patch("os.unlink", side_effect=OSError(16, "Device or resource busy")):
            unlink_pidfile_safely(pidfile)  # must not raise
