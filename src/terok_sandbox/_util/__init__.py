# SPDX-FileCopyrightText: 2025 Jiri Vyskocil
# SPDX-FileCopyrightText: 2026 Jiri Vyskocil
# SPDX-License-Identifier: Apache-2.0

"""Vendored utility functions for filesystem, templates, logging, naming, and sanitization."""

from terok_util import (
    ensure_dir,
    ensure_dir_writable,
    render_template,
    sanitize_tty,
    write_sensitive_file,
)

from ._fs import systemd_user_unit_dir
from ._logging import BestEffortLogger, log_debug, log_warning, warn_user
from ._naming import effective_ssh_key_name
from ._pidfile import read_pidfile_safely, unlink_pidfile_safely
from ._templates import systemd_escape, systemd_exec_argv

__all__ = [
    "BestEffortLogger",
    "effective_ssh_key_name",
    "ensure_dir",
    "ensure_dir_writable",
    "log_debug",
    "log_warning",
    "read_pidfile_safely",
    "render_template",
    "sanitize_tty",
    "systemd_escape",
    "systemd_exec_argv",
    "systemd_user_unit_dir",
    "unlink_pidfile_safely",
    "warn_user",
    "write_sensitive_file",
]
