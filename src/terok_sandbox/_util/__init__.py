# SPDX-FileCopyrightText: 2025 Jiri Vyskocil
# SPDX-License-Identifier: Apache-2.0

"""Vendored utility functions for filesystem, templates, logging, naming, and sanitization."""

from ._fs import ensure_dir, ensure_dir_writable, systemd_user_unit_dir, write_sensitive_file
from ._logging import BestEffortLogger, log_debug, log_warning, warn_user
from ._naming import effective_ssh_key_name
from ._sanitize import sanitize_tty
from ._templates import render_template

__all__ = [
    "BestEffortLogger",
    "effective_ssh_key_name",
    "ensure_dir",
    "ensure_dir_writable",
    "log_debug",
    "log_warning",
    "render_template",
    "sanitize_tty",
    "systemd_user_unit_dir",
    "warn_user",
    "write_sensitive_file",
]
