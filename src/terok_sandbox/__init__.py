# SPDX-FileCopyrightText: 2025 Jiri Vyskocil
# SPDX-License-Identifier: Apache-2.0

"""terok-sandbox: hardened Podman container runner with gate and shield integration.

Public API for standalone use and integration with terok.
"""

__version__: str = "0.0.0"  # placeholder; replaced at build time

from importlib.metadata import PackageNotFoundError, version as _meta_version

try:
    __version__ = _meta_version("terok-sandbox")
except PackageNotFoundError:
    pass  # editable install or running from source without metadata

__all__ = ["__version__"]
