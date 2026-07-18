# SPDX-FileCopyrightText: 2026 Jiri Vyskocil
# SPDX-License-Identifier: Apache-2.0

"""Vulture whitelist — suppress false-positive dead-code findings."""

# Re-exported types in shield.py for terok consumers
_ = NftNotFoundError  # type: ignore[name-defined]  # noqa: F821
_ = EnvironmentCheck  # type: ignore[name-defined]  # noqa: F821
_ = ShieldNeedsSetup  # type: ignore[name-defined]  # noqa: F821
_ = ShieldState  # type: ignore[name-defined]  # noqa: F821

# cli._VersionAction.__call__ — parameter names fixed by the argparse.Action interface
_ = namespace  # type: ignore[name-defined]  # noqa: F821
_ = option_string  # type: ignore[name-defined]  # noqa: F821
