# SPDX-FileCopyrightText: 2026 Jiri Vyskocil
# SPDX-License-Identifier: Apache-2.0

"""Vulture whitelist — suppress false-positive dead-code findings."""

# Re-exported types in shield.py for terok consumers
_ = NftNotFoundError  # type: ignore[name-defined]  # noqa: F821
_ = EnvironmentCheck  # type: ignore[name-defined]  # noqa: F821
_ = ShieldNeedsSetup  # type: ignore[name-defined]  # noqa: F821
_ = ShieldState  # type: ignore[name-defined]  # noqa: F821
