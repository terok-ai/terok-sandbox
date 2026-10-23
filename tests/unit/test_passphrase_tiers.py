# SPDX-FileCopyrightText: 2026 Jiri Vyskocil
# SPDX-License-Identifier: Apache-2.0

"""Human-facing passphrase tier labels stay distinct from stable machine IDs."""

import pytest

import terok_sandbox
from terok_sandbox import PassphraseTier


@pytest.mark.parametrize(
    ("value", "display_name"),
    [
        ("desktop-keyring", "desktop keyring"),
        ("session-cache", "session cache"),
        ("systemd-creds", "systemd-creds"),
        ("passphrase-command", "passphrase-command"),
        ("prompt", "prompt"),
    ],
)
def test_display_names_preserve_machine_ids(value: str, display_name: str) -> None:
    """CLI/config and JSON keep their IDs while operators see explicit labels."""
    tier = PassphraseTier(value)
    assert str(tier) == value
    assert tier.display_name == display_name


@pytest.mark.parametrize("old_value", ["keyring", "kernel-keyring"])
def test_ambiguous_tier_ids_are_rejected(old_value: str) -> None:
    """Old tier spellings are not compatibility aliases for the explicit vocabulary."""
    with pytest.raises(ValueError):
        PassphraseTier(old_value)


def test_tier_members_have_no_legacy_aliases() -> None:
    """The temporary cache is not mislabeled as a kernel-only backing."""
    assert "KEYRING" not in PassphraseTier.__members__
    assert "KERNEL_KEYRING" not in PassphraseTier.__members__
    assert PassphraseTier.DESKTOP_KEYRING.value == "desktop-keyring"
    assert PassphraseTier.SESSION_CACHE.value == "session-cache"


@pytest.mark.parametrize("old_name", ["keyring_backend_available", "handle_vault_to_keyring"])
def test_ambiguous_public_helpers_have_no_aliases(old_name: str) -> None:
    """The public package exposes only the explicit desktop-keyring helper names."""
    assert old_name not in terok_sandbox.__all__
    assert not hasattr(terok_sandbox, old_name)
    assert callable(terok_sandbox.desktop_keyring_backend_available)
    assert callable(terok_sandbox.handle_vault_to_desktop_keyring)
