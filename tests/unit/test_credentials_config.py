# SPDX-FileCopyrightText: 2026 Jiri Vyskocil
# SPDX-License-Identifier: Apache-2.0

"""Passphrase-source policy stays fresh across processes and fails closed."""

from __future__ import annotations

import os
import subprocess
import sys
import traceback
from pathlib import Path

import pytest
from terok_util import read_config_section

from terok_sandbox import config
from terok_sandbox._yaml import update_section
from terok_sandbox.commands.credentials import _persist_mode_choice
from terok_sandbox.vault.store.tiers import PassphraseTier

_REAL_CREDENTIALS_USE_DESKTOP_KEYRING = config.credentials_use_desktop_keyring
_OLD_COMMAND = "pass show previous-vault"
_NEW_COMMAND = "pass show current-vault"
_SECRET_SENTINEL = "never-include-this-value-in-diagnostics"


@pytest.fixture(autouse=True)
def _restore_credentials_reader(monkeypatch: pytest.MonkeyPatch) -> None:
    """Exercise policy reads instead of the suite's desktop-keyring opt-in stub."""
    monkeypatch.setattr(
        config, "credentials_use_desktop_keyring", _REAL_CREDENTIALS_USE_DESKTOP_KEYRING
    )


def test_ambiguous_config_key_is_rejected(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """The old key cannot silently opt into the desktop keyring after a rename."""
    config_file = tmp_path / "config.yml"
    config_file.write_text("credentials:\n  use_keyring: false\n")
    monkeypatch.setenv("TEROK_CONFIG_FILE", str(config_file))
    with pytest.raises(RuntimeError, match="Cannot read or validate credentials configuration"):
        config.credentials_use_desktop_keyring()


@pytest.mark.parametrize("external_update", [False, True], ids=["same-process", "child-process"])
@pytest.mark.parametrize("initial_desktop_keyring", [False, True], ids=["enable", "disable"])
def test_config_refreshes_after_persisted_change(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    external_update: bool,
    initial_desktop_keyring: bool,
) -> None:
    """A running parent sees changed policy even after both old caches were primed."""
    config_file = tmp_path / "config.yml"
    monkeypatch.setenv("TEROK_CONFIG_FILE", str(config_file))
    update_section(
        config_file,
        "credentials",
        {"use_desktop_keyring": initial_desktop_keyring, "passphrase_command": _OLD_COMMAND},
    )
    assert read_config_section("credentials")["use_desktop_keyring"] == str(initial_desktop_keyring)
    original = config.SandboxConfig()
    assert original.credentials_use_desktop_keyring is initial_desktop_keyring
    assert original.credentials_passphrase_command == _OLD_COMMAND

    if external_update:
        subprocess.run(
            [
                sys.executable,
                "-c",
                "import sys; from pathlib import Path; "
                "from terok_sandbox._yaml import update_section; "
                "update_section(Path(sys.argv[1]), 'credentials', "
                "{'use_desktop_keyring': sys.argv[3] == 'True', 'passphrase_command': sys.argv[2]})",
                str(config_file),
                _NEW_COMMAND,
                str(not initial_desktop_keyring),
            ],
            check=True,
            capture_output=True,
            text=True,
            timeout=10,
        )
    else:
        update_section(
            config_file,
            "credentials",
            {
                "use_desktop_keyring": not initial_desktop_keyring,
                "passphrase_command": _NEW_COMMAND,
            },
        )

    refreshed = config.SandboxConfig()
    assert refreshed.credentials_use_desktop_keyring is not initial_desktop_keyring
    assert refreshed.credentials_passphrase_command == _NEW_COMMAND
    assert original.credentials_use_desktop_keyring is initial_desktop_keyring
    assert original.credentials_passphrase_command == _OLD_COMMAND


def test_credentials_follow_layered_config_and_null_deletion(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Fresh reads retain system defaults, user overrides, and explicit deletions."""
    system_config = tmp_path / "system.yml"
    user_config = tmp_path / "user.yml"
    system_config.write_text(
        f"credentials:\n  use_desktop_keyring: false\n  passphrase_command: {_OLD_COMMAND}\n"
    )
    user_config.write_text("credentials:\n  passphrase_command: null\n")
    monkeypatch.setattr(
        "terok_sandbox.paths.config_file_paths",
        lambda: [("system", system_config), ("user", user_config)],
    )

    current = config.SandboxConfig()
    assert current.credentials_use_desktop_keyring is False
    assert current.credentials_passphrase_command is None

    update_section(user_config, "credentials", {"use_desktop_keyring": True})
    assert config.SandboxConfig().credentials_use_desktop_keyring is True


@pytest.mark.parametrize(
    "content",
    [
        f"credentials: [\n  {_SECRET_SENTINEL}\n",
        f"credentials: [{_SECRET_SENTINEL}]\n",
        f"credentials:\n  use_desktop_keyring: {_SECRET_SENTINEL}\n",
        f"credentials:\n  unknown_setting: {_SECRET_SENTINEL}\n",
        f"credentials:\n  passphrase: {_SECRET_SENTINEL}\n",
        f"[{_SECRET_SENTINEL}]\n",
    ],
    ids=["invalid-yaml", "wrong-shape", "invalid-value", "unknown-key", "plaintext", "non-mapping"],
)
def test_invalid_configuration_never_enables_a_source_or_discloses_values(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
    content: str,
) -> None:
    """Parser and schema errors fail closed without echoing secret-bearing input."""
    config_file = tmp_path / "config.yml"
    config_file.write_text(content)
    monkeypatch.setenv("TEROK_CONFIG_FILE", str(config_file))

    with pytest.raises(RuntimeError, match="credentials configuration") as error:
        config.SandboxConfig()

    assert _SECRET_SENTINEL not in "".join(traceback.format_exception(error.value))
    captured = capsys.readouterr()
    assert _SECRET_SENTINEL not in captured.out + captured.err


def test_unreadable_configuration_never_enables_a_source(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A read failure cannot turn an explicit desktop-keyring opt-out into opt-in."""
    config_file = tmp_path / "config.yml"
    config_file.write_text("credentials:\n  use_desktop_keyring: false\n")
    monkeypatch.setenv("TEROK_CONFIG_FILE", str(config_file))
    assert config.SandboxConfig().credentials_use_desktop_keyring is False

    def deny_read(*_args: object, **_kwargs: object) -> str:
        """Simulate read denial without relying on the test runner's UID."""
        raise PermissionError(_SECRET_SENTINEL)

    monkeypatch.setattr(Path, "read_text", deny_read)
    with pytest.raises(RuntimeError, match="credentials configuration") as error:
        config.SandboxConfig()
    assert _SECRET_SENTINEL not in "".join(traceback.format_exception(error.value))


def test_directory_is_not_treated_as_absent_configuration(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """An existing non-file config path fails closed instead of enabling defaults."""
    monkeypatch.setenv("TEROK_CONFIG_FILE", str(tmp_path))
    with pytest.raises(RuntimeError, match="credentials configuration"):
        config.SandboxConfig()


def test_missing_configuration_preserves_defaults(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """An unprovisioned config file remains a valid default configuration."""
    monkeypatch.setenv("TEROK_CONFIG_FILE", str(tmp_path / "not-created.yml"))
    current = config.SandboxConfig()
    assert current.credentials_use_desktop_keyring is True
    assert current.credentials_passphrase_command is None


def test_null_device_override_preserves_raw_mode_defaults(monkeypatch: pytest.MonkeyPatch) -> None:
    """The CLI's explicit --raw override remains empty without reading a device."""
    monkeypatch.setenv("TEROK_CONFIG_FILE", os.devnull)
    current = config.SandboxConfig()
    assert current.credentials_use_desktop_keyring is True
    assert current.credentials_passphrase_command is None


@pytest.mark.parametrize("scope", ["override", "system", "user"])
def test_persist_mode_enables_highest_priority_config(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, scope: str
) -> None:
    """Provisioning verifies the effective override, root-system, or layered user policy."""
    config_file = tmp_path / f"{scope}.yml"
    original = f"credentials:\n  use_desktop_keyring: false\n  passphrase_command: {_OLD_COMMAND}\n"
    config_file.write_text(original)
    if scope == "override":
        monkeypatch.setenv("TEROK_CONFIG_FILE", str(config_file))
    else:
        paths = [(scope, config_file)]
        if scope == "user":
            system_file = tmp_path / "system.yml"
            system_file.write_text(original)
            paths.insert(0, ("system", system_file))
        monkeypatch.setattr("terok_sandbox.paths.config_file_paths", lambda: paths)
        monkeypatch.setattr("terok_util.paths.config_file_paths", lambda: paths)

    assert read_config_section("credentials")["use_desktop_keyring"] == "False"
    assert config.credentials_use_desktop_keyring() is False
    _persist_mode_choice(PassphraseTier.DESKTOP_KEYRING)

    assert "use_desktop_keyring: true" in config_file.read_text()
    assert config.credentials_use_desktop_keyring() is True
    assert config.credentials_passphrase_command() == _OLD_COMMAND
    if scope == "user":
        assert system_file.read_text() == original


@pytest.mark.parametrize("failure", ["write", "readback-error", "still-disabled", "invalid-yaml"])
def test_persist_mode_rejects_unverified_policy_without_disclosing_secrets(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
    failure: str,
) -> None:
    """Persistence errors never become success or expose backend/parser diagnostics."""
    config_file = tmp_path / "config.yml"
    original = "credentials:\n  use_desktop_keyring: false\n"
    config_file.write_text(original)
    monkeypatch.setenv("TEROK_CONFIG_FILE", str(config_file))

    def fail_with_secret(*_args: object, **_kwargs: object) -> None:
        """Simulate diagnostics containing input values that must not be propagated."""
        raise PermissionError(_SECRET_SENTINEL)

    match failure:
        case "write":
            monkeypatch.setattr("terok_sandbox._yaml.update_section", fail_with_secret)
        case "readback-error":
            monkeypatch.setattr(config, "credentials_use_desktop_keyring", fail_with_secret)
        case "still-disabled":
            monkeypatch.setattr(config, "credentials_use_desktop_keyring", lambda: False)
        case "invalid-yaml":
            original = f"credentials: [\n  {_SECRET_SENTINEL}\n"
            config_file.write_text(original)

    with pytest.raises(RuntimeError, match="could not enable the desktop keyring") as error:
        _persist_mode_choice(PassphraseTier.DESKTOP_KEYRING)

    assert _SECRET_SENTINEL not in "".join(traceback.format_exception(error.value))
    captured = capsys.readouterr()
    assert not captured.out
    assert _SECRET_SENTINEL not in captured.err
    if failure in {"write", "invalid-yaml"}:
        assert config_file.read_text() == original


def test_session_cache_mode_does_not_read_or_modify_desktop_policy(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Choosing the temporary cache leaves desktop-keyring configuration untouched."""
    config_file = tmp_path / "config.yml"
    original = f"credentials: [\n  {_SECRET_SENTINEL}\n"
    config_file.write_text(original)
    monkeypatch.setenv("TEROK_CONFIG_FILE", str(config_file))

    _persist_mode_choice(PassphraseTier.SESSION_CACHE)

    assert config_file.read_text() == original
