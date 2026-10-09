# SPDX-FileCopyrightText: 2026 Jiri Vyskocil
# SPDX-License-Identifier: Apache-2.0

"""Desktop-keyring transfers preserve access across failures and subprocesses."""

from __future__ import annotations

import subprocess
import sys
from collections.abc import Iterator
from dataclasses import dataclass
from pathlib import Path

import keyring
import pytest
from terok_util import read_config_section

from terok_sandbox import config
from terok_sandbox.commands import handle_vault_to_desktop_keyring
from terok_sandbox.commands.credentials import TierProvisionResult
from terok_sandbox.commands.vault import PassphraseChangeResult
from terok_sandbox.vault.store import encryption, kernel_keyring, session_cache, session_file
from terok_sandbox.vault.store.db import CredentialDB
from terok_sandbox.vault.store.encryption import (
    DESKTOP_KEYRING_SERVICE,
    DESKTOP_KEYRING_USERNAME,
    load_passphrase_from_desktop_keyring,
    store_passphrase_in_desktop_keyring,
)
from terok_sandbox.vault.store.recovery import acknowledge
from terok_sandbox.vault.store.status import VaultState, VaultStatus
from terok_sandbox.vault.store.tiers import PassphraseTier

_PASSPHRASE = "dummy-transfer-pässphrase"
_REAL_CREDENTIALS_USE_DESKTOP_KEYRING = config.credentials_use_desktop_keyring
_DISABLED_CONFIG = "credentials:\n  use_desktop_keyring: false\n"


class _FileDesktopKeyring:
    """A fake desktop keyring shared between processes through a temporary file."""

    def __init__(self, path: Path) -> None:
        self.path = path
        self.readback = "stored"
        self.writes = 0

    def get_password(self, service: str, username: str) -> str | None:
        """Read the dummy secret or simulate a backend's faulty readback."""
        assert (service, username) == (DESKTOP_KEYRING_SERVICE, DESKTOP_KEYRING_USERNAME)
        match self.readback:
            case "absent":
                return None
            case "wrong":
                return "different-dummy-passphrase"
            case "error":
                raise RuntimeError(_PASSPHRASE)
            case _:
                return _read_file(self.path)

    def set_password(self, service: str, username: str, passphrase: str) -> None:
        """Accept a write independently of whether later reads work."""
        assert (service, username) == (DESKTOP_KEYRING_SERVICE, DESKTOP_KEYRING_USERNAME)
        self.path.write_text(passphrase)
        self.writes += 1


@dataclass
class _TransferHarness:
    """Temporary policy and stores surrounding a real encrypted vault."""

    root: Path
    config_file: Path
    kernel_file: Path
    desktop: _FileDesktopKeyring


@pytest.fixture
def transfer(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Iterator[_TransferHarness]:
    """Provision an acknowledged encrypted vault, initially unlocked from the kernel cache."""
    harness = _install_fake_stores(tmp_path, monkeypatch)
    harness.config_file.write_text(_DISABLED_CONFIG)
    harness.kernel_file.write_text(_PASSPHRASE)
    cfg = config.SandboxConfig()
    db = CredentialDB(cfg.db_path, passphrase=_PASSPHRASE)
    db.store_credential("default", "test-provider", {"type": "api_key", "value": "dummy-token"})
    db.close()
    acknowledge(cfg.vault_recovery_marker_file)
    yield harness
    encryption.retire_desktop_keyring_worker()


@pytest.mark.parametrize(
    "result",
    [
        TierProvisionResult(_PASSPHRASE, PassphraseTier.DESKTOP_KEYRING, generated=True),
        PassphraseChangeResult(_PASSPHRASE, generated=True, rekeyed=True, rewrites=()),
    ],
    ids=["provision-result", "change-result"],
)
def test_passphrase_results_do_not_expose_secrets_in_representations(
    result: TierProvisionResult | PassphraseChangeResult,
) -> None:
    """Ordinary logging and nested container reprs retain metadata, not the passphrase."""
    assert result.passphrase == _PASSPHRASE
    assert "generated=True" in repr(result)
    assert _PASSPHRASE not in repr(result)
    assert _PASSPHRASE not in str(result)
    assert _PASSPHRASE not in repr({"result": result})


def test_child_transfer_refreshes_running_parent_status(transfer: _TransferHarness) -> None:
    """A child transfer leaves the already-running parent unlocked through the desktop keyring."""
    assert read_config_section("credentials")["use_desktop_keyring"] == "False"
    original = config.SandboxConfig()
    assert original.credentials_use_desktop_keyring is False
    before = VaultStatus.load(original)
    assert before.state is VaultState.UNLOCKED
    assert before.source is PassphraseTier.SESSION_CACHE

    completed = subprocess.run(
        [
            sys.executable,
            "-c",
            "from pathlib import Path; import sys; "
            "from tests.unit.test_vault_desktop_keyring_transfer import _run_child_transfer; "
            "_run_child_transfer(Path(sys.argv[1]))",
            str(transfer.root),
        ],
        check=True,
        capture_output=True,
        text=True,
        timeout=20,
    )

    assert "stored and verified passphrase in desktop keyring" in completed.stdout
    assert _PASSPHRASE not in completed.stdout + completed.stderr
    assert not transfer.kernel_file.exists()
    assert transfer.desktop.path.read_text() == _PASSPHRASE
    assert "use_desktop_keyring: true" in transfer.config_file.read_text()
    after = VaultStatus.load()
    assert after.state is VaultState.UNLOCKED
    assert after.source is PassphraseTier.DESKTOP_KEYRING
    assert after.providers == ("test-provider",)
    assert not next(row for row in after.chain if row.tier is PassphraseTier.SESSION_CACHE).present


@pytest.mark.parametrize("readback", ["absent", "wrong", "error"])
def test_failed_desktop_readback_preserves_source_and_policy(
    transfer: _TransferHarness,
    capsys: pytest.CaptureFixture[str],
    caplog: pytest.LogCaptureFixture,
    readback: str,
) -> None:
    """A backend accepting the write is not sufficient to remove the working source."""
    transfer.desktop.readback = readback

    with pytest.raises(SystemExit, match="could not store and verify") as error:
        handle_vault_to_desktop_keyring()

    assert transfer.desktop.writes == 1
    assert transfer.kernel_file.read_text() == _PASSPHRASE
    assert transfer.config_file.read_text() == _DISABLED_CONFIG
    assert VaultStatus.load().source is PassphraseTier.SESSION_CACHE
    captured = capsys.readouterr()
    assert "stored and verified" not in captured.out
    assert _PASSPHRASE not in captured.out + captured.err + caplog.text + str(error.value)


def test_wrong_prompted_passphrase_never_reaches_destination(
    transfer: _TransferHarness,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    """A typing error cannot replace the saved passphrase or change source policy."""
    transfer.kernel_file.unlink()
    monkeypatch.setattr("sys.stdin.isatty", lambda: True)
    monkeypatch.setattr(encryption, "prompt_passphrase", lambda: "mistyped-dummy-passphrase")

    with pytest.raises(SystemExit, match="does not open the encrypted vault"):
        handle_vault_to_desktop_keyring()

    assert transfer.desktop.writes == 0
    assert not transfer.desktop.path.exists()
    assert transfer.config_file.read_text() == _DISABLED_CONFIG
    assert "stored and verified" not in capsys.readouterr().out
    CredentialDB(config.SandboxConfig().db_path, passphrase=_PASSPHRASE).close()


@pytest.mark.parametrize("failure", ["write", "verification"])
def test_config_failure_keeps_working_source(
    transfer: _TransferHarness,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
    failure: str,
) -> None:
    """The source remains usable until the persisted desktop-keyring policy is confirmed."""
    cfg = config.SandboxConfig()
    if failure == "write":

        def deny_write(*_args: object, **_kwargs: object) -> None:
            """Model a secret-bearing parser or filesystem failure."""
            raise PermissionError(_PASSPHRASE)

        monkeypatch.setattr("terok_sandbox._yaml.update_section", deny_write)
    else:
        monkeypatch.setattr(config, "credentials_use_desktop_keyring", lambda: False)

    with pytest.raises(SystemExit, match="could not enable it in configuration") as error:
        handle_vault_to_desktop_keyring(cfg=cfg)

    assert transfer.desktop.path.read_text() == _PASSPHRASE
    assert transfer.kernel_file.read_text() == _PASSPHRASE
    if failure == "write":
        assert transfer.config_file.read_text() == _DISABLED_CONFIG
    captured = capsys.readouterr()
    assert "stored and verified" not in captured.out
    assert _PASSPHRASE not in captured.out + captured.err + str(error.value)
    assert VaultStatus.load(cfg).state is VaultState.UNLOCKED


def test_sealed_cleanup_failure_preserves_kernel_cache(
    transfer: _TransferHarness,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    """An undeletable higher-priority sealed source aborts before clearing the kernel copy."""
    cfg = config.SandboxConfig()
    sealed = cfg.vault_systemd_creds_file
    sealed.write_bytes(b"dummy-sealed-credential")
    monkeypatch.setattr("terok_sandbox.vault.store.systemd_creds.unseal", lambda _path: _PASSPHRASE)
    real_unlink = Path.unlink

    def reject_sealed_unlink(path: Path, *args: object, **kwargs: object) -> None:
        """Keep ordinary temporary-file cleanup working while denying the sealed source."""
        if path == sealed:
            raise PermissionError(_PASSPHRASE)
        real_unlink(path, *args, **kwargs)

    monkeypatch.setattr(Path, "unlink", reject_sealed_unlink)
    with pytest.raises(SystemExit, match="could not remove the sealed source") as error:
        handle_vault_to_desktop_keyring(cfg=cfg)

    assert sealed.exists()
    assert transfer.kernel_file.read_text() == _PASSPHRASE
    assert transfer.desktop.path.read_text() == _PASSPHRASE
    captured = capsys.readouterr()
    assert "stored and verified" not in captured.out
    assert _PASSPHRASE not in captured.out + captured.err + str(error.value)


def _install_fake_stores(root: Path, monkeypatch: pytest.MonkeyPatch) -> _TransferHarness:
    """Install identical, filesystem-isolated stores in the parent or transfer subprocess."""
    config_file = root / "config.yml"
    kernel_file = root / "fake-kernel-secret"
    desktop = _FileDesktopKeyring(root / "fake-desktop-secret")
    for name, relative in (
        ("TEROK_CONFIG_FILE", "config.yml"),
        ("TEROK_VAULT_DIR", "vault"),
        ("TEROK_SANDBOX_STATE_DIR", "state"),
        ("TEROK_SANDBOX_RUNTIME_DIR", "runtime"),
        ("TEROK_SANDBOX_CONFIG_DIR", "sandbox-config"),
    ):
        monkeypatch.setenv(name, str(root / relative))
    monkeypatch.setattr(
        config, "credentials_use_desktop_keyring", _REAL_CREDENTIALS_USE_DESKTOP_KEYRING
    )
    monkeypatch.setattr(
        encryption, "load_passphrase_from_desktop_keyring", load_passphrase_from_desktop_keyring
    )
    monkeypatch.setattr(
        encryption, "store_passphrase_in_desktop_keyring", store_passphrase_in_desktop_keyring
    )
    monkeypatch.setattr(keyring, "get_keyring", lambda: desktop)
    monkeypatch.setattr(session_cache, "_backend", lambda: kernel_keyring)
    monkeypatch.setattr(kernel_keyring, "load", lambda _db: _read_file(kernel_file))
    monkeypatch.setattr(kernel_keyring, "is_cached", lambda _db: kernel_file.is_file())
    monkeypatch.setattr(kernel_keyring, "unavailable_reason", lambda: None)
    monkeypatch.setattr(session_file, "forget", lambda _db: True)
    monkeypatch.setattr(
        "terok_sandbox.vault.store.systemd_creds.unavailable_reason",
        lambda: "disabled in transfer regression tests",
    )

    def forget_kernel(_db: object) -> bool:
        """Delete only the temporary file that models the kernel-keyring entry."""
        kernel_file.unlink(missing_ok=True)
        return True

    monkeypatch.setattr(kernel_keyring, "forget", forget_kernel)
    return _TransferHarness(root, config_file, kernel_file, desktop)


def _read_file(path: Path) -> str | None:
    """Read one fake store; missing means it has no passphrase."""
    try:
        return path.read_text()
    except FileNotFoundError:
        return None


def _run_child_transfer(root: Path) -> None:
    """Run the real transfer with only a temporary path supplied through argv."""
    with pytest.MonkeyPatch.context() as monkeypatch:
        _install_fake_stores(root, monkeypatch)
        try:
            handle_vault_to_desktop_keyring()
        finally:
            encryption.retire_desktop_keyring_worker()
