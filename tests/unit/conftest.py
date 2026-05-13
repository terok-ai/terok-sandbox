# SPDX-FileCopyrightText: 2026 Jiri Vyskocil
# SPDX-License-Identifier: Apache-2.0

"""Unit-test fixtures for terok-sandbox.

Ensures port registry never touches the real ``/tmp/terok-ports/`` or
persists claims to the real state directory.
"""

from collections.abc import Iterator
from pathlib import Path

import pytest


@pytest.fixture(autouse=True)
def _reset_config_caches() -> Iterator[None]:
    """Clear config caches between tests to prevent cross-test pollution."""
    import terok_sandbox.paths as _paths

    _paths._config_section_cache.clear()
    yield
    _paths._config_section_cache.clear()


@pytest.fixture(autouse=True)
def _isolate_port_registry(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """Redirect the default port registry to tmp dirs and clear state."""
    import terok_sandbox.port_registry as _reg

    registry = tmp_path / "terok-ports"
    registry.mkdir()
    monkeypatch.setattr(_reg._default, "registry_dir", registry)
    monkeypatch.setattr(_reg, "_save_ports", lambda _sd, _p: None)
    _reg.reset_cache()


@pytest.fixture(autouse=True)
def _isolate_systemd_creds_version_cache() -> Iterator[None]:
    """Clear the ``_systemd_creds_version`` ``@cache`` between tests.

    The production cache is process-lifetime correct (systemd doesn't
    re-version mid-process), but tests stub ``subprocess.run`` /
    ``shutil.which`` with different return values per case; leaking the
    first test's probe result into the next is a guaranteed false-pass.
    """
    from terok_sandbox.credentials import systemd_creds as _sc

    _sc._systemd_creds_version.cache_clear()
    yield
    _sc._systemd_creds_version.cache_clear()


@pytest.fixture(autouse=True)
def _isolate_credential_keyring(monkeypatch: pytest.MonkeyPatch) -> None:
    """Stub the resolution chain so tests get a deterministic ``"test"`` passphrase.

    Tests pick up the platform-default ``runtime_dir`` which can hold
    a stale ``vault.passphrase`` from a prior run, so we blank the
    file tier as well as the keyring tier; the 4 tests that exercise
    the file tier explicitly restore ``load_passphrase_from_file``
    via a local monkeypatch.
    """
    import terok_sandbox.config as _config
    import terok_sandbox.credentials.encryption as _enc

    monkeypatch.setattr(_enc, "load_passphrase_from_file", lambda _path: None)
    monkeypatch.setattr(_enc, "load_passphrase_from_keyring", lambda: "test")
    monkeypatch.setattr(_enc, "store_passphrase_in_keyring", lambda _pw: True)
    monkeypatch.setattr(_enc, "forget_passphrase_in_keyring", lambda: True)
    monkeypatch.setattr(_config, "credentials_use_keyring", lambda: True)
