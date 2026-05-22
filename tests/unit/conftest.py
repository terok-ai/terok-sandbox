# SPDX-FileCopyrightText: 2026 Jiri Vyskocil
# SPDX-License-Identifier: Apache-2.0

"""Unit-test fixtures for terok-sandbox.

Ensures port registry never touches the real ``/tmp/terok-ports/`` or
persists claims to the real state directory, and (via
``_isolate_user_paths``) no test ever resolves to a real
``~/.config/terok`` / XDG state path.
"""

from collections.abc import Iterator
from pathlib import Path

import pytest

# Terok-specific env vars that override path resolution.  The autouse
# isolation fixture unsets each so resolution falls back through the
# tmp-rooted ``HOME`` / ``XDG_*`` chain — never to the operator's real
# state.  Kept in one place so a new ``TEROK_*_DIR`` knob added to
# either sandbox or its consumers only needs one edit here.
_TEROK_PATH_OVERRIDE_ENV_VARS = (
    "TEROK_CONFIG_DIR",
    "TEROK_STATE_DIR",
    "TEROK_VAULT_DIR",
    "TEROK_RUNTIME_DIR",
    "TEROK_ROOT",
    "TEROK_SANDBOX_LIVE_DIR",
    "TEROK_SANDBOX_STATE_DIR",
    "TEROK_SANDBOX_RUNTIME_DIR",
    "TEROK_EXECUTOR_STATE_DIR",
    "TEROK_PORT_REGISTRY_DIR",
)


@pytest.fixture(autouse=True)
def _isolate_user_paths(
    tmp_path_factory: pytest.TempPathFactory, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Redirect ``HOME`` and every ``XDG_*`` / ``TEROK_*_DIR`` knob to a fresh tmp dir.

    Without this, tests that exercise default-config code paths (e.g.
    ``SandboxConfig()`` with no overrides, ``handle_*(cfg=None)``) fall
    through to the operator's real ``~/.config/terok/config.yml`` and
    XDG state dirs — silently passing on a clean machine and mutating
    those files on a populated one.  The known reproducer was
    ``TestVaultToKeyring::test_default_cfg_branch``, where a leaked
    passphrase in the real config file flipped the test's expected
    SystemExit into a successful write of ``use_keyring: true`` to the
    operator's config.

    Uses ``tmp_path_factory`` rather than ``tmp_path`` so the fake home
    lives outside the per-test ``tmp_path`` — otherwise tests that
    iterate ``tmp_path`` looking for fixtures see a stray ``fake-home``
    entry.  The per-test ``monkeypatch`` undoes the env overrides at
    teardown, so tests that need different env state can layer their
    own ``setenv`` / ``delenv`` calls on top without leaking across
    cases.
    """
    fake_home = tmp_path_factory.mktemp("fake-home")
    monkeypatch.setenv("HOME", str(fake_home))
    monkeypatch.setenv("XDG_CONFIG_HOME", str(fake_home / ".config"))
    monkeypatch.setenv("XDG_DATA_HOME", str(fake_home / ".local" / "share"))
    monkeypatch.setenv("XDG_STATE_HOME", str(fake_home / ".local" / "state"))
    monkeypatch.setenv("XDG_CACHE_HOME", str(fake_home / ".cache"))
    monkeypatch.setenv("XDG_RUNTIME_DIR", str(fake_home / "run"))
    for var in _TEROK_PATH_OVERRIDE_ENV_VARS:
        monkeypatch.delenv(var, raising=False)


@pytest.fixture(autouse=True)
def _reset_config_caches() -> Iterator[None]:
    """Clear config caches between tests to prevent cross-test pollution."""
    from terok_util.paths import _reset_config_caches_for_tests

    _reset_config_caches_for_tests()
    yield
    _reset_config_caches_for_tests()


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
    """Clear the ``systemd_creds`` ``@cache``s between tests.

    The production caches are process-lifetime correct (systemd doesn't
    re-version mid-process and ``PATH`` resolution is stable for the
    run), but tests stub ``subprocess.run`` / ``shutil.which`` with
    different return values per case; leaking the first test's probe
    result into the next is a guaranteed false-pass.
    """
    from terok_sandbox.vault.store import systemd_creds as _sc

    _sc._systemd_creds_version.cache_clear()
    _sc._systemd_creds_exe.cache_clear()
    yield
    _sc._systemd_creds_version.cache_clear()
    _sc._systemd_creds_exe.cache_clear()


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
    import terok_sandbox.vault.store.encryption as _enc

    monkeypatch.setattr(_enc, "load_passphrase_from_file", lambda _path: None)
    monkeypatch.setattr(_enc, "load_passphrase_from_keyring", lambda: "test")
    monkeypatch.setattr(_enc, "store_passphrase_in_keyring", lambda _pw: True)
    monkeypatch.setattr(_enc, "forget_passphrase_in_keyring", lambda: True)
    monkeypatch.setattr(_config, "credentials_use_keyring", lambda: True)
