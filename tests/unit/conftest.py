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
    """Clear the ``systemd_creds`` ``@cache``s between tests.

    The production caches are process-lifetime correct (systemd doesn't
    re-version mid-process and ``PATH`` resolution is stable for the
    run), but tests stub ``subprocess.run`` / ``shutil.which`` with
    different return values per case; leaking the first test's probe
    result into the next is a guaranteed false-pass.
    """
    from terok_sandbox.credentials import systemd_creds as _sc

    _sc._systemd_creds_version.cache_clear()
    _sc._systemd_creds_exe.cache_clear()
    yield
    _sc._systemd_creds_version.cache_clear()
    _sc._systemd_creds_exe.cache_clear()


@pytest.fixture()
def _systemctl_on_path(monkeypatch: pytest.MonkeyPatch) -> None:
    """Pretend ``systemctl`` is on ``PATH`` so ``_systemctl`` helpers don't short-circuit.

    [`run_best_effort`][terok_sandbox._util._systemctl.run_best_effort]
    and [`run`][terok_sandbox._util._systemctl.run] both probe via
    ``shutil.which`` and return silently when it isn't there — the
    documented containerized-host leniency.  Tests that mock
    ``subprocess.run`` to assert on the systemctl invocation need the
    gate open so the call actually reaches the mock.  Opt-in via
    ``@pytest.mark.usefixtures("_systemctl_on_path")`` rather than
    autouse so other tests can still exercise the absent-binary path.
    """
    import shutil

    real_which = shutil.which

    def _which(cmd: str, *args: object, **kwargs: object) -> str | None:
        if cmd == "systemctl":
            return "/usr/bin/systemctl"
        return real_which(cmd, *args, **kwargs)  # type: ignore[arg-type]

    monkeypatch.setattr("shutil.which", _which)


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
