# SPDX-FileCopyrightText: 2026 Jiri Vyskocil
# SPDX-License-Identifier: Apache-2.0

"""The volatile session cache and the non-blocking desktop-keyring read.

Locks the two guarantees the TUI-freeze fix introduced: an desktop-keyring
read can never block its caller (locked collections are skipped, wedged
backends time out), and the cache tier degrades to a tmpfs session file
on hosts where the kernel key facility is unusable.
"""

from __future__ import annotations

import sys
import threading
import time
import types
from pathlib import Path

import keyring
import pytest

from terok_sandbox._util._placement import SupervisorPlacement
from terok_sandbox.vault.store import (
    encryption,
    kernel_keyring,
    session_cache,
    session_file,
)

_DB = "credentials.db"

#: The real read, captured at import — the autouse conftest fixture stubs
#: the module attribute for every test, and this class tests the real thing.
_REAL_LOAD = encryption.load_passphrase_from_desktop_keyring


# ── Non-blocking desktop-keyring read ────────────────────────────────────


class TestDesktopKeyringReadNeverBlocks:
    """`load_passphrase_from_desktop_keyring` must return, whatever the backend does."""

    def test_blocked_interactive_read_is_skipped_without_touching_the_backend(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """An interactive read the probe marks blocked never reaches the backend."""
        monkeypatch.setenv("DISPLAY", ":0")
        monkeypatch.setattr(
            encryption, "desktop_keyring_read_blocked", lambda **_kw: "desktop keyring locked"
        )
        forbidden = types.ModuleType("keyring")

        def _explode(*_args: object) -> str:
            raise AssertionError("a blocked read must not reach the backend")

        forbidden.get_password = _explode  # type: ignore[attr-defined]
        monkeypatch.setitem(sys.modules, "keyring", forbidden)

        assert _REAL_LOAD(allow_prompt=True) is None

    def test_wedged_backend_times_out_instead_of_freezing(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """A backend that never answers costs the timeout, not the process."""
        monkeypatch.setattr(encryption, "desktop_keyring_read_blocked", lambda **_kw: None)
        monkeypatch.setattr(encryption, "_DESKTOP_KEYRING_READ_TIMEOUT_S", 0.2)
        release = threading.Event()
        stuck = types.ModuleType("keyring")

        def _hang(*_args: object) -> str | None:
            release.wait()  # blocks until released, like a locked-collection prompt
            return None

        stuck.get_password = _hang  # type: ignore[attr-defined]
        monkeypatch.setattr(keyring, "get_keyring", lambda: stuck)

        try:
            started = time.monotonic()
            assert _REAL_LOAD() is None
            assert time.monotonic() - started < 2.0
        finally:
            release.set()  # frees the shared worker for the next test

    def test_repeated_timeouts_do_not_stack_workers(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """A wedged backend occupies the single worker; later reads never reach it."""
        monkeypatch.setattr(encryption, "desktop_keyring_read_blocked", lambda **_kw: None)
        monkeypatch.setattr(encryption, "_DESKTOP_KEYRING_READ_TIMEOUT_S", 0.2)
        release = threading.Event()
        backend_calls: list[int] = []
        stuck = types.ModuleType("keyring")

        def _hang(*_args: object) -> str | None:
            backend_calls.append(1)
            release.wait()
            return None

        stuck.get_password = _hang  # type: ignore[attr-defined]
        monkeypatch.setattr(keyring, "get_keyring", lambda: stuck)

        try:
            assert _REAL_LOAD() is None  # times out; the call occupies the worker
            assert _REAL_LOAD() is None  # queues, times out, and cancels out of the queue
            assert len(backend_calls) == 1  # the wedged call is the only backend entry
        finally:
            release.set()  # frees the shared worker for the next test

    def test_healthy_backend_answers_normally(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """The guards are transparent to a backend that just answers."""
        monkeypatch.setattr(encryption, "desktop_keyring_read_blocked", lambda **_kw: None)
        healthy = types.ModuleType("keyring")
        healthy.get_password = lambda *_args: "the-passphrase"  # type: ignore[attr-defined]
        monkeypatch.setattr(keyring, "get_keyring", lambda: healthy)

        assert _REAL_LOAD() == "the-passphrase"


# ── Session-file backing ────────────────────────────────────────────


class TestSessionFile:
    """The tmpfs cache file mirrors the kernel-keyring verb contract."""

    @pytest.fixture(autouse=True)
    def _runtime_dir(self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
        runtime = tmp_path / "xdg-runtime"
        runtime.mkdir()
        monkeypatch.setenv("XDG_RUNTIME_DIR", str(runtime))
        return runtime

    def test_store_load_forget_roundtrip(self) -> None:
        assert not session_file.is_cached(_DB)
        assert session_file.store("pw", _DB)
        assert session_file.is_cached(_DB)
        assert session_file.load(_DB) == "pw"
        assert session_file.forget(_DB)
        assert session_file.load(_DB) is None

    def test_cache_file_is_user_only(self, _runtime_dir: Path) -> None:
        """The file carries 0600 in a 0700 directory — kernel-keyring exposure, on tmpfs."""
        session_file.store("pw", _DB)
        [cache] = list((_runtime_dir / "terok" / "vault-session").iterdir())
        assert cache.stat().st_mode & 0o777 == 0o600
        assert cache.parent.stat().st_mode & 0o777 == 0o700

    def test_two_vaults_never_share_a_cache(self) -> None:
        session_file.store("first", "/one/credentials.db")
        assert session_file.load("/two/credentials.db") is None

    def test_no_runtime_dir_is_a_clean_miss(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.delenv("XDG_RUNTIME_DIR")
        assert session_file.unavailable_reason() is not None
        assert not session_file.store("pw", _DB)
        assert session_file.load(_DB) is None
        assert session_file.forget(_DB)  # nothing can exist → cleared by definition


# ── Backing selection ───────────────────────────────────────────────


class TestSessionCacheFacade:
    """One tier, two backings — the kernel keyring first, the file as fallback."""

    @pytest.fixture(autouse=True)
    def _runtime_dir(self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
        runtime = tmp_path / "xdg-runtime"
        runtime.mkdir()
        monkeypatch.setenv("XDG_RUNTIME_DIR", str(runtime))

    def test_falls_back_to_the_session_file_when_the_kernel_facility_is_gone(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.setattr(kernel_keyring, "unavailable_reason", lambda: "no libkeyutils")
        assert session_cache.store("pw", _DB)
        assert session_cache.load(_DB) == "pw"
        assert session_file.is_cached(_DB)  # it landed in the file, not the kernel keyring

    def test_forget_clears_both_backings(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """`vault lock` must not leave a cache in the backing this boot doesn't prefer."""
        forgotten: list[str] = []
        monkeypatch.setattr(
            kernel_keyring, "forget", lambda _db: forgotten.append("kernel") or True
        )
        session_file.store("pw", _DB)
        assert session_cache.forget(_DB)
        assert forgotten == ["kernel"]
        assert not session_file.is_cached(_DB)

    def test_a_user_unit_supervisor_reads_the_kernel_keyring(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Where the supervisor is a user unit, the operator's kernel keyring is its cache."""
        stored: list[str] = []
        monkeypatch.setattr(
            session_cache, "supervisor_placement", lambda: SupervisorPlacement.USER_UNIT
        )
        monkeypatch.setattr(kernel_keyring, "unavailable_reason", lambda: None)
        monkeypatch.setattr(
            kernel_keyring, "store", lambda pw, _db: bool(stored.append(pw)) or True
        )
        assert session_cache.store("pw", _DB)
        assert stored == ["pw"]
        assert not session_file.is_cached(_DB)

    def test_a_namespace_daemon_gets_the_session_file(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Inside the container namespace the kernel keyring is a stranger; a path is a path."""
        monkeypatch.setattr(
            session_cache, "supervisor_placement", lambda: SupervisorPlacement.NAMESPACE_DAEMON
        )
        monkeypatch.setattr(kernel_keyring, "unavailable_reason", lambda: None)
        assert session_cache.store("pw", _DB)
        assert session_file.is_cached(_DB)
        assert "the supervisor runs in the container namespace" in session_cache.backing_detail(
            cached=True
        )

    def test_unavailable_is_the_chosen_backings_reason(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.setattr(
            session_cache, "supervisor_placement", lambda: SupervisorPlacement.NAMESPACE_DAEMON
        )
        assert session_cache.unavailable_reason() is None
        monkeypatch.setattr(session_file, "unavailable_reason", lambda: "no runtime dir")
        assert session_cache.unavailable_reason() == (
            "the supervisor runs in the container namespace, without a user manager; no runtime dir"
        )
        assert session_cache.backing_detail(cached=False).startswith("unusable here: ")

    def test_detail_names_the_degraded_backing(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """A unit host whose kernel facility is gone still says why the file serves."""
        monkeypatch.setattr(
            session_cache, "supervisor_placement", lambda: SupervisorPlacement.USER_UNIT
        )
        monkeypatch.setattr(kernel_keyring, "unavailable_reason", lambda: "no libkeyutils")
        detail = session_cache.backing_detail(cached=True)
        assert "session file" in detail
        assert "no libkeyutils" in detail


class TestDesktopKeyringWorkerRetirement:
    """The desktop-keyring read leaves a worker thread; a child retires it before Landlock."""

    @pytest.fixture(autouse=True)
    def _fresh_worker(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """Start from no worker and no wedge, whatever an earlier test left behind."""
        monkeypatch.setattr(encryption, "_desktop_keyring_executor", None)
        monkeypatch.setattr(encryption, "_desktop_keyring_worker_wedged", False)

    def test_a_finished_read_leaves_no_thread_behind(self) -> None:
        before = set(threading.enumerate())
        assert encryption._call_with_timeout(lambda: "pw", 1.0) == "pw"
        spawned = set(threading.enumerate()) - before
        assert spawned
        encryption.retire_desktop_keyring_worker()
        assert not any(thread.is_alive() for thread in spawned)
        # A later read starts a fresh worker; retirement is not a one-way door.
        assert encryption._call_with_timeout(lambda: "again", 1.0) == "again"
        encryption.retire_desktop_keyring_worker()

    def test_a_wedged_worker_is_abandoned_not_joined(self) -> None:
        """Joining a stuck read would hang the child; the thread stays and Landlock says so."""
        release = threading.Event()
        before = set(threading.enumerate())
        with pytest.raises(TimeoutError):
            encryption._call_with_timeout(lambda: release.wait(5) and "late", 0.05)
        spawned = set(threading.enumerate()) - before
        encryption.retire_desktop_keyring_worker()  # returns at once
        assert any(thread.is_alive() for thread in spawned)
        release.set()
        # The wedge belonged to that worker: the next one is joined again.
        assert encryption._call_with_timeout(lambda: "fresh", 1.0) == "fresh"
        assert encryption._desktop_keyring_worker_wedged is False
        encryption.retire_desktop_keyring_worker()


class TestLockedCollectionPromptPolicy:
    """A locked collection prompts only an interactive caller on a desktop.

    The unlock dialog is legitimate when the operator sees it and can
    answer or cancel it.  A background read, and any read on a host
    without a display server, must skip the locked collection instead.
    """

    @pytest.fixture(autouse=True)
    def _locked_secret_service(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """Present a locked Secret Service collection without real D-Bus."""
        import keyring
        from keyring.backends.SecretService import Keyring as SecretService

        backend = SecretService.__new__(SecretService)
        monkeypatch.setattr(keyring, "get_keyring", lambda: backend)
        fake = types.ModuleType("secretstorage")
        connection = types.SimpleNamespace(close=lambda: None)
        collection = types.SimpleNamespace(is_locked=lambda: True)
        fake.dbus_init = lambda: connection  # type: ignore[attr-defined]
        fake.Collection = lambda _c: collection  # type: ignore[attr-defined]
        monkeypatch.setitem(sys.modules, "secretstorage", fake)

    def test_background_read_skips_the_locked_collection(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.setenv("DISPLAY", ":0")  # a desktop alone does not permit a prompt
        assert encryption.desktop_keyring_read_blocked() is not None

    def test_interactive_read_on_a_desktop_may_prompt(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        import keyring

        monkeypatch.setenv("DISPLAY", ":0")
        assert encryption.desktop_keyring_read_blocked(allow_prompt=True) is None
        # The read reaches the backend — the prompt is the backend's business.
        monkeypatch.setattr(keyring.get_keyring(), "get_password", lambda *_a: "unlocked-by-dialog")
        assert _REAL_LOAD(allow_prompt=True) == "unlocked-by-dialog"

    def test_interactive_read_without_a_display_skips(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.delenv("DISPLAY", raising=False)
        monkeypatch.delenv("WAYLAND_DISPLAY", raising=False)
        assert encryption.desktop_keyring_read_blocked(allow_prompt=True) is not None
        assert _REAL_LOAD(allow_prompt=True) is None
