# SPDX-FileCopyrightText: 2026 Jiri Vyskocil
# SPDX-License-Identifier: Apache-2.0

"""Desktop-keyring transfers require verified retrieval without background prompts."""

from __future__ import annotations

import threading
from unittest.mock import Mock

import keyring
import pytest
import secretstorage
from keyring.backends import null
from keyring.backends.chainer import ChainerBackend
from keyring.backends.SecretService import Keyring as SecretService
from keyring.errors import PasswordDeleteError
from secretstorage.exceptions import ItemNotFoundException, LockedException

from terok_sandbox.vault.store import encryption
from terok_sandbox.vault.store.encryption import (
    DESKTOP_KEYRING_SERVICE,
    DESKTOP_KEYRING_USERNAME,
    forget_passphrase_in_desktop_keyring,
    load_passphrase_from_desktop_keyring,
    store_passphrase_in_desktop_keyring,
)

_PASSPHRASE = "dummy-pässphrase"
_PREFERRED_COLLECTION = "/org/freedesktop/secrets/collection/terok_testing"
_DELETE_PROMPT_PATH = "/org/freedesktop/secrets/prompt/terok_testing"


@pytest.fixture
def backend(monkeypatch: pytest.MonkeyPatch) -> Mock:
    """Route every desktop-keyring operation to a fake backend, never the operator's stores."""
    selected = Mock()
    selected.get_password.return_value = _PASSPHRASE
    monkeypatch.setattr(keyring, "get_keyring", lambda: selected)
    return selected


@pytest.fixture
def secret_service(monkeypatch: pytest.MonkeyPatch) -> tuple[SecretService, Mock, Mock]:
    """Expose a fake Secret Service collection behind the real backend type."""
    backend = SecretService.__new__(SecretService)
    connection = Mock()
    collection = Mock()
    collection.is_locked.return_value = False
    item = Mock()
    item.get_secret.return_value = _PASSPHRASE.encode()
    collection.search_items.return_value = [item]
    monkeypatch.setattr(keyring, "get_keyring", lambda: backend)
    monkeypatch.setattr(secretstorage, "dbus_init", Mock(return_value=connection))
    monkeypatch.setattr(secretstorage, "Collection", Mock(return_value=collection))
    monkeypatch.setattr(
        secretstorage, "get_default_collection", Mock(side_effect=AssertionError("may prompt"))
    )
    monkeypatch.setattr(backend, "get_password", Mock(side_effect=AssertionError("may unlock")))
    return backend, collection, connection


class TestVerifiedDesktopStore:
    """Writing alone is not a transfer success; the value must be retrievable."""

    def test_exact_value_read_back(self, backend: Mock) -> None:
        assert store_passphrase_in_desktop_keyring(_PASSPHRASE)
        backend.set_password.assert_called_once_with(
            DESKTOP_KEYRING_SERVICE, DESKTOP_KEYRING_USERNAME, _PASSPHRASE
        )
        backend.get_password.assert_called_once_with(
            DESKTOP_KEYRING_SERVICE, DESKTOP_KEYRING_USERNAME
        )

    @pytest.mark.parametrize("retrieved", [None, "", "wrong-passphrase"])
    def test_missing_or_wrong_readback_is_not_success(self, backend: Mock, retrieved: str | None):
        backend.get_password.return_value = retrieved
        assert not store_passphrase_in_desktop_keyring(_PASSPHRASE)
        backend.delete_password.assert_not_called()

    @pytest.mark.parametrize("operation", ["set_password", "get_password"])
    def test_backend_error_does_not_expose_secret(
        self, backend: Mock, operation: str, caplog: pytest.LogCaptureFixture
    ) -> None:
        getattr(backend, operation).side_effect = RuntimeError(_PASSPHRASE)
        assert not store_passphrase_in_desktop_keyring(_PASSPHRASE)
        assert _PASSPHRASE not in caplog.text

    def test_noop_null_backend_is_not_success(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setattr(keyring, "get_keyring", lambda: null.Keyring())
        assert not store_passphrase_in_desktop_keyring(_PASSPHRASE)

    def test_empty_value_never_reaches_backend(self, backend: Mock) -> None:
        with pytest.raises(ValueError, match="empty passphrase"):
            store_passphrase_in_desktop_keyring("")
        backend.set_password.assert_not_called()

    def test_readback_timeout_preserves_failure(
        self, backend: Mock, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        release = threading.Event()
        finished = threading.Event()

        def delayed_read(*_args: object) -> str:
            try:
                release.wait()
                return _PASSPHRASE
            finally:
                finished.set()

        backend.get_password.side_effect = delayed_read
        monkeypatch.setattr(encryption, "_DESKTOP_KEYRING_READ_TIMEOUT_S", 0.01)
        try:
            assert not store_passphrase_in_desktop_keyring(_PASSPHRASE)
            backend.set_password.assert_called_once()
        finally:
            release.set()
            assert finished.wait(2)

    def test_chainer_does_not_verify_against_another_backend(
        self, backend: Mock, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        backend.get_password.return_value = None
        other = Mock()
        other.get_password.return_value = _PASSPHRASE
        chain = _select_chain(monkeypatch, backend, other)
        assert not store_passphrase_in_desktop_keyring(_PASSPHRASE)
        other.set_password.assert_not_called()
        assert keyring.get_keyring() is chain

    def test_readonly_backend_cannot_shadow_a_verified_write(
        self, backend: Mock, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        backend.set_password.side_effect = NotImplementedError
        backend.get_password.return_value = "old-passphrase"
        writable = Mock()
        writable.get_password.return_value = _PASSPHRASE
        _select_chain(monkeypatch, backend, writable)
        assert not store_passphrase_in_desktop_keyring(_PASSPHRASE)
        writable.set_password.assert_called_once()

    def test_chainer_write_verifies_destination_and_resolution(
        self, backend: Mock, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        _select_chain(monkeypatch, backend)
        assert store_passphrase_in_desktop_keyring(_PASSPHRASE)
        assert backend.get_password.call_count == 2

    def test_secret_service_readback_does_not_unlock(
        self, secret_service: tuple[SecretService, Mock, Mock], monkeypatch: pytest.MonkeyPatch
    ) -> None:
        backend, collection, _connection = secret_service
        monkeypatch.setattr(backend, "set_password", Mock())
        assert store_passphrase_in_desktop_keyring(_PASSPHRASE)
        collection.unlock.assert_not_called()
        backend.get_password.assert_not_called()


class TestPromptlessSecretService:
    """A collection probe is not permission for a later read to unlock anything."""

    def test_preferred_collection_and_attribute_scheme_are_respected(
        self, secret_service: tuple[SecretService, Mock, Mock]
    ) -> None:
        backend, collection, connection = secret_service
        backend.preferred_collection = _PREFERRED_COLLECTION
        backend.scheme = "KeePassXC"
        assert load_passphrase_from_desktop_keyring() == _PASSPHRASE
        secretstorage.Collection.assert_called_with(connection, _PREFERRED_COLLECTION)
        collection.search_items.assert_called_once_with(
            {"UserName": DESKTOP_KEYRING_USERNAME, "Title": DESKTOP_KEYRING_SERVICE}
        )
        connection.close.assert_called_once()

    def test_absent_collection_never_creates_one(
        self, secret_service: tuple[SecretService, Mock, Mock]
    ) -> None:
        _backend, _collection, connection = secret_service
        secretstorage.Collection.side_effect = ItemNotFoundException("absent")
        assert load_passphrase_from_desktop_keyring() is None
        secretstorage.get_default_collection.assert_not_called()
        connection.close.assert_called_once()

    def test_absent_collection_is_verified_absence_on_delete(
        self, secret_service: tuple[SecretService, Mock, Mock]
    ) -> None:
        secretstorage.Collection.side_effect = ItemNotFoundException("absent")
        assert forget_passphrase_in_desktop_keyring() is None
        secretstorage.get_default_collection.assert_not_called()

    def test_locked_collection_skips_read(
        self, secret_service: tuple[SecretService, Mock, Mock]
    ) -> None:
        backend, collection, _connection = secret_service
        collection.is_locked.return_value = True
        collection.ensure_not_locked.side_effect = LockedException("locked")
        assert load_passphrase_from_desktop_keyring() is None
        backend.get_password.assert_not_called()
        collection.search_items.assert_not_called()

    @pytest.mark.parametrize("locked_after_probe", ["collection", "item"])
    def test_lock_race_cannot_trigger_unlock(
        self, secret_service: tuple[SecretService, Mock, Mock], locked_after_probe: str
    ) -> None:
        backend, collection, _connection = secret_service
        item = collection.search_items.return_value[0]
        operation = (
            collection.ensure_not_locked if locked_after_probe == "collection" else item.get_secret
        )
        operation.side_effect = LockedException("locked after probe")
        assert load_passphrase_from_desktop_keyring() is None
        backend.get_password.assert_not_called()
        collection.unlock.assert_not_called()
        item.unlock.assert_not_called()

    def test_chained_locked_collection_is_not_bypassed(
        self, secret_service: tuple[SecretService, Mock, Mock], monkeypatch: pytest.MonkeyPatch
    ) -> None:
        backend, collection, _connection = secret_service
        collection.is_locked.return_value = True
        collection.ensure_not_locked.side_effect = LockedException("locked")
        _select_chain(monkeypatch, backend)
        assert encryption.desktop_keyring_read_blocked() is not None
        assert load_passphrase_from_desktop_keyring() is None
        backend.get_password.assert_not_called()

    def test_chained_unlocked_read_never_calls_generic_getter(
        self, secret_service: tuple[SecretService, Mock, Mock], monkeypatch: pytest.MonkeyPatch
    ) -> None:
        backend, _collection, _connection = secret_service
        _select_chain(monkeypatch, backend)
        assert load_passphrase_from_desktop_keyring() == _PASSPHRASE
        backend.get_password.assert_not_called()

    def test_unused_native_backend_does_not_mask_secret_service(
        self, secret_service: tuple[SecretService, Mock, Mock], monkeypatch: pytest.MonkeyPatch
    ) -> None:
        backend, _collection, _connection = secret_service
        native = type("Keyring", (), {"__module__": "keyring.backends.libsecret"})()
        native.get_password = Mock(side_effect=AssertionError("would prompt"))
        _select_chain(monkeypatch, backend, native)
        assert load_passphrase_from_desktop_keyring() == _PASSPHRASE
        native.get_password.assert_not_called()


class TestUnsupportedImplicitUnlock:
    """A native backend without prompt control cannot act as a background reader."""

    @pytest.mark.parametrize("chained", [False, True])
    def test_libsecret_never_gets_a_background_read(
        self, monkeypatch: pytest.MonkeyPatch, chained: bool
    ) -> None:
        backend_type = type("Keyring", (), {"__module__": "keyring.backends.libsecret"})
        backend = backend_type()
        backend.get_password = Mock(side_effect=AssertionError("would prompt"))
        backend.set_password = Mock()
        if chained:
            _select_chain(monkeypatch, backend)
        else:
            monkeypatch.setattr(keyring, "get_keyring", lambda: backend)
        assert encryption.desktop_keyring_read_blocked() is not None
        assert not encryption.desktop_keyring_backend_available()
        assert load_passphrase_from_desktop_keyring() is None
        assert not store_passphrase_in_desktop_keyring(_PASSPHRASE)
        backend.get_password.assert_not_called()
        backend.set_password.assert_not_called()

    def test_headless_interactive_request_is_still_bounded(
        self, backend: Mock, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.delenv("DISPLAY", raising=False)
        monkeypatch.delenv("WAYLAND_DISPLAY", raising=False)
        bounded = Mock(wraps=encryption._call_with_timeout)
        monkeypatch.setattr(encryption, "_call_with_timeout", bounded)
        assert load_passphrase_from_desktop_keyring(allow_prompt=True) == _PASSPHRASE
        assert bounded.call_count == 1


class TestVerifiedDesktopDelete:
    """Unavailable readback is not evidence that the passphrase was removed."""

    @pytest.mark.parametrize("missing_error", [False, True])
    def test_absence_is_verified(self, backend: Mock, missing_error: bool) -> None:
        backend.get_password.return_value = None
        if missing_error:
            backend.delete_password.side_effect = PasswordDeleteError("absent")
        assert forget_passphrase_in_desktop_keyring() is None
        backend.get_password.assert_called_once()

    @pytest.mark.parametrize("missing_error", [False, True])
    def test_silent_noop_and_false_missing_are_failures(self, backend: Mock, missing_error: bool):
        if missing_error:
            backend.delete_password.side_effect = PasswordDeleteError("still present")
        assert forget_passphrase_in_desktop_keyring() is not None

    @pytest.mark.parametrize("operation", ["delete_password", "get_password"])
    def test_backend_error_is_not_absence_or_secret_disclosure(
        self, backend: Mock, operation: str, caplog: pytest.LogCaptureFixture
    ) -> None:
        getattr(backend, operation).side_effect = RuntimeError(_PASSPHRASE)
        reason = forget_passphrase_in_desktop_keyring()
        assert reason is not None
        assert _PASSPHRASE not in reason
        assert _PASSPHRASE not in caplog.text

    def test_secret_service_delete_does_not_unlock_collection(
        self, secret_service: tuple[SecretService, Mock, Mock]
    ) -> None:
        _backend, collection, _connection = secret_service
        collection.ensure_not_locked.side_effect = LockedException("locked after probe")
        assert forget_passphrase_in_desktop_keyring() is not None
        collection.unlock.assert_not_called()

    def test_secret_service_delete_verifies_the_actual_item(
        self, secret_service: tuple[SecretService, Mock, Mock]
    ) -> None:
        _backend, collection, _connection = secret_service
        [item] = collection.search_items.return_value

        def delete(*_args: object) -> tuple[str]:
            collection.search_items.return_value = []
            return (encryption._SECRET_SERVICE_NO_PROMPT,)

        item._item.call.side_effect = delete
        assert forget_passphrase_in_desktop_keyring() is None
        item._item.call.assert_called_once_with("Delete", "")
        item.delete.assert_not_called()
        collection.unlock.assert_not_called()

    def test_secret_service_confirmation_never_opens_a_dialog(
        self,
        secret_service: tuple[SecretService, Mock, Mock],
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        _backend, collection, _connection = secret_service
        [item] = collection.search_items.return_value
        item._item.call.return_value = (_DELETE_PROMPT_PATH,)
        prompt = Mock(side_effect=AssertionError("must not prompt"))
        monkeypatch.setattr(secretstorage.item, "exec_prompt", prompt)
        assert (
            forget_passphrase_in_desktop_keyring()
            == "desktop keyring requires deletion confirmation"
        )
        item._item.call.assert_called_once_with("Delete", "")
        item.delete.assert_not_called()
        prompt.assert_not_called()

    def test_delete_checks_other_chained_backends(
        self, backend: Mock, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        backend.get_password.return_value = None
        other = Mock()
        other.get_password.return_value = _PASSPHRASE
        _select_chain(monkeypatch, backend, other)
        assert forget_passphrase_in_desktop_keyring() is not None
        other.delete_password.assert_not_called()


def _select_chain(monkeypatch: pytest.MonkeyPatch, *backends: object) -> ChainerBackend:
    """Select a fixed backend chain without host backend discovery."""
    monkeypatch.setattr(ChainerBackend, "backends", list(backends))
    chain = ChainerBackend()
    monkeypatch.setattr(keyring, "get_keyring", lambda: chain)
    return chain
