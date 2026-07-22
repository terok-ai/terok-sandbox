# SPDX-FileCopyrightText: 2026 Jiri Vyskocil
# SPDX-License-Identifier: Apache-2.0

"""Unit tests for the kernel-keyring passphrase tier binding.

The real ``add_key(2)`` syscall is blocked by the default Podman seccomp
profile (``ENOSYS``) inside the CI/dev container, so these tests drive
the module against an **in-memory fake of ``libkeyutils``** — swapped in
via ``_load_library`` — which exercises the store/load/forget/probe
logic and every error branch deterministically on any host.  A single
opt-in round-trip against the *real* keyring is ``skipif``-gated on the
facility actually being available, so it validates the live ctypes
signatures on an unconfined host (a bare-metal CI runner) while staying
inert in the seccomp sandbox.
"""

from __future__ import annotations

import ctypes

import pytest

from terok_sandbox.vault.store import kernel_keyring

# Captured at import, before the package-level autouse ``_isolate_credential_keyring``
# fixture swaps these for deterministic stubs.  This module tests the real
# implementations (against a fake library), so an autouse fixture below restores
# them for every test here.
_REAL_FUNCS = {
    name: getattr(kernel_keyring, name)
    for name in ("load", "store", "forget", "unavailable_reason")
}


@pytest.fixture(autouse=True)
def _restore_real_kernel_keyring(monkeypatch: pytest.MonkeyPatch) -> None:
    """Undo conftest's global kernel-keyring stubs so these tests hit the real code."""
    for name, func in _REAL_FUNCS.items():
        monkeypatch.setattr(kernel_keyring, name, func)


class FakeKeyutils:
    """In-memory stand-in for the ``libkeyutils`` handle.

    Models exactly the six entry points
    [`kernel_keyring._load_library`][terok_sandbox.vault.store.kernel_keyring._load_library]
    configures, with knobs to force each failure mode.  ``errno`` is set
    through ``ctypes`` so the module's ``os.strerror(ctypes.get_errno())``
    diagnostics render as they would against the real library.
    """

    def __init__(
        self,
        *,
        get_keyring_id: int = 100,
        add_key_errno: int | None = None,
        setperm_ok: bool = True,
    ) -> None:
        self._keys: dict[bytes, tuple[int, bytes]] = {}
        self._by_serial: dict[int, bytes] = {}
        self._next_serial = 1000
        self._get_keyring_id = get_keyring_id
        self._add_key_errno = add_key_errno
        self._setperm_ok = setperm_ok
        self.perms: dict[int, int] = {}

    def keyctl_get_keyring_ID(self, _ring: int, _create: int) -> int:  # noqa: N802
        if self._get_keyring_id < 0:
            ctypes.set_errno(38)  # ENOSYS
        return self._get_keyring_id

    def add_key(self, _ktype: bytes, desc: bytes, payload: bytes, plen: int, _ring: int) -> int:
        if self._add_key_errno is not None:
            ctypes.set_errno(self._add_key_errno)
            return -1
        serial = self._next_serial
        self._next_serial += 1
        self._keys[desc] = (serial, payload[:plen])
        self._by_serial[serial] = desc
        return serial

    def keyctl_search(self, _ring: int, _ktype: bytes, desc: bytes, _dest: int) -> int:
        if desc not in self._keys:
            ctypes.set_errno(126)  # ENOKEY
            return -1
        return self._keys[desc][0]

    def keyctl_read(self, serial: int, buf: object, _buflen: int) -> int:
        desc = self._by_serial.get(serial)
        if desc is None:
            return -1
        payload = self._keys[desc][1]
        if buf is not None:
            ctypes.memmove(buf, payload, len(payload))
        return len(payload)

    def keyctl_setperm(self, serial: int, perm: int) -> int:  # noqa: N802 (mirror C name)
        if not self._setperm_ok:
            ctypes.set_errno(1)
            return -1
        self.perms[serial] = perm
        return 0

    def keyctl_link(self, _key: int, _ring: int) -> int:  # noqa: N802 (mirror C name)
        return 0

    def keyctl_unlink(self, serial: int, _ring: int) -> int:
        desc = self._by_serial.pop(serial, None)
        if desc is not None:
            self._keys.pop(desc, None)
        return 0


@pytest.fixture
def fake_lib(monkeypatch: pytest.MonkeyPatch) -> FakeKeyutils:
    """Install a fresh [`FakeKeyutils`][tests.unit.test_kernel_keyring.FakeKeyutils] as the library."""
    lib = FakeKeyutils()
    monkeypatch.setattr(kernel_keyring, "_load_library", lambda: lib)
    return lib


# ── unavailable_reason ──────────────────────────────────────────────


def test_unavailable_reason_none_when_facility_present(fake_lib: FakeKeyutils) -> None:
    assert kernel_keyring.unavailable_reason() is None


def test_unavailable_reason_reports_enosys(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(kernel_keyring, "_load_library", lambda: FakeKeyutils(get_keyring_id=-1))
    reason = kernel_keyring.unavailable_reason()
    assert reason is not None
    assert "CONFIG_KEYS" in reason


def test_unavailable_reason_reports_missing_library(monkeypatch: pytest.MonkeyPatch) -> None:
    def _raise() -> object:
        raise kernel_keyring._KeyutilsUnavailable("libkeyutils not loadable")

    monkeypatch.setattr(kernel_keyring, "_load_library", _raise)
    assert kernel_keyring.unavailable_reason() == "libkeyutils not loadable"


# ── store / load round-trip ─────────────────────────────────────────


def test_store_then_load_round_trips(fake_lib: FakeKeyutils) -> None:
    assert kernel_keyring.store("s3cr3t-éé with spaces") is True
    assert kernel_keyring.load() == "s3cr3t-éé with spaces"


def test_store_locks_perms_possessor_and_uid_only(fake_lib: FakeKeyutils) -> None:
    assert kernel_keyring.store("pw") is True
    (serial,) = fake_lib.perms
    # Possessor-all + uid view/read/write/search/setattr, group/other zero.
    assert fake_lib.perms[serial] == 0x3F2F0000


def test_store_arms_no_timeout(fake_lib: FakeKeyutils) -> None:
    # The cache persists until an explicit forget — never a timed expiry.
    assert not hasattr(fake_lib, "keyctl_set_timeout")


def test_store_updates_in_place(fake_lib: FakeKeyutils) -> None:
    kernel_keyring.store("first")
    kernel_keyring.store("second")
    assert kernel_keyring.load() == "second"


def test_store_rejects_empty_passphrase(fake_lib: FakeKeyutils) -> None:
    with pytest.raises(ValueError, match="empty passphrase"):
        kernel_keyring.store("")


def test_store_rejects_oversize_passphrase(fake_lib: FakeKeyutils) -> None:
    with pytest.raises(ValueError, match="exceeds"):
        kernel_keyring.store("x" * 5000)


# ── store failure branches ──────────────────────────────────────────


def test_store_returns_false_on_add_key_failure(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(
        kernel_keyring,
        "_load_library",
        lambda: FakeKeyutils(add_key_errno=122),  # EDQUOT
    )
    assert kernel_keyring.store("pw") is False


def test_store_unlinks_when_setperm_fails(monkeypatch: pytest.MonkeyPatch) -> None:
    lib = FakeKeyutils(setperm_ok=False)
    monkeypatch.setattr(kernel_keyring, "_load_library", lambda: lib)
    assert kernel_keyring.store("pw") is False
    # Rolled back — no readable key left behind.
    assert kernel_keyring.load() is None


def test_store_returns_false_when_library_unavailable(monkeypatch: pytest.MonkeyPatch) -> None:
    def _raise() -> object:
        raise kernel_keyring._KeyutilsUnavailable("nope")

    monkeypatch.setattr(kernel_keyring, "_load_library", _raise)
    assert kernel_keyring.store("pw") is False


# ── load / forget ───────────────────────────────────────────────────


def test_load_returns_none_when_absent(fake_lib: FakeKeyutils) -> None:
    assert kernel_keyring.load() is None


def test_load_returns_none_when_library_unavailable(monkeypatch: pytest.MonkeyPatch) -> None:
    def _raise() -> object:
        raise kernel_keyring._KeyutilsUnavailable("nope")

    monkeypatch.setattr(kernel_keyring, "_load_library", _raise)
    assert kernel_keyring.load() is None


def test_forget_removes_the_key(fake_lib: FakeKeyutils) -> None:
    kernel_keyring.store("pw")
    assert kernel_keyring.load() == "pw"
    assert kernel_keyring.forget() is True
    assert kernel_keyring.load() is None


def test_forget_is_idempotent_when_absent(fake_lib: FakeKeyutils) -> None:
    assert kernel_keyring.forget() is True


def test_forget_true_when_library_unavailable(monkeypatch: pytest.MonkeyPatch) -> None:
    def _raise() -> object:
        raise kernel_keyring._KeyutilsUnavailable("nope")

    monkeypatch.setattr(kernel_keyring, "_load_library", _raise)
    assert kernel_keyring.forget() is True


# ── live facility (only where the kernel keyring actually works) ─────


@pytest.mark.skipif(
    kernel_keyring.unavailable_reason() is not None,
    reason="kernel keyring facility unavailable here (no CONFIG_KEYS / no libkeyutils)",
)
def test_real_round_trip(monkeypatch: pytest.MonkeyPatch) -> None:
    """Validate the live ctypes signatures against the real keyring.

    Uses a test-unique description so it can't collide with a real
    vault cache, and cleans up after itself.  ``unavailable_reason`` is
    a side-effect-free probe (``store``'s return is the definitive
    answer), so on a host where ``add_key`` is filtered even though
    ``keyctl`` isn't — e.g. inside a default-seccomp Podman container —
    the write fails and the test skips rather than failing; it runs for
    real on an unconfined runner.
    """
    monkeypatch.setattr(kernel_keyring, "KEY_DESCRIPTION", b"terok-sandbox:pytest-probe")
    if not kernel_keyring.store("live-value"):
        pytest.skip("add_key not permitted here (seccomp) — nothing to validate live")
    try:
        assert kernel_keyring.load() == "live-value"
    finally:
        kernel_keyring.forget()
    assert kernel_keyring.load() is None
