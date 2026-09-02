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
import ctypes.util
import errno

import pytest

from terok_sandbox.vault.store import kernel_keyring
from tests.constants import MOCK_BASE

#: A stable, vault-unique identity for the per-DB key scoping.  Nothing is
#: written here — the path only seeds
#: [`kernel_keyring.key_description`][terok_sandbox.vault.store.kernel_keyring.key_description],
#: so a plain ``MOCK_BASE`` constant (never the operator's real DB) is enough.
MOCK_DB_PATH = MOCK_BASE / "kernel-keyring" / "credentials.db"

# Captured at import, before the package-level autouse ``_isolate_credential_keyring``
# fixture swaps these for deterministic stubs.  This module tests the real
# implementations (against a fake library), so an autouse fixture below restores
# them for every test here.
_REAL_FUNCS = {
    name: getattr(kernel_keyring, name)
    for name in ("load", "store", "forget", "is_cached", "unavailable_reason")
}


@pytest.fixture(autouse=True)
def _restore_real_kernel_keyring(monkeypatch: pytest.MonkeyPatch) -> None:
    """Undo conftest's global kernel-keyring stubs so these tests hit the real code."""
    for name, func in _REAL_FUNCS.items():
        monkeypatch.setattr(kernel_keyring, name, func)


#: Ring specs from ``linux/keyctl.h`` the fake resolves, mirroring the
#: module's own constants.
_UID_RING = -4
_SESSION_RING = -3


class FakeKeyutils:
    """In-memory stand-in for the ``libkeyutils`` handle.

    Models exactly the six entry points
    [`kernel_keyring._load_library`][terok_sandbox.vault.store.kernel_keyring._load_library]
    configures, with knobs to force each failure mode.  ``errno`` is set
    through ``ctypes`` so the module's ``os.strerror(ctypes.get_errno())``
    diagnostics render as they would against the real library.

    Keyrings are modelled by *identity* rather than by spec, because the
    two differ exactly where this tier is hard: a ring spec resolves
    per-namespace, so ``@u`` names one keyring for the operator and
    another for a process inside podman's rootless namespace, while
    ``@s`` names the same keyring for both.
    [`enter_user_namespace`][tests.unit.test_kernel_keyring.FakeKeyutils.enter_user_namespace]
    reproduces that crossing.  Searches follow links between rings, as
    ``keyctl_search`` does.
    """

    def __init__(
        self,
        *,
        get_keyring_id: int = 100,
        add_key_errno: int | None = None,
        setperm_ok: bool = True,
        search_errno: int | None = None,
        link_ok: bool = True,
    ) -> None:
        self._keys: dict[bytes, tuple[int, bytes]] = {}
        self._by_serial: dict[int, bytes] = {}
        self._next_serial = 1000
        self._get_keyring_id = get_keyring_id
        self._add_key_errno = add_key_errno
        self._setperm_ok = setperm_ok
        self._search_errno = search_errno
        self._link_ok = link_ok
        self.perms: dict[int, int] = {}
        # Ring identity → the descriptions it holds, and the rings linked
        # into it.  ``_rings`` maps the spec a caller passes to one identity.
        self._next_ring = 1
        self._rings = {_UID_RING: self._new_ring(), _SESSION_RING: self._new_ring()}
        self._holds: dict[int, set[bytes]] = {r: set() for r in self._rings.values()}
        self._nested: dict[int, set[int]] = {r: set() for r in self._rings.values()}

    def enter_user_namespace(self) -> None:
        """Re-resolve ``@u`` to a fresh empty keyring, leaving ``@s`` as it was.

        What a rootless supervisor child sees: its own per-namespace user
        keyring, and the session keyring it inherited untouched.
        """
        ring = self._new_ring()
        self._rings[_UID_RING] = ring
        self._holds[ring] = set()
        self._nested[ring] = set()

    def _new_ring(self) -> int:
        """Return a fresh keyring identity."""
        ring = self._next_ring
        self._next_ring += 1
        return ring

    def keyctl_get_keyring_ID(self, _ring: int, _create: int) -> int:  # noqa: N802
        if self._get_keyring_id < 0:
            ctypes.set_errno(38)  # ENOSYS
        return self._get_keyring_id

    def add_key(self, _ktype: bytes, desc: bytes, payload: bytes, plen: int, ring: int) -> int:
        if self._add_key_errno is not None:
            ctypes.set_errno(self._add_key_errno)
            return -1
        serial = self._next_serial
        self._next_serial += 1
        self._keys[desc] = (serial, payload[:plen])
        self._by_serial[serial] = desc
        self._holds[self._rings[ring]].add(desc)
        return serial

    def keyctl_search(self, ring: int, _ktype: bytes, desc: bytes, _dest: int) -> int:
        if self._search_errno is not None:
            ctypes.set_errno(self._search_errno)
            return -1
        if not self._reaches(self._rings[ring], desc):
            ctypes.set_errno(126)  # ENOKEY
            return -1
        return self._keys[desc][0]

    def _reaches(self, ring: int, desc: bytes, seen: frozenset[int] = frozenset()) -> bool:
        """Is *desc* in *ring* or in any keyring linked into it?"""
        if desc in self._holds[ring]:
            return True
        return any(
            self._reaches(nested, desc, seen | {ring}) for nested in self._nested[ring] - seen
        )

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

    def keyctl_link(self, key: int, ring: int) -> int:  # noqa: N802 (mirror C name)
        if not self._link_ok:
            ctypes.set_errno(13)  # EACCES
            return -1
        self._nested[self._rings[ring]].add(self._rings[key])
        return 0

    def keyctl_unlink(self, serial: int, _ring: int) -> int:
        desc = self._by_serial.pop(serial, None)
        if desc is not None:
            self._keys.pop(desc, None)
            for held in self._holds.values():
                held.discard(desc)
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


def test_key_description_scopes_by_host_and_path(monkeypatch: pytest.MonkeyPatch) -> None:
    """The ``@u`` key id folds in both hostname and DB path — neither collides alone.

    Path keeps a test's throwaway DB off the operator's real key on one
    host; hostname separates environments that share a ``@u`` but differ by
    UTS namespace (concurrent rootless containers with identical paths).
    """
    monkeypatch.setattr("socket.gethostname", lambda: "host-a")
    a_one = kernel_keyring.key_description("/vault/one/credentials.db")
    a_two = kernel_keyring.key_description("/vault/two/credentials.db")
    monkeypatch.setattr("socket.gethostname", lambda: "host-b")
    b_one = kernel_keyring.key_description("/vault/one/credentials.db")

    assert a_one != a_two  # same host, different path
    assert a_one != b_one  # same path, different host
    assert a_one.startswith(kernel_keyring.KEY_DESCRIPTION_PREFIX)


def test_store_then_load_round_trips(fake_lib: FakeKeyutils) -> None:
    assert kernel_keyring.store("s3cr3t-éé with spaces", MOCK_DB_PATH) is True
    assert kernel_keyring.load(MOCK_DB_PATH) == "s3cr3t-éé with spaces"


def test_store_locks_perms_possessor_and_uid_only(fake_lib: FakeKeyutils) -> None:
    assert kernel_keyring.store("pw", MOCK_DB_PATH) is True
    (serial,) = fake_lib.perms
    # Possessor-all + uid view/read/write/search/setattr, group/other zero.
    assert fake_lib.perms[serial] == 0x3F2F0000


def test_store_arms_no_timeout(fake_lib: FakeKeyutils) -> None:
    # The cache persists until an explicit forget — never a timed expiry.
    assert not hasattr(fake_lib, "keyctl_set_timeout")


def test_store_updates_in_place(fake_lib: FakeKeyutils) -> None:
    kernel_keyring.store("first", MOCK_DB_PATH)
    kernel_keyring.store("second", MOCK_DB_PATH)
    assert kernel_keyring.load(MOCK_DB_PATH) == "second"


def test_store_rejects_empty_passphrase(fake_lib: FakeKeyutils) -> None:
    with pytest.raises(ValueError, match="empty passphrase"):
        kernel_keyring.store("", MOCK_DB_PATH)


def test_store_rejects_oversize_passphrase(fake_lib: FakeKeyutils) -> None:
    with pytest.raises(ValueError, match="exceeds"):
        kernel_keyring.store("x" * 5000, MOCK_DB_PATH)


# ── store failure branches ──────────────────────────────────────────


def test_store_returns_false_on_add_key_failure(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(
        kernel_keyring,
        "_load_library",
        lambda: FakeKeyutils(add_key_errno=122),  # EDQUOT
    )
    assert kernel_keyring.store("pw", MOCK_DB_PATH) is False


def test_store_unlinks_when_setperm_fails(monkeypatch: pytest.MonkeyPatch) -> None:
    lib = FakeKeyutils(setperm_ok=False)
    monkeypatch.setattr(kernel_keyring, "_load_library", lambda: lib)
    assert kernel_keyring.store("pw", MOCK_DB_PATH) is False
    # Rolled back — no readable key left behind.
    assert kernel_keyring.load(MOCK_DB_PATH) is None


def test_store_returns_false_when_library_unavailable(monkeypatch: pytest.MonkeyPatch) -> None:
    def _raise() -> object:
        raise kernel_keyring._KeyutilsUnavailable("nope")

    monkeypatch.setattr(kernel_keyring, "_load_library", _raise)
    assert kernel_keyring.store("pw", MOCK_DB_PATH) is False


# ── reading across a user namespace ─────────────────────────────────


def test_load_crosses_a_user_namespace_through_the_session_keyring(
    fake_lib: FakeKeyutils,
) -> None:
    """The supervisor's children read the operator's cache, or the vault won't open.

    They run inside podman's rootless user namespace, where ``@u`` is a
    different keyring — an empty one.  The session keyring crosses that
    boundary unchanged and holds the link ``store`` left behind, which is
    the only route to the key from in there.
    """
    kernel_keyring.store("s3cret", MOCK_DB_PATH)

    fake_lib.enter_user_namespace()

    assert kernel_keyring.load(MOCK_DB_PATH) == "s3cret"
    assert kernel_keyring.is_cached(MOCK_DB_PATH) is True


def test_load_stays_blind_across_a_namespace_without_the_session_link(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Without the ``@u`` link there is no bridge, and the miss is honest.

    The pair to the test above: it is the link that carries the cache
    across, not the crossing being harmless.
    """
    lib = FakeKeyutils(link_ok=False)
    monkeypatch.setattr(kernel_keyring, "_load_library", lambda: lib)
    kernel_keyring.store("s3cret", MOCK_DB_PATH)

    lib.enter_user_namespace()

    assert kernel_keyring.load(MOCK_DB_PATH) is None


def test_is_bridged_answers_for_the_reader_in_the_other_namespace(
    fake_lib: FakeKeyutils,
) -> None:
    """It asks the ``@s`` leg alone, because that is the only leg those readers have."""
    assert kernel_keyring.is_bridged(MOCK_DB_PATH) is False
    kernel_keyring.store("s3cret", MOCK_DB_PATH)
    assert kernel_keyring.is_bridged(MOCK_DB_PATH) is True


def test_is_bridged_is_false_without_the_session_link(monkeypatch: pytest.MonkeyPatch) -> None:
    """A cache the operator can read and the supervisor cannot is the reported state."""
    lib = FakeKeyutils(link_ok=False)
    monkeypatch.setattr(kernel_keyring, "_load_library", lambda: lib)
    kernel_keyring.store("s3cret", MOCK_DB_PATH)

    assert kernel_keyring.load(MOCK_DB_PATH) == "s3cret"
    assert kernel_keyring.is_bridged(MOCK_DB_PATH) is False


def test_is_bridged_is_false_without_the_facility(monkeypatch: pytest.MonkeyPatch) -> None:
    """No keyring at all is one more way for the supervisor not to find it."""

    def _raise() -> object:
        raise kernel_keyring._KeyutilsUnavailable("libkeyutils not loadable")

    monkeypatch.setattr(kernel_keyring, "_load_library", _raise)
    assert kernel_keyring.is_bridged(MOCK_DB_PATH) is False


def test_a_revoked_key_is_a_miss_and_says_nothing(
    monkeypatch: pytest.MonkeyPatch, fake_lib: FakeKeyutils, caplog: pytest.LogCaptureFixture
) -> None:
    """A dead key is the same answer as no key, and not worth a word.

    The session-keyring leg walks keys the user-keyring leg never saw, and
    a revoked one among them used to raise — so an operator whose vault
    runs on a different tier entirely got two lines about the keyring on
    every single container start, about a key that can never be read
    again and that no action of theirs would change.
    """

    def _revoked(_ring: int, _ktype: bytes, _desc: bytes, _dest: int) -> int:
        ctypes.set_errno(errno.EKEYREVOKED)
        return -1

    monkeypatch.setattr(fake_lib, "keyctl_search", _revoked)

    with caplog.at_level("WARNING"):
        assert kernel_keyring.load(MOCK_DB_PATH) is None
        assert kernel_keyring.is_cached(MOCK_DB_PATH) is False
        # Its contract is the end state, and a key that yields nothing meets it.
        assert kernel_keyring.forget(MOCK_DB_PATH) is True

    assert caplog.text == ""


def test_a_permission_fault_is_still_reported(
    monkeypatch: pytest.MonkeyPatch, fake_lib: FakeKeyutils, caplog: pytest.LogCaptureFixture
) -> None:
    """The other half of the rule: a lookup that did not complete is not an answer."""

    def _denied(_ring: int, _ktype: bytes, _desc: bytes, _dest: int) -> int:
        ctypes.set_errno(errno.EACCES)
        return -1

    monkeypatch.setattr(fake_lib, "keyctl_search", _denied)

    with caplog.at_level("WARNING"):
        assert kernel_keyring.load(MOCK_DB_PATH) is None
        # A cache that may still be there must never be reported as cleared.
        assert kernel_keyring.forget(MOCK_DB_PATH) is False

    assert "Permission denied" in caplog.text


def test_store_warns_when_the_session_link_fails(
    monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture
) -> None:
    """A failed link is the supervisor losing the cache — say so."""
    monkeypatch.setattr(kernel_keyring, "_load_library", lambda: FakeKeyutils(link_ok=False))

    with caplog.at_level("WARNING"):
        kernel_keyring.store("s3cret", MOCK_DB_PATH)

    assert "supervisor child" in caplog.text


def test_a_faulting_keyring_does_not_veto_the_next_one(
    monkeypatch: pytest.MonkeyPatch, fake_lib: FakeKeyutils
) -> None:
    """A hit anywhere is a hit — an earlier permission fault is not the answer."""
    kernel_keyring.store("s3cret", MOCK_DB_PATH)
    real_search = fake_lib.keyctl_search

    def _faulting(ring: int, ktype: bytes, desc: bytes, dest: int) -> int:
        if ring == _UID_RING:
            ctypes.set_errno(13)  # EACCES
            return -1
        return real_search(ring, ktype, desc, dest)

    monkeypatch.setattr(fake_lib, "keyctl_search", _faulting)

    assert kernel_keyring.load(MOCK_DB_PATH) == "s3cret"


# ── load / forget ───────────────────────────────────────────────────


def test_load_returns_none_when_absent(fake_lib: FakeKeyutils) -> None:
    assert kernel_keyring.load(MOCK_DB_PATH) is None


def test_load_returns_none_when_library_unavailable(monkeypatch: pytest.MonkeyPatch) -> None:
    def _raise() -> object:
        raise kernel_keyring._KeyutilsUnavailable("nope")

    monkeypatch.setattr(kernel_keyring, "_load_library", _raise)
    assert kernel_keyring.load(MOCK_DB_PATH) is None


def test_forget_removes_the_key(fake_lib: FakeKeyutils) -> None:
    kernel_keyring.store("pw", MOCK_DB_PATH)
    assert kernel_keyring.load(MOCK_DB_PATH) == "pw"
    assert kernel_keyring.forget(MOCK_DB_PATH) is True
    assert kernel_keyring.load(MOCK_DB_PATH) is None


def test_forget_is_idempotent_when_absent(fake_lib: FakeKeyutils) -> None:
    assert kernel_keyring.forget(MOCK_DB_PATH) is True


def test_is_cached_reflects_presence(fake_lib: FakeKeyutils) -> None:
    assert kernel_keyring.is_cached(MOCK_DB_PATH) is False
    kernel_keyring.store("pw", MOCK_DB_PATH)
    assert kernel_keyring.is_cached(MOCK_DB_PATH) is True
    kernel_keyring.forget(MOCK_DB_PATH)
    assert kernel_keyring.is_cached(MOCK_DB_PATH) is False


def test_is_cached_never_reads_the_payload(
    fake_lib: FakeKeyutils, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Presence must not materialise the secret — status surfaces poll this."""
    kernel_keyring.store("pw", MOCK_DB_PATH)

    def _explode(*_args: object) -> int:
        raise AssertionError("is_cached must not call keyctl_read")

    monkeypatch.setattr(fake_lib, "keyctl_read", _explode)
    assert kernel_keyring.is_cached(MOCK_DB_PATH) is True


def test_is_cached_false_when_library_unavailable(monkeypatch: pytest.MonkeyPatch) -> None:
    def _raise() -> object:
        raise kernel_keyring._KeyutilsUnavailable("nope")

    monkeypatch.setattr(kernel_keyring, "_load_library", _raise)
    assert kernel_keyring.is_cached(MOCK_DB_PATH) is False


def test_forget_reports_failure_when_unlink_fails(
    fake_lib: FakeKeyutils, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A present key whose unlink is refused must report failure, not success."""
    kernel_keyring.store("pw", MOCK_DB_PATH)
    monkeypatch.setattr(fake_lib, "keyctl_unlink", lambda _serial, _ring: -1)
    assert kernel_keyring.forget(MOCK_DB_PATH) is False


def test_load_returns_none_when_read_reports_no_length(
    fake_lib: FakeKeyutils, monkeypatch: pytest.MonkeyPatch
) -> None:
    kernel_keyring.store("pw", MOCK_DB_PATH)
    monkeypatch.setattr(fake_lib, "keyctl_read", lambda _s, _b, _l: 0)
    assert kernel_keyring.load(MOCK_DB_PATH) is None


def test_load_returns_none_when_second_read_fails(
    fake_lib: FakeKeyutils, monkeypatch: pytest.MonkeyPatch
) -> None:
    kernel_keyring.store("pw", MOCK_DB_PATH)
    # First pass (buf is None) sizes the payload; the fill pass then fails.
    monkeypatch.setattr(fake_lib, "keyctl_read", lambda _s, buf, _l: 8 if buf is None else 0)
    assert kernel_keyring.load(MOCK_DB_PATH) is None


def test_unavailable_reason_reports_non_enosys_errno(monkeypatch: pytest.MonkeyPatch) -> None:
    lib = FakeKeyutils()
    monkeypatch.setattr(lib, "keyctl_get_keyring_ID", lambda _r, _c: (ctypes.set_errno(13), -1)[1])
    monkeypatch.setattr(kernel_keyring, "_load_library", lambda: lib)
    reason = kernel_keyring.unavailable_reason()
    assert reason is not None
    assert "user keyring unreachable" in reason


def test_load_library_reports_unloadable(monkeypatch: pytest.MonkeyPatch) -> None:
    """A soname that won't load degrades to unavailable, not a crash."""
    kernel_keyring._load_library.cache_clear()
    monkeypatch.setattr(ctypes.util, "find_library", lambda _n: "libkeyutils.so.1")

    def _boom(*_a: object, **_k: object) -> object:
        raise OSError("cannot open shared object")

    monkeypatch.setattr(ctypes, "CDLL", _boom)
    reason = kernel_keyring.unavailable_reason()
    kernel_keyring._load_library.cache_clear()
    assert reason is not None
    assert "not loadable" in reason


def test_load_library_reports_missing_symbol(monkeypatch: pytest.MonkeyPatch) -> None:
    """A wrong/incompatible libkeyutils (missing an expected symbol) degrades too."""
    kernel_keyring._load_library.cache_clear()
    monkeypatch.setattr(ctypes.util, "find_library", lambda _n: "libkeyutils.so.1")
    # A bare object has no ``add_key`` — binding its restype raises AttributeError.
    monkeypatch.setattr(ctypes, "CDLL", lambda *_a, **_k: object())
    reason = kernel_keyring.unavailable_reason()
    kernel_keyring._load_library.cache_clear()
    assert reason is not None
    assert "missing expected symbol" in reason


def test_forget_reports_failure_on_lookup_error(monkeypatch: pytest.MonkeyPatch) -> None:
    """A non-ENOKEY search failure must NOT read as 'nothing to forget'.

    Otherwise ``vault lock`` claims the passphrase is cleared while a key
    may still be cached.
    """
    lib = FakeKeyutils(search_errno=13)  # EACCES — not ENOKEY
    monkeypatch.setattr(kernel_keyring, "_load_library", lambda: lib)
    assert kernel_keyring.forget(MOCK_DB_PATH) is False


def test_load_none_on_lookup_error(monkeypatch: pytest.MonkeyPatch) -> None:
    lib = FakeKeyutils(search_errno=13)  # EACCES
    monkeypatch.setattr(kernel_keyring, "_load_library", lambda: lib)
    assert kernel_keyring.load(MOCK_DB_PATH) is None


def test_is_cached_false_on_lookup_error(monkeypatch: pytest.MonkeyPatch) -> None:
    lib = FakeKeyutils(search_errno=13)  # EACCES
    monkeypatch.setattr(kernel_keyring, "_load_library", lambda: lib)
    assert kernel_keyring.is_cached(MOCK_DB_PATH) is False


def test_forget_true_when_library_unavailable(monkeypatch: pytest.MonkeyPatch) -> None:
    def _raise() -> object:
        raise kernel_keyring._KeyutilsUnavailable("nope")

    monkeypatch.setattr(kernel_keyring, "_load_library", _raise)
    assert kernel_keyring.forget(MOCK_DB_PATH) is True


# ── live facility (only where the kernel keyring actually works) ─────


@pytest.mark.skipif(
    kernel_keyring.unavailable_reason() is not None,
    reason="kernel keyring facility unavailable here (no CONFIG_KEYS / no libkeyutils)",
)
def test_real_round_trip() -> None:
    """Validate the live ctypes signatures against the real keyring.

    The mock DB path scopes the key to a test-unique description that
    can't collide with a real vault cache, and it cleans up after
    itself.  ``unavailable_reason`` is a side-effect-free probe
    (``store``'s return is the definitive answer), so on a host where
    ``add_key`` is filtered even though ``keyctl`` isn't — e.g. inside a
    default-seccomp Podman container — the write fails and the test skips
    rather than failing; it runs for real on an unconfined runner.
    """
    if not kernel_keyring.store("live-value", MOCK_DB_PATH):
        pytest.skip("add_key not permitted here (seccomp) — nothing to validate live")
    try:
        assert kernel_keyring.load(MOCK_DB_PATH) == "live-value"
    finally:
        kernel_keyring.forget(MOCK_DB_PATH)
    assert kernel_keyring.load(MOCK_DB_PATH) is None
