# SPDX-FileCopyrightText: 2026 Jiri Vyskocil
# SPDX-License-Identifier: Apache-2.0

"""Tests for the systemd-creds subprocess wrapper.

The wrapper is intentionally thin — every test stubs ``subprocess.run``
or ``shutil.which`` and asserts on argv / return-code translation.
Integration with a real ``systemd-creds`` binary is covered in
``tests/integration/`` (issue #277).
"""

from __future__ import annotations

import subprocess
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from terok_sandbox.credentials import systemd_creds


class TestAvailability:
    """The presence probes that gate every other operation."""

    def test_is_available_when_binary_recent_enough(self) -> None:
        """Fedora 44 / Debian 13 ship systemd 257+ which has the Varlink delegation."""
        with (
            patch("shutil.which", return_value="/usr/bin/systemd-creds"),
            patch("subprocess.run", return_value=_version_output(259)),
        ):
            assert systemd_creds.is_available() is True

    def test_is_available_false_when_systemd_too_old(self) -> None:
        """systemd < 257 lacks the non-root --user decrypt path — tier is unusable."""
        with (
            patch("shutil.which", return_value="/usr/bin/systemd-creds"),
            patch("subprocess.run", return_value=_version_output(256)),
        ):
            assert systemd_creds.is_available() is False

    def test_is_available_when_missing(self) -> None:
        with patch("shutil.which", return_value=None):
            assert systemd_creds.is_available() is False

    def test_is_available_when_version_probe_fails(self) -> None:
        with (
            patch("shutil.which", return_value="/usr/bin/systemd-creds"),
            patch(
                "subprocess.run",
                side_effect=subprocess.CalledProcessError(returncode=1, cmd=["systemd-creds"]),
            ),
        ):
            assert systemd_creds.is_available() is False

    def test_is_available_when_version_output_has_no_integer(self) -> None:
        """Garbage output (no version number) is treated as unavailable, not crash."""
        with (
            patch("shutil.which", return_value="/usr/bin/systemd-creds"),
            patch("subprocess.run", return_value=MagicMock(returncode=0, stdout="garbage\n")),
        ):
            assert systemd_creds.is_available() is False

    def test_has_tpm2_short_circuits_when_unavailable(self) -> None:
        """``is_available`` False → no TPM probe; no extra subprocess spawned."""
        with (
            patch("shutil.which", return_value=None),
            patch("subprocess.run") as run,
        ):
            assert systemd_creds.has_tpm2() is False
            run.assert_not_called()

    def test_has_tpm2_when_command_succeeds(self) -> None:
        with (
            patch("shutil.which", return_value="/usr/bin/systemd-creds"),
            patch(
                "subprocess.run",
                side_effect=[
                    _version_output(259),  # version probe inside is_available()
                    MagicMock(returncode=0),  # has-tpm2 probe
                ],
            ),
        ):
            assert systemd_creds.has_tpm2() is True

    def test_has_tpm2_when_command_fails(self) -> None:
        with (
            patch("shutil.which", return_value="/usr/bin/systemd-creds"),
            patch(
                "subprocess.run",
                side_effect=[_version_output(259), MagicMock(returncode=1)],
            ),
        ):
            assert systemd_creds.has_tpm2() is False

    def test_has_tpm2_swallows_timeout(self) -> None:
        with (
            patch("shutil.which", return_value="/usr/bin/systemd-creds"),
            patch(
                "subprocess.run",
                side_effect=[
                    _version_output(259),
                    subprocess.TimeoutExpired(cmd="systemd-creds", timeout=5),
                ],
            ),
        ):
            assert systemd_creds.has_tpm2() is False


@pytest.mark.usefixtures("_have_systemd_creds")
class TestSeal:
    """``seal`` shells out to ``systemd-creds encrypt`` and writes the captured blob atomically."""

    def test_user_mode_always_set(self, tmp_path: Path) -> None:
        """Every seal goes through ``--user`` so non-root Varlink delegation engages."""
        cred = tmp_path / "v.cred"
        with patch("subprocess.run", side_effect=_mock_seal_subprocess()) as run:
            systemd_creds.seal("pw", cred)
        assert "--user" in _seal_argv(run)

    def test_absolute_binary_path_used(self, tmp_path: Path) -> None:
        """Subprocess argv pins the resolved absolute path, not the bare name on PATH."""
        cred = tmp_path / "v.cred"
        with patch("subprocess.run", side_effect=_mock_seal_subprocess()) as run:
            systemd_creds.seal("pw", cred)
        argv = _seal_argv(run)
        assert argv[0] == _SYSTEMD_CREDS_EXE
        assert argv[1] == "encrypt"

    def test_namespaced_credential_name_embedded(self, tmp_path: Path) -> None:
        """``--name=`` is set to prevent cross-purpose reuse of the sealed blob."""
        cred = tmp_path / "v.cred"
        with patch("subprocess.run", side_effect=_mock_seal_subprocess()) as run:
            systemd_creds.seal("pw", cred)
        assert "--name=terok-sandbox.vault-passphrase" in _seal_argv(run)

    def test_default_key_mode_is_auto(self, tmp_path: Path) -> None:
        """Default delegates the host-vs-TPM choice to systemd's own auto-detection."""
        cred = tmp_path / "v.cred"
        with patch("subprocess.run", side_effect=_mock_seal_subprocess()) as run:
            systemd_creds.seal("pw", cred)
        assert "--with-key=auto" in _seal_argv(run)

    @pytest.mark.parametrize(
        ("key_mode", "expected_flag"),
        [
            ("auto", "--with-key=auto"),
            ("host", "--with-key=host"),
            ("tpm2", "--with-key=tpm2"),
            ("host+tpm2", "--with-key=host+tpm2"),
        ],
    )
    def test_key_mode_maps_through_to_systemd_flag(
        self, tmp_path: Path, key_mode: systemd_creds.KeyMode, expected_flag: str
    ) -> None:
        """The wrapper passes the operator's key_mode through verbatim — no reinterpretation."""
        cred = tmp_path / "v.cred"
        with patch("subprocess.run", side_effect=_mock_seal_subprocess()) as run:
            systemd_creds.seal("pw", cred, key_mode=key_mode)
        assert expected_flag in _seal_argv(run)

    def test_sealed_blob_routed_to_stdout_and_written_atomically(self, tmp_path: Path) -> None:
        """systemd-creds writes the blob to stdout; we capture and rename into place."""
        cred = tmp_path / "v.cred"
        with patch("subprocess.run", side_effect=_mock_seal_subprocess(b"opaque-blob")) as run:
            systemd_creds.seal("pw", cred)
        # ``-`` ``-`` means stdin → stdout, never touching the destination directly
        argv = _seal_argv(run)
        assert argv[-2:] == ["-", "-"]
        assert cred.read_bytes() == b"opaque-blob"

    def test_passphrase_piped_as_bytes_not_in_argv(self, tmp_path: Path) -> None:
        """The passphrase reaches systemd-creds as stdin bytes, never as an argv element."""
        cred = tmp_path / "v.cred"
        with patch("subprocess.run", side_effect=_mock_seal_subprocess()) as run:
            systemd_creds.seal("secret", cred)
        encrypt_call = run.call_args_list[1]
        assert encrypt_call.kwargs["input"] == b"secret"
        assert "secret" not in _seal_argv(run)

    def test_empty_passphrase_rejected(self, tmp_path: Path) -> None:
        with pytest.raises(ValueError, match="empty"):
            systemd_creds.seal("", tmp_path / "v.cred")

    def test_refuses_when_systemd_too_old(self, tmp_path: Path) -> None:
        """seal() bails before spawning encrypt when ``is_available`` says no."""
        with (
            patch("subprocess.run", return_value=_version_output(256)),
            pytest.raises(RuntimeError, match="needs systemd"),
        ):
            systemd_creds.seal("pw", tmp_path / "v.cred")

    def test_credential_file_created_at_0o600(self, tmp_path: Path) -> None:
        """Atomic-write via ``mkstemp`` materialises the leaf at 0o600 from inception — no umask window."""
        cred = tmp_path / "v.cred"
        with patch("subprocess.run", side_effect=_mock_seal_subprocess()):
            systemd_creds.seal("pw", cred)
        assert oct(cred.stat().st_mode & 0o777) == oct(0o600)

    def test_refuses_symlinked_leaf(self, tmp_path: Path) -> None:
        """A pre-existing symlink at the leaf would redirect the rename — refuse."""
        cred = tmp_path / "v.cred"
        target = tmp_path / "victim"
        target.touch()
        cred.symlink_to(target)
        with (
            patch("subprocess.run", side_effect=_mock_seal_subprocess()),
            pytest.raises(RuntimeError, match="symlinked credential path"),
        ):
            systemd_creds.seal("pw", cred)
        # The symlink target stays untouched.
        assert target.read_bytes() == b""

    def test_refuses_symlinked_parent(self, tmp_path: Path) -> None:
        """A symlinked parent would let ``mkstemp(dir=…)`` write into the link target — refuse."""
        real_dir = tmp_path / "real"
        real_dir.mkdir()
        link_dir = tmp_path / "vault"
        link_dir.symlink_to(real_dir)
        cred = link_dir / "v.cred"
        with (
            patch("subprocess.run", side_effect=_mock_seal_subprocess()),
            pytest.raises(RuntimeError, match="symlinked parent"),
        ):
            systemd_creds.seal("pw", cred)

    def test_binary_missing_raises_runtime_error(self, tmp_path: Path) -> None:
        """Race: ``is_available`` saw the binary but it vanished before encrypt."""
        cred = tmp_path / "v.cred"

        def _run(*_a: object, **kw: object) -> MagicMock:
            argv = _a[0] if _a else kw.get("args")
            if argv and "--version" in argv:
                return _version_output(259)
            raise FileNotFoundError("systemd-creds")

        with (
            patch("subprocess.run", side_effect=_run),
            pytest.raises(RuntimeError, match="not found on PATH"),
        ):
            systemd_creds.seal("pw", cred)

    def test_called_process_error_surfaces_stderr(self, tmp_path: Path) -> None:
        def _run(*_a: object, **kw: object) -> MagicMock:
            argv = _a[0] if _a else kw.get("args")
            if argv and "--version" in argv:
                return _version_output(259)
            raise subprocess.CalledProcessError(
                returncode=1, cmd=["systemd-creds"], stderr=b"TPM unavailable"
            )

        with (
            patch("subprocess.run", side_effect=_run),
            pytest.raises(RuntimeError, match="TPM unavailable"),
        ):
            systemd_creds.seal("pw", tmp_path / "v.cred")

    def test_timeout_translated_to_runtime_error(self, tmp_path: Path) -> None:
        """A hung ``systemd-creds`` surfaces as the documented RuntimeError, not TimeoutExpired."""

        def _run(*_a: object, **kw: object) -> MagicMock:
            argv = _a[0] if _a else kw.get("args")
            if argv and "--version" in argv:
                return _version_output(259)
            raise subprocess.TimeoutExpired(cmd=["systemd-creds"], timeout=10)

        with (
            patch("subprocess.run", side_effect=_run),
            pytest.raises(RuntimeError, match="timed out"),
        ):
            systemd_creds.seal("pw", tmp_path / "v.cred")

    def test_replace_failure_translated_to_runtime_error(self, tmp_path: Path) -> None:
        """An OSError on the atomic rename stays inside the documented contract."""
        cred = tmp_path / "v.cred"
        with (
            patch("subprocess.run", side_effect=_mock_seal_subprocess()),
            patch("os.replace", side_effect=OSError("read-only fs")),
            pytest.raises(RuntimeError, match="failed to materialise sealed credential"),
        ):
            systemd_creds.seal("pw", cred)

    def test_generic_os_error_translated_to_runtime_error(self, tmp_path: Path) -> None:
        """Any OSError from the subprocess (not just FileNotFoundError) folds to RuntimeError."""
        cred = tmp_path / "v.cred"

        def _run(*_a: object, **kw: object) -> MagicMock:
            argv = _a[0] if _a else kw.get("args")
            if argv and "--version" in argv:
                return _version_output(259)
            raise OSError("too many open files")

        with (
            patch("subprocess.run", side_effect=_run),
            pytest.raises(RuntimeError, match="too many open files"),
        ):
            systemd_creds.seal("pw", cred)


class TestUnseal:
    """``unseal`` returns ``None`` on every failure mode — the resolver falls through."""

    def test_missing_file_returns_none(self, tmp_path: Path) -> None:
        assert systemd_creds.unseal(tmp_path / "nope.cred") is None

    def test_successful_decrypt_returns_passphrase(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        cred = tmp_path / "v.cred"
        cred.write_bytes(b"sealed-blob")
        monkeypatch.setattr("shutil.which", lambda _name: _SYSTEMD_CREDS_EXE)
        result = MagicMock(returncode=0, stdout="my-passphrase\n")
        with patch("subprocess.run", return_value=result) as run:
            assert systemd_creds.unseal(cred) == "my-passphrase"
        argv = run.call_args.args[0]
        # Absolute path pins the binary (PATH-hijack defense); ``--user``
        # engages the non-root Varlink delegation; ``--name=`` matches the
        # encrypt side so systemd refuses cross-purpose reuse.
        assert argv[0] == _SYSTEMD_CREDS_EXE
        assert argv[1] == "decrypt"
        assert "--user" in argv
        assert "--name=terok-sandbox.vault-passphrase" in argv

    def test_returns_none_when_binary_absent(self, tmp_path: Path) -> None:
        """Without the binary we can't decrypt — fall through cleanly, no subprocess spawn."""
        cred = tmp_path / "v.cred"
        cred.write_bytes(b"sealed-blob")
        with (
            patch("shutil.which", return_value=None),
            patch("subprocess.run") as run,
        ):
            assert systemd_creds.unseal(cred) is None
            run.assert_not_called()

    def test_empty_decrypt_output_returns_none(self, tmp_path: Path) -> None:
        """Empty plaintext is SQLCipher's no-encryption sentinel — collapse to None."""
        cred = tmp_path / "v.cred"
        cred.write_bytes(b"sealed-blob")
        with patch("subprocess.run", return_value=MagicMock(returncode=0, stdout="\n")):
            assert systemd_creds.unseal(cred) is None

    def test_decrypt_failure_returns_none(self, tmp_path: Path) -> None:
        """systemd-creds failing (wrong machine, TPM state changed, etc.) → fall through."""
        cred = tmp_path / "v.cred"
        cred.write_bytes(b"sealed-blob")
        err = subprocess.CalledProcessError(
            returncode=1, cmd=["systemd-creds"], stderr="Failed to decrypt"
        )
        with patch("subprocess.run", side_effect=err):
            assert systemd_creds.unseal(cred) is None

    def test_binary_missing_returns_none(self, tmp_path: Path) -> None:
        cred = tmp_path / "v.cred"
        cred.write_bytes(b"sealed-blob")
        with patch("subprocess.run", side_effect=FileNotFoundError):
            assert systemd_creds.unseal(cred) is None

    def test_timeout_returns_none(self, tmp_path: Path) -> None:
        cred = tmp_path / "v.cred"
        cred.write_bytes(b"sealed-blob")
        with patch(
            "subprocess.run",
            side_effect=subprocess.TimeoutExpired(cmd="systemd-creds", timeout=10),
        ):
            assert systemd_creds.unseal(cred) is None


# ── Test helpers ────────────────────────────────────────────────────

_SYSTEMD_CREDS_EXE = "/usr/bin/systemd-creds"
"""Stable absolute path returned by ``shutil.which`` in the
``_have_systemd_creds`` fixture; tests assert on this so subprocess
calls visibly pin the binary instead of resolving via PATH."""


def _version_output(version: int) -> MagicMock:
    """Return a ``subprocess.run`` result that mimics ``systemd-creds --version``."""
    return MagicMock(returncode=0, stdout=f"systemd {version} ({version}.5-1.fc44)\n+PAM +AUDIT\n")


def _mock_seal_subprocess(sealed_blob: bytes = b"sealed-blob"):
    """Return a ``subprocess.run`` side_effect for the [version, encrypt] pair.

    ``seal()`` runs ``--version`` first (via ``is_available()``) then
    ``encrypt`` (which captures the sealed blob from stdout).  Tests
    that don't override the encrypt output get the default
    ``b"sealed-blob"`` value.
    """
    calls = {"n": 0}

    def _run(*_a: object, **_kw: object) -> MagicMock:
        calls["n"] += 1
        if calls["n"] == 1:
            return _version_output(259)
        return MagicMock(returncode=0, stdout=sealed_blob)

    return _run


def _seal_argv(run_mock: MagicMock) -> list[str]:
    """Return the argv of the *encrypt* call (second subprocess.run invocation)."""
    return run_mock.call_args_list[1].args[0]


@pytest.fixture()
def _have_systemd_creds(monkeypatch: pytest.MonkeyPatch) -> None:
    """Make ``shutil.which`` report systemd-creds present for the duration of the test.

    ``is_available()`` short-circuits to False when the binary isn't on
    PATH, which it never is in the test container.  Tests that exercise
    seal / unseal need that gate open so they can drive
    ``subprocess.run`` directly.
    """
    monkeypatch.setattr("shutil.which", lambda _name: _SYSTEMD_CREDS_EXE)
