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


def _version_output(version: int) -> MagicMock:
    """Return a ``subprocess.run`` result that mimics ``systemd-creds --version``."""
    return MagicMock(returncode=0, stdout=f"systemd {version} ({version}.5-1.fc44)\n+PAM +AUDIT\n")


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


def _mock_seal_subprocess(cred: Path):
    """Return a ``subprocess.run`` side_effect for the [version, encrypt] pair.

    ``seal()`` calls ``is_available()`` first (one subprocess for the
    ``--version`` probe) then ``systemd-creds encrypt`` (which is the
    call the test wants to inspect).  This helper materialises *cred*
    on the encrypt call so the post-write ``chmod`` finds the file.
    """
    calls = {"n": 0}

    def _run(*_a: object, **_kw: object) -> MagicMock:
        calls["n"] += 1
        if calls["n"] == 1:
            return _version_output(259)
        cred.write_bytes(b"sealed-blob")
        return MagicMock(returncode=0)

    return _run


@pytest.fixture()
def _have_systemd_creds(monkeypatch: pytest.MonkeyPatch) -> None:
    """Make ``shutil.which`` report systemd-creds present for the duration of the test.

    ``is_available()`` short-circuits to False when the binary isn't on
    PATH, which it never is in the test container.  Tests that exercise
    seal / unseal need that gate open so they can drive
    ``subprocess.run`` directly.
    """
    monkeypatch.setattr("shutil.which", lambda _name: "/usr/bin/systemd-creds")


def _seal_argv(run_mock: MagicMock) -> list[str]:
    """Return the argv of the *encrypt* call (second subprocess.run invocation)."""
    return run_mock.call_args_list[1].args[0]


@pytest.mark.usefixtures("_have_systemd_creds")
class TestSeal:
    """``seal`` shells out to ``systemd-creds encrypt``."""

    def test_user_mode_always_set(self, tmp_path: Path) -> None:
        """Every seal goes through ``--user`` so non-root Varlink delegation engages."""
        cred = tmp_path / "v.cred"
        with patch("subprocess.run", side_effect=_mock_seal_subprocess(cred)) as run:
            systemd_creds.seal("pw", cred)
        assert "--user" in _seal_argv(run)

    def test_tpm_key_argv(self, tmp_path: Path) -> None:
        cred = tmp_path / "v.cred"
        with patch("subprocess.run", side_effect=_mock_seal_subprocess(cred)) as run:
            systemd_creds.seal("pw", cred, tpm=True)
        argv = _seal_argv(run)
        assert argv[:2] == ["systemd-creds", "encrypt"]
        assert "--with-key=tpm2" in argv

    def test_host_key_argv(self, tmp_path: Path) -> None:
        cred = tmp_path / "v.cred"
        with patch("subprocess.run", side_effect=_mock_seal_subprocess(cred)) as run:
            systemd_creds.seal("pw", cred, tpm=False)
        assert "--with-key=host" in _seal_argv(run)

    def test_passphrase_goes_to_stdin(self, tmp_path: Path) -> None:
        """The passphrase is piped to systemd-creds, not embedded in argv."""
        cred = tmp_path / "v.cred"
        with patch("subprocess.run", side_effect=_mock_seal_subprocess(cred)) as run:
            systemd_creds.seal("secret", cred)
        encrypt_call = run.call_args_list[1]
        assert encrypt_call.kwargs["input"] == "secret"
        # And explicitly not in argv:
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

    def test_credential_path_locked_to_0o600(self, tmp_path: Path) -> None:
        cred = tmp_path / "v.cred"
        with patch("subprocess.run", side_effect=_mock_seal_subprocess(cred)):
            systemd_creds.seal("pw", cred)
        assert oct(cred.stat().st_mode & 0o777) == oct(0o600)

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
                returncode=1, cmd=["systemd-creds"], stderr="TPM unavailable"
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

    def test_chmod_failure_translated_to_runtime_error(self, tmp_path: Path) -> None:
        """An OSError from the post-write chmod stays inside the documented contract."""
        cred = tmp_path / "v.cred"
        with (
            patch("subprocess.run", side_effect=_mock_seal_subprocess(cred)),
            patch("pathlib.Path.chmod", side_effect=OSError("read-only fs")),
            pytest.raises(RuntimeError, match="failed to secure sealed credential"),
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

    def test_successful_decrypt_returns_passphrase(self, tmp_path: Path) -> None:
        cred = tmp_path / "v.cred"
        cred.write_bytes(b"sealed-blob")
        result = MagicMock(returncode=0, stdout="my-passphrase\n")
        with patch("subprocess.run", return_value=result) as run:
            assert systemd_creds.unseal(cred) == "my-passphrase"
        argv = run.call_args.args[0]
        # ``--user`` engages the non-root Varlink delegation path
        assert "--user" in argv
        assert argv[:2] == ["systemd-creds", "decrypt"]

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
