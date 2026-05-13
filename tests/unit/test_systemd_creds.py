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

    def test_is_available_when_binary_on_path(self) -> None:
        with patch("shutil.which", return_value="/usr/bin/systemd-creds"):
            assert systemd_creds.is_available() is True

    def test_is_available_when_missing(self) -> None:
        with patch("shutil.which", return_value=None):
            assert systemd_creds.is_available() is False

    def test_has_tpm2_short_circuits_when_binary_absent(self) -> None:
        """No binary → no TPM probe; no subprocess spawned."""
        with (
            patch("shutil.which", return_value=None),
            patch("subprocess.run") as run,
        ):
            assert systemd_creds.has_tpm2() is False
            run.assert_not_called()

    def test_has_tpm2_when_command_succeeds(self) -> None:
        with (
            patch("shutil.which", return_value="/usr/bin/systemd-creds"),
            patch("subprocess.run", return_value=MagicMock(returncode=0)),
        ):
            assert systemd_creds.has_tpm2() is True

    def test_has_tpm2_when_command_fails(self) -> None:
        with (
            patch("shutil.which", return_value="/usr/bin/systemd-creds"),
            patch("subprocess.run", return_value=MagicMock(returncode=1)),
        ):
            assert systemd_creds.has_tpm2() is False

    def test_has_tpm2_swallows_timeout(self) -> None:
        with (
            patch("shutil.which", return_value="/usr/bin/systemd-creds"),
            patch(
                "subprocess.run",
                side_effect=subprocess.TimeoutExpired(cmd="systemd-creds", timeout=5),
            ),
        ):
            assert systemd_creds.has_tpm2() is False


def _mock_seal_subprocess(cred: Path):
    """Return a ``subprocess.run`` side_effect that materialises *cred*.

    Real ``systemd-creds encrypt`` writes the sealed blob to the file;
    the wrapper then ``chmod``s it.  Tests that exercise that
    post-write step need the file to exist.
    """

    def _run(*_a: object, **_kw: object) -> MagicMock:
        cred.write_bytes(b"sealed-blob")
        return MagicMock(returncode=0)

    return _run


class TestSeal:
    """``seal`` shells out to ``systemd-creds encrypt``."""

    def test_tpm_key_argv(self, tmp_path: Path) -> None:
        cred = tmp_path / "v.cred"
        with patch("subprocess.run", side_effect=_mock_seal_subprocess(cred)) as run:
            systemd_creds.seal("pw", cred, tpm=True)
        argv = run.call_args.args[0]
        assert argv[:2] == ["systemd-creds", "encrypt"]
        assert "--with-key=tpm2" in argv

    def test_host_key_argv(self, tmp_path: Path) -> None:
        cred = tmp_path / "v.cred"
        with patch("subprocess.run", side_effect=_mock_seal_subprocess(cred)) as run:
            systemd_creds.seal("pw", cred, tpm=False)
        assert "--with-key=host" in run.call_args.args[0]

    def test_passphrase_goes_to_stdin(self, tmp_path: Path) -> None:
        """The passphrase is piped to systemd-creds, not embedded in argv."""
        cred = tmp_path / "v.cred"
        with patch("subprocess.run", side_effect=_mock_seal_subprocess(cred)) as run:
            systemd_creds.seal("secret", cred)
        assert run.call_args.kwargs["input"] == "secret"
        # And explicitly not in argv:
        assert "secret" not in run.call_args.args[0]

    def test_empty_passphrase_rejected(self, tmp_path: Path) -> None:
        with pytest.raises(ValueError, match="empty"):
            systemd_creds.seal("", tmp_path / "v.cred")

    def test_credential_path_locked_to_0o600(self, tmp_path: Path) -> None:
        cred = tmp_path / "v.cred"
        with patch("subprocess.run", side_effect=_mock_seal_subprocess(cred)):
            systemd_creds.seal("pw", cred)
        assert oct(cred.stat().st_mode & 0o777) == oct(0o600)

    def test_binary_missing_raises_runtime_error(self, tmp_path: Path) -> None:
        with (
            patch("subprocess.run", side_effect=FileNotFoundError("systemd-creds")),
            pytest.raises(RuntimeError, match="not found on PATH"),
        ):
            systemd_creds.seal("pw", tmp_path / "v.cred")

    def test_called_process_error_surfaces_stderr(self, tmp_path: Path) -> None:
        err = subprocess.CalledProcessError(
            returncode=1, cmd=["systemd-creds"], stderr="TPM unavailable"
        )
        with (
            patch("subprocess.run", side_effect=err),
            pytest.raises(RuntimeError, match="TPM unavailable"),
        ):
            systemd_creds.seal("pw", tmp_path / "v.cred")


class TestUnseal:
    """``unseal`` returns ``None`` on every failure mode — the resolver falls through."""

    def test_missing_file_returns_none(self, tmp_path: Path) -> None:
        assert systemd_creds.unseal(tmp_path / "nope.cred") is None

    def test_successful_decrypt_returns_passphrase(self, tmp_path: Path) -> None:
        cred = tmp_path / "v.cred"
        cred.write_bytes(b"sealed-blob")
        result = MagicMock(returncode=0, stdout="my-passphrase\n")
        with patch("subprocess.run", return_value=result):
            assert systemd_creds.unseal(cred) == "my-passphrase"

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
