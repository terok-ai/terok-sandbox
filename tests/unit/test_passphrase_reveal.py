# SPDX-FileCopyrightText: 2026 Jiri Vyskocil
# SPDX-License-Identifier: Apache-2.0

"""Tests for the ``vault passphrase reveal`` / ``acknowledge`` CLI verbs.

The reveal verb resolves the current passphrase via the chain and
prints it (default to ``/dev/tty``, opt-in stdout via
``--allow-redirect``).  The acknowledge verb is the silent counterpart
the TUI uses after showing the passphrase in its own modal.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from terok_sandbox import SandboxConfig

# These handlers stay internal to ``terok_sandbox.commands.vault`` —
# importing them via the package facade would add stable-API contract
# weight to symbols that the package treats as private.  Tests reach
# into the implementing module directly.
from terok_sandbox.commands.vault import (
    _handle_vault_passphrase_acknowledge,
    _handle_vault_passphrase_reveal,
)
from terok_sandbox.vault.store.recovery import (
    fingerprint,
    is_acknowledged,
)

_PASSPHRASE = "correct-horse-battery-staple"


def _cfg(tmp_path: Path, *, passphrase: str | None = _PASSPHRASE) -> SandboxConfig:
    """Sandbox config with the config-tier passphrase wired in (no keyring)."""
    return SandboxConfig(
        state_dir=tmp_path / "state",
        runtime_dir=tmp_path / "rt",
        config_dir=tmp_path / "cfg",
        vault_dir=tmp_path / "vault",
        services_mode="socket",
        credentials_passphrase=passphrase,
        credentials_use_keyring=False,
    )


class _TtyChannel:
    """Captures both directions of ``/dev/tty`` I/O for the reveal flow."""

    def __init__(self, response_lines: tuple[str, ...] = ()) -> None:
        self.value = ""
        self._responses = list(response_lines)

    def __enter__(self) -> _TtyChannel:
        return self

    def __exit__(self, *_exc: object) -> None:
        return None

    def write(self, s: str) -> int:
        self.value += s
        return len(s)

    def flush(self) -> None:
        return None

    def readline(self) -> str:
        if not self._responses:
            return ""
        return self._responses.pop(0) + "\n"


def _patch_tty(
    monkeypatch: pytest.MonkeyPatch,
    *,
    responses: tuple[str, ...] = (),
) -> _TtyChannel:
    """Redirect ``Path('/dev/tty').open(...)`` into an in-memory channel."""
    from pathlib import Path as _Path

    channel = _TtyChannel(response_lines=responses)
    real_open = _Path.open

    def _selective_open(self, *args, **kwargs):
        if str(self) == "/dev/tty":
            return channel
        return real_open(self, *args, **kwargs)

    monkeypatch.setattr(_Path, "open", _selective_open)
    return channel


class TestRevealDefault:
    """Default path: render to /dev/tty, never stdout."""

    def test_writes_passphrase_to_tty(
        self,
        tmp_path: Path,
        monkeypatch: pytest.MonkeyPatch,
        capsys: pytest.CaptureFixture[str],
    ) -> None:
        """The cleartext lands on /dev/tty even when stdout is captured."""
        cfg = _cfg(tmp_path)
        tty = _patch_tty(monkeypatch)
        _handle_vault_passphrase_reveal(cfg=cfg)
        assert _PASSPHRASE in tty.value
        # Stdout (captured by pytest) must not carry the cleartext.
        assert _PASSPHRASE not in capsys.readouterr().out

    def test_locked_vault_exits(
        self,
        tmp_path: Path,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        """No resolvable passphrase → SystemExit with `vault unlock` hint."""
        _patch_tty(monkeypatch)
        with pytest.raises(SystemExit, match="vault unlock"):
            _handle_vault_passphrase_reveal(cfg=_cfg(tmp_path, passphrase=None))


class TestRevealAllowRedirect:
    """``--allow-redirect`` flips output to stdout for explicit piping."""

    def test_writes_to_stdout(
        self,
        tmp_path: Path,
        capsys: pytest.CaptureFixture[str],
    ) -> None:
        """The cleartext lands on stdout — no /dev/tty path needed."""
        _handle_vault_passphrase_reveal(cfg=_cfg(tmp_path), allow_redirect=True)
        assert _PASSPHRASE in capsys.readouterr().out

    def test_stdout_carries_only_the_passphrase(
        self,
        tmp_path: Path,
        capsys: pytest.CaptureFixture[str],
    ) -> None:
        """Pipe payload must be exactly ``passphrase\\n`` — banner goes to stderr."""
        _handle_vault_passphrase_reveal(cfg=_cfg(tmp_path), allow_redirect=True)
        captured = capsys.readouterr()
        # stdout = passphrase + single newline; nothing else.
        assert captured.out == _PASSPHRASE + "\n"
        # The banner + recovery reminder + acknowledgement messaging
        # live on stderr so ``| pass insert -e`` gets clean input.
        assert "Recovery key" in captured.err
        assert _PASSPHRASE in captured.err  # also echoed in stderr banner

    def test_already_acked_status_routes_to_stderr_under_redirect(
        self,
        tmp_path: Path,
        capsys: pytest.CaptureFixture[str],
    ) -> None:
        """``allow_redirect=True`` keeps even post-reveal status off stdout."""
        cfg = _cfg(tmp_path)
        cfg.vault_recovery_marker_file.parent.mkdir(parents=True, exist_ok=True)
        cfg.vault_recovery_marker_file.write_text(fingerprint(_PASSPHRASE) + "\n")
        _handle_vault_passphrase_reveal(cfg=cfg, allow_redirect=True)
        captured = capsys.readouterr()
        assert captured.out == _PASSPHRASE + "\n"
        assert "already marked as saved" in captured.err


class TestRevealAckPrompt:
    """The reveal verb offers an inline "mark as saved" affordance."""

    def test_typing_saved_writes_marker(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """``SAVED`` lands the fingerprint marker — same shape as the standalone ack."""
        cfg = _cfg(tmp_path)
        _patch_tty(monkeypatch, responses=("SAVED",))
        _handle_vault_passphrase_reveal(cfg=cfg)
        assert is_acknowledged(cfg.vault_recovery_marker_file, _PASSPHRASE)

    def test_empty_response_does_not_write_marker(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Anything other than ``SAVED`` leaves the unconfirmed warning on."""
        cfg = _cfg(tmp_path)
        _patch_tty(monkeypatch)  # No responses → readline returns ""
        _handle_vault_passphrase_reveal(cfg=cfg)
        assert not cfg.vault_recovery_marker_file.exists()

    def test_already_acked_skips_prompt(
        self,
        tmp_path: Path,
        monkeypatch: pytest.MonkeyPatch,
        capsys: pytest.CaptureFixture[str],
    ) -> None:
        """When the marker already matches, no second confirmation is needed."""
        cfg = _cfg(tmp_path)
        # Pre-seed the marker; reveal should short-circuit the ack prompt.
        cfg.vault_recovery_marker_file.parent.mkdir(parents=True, exist_ok=True)
        cfg.vault_recovery_marker_file.write_text(fingerprint(_PASSPHRASE) + "\n")
        _patch_tty(monkeypatch)
        _handle_vault_passphrase_reveal(cfg=cfg)
        out = capsys.readouterr().out
        assert "already marked as saved" in out


class TestAcknowledge:
    """``vault passphrase acknowledge`` — silent ack from the TUI / CI."""

    def test_writes_marker(self, tmp_path: Path) -> None:
        """Acknowledging an unlocked vault lands the fingerprint sidecar."""
        cfg = _cfg(tmp_path)
        _handle_vault_passphrase_acknowledge(cfg=cfg)
        assert is_acknowledged(cfg.vault_recovery_marker_file, _PASSPHRASE)

    def test_idempotent(self, tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
        """Re-running on an already-acked marker is a no-op with a hint."""
        cfg = _cfg(tmp_path)
        _handle_vault_passphrase_acknowledge(cfg=cfg)
        capsys.readouterr()  # discard first call's output
        _handle_vault_passphrase_acknowledge(cfg=cfg)
        assert "already marked as saved" in capsys.readouterr().out

    def test_locked_vault_exits(self, tmp_path: Path) -> None:
        """Locked vault → SystemExit (matching the rest of the verbs)."""
        with pytest.raises(SystemExit, match="vault unlock"):
            _handle_vault_passphrase_acknowledge(cfg=_cfg(tmp_path, passphrase=None))


class TestRevealEdgeCases:
    """Boundary conditions: wrong passphrase, no-TTY ack, non-SAVED response."""

    def test_wrong_passphrase_translates_to_systemexit(
        self,
        tmp_path: Path,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        """A ``WrongPassphraseError`` from the resolver lands as a clean exit."""
        from terok_sandbox.vault.store.encryption import WrongPassphraseError

        cfg = _cfg(tmp_path)

        def _raise(*_a: object, **_kw: object) -> object:
            raise WrongPassphraseError("DB rejected the passphrase")

        monkeypatch.setattr(
            type(cfg), "resolve_passphrase_with_source", lambda self, **_kw: _raise()
        )
        _patch_tty(monkeypatch)
        with pytest.raises(SystemExit, match="cannot reveal passphrase"):
            _handle_vault_passphrase_reveal(cfg=cfg)

    def test_no_tty_skips_ack_prompt_with_hint(
        self,
        tmp_path: Path,
        monkeypatch: pytest.MonkeyPatch,
        capsys: pytest.CaptureFixture[str],
    ) -> None:
        """When /dev/tty is unreachable, the ack prompt is skipped with an
        actionable hint pointing at ``vault passphrase acknowledge``.
        """
        from pathlib import Path as _Path

        cfg = _cfg(tmp_path)
        real_open = _Path.open

        def _no_tty_open(self, *args, **kwargs):
            if str(self) == "/dev/tty":
                raise OSError("no /dev/tty in test")
            return real_open(self, *args, **kwargs)

        # No /dev/tty available → reveal must still succeed under
        # --allow-redirect (cleartext on stdout) and emit the hint.
        monkeypatch.setattr(_Path, "open", _no_tty_open)
        _handle_vault_passphrase_reveal(cfg=cfg, allow_redirect=True)
        captured = capsys.readouterr()
        assert "vault passphrase acknowledge" in captured.err

    def test_non_saved_response_does_not_acknowledge(
        self,
        tmp_path: Path,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        """Anything except ``SAVED`` (e.g. ``no``, ``nope``) leaves the marker absent."""
        cfg = _cfg(tmp_path)
        _patch_tty(monkeypatch, responses=("nope",))
        _handle_vault_passphrase_reveal(cfg=cfg)
        assert not cfg.vault_recovery_marker_file.exists()


class TestAcknowledgeEdgeCases:
    """Edge cases for the silent ack verb."""

    def test_wrong_passphrase_translates_to_systemexit(
        self,
        tmp_path: Path,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        """``WrongPassphraseError`` surfaces as ``cannot acknowledge: …`` exit."""
        from terok_sandbox.vault.store.encryption import WrongPassphraseError

        cfg = _cfg(tmp_path)

        def _raise(*_a: object, **_kw: object) -> object:
            raise WrongPassphraseError("DB rejected the passphrase")

        monkeypatch.setattr(type(cfg), "resolve_passphrase", lambda self, **_kw: _raise())
        with pytest.raises(SystemExit, match="cannot acknowledge"):
            _handle_vault_passphrase_acknowledge(cfg=cfg)
