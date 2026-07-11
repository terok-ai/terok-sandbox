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
    acknowledge as _ack_marker,
    acknowledged,
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
        """The cleartext lands on /dev/tty — never stdout, never stderr."""
        cfg = _cfg(tmp_path)
        tty = _patch_tty(monkeypatch)
        _handle_vault_passphrase_reveal(cfg=cfg)
        assert _PASSPHRASE in tty.value
        # Neither stdout nor stderr (both captured by pytest) may carry
        # the cleartext — a redirected ``terok-sandbox vault passphrase
        # reveal 2>/dev/null > out`` must not leak the recovery key.
        captured = capsys.readouterr()
        assert _PASSPHRASE not in captured.out
        assert _PASSPHRASE not in captured.err

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
        """Pipe payload is exactly ``passphrase\\n``; stderr UX never echoes it."""
        _handle_vault_passphrase_reveal(cfg=_cfg(tmp_path), allow_redirect=True)
        captured = capsys.readouterr()
        # stdout = passphrase + single newline; nothing else.
        assert captured.out == _PASSPHRASE + "\n"
        # The banner + recovery reminder + acknowledgement messaging
        # live on stderr so ``| pass insert -e`` gets clean input.
        assert "Recovery key" in captured.err
        # Audit finding #1: stderr must not contain the cleartext — CI
        # log capture would otherwise persist the recovery key.
        assert _PASSPHRASE not in captured.err

    def test_already_acked_status_routes_to_stderr_under_redirect(
        self,
        tmp_path: Path,
        capsys: pytest.CaptureFixture[str],
    ) -> None:
        """``allow_redirect=True`` keeps even post-reveal status off stdout."""
        cfg = _cfg(tmp_path)
        _ack_marker(cfg.vault_recovery_marker_file)
        _handle_vault_passphrase_reveal(cfg=cfg, allow_redirect=True)
        captured = capsys.readouterr()
        assert captured.out == _PASSPHRASE + "\n"
        assert "already marked as saved" in captured.err

    def test_allow_redirect_banner_does_not_leak_passphrase(
        self,
        tmp_path: Path,
        capsys: pytest.CaptureFixture[str],
    ) -> None:
        """Audit finding #1: stderr banner must not echo the passphrase.

        Many CI / log setups persist stderr by default — embedding the
        cleartext there would silently leak the recovery key even when
        the operator carefully redirects stdout into a secret manager.
        """
        _handle_vault_passphrase_reveal(cfg=_cfg(tmp_path), allow_redirect=True)
        captured = capsys.readouterr()
        assert captured.out == _PASSPHRASE + "\n"
        assert _PASSPHRASE not in captured.err
        # The stderr banner still surfaces the source label + the
        # save-it-off-host reminder, just not the cleartext.
        assert "Recovery key" in captured.err


class TestRevealAckPrompt:
    """The reveal verb offers an inline "mark as saved" affordance."""

    def test_typing_saved_writes_marker(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """``SAVED`` lands the (zero-byte) marker — same shape as the standalone ack."""
        cfg = _cfg(tmp_path)
        _patch_tty(monkeypatch, responses=("SAVED",))
        _handle_vault_passphrase_reveal(cfg=cfg)
        assert acknowledged(cfg.vault_recovery_marker_file)

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
        """When the marker is present, no second confirmation is needed."""
        cfg = _cfg(tmp_path)
        # Pre-seed the marker; reveal should short-circuit the ack prompt.
        _ack_marker(cfg.vault_recovery_marker_file)
        _patch_tty(monkeypatch)
        _handle_vault_passphrase_reveal(cfg=cfg)
        out = capsys.readouterr().out
        assert "already marked as saved" in out


class TestAcknowledge:
    """``vault passphrase acknowledge`` — silent ack from the TUI / CI."""

    def test_writes_marker(self, tmp_path: Path) -> None:
        """Acknowledging lands the (zero-byte) sidecar marker."""
        cfg = _cfg(tmp_path)
        _handle_vault_passphrase_acknowledge(cfg=cfg)
        assert acknowledged(cfg.vault_recovery_marker_file)

    def test_idempotent(self, tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
        """Re-running on an already-acked marker is a no-op with a hint."""
        cfg = _cfg(tmp_path)
        _handle_vault_passphrase_acknowledge(cfg=cfg)
        capsys.readouterr()  # discard first call's output
        _handle_vault_passphrase_acknowledge(cfg=cfg)
        assert "already marked as saved" in capsys.readouterr().out

    def test_locked_vault_still_acks(self, tmp_path: Path) -> None:
        """Marker is passphrase-independent — ack works even on a locked vault."""
        cfg = _cfg(tmp_path, passphrase=None)
        _handle_vault_passphrase_acknowledge(cfg=cfg)
        assert acknowledged(cfg.vault_recovery_marker_file)


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


class TestDefaultConfigConstruction:
    """``cfg=None`` branches in the reveal + acknowledge handlers."""

    def test_reveal_constructs_default_config(
        self,
        tmp_path: Path,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        """Passing ``cfg=None`` runs the lazy ``SandboxConfig()`` construction."""
        sentinel = _cfg(tmp_path)
        monkeypatch.setattr("terok_sandbox.config.SandboxConfig", lambda: sentinel)
        _handle_vault_passphrase_reveal(allow_redirect=True)
        # Marker still missing (no SAVED typed), but the handler had to
        # construct the cfg itself to even get this far.
        assert not sentinel.vault_recovery_marker_file.exists()

    def test_acknowledge_constructs_default_config(
        self,
        tmp_path: Path,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        """``cfg=None`` ack flow lands the marker against the default config."""
        sentinel = _cfg(tmp_path)
        monkeypatch.setattr("terok_sandbox.config.SandboxConfig", lambda: sentinel)
        _handle_vault_passphrase_acknowledge()
        assert acknowledged(sentinel.vault_recovery_marker_file)


class TestRevealStdoutIsattyHint:
    """``--allow-redirect=False`` + interactive stdout surfaces the routing hint."""

    def test_isatty_stdout_prints_routing_hint(
        self,
        tmp_path: Path,
        monkeypatch: pytest.MonkeyPatch,
        capsys: pytest.CaptureFixture[str],
    ) -> None:
        """The "passphrase routed to /dev/tty" message fires only on an interactive stdout."""
        cfg = _cfg(tmp_path)
        _patch_tty(monkeypatch)
        monkeypatch.setattr("sys.stdout.isatty", lambda: True)
        _handle_vault_passphrase_reveal(cfg=cfg)
        out = capsys.readouterr().out
        assert "passphrase routed to /dev/tty" in out
        assert "--allow-redirect" in out


class TestAcknowledgeDecoupling:
    """The ack verb is independent of the passphrase resolver.

    Pinned here because the security-audit fix decoupled the marker
    from any passphrase-derived bytes — a regression that tried to
    re-introduce the coupling (e.g. recompute a fingerprint, check
    DB openability) would break these tests.
    """

    def test_does_not_touch_resolver(
        self,
        tmp_path: Path,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        """Even if the resolver would raise, ack still lands the marker."""
        from terok_sandbox.vault.store.encryption import WrongPassphraseError

        cfg = _cfg(tmp_path)

        def _trap(*_a: object, **_kw: object) -> object:
            raise WrongPassphraseError("resolver must not be called")

        # If the handler regresses to calling the resolver, this will
        # raise WrongPassphraseError instead of completing the ack.
        monkeypatch.setattr(type(cfg), "resolve_passphrase", lambda self, **_kw: _trap())
        monkeypatch.setattr(
            type(cfg), "resolve_passphrase_with_source", lambda self, **_kw: _trap()
        )
        _handle_vault_passphrase_acknowledge(cfg=cfg)
        assert acknowledged(cfg.vault_recovery_marker_file)
