# SPDX-FileCopyrightText: 2025 Jiri Vyskocil
# SPDX-License-Identifier: Apache-2.0

"""Tests for ``GitGate._ssh_env`` — three-branch policy + ephemeral signer.

The gate prefers a per-instance ephemeral SSH signer (bound the first
time ``_ssh_env`` is called and torn down with the gate); explicit
opt-in (``use_personal_ssh=True``) falls through to the user's ambient
SSH; HTTPS upstreams skip SSH entirely.  When the DB has no keys
assigned to the scope, [`GateAuthNotConfigured`][terok_sandbox.GateAuthNotConfigured]
is raised straight from ``_EphemeralSigner.start``.

The vault branch additionally pins OpenSSH so personal keys can never
leak in and no interactive prompt can ever leak out — asserted below as
``IdentityAgent=<sock>`` + ``-F /dev/null`` + ``BatchMode=yes``.
"""

from __future__ import annotations

import socket
from contextlib import contextmanager
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from terok_sandbox.gate.mirror import GateAuthNotConfigured, GitGate


@contextmanager
def _stub_credential_db(scopes_with_keys: list[str] | None):
    """Stub ``SandboxConfig.open_credential_db`` to return a fake DB.

    When *scopes_with_keys* is ``None`` the helper raises from
    ``open_credential_db`` (simulating ``NoPassphraseError`` / locked
    vault); otherwise ``list_ssh_keys_for_scope`` returns a non-empty
    list for any scope in the set and an empty list otherwise.
    """
    fake_db = MagicMock()
    fake_db.list_ssh_keys_for_scope.side_effect = lambda scope: (
        ["fake-key"] if scopes_with_keys and scope in scopes_with_keys else []
    )
    if scopes_with_keys is None:
        with patch(
            "terok_sandbox.config.SandboxConfig.open_credential_db",
            side_effect=RuntimeError("locked"),
        ):
            yield fake_db
    else:
        with patch(
            "terok_sandbox.config.SandboxConfig.open_credential_db",
            return_value=fake_db,
        ):
            yield fake_db


def _bind_real_unix_socket(path: Path) -> socket.socket:
    """Bind a real Unix domain socket at *path* — used as the signer stand-in."""
    s = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
    s.bind(str(path))
    return s


def _make_gate(scope: str = "proj", **overrides: object) -> GitGate:
    """Construct a ``GitGate`` with an SSH upstream and the given overrides."""
    kwargs: dict[str, object] = {
        "scope": scope,
        "gate_path": Path("/tmp/test-gate.git"),  # noqa: S108 — never written
        "upstream_url": "git@github.com:example/repo.git",
    }
    kwargs.update(overrides)
    return GitGate(**kwargs)  # type: ignore[arg-type]


class TestVaultPath:
    """Verify the default (ephemeral-signer) branch."""

    def test_starts_signer_and_pins_ssh_options(self, tmp_path: Path) -> None:
        """When the DB has keys, ``_ssh_env`` binds a signer and pins OpenSSH at it."""
        sock_path = tmp_path / "agent.sock"
        bound_sock = _bind_real_unix_socket(sock_path)

        from terok_sandbox.gate.mirror import _EphemeralSigner

        fake_signer = _EphemeralSigner.__new__(_EphemeralSigner)
        object.__setattr__(fake_signer, "socket_path", sock_path)
        object.__setattr__(fake_signer, "_tmpdir", None)
        object.__setattr__(fake_signer, "_thread", None)
        object.__setattr__(fake_signer, "_loop", None)
        object.__setattr__(fake_signer, "_server", None)

        try:
            with (
                _stub_credential_db(scopes_with_keys=["proj"]),
                patch.object(_EphemeralSigner, "start", return_value=fake_signer) as mock_start,
            ):
                gate = _make_gate()
                env = gate._ssh_env()
                # Second call must reuse the same signer — no re-bind churn.
                env2 = gate._ssh_env()
        finally:
            bound_sock.close()

        assert mock_start.call_count == 1
        assert env["SSH_AUTH_SOCK"] == str(sock_path)
        assert env2["SSH_AUTH_SOCK"] == str(sock_path)
        cmd = env["GIT_SSH_COMMAND"]
        # Ephemeral signer is the pinned identity source.
        assert f"IdentityAgent={sock_path}" in cmd
        assert "IdentityFile=none" in cmd
        # User's ~/.ssh/config must not influence auth.
        assert "-F /dev/null" in cmd
        # No passphrase / host-key / password prompts may leak to the caller.
        assert "BatchMode=yes" in cmd
        assert "StrictHostKeyChecking=no" in cmd
        # Agent must remain consulted — IdentitiesOnly with IdentityFile=none
        # would leave ssh with no identities at all.
        assert "IdentitiesOnly=yes" not in cmd

    def test_raises_when_db_has_no_keys(self) -> None:
        """No DB rows for the scope → ``GateAuthNotConfigured`` immediately."""
        with (
            _stub_credential_db(scopes_with_keys=[]),
            pytest.raises(GateAuthNotConfigured) as excinfo,
        ):
            _make_gate(scope="nowhere")._ssh_env()
        assert "nowhere" in str(excinfo.value)
        assert "ssh-init" in str(excinfo.value)
        assert "use-personal-ssh" in str(excinfo.value)

    def test_raises_when_vault_locked(self) -> None:
        """``open_credential_db`` raises → ``GateAuthNotConfigured`` (fail-soft)."""
        with _stub_credential_db(scopes_with_keys=None), pytest.raises(GateAuthNotConfigured):
            _make_gate()._ssh_env()


class TestPersonalOptIn:
    """Verify ``use_personal_ssh=True`` leaves the env alone."""

    def test_returns_untouched_env(self) -> None:
        """Opt-in: no GIT_SSH_COMMAND injected, SSH_AUTH_SOCK not forced."""
        import os

        env = _make_gate(use_personal_ssh=True)._ssh_env()

        assert env.get("SSH_AUTH_SOCK") == os.environ.get("SSH_AUTH_SOCK")
        assert "GIT_SSH_COMMAND" not in env or env["GIT_SSH_COMMAND"] == os.environ.get(
            "GIT_SSH_COMMAND", ""
        )

    def test_opt_in_does_not_start_signer(self, tmp_path: Path) -> None:
        """Opt-in bypasses the ephemeral signer entirely — no DB read either."""
        from terok_sandbox.gate.mirror import _EphemeralSigner

        with patch.object(_EphemeralSigner, "start") as mock_start:
            _make_gate(use_personal_ssh=True)._ssh_env()

        mock_start.assert_not_called()


class TestHttpsUpstream:
    """HTTPS upstreams skip the SSH machinery entirely."""

    def test_https_upstream_skips_signer(self) -> None:
        """An ``https://`` upstream gets an unmodified env — no signer bound."""
        from terok_sandbox.gate.mirror import _EphemeralSigner

        with patch.object(_EphemeralSigner, "start") as mock_start:
            env = _make_gate(upstream_url="https://github.com/example/repo.git")._ssh_env()

        mock_start.assert_not_called()
        assert "GIT_SSH_COMMAND" not in env


class TestSignerLifecycle:
    """``close()`` and ``__del__`` tear the signer down idempotently."""

    def test_close_stops_signer_once(self) -> None:
        """``close()`` calls ``signer.stop()`` once and clears the handle."""
        from terok_sandbox.gate.mirror import _EphemeralSigner

        fake_signer = _EphemeralSigner.__new__(_EphemeralSigner)
        object.__setattr__(fake_signer, "socket_path", Path("/tmp/x"))  # noqa: S108
        stops: list[int] = []
        object.__setattr__(fake_signer, "stop", lambda: stops.append(1))

        gate = _make_gate()
        gate._signer = fake_signer
        gate.close()
        gate.close()  # idempotent

        assert stops == [1]
        assert gate._signer is None
