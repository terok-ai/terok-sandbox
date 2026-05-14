# SPDX-FileCopyrightText: 2026 Jiri Vyskocil
# SPDX-License-Identifier: Apache-2.0

"""Vault-daemon CLI verbs — daemon lifecycle plus the ``vault passphrase`` subverbs.

The unlock/lock pair drives the session-tier slot of the SQLCipher
passphrase resolution chain: ``unlock`` lands a passphrase on the
session-unlock tmpfs file; ``lock`` removes it.  Everything else lives
under ``vault passphrase``:

- ``vault passphrase seal`` promotes the current passphrase into a
  machine-bound ``systemd-creds`` credential.
- ``vault passphrase to-keyring`` moves it from whichever tier holds it
  now into the OS keyring (the recommended upgrade path off the
  session-file / plaintext-config tiers).
- ``vault passphrase destroy`` clears every persistent tier so the
  vault becomes irrecoverable without an external copy of the
  passphrase.
"""

from __future__ import annotations

import sys
from pathlib import Path
from typing import TYPE_CHECKING

from .._util import sanitize_tty
from .._yaml import update_section as _yaml_update_section
from ..config import SandboxConfig
from ._types import ArgDef, CommandDef

if TYPE_CHECKING:
    from ..vault.store.systemd_creds import KeyMode


def _handle_vault_start() -> None:
    """Start the vault daemon."""
    from ..vault.daemon.lifecycle import VaultManager

    mgr = VaultManager()
    if mgr.get_status().running:
        print("Vault is already running.")
        return
    mgr.start_daemon()
    print("Vault started.")


def _handle_vault_stop() -> None:
    """Stop the vault daemon."""
    from ..vault.daemon.lifecycle import VaultManager

    mgr = VaultManager()
    if not mgr.is_daemon_running():
        print("Vault is not running.")
        return
    mgr.stop_daemon()
    print("Vault stopped.")


def _handle_vault_status() -> None:
    """Show vault status."""
    from ..vault.daemon.lifecycle import VaultManager

    status = VaultManager().get_status()
    state = "running" if status.running else "stopped"
    print(f"Status: {state}")
    print(f"Socket: {sanitize_tty(str(status.socket_path))}")
    print(f"DB:     {sanitize_tty(str(status.db_path))}")
    print(
        f"Routes: {sanitize_tty(str(status.routes_path))} ({status.routes_configured} configured)"
    )
    if status.credentials_stored:
        print(f"Credentials: {', '.join(sanitize_tty(c) for c in status.credentials_stored)}")
    else:
        print("Credentials: none stored")
    if status.plaintext_passphrase_path is not None:
        _print_plaintext_passphrase_warning(status.plaintext_passphrase_path)


def _print_plaintext_passphrase_warning(path: Path) -> None:
    """Stderr WARNING that the vault passphrase lives in plaintext on disk.

    Fires whenever ``credentials.passphrase`` is configured, regardless
    of which tier actually unlocked this call — the file is a passive
    re-unlock vector and operators deserve to know it's there.
    """
    use_color = sys.stderr.isatty()
    red = "\033[1;31m" if use_color else ""
    reset = "\033[0m" if use_color else ""
    print(
        f"{red}WARNING: vault passphrase stored in plaintext at {sanitize_tty(str(path))}{reset}\n"
        f"{red}         accept on-disk plaintext as your trust boundary,"
        f" or migrate to keyring/systemd-creds.{reset}",
        file=sys.stderr,
    )


def _handle_vault_install() -> None:
    """Install and start systemd socket activation for the vault."""
    from ..vault.daemon.lifecycle import VaultManager

    mgr = VaultManager()
    if not mgr.is_systemd_available():
        print("Error: systemd user services are not available on this host.")
        raise SystemExit(1)
    mgr.install_systemd_units()
    print("Vault installed via systemd socket activation.")


def _handle_vault_uninstall() -> None:
    """Remove vault systemd units."""
    from ..vault.daemon.lifecycle import VaultManager

    mgr = VaultManager()
    if not mgr.is_systemd_available():
        print("Error: systemd user services are not available. Nothing to uninstall.")
        raise SystemExit(1)
    mgr.uninstall_systemd_units()
    print("Vault systemd units removed.")


def _handle_vault_unlock(*, cfg: SandboxConfig | None = None) -> None:
    """Write the credentials-DB passphrase to the session-unlock tmpfs file; restart the daemon."""
    from .._yaml import write_secret_text
    from ..vault.daemon.lifecycle import VaultManager
    from ..vault.store.encryption import prompt_passphrase

    if cfg is None:
        cfg = SandboxConfig()

    write_secret_text(cfg.vault_passphrase_file, prompt_passphrase() + "\n")
    print(f"→ wrote passphrase to {cfg.vault_passphrase_file} (RAM-backed, cleared on reboot)")

    mgr = VaultManager(cfg)
    if mgr.is_daemon_running():
        mgr.stop_daemon()
        mgr.start_daemon()
        print("→ vault daemon restarted")
    else:
        print("→ vault daemon is not running; start it with `terok-sandbox vault start`")


def _forget_config_tier_updates(cfg: SandboxConfig) -> dict[str, object | None]:
    """Return the config-section patch ``vault passphrase destroy`` should apply.

    Both fields are auto-resolution wirings — leaving either would let
    the daemon re-unlock on next socket activation and defeat the lock.
    """
    updates: dict[str, object | None] = {}
    if cfg.credentials_passphrase:
        updates["passphrase"] = None
    if cfg.credentials_passphrase_command:
        updates["passphrase_command"] = None
    return updates


def _handle_vault_lock(*, cfg: SandboxConfig | None = None, forget: bool = False) -> None:
    """Delete the session-unlock tmpfs file and stop the vault daemon.

    By default this only clears the *session* tier; any of keyring,
    ``credentials.passphrase``, ``credentials.passphrase_command``, or a
    sealed systemd-creds credential can re-unlock the daemon at the next
    socket activation — and the operator may not realise that's not
    "locked".  Pass ``forget=True`` (CLI:
    ``terok-sandbox vault passphrase destroy``) to also remove the
    keyring entry, clear those config keys from ``config.yml``, and
    delete the sealed systemd-creds credential so the next daemon start
    *must* have an explicit ``vault unlock``.
    """
    from ..vault.daemon.lifecycle import VaultManager
    from ..vault.store.encryption import forget_passphrase_in_keyring

    if cfg is None:
        cfg = SandboxConfig()

    path = cfg.vault_passphrase_file
    if path.exists():
        path.unlink()
        print(f"→ removed {path}")
    else:
        print(f"→ {path} was not present")

    sealed_cred = cfg.vault_systemd_creds_file
    if forget:
        if cfg.credentials_use_keyring:
            from ..vault.store.encryption import load_passphrase_from_keyring

            if forget_passphrase_in_keyring():
                print("→ cleared keyring entry")
            elif load_passphrase_from_keyring() is None:
                # ``keyring.delete_password`` raises on a missing entry on most
                # backends, which the helper folds to False — a residual entry
                # after that means the backend rejected the delete.
                print("→ keyring entry already absent")
            else:
                raise SystemExit(
                    "failed to clear keyring entry; daemon may still auto-unlock from keyring"
                )
        config_updates = _forget_config_tier_updates(cfg)
        if config_updates:
            from .. import config as _config
            from ..paths import _config_file_paths

            user_config = next((p for label, p in _config_file_paths() if label == "user"), None)
            if user_config is not None and user_config.exists():
                _yaml_update_section(
                    user_config,
                    "credentials",
                    config_updates,
                )
                _config._credentials_section.cache_clear()
                for key in config_updates:
                    print(f"→ cleared credentials.{key} from config.yml")
        if sealed_cred.exists():
            try:
                sealed_cred.unlink()
            except OSError as exc:
                raise SystemExit(
                    f"failed to remove sealed credential at {sealed_cred}: {exc}"
                ) from exc
            print(f"→ removed sealed credential at {sealed_cred}")
    else:
        active = []
        if cfg.credentials_use_keyring:
            active.append("keyring")
        if cfg.credentials_passphrase_command:
            active.append("passphrase_command")
        if cfg.credentials_passphrase:
            active.append("config.yml")
        if sealed_cred.is_file():
            active.append("systemd-creds")
        if active:
            print(
                f"warning: non-session passphrase tiers still active ({', '.join(active)});"
                " the daemon may auto-unlock on next socket activation.\n"
                "         Use `terok-sandbox vault passphrase destroy` to clear them too.",
                file=sys.stderr,
            )

    mgr = VaultManager(cfg)
    if mgr.is_daemon_running():
        mgr.stop_daemon()
        print("→ vault daemon stopped")


#: CLI verbs for ``vault seal --key=``, mapped to systemd-creds' own
#: ``--with-key=`` vocabulary.  ``"auto"`` is the natural default —
#: systemd already picks ``host+tpm2`` on TPM-equipped hosts and
#: ``host`` otherwise, and second-guessing that decision here would
#: silently weaken the dual-factor default.
_SEAL_KEY_MODES: dict[str, KeyMode] = {
    "auto": "auto",
    "tpm": "tpm2",
    "host": "host",
    "tpm+host": "host+tpm2",
}


def _handle_vault_seal(*, cfg: SandboxConfig | None = None, key: str = "auto") -> None:
    """Seal the credentials-DB passphrase into a systemd-creds credential.

    Adds the systemd-creds tier to the resolution chain: machine-bound
    (TPM2 + host key, or either alone), survives reboot, no OS
    keyring required.  After sealing, the daemon resolves the
    passphrase via ``systemd-creds decrypt`` on every start — no
    operator interaction needed at boot, no plaintext-on-disk.

    Requires an already-resolvable passphrase — typically from a fresh
    ``vault unlock`` in the current session.  Doesn't touch other tiers
    or restart the daemon; the new tier is picked up on the next chain
    walk.
    """
    from ..vault.store import systemd_creds
    from ..vault.store.encryption import WrongPassphraseError

    if cfg is None:
        cfg = SandboxConfig()

    if not systemd_creds.is_available():
        raise SystemExit(
            "systemd-creds unavailable: needs systemd ≥ 257 with the Varlink"
            " io.systemd.Credentials interface (Fedora ≥ 42, Debian ≥ 13)"
        )

    key_mode = _SEAL_KEY_MODES.get(key)
    if key_mode is None:
        choices = ", ".join(sorted(_SEAL_KEY_MODES))
        raise SystemExit(f"unknown --key value: {key!r} (expected one of: {choices})")

    # A prompt here would accept a freshly-typed value and seal *that*,
    # leaving the next chain walk holding a key that doesn't open the DB.
    try:
        passphrase = cfg.resolve_passphrase()
    except WrongPassphraseError as exc:
        raise SystemExit(f"cannot seal: {exc}") from exc
    if passphrase is None:
        raise SystemExit("no current passphrase to seal — run `terok-sandbox vault unlock` first")

    try:
        systemd_creds.seal(passphrase, cfg.vault_systemd_creds_file, key_mode=key_mode)
    except RuntimeError as exc:
        # ``tpm2`` requested on a TPM-less host surfaces as a CalledProcessError
        # bubbled to RuntimeError — pass it through with the hint attached.
        raise SystemExit(str(exc)) from exc

    print(f"→ sealed passphrase to {cfg.vault_systemd_creds_file} (--with-key={key_mode})")
    print("  the resolution chain will pick this up on the next daemon start; no restart required")


def _handle_vault_to_keyring(*, cfg: SandboxConfig | None = None) -> None:
    """Move the current passphrase from its current tier into the OS keyring.

    Resolves the passphrase via the chain (or prompts as a last resort),
    writes it to the keyring, flips ``credentials.use_keyring`` to true
    in ``config.yml``, clears any plaintext ``credentials.passphrase`` /
    ``credentials.passphrase_command`` wiring, removes the session-file
    and sealed systemd-creds copies, and restarts the daemon so the
    next chain walk hits keyring.

    The validate-before-destroy ordering is deliberate: if the keyring
    write fails, the source tier is still intact.
    """
    from .. import config as _config
    from ..vault.daemon.lifecycle import VaultManager
    from ..vault.store.encryption import (
        WrongPassphraseError,
        store_passphrase_in_keyring,
    )

    if cfg is None:
        cfg = SandboxConfig()

    try:
        passphrase, source = cfg.resolve_passphrase_with_source(prompt_on_tty=True)
    except WrongPassphraseError as exc:
        raise SystemExit(f"cannot move to keyring: {exc}") from exc

    if not passphrase:
        raise SystemExit("no current passphrase resolvable; run `terok-sandbox vault unlock` first")
    if source == "keyring":
        print("→ passphrase is already in the keyring; nothing to do")
        return

    if not store_passphrase_in_keyring(passphrase):
        raise SystemExit("OS keyring is unreachable or denied; aborting (nothing was changed)")
    print(f"→ stored passphrase in keyring (was: {source})")

    # Switch the config's tier wiring atomically: flip use_keyring on,
    # drop the plaintext + helper fallbacks so the chain can't re-resolve
    # via a stale lower tier.
    from ..paths import _config_file_paths

    user_config = next((p for label, p in _config_file_paths() if label == "user"), None)
    if user_config is not None:
        # nosec: B105 — clearing config keys to None, not hardcoding secrets
        updates = {  # nosec: B105
            "use_keyring": True,
            "passphrase": None,  # nosec: B105
            "passphrase_command": None,  # nosec: B105
        }
        _yaml_update_section(user_config, "credentials", updates)
        _config._credentials_section.cache_clear()
        print(f"→ updated {user_config} (use_keyring: true, plaintext fields cleared)")

    # Remove the old tier's persistent copy.  Session file is removed
    # because the chain prefers it over keyring; sealed systemd-creds
    # likewise outranks keyring on the resolution order.
    for stale in (cfg.vault_passphrase_file, cfg.vault_systemd_creds_file):
        if stale.exists():
            stale.unlink()
            print(f"→ removed {sanitize_tty(str(stale))}")

    mgr = VaultManager(cfg)
    if mgr.is_daemon_running():
        mgr.stop_daemon()
        mgr.start_daemon()
        print("→ vault daemon restarted")


def _handle_vault_destroy_passphrase(*, cfg: SandboxConfig | None = None) -> None:
    """Clear every persistent passphrase tier — the destructive lock.

    Removes the session file, keyring entry, sealed systemd-creds
    credential, and plaintext ``config.yml`` fields, then stops the
    daemon.  The vault becomes unrecoverable unless the operator has an
    external copy of the passphrase.

    See [`_handle_vault_lock`][terok_sandbox.commands.vault._handle_vault_lock]
    — this is its ``forget=True`` mode, exposed as a distinct verb so
    the operation reads as "I am throwing away the key" in shell
    history rather than as a flag on the routine "lock for this
    session" verb.
    """
    _handle_vault_lock(cfg=cfg, forget=True)


VAULT_COMMANDS: tuple[CommandDef, ...] = (
    CommandDef(
        name="start",
        help="Start the vault daemon",
        handler=_handle_vault_start,
        group="vault",
    ),
    CommandDef(
        name="stop",
        help="Stop the vault daemon",
        handler=_handle_vault_stop,
        group="vault",
    ),
    CommandDef(
        name="status",
        help="Show vault status",
        handler=_handle_vault_status,
        group="vault",
    ),
    CommandDef(
        name="install",
        help="Install systemd socket activation",
        handler=_handle_vault_install,
        group="vault",
    ),
    CommandDef(
        name="uninstall",
        help="Remove systemd units",
        handler=_handle_vault_uninstall,
        group="vault",
    ),
    CommandDef(
        name="unlock",
        help="Provision the credentials-DB passphrase for this session (tmpfs file)",
        handler=_handle_vault_unlock,
        group="vault",
    ),
    CommandDef(
        name="lock",
        help="Remove the session-unlock tmpfs file and stop the vault daemon",
        handler=_handle_vault_lock,
        group="vault",
    ),
)


VAULT_PASSPHRASE_COMMANDS: tuple[CommandDef, ...] = (
    CommandDef(
        name="seal",
        help="Seal the current passphrase into a systemd-creds credential",
        handler=_handle_vault_seal,
        group="vault-passphrase",
        args=(
            ArgDef(
                name="--key",
                default="auto",
                help=(
                    "Sealing key: 'auto' (host+TPM2 if a TPM is present,"
                    " host alone otherwise), 'tpm' (require TPM2),"
                    " 'host' (host key only), 'tpm+host' (pin both)"
                ),
            ),
        ),
    ),
    CommandDef(
        name="to-keyring",
        help="Move the current passphrase from its current tier into the OS keyring",
        handler=_handle_vault_to_keyring,
        group="vault-passphrase",
    ),
    CommandDef(
        name="destroy",
        help=(
            "Clear every persistent passphrase tier — the vault becomes"
            " unrecoverable without an external copy"
        ),
        handler=_handle_vault_destroy_passphrase,
        group="vault-passphrase",
    ),
)


__all__ = ["VAULT_COMMANDS", "VAULT_PASSPHRASE_COMMANDS"]
