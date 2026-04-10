# SPDX-FileCopyrightText: 2026 Jiri Vyskocil
# SPDX-License-Identifier: Apache-2.0

"""Command registry for terok-sandbox.

Follows the same :class:`CommandDef` / :class:`ArgDef` pattern as
``terok_shield.registry``.  Higher-level consumers (terok, terok-agent)
can import ``COMMANDS`` to build their own CLI frontends without
duplicating argument definitions or handler logic.

Shield commands are delegated to terok-shield's own registry —
``SHIELD_COMMANDS`` re-exports the non-standalone subset.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, NamedTuple

from ._util import sanitize_tty

if TYPE_CHECKING:
    from collections.abc import Callable
    from pathlib import Path

    from .config import SandboxConfig


class KeyRow(NamedTuple):
    """One registered SSH key, fully resolved for display and matching."""

    scope: str
    comment: str
    key_type: str
    fingerprint: str
    private_key: str
    public_key: str


@dataclass(frozen=True)
class ArgDef:
    """Definition of a single CLI argument."""

    name: str
    help: str = ""
    type: Callable[[str], Any] | None = None
    default: Any = None
    action: str | None = None
    dest: str | None = None
    nargs: int | str | None = None
    required: bool = False


@dataclass(frozen=True)
class CommandDef:
    """Definition of a sandbox subcommand.

    Attributes:
        name: Subcommand name (e.g. ``"gate start"``).
        help: One-line help string.
        handler: Callable implementing the command.
        args: Argument definitions.
        group: Command group (e.g. ``"gate"``, ``"shield"``).
    """

    name: str
    help: str = ""
    handler: Callable[..., None] | None = None
    args: tuple[ArgDef, ...] = ()
    group: str = ""


# ---------------------------------------------------------------------------
# Gate handlers
# ---------------------------------------------------------------------------


def _handle_gate_start(
    *, port: int | None = None, daemon: bool = False, cfg: SandboxConfig | None = None
) -> None:
    """Start the gate server (systemd preferred, daemon fallback)."""
    from .gate.lifecycle import install_systemd_units, is_systemd_available, start_daemon

    if is_systemd_available() and not daemon:
        install_systemd_units(cfg=cfg)
        print("Gate server started via systemd socket activation.")
    else:
        start_daemon(port=port, cfg=cfg)
        print("Gate server daemon started.")


def _handle_gate_stop(*, cfg: SandboxConfig | None = None) -> None:
    """Stop the gate server."""
    from .gate.lifecycle import get_server_status, stop_daemon, uninstall_systemd_units

    status = get_server_status(cfg=cfg)
    if status.mode == "systemd":
        uninstall_systemd_units(cfg=cfg)
        print("Gate server systemd units removed.")
    elif status.mode == "daemon":
        stop_daemon(cfg=cfg)
        print("Gate server daemon stopped.")
    else:
        print("Gate server is not running.")


def _handle_gate_status(*, cfg: SandboxConfig | None = None) -> None:
    """Show gate server status."""
    from .gate.lifecycle import check_units_outdated, get_gate_base_path, get_server_status

    status = get_server_status(cfg=cfg)
    print(f"Mode:      {status.mode}")
    print(f"Running:   {'yes' if status.running else 'no'}")
    print(f"Port:      {status.port}")
    print(f"Base path: {get_gate_base_path(cfg=cfg)}")

    warning = check_units_outdated(cfg=cfg)
    if warning:
        import sys

        print(f"\nWarning: {warning}", file=sys.stderr)
        print("Run 'terok-sandbox gate start' to update.", file=sys.stderr)


# ---------------------------------------------------------------------------
# Shield handlers (thin wrappers around terok_sandbox.shield)
# ---------------------------------------------------------------------------


def _handle_shield_setup(*, root: bool = False, user: bool = False) -> None:
    """Install OCI hooks for the shield firewall."""
    from .shield import run_setup

    run_setup(root=root, user=user)


def _handle_shield_status() -> None:
    """Show shield configuration and environment check."""
    import sys

    from .shield import check_environment, status

    env = check_environment()
    cfg = status()

    print(f"Shield mode:    {cfg.get('mode', '?')}")
    print(f"Profiles:       {', '.join(cfg.get('profiles', []))}")
    print(f"Audit:          {'enabled' if cfg.get('audit_enabled') else 'disabled'}")
    print(f"Hooks:          {env.hooks}")
    print(f"Health:         {env.health}")
    if env.needs_setup:
        print(f"\n{env.setup_hint}", file=sys.stderr)


# ---------------------------------------------------------------------------
# Command definitions
# ---------------------------------------------------------------------------

GATE_COMMANDS: tuple[CommandDef, ...] = (
    CommandDef(
        name="start",
        help="Start the gate server",
        handler=_handle_gate_start,
        group="gate",
        args=(
            ArgDef(name="--port", type=int, default=None, help="Override port (default: 9418)"),
            ArgDef(name="--daemon", action="store_true", help="Force daemon mode (skip systemd)"),
        ),
    ),
    CommandDef(
        name="stop",
        help="Stop the gate server",
        handler=_handle_gate_stop,
        group="gate",
    ),
    CommandDef(
        name="status",
        help="Show gate server status",
        handler=_handle_gate_status,
        group="gate",
    ),
)

SHIELD_COMMANDS: tuple[CommandDef, ...] = (
    CommandDef(
        name="setup",
        help="Install OCI hooks for the shield firewall",
        handler=_handle_shield_setup,
        group="shield",
        args=(
            ArgDef(name="--root", action="store_true", help="Install system-wide (requires sudo)"),
            ArgDef(name="--user", action="store_true", help="Install to user hooks directory"),
        ),
    ),
    CommandDef(
        name="status",
        help="Show shield status",
        handler=_handle_shield_status,
        group="shield",
    ),
)

# ---------------------------------------------------------------------------
# Credential proxy handlers
# ---------------------------------------------------------------------------


def _handle_proxy_start() -> None:
    """Start the credential proxy daemon."""
    from .credentials.lifecycle import get_proxy_status, start_daemon

    status = get_proxy_status()
    if status.running:
        print("Credential proxy is already running.")
        return
    start_daemon()
    print("Credential proxy started.")


def _handle_proxy_stop() -> None:
    """Stop the credential proxy daemon."""
    from .credentials.lifecycle import is_daemon_running, stop_daemon

    if not is_daemon_running():
        print("Credential proxy is not running.")
        return
    stop_daemon()
    print("Credential proxy stopped.")


def _handle_proxy_status() -> None:
    """Show credential proxy status."""
    from .credentials.lifecycle import get_proxy_status

    status = get_proxy_status()
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


def _handle_proxy_install() -> None:
    """Install and start systemd socket activation for the credential proxy."""
    from .credentials.lifecycle import install_systemd_units, is_systemd_available

    if not is_systemd_available():
        print("Error: systemd user services are not available on this host.")
        raise SystemExit(1)
    install_systemd_units()
    print("Credential proxy installed via systemd socket activation.")


def _handle_proxy_uninstall() -> None:
    """Remove credential proxy systemd units."""
    from .credentials.lifecycle import is_systemd_available, uninstall_systemd_units

    if not is_systemd_available():
        print("Error: systemd user services are not available. Nothing to uninstall.")
        raise SystemExit(1)
    uninstall_systemd_units()
    print("Credential proxy systemd units removed.")


PROXY_COMMANDS: tuple[CommandDef, ...] = (
    CommandDef(
        name="start",
        help="Start the credential proxy daemon",
        handler=_handle_proxy_start,
        group="proxy",
    ),
    CommandDef(
        name="stop",
        help="Stop the credential proxy daemon",
        handler=_handle_proxy_stop,
        group="proxy",
    ),
    CommandDef(
        name="status",
        help="Show credential proxy status",
        handler=_handle_proxy_status,
        group="proxy",
    ),
    CommandDef(
        name="install",
        help="Install systemd socket activation",
        handler=_handle_proxy_install,
        group="proxy",
    ),
    CommandDef(
        name="uninstall",
        help="Remove systemd units",
        handler=_handle_proxy_uninstall,
        group="proxy",
    ),
)

# ---------------------------------------------------------------------------
# SSH handlers
# ---------------------------------------------------------------------------


def _build_key_rows(cfg: SandboxConfig) -> list[KeyRow]:
    """Load ssh-keys.json and resolve each entry into a displayable row.

    Shared by ``list`` and ``remove-key`` so both present identical
    information.  Returns an empty list when no keys are registered.
    """
    import base64
    import hashlib
    import json
    from pathlib import Path

    keys_path = cfg.ssh_keys_json_path
    if not keys_path.is_file():
        return []

    try:
        data = json.loads(keys_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise SystemExit(f"Cannot read {keys_path}: {exc}") from exc

    if not isinstance(data, dict):
        raise SystemExit(f"Cannot read {keys_path}: expected top-level JSON object")

    rows: list[KeyRow] = []
    for scope in sorted(data):
        entries = data[scope]
        if not isinstance(entries, list):
            continue
        for entry in entries:
            if not isinstance(entry, dict):
                continue
            pub_path = Path(entry.get("public_key", ""))
            priv_path = entry.get("private_key", "")
            if pub_path.is_file():
                try:
                    parts = pub_path.read_text(encoding="utf-8").strip().split()
                    key_type = parts[0].removeprefix("ssh-") if parts else "?"
                    blob = base64.b64decode(parts[1]) if len(parts) > 1 else b""
                    comment = " ".join(parts[2:]) if len(parts) > 2 else pub_path.stem
                    digest = base64.b64encode(hashlib.sha256(blob).digest()).rstrip(b"=")
                    fingerprint = f"SHA256:{digest.decode()}"
                except Exception:
                    key_type, comment, fingerprint = "?", pub_path.stem, "(error)"
            else:
                key_type, comment, fingerprint = "?", Path(priv_path).stem, "(pub missing)"
            rows.append(KeyRow(scope, comment, key_type, fingerprint, priv_path, str(pub_path)))
    return rows


def _print_key_table(rows: list[KeyRow], *, numbered: bool = False) -> None:
    """Print a formatted table of SSH key rows.

    All untrusted fields are sanitized before display to prevent
    terminal escape injection from crafted key comments or paths.

    Args:
        rows: Key rows to display.
        numbered: Prefix each row with a 1-based index for interactive selection.
    """
    if not rows:
        print("No SSH keys registered.")
        return

    headers = ("SCOPE", "KEY", "TYPE", "FINGERPRINT", "PATH")
    # Sanitize untrusted fields before computing widths and formatting
    display = [
        tuple(
            sanitize_tty(f) for f in (r.scope, r.comment, r.key_type, r.fingerprint, r.public_key)
        )
        for r in rows
    ]
    widths = [max(len(h), *(len(d[i]) for d in display)) for i, h in enumerate(headers)]

    if numbered:
        idx_w = len(str(len(display)))
        prefix_w = idx_w + 2  # "N) " or "   "
        fmt = f"{{:<{prefix_w}}}" + "  ".join(f"{{:<{w}}}" for w in widths)
        print(fmt.format("", *headers))
        for i, d in enumerate(display, 1):
            print(fmt.format(f"{i})", *d))
    else:
        fmt = "  ".join(f"{{:<{w}}}" for w in widths)
        print(fmt.format(*headers))
        for d in display:
            print(fmt.format(*d))


def _handle_ssh_import(
    *,
    scope: str,
    private_key: str,
    public_key: str | None = None,
    create_scope: bool = False,
    cfg: SandboxConfig | None = None,
) -> None:
    """Copy an SSH keypair into the managed key store and register it in ssh-keys.json."""
    import os
    import re
    import shutil
    from pathlib import Path

    from .config import SandboxConfig as _SandboxConfig
    from .credentials.ssh import SSHInitResult, update_ssh_keys_json

    if not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9._-]*", scope):
        raise SystemExit(
            f"Invalid scope {scope!r}: must start with a letter or digit "
            "and contain only [A-Za-z0-9._-]"
        )

    if cfg is None:
        cfg = _SandboxConfig()

    _validate_scope_exists(scope, create_scope, cfg)

    priv_src = Path(private_key).expanduser().resolve()
    pub_src = Path(public_key).expanduser().resolve() if public_key else Path(f"{priv_src}.pub")

    if not priv_src.exists():
        raise SystemExit(f"Private key not found: {priv_src}")
    if not pub_src.exists():
        raise SystemExit(
            f"Public key not found: {pub_src} (use --public-key to specify explicitly)"
        )

    dest_dir = cfg.ssh_keys_dir / scope
    dest_dir.mkdir(parents=True, exist_ok=True)

    def _unique_dst(src: Path) -> Path:
        """Return dest path for *src* inside *dest_dir*, with numeric suffix on collision.

        A destination that already exists with identical content is considered
        the same key (re-import) and returns the existing path unchanged.
        """
        p = dest_dir / src.name
        if not p.exists() or p.read_bytes() == src.read_bytes():
            return p
        stem, suffix = p.stem, p.suffix
        n = 1
        while True:
            p = dest_dir / f"{stem}_{n}{suffix}"
            if not p.exists() or p.read_bytes() == src.read_bytes():
                return p
            n += 1

    priv_dst = _unique_dst(priv_src)
    pub_dst = _unique_dst(pub_src)

    print(f"Copying private key: {sanitize_tty(str(priv_src))}")
    print(f"              → {sanitize_tty(str(priv_dst))}")
    shutil.copy2(str(priv_src), str(priv_dst))
    os.chmod(priv_dst, 0o600)

    print(f"Copying public key:  {sanitize_tty(str(pub_src))}")
    print(f"              → {sanitize_tty(str(pub_dst))}")
    shutil.copy2(str(pub_src), str(pub_dst))

    result = SSHInitResult(
        dir=str(dest_dir),
        private_key=str(priv_dst),
        public_key=str(pub_dst),
        config_path="",
        key_name=priv_dst.name,
    )
    keys_path = cfg.ssh_keys_json_path
    update_ssh_keys_json(keys_path, scope, result)
    print(f"Registered key for scope '{sanitize_tty(scope)}': {sanitize_tty(str(priv_dst))}")


def _handle_ssh_add_key(
    *,
    scope: str,
    name: str | None = None,
    key_type: str = "ed25519",
    create_scope: bool = False,
    cfg: SandboxConfig | None = None,
) -> None:
    """Generate a new SSH keypair and register it for a credential scope."""
    import re

    from .config import SandboxConfig as _SandboxConfig
    from .credentials.ssh import (
        SSHInitResult,
        _harden_permissions,
        _next_key_number,
        generate_keypair,
        update_ssh_keys_json,
    )

    if not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9._-]*", scope):
        raise SystemExit(
            f"Invalid scope {scope!r}: must start with a letter or digit "
            "and contain only [A-Za-z0-9._-]"
        )
    if key_type not in ("ed25519", "rsa"):
        raise SystemExit("Unsupported --key-type. Use 'ed25519' or 'rsa'.")

    algo = "ed25519" if key_type == "ed25519" else "rsa"
    if cfg is None:
        cfg = _SandboxConfig()

    _validate_scope_exists(scope, create_scope, cfg)

    dest_dir = cfg.ssh_keys_dir / scope
    dest_dir.mkdir(parents=True, exist_ok=True)

    if name is not None:
        if not re.fullmatch(r"[a-zA-Z_-]+", name):
            raise SystemExit(
                f"Invalid key name {name!r}: must contain only letters, underscores, and hyphens"
            )
        key_name = name
    else:
        key_name = f"key-{_next_key_number(dest_dir, algo)}"

    filename = f"id_{algo}_{key_name}"
    priv_path = dest_dir / filename
    pub_path = dest_dir / f"{filename}.pub"

    existing = priv_path if priv_path.exists() else pub_path if pub_path.exists() else None
    if existing:
        raise SystemExit(
            f"Key file already exists: {existing}\n"
            "Use a different --name or remove the existing key first."
        )

    comment = f"tk-side:{scope}:{key_name}"
    generate_keypair(key_type, priv_path, pub_path, comment)

    try:
        _harden_permissions(dest_dir, priv_path, pub_path, dest_dir / "config")
    except OSError as e:
        raise SystemExit(f"Failed to set permissions: {e}") from e

    result = SSHInitResult(
        dir=str(dest_dir),
        private_key=str(priv_path),
        public_key=str(pub_path),
        config_path="",
        key_name=filename,
    )
    update_ssh_keys_json(cfg.ssh_keys_json_path, scope, result)

    print(f"SSH key generated for scope '{sanitize_tty(scope)}':")
    print(f"  name:        {sanitize_tty(key_name)}")
    print(f"  private key: {sanitize_tty(str(priv_path))}")
    print(f"  public key:  {sanitize_tty(str(pub_path))}")
    print(f"  comment:     {sanitize_tty(comment)}")
    try:
        pub_text = pub_path.read_text(encoding="utf-8", errors="ignore").strip()
        if pub_text:
            print("Public key (add as deploy key):")
            print(f"  {sanitize_tty(pub_text)}")
    except Exception:
        pass


def _validate_scope_exists(scope: str, create_scope: bool, cfg: SandboxConfig) -> None:
    """Reject unknown scopes unless ``--create-scope`` was passed.

    Scopes are considered "known" if they appear in ``ssh-keys.json``.
    """
    import json

    keys_path = cfg.ssh_keys_json_path
    existing: dict = {}
    if keys_path.is_file():
        try:
            existing = json.loads(keys_path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            pass
    if scope in existing or create_scope:
        return
    known = sorted(existing)
    msg = f"Unknown scope {scope!r}."
    if known:
        msg += f" Known scopes: {', '.join(known)}"
    msg += "\nUse --create-scope to create a new credential scope."
    raise SystemExit(msg)


def _handle_ssh_list(
    *,
    scope: str | None = None,
    cfg: SandboxConfig | None = None,
) -> None:
    """List SSH keys registered in the auth proxy's key store."""
    from .config import SandboxConfig as _SandboxConfig

    if cfg is None:
        cfg = _SandboxConfig()

    rows = _build_key_rows(cfg)
    if scope:
        filtered = [r for r in rows if r.scope == scope]
        if not filtered:
            raise SystemExit(f"No keys registered for scope {scope!r}")
        rows = filtered

    _print_key_table(rows)


def _remove_keys_from_json(keys_json_path: Path, removals: list[KeyRow]) -> None:
    """Delete key entries from ssh-keys.json, removing empty scopes.

    Matches by ``private_key`` path — the stable identifier across
    renames of the public key or comment changes.  Uses the same
    ``fcntl.flock`` concurrency guard as :func:`update_ssh_keys_json`.
    """
    import fcntl
    import json
    import os

    remove_set = {r.private_key for r in removals}

    fd = os.open(str(keys_json_path), os.O_RDWR | os.O_CREAT, 0o600)
    try:
        fcntl.flock(fd, fcntl.LOCK_EX)
        chunks: list[bytes] = []
        while chunk := os.read(fd, 8192):
            chunks.append(chunk)
        raw = b"".join(chunks)
        try:
            parsed = json.loads(raw) if raw.strip() else {}
        except json.JSONDecodeError as exc:
            raise SystemExit(f"Cannot read {keys_json_path}: {exc}") from exc
        if not isinstance(parsed, dict):
            raise SystemExit(f"Cannot read {keys_json_path}: expected top-level JSON object")
        mapping: dict = parsed

        for scope in list(mapping):
            entries = mapping[scope]
            if not isinstance(entries, list):
                continue
            mapping[scope] = [
                e for e in entries if isinstance(e, dict) and e.get("private_key") not in remove_set
            ]
            if not mapping[scope]:
                del mapping[scope]

        data = (json.dumps(mapping, indent=2) + "\n").encode("utf-8")
        os.lseek(fd, 0, os.SEEK_SET)
        os.ftruncate(fd, 0)
        os.write(fd, data)
    finally:
        fcntl.flock(fd, fcntl.LOCK_UN)
        os.close(fd)


def _delete_key_files(rows: list[KeyRow], cfg: SandboxConfig) -> tuple[int, list[str]]:
    """Remove private and public key files from disk.

    Paths are validated against ``cfg.ssh_keys_dir`` to prevent a
    tampered ``ssh-keys.json`` from steering deletions outside the
    managed key directory (CWE-73).

    Continues past per-file failures so partial cleanup still completes.
    The registry has already been updated by the time this runs, so
    leaving a stale file is better than aborting mid-deletion.

    Returns:
        Tuple of (files deleted, list of error messages for failed deletions).
    """
    from pathlib import Path

    base = cfg.ssh_keys_dir.resolve()
    deleted = 0
    errors: list[str] = []
    for row in rows:
        for p in (row.private_key, row.public_key):
            path = Path(p)
            try:
                resolved = path.resolve(strict=True)
            except (FileNotFoundError, OSError):
                continue
            if not resolved.is_file():
                continue
            if not resolved.is_relative_to(base):
                errors.append(f"Refusing to delete outside managed dir: {resolved}")
                continue
            try:
                resolved.unlink()
                deleted += 1
            except OSError as exc:
                errors.append(f"{resolved}: {exc}")
    return deleted, errors


def _filter_key_rows(
    rows: list[KeyRow],
    *,
    scope: str | None = None,
    name: str | None = None,
    fingerprint: str | None = None,
) -> list[KeyRow]:
    """Narrow key rows by scope (exact), name (glob), and fingerprint (prefix)."""
    from fnmatch import fnmatch

    if scope:
        rows = [r for r in rows if r.scope == scope]
    if name:
        rows = [r for r in rows if fnmatch(r.comment, name)]
    if fingerprint:
        # Accept both "SHA256:..." and raw prefix
        fp = fingerprint.removeprefix("SHA256:")
        rows = [r for r in rows if r.fingerprint.removeprefix("SHA256:").startswith(fp)]
    return rows


def _prompt_file_action(*, delete_files: bool, keep_files: bool, yes: bool = False) -> bool:
    """Determine whether to delete key files, prompting if neither flag is set.

    Raises:
        SystemExit: If both ``--delete-files`` and ``--keep-files`` are set.
    """
    if delete_files and keep_files:
        raise SystemExit("Cannot use both --delete-files and --keep-files.")
    if delete_files:
        return True
    if keep_files or yes:
        return False
    try:
        answer = input("Also delete key files from disk? [y/N]: ").strip().lower()
    except (EOFError, KeyboardInterrupt):
        print()
        return False
    return answer in ("y", "yes")


def _handle_ssh_remove_key(
    *,
    scope: str | None = None,
    name: str | None = None,
    fingerprint: str | None = None,
    yes: bool = False,
    delete_files: bool = False,
    keep_files: bool = False,
    cfg: SandboxConfig | None = None,
) -> None:
    """Remove SSH keys from the auth proxy's key store.

    Two modes: interactive selection when called without filters, or
    direct matching when any of ``--scope``, ``--name``, or
    ``--fingerprint`` is provided.
    """
    from .config import SandboxConfig as _SandboxConfig

    if cfg is None:
        cfg = _SandboxConfig()

    all_rows = _build_key_rows(cfg)
    if not all_rows:
        raise SystemExit("No SSH keys registered.")

    has_filters = any((scope, name, fingerprint))

    if has_filters:
        # Parameterized mode — filter and confirm
        candidates = _filter_key_rows(all_rows, scope=scope, name=name, fingerprint=fingerprint)
        if not candidates:
            raise SystemExit("No keys match the given filters.")
        if not yes:
            n = len(candidates)
            if n > 1:
                print(f"Multiple keys match ({n}):\n")
            _print_key_table(candidates)
            prompt = f"\nRemove all {n} keys? [y/N]: " if n > 1 else "\nRemove this key? [y/N]: "
            try:
                answer = input(prompt).strip().lower()
            except (EOFError, KeyboardInterrupt):
                print()
                raise SystemExit("Aborted.") from None
            if answer not in ("y", "yes"):
                raise SystemExit("Aborted.")
    else:
        # Interactive mode — numbered list, user picks
        if yes:
            raise SystemExit("Cannot use --yes without at least one filter flag.")
        _print_key_table(all_rows, numbered=True)
        print()
        try:
            selection = input("Select key(s) to remove (number, comma-separated, or 'all'): ")
        except (EOFError, KeyboardInterrupt):
            print()
            raise SystemExit("Aborted.") from None
        selection = selection.strip().lower()
        if not selection:
            raise SystemExit("Aborted.")
        if selection == "all":
            candidates = list(all_rows)
        else:
            indices: list[int] = []
            for part in selection.split(","):
                part = part.strip()
                if not part.isdigit() or not (1 <= int(part) <= len(all_rows)):
                    raise SystemExit(
                        f"Invalid selection {part!r}. "
                        f"Enter a number 1–{len(all_rows)}, comma-separated, or 'all'."
                    )
                indices.append(int(part) - 1)
            candidates = [all_rows[i] for i in dict.fromkeys(indices)]

    # Determine file action
    do_delete = _prompt_file_action(delete_files=delete_files, keep_files=keep_files, yes=yes)

    # Execute removal
    _remove_keys_from_json(cfg.ssh_keys_json_path, candidates)
    files_deleted, file_errors = _delete_key_files(candidates, cfg) if do_delete else (0, [])

    n = len(candidates)
    msg = f"Removed {n} key{'s' if n != 1 else ''} from registry."
    if files_deleted:
        msg += f" Deleted {files_deleted} file{'s' if files_deleted != 1 else ''} from disk."
    elif not do_delete:
        msg += " Key files kept on disk."
    print(msg)
    for err in file_errors:
        print(f"  Warning: could not delete {err}", file=__import__("sys").stderr)


SSH_COMMANDS: tuple[CommandDef, ...] = (
    CommandDef(
        name="list",
        help="List SSH keys registered in the auth proxy",
        handler=_handle_ssh_list,
        group="ssh",
        args=(
            ArgDef(
                name="--scope",
                help="Show keys for a specific credential scope only",
                default=None,
            ),
        ),
    ),
    CommandDef(
        name="import",
        help="Register an existing SSH keypair in ssh-keys.json",
        handler=_handle_ssh_import,
        group="ssh",
        args=(
            ArgDef(name="scope", help="Credential scope to associate the key with"),
            ArgDef(
                name="--private-key",
                help="Path to the private key file",
                dest="private_key",
                required=True,
            ),
            ArgDef(
                name="--public-key",
                help="Path to the .pub file (default: <private-key>.pub)",
                default=None,
                dest="public_key",
            ),
            ArgDef(
                name="--create-scope",
                help="Allow creating a new credential scope",
                action="store_true",
                dest="create_scope",
            ),
        ),
    ),
    CommandDef(
        name="add-key",
        help="Generate a new SSH keypair for a credential scope",
        handler=_handle_ssh_add_key,
        group="ssh",
        args=(
            ArgDef(name="scope", help="Credential scope to associate the key with"),
            ArgDef(
                name="--name",
                help="Key name ([a-zA-Z_-]; auto-generates key-1, key-2, ... if omitted)",
                default=None,
            ),
            ArgDef(
                name="--key-type",
                help="Key algorithm: ed25519 (default) or rsa",
                default="ed25519",
                dest="key_type",
            ),
            ArgDef(
                name="--create-scope",
                help="Allow creating a new credential scope",
                action="store_true",
                dest="create_scope",
            ),
        ),
    ),
    CommandDef(
        name="remove-key",
        help="Remove SSH keys from the auth proxy's key store",
        handler=_handle_ssh_remove_key,
        group="ssh",
        args=(
            ArgDef(
                name="--scope",
                help="Filter by credential scope (exact match)",
                default=None,
            ),
            ArgDef(
                name="--name",
                help="Filter by key name/comment (supports glob wildcards)",
                default=None,
            ),
            ArgDef(
                name="--fingerprint",
                help="Filter by fingerprint prefix (min 8 chars recommended)",
                default=None,
            ),
            ArgDef(
                name="--yes",
                help="Skip confirmation prompts",
                action="store_true",
                dest="yes",
            ),
            ArgDef(
                name="--delete-files",
                help="Delete key files from disk (skip prompt)",
                action="store_true",
                dest="delete_files",
            ),
            ArgDef(
                name="--keep-files",
                help="Keep key files on disk (skip prompt)",
                action="store_true",
                dest="keep_files",
            ),
        ),
    ),
)

# ---------------------------------------------------------------------------
# Doctor handler
# ---------------------------------------------------------------------------


def _handle_doctor(*, cfg: SandboxConfig | None = None) -> None:
    """Run sandbox-level health checks and print results.

    This is the standalone host-side doctor — it runs on the host, not
    inside a container.  For non-host_side checks (network probes), we
    execute the probe_cmd directly via subprocess instead of ``podman exec``.
    For host_side checks (e.g. shield), we delegate to ``evaluate`` which
    performs the check itself using Python APIs.
    """
    import subprocess
    import sys

    from .config import SandboxConfig as _SandboxConfig
    from .credentials.lifecycle import get_proxy_port, get_ssh_agent_port
    from .doctor import sandbox_doctor_checks

    if cfg is None:
        cfg = _SandboxConfig()
    checks = sandbox_doctor_checks(
        proxy_port=get_proxy_port(cfg),
        ssh_agent_port=get_ssh_agent_port(cfg),
        desired_shield_state=None,  # standalone mode — no task context
    )
    worst = "ok"
    markers = {"ok": "ok", "warn": "WARN", "error": "ERROR"}
    for check in checks:
        if check.host_side:
            # Host-side checks perform the check inside evaluate() itself.
            verdict = check.evaluate(0, "", "")
        elif check.probe_cmd:
            # Non-host_side checks: run probe_cmd directly on the host
            # (the command targets host.containers.internal which resolves
            # to localhost when not inside a container, so we rewrite to
            # localhost for standalone execution).
            try:
                result = subprocess.run(  # noqa: S603
                    check.probe_cmd,
                    capture_output=True,
                    text=True,
                    timeout=5,
                )
                verdict = check.evaluate(result.returncode, result.stdout, result.stderr)
            except (FileNotFoundError, subprocess.TimeoutExpired):
                verdict = check.evaluate(1, "", "probe command unavailable or timed out")
        else:
            verdict = check.evaluate(0, "", "")
        tag = markers.get(verdict.severity, verdict.severity)
        print(f"  {check.label} .... {tag} ({verdict.detail})")
        if verdict.severity == "error" or worst == "error":
            worst = "error"
        elif verdict.severity == "warn" or worst == "warn":
            worst = "warn"

    if worst == "error":
        sys.exit(2)
    elif worst == "warn":
        sys.exit(1)


DOCTOR_COMMANDS: tuple[CommandDef, ...] = (
    CommandDef(
        name="doctor",
        help="Run sandbox health checks",
        handler=_handle_doctor,
        group="doctor",
    ),
)

#: All sandbox commands, grouped by subsystem.
COMMANDS: tuple[CommandDef, ...] = (
    GATE_COMMANDS + SHIELD_COMMANDS + PROXY_COMMANDS + SSH_COMMANDS + DOCTOR_COMMANDS
)
