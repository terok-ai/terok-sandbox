# SPDX-FileCopyrightText: 2026 Jiri Vyskocil
# SPDX-License-Identifier: Apache-2.0

"""SSH-key CLI verbs — list, import, add, export, pub, link, rename, remove.

Operates on the SSH key tables of the credentials DB.  Each handler
opens the DB with the CLI's prompt-on-tty policy and closes it before
returning.  Display formatting (``_print_key_table``) sanitises every
field via ``sanitize_tty`` so a hostile key comment can't inject
terminal escapes into operator output.
"""

from __future__ import annotations

from fnmatch import fnmatch
from pathlib import Path
from typing import TYPE_CHECKING

from .._util import sanitize_tty
from ..config import SandboxConfig
from ._types import ArgDef, CommandDef, KeyRow

if TYPE_CHECKING:
    from ..vault.store.db import CredentialDB


def _open_db(cfg: SandboxConfig) -> CredentialDB:
    """Open the vault credential DB for SSH operations (CLI flavour, TTY-prompt enabled)."""
    return cfg.open_credential_db(prompt_on_tty=True)


def _build_key_rows(cfg: SandboxConfig) -> list[KeyRow]:
    """Enumerate every registered SSH key as a displayable [`KeyRow`][terok_sandbox.commands.KeyRow]."""
    db = _open_db(cfg)
    try:
        rows: list[KeyRow] = []
        for scope in db.list_scopes_with_ssh_keys():
            for r in db.list_ssh_keys_for_scope(scope):
                rows.append(
                    KeyRow(
                        scope=scope,
                        comment=r.comment or f"id={r.id}",
                        key_type=r.key_type,
                        fingerprint=r.fingerprint,
                        private_key=f"db:ssh_keys/{r.id}",
                        public_key=f"db:ssh_keys/{r.id}",
                    )
                )
        rows.sort(key=lambda r: (r.scope, r.comment))
        return rows
    finally:
        db.close()


def _print_key_table(rows: list[KeyRow], *, numbered: bool = False) -> None:
    """Print a formatted table of SSH key rows.

    All untrusted fields are sanitised before display so a crafted
    comment or path can't inject terminal escapes.

    Args:
        rows: Key rows to display.
        numbered: Prefix each row with a 1-based index for interactive selection.
    """
    if not rows:
        print("No SSH keys registered.")
        return

    headers = ("SCOPE", "KEY", "TYPE", "FINGERPRINT", "PATH")
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


def _validate_scope_name(scope: str) -> None:
    """Raise [`SystemExit`][SystemExit] if *scope* is not a safe filesystem identifier.

    Delegates to the canonical DB-layer validator so the character set
    *and* the length bound stay co-located with the write sites that
    depend on them.
    """
    from ..vault.store.db import InvalidScopeName, _require_safe_scope

    try:
        _require_safe_scope(scope)
    except InvalidScopeName as exc:
        raise SystemExit(str(exc)) from exc


def _filter_key_rows(
    rows: list[KeyRow],
    *,
    scope: str | None = None,
    comment: str | None = None,
    fingerprint: str | None = None,
) -> list[KeyRow]:
    """Narrow key rows by scope (exact), comment (glob), and fingerprint (prefix)."""
    if scope:
        rows = [r for r in rows if r.scope == scope]
    if comment:
        rows = [r for r in rows if fnmatch(r.comment, comment)]
    if fingerprint:
        fp = fingerprint.removeprefix("SHA256:")
        rows = [r for r in rows if r.fingerprint.removeprefix("SHA256:").startswith(fp)]
    return rows


def _key_id_from_row(row: KeyRow) -> int:
    """Extract the ``ssh_keys.id`` from a row's pseudo-path ``db:ssh_keys/<id>``."""
    return int(row.private_key.rsplit("/", 1)[-1])


def _handle_ssh_list(
    *,
    scope: str | None = None,
    cfg: SandboxConfig | None = None,
) -> None:
    """List SSH keys registered in the auth proxy's key store."""
    if cfg is None:
        cfg = SandboxConfig()

    rows = _build_key_rows(cfg)
    if scope:
        filtered = [r for r in rows if r.scope == scope]
        if not filtered:
            raise SystemExit(f"No keys registered for scope {scope!r}")
        rows = filtered

    _print_key_table(rows)


def _handle_ssh_import(
    *,
    scope: str,
    private_key: str,
    public_key: str | None = None,
    comment: str | None = None,
    cfg: SandboxConfig | None = None,
) -> None:
    """Import an OpenSSH keypair from files into the vault DB for *scope*."""
    from ..vault.ssh.keypair import (
        KeypairMismatchError,
        PasswordProtectedKeyError,
        import_ssh_keypair,
    )
    from ..vault.store.db import UnsafeCommentError

    _validate_scope_name(scope)
    if cfg is None:
        cfg = SandboxConfig()

    priv_path = Path(private_key).expanduser().resolve()
    pub_path = Path(public_key).expanduser().resolve() if public_key else None

    if not priv_path.is_file():
        raise SystemExit(f"Private key not found: {priv_path}")
    if pub_path is not None and not pub_path.is_file():
        raise SystemExit(f"Public key not found: {pub_path}")

    db = _open_db(cfg)
    try:
        try:
            result = import_ssh_keypair(
                db,
                scope,
                priv_path,
                pub_path=pub_path,
                comment=comment,
            )
        except PasswordProtectedKeyError as exc:
            raise SystemExit(
                f"{exc}  Run `ssh-keygen -p -f {priv_path}` to strip the passphrase."
            ) from exc
        except (KeypairMismatchError, UnsafeCommentError, ValueError) as exc:
            raise SystemExit(f"Import failed: {exc}") from exc

        pretty_scope = sanitize_tty(scope)
        if not result.already_present:
            headline = f"Imported new key to scope '{pretty_scope}':"
        elif result.scope_was_assigned:
            headline = f"Key already linked to scope '{pretty_scope}' — nothing to do:"
        else:
            headline = f"Linked existing vault key to scope '{pretty_scope}':"
        print(
            f"{headline}\n"
            f"  id:          {result.key_id}\n"
            f"  fingerprint: {sanitize_tty(result.fingerprint)}\n"
            f"  comment:     {sanitize_tty(result.comment)}"
        )
    finally:
        db.close()


def _handle_ssh_add(
    *,
    scope: str,
    key_type: str = "ed25519",
    comment: str | None = None,
    force: bool = False,
    cfg: SandboxConfig | None = None,
) -> None:
    """Generate a new SSH keypair in the vault for *scope*."""
    from ..vault.ssh.manager import SSHManager

    _validate_scope_name(scope)
    if key_type not in ("ed25519", "rsa"):
        raise SystemExit("Unsupported --key-type. Use 'ed25519' or 'rsa'.")
    if cfg is None:
        cfg = SandboxConfig()

    db = _open_db(cfg)
    try:
        manager = SSHManager(scope=scope, db=db)
        result = manager.init(key_type=key_type, comment=comment, force=force)
        print(f"SSH key ready for scope '{sanitize_tty(scope)}':")
        print(f"  id:          {result['key_id']}")
        print(f"  type:        {sanitize_tty(result['key_type'])}")
        print(f"  fingerprint: {sanitize_tty(result['fingerprint'])}")
        print(f"  comment:     {sanitize_tty(result['comment'])}")
        print("Public key (register as a deploy key):")
        print(f"  {sanitize_tty(result['public_line'])}")
    finally:
        db.close()


def _handle_ssh_export(
    *,
    scope: str,
    out_dir: str,
    key_id: int | None = None,
    out_name: str | None = None,
    cfg: SandboxConfig | None = None,
) -> None:
    """Write a scope's key back to a standard OpenSSH file pair."""
    from ..vault.ssh.keypair import export_ssh_keypair

    _validate_scope_name(scope)
    if cfg is None:
        cfg = SandboxConfig()

    db = _open_db(cfg)
    try:
        try:
            result = export_ssh_keypair(
                db,
                scope,
                Path(out_dir).expanduser(),
                key_id=key_id,
                out_name=out_name,
            )
        except ValueError as exc:
            raise SystemExit(str(exc)) from exc
        except FileExistsError as exc:
            raise SystemExit(f"Refusing to overwrite existing file: {exc.filename}") from exc

        print(f"Exported key id={result.key_id} ({sanitize_tty(result.fingerprint)}):")
        print(f"  private key: {sanitize_tty(str(result.private_path))}")
        print(f"  public key:  {sanitize_tty(str(result.public_path))}")
    finally:
        db.close()


def _handle_ssh_pub(
    *,
    scope: str,
    key_id: int | None = None,
    all_keys: bool = False,
    cfg: SandboxConfig | None = None,
) -> None:
    """Print a scope's public key line(s) to stdout.

    Default: the most recently assigned key — the one likely to be the
    primary deploy key.  ``--all`` prints every key assigned to the scope
    (one per line, newest last); ``--key-id`` targets a specific row.
    """
    from ..vault.ssh.keypair import public_line_of

    _validate_scope_name(scope)
    if cfg is None:
        cfg = SandboxConfig()

    if all_keys and key_id is not None:
        raise SystemExit("--all and --key-id are mutually exclusive")

    db = _open_db(cfg)
    try:
        records = db.load_ssh_keys_for_scope(scope)
        if not records:
            raise SystemExit(f"scope {scope!r} has no SSH keys assigned")
        if all_keys:
            for record in records:
                print(public_line_of(record))
            return
        if key_id is None:
            record = records[-1]
        else:
            matches = [r for r in records if r.id == key_id]
            if not matches:
                raise SystemExit(f"key_id {key_id} is not assigned to scope {scope!r}")
            record = matches[0]
        print(public_line_of(record))
    finally:
        db.close()


def _handle_ssh_link(
    *,
    key_id: int,
    scope: str,
    cfg: SandboxConfig | None = None,
) -> None:
    """Assign an already-stored ssh_keys row to an additional scope.

    The inverse of ``ssh remove`` — adds a row in ``ssh_key_assignments``
    linking *scope* to *key_id*.  Idempotent: re-linking an existing
    pair is a no-op.  Useful when several projects legitimately share a
    single deploy key.
    """
    _validate_scope_name(scope)
    if cfg is None:
        cfg = SandboxConfig()

    db = _open_db(cfg)
    try:
        # Existence check up front so a stale id surfaces here rather
        # than as a raw foreign-key error from assign_ssh_key.
        key_exists = db._conn.execute(  # noqa: SLF001 — one-shot read
            "SELECT 1 FROM ssh_keys WHERE id = ?", (key_id,)
        ).fetchone()
        if not key_exists:
            raise SystemExit(f"No ssh_keys row with id={key_id}")

        already_linked = db._conn.execute(  # noqa: SLF001
            "SELECT 1 FROM ssh_key_assignments WHERE scope = ? AND key_id = ?",
            (scope, key_id),
        ).fetchone()
        if already_linked:
            print(f"Scope '{sanitize_tty(scope)}' is already linked to key id={key_id}")
            return

        db.assign_ssh_key(scope, key_id)
        print(f"Linked key id={key_id} to scope '{sanitize_tty(scope)}'")
    finally:
        db.close()


def _handle_ssh_rename(
    *,
    fingerprint: str,
    comment: str,
    cfg: SandboxConfig | None = None,
) -> None:
    """Change the comment of an existing SSH key, identified by fingerprint prefix.

    The comment lives on the ``ssh_keys`` row, so renaming applies across
    every scope that key is linked to.  Ambiguous prefixes that match
    more than one distinct fingerprint print the candidates and exit
    without writing anything.
    """
    from ..vault.store.db import UnsafeCommentError

    if cfg is None:
        cfg = SandboxConfig()

    matches = _filter_key_rows(_build_key_rows(cfg), fingerprint=fingerprint)
    distinct = {r.fingerprint for r in matches}
    if not distinct:
        raise SystemExit(f"No SSH key matches fingerprint prefix {fingerprint!r}.")
    if len(distinct) > 1:
        print(f"Ambiguous fingerprint prefix {fingerprint!r} matched {len(distinct)} keys:\n")
        _print_key_table(matches)
        raise SystemExit("Refine the prefix to a single key.")

    full_fp = distinct.pop()
    db = _open_db(cfg)
    try:
        try:
            db.set_ssh_key_comment(full_fp, comment)
        except UnsafeCommentError as exc:
            raise SystemExit(f"Invalid comment: {exc}") from exc
    finally:
        db.close()

    print(f"Renamed {full_fp} → {sanitize_tty(comment)!r}")


def _handle_ssh_remove(
    *,
    scope: str | None = None,
    comment: str | None = None,
    fingerprint: str | None = None,
    yes: bool = False,
    cfg: SandboxConfig | None = None,
) -> None:
    """Unassign SSH keys from their scope(s); cascade-delete orphaned key rows.

    Two modes: interactive selection when called without filters, or
    direct matching when any of ``--scope``, ``--comment``, or
    ``--fingerprint`` is provided.
    """
    if cfg is None:
        cfg = SandboxConfig()

    all_rows = _build_key_rows(cfg)
    if not all_rows:
        raise SystemExit("No SSH keys registered.")

    has_filters = any((scope, comment, fingerprint))

    if has_filters:
        candidates = _filter_key_rows(
            all_rows,
            scope=scope,
            comment=comment,
            fingerprint=fingerprint,
        )
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

    db = _open_db(cfg)
    try:
        for row in candidates:
            db.unassign_ssh_key(row.scope, _key_id_from_row(row))
    finally:
        db.close()

    n = len(candidates)
    # "Unassigned" not "removed" — a key still linked to another scope survives.
    print(f"Unassigned {n} key{'s' if n != 1 else ''} from their scope(s).")


SSH_COMMANDS: tuple[CommandDef, ...] = (
    CommandDef(
        name="list",
        help="List SSH keys stored in the vault",
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
        help="Import an OpenSSH keypair from files into the vault DB",
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
                help="Path to the .pub file (default: derive from the private key)",
                default=None,
                dest="public_key",
            ),
            ArgDef(
                name="--comment",
                help="Override the key's comment string",
                default=None,
            ),
        ),
    ),
    CommandDef(
        name="add",
        help="Generate a new SSH keypair in the vault for a credential scope",
        handler=_handle_ssh_add,
        group="ssh",
        args=(
            ArgDef(name="scope", help="Credential scope to associate the key with"),
            ArgDef(
                name="--key-type",
                help="Key algorithm: ed25519 (default) or rsa",
                default="ed25519",
                dest="key_type",
            ),
            ArgDef(
                name="--comment",
                help="Comment embedded in the public key (default: tk-main:<scope>)",
                default=None,
            ),
            ArgDef(
                name="--force",
                help="Rotate — unassign all existing keys from the scope and generate fresh",
                action="store_true",
            ),
        ),
    ),
    CommandDef(
        name="export",
        help="Export a scope's SSH keypair to standard OpenSSH files",
        handler=_handle_ssh_export,
        group="ssh",
        args=(
            ArgDef(name="scope", help="Credential scope to export"),
            ArgDef(
                name="--out-dir",
                help="Directory to write files into",
                dest="out_dir",
                required=True,
            ),
            ArgDef(
                name="--key-id",
                help="Export a specific ssh_keys.id (default: most recently added)",
                default=None,
                dest="key_id",
                type=int,
            ),
            ArgDef(
                name="--out-name",
                help="Override the output filename stem (default: id_<type>_<fp8>)",
                default=None,
                dest="out_name",
            ),
        ),
    ),
    CommandDef(
        name="pub",
        help="Print a scope's public key to stdout",
        handler=_handle_ssh_pub,
        group="ssh",
        args=(
            ArgDef(name="scope", help="Credential scope"),
            ArgDef(
                name="--key-id",
                help="Specific ssh_keys.id (default: most recently added)",
                default=None,
                dest="key_id",
                type=int,
            ),
            ArgDef(
                name="--all",
                help="Print every key assigned to the scope, one per line",
                action="store_true",
                dest="all_keys",
            ),
        ),
    ),
    CommandDef(
        name="link",
        help="Link an existing vault key to an additional scope",
        handler=_handle_ssh_link,
        group="ssh",
        args=(
            ArgDef(name="scope", help="Credential scope to link the key to"),
            ArgDef(
                name="--key-id",
                help="ssh_keys.id of the key already stored in the vault",
                dest="key_id",
                type=int,
                required=True,
            ),
        ),
    ),
    CommandDef(
        name="rename",
        help="Change the comment of a stored SSH key (selected by fingerprint prefix)",
        handler=_handle_ssh_rename,
        group="ssh",
        args=(
            ArgDef(
                name="fingerprint",
                help="Fingerprint prefix identifying the key (min 8 chars recommended)",
            ),
            ArgDef(
                name="comment",
                help="New comment text",
            ),
        ),
    ),
    CommandDef(
        name="remove",
        help="Unassign SSH keys from scopes (orphaned keys cascade-delete)",
        handler=_handle_ssh_remove,
        group="ssh",
        args=(
            ArgDef(
                name="--scope",
                help="Filter by credential scope (exact match)",
                default=None,
            ),
            ArgDef(
                name="--comment",
                help="Filter by comment (supports glob wildcards)",
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
        ),
    ),
)


__all__ = ["SSH_COMMANDS"]
