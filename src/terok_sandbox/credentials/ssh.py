# SPDX-FileCopyrightText: 2025 Jiri Vyskocil
# SPDX-License-Identifier: Apache-2.0

"""SSH keypair generation and config directory setup.

Manages the lifecycle of SSH keypairs and config for sandboxed containers.
:class:`SSHManager` generates keys, renders the SSH config from a template,
and sets permissions so that the container's ``/home/dev/.ssh`` mount works
correctly.

All constructor parameters are plain values (strings, paths) — no
terok-specific types.  The orchestration layer constructs the manager
from project configuration.
"""

import fcntl
import json
import os
import re
import subprocess
from importlib import resources
from pathlib import Path
from typing import TypedDict

from .._util import effective_ssh_key_name, ensure_dir_writable, render_template
from ..config import SandboxConfig


def generate_keypair(key_type: str, priv_path: Path, pub_path: Path, comment: str) -> None:
    """Generate an SSH keypair via ``ssh-keygen``.

    Removes any stale half-existing files first, then invokes
    ``ssh-keygen`` with the given *comment* embedded in the public key.
    """
    for p in (priv_path, pub_path):
        p.unlink(missing_ok=True)

    cmd = [
        "ssh-keygen",
        "-t",
        key_type,
        "-f",
        str(priv_path),
        "-N",
        "",
        "-C",
        comment,
    ]
    try:
        subprocess.run(cmd, check=True)
    except FileNotFoundError:
        raise SystemExit("ssh-keygen not found. Please install OpenSSH client tools.")
    except subprocess.CalledProcessError as e:
        raise SystemExit(f"ssh-keygen failed: {e}")


class SSHInitResult(TypedDict):
    """Result of SSH directory initialization."""

    dir: str
    private_key: str
    public_key: str
    config_path: str
    key_name: str


class SSHManager:
    """SSH keypair generation and config directory management.

    Handles the full SSH setup lifecycle: directory creation, keypair
    generation (ed25519 or RSA), config file rendering from templates, and
    permission hardening.  Keys are stored under ``ssh_keys_dir/<project>``
    and used by the credential proxy's SSH agent for container access.
    """

    def __init__(
        self,
        *,
        project_id: str,
        ssh_host_dir: Path | str | None = None,
        ssh_key_name: str | None = None,
        ssh_config_template: Path | str | None = None,
    ) -> None:
        """Initialize with plain parameters.

        Parameters
        ----------
        project_id:
            Identifier used for key naming and directory layout.
        ssh_host_dir:
            Explicit SSH directory (overrides default ``<ssh_keys_dir>/<id>``).
        ssh_key_name:
            Explicit key filename (overrides derived ``id_<type>_<id>``).
        ssh_config_template:
            Path to a user-provided SSH config template file.
        """
        self._project_id = project_id
        self._ssh_host_dir = Path(ssh_host_dir) if ssh_host_dir else None
        self._ssh_key_name = ssh_key_name
        self._ssh_config_template = Path(ssh_config_template) if ssh_config_template else None

    @property
    def key_name(self) -> str:
        """Return the effective SSH key name."""
        return effective_ssh_key_name(self._project_id, ssh_key_name=self._ssh_key_name)

    def init(
        self,
        key_type: str = "ed25519",
        key_name: str | None = None,
        force: bool = False,
    ) -> SSHInitResult:
        """Initialize the SSH directory and generate a keypair.

        Location resolution:
          - If *ssh_host_dir* was provided, use that path.
          - Otherwise: ``<ssh_keys_dir>/<project_id>``

        Key name defaults to ``id_<type>_<project_id>`` (e.g. ``id_ed25519_proj``).
        """
        if key_type not in ("ed25519", "rsa"):
            raise SystemExit("Unsupported --key-type. Use 'ed25519' or 'rsa'.")

        target_dir = self._ssh_host_dir or (SandboxConfig().ssh_keys_dir / self._project_id)
        target_dir = Path(target_dir).expanduser().resolve()
        ensure_dir_writable(target_dir, "SSH host dir")

        if not key_name:
            key_name = effective_ssh_key_name(
                self._project_id, ssh_key_name=self._ssh_key_name, key_type=key_type
            )

        # Reject path-like or reserved key names
        _RESERVED_NAMES = {"config", "known_hosts", "authorized_keys"}
        key_path = Path(key_name)
        if key_path.is_absolute() or ".." in key_path.parts or "/" in key_name or "\\" in key_name:
            raise SystemExit(
                f"Invalid SSH key name {key_name!r}: must be a plain filename, "
                "not an absolute path or traversal sequence"
            )
        if key_name.lower() in _RESERVED_NAMES:
            raise SystemExit(
                f"Invalid SSH key name {key_name!r}: collides with reserved "
                f"filename (reserved: {', '.join(sorted(_RESERVED_NAMES))})"
            )

        priv_path = target_dir / key_name
        pub_path = target_dir / f"{key_name}.pub"
        cfg_path = target_dir / "config"

        # Refuse to reuse artifacts that are symlinks or non-regular files
        for p in (priv_path, pub_path, cfg_path):
            if p.exists() or p.is_symlink():
                if p.is_symlink() or not p.is_file():
                    raise SystemExit(
                        f"Refusing to use {p}: expected a regular file but found "
                        f"{'a symlink' if p.is_symlink() else 'a non-regular file'}. "
                        "Remove it manually and retry."
                    )

        if force or not priv_path.exists() or not pub_path.exists():
            self._generate_keypair(key_type, priv_path, pub_path, self._project_id)

        if force or not cfg_path.exists():
            self._render_config(
                cfg_path, key_name, priv_path, self._project_id, self._ssh_config_template
            )

        try:
            _harden_permissions(target_dir, priv_path, pub_path, cfg_path)
        except OSError as e:
            raise SystemExit(f"Failed to set SSH directory permissions on {target_dir}: {e}") from e
        _print_init_summary(target_dir, priv_path, pub_path, cfg_path)
        return SSHInitResult(
            dir=str(target_dir),
            private_key=str(priv_path),
            public_key=str(pub_path),
            config_path=str(cfg_path),
            key_name=key_name,
        )

    @staticmethod
    def _generate_keypair(key_type: str, priv_path: Path, pub_path: Path, project_id: str) -> None:
        """Generate an SSH keypair, removing any stale half-existing files first."""
        generate_keypair(key_type, priv_path, pub_path, f"tk-main:{project_id}")

    @staticmethod
    def _render_config(
        cfg_path: Path,
        key_name: str,
        priv_path: Path,
        project_id: str,
        config_template: Path | None = None,
    ) -> None:
        """Render the SSH config from a user or packaged template."""
        variables = {
            "KEY_NAME": key_name,
            "IDENTITY_FILE": str(priv_path),
            "PROJECT_ID": project_id,
        }
        user_config = _try_render_user_template(config_template, variables)
        config_text = (
            user_config if user_config is not None else _try_render_packaged_template(variables)
        )
        if config_text is None:
            raise SystemExit(
                "Failed to render SSH config: no valid template. "
                "Ensure a project ssh.config_template is set or the packaged template exists."
            )
        try:
            cfg_path.write_text(config_text)
        except Exception as e:
            raise SystemExit(f"Failed to write SSH config at {cfg_path}: {e}")


# ---------------------------------------------------------------------------
# Module-private helpers (extracted to reduce cognitive complexity)
# ---------------------------------------------------------------------------


def _try_render_user_template(template_path: Path | None, variables: dict[str, str]) -> str | None:
    """Render the user-provided SSH config template, if configured.

    Raises ``SystemExit`` if the template path is configured but the file
    is missing or rendering fails — explicit misconfiguration should fail
    fast rather than silently falling back to the packaged template.
    """
    if not template_path:
        return None
    p = Path(template_path)
    if not p.is_file():
        raise SystemExit(f"SSH config template not found: {p}")
    try:
        return render_template(p, variables)
    except Exception as exc:
        raise SystemExit(f"Failed to render SSH config template {p}: {exc}") from exc


def _try_render_packaged_template(variables: dict[str, str]) -> str | None:
    """Attempt to render the bundled SSH config template from package resources."""
    try:
        raw = (
            resources.files("terok_sandbox") / "resources" / "templates" / "ssh_config.template"
        ).read_text()
    except Exception:
        return None
    for k, v in variables.items():
        raw = raw.replace(f"{{{{{k}}}}}", v)
    return raw


def update_ssh_keys_json(keys_json_path: Path, project_id: str, result: SSHInitResult) -> None:
    """Update the SSH key mapping JSON with a project's key paths.

    The JSON file maps project IDs to their SSH key file paths, similar
    to how ``routes.json`` maps provider names to proxy routes.  The
    credential proxy's SSH agent handler reads this file to locate the
    private key for signing requests.

    Key management rules (keyed by ``private_key`` path):

    - **No existing entry**: write a single-dict entry (simple case).
    - **Same private_key path**: replace in-place (idempotent re-run of ``ssh-init``).
    - **Different private_key path**: expand to / append to a list, so a project can
      hold multiple independent SSH keys (e.g. GitHub + GitLab).

    Uses ``fcntl.flock`` to prevent concurrent ``ssh-init`` invocations
    from corrupting the file.
    """
    new_entry: dict[str, str] = {
        "private_key": result["private_key"],
        "public_key": result["public_key"],
    }
    keys_json_path.parent.mkdir(parents=True, exist_ok=True)
    fd = os.open(str(keys_json_path), os.O_RDWR | os.O_CREAT, 0o600)
    try:
        fcntl.flock(fd, fcntl.LOCK_EX)
        chunks: list[bytes] = []
        while chunk := os.read(fd, 8192):
            chunks.append(chunk)
        raw = b"".join(chunks)
        mapping: dict = json.loads(raw) if raw.strip() else {}
        entries: list[dict[str, str]] = mapping.get(project_id) or []
        if not isinstance(entries, list):
            entries = []
        for i, entry in enumerate(entries):
            if isinstance(entry, dict) and entry.get("private_key") == new_entry["private_key"]:
                entries[i] = new_entry  # same path — idempotent update
                break
        else:
            entries.append(new_entry)  # new path — append
        mapping[project_id] = entries
        data = (json.dumps(mapping, indent=2) + "\n").encode("utf-8")
        os.lseek(fd, 0, os.SEEK_SET)
        os.ftruncate(fd, 0)
        os.write(fd, data)
    finally:
        fcntl.flock(fd, fcntl.LOCK_UN)
        os.close(fd)


def _harden_permissions(target_dir: Path, priv_path: Path, pub_path: Path, cfg_path: Path) -> None:
    """Set restrictive permissions on the SSH directory and key files.

    Raises ``OSError`` if any chmod operation fails.
    """
    os.chmod(target_dir, 0o700)
    if priv_path.exists():
        os.chmod(priv_path, 0o600)
    if pub_path.exists():
        os.chmod(pub_path, 0o644)
    if cfg_path.exists():
        os.chmod(cfg_path, 0o644)


def _next_key_number(project_dir: Path, algo: str) -> int:
    """Scan *project_dir* for ``id_<algo>_key-N`` files and return the next *N*.

    Returns 1 when no numbered keys exist yet.  Uses max+1 (no gap-filling)
    so that retired key numbers are never reused.
    """
    pattern = re.compile(rf"^id_{re.escape(algo)}_key-(\d+)$")
    max_n = 0
    if project_dir.is_dir():
        for entry in project_dir.iterdir():
            if m := pattern.match(entry.name):
                max_n = max(max_n, int(m.group(1)))
    return max_n + 1


def _print_init_summary(target_dir: Path, priv_path: Path, pub_path: Path, cfg_path: Path) -> None:
    """Print a human-readable summary of the initialized SSH directory."""
    print("SSH directory initialized:")
    print(f"  dir:         {target_dir}")
    print(f"  private key: {priv_path}")
    print(f"  public key:  {pub_path}")
    print(f"  config:      {cfg_path}")
    try:
        if pub_path.exists():
            pub_key_text = pub_path.read_text(encoding="utf-8", errors="ignore").strip()
            if pub_key_text:
                print("Public key:")
                print(f"  {pub_key_text}")
    except Exception:
        pass
