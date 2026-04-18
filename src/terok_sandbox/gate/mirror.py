# SPDX-FileCopyrightText: 2025 Jiri Vyskocil
# SPDX-License-Identifier: Apache-2.0

"""Host-side git gate (mirror) management and upstream comparison.

The git gate is a bare mirror of the upstream repository stored on the host.
In **gatekeeping mode**, it is the *only* repository the container can access,
enforcing human review before changes reach upstream.  In **online mode**, it
serves as a read-only clone accelerator (faster than cloning over the network).

:class:`GitGate` is the main service class — wraps git CLI operations for
syncing, comparing, and querying the mirror.

All constructor parameters are plain values (strings, paths) — no
terok-specific types like ``ProjectConfig``.

Value types returned by ``GitGate`` methods:

- :class:`GateSyncResult` — full sync outcome (created, updated branches, errors)
- :class:`BranchSyncResult` — selective branch sync outcome
- :class:`CommitInfo` — single commit metadata (hash, date, author, message)
- :class:`GateStalenessInfo` — frozen comparison of gate HEAD vs upstream HEAD
"""

import logging
import os
import shlex
import shutil
import subprocess
from collections.abc import Callable
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import TypedDict

from .._util import effective_ssh_key_name

logger = logging.getLogger(__name__)

# ---------- Vocabulary ----------


class GateSyncResult(TypedDict):
    """Result of a full gate sync operation."""

    path: str
    upstream_url: str
    created: bool
    success: bool
    updated_branches: list[str]
    errors: list[str]
    cache_refreshed: bool


class BranchSyncResult(TypedDict):
    """Result of a branch sync operation."""

    success: bool
    updated_branches: list[str]
    errors: list[str]


class CommitInfo(TypedDict):
    """Information about a single git commit."""

    commit_hash: str
    commit_date: str
    commit_message: str
    commit_author: str


@dataclass(frozen=True)
class GateStalenessInfo:
    """Result of comparing gate vs upstream."""

    branch: str | None
    gate_head: str | None
    upstream_head: str | None
    is_stale: bool
    commits_behind: int | None  # None if couldn't determine
    commits_ahead: int | None  # None if couldn't determine
    last_checked: str  # ISO timestamp
    error: str | None


# ---------- GitGate class (Repository + Gateway pattern) ----------


class GitGate:
    """Repository + Gateway for a host-side git gate mirror.

    Manages the bare git mirror that containers clone from.  Provides
    operations for initial creation, incremental sync from upstream,
    selective branch fetching, and staleness detection.

    Constructor takes plain parameters — no terok-specific types.
    """

    def __init__(
        self,
        *,
        scope: str,
        gate_path: Path | str,
        upstream_url: str | None = None,
        default_branch: str | None = None,
        ssh_host_dir: Path | str | None = None,
        ssh_key_name: str | None = None,
        allow_host_keys: bool = False,
        validate_gate_fn: Callable[[str], None] | None = None,
        clone_cache_base: Path | str | None = None,
    ) -> None:
        """Initialise with plain parameters.

        Parameters
        ----------
        scope:
            Credential scope for this gate's owner.
        gate_path:
            Path to the bare git mirror on the host.
        upstream_url:
            Git upstream URL to sync from.
        default_branch:
            Branch name used for staleness comparisons.
        ssh_host_dir:
            Explicit SSH directory for git operations.  When ``None``,
            falls back to ``SandboxConfig().ssh_keys_dir / scope``.
        ssh_key_name:
            Explicit SSH key filename.
        allow_host_keys:
            When ``True``, fall back to the user's ``~/.ssh`` keys and
            SSH agent if no terok-managed key is found.  Default
            ``False`` — blocks host-key probing to prevent accidental
            exposure of personal credentials.
        validate_gate_fn:
            Optional callback ``(scope) -> None`` that validates no other
            scope uses the same gate with a different upstream.  Injected by
            the orchestration layer; omitted for standalone use.
        clone_cache_base:
            Base directory for non-bare clone caches.  When set,
            :meth:`sync` refreshes a working-tree cache at
            ``clone_cache_base / scope`` after updating the bare mirror.
            The cache accelerates task startup by enabling a host-side
            file copy instead of a full ``git clone``.
        """
        self._scope = scope
        self._gate_path = Path(gate_path)
        self._upstream_url = upstream_url
        self._default_branch = default_branch
        self._ssh_host_dir = Path(ssh_host_dir) if ssh_host_dir else None
        self._ssh_key_name = ssh_key_name
        self._allow_host_keys = allow_host_keys
        self._validate_gate_fn = validate_gate_fn
        self._clone_cache_base = Path(clone_cache_base) if clone_cache_base else None

    @property
    def cache_path(self) -> Path | None:
        """Clone cache directory for this scope, or ``None`` if caching is disabled."""
        return (self._clone_cache_base / self._scope) if self._clone_cache_base else None

    def _ssh_env(self) -> dict:
        """Return a subprocess env dict with SSH configuration."""
        return _git_env_with_ssh(
            scope=self._scope,
            ssh_host_dir=self._ssh_host_dir,
            ssh_key_name=self._ssh_key_name,
            allow_host_keys=self._allow_host_keys,
        )

    def _validate_gate(self) -> None:
        """Run the injected gate validation callback, if any."""
        if self._validate_gate_fn:
            self._validate_gate_fn(self._scope)

    def sync(
        self,
        branches: list[str] | None = None,
        force_reinit: bool = False,
    ) -> GateSyncResult:
        """Sync the host-side git mirror gate.

        - Uses SSH configuration via GIT_SSH_COMMAND.
        - If gate doesn't exist (or *force_reinit*), performs a fresh ``git clone --mirror``.
        - Always runs the sync logic afterward for consistent side effects.

        Returns:
            Dict with keys: path, upstream_url, created (bool), success,
            updated_branches, errors.
        """
        if not self._upstream_url:
            raise SystemExit("Project has no git.upstream_url configured")

        self._validate_gate()

        gate_dir = self._gate_path
        gate_exists = gate_dir.exists()
        gate_dir.parent.mkdir(parents=True, exist_ok=True)

        env = self._ssh_env()
        created = False
        if force_reinit and gate_exists:
            try:
                if gate_dir.is_dir():
                    shutil.rmtree(gate_dir)
            except Exception as exc:
                logger.warning(f"Failed to remove gate dir {gate_dir}: {exc}")
            gate_exists = False

        if not gate_exists:
            _clone_gate_mirror(self._upstream_url, gate_dir, env)
            created = True

        sync_result = self.sync_branches(branches)

        # Refresh the non-bare clone cache from the bare mirror (best-effort).
        cache_refreshed = False
        if sync_result["success"] and self._clone_cache_base:
            cache_refreshed = self._refresh_clone_cache()

        return {
            "path": str(gate_dir),
            "upstream_url": self._upstream_url,
            "created": created,
            "success": sync_result["success"],
            "updated_branches": sync_result["updated_branches"],
            "errors": sync_result["errors"],
            "cache_refreshed": cache_refreshed,
        }

    def _refresh_clone_cache(self) -> bool:
        """Refresh the non-bare clone cache from the local bare mirror.

        Creates the cache via ``git clone`` if it doesn't exist, or
        fetches updates if it does.  Returns ``True`` on success.
        Failures are logged and swallowed — the cache is purely an
        optimization for faster task startup.
        """
        cache_dir = self.cache_path
        if cache_dir is None:
            return False

        gate_file_url = self._gate_path.resolve().as_uri()
        try:
            cache_dir.parent.mkdir(parents=True, exist_ok=True)

            if not cache_dir.exists():
                logger.info("Creating clone cache at %s", cache_dir)
                subprocess.run(
                    ["git", "clone", gate_file_url, str(cache_dir)],
                    check=True,
                    capture_output=True,
                    timeout=300,
                )
            else:
                # Ensure origin points to current bare mirror
                subprocess.run(
                    ["git", "-C", str(cache_dir), "remote", "set-url", "origin", gate_file_url],
                    check=True,
                    capture_output=True,
                    timeout=10,
                )
                subprocess.run(
                    ["git", "-C", str(cache_dir), "fetch", "--all", "--prune"],
                    check=True,
                    capture_output=True,
                    timeout=120,
                )
                # Update working tree to match fetched HEAD — the cache is
                # copied as-is into task workspaces, so stale files matter.
                subprocess.run(
                    ["git", "-C", str(cache_dir), "reset", "--hard", "origin/HEAD"],
                    check=True,
                    capture_output=True,
                    timeout=30,
                )
                # Remove untracked/ignored files so the cache stays pristine.
                subprocess.run(
                    ["git", "-C", str(cache_dir), "clean", "-ffdx"],
                    check=True,
                    capture_output=True,
                    timeout=30,
                )
            return True
        except (subprocess.CalledProcessError, subprocess.TimeoutExpired, OSError) as exc:
            logger.warning("Clone cache refresh failed (non-fatal): %s", exc)
            return False

    def sync_branches(self, branches: list[str] | None = None) -> BranchSyncResult:
        """Sync specific branches in the gate from upstream.

        Args:
            branches: List of branches to sync (default: all via remote update)

        Returns:
            Dict with keys: success, updated_branches, errors
        """
        gate_dir = self._gate_path

        if not gate_dir.exists():
            return {"success": False, "updated_branches": [], "errors": ["Gate not initialized"]}

        self._validate_gate()

        env = self._ssh_env()
        errors: list[str] = []
        updated: list[str] = []

        try:
            cmd = ["git", "-C", str(gate_dir), "remote", "update", "--prune"]
            result = subprocess.run(cmd, capture_output=True, text=True, env=env, timeout=120)

            if result.returncode != 0:
                errors.append(f"remote update failed: {result.stderr}")
            else:
                updated = branches if branches else ["all"]

        except subprocess.TimeoutExpired:
            errors.append("Sync timed out")
        except Exception as e:
            errors.append(str(e))

        return {"success": len(errors) == 0, "updated_branches": updated, "errors": errors}

    def compare_vs_upstream(self, branch: str | None = None) -> GateStalenessInfo:
        """Compare gate HEAD vs upstream HEAD for a branch.

        Args:
            branch: Branch to compare (default: configured default_branch)

        Returns:
            GateStalenessInfo with comparison results
        """
        branch = branch or self._default_branch
        now = datetime.now().isoformat()

        if not branch:
            return GateStalenessInfo(
                branch=None,
                gate_head=None,
                upstream_head=None,
                is_stale=False,
                commits_behind=None,
                commits_ahead=None,
                last_checked=now,
                error="No branch configured",
            )

        env = self._ssh_env()

        # Get gate HEAD
        gate_head = _get_gate_branch_head(self._gate_path, branch, env)
        if gate_head is None:
            return GateStalenessInfo(
                branch=branch,
                gate_head=None,
                upstream_head=None,
                is_stale=False,
                commits_behind=None,
                commits_ahead=None,
                last_checked=now,
                error="Gate not initialized",
            )

        # Get upstream HEAD
        if not self._upstream_url:
            return GateStalenessInfo(
                branch=branch,
                gate_head=gate_head,
                upstream_head=None,
                is_stale=False,
                commits_behind=None,
                commits_ahead=None,
                last_checked=now,
                error="No upstream URL configured",
            )

        upstream_info = _get_upstream_head(self._upstream_url, branch, env)
        if upstream_info is None:
            return GateStalenessInfo(
                branch=branch,
                gate_head=gate_head,
                upstream_head=None,
                is_stale=False,
                commits_behind=None,
                commits_ahead=None,
                last_checked=now,
                error="Could not reach upstream",
            )

        upstream_head = upstream_info["commit_hash"]
        is_stale = gate_head != upstream_head

        commits_behind = None
        commits_ahead = None
        if is_stale:
            commits_behind = _count_commits_range(self._gate_path, gate_head, upstream_head, env)
            commits_ahead = _count_commits_range(self._gate_path, upstream_head, gate_head, env)

        return GateStalenessInfo(
            branch=branch,
            gate_head=gate_head,
            upstream_head=upstream_head,
            is_stale=is_stale,
            commits_behind=commits_behind if is_stale else 0,
            commits_ahead=commits_ahead if is_stale else 0,
            last_checked=now,
            error=None,
        )

    def last_commit(self) -> CommitInfo | None:
        """Get information about the last commit on the configured branch.

        Returns ``None`` if the gate doesn't exist or is not accessible.
        """
        try:
            gate_dir = self._gate_path

            if not gate_dir.exists() or not gate_dir.is_dir():
                return None

            env = self._ssh_env()

            rev = f"refs/heads/{self._default_branch}" if self._default_branch else "HEAD"
            cmd = [
                "git",
                "-C",
                str(gate_dir),
                "log",
                "-1",
                rev,
                "--pretty=format:%H%x00%ad%x00%an%x00%s",
                "--date=iso",
            ]

            result = subprocess.run(cmd, capture_output=True, text=True, env=env)
            if result.returncode != 0 and self._default_branch:
                cmd[5] = "HEAD"
                result = subprocess.run(cmd, capture_output=True, text=True, env=env)
            if result.returncode != 0:
                return None

            parts = result.stdout.strip().split("\x00", 3)
            if len(parts) == 4:
                return {
                    "commit_hash": parts[0],
                    "commit_date": parts[1],
                    "commit_author": parts[2],
                    "commit_message": parts[3],
                }
            return None

        except Exception:
            return None


# ---------- Private helpers ----------


def _git_env_with_ssh(
    *,
    scope: str,
    ssh_host_dir: Path | None,
    ssh_key_name: str | None,
    allow_host_keys: bool = False,
) -> dict:
    """Return an env that forces git to use the scope's SSH key directly.

    Builds ``GIT_SSH_COMMAND`` from the private key file — no SSH config file
    required.  The vault handles container-side SSH auth; this
    helper only covers host-side gate operations (clone, fetch).

    When the key file is not found and *allow_host_keys* is ``False``
    (default), SSH is configured to reject all identities so that the
    user's personal ``~/.ssh`` keys are never tried silently.
    """
    from ..config import SandboxConfig

    env = os.environ.copy()
    ssh_dir = ssh_host_dir or (SandboxConfig().ssh_keys_dir / scope)
    eff_name = effective_ssh_key_name(scope, ssh_key_name=ssh_key_name, key_type="ed25519")
    key_path = Path(ssh_dir) / eff_name
    if key_path.is_file():
        ssh_cmd = [
            "ssh",
            "-o",
            "IdentitiesOnly=yes",
            "-o",
            f"IdentityFile={key_path}",
            "-o",
            "StrictHostKeyChecking=no",
        ]
        env["GIT_SSH_COMMAND"] = shlex.join(ssh_cmd)
        env["SSH_AUTH_SOCK"] = ""
    elif allow_host_keys:
        pass  # unmodified env — host SSH agent + ~/.ssh keys allowed
    else:
        # Block host-key probing: IdentitiesOnly with no IdentityFile.
        logger.warning("SSH key not found at %s — host keys blocked", key_path)
        ssh_cmd = ["ssh", "-o", "IdentitiesOnly=yes", "-o", "StrictHostKeyChecking=no"]
        env["GIT_SSH_COMMAND"] = shlex.join(ssh_cmd)
        env["SSH_AUTH_SOCK"] = ""
    return env


def _clone_gate_mirror(upstream_url: str, gate_dir: Path, env: dict) -> None:
    """Clone the upstream repository as a bare mirror into *gate_dir*."""
    cmd = ["git", "clone", "--mirror", upstream_url, str(gate_dir)]
    try:
        subprocess.run(cmd, check=True, env=env)
    except FileNotFoundError:
        raise SystemExit("git not found on host; please install git")
    except subprocess.CalledProcessError as e:
        raise SystemExit(f"git clone --mirror failed: {e}")


def _get_upstream_head(upstream_url: str, branch: str, env: dict) -> dict | None:
    """Query upstream HEAD ref using git ls-remote (cheap, no object download).

    Returns:
        Dict with keys: commit_hash, ref_name, upstream_url
        or None if query fails.
    """
    try:
        cmd = ["git", "ls-remote", upstream_url, f"refs/heads/{branch}"]
        result = subprocess.run(cmd, capture_output=True, text=True, env=env, timeout=30)

        if result.returncode != 0:
            return None

        line = result.stdout.strip()
        if not line:
            return None

        parts = line.split("\t")
        if len(parts) >= 2:
            return {
                "commit_hash": parts[0],
                "ref_name": parts[1],
                "upstream_url": upstream_url,
            }
        return None
    except (
        subprocess.CalledProcessError,
        subprocess.TimeoutExpired,
        FileNotFoundError,
        OSError,
    ) as exc:
        logger.debug(f"_get_upstream_head({branch}) failed: {exc}")
        return None


def _get_gate_branch_head(gate_dir: Path, branch: str, env: dict) -> str | None:
    """Get the commit hash for a specific branch in the gate.

    Returns:
        Commit hash string or None if not found.
    """
    try:
        if not gate_dir.exists():
            return None

        cmd = ["git", "-C", str(gate_dir), "rev-parse", f"refs/heads/{branch}"]
        result = subprocess.run(cmd, capture_output=True, text=True, env=env)

        if result.returncode == 0:
            return result.stdout.strip()
        return None
    except (
        subprocess.CalledProcessError,
        subprocess.TimeoutExpired,
        FileNotFoundError,
        OSError,
    ) as exc:
        logger.debug(f"_get_gate_branch_head({branch}) failed: {exc}")
        return None


def _count_commits_range(gate_dir: Path, from_ref: str, to_ref: str, env: dict) -> int | None:
    """Count commits reachable from *to_ref* but not from *from_ref*.

    Uses ``git rev-list --count from..to``.  Returns ``None`` when the
    count cannot be determined (e.g. refs not yet fetched).
    """
    try:
        cmd = [
            "git",
            "-C",
            str(gate_dir),
            "rev-list",
            "--count",
            f"{from_ref}..{to_ref}",
        ]
        result = subprocess.run(cmd, capture_output=True, text=True, env=env)
        if result.returncode == 0:
            return int(result.stdout.strip())
        return None
    except (
        subprocess.CalledProcessError,
        subprocess.TimeoutExpired,
        FileNotFoundError,
        OSError,
    ) as exc:
        logger.debug(f"_count_commits_range({from_ref}..{to_ref}) failed: {exc}")
        return None
