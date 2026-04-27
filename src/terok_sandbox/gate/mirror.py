# SPDX-FileCopyrightText: 2025 Jiri Vyskocil
# SPDX-License-Identifier: Apache-2.0

"""Host-side git gate (mirror) management and upstream comparison.

The git gate is a bare git repository stored on the host.  Its role varies
with how the caller configures it:

- **Upstream set, gatekeeping mode** — the gate is a mirror of upstream
  and is the *only* repository the container can access, enforcing human
  review before changes reach upstream.
- **Upstream set, online mode** — the gate mirrors upstream and serves as
  a read-only clone accelerator (faster than cloning over the network).
- **No upstream** — ``sync()`` initialises the gate as a remoteless bare
  repo that the container can still push to.  Nothing is fetched because
  there is no remote; subsequent syncs are no-ops.

[`GitGate`][terok_sandbox.gate.mirror.GitGate] is the main service class — wraps git CLI operations for
syncing, comparing, and querying the gate.

All constructor parameters are plain values (strings, paths) — no
terok-specific types like ``ProjectConfig``.

Value types returned by ``GitGate`` methods:

- [`GateSyncResult`][terok_sandbox.gate.mirror.GateSyncResult] — full sync outcome (created, updated branches,
  errors; ``upstream_url`` is ``None`` for remoteless gates)
- [`BranchSyncResult`][terok_sandbox.gate.mirror.BranchSyncResult] — selective branch sync outcome
- [`CommitInfo`][terok_sandbox.gate.mirror.CommitInfo] — single commit metadata (hash, date, author, message)
- [`GateStalenessInfo`][terok_sandbox.gate.mirror.GateStalenessInfo] — frozen comparison of gate HEAD vs upstream HEAD
"""

import logging
import os
import re
import shlex
import shutil
import sqlite3
import stat
import subprocess
import time
from collections.abc import Callable
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import TypedDict


class GateAuthNotConfigured(RuntimeError):
    """Raised when a scope has no vault key and personal-SSH fallback is not opted in.

    Callers (the ``gate-sync`` CLI dispatch) turn this into a two-door
    remediation hint:

    - generate a terok-managed key with ``terok ssh-init <project>`` and
      register it upstream, or
    - opt in to the user's own ``~/.ssh`` keys with ``--use-personal-ssh``
      (or ``ssh.use_personal: true`` in the project YAML).
    """

    def __init__(self, scope: str) -> None:
        self.scope = scope
        super().__init__(
            f"No SSH key is assigned to scope {scope!r} and personal-SSH "
            "fallback is not enabled.  Either run `terok ssh-init` to "
            "generate one, or pass --use-personal-ssh."
        )


logger = logging.getLogger(__name__)

# ---------- Vocabulary ----------


class GateSyncResult(TypedDict):
    """Result of a full gate sync operation.

    ``upstream_url`` is ``None`` when the gate is initialised without a
    remote — a local-only mirror that the container can push to but that
    never fetches external commits.
    """

    path: str
    upstream_url: str | None
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
        use_personal_ssh: bool = False,
        validate_gate_fn: Callable[[str], None] | None = None,
        clone_cache_base: Path | str | None = None,
    ) -> None:
        """Initialise with plain parameters.

        Parameters
        ----------
        scope:
            Credential scope for this gate's owner.  Used to locate the
            per-scope vault SSH-agent socket.
        gate_path:
            Path to the bare git mirror on the host.
        upstream_url:
            Git upstream URL to sync from.
        default_branch:
            Branch name used for staleness comparisons.
        use_personal_ssh:
            When ``True``, skip the vault socket entirely and let git fall
            through to the user's ``~/.ssh`` keys / loaded agent.  Default
            ``False`` — "terok never touches your real keys" is the advertised
            property.  Opt in per-invocation (``--use-personal-ssh``) or
            per-project (``ssh.use_personal: true`` in project YAML).
        validate_gate_fn:
            Optional callback ``(scope) -> None`` that validates no other
            scope uses the same gate with a different upstream.  Injected by
            the orchestration layer; omitted for standalone use.
        clone_cache_base:
            Base directory for non-bare clone caches.  When set,
            [`sync`][terok_sandbox.gate.mirror.GitGate.sync] refreshes a working-tree cache at
            ``clone_cache_base / scope`` after updating the bare mirror.
            The cache accelerates task startup by enabling a host-side
            file copy instead of a full ``git clone``.
        """
        self._scope = scope
        self._gate_path = Path(gate_path)
        self._upstream_url = upstream_url
        self._default_branch = default_branch
        self._use_personal_ssh = use_personal_ssh
        self._validate_gate_fn = validate_gate_fn
        self._clone_cache_base = Path(clone_cache_base) if clone_cache_base else None

    @property
    def cache_path(self) -> Path | None:
        """Clone cache directory for this scope, or ``None`` if caching is disabled."""
        return (self._clone_cache_base / self._scope) if self._clone_cache_base else None

    def _ssh_env(self) -> dict:
        """Return a subprocess env dict, injecting SSH config only for SSH upstreams.

        HTTPS upstreams don't use SSH at all, so we hand git an unmodified
        env in that case — fetching `GateAuthNotConfigured` on an HTTPS
        project would be absurd.
        """
        if not is_ssh_url(self._upstream_url):
            return os.environ.copy()
        return _git_env_with_ssh(
            scope=self._scope,
            use_personal_ssh=self._use_personal_ssh,
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
        """Sync the host-side git mirror gate from upstream.

        With an upstream configured, clones (or fetches) from it using the
        project's SSH setup.  Without one, initialises a bare repo in place
        and returns a no-op sync — the gate then serves as a local-only
        remote that the container can push to, giving the agent somewhere
        to stage commits even when there is nothing external to mirror.

        A remoteless gate that already exists is a proper no-op: nothing
        re-initialises, and the returned branch list is empty.
        """
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
            if self._upstream_url:
                _clone_gate_mirror(self._upstream_url, gate_dir, env)
            else:
                _init_remoteless_gate(gate_dir)
            created = True

        # A remoteless gate has nothing to fetch — skip ``git remote update``
        # (which would fail on a repo with no origin) and the clone-cache
        # refresh (there is no bare mirror to track).
        if not self._upstream_url:
            return {
                "path": str(gate_dir),
                "upstream_url": None,
                "created": created,
                "success": True,
                "updated_branches": [],
                "errors": [],
                "cache_refreshed": False,
            }

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


# ---------- Public predicates ----------


# scp-style SSH URL: optional ``user@`` prefix, then ``host:path``.  Host
# must be ≥2 chars so Windows ``C:\…`` paths (1-char "host") stay out.
# Any other URL scheme is ruled out before we consult this pattern.
_SCP_SSH_RE = re.compile(r"^(?:[^@/\s:]+@)?[^:/\s]{2,}:.+")


def is_ssh_url(url: str | None) -> bool:
    """Return ``True`` for SSH-scheme git URLs.

    Accepts the two forms git itself accepts:

    - ``ssh://[user@]host[:port]/path`` — explicit URL scheme.
    - ``[user@]host:path`` — scp-style shorthand.  The user part is
      optional (``git@github.com:foo.git``, ``deploy@host:repo.git``,
      bare ``github.com:foo.git``).

    Shared with terok-main: both the gate's env builder and callers that
    branch on "does this project use SSH?" (e.g. deploy-key prompts,
    gate-sync fallback hints) must agree on one definition.
    """
    if not url:
        return False
    candidate = url.strip()
    lowered = candidate.lower()
    if lowered.startswith("ssh://"):
        return True
    if "://" in candidate:
        return False
    return bool(_SCP_SSH_RE.match(candidate))


# ---------- Private helpers ----------


_SOCKET_BIND_WAIT_SECONDS = 4.0
"""Client-side tolerance for the daemon's reconciler to bind a fresh scope socket.

Roughly two of the reconciler's own poll ticks
(`terok_sandbox.vault.scope_sockets._POLL_INTERVAL_SECONDS`) — enough to
absorb one full miss plus the next bind attempt.
"""

_SOCKET_POLL_INTERVAL = 0.1
"""How often `_wait_for_socket` rechecks while inside the grace window."""


def _git_env_with_ssh(*, scope: str, use_personal_ssh: bool = False) -> dict:
    """Return a subprocess env for *scope*'s git operations.

    Three branches:

    - **Vault-only (default).**  The per-scope vault agent is the sole
      identity source.  ``GIT_SSH_COMMAND`` pins OpenSSH to the vault
      socket (``IdentityAgent=<sock>``), ignores the user's
      ``~/.ssh/config`` (``-F /dev/null``), suppresses default
      ``~/.ssh/id_*`` (``IdentityFile=none``), and forbids interactive
      prompts (``BatchMode=yes``).  This combination guarantees the
      user's personal keys are never offered *and* no passphrase,
      host-key, or password prompt can ever leak out to the caller's
      terminal — auth either succeeds from the vault or fails cleanly.
    - **Personal-SSH opt-in.**  ``use_personal_ssh=True`` returns the
      process env unmodified so OpenSSH behaves normally (user's loaded
      agent, ``~/.ssh/config``, default ``~/.ssh/id_*`` files, and any
      passphrase prompts go through the user's ambient askpass / tty).
      This branch is *either-or* with the vault — personal opt-in
      bypasses the vault entirely; it is not additive.
    - **Unconfigured.**  No vault socket and no opt-in — raise
      [`GateAuthNotConfigured`][terok_sandbox.gate.mirror.GateAuthNotConfigured] so the CLI layer can surface the
      two remediation paths.
    """
    from ..config import SandboxConfig

    env = os.environ.copy()
    if use_personal_ssh:
        return env  # let the user's ambient SSH handle it

    cfg = SandboxConfig()
    sock = cfg.ssh_signer_local_socket_path(scope)

    # The vault daemon's reconciler binds the per-scope socket on a short
    # poll interval, so a ``gate-sync`` fired right after ``ssh-init`` /
    # ``ssh-import`` can race the bind.  If the DB already says this scope
    # owns keys, give the daemon a bounded grace window to catch up before
    # declaring it unconfigured.
    if not _is_unix_socket(sock) and _db_has_keys_for_scope(cfg.db_path, scope):
        _wait_for_socket(sock, _SOCKET_BIND_WAIT_SECONDS)
    if not _is_unix_socket(sock):
        raise GateAuthNotConfigured(scope)

    env["SSH_AUTH_SOCK"] = str(sock)
    env["GIT_SSH_COMMAND"] = shlex.join(
        [
            "ssh",
            "-F",
            "/dev/null",
            "-o",
            f"IdentityAgent={sock}",
            "-o",
            "IdentityFile=none",
            "-o",
            "BatchMode=yes",
            "-o",
            "StrictHostKeyChecking=no",
        ]
    )
    return env


def _wait_for_socket(path: Path, timeout: float) -> None:
    """Block up to *timeout* seconds for *path* to become a Unix socket."""
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        if _is_unix_socket(path):
            return
        time.sleep(_SOCKET_POLL_INTERVAL)


def _db_has_keys_for_scope(db_path: Path, scope: str) -> bool:
    """Return ``True`` iff *scope* owns at least one row in ``ssh_key_assignments``.

    Opens the DB read-only via the ``file:?mode=ro`` URI so a missing DB
    file raises cleanly instead of being auto-created as a side effect.
    Any ``sqlite3.Error`` — missing file, missing table, locked DB —
    collapses to ``False`` so the caller falls straight through to
    [`GateAuthNotConfigured`][terok_sandbox.gate.mirror.GateAuthNotConfigured] instead of hanging.
    """
    try:
        conn = sqlite3.connect(f"file:{db_path}?mode=ro", uri=True)
        try:
            row = conn.execute(
                "SELECT 1 FROM ssh_key_assignments WHERE scope = ? LIMIT 1",
                (scope,),
            ).fetchone()
        finally:
            conn.close()
    except sqlite3.Error:
        return False
    return row is not None


def _is_unix_socket(path: Path) -> bool:
    """Return ``True`` iff *path* refers to an existing Unix domain socket."""
    try:
        return stat.S_ISSOCK(path.stat().st_mode)
    except FileNotFoundError:
        return False


def _clone_gate_mirror(upstream_url: str, gate_dir: Path, env: dict) -> None:
    """Clone the upstream repository as a bare mirror into *gate_dir*."""
    cmd = ["git", "clone", "--mirror", upstream_url, str(gate_dir)]
    try:
        subprocess.run(cmd, check=True, env=env)
    except FileNotFoundError:
        raise SystemExit("git not found on host; please install git")
    except subprocess.CalledProcessError as e:
        raise SystemExit(f"git clone --mirror failed: {e}")


def _init_remoteless_gate(gate_dir: Path) -> None:
    """Initialise an empty bare repo at *gate_dir* with no configured remote.

    Used for projects without an upstream: the container can still push to
    the gate (it behaves like any other bare repo), but there is nothing
    for the host to fetch.
    """
    gate_dir.mkdir(parents=True, exist_ok=True)
    try:
        subprocess.run(
            ["git", "init", "--bare", str(gate_dir)],
            check=True,
            capture_output=True,
            text=True,
            timeout=30,
        )
    except FileNotFoundError:
        raise SystemExit("git not found on host; please install git")
    except subprocess.CalledProcessError as e:
        raise SystemExit(f"git init --bare failed: {e.stderr or e}")


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
