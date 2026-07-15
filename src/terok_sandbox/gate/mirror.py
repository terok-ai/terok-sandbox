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

from __future__ import annotations

import asyncio
import contextlib
import logging
import os
import re
import shlex
import shutil
import subprocess  # nosec B404 — driving git for upstream mirror operations
import tempfile
import threading
from collections.abc import Callable
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import TYPE_CHECKING, TypedDict

if TYPE_CHECKING:
    from ..config import SandboxConfig


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

    The clone-cache refresh is best-effort: ``cache_error`` carries the
    failure description when the refresh was attempted and failed, so
    callers can report it instead of silently claiming a clean sync.
    ``cache_refreshed`` stays ``False`` both on failure and when no
    cache is configured; ``cache_error`` distinguishes the two.
    """

    path: str
    upstream_url: str | None
    created: bool
    success: bool
    updated_branches: list[str]
    errors: list[str]
    cache_refreshed: bool
    cache_error: str | None


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
        self._signer: _EphemeralSigner | None = None

    @property
    def cache_path(self) -> Path | None:
        """Clone cache directory for this scope, or ``None`` if caching is disabled."""
        return (self._clone_cache_base / self._scope) if self._clone_cache_base else None

    def _ssh_env(self) -> dict:
        """Return a subprocess env dict, injecting SSH config only for SSH upstreams.

        HTTPS upstreams don't use SSH at all, so we hand git an unmodified
        env in that case — fetching `GateAuthNotConfigured` on an HTTPS
        project would be absurd.  For SSH upstreams the ephemeral signer
        is lazy-started on first call and reused for the gate's lifetime.
        """
        if not is_ssh_url(self._upstream_url):
            return os.environ.copy()
        if self._use_personal_ssh:
            return os.environ.copy()  # let the user's ambient SSH handle it
        if self._signer is None:
            self._signer = _EphemeralSigner.start(self._scope)
        env = os.environ.copy()
        env["SSH_AUTH_SOCK"] = str(self._signer.socket_path)
        # Trust-on-first-use against a persistent known_hosts file rather
        # than disabling host-key verification: the first sync of a given
        # upstream pins the host key, and every subsequent sync verifies
        # against the pin.  The file lives beside the bare mirrors (one
        # per gate base path) so it survives across syncs and repos.
        known_hosts = self._known_hosts_path()
        env["GIT_SSH_COMMAND"] = shlex.join(
            [
                "ssh",
                "-F",
                "/dev/null",
                "-o",
                f"IdentityAgent={self._signer.socket_path}",
                "-o",
                "IdentityFile=none",
                "-o",
                "BatchMode=yes",
                "-o",
                "StrictHostKeyChecking=accept-new",
                "-o",
                f"UserKnownHostsFile={known_hosts}",
            ]
        )
        return env

    def _known_hosts_path(self) -> Path:
        """Return the persistent known_hosts file for this gate's mirrors.

        Lives at ``<gate base>/.known_hosts`` (the gate base is the parent
        of the per-repo bare mirror).  Created empty 0600 on first use so
        ``StrictHostKeyChecking=accept-new`` can append the pinned key.
        """
        path = self._gate_path.parent / ".known_hosts"
        if not path.exists():
            path.parent.mkdir(parents=True, exist_ok=True)
            path.touch(mode=0o600, exist_ok=True)
        return path

    def close(self) -> None:
        """Stop the ephemeral signer this gate started, if any.

        Idempotent.  Long-lived processes (the TUI) should call this
        explicitly so the signer thread and temp socket don't outlive
        the gate's last use.
        """
        if self._signer is not None:
            self._signer.stop()
            self._signer = None

    def __del__(self) -> None:
        """Best-effort signer teardown on GC."""
        with contextlib.suppress(Exception):  # __del__ never raises
            self.close()

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

        ``force_reinit`` recreates the whole local footprint — the bare
        mirror *and* the clone cache — so a hopeless state can always be
        recovered with one flag.  Deletion failures propagate: rebuilding
        over stale or partial data would silently defeat the point of a
        from-scratch recovery.
        """
        self._validate_gate()

        gate_dir = self._gate_path
        gate_exists = gate_dir.exists()
        gate_dir.parent.mkdir(parents=True, exist_ok=True)

        env = self._ssh_env()
        created = False
        if force_reinit:
            if gate_exists:
                shutil.rmtree(gate_dir)
                gate_exists = False
            if self.cache_path is not None and self.cache_path.exists():
                shutil.rmtree(self.cache_path)

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
                "cache_error": None,
            }

        sync_result = self.sync_branches(branches)

        # Refresh the non-bare clone cache from the bare mirror (best-effort).
        cache_error: str | None = None
        cache_refreshed = False
        if sync_result["success"]:
            self._heal_gate_head(env)
            if self._clone_cache_base:
                cache_error = self._refresh_clone_cache()
                cache_refreshed = cache_error is None

        return {
            "path": str(gate_dir),
            "upstream_url": self._upstream_url,
            "created": created,
            "success": sync_result["success"],
            "updated_branches": sync_result["updated_branches"],
            "errors": sync_result["errors"],
            "cache_refreshed": cache_refreshed,
            "cache_error": cache_error,
        }

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
            result = subprocess.run(cmd, capture_output=True, text=True, env=env, timeout=120)  # nosec B603 — argv is a fixed list controlled by this module

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

            result = subprocess.run(cmd, capture_output=True, text=True, env=env)  # nosec B603 — argv is a fixed list controlled by this module
            if result.returncode != 0 and self._default_branch:
                cmd[5] = "HEAD"
                result = subprocess.run(cmd, capture_output=True, text=True, env=env)  # nosec B603 — argv is a fixed list controlled by this module
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

    def _heal_gate_head(self, env: dict) -> None:
        """Re-point the gate's ``HEAD`` after an upstream default-branch rename.

        ``git remote update --prune`` moves the mirror's branch refs but
        never touches its ``HEAD`` symref, so a rename upstream leaves it
        dangling — and everything that asks the gate for its HEAD (clone
        cache ``set-head``, fresh task clones) then fails or silently
        checks out nothing.  The happy path costs one local ref lookup;
        the upstream roundtrip runs only when HEAD is actually dangling.

        Best-effort: on any failure the gate keeps its current HEAD and
        the clone-cache refresh reports the fallout.
        """
        gate_dir = str(self._gate_path)
        try:
            target = subprocess.run(  # nosec B603 B607 — argv built from fixed verbs + repo-relative paths — binary PATH lookup is the cross-distro contract
                ["git", "-C", gate_dir, "symbolic-ref", "--quiet", "HEAD"],
                capture_output=True,
                text=True,
                timeout=10,
            ).stdout.strip()
            dangling = not target or (
                subprocess.run(  # nosec B603 B607 — argv built from fixed verbs + repo-relative paths — binary PATH lookup is the cross-distro contract
                    ["git", "-C", gate_dir, "show-ref", "--verify", "--quiet", target],
                    capture_output=True,
                    timeout=10,
                ).returncode
                != 0
            )
            if not dangling:
                return

            upstream_head = _query_upstream_head_ref(gate_dir, env)
            if upstream_head is None:
                logger.warning(
                    "Gate HEAD %r is dangling and upstream's default branch "
                    "could not be determined; leaving HEAD unchanged",
                    target,
                )
                return
            subprocess.run(  # nosec B603 B607 — argv built from fixed verbs + repo-relative paths — binary PATH lookup is the cross-distro contract
                ["git", "-C", gate_dir, "symbolic-ref", "HEAD", upstream_head],
                check=True,
                capture_output=True,
                timeout=10,
            )
            logger.info("Gate HEAD healed: %s -> %s", target or "(unset)", upstream_head)
        except (subprocess.CalledProcessError, subprocess.TimeoutExpired, OSError) as exc:
            logger.warning("Gate HEAD heal failed (non-fatal): %s", _describe_git_failure(exc))

    def _refresh_clone_cache(self) -> str | None:
        """Refresh the non-bare clone cache from the local bare mirror.

        The cache exists purely to make task startup fast, so it is never
        worth preserving in a broken state: the happy path is a cheap
        in-place update, and *any* failure there discards the cache and
        rebuilds it from the gate (a local ``file://`` clone).  Returns
        ``None`` on success, or a failure description (including git's
        stderr) only when the rebuild itself failed too.
        """
        cache_dir = self.cache_path
        if cache_dir is None:
            return "no clone cache configured"

        gate_file_url = self._gate_path.resolve().as_uri()
        if cache_dir.exists():
            try:
                self._update_cache_in_place(cache_dir, gate_file_url)
                return None
            except (subprocess.CalledProcessError, subprocess.TimeoutExpired, OSError) as exc:
                logger.warning(
                    "Clone cache update failed; rebuilding from scratch: %s",
                    _describe_git_failure(exc),
                )
        return self._rebuild_cache(cache_dir, gate_file_url)

    def _update_cache_in_place(self, cache_dir: Path, gate_file_url: str) -> None:
        """Fast-forward an existing cache working tree to the gate's HEAD.

        Raises on any git failure — the caller answers every failure the
        same way (discard and rebuild), so no per-step handling here.
        """
        # Ensure origin points to current bare mirror
        subprocess.run(  # nosec B603 B607 — argv built from fixed verbs + repo-relative paths — binary PATH lookup is the cross-distro contract
            ["git", "-C", str(cache_dir), "remote", "set-url", "origin", gate_file_url],
            check=True,
            capture_output=True,
            timeout=10,
        )
        subprocess.run(  # nosec B603 B607 — argv built from fixed verbs + repo-relative paths — binary PATH lookup is the cross-distro contract
            ["git", "-C", str(cache_dir), "fetch", "--all", "--prune"],
            check=True,
            capture_output=True,
            timeout=120,
        )
        # ``fetch`` never touches ``refs/remotes/origin/HEAD`` — only
        # ``clone`` creates it, and the mirror's default branch can
        # move after the cache was cloned.  Re-resolve it explicitly
        # or the reset below fails on caches missing the ref.
        subprocess.run(  # nosec B603 B607 — argv built from fixed verbs + repo-relative paths — binary PATH lookup is the cross-distro contract
            ["git", "-C", str(cache_dir), "remote", "set-head", "origin", "--auto"],
            check=True,
            capture_output=True,
            timeout=30,
        )
        # Update working tree to match fetched HEAD — the cache is
        # copied as-is into task workspaces, so stale files matter.
        subprocess.run(  # nosec B603 B607 — argv built from fixed verbs + repo-relative paths — binary PATH lookup is the cross-distro contract
            ["git", "-C", str(cache_dir), "reset", "--hard", "origin/HEAD"],
            check=True,
            capture_output=True,
            timeout=30,
        )
        # Remove untracked/ignored files so the cache stays pristine.
        subprocess.run(  # nosec B603 B607 — argv built from fixed verbs + repo-relative paths — binary PATH lookup is the cross-distro contract
            ["git", "-C", str(cache_dir), "clean", "-ffdx"],
            check=True,
            capture_output=True,
            timeout=30,
        )

    def _rebuild_cache(self, cache_dir: Path, gate_file_url: str) -> str | None:
        """Build the cache from scratch with a fresh clone of the gate.

        A cache that cannot be built whole must not exist at all: a fresh
        ``git clone`` exits 0 even when the remote's HEAD is unresolvable,
        leaving an *empty* working tree that would seed empty task
        workspaces.  The post-clone ``rev-parse`` catches that, and any
        failure removes the half-built cache so task startup falls back
        to a full clone instead of copying garbage.
        """
        try:
            if cache_dir.exists():
                shutil.rmtree(cache_dir)
            cache_dir.parent.mkdir(parents=True, exist_ok=True)
            logger.info("Creating clone cache at %s", cache_dir)
            subprocess.run(  # nosec B603 B607 — argv built from fixed verbs + repo-relative paths — binary PATH lookup is the cross-distro contract
                ["git", "clone", gate_file_url, str(cache_dir)],
                check=True,
                capture_output=True,
                timeout=300,
            )
            subprocess.run(  # nosec B603 B607 — argv built from fixed verbs + repo-relative paths — binary PATH lookup is the cross-distro contract
                ["git", "-C", str(cache_dir), "rev-parse", "--verify", "HEAD"],
                check=True,
                capture_output=True,
                timeout=10,
            )
            return None
        except (subprocess.CalledProcessError, subprocess.TimeoutExpired, OSError) as exc:
            error = _describe_git_failure(exc)
            logger.warning("Clone cache rebuild failed (non-fatal): %s", error)
            shutil.rmtree(cache_dir, ignore_errors=True)
            return error


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


# Guards only the unix-socket bind on the signer thread; passphrase
# resolution runs on the caller (see ``_EphemeralSigner.start``).
_SIGNER_BIND_TIMEOUT_S = 5.0
_SIGNER_STOP_TIMEOUT_S = 5.0


@dataclass(frozen=True)
class _EphemeralSigner:
    """A host-local SSH-agent signer bound for a single ``GitGate`` lifetime.

    An asyncio loop running on a daemon background thread hosts
    [`start_ssh_signer_local`][terok_sandbox.vault.ssh.signer.start_ssh_signer_local];
    the caller reads ``socket_path`` and points OpenSSH at it via
    ``IdentityAgent``.
    """

    socket_path: Path
    _tmpdir: tempfile.TemporaryDirectory
    _thread: threading.Thread
    _loop: asyncio.AbstractEventLoop
    _server: asyncio.Server

    @classmethod
    def start(cls, scope: str) -> _EphemeralSigner:
        """Bind a fresh signer for *scope*; raise ``GateAuthNotConfigured`` if no keys."""
        from ..config import SandboxConfig
        from ..vault.ssh.signer import start_ssh_signer_local

        cfg = SandboxConfig()
        # Resolve once on the calling thread, where waiting is free: the
        # bind timeout below must not wrap a slow keystore tier (TPM2
        # unseal, secret-manager CLI), and the key pre-check reuses it.
        passphrase = cfg.resolve_passphrase(prompt_on_tty=False)
        if passphrase is None or not _db_has_keys_for_scope(cfg, scope, passphrase):
            raise GateAuthNotConfigured(scope)

        tmpdir = tempfile.TemporaryDirectory(prefix="terok-gate-signer-")
        socket_path = Path(tmpdir.name) / "agent.sock"
        loop = asyncio.new_event_loop()
        ready = threading.Event()
        server_box: list[asyncio.Server] = []
        start_error: list[BaseException] = []

        async def _start() -> None:
            """Bind the signer; serve forever until the loop is stopped."""
            try:
                server = await start_ssh_signer_local(
                    scope=scope,
                    socket_path=socket_path,
                    db_path=str(cfg.db_path),
                    passphrase=passphrase,
                )
            except Exception as exc:  # noqa: BLE001 — surface bind failure
                start_error.append(exc)
                ready.set()
                return
            server_box.append(server)
            ready.set()
            async with server:
                with contextlib.suppress(asyncio.CancelledError):
                    await server.serve_forever()

        def _run_loop() -> None:
            asyncio.set_event_loop(loop)
            try:
                loop.run_until_complete(_start())
            finally:
                loop.close()

        thread = threading.Thread(target=_run_loop, daemon=True, name=f"terok-signer-{scope}")
        thread.start()
        if not ready.wait(timeout=_SIGNER_BIND_TIMEOUT_S):
            # A timed-out bind leaves the loop running on the background
            # thread; stop it and join before cleanup so we don't leak a
            # live loop/thread (``call_soon_threadsafe`` is the only safe
            # cross-thread way to reach into the signer's own loop).
            with contextlib.suppress(RuntimeError):
                loop.call_soon_threadsafe(loop.stop)
            thread.join(timeout=_SIGNER_STOP_TIMEOUT_S)
            tmpdir.cleanup()
            raise RuntimeError(
                f"ephemeral SSH signer for scope {scope!r} did not bind within "
                f"{_SIGNER_BIND_TIMEOUT_S}s"
            )
        if start_error:
            thread.join(timeout=_SIGNER_STOP_TIMEOUT_S)
            tmpdir.cleanup()
            raise RuntimeError(
                f"ephemeral SSH signer for scope {scope!r} failed to bind: {start_error[0]}"
            ) from start_error[0]
        return cls(
            socket_path=socket_path,
            _tmpdir=tmpdir,
            _thread=thread,
            _loop=loop,
            _server=server_box[0],
        )

    def stop(self) -> None:
        """Close the server, drain the asyncio loop, remove the temp dir."""
        if not self._thread.is_alive():
            self._tmpdir.cleanup()
            return
        # call_soon_threadsafe schedules ``server.close()`` on the
        # server's own loop — calling it directly across threads is UB.
        self._loop.call_soon_threadsafe(self._server.close)
        self._thread.join(timeout=_SIGNER_STOP_TIMEOUT_S)
        self._tmpdir.cleanup()


def _db_has_keys_for_scope(cfg: SandboxConfig, scope: str, passphrase: str) -> bool:
    """Return ``True`` iff *scope* has at least one assigned SSH key.

    Opens the credentials DB with the caller's pre-resolved *passphrase*.
    Any exception (wrong passphrase, no DB file, schema not yet
    bootstrapped) collapses to ``False`` so the caller surfaces
    [`GateAuthNotConfigured`][terok_sandbox.gate.mirror.GateAuthNotConfigured]
    with its actionable two-door hint instead of leaking a stack trace.
    """
    from ..vault.store.db import CredentialDB

    try:
        db = CredentialDB(cfg.db_path, passphrase=passphrase)
    except Exception:  # noqa: BLE001 — fail-soft to the friendly GateAuthNotConfigured
        return False
    try:
        return bool(db.list_ssh_keys_for_scope(scope))
    except Exception:  # noqa: BLE001 — same rationale (e.g. schema not bootstrapped)
        return False
    finally:
        db.close()


def _describe_git_failure(exc: Exception) -> str:
    """Render a git subprocess failure with its captured stderr.

    ``CalledProcessError``'s own ``str()`` names only the argv and exit
    status; the actionable part — git's complaint — sits in ``stderr``,
    which ``capture_output=True`` collected but nobody would otherwise
    show.  ``TimeoutExpired`` / ``OSError`` carry no stderr and fall
    through to their plain ``str()``.
    """
    stderr = getattr(exc, "stderr", None)
    if isinstance(stderr, bytes):
        stderr = stderr.decode("utf-8", errors="replace")
    detail = (stderr or "").strip()
    return f"{exc}: {detail}" if detail else str(exc)


def _query_upstream_head_ref(gate_dir: str, env: dict) -> str | None:
    """Ask upstream which ref its HEAD points at (e.g. ``refs/heads/main``).

    Runs ``git ls-remote --symref origin HEAD`` inside the gate so the
    mirror's own origin URL and the caller's SSH env do the routing.
    Returns ``None`` when upstream is unreachable or advertises no
    symref (very old servers) — callers treat that as "cannot heal".
    """
    result = subprocess.run(  # nosec B603 B607 — argv built from fixed verbs + repo-relative paths — binary PATH lookup is the cross-distro contract
        ["git", "-C", gate_dir, "ls-remote", "--symref", "origin", "HEAD"],
        capture_output=True,
        text=True,
        env=env,
        timeout=30,
    )
    if result.returncode != 0:
        return None
    for line in result.stdout.splitlines():
        if line.startswith("ref:"):
            # "ref: refs/heads/main\tHEAD"
            return line.split()[1]
    return None


def _clone_gate_mirror(upstream_url: str, gate_dir: Path, env: dict) -> None:
    """Clone the upstream repository as a bare mirror into *gate_dir*."""
    cmd = ["git", "clone", "--mirror", upstream_url, str(gate_dir)]
    try:
        subprocess.run(cmd, check=True, env=env)  # nosec B603 — argv is a fixed list controlled by this module
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
        subprocess.run(  # nosec B603 B607 — argv built from fixed verbs + repo-relative paths — binary PATH lookup is the cross-distro contract
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
        result = subprocess.run(cmd, capture_output=True, text=True, env=env, timeout=30)  # nosec B603 — argv is a fixed list controlled by this module

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
        result = subprocess.run(cmd, capture_output=True, text=True, env=env)  # nosec B603 — argv is a fixed list controlled by this module

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
        result = subprocess.run(cmd, capture_output=True, text=True, env=env)  # nosec B603 — argv is a fixed list controlled by this module
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
