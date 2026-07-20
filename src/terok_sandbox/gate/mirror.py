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

Ref model
---------

The gate keeps three ref namespaces, and the split is what makes sync safe:

- ``refs/heads/*`` — the container-facing view.  Sync on its own only ever
  *creates* branches or *fast-forwards* them; every other change (delete,
  non-fast-forward move) is returned as a pending op and applied only via
  [`apply_pending_ops`][terok_sandbox.gate.mirror.GitGate.apply_pending_ops]
  after the operator confirmed it.  Agent work pushed to the gate but not
  yet upstream therefore survives any number of syncs.
- ``refs/terok/upstream/*`` — a private snapshot of upstream's heads,
  force-updated and pruned freely on every fetch.  It is the "what did
  upstream have last time" memory that classifies each branch: a gate head
  equal to its snapshot entry has no gate-local work (a destructive op on
  it is *lossless*), anything else diverged locally (*lossy*).
- ``refs/terok/backup/<branch>/<stamp>-<sha12>`` — the old tip of every
  destructively changed branch, written before the change so nothing ever
  becomes unreachable.  Expired per the retention policy by
  [`prune_backups`][terok_sandbox.gate.mirror.GitGate.prune_backups].
- ``refs/terok/attic/<branch>`` — the last upstream tip a branch had
  before it went pending (deleted upstream, or rewritten while the gate
  head stayed behind).  The pruning, force-updating snapshot forgets that
  tip immediately, but it is exactly what proves a pending op lossless —
  so the attic keeps it until the branch is resolved (op applied, or gate
  and upstream agree again).  First writer wins: across repeated upstream
  rewrites the attic still names the last tip the gate was actually in
  sync with.

The whole ``refs/terok`` namespace is hidden (``transfer.hideRefs``), so
containers can neither see nor overwrite the snapshot, attic, or backups —
a container that could push the snapshot could forge "lossless".

Value types returned by ``GitGate`` methods:

- [`GateSyncResult`][terok_sandbox.gate.mirror.GateSyncResult] — full sync outcome: applied ops, pending
  destructive ops, kept gate-only branches (``upstream_url`` is ``None``
  for remoteless gates)
- [`AppliedOp`][terok_sandbox.gate.mirror.AppliedOp] / [`PendingOp`][terok_sandbox.gate.mirror.PendingOp] — one branch-level ref change,
  performed or awaiting confirmation
- [`ApplyPendingResult`][terok_sandbox.gate.mirror.ApplyPendingResult] — outcome of applying confirmed pending ops
- [`BackupRef`][terok_sandbox.gate.mirror.BackupRef] — one parsed backup ref
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
from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import TYPE_CHECKING, Literal, TypedDict, cast

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


OpKind = Literal["create", "fast_forward", "force_update", "delete"]
"""Branch-level ref change kinds.  ``create``/``fast_forward`` are the only
kinds sync applies on its own; ``force_update``/``delete`` only ever appear
as pending ops."""

PendingReason = Literal["upstream_rewrite", "upstream_delete", "unknown_provenance"]
"""Why a destructive op is proposed.  ``unknown_provenance`` marks branches
found during migration from a pre-snapshot mirror gate — upstream doesn't
have them *now*, but whether it ever did is unknowable, so they are
surfaced once and otherwise left alone."""


class AppliedOp(TypedDict):
    """One branch-level ref change sync (or an approved apply) performed."""

    branch: str
    kind: OpKind
    old_sha: str | None  # None for create
    new_sha: str | None  # None for delete


class PendingOp(TypedDict):
    """A destructive branch change awaiting operator confirmation.

    ``lossless`` is the heart of the safety story: ``True`` means the gate
    tip equals what upstream last advertised for this branch — no agent
    commits would be discarded.  ``gate_only_commits`` quantifies the lossy
    case (``None`` when provenance is unknown and the count would be
    meaningless).  ``gate_sha`` doubles as the compare-and-swap guard in
    [`apply_pending_ops`][terok_sandbox.gate.mirror.GitGate.apply_pending_ops]:
    an op is refused if the branch moved since it was proposed.
    """

    branch: str
    kind: Literal["force_update", "delete"]
    reason: PendingReason
    gate_sha: str
    upstream_sha: str | None  # None for delete
    old_snapshot_sha: str | None  # None when provenance is unknown
    lossless: bool
    gate_only_commits: int | None


class GateSyncResult(TypedDict):
    """Result of a gate sync operation.

    ``upstream_url`` is ``None`` when the gate is initialised without a
    remote — a local-only mirror that the container can push to but that
    never fetches external commits.

    ``pending`` ops are *proposals*, not failures — a sync that fetched
    cleanly and applied its safe ops reports ``success: True`` regardless
    of how much destructive work awaits confirmation.  ``notes`` carries
    non-fatal observations (moved tags, expired backups, foreign refs).

    The clone-cache refresh is best-effort: ``cache_error`` carries the
    failure description when the refresh was attempted and failed, so
    callers can report it instead of silently claiming a clean sync.
    ``cache_refreshed`` stays ``False`` both on failure and when no
    cache is configured; ``cache_error`` distinguishes the two.
    """

    path: str
    upstream_url: str | None
    created: bool
    migrated: bool
    success: bool
    errors: list[str]
    notes: list[str]
    applied: list[AppliedOp]
    pending: list[PendingOp]
    gate_only_branches: list[str]
    cache_refreshed: bool
    cache_error: str | None


class ApplyPendingResult(TypedDict):
    """Result of applying operator-confirmed pending ops.

    Ops whose branch moved between proposal and apply land in ``errors``
    and leave the branch untouched — the rest are still applied, so a
    single race never voids a whole confirmation.  ``backups`` maps each
    changed branch to the backup ref holding its previous tip (empty when
    backups are disabled).
    """

    success: bool
    applied: list[AppliedOp]
    backups: dict[str, str]
    errors: list[str]


class BackupRef(TypedDict):
    """One parsed ``refs/terok/backup/…`` entry."""

    ref: str
    branch: str
    saved_at: str  # ISO timestamp parsed from the ref name
    sha: str  # the backed-up tip (full sha the ref points at)


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


# ---------- Ref-model constants ----------

#: Hidden namespace root — everything terok-private lives under it, and
#: ``transfer.hideRefs`` keeps containers away from all of it at once.
_TEROK_NAMESPACE = "refs/terok"
_SNAPSHOT_PREFIX = "refs/terok/upstream/"
_BACKUP_PREFIX = "refs/terok/backup/"
_ATTIC_PREFIX = "refs/terok/attic/"
_HEADS_PREFIX = "refs/heads/"
_FETCH_REFSPEC = f"+{_HEADS_PREFIX}*:{_SNAPSHOT_PREFIX}*"
#: git's "ref must not exist yet" / "delete unconditionally" CAS sentinel.
_ZERO_SHA = "0" * 40
#: Timestamp format baked into backup ref names — sortable, filesystem- and
#: refname-safe, and the retention clock (commit dates would lie about when
#: the *backup* was taken).
_BACKUP_STAMP_FORMAT = "%Y%m%dT%H%M%SZ"
_BACKUP_STAMP_RE = re.compile(r"^(\d{8}T\d{6}Z)-([0-9a-f]{12})$")


def _git(
    gate_dir: Path | str,
    *args: str,
    env: dict | None = None,
    timeout: float = 30,
    check: bool = False,
) -> subprocess.CompletedProcess[str]:
    """Run a git subcommand inside *gate_dir* with captured text output."""
    return subprocess.run(  # nosec B603 B607 — argv built from fixed verbs + repo-relative paths — binary PATH lookup is the cross-distro contract
        ["git", "-C", str(gate_dir), *args],
        capture_output=True,
        text=True,
        env=env,
        timeout=timeout,
        check=check,
    )


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
        backups_enabled: bool = True,
        backup_retention_days: int = 30,
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
        backups_enabled:
            When ``True`` (default), every destructive branch change applied
            via [`apply_pending_ops`][terok_sandbox.gate.mirror.GitGate.apply_pending_ops]
            first saves the old tip under ``refs/terok/backup/``.  Opt out
            per project when the reflog alone is protection enough.
        backup_retention_days:
            Backups older than this are expired by
            [`prune_backups`][terok_sandbox.gate.mirror.GitGate.prune_backups]
            (which a successful sync runs automatically).  ``0`` keeps
            backups forever.
        """
        self._scope = scope
        self._gate_path = Path(gate_path)
        self._upstream_url = upstream_url
        self._default_branch = default_branch
        self._use_personal_ssh = use_personal_ssh
        self._validate_gate_fn = validate_gate_fn
        self._clone_cache_base = Path(clone_cache_base) if clone_cache_base else None
        self._backups_enabled = backups_enabled
        self._backup_retention_days = backup_retention_days
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
        """Sync the host-side git gate from upstream, destroying nothing.

        With an upstream configured, clones (or fetches) from it using the
        project's SSH setup.  Without one, initialises a bare repo in place
        and returns a no-op sync — the gate then serves as a local-only
        remote that the container can push to, giving the agent somewhere
        to stage commits even when there is nothing external to mirror.

        A remoteless gate that already exists is a proper no-op: nothing
        re-initialises, and the returned op lists are empty.

        Only safe branch changes are applied here — creates and
        fast-forwards.  Deletes and non-fast-forward moves come back as
        ``pending`` proposals for
        [`apply_pending_ops`][terok_sandbox.gate.mirror.GitGate.apply_pending_ops];
        branches that exist only on the gate (agent work not yet upstream)
        are listed in ``gate_only_branches`` and never touched.

        *branches* restricts the sync to the named branches (the auto-sync
        allowlist); the default full sync covers everything upstream has.

        ``force_reinit`` recreates the whole local footprint — the bare
        mirror *and* the clone cache — so a hopeless state can always be
        recovered with one flag.  Deletion failures propagate: rebuilding
        over stale or partial data would silently defeat the point of a
        from-scratch recovery.  This is the one path that still discards
        gate-local work, which is exactly why it hides behind an explicit
        operator flag and never runs implicitly.
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
                _normalise_fresh_gate(gate_dir)
            else:
                _init_remoteless_gate(gate_dir)
            created = True

        # A remoteless gate has nothing to fetch — skip the upstream fetch
        # (which would fail on a repo with no origin) and the clone-cache
        # refresh (there is no upstream state to track).
        if not self._upstream_url:
            return _empty_sync_result(str(gate_dir), created=created)

        migrated, errors, notes, applied, pending, gate_only = self._sync_from_upstream(
            env, branches, freshly_created=created
        )
        success = not errors

        # A dangling gate HEAD breaks every fresh clone of the gate, so a
        # failed heal is a sync failure, not a footnote.
        if success and (head_error := self._align_gate_head(env)):
            errors.append(head_error)
            success = False

        if success and (expired := self.prune_backups()):
            notes.append(f"expired {len(expired)} backup ref(s) past retention")

        # Refresh the non-bare clone cache from the bare mirror (best-effort).
        cache_error: str | None = None
        cache_refreshed = False
        if success and self._clone_cache_base:
            cache_error = self._refresh_clone_cache()
            cache_refreshed = cache_error is None

        return {
            "path": str(gate_dir),
            "upstream_url": self._upstream_url,
            "created": created,
            "migrated": migrated,
            "success": success,
            "errors": errors,
            "notes": notes,
            "applied": applied,
            "pending": pending,
            "gate_only_branches": gate_only,
            "cache_refreshed": cache_refreshed,
            "cache_error": cache_error,
        }

    def _sync_from_upstream(
        self, env: dict, branches: list[str] | None, *, freshly_created: bool
    ) -> tuple[bool, list[str], list[str], list[AppliedOp], list[PendingOp], list[str]]:
        """Normalise config, fetch, classify, apply safe ops — the sync core.

        Returns ``(migrated, errors, notes, applied, pending,
        gate_only_branches)``.  A pre-existing gate may still carry the
        destructive mirror configuration (``+refs/*:refs/*`` + prune);
        that is normalised exactly once here, and the migration sync
        classifies differently (no snapshot memory exists yet).  Any
        exception collapses into ``errors``: sync is a boundary the TUI
        poller and CLI call in loops, and a transient git hiccup must
        degrade into a reportable failed sync, not a crash.
        """
        migrated = False
        errors: list[str] = []
        notes: list[str] = []
        applied: list[AppliedOp] = []
        pending: list[PendingOp] = []
        gate_only: list[str] = []
        try:
            migrated = False if freshly_created else _ensure_gate_config(self._gate_path)
            old_snapshot = _read_refs(self._gate_path, _SNAPSHOT_PREFIX)
            fetch_error = self._fetch_upstream(env, branches, notes)
            if fetch_error is not None:
                errors.append(fetch_error)
            else:
                applied, pending, gate_only = self._plan_and_apply_safe_ops(
                    old_snapshot, branches, first_sync=migrated, notes=notes
                )
        except subprocess.TimeoutExpired:
            errors.append("Sync timed out")
        except Exception as e:  # noqa: BLE001 — boundary: report, don't crash callers
            errors.append(str(e))
        return migrated, errors, notes, applied, pending, gate_only

    def _fetch_upstream(
        self, env: dict, branches: list[str] | None, notes: list[str]
    ) -> str | None:
        """Fetch upstream state into the snapshot namespace.

        Full mode fetches with the configured snapshot refspec plus
        ``--prune`` (scoped to that refspec's target, so it never touches
        ``refs/heads``) and ``--tags`` (without ``--prune-tags`` — tags an
        agent created on the gate are not sync's to delete).

        Selective mode fetches one explicit refspec per requested branch
        and prunes nothing; a branch upstream no longer advertises is
        detected via ``ls-remote`` and its snapshot entry dropped by hand —
        that *is* the deletion signal in this mode.

        Returns an error description, or ``None`` on success.  A fetch
        whose only complaint is force-moved upstream tags is a success
        with a note: git rejects clobbering existing tags (rc 1) on every
        subsequent fetch, and one moved tag must not wedge sync forever.
        """
        gate_dir = self._gate_path
        if branches:
            missing = self._drop_snapshots_of_deleted(env, branches)
            to_fetch = [b for b in branches if b not in missing]
            if not to_fetch:
                return None
            refspecs = [f"+{_HEADS_PREFIX}{b}:{_SNAPSHOT_PREFIX}{b}" for b in to_fetch]
            result = _git(gate_dir, "fetch", "origin", *refspecs, env=env, timeout=120)
        else:
            result = _git(gate_dir, "fetch", "--prune", "--tags", "origin", env=env, timeout=120)

        if result.returncode == 0:
            return None
        if _only_tag_clobbers(result.stderr):
            moved = ", ".join(sorted(_rejected_tags(result.stderr))) or "unknown"
            notes.append(f"upstream moved existing tag(s) not updated: {moved}")
            return None
        return f"fetch failed: {result.stderr.strip()}"

    def _drop_snapshots_of_deleted(self, env: dict, branches: list[str]) -> set[str]:
        """Return requested *branches* gone from upstream; drop their snapshots.

        One batched ``ls-remote --heads`` roundtrip answers "which of these
        still exist" — fetching a deleted ref would otherwise fail the whole
        selective fetch.  The snapshot entry of a deleted branch is removed
        here so the classifier sees the deletion the same way a pruning
        full fetch would show it.
        """
        listed = _git(
            self._gate_path,
            "ls-remote",
            "--heads",
            "origin",
            *(f"{_HEADS_PREFIX}{b}" for b in branches),
            env=env,
            timeout=60,
            check=True,
        ).stdout
        alive = {line.split("\t")[1].removeprefix(_HEADS_PREFIX) for line in listed.splitlines()}
        missing = set(branches) - alive
        for branch in missing:
            _git(self._gate_path, "update-ref", "-d", f"{_SNAPSHOT_PREFIX}{branch}")
        return missing

    def _plan_and_apply_safe_ops(
        self,
        old_snapshot: dict[str, str],
        branches: list[str] | None,
        *,
        first_sync: bool,
        notes: list[str],
    ) -> tuple[list[AppliedOp], list[PendingOp], list[str]]:
        """Classify every branch and apply the safe subset of the plan.

        The three-way comparison — snapshot before the fetch, snapshot
        after it, current gate heads — decides each branch's fate exactly
        as documented in the module docstring.  Creates and fast-forwards
        are applied immediately through compare-and-swap ``update-ref``
        (a concurrent container push makes the op fail into a note rather
        than clobbering the push).  Everything destructive is returned as
        pending.

        *first_sync* marks the one sync right after migrating a mirror
        gate: with no snapshot memory, heads absent upstream are surfaced
        once as ``unknown_provenance`` pending deletes so stale
        squash-merge residue gets a cleanup offer; declining leaves them
        as ordinary gate-only branches from the next sync on.
        """
        gate_dir = self._gate_path
        new_snapshot = _read_refs(gate_dir, _SNAPSHOT_PREFIX)
        heads = _read_refs(gate_dir, _HEADS_PREFIX)
        attic = _read_refs(gate_dir, _ATTIC_PREFIX)
        restrict = set(branches) if branches else None

        applied: list[AppliedOp] = []
        pending: list[PendingOp] = []
        gate_only: list[str] = []

        for branch in sorted(heads | new_snapshot | attic):
            if restrict is not None and branch not in restrict:
                continue
            # Attic first: it names the last tip the gate was in sync with,
            # which survives repeated upstream rewrites; the pre-fetch
            # snapshot only covers the most recent one.
            op = self._classify_branch(
                branch,
                gate_sha=heads.get(branch),
                upstream_sha=new_snapshot.get(branch),
                old_sha=attic.get(branch) or old_snapshot.get(branch),
                first_sync=first_sync,
            )
            if op is None:
                continue
            if op == "gate-only":
                gate_only.append(branch)
            elif op["kind"] in ("create", "fast_forward"):
                if error := self._apply_ref_cas(op):
                    notes.append(error)
                else:
                    applied.append(op)
            else:
                pending.append(cast("PendingOp", op))

        self._update_attic(old_snapshot, new_snapshot, heads)
        return applied, pending, gate_only

    def _classify_branch(
        self,
        branch: str,
        *,
        gate_sha: str | None,
        upstream_sha: str | None,
        old_sha: str | None,
        first_sync: bool,
    ) -> AppliedOp | PendingOp | Literal["gate-only"] | None:
        """Decide one branch's fate from the three-way ref comparison.

        Returns a safe [`AppliedOp`][terok_sandbox.gate.mirror.AppliedOp]
        (not yet performed), a destructive
        [`PendingOp`][terok_sandbox.gate.mirror.PendingOp] proposal,
        ``"gate-only"`` for agent branches upstream never had, or ``None``
        for a branch already up to date.
        """
        if gate_sha is None:
            if upstream_sha is None:
                return None  # attic residue only — handled by _update_attic
            return {"branch": branch, "kind": "create", "old_sha": None, "new_sha": upstream_sha}

        if upstream_sha is not None:
            if gate_sha == upstream_sha:
                return None
            if self._is_ancestor(gate_sha, upstream_sha):
                return {
                    "branch": branch,
                    "kind": "fast_forward",
                    "old_sha": gate_sha,
                    "new_sha": upstream_sha,
                }
            return self._pending(
                branch,
                "force_update",
                "upstream_rewrite",
                gate_sha=gate_sha,
                upstream_sha=upstream_sha,
                old_sha=old_sha,
            )

        # Upstream doesn't have the branch.  With provenance (snapshot or
        # attic memory) that's an upstream deletion; without it, it is an
        # agent branch — except on the first post-migration sync, where
        # history is unknowable and we offer the cleanup exactly once.
        if old_sha is not None:
            return self._pending(
                branch,
                "delete",
                "upstream_delete",
                gate_sha=gate_sha,
                upstream_sha=None,
                old_sha=old_sha,
            )
        if first_sync:
            return self._pending(
                branch,
                "delete",
                "unknown_provenance",
                gate_sha=gate_sha,
                upstream_sha=None,
                old_sha=None,
            )
        return "gate-only"

    def _pending(
        self,
        branch: str,
        kind: Literal["force_update", "delete"],
        reason: PendingReason,
        *,
        gate_sha: str,
        upstream_sha: str | None,
        old_sha: str | None,
    ) -> PendingOp:
        """Build a pending op with its lossless/lossy classification."""
        lossless = old_sha is not None and gate_sha == old_sha
        base = old_sha or upstream_sha
        gate_only_commits: int | None = None
        if lossless:
            gate_only_commits = 0
        elif base is not None:
            gate_only_commits = _count_commits_range(self._gate_path, base, gate_sha, None)
        return {
            "branch": branch,
            "kind": kind,
            "reason": reason,
            "gate_sha": gate_sha,
            "upstream_sha": upstream_sha,
            "old_snapshot_sha": old_sha,
            "lossless": lossless,
            "gate_only_commits": gate_only_commits,
        }

    def _apply_ref_cas(self, op: AppliedOp) -> str | None:
        """Apply a safe op via compare-and-swap; return an error note on race.

        ``git update-ref <ref> <new> <old>`` refuses to move a ref that
        isn't exactly at ``<old>`` — the all-zeros form means "must not
        exist yet".  A concurrent container push therefore fails the op
        cleanly instead of being clobbered; the next sync re-plans against
        the pushed state.
        """
        ref = f"{_HEADS_PREFIX}{op['branch']}"
        new_sha = op["new_sha"] or _ZERO_SHA
        old_sha = op["old_sha"] or _ZERO_SHA
        result = _git(self._gate_path, "update-ref", ref, new_sha, old_sha)
        if result.returncode != 0:
            return f"skipped {op['kind']} of {op['branch']}: ref moved during sync"
        return None

    def _update_attic(
        self, old_snapshot: dict[str, str], new_snapshot: dict[str, str], heads: dict[str, str]
    ) -> None:
        """Maintain the attic — provenance memory for branches gone pending.

        A pruning, force-updating fetch erases the snapshot entry that
        proved a pending op lossless (and, for deletes, that the branch
        ever came from upstream at all), so the *next* sync — and any
        offline [`pending_ops`][terok_sandbox.gate.mirror.GitGate.pending_ops]
        call — would be unable to classify it.  The attic records the
        pre-fetch tip the first time a branch goes pending (first writer
        wins) and drops it once the branch is resolved: gate and upstream
        agree again (equal or fast-forwardable), or the gate head is gone.
        """
        attic = _read_refs(self._gate_path, _ATTIC_PREFIX)
        for branch, gate_sha in heads.items():
            upstream_sha = new_snapshot.get(branch)
            resolved = upstream_sha is not None and (
                gate_sha == upstream_sha or self._is_ancestor(gate_sha, upstream_sha)
            )
            if resolved:
                if branch in attic:
                    _git(self._gate_path, "update-ref", "-d", f"{_ATTIC_PREFIX}{branch}")
            elif branch not in attic and (old_sha := old_snapshot.get(branch)) is not None:
                _git(self._gate_path, "update-ref", f"{_ATTIC_PREFIX}{branch}", old_sha)
        for branch in attic.keys() - heads.keys():
            _git(self._gate_path, "update-ref", "-d", f"{_ATTIC_PREFIX}{branch}")

    def pending_ops(self) -> list[PendingOp]:
        """Recompute the pending destructive ops without touching the network.

        Compares the current gate heads against the snapshot and attic —
        state the last sync left behind — so TUI badges and confirmation
        dialogs can refresh cheaply (no fetch, no SSH signer).  The one
        thing this cannot see is the one-shot ``unknown_provenance`` batch
        a migration sync reports.
        """
        if not self._gate_path.exists():
            return []
        snapshot = _read_refs(self._gate_path, _SNAPSHOT_PREFIX)
        attic = _read_refs(self._gate_path, _ATTIC_PREFIX)
        heads = _read_refs(self._gate_path, _HEADS_PREFIX)

        pending: list[PendingOp] = []
        for branch, gate_sha in sorted(heads.items()):
            if (upstream_sha := snapshot.get(branch)) is not None:
                if gate_sha != upstream_sha and not self._is_ancestor(gate_sha, upstream_sha):
                    pending.append(
                        self._pending(
                            branch,
                            "force_update",
                            "upstream_rewrite",
                            gate_sha=gate_sha,
                            upstream_sha=upstream_sha,
                            old_sha=attic.get(branch),
                        )
                    )
            elif (attic_sha := attic.get(branch)) is not None:
                pending.append(
                    self._pending(
                        branch,
                        "delete",
                        "upstream_delete",
                        gate_sha=gate_sha,
                        upstream_sha=None,
                        old_sha=attic_sha,
                    )
                )
        return pending

    def apply_pending_ops(self, ops: list[PendingOp]) -> ApplyPendingResult:
        """Apply operator-confirmed destructive ops, backing up every old tip.

        Each op is guarded by compare-and-swap on the ``gate_sha`` it was
        proposed against: a branch that moved since (an agent pushed) fails
        *that op only* and stays untouched — confirmations never apply to
        state the operator didn't see.  Unless backups are disabled, the
        old tip is first saved under ``refs/terok/backup/`` so even an
        approved mistake is one ``update-ref`` away from recovery.

        Finishes with the same HEAD-alignment and clone-cache refresh a
        sync performs — deleting the branch HEAD points at is precisely
        when healing matters.
        """
        applied: list[AppliedOp] = []
        backups: dict[str, str] = {}
        errors: list[str] = []

        for op in ops:
            backup_ref, error = self._apply_one_pending_op(op)
            if error is not None:
                errors.append(error)
                continue
            if backup_ref is not None:
                backups[op["branch"]] = backup_ref
            applied.append(
                {
                    "branch": op["branch"],
                    "kind": op["kind"],
                    "old_sha": op["gate_sha"],
                    "new_sha": op["upstream_sha"],
                }
            )

        if applied:
            env = self._ssh_env()
            if head_error := self._align_gate_head(env):
                errors.append(head_error)
            elif self._clone_cache_base and (cache_error := self._refresh_clone_cache()):
                errors.append(f"clone cache refresh failed: {cache_error}")

        return {"success": not errors, "applied": applied, "backups": backups, "errors": errors}

    def _apply_one_pending_op(self, op: PendingOp) -> tuple[str | None, str | None]:
        """Perform one confirmed op; return ``(backup_ref, error)``.

        Exactly one of the two is meaningful: an error means the branch was
        left untouched (malformed op, or the CAS guard found the branch
        moved since the proposal — in which case the just-written backup is
        removed again, since nothing was changed that could need it).
        """
        ref = f"{_HEADS_PREFIX}{op['branch']}"
        new_sha = op["upstream_sha"]
        if op["kind"] != "delete" and new_sha is None:
            return None, f"{op['branch']}: force_update op carries no upstream sha"
        # Backup before the change: between a delete and a later backup
        # there would be a window where the old tip is unreferenced.
        backup_ref = (
            self._write_backup(op["branch"], op["gate_sha"]) if self._backups_enabled else None
        )
        if op["kind"] == "delete":
            result = _git(self._gate_path, "update-ref", "-d", ref, op["gate_sha"])
        else:
            result = _git(self._gate_path, "update-ref", ref, cast("str", new_sha), op["gate_sha"])
        if result.returncode != 0:
            if backup_ref is not None:
                _git(self._gate_path, "update-ref", "-d", backup_ref)
            return None, f"{op['branch']}: branch moved since the op was proposed — not applied"
        # The op resolved the branch either way — the attic memory is moot.
        _git(self._gate_path, "update-ref", "-d", f"{_ATTIC_PREFIX}{op['branch']}")
        return backup_ref, None

    def _write_backup(self, branch: str, sha: str) -> str | None:
        """Save *sha* as a timestamped backup ref for *branch*; return its name.

        The ref name carries the wall-clock stamp retention runs on and the
        abbreviated tip for human eyes; the ref itself pins the full sha.
        Failure to back up is deliberately non-fatal — ``None`` tells the
        caller no backup exists, and the ``logAllRefUpdates=always`` reflog
        remains the last-resort trail.
        """
        stamp = datetime.now(UTC).strftime(_BACKUP_STAMP_FORMAT)
        ref = f"{_BACKUP_PREFIX}{branch}/{stamp}-{sha[:12]}"
        result = _git(self._gate_path, "update-ref", ref, sha)
        return ref if result.returncode == 0 else None

    def list_backups(self) -> list[BackupRef]:
        """Return all backup refs, newest first, parsed from their names."""
        entries: list[BackupRef] = []
        for path, sha in _read_refs(self._gate_path, _BACKUP_PREFIX).items():
            branch, _, leaf = path.rpartition("/")
            if not branch or not (m := _BACKUP_STAMP_RE.match(leaf)):
                continue
            saved_at = datetime.strptime(m.group(1), _BACKUP_STAMP_FORMAT).replace(tzinfo=UTC)
            entries.append(
                {
                    "ref": f"{_BACKUP_PREFIX}{path}",
                    "branch": branch,
                    "saved_at": saved_at.isoformat(),
                    "sha": sha,
                }
            )
        return sorted(entries, key=lambda e: e["saved_at"], reverse=True)

    def prune_backups(self, older_than_days: int | None = None) -> list[str]:
        """Delete backup refs older than the retention window; return them.

        Age is the ref-name timestamp — when the backup was *taken*, the
        only clock that matters for "have I had a chance to notice".  With
        retention ``0`` (or a nonexistent gate) nothing is ever expired.
        """
        days = self._backup_retention_days if older_than_days is None else older_than_days
        if days <= 0 or not self._gate_path.exists():
            return []
        cutoff = datetime.now(UTC) - timedelta(days=days)
        expired = [
            entry["ref"]
            for entry in self.list_backups()
            if datetime.fromisoformat(entry["saved_at"]) < cutoff
        ]
        for ref in expired:
            _git(self._gate_path, "update-ref", "-d", ref)
        return expired

    def _is_ancestor(self, ancestor: str, descendant: str) -> bool:
        """Return ``True`` iff *ancestor* is reachable from *descendant*."""
        result = _git(self._gate_path, "merge-base", "--is-ancestor", ancestor, descendant)
        return result.returncode == 0

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

    def _align_gate_head(self, env: dict) -> str | None:
        """Keep the gate's ``HEAD`` symref pointing at upstream's default branch.

        Sync moves branch refs but never the ``HEAD`` symref, so an
        upstream default-branch change leaves the gate advertising the old
        default — and once the old branch's (always-gated) deletion is
        approved, a dangling HEAD that breaks every fresh clone.  Aligning
        eagerly, not just when dangling, keeps fresh clones and the clone
        cache's ``remote set-head --auto`` on the branch upstream actually
        develops on; the old branch itself is untouched (its deletion
        remains a pending op like any other).

        Re-pointing HEAD is deliberately conservative: only at a branch
        that exists in the gate.  A detached or dangling HEAD is
        normalised the same way — a bare gate's HEAD must be a valid
        symref for ``git clone`` to work at all.

        Returns ``None`` when HEAD is healthy (or was aligned), or a
        failure description — a gate whose HEAD stays dangling breaks
        every fresh clone, so ``sync()`` reports it as a sync failure
        rather than pressing on quietly.  One exception: an empty gate
        with an empty (or unreachable) upstream keeps its unborn HEAD and
        is healthy — there is simply nothing to point at yet.
        """
        gate_dir = self._gate_path
        try:
            target = _git(gate_dir, "symbolic-ref", "--quiet", "HEAD", timeout=10).stdout.strip()
            healthy = bool(target) and (
                _git(gate_dir, "show-ref", "--verify", "--quiet", target, timeout=10).returncode
                == 0
            )

            upstream_head = _query_upstream_head_ref(str(gate_dir), env)
            if upstream_head is None:
                if healthy or not _read_refs(gate_dir, _HEADS_PREFIX):
                    return None  # aligned enough, or healthy-empty (unborn HEAD)
                return (
                    f"gate HEAD {target or '(unset)'!r} is dangling and upstream's "
                    "default branch could not be determined"
                )
            if upstream_head == target and healthy:
                return None
            # The target must exist in the gate before HEAD is re-pointed —
            # otherwise this would swap one dangling symref for another and
            # report success (possible when upstream's HEAD moved between
            # our fetch and the ls-remote, or the new default's creation is
            # itself still pending).
            exists = (
                _git(
                    gate_dir, "show-ref", "--verify", "--quiet", upstream_head, timeout=10
                ).returncode
                == 0
            )
            if not exists:
                if healthy:
                    return None  # keep the old, valid default until the new one lands
                if not _read_refs(gate_dir, _HEADS_PREFIX):
                    return None  # empty gate, empty upstream — unborn HEAD is correct
                return (
                    f"gate HEAD {target or '(unset)'!r} is dangling and upstream's "
                    f"default branch {upstream_head!r} is not present in the gate"
                )
            _git(gate_dir, "symbolic-ref", "HEAD", upstream_head, timeout=10, check=True)
            logger.info("Gate HEAD aligned: %s -> %s", target or "(unset)", upstream_head)
            return None
        except (subprocess.CalledProcessError, subprocess.TimeoutExpired, OSError) as exc:
            return f"gate HEAD alignment failed: {_describe_git_failure(exc)}"

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
            except (
                subprocess.CalledProcessError,
                subprocess.TimeoutExpired,
                OSError,
                RuntimeError,
            ) as exc:
                logger.warning(
                    "Clone cache update failed; rebuilding from scratch: %s",
                    _describe_git_failure(exc),
                )
        return self._rebuild_cache(cache_dir, gate_file_url)

    def _update_cache_in_place(self, cache_dir: Path, gate_file_url: str) -> None:
        """Fast-forward an existing cache working tree to the gate's default branch.

        Raises on any git failure, and ``RuntimeError`` if the cache ends
        up on a branch other than the gate's default — the caller answers
        every failure the same way (discard and rebuild), so no per-step
        handling here.
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
        # or the branch alignment below targets a stale default.
        subprocess.run(  # nosec B603 B607 — argv built from fixed verbs + repo-relative paths — binary PATH lookup is the cross-distro contract
            ["git", "-C", str(cache_dir), "remote", "set-head", "origin", "--auto"],
            check=True,
            capture_output=True,
            timeout=30,
        )
        branch = _resolve_origin_default_branch(cache_dir)
        # The cache is copied as-is into task workspaces, so the checked-out
        # branch *name* matters, not just the tree: after a default-branch
        # rename, hard-resetting the old branch would hand tasks the right
        # files under a branch that no longer exists upstream.  ``-B``
        # resets the branch to the remote tip and ``-f`` discards local
        # edits — together, the hard reset this refresh always performed.
        subprocess.run(  # nosec B603 B607 — argv built from fixed verbs + repo-relative paths — binary PATH lookup is the cross-distro contract
            ["git", "-C", str(cache_dir), "checkout", "-q", "-f", "-B", branch, f"origin/{branch}"],
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
        current = subprocess.run(  # nosec B603 B607 — argv built from fixed verbs + repo-relative paths — binary PATH lookup is the cross-distro contract
            ["git", "-C", str(cache_dir), "symbolic-ref", "--short", "HEAD"],
            check=True,
            capture_output=True,
            text=True,
            timeout=10,
        ).stdout.strip()
        if current != branch:
            raise RuntimeError(f"clone cache is on {current!r} after refresh, expected {branch!r}")

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


def _resolve_origin_default_branch(cache_dir: Path) -> str:
    """Return the branch name ``refs/remotes/origin/HEAD`` points at.

    Reads the local symref (no network) that ``remote set-head --auto``
    just resolved, e.g. ``refs/remotes/origin/main`` → ``main``.  The
    prefix strip keeps slashed branch names (``release/1.0``) intact.
    """
    head_ref = subprocess.run(  # nosec B603 B607 — argv built from fixed verbs + repo-relative paths — binary PATH lookup is the cross-distro contract
        ["git", "-C", str(cache_dir), "symbolic-ref", "refs/remotes/origin/HEAD"],
        check=True,
        capture_output=True,
        text=True,
        timeout=10,
    ).stdout.strip()
    return head_ref.removeprefix("refs/remotes/origin/")


def _read_refs(gate_dir: Path | str, prefix: str) -> dict[str, str]:
    """Return ``{name-under-prefix: sha}`` for every ref below *prefix*."""
    result = _git(gate_dir, "for-each-ref", "--format=%(objectname) %(refname)", prefix.rstrip("/"))
    refs: dict[str, str] = {}
    for line in result.stdout.splitlines():
        sha, _, refname = line.partition(" ")
        refs[refname.removeprefix(prefix)] = sha
    return refs


def _ensure_gate_config(gate_dir: Path | str) -> bool:
    """Normalise the gate's git config; return whether this was a migration.

    Idempotently establishes the three config facts the ref model rests
    on: the snapshot fetch refspec (replacing the mirror-clone
    ``+refs/*:refs/*`` that force-overwrote and pruned agent branches),
    the hidden ``refs/terok`` namespace (containers must not see or forge
    the snapshot/attic/backups), and always-on reflogs as the last-resort
    recovery trail.  ``True`` means the destructive mirror configuration
    was actually present — the caller reports that one-time migration and
    classifies the sync accordingly (no snapshot memory exists yet).
    """
    had_mirror = _git(gate_dir, "config", "--get", "remote.origin.mirror").stdout.strip() == "true"
    refspecs = _git(gate_dir, "config", "--get-all", "remote.origin.fetch").stdout.split()
    migrating = had_mirror or _FETCH_REFSPEC not in refspecs

    if migrating:
        _git(gate_dir, "config", "--unset-all", "remote.origin.mirror")
        _git(gate_dir, "config", "--replace-all", "remote.origin.fetch", _FETCH_REFSPEC)
    hidden = _git(gate_dir, "config", "--get-all", "transfer.hideRefs").stdout.split()
    if _TEROK_NAMESPACE not in hidden:
        _git(gate_dir, "config", "--add", "transfer.hideRefs", _TEROK_NAMESPACE)
    _git(gate_dir, "config", "core.logAllRefUpdates", "always")
    return migrating


def _normalise_fresh_gate(gate_dir: Path | str) -> None:
    """Turn a just-cloned mirror into a snapshot-model gate.

    ``git clone --mirror`` is kept for the initial copy (one efficient,
    atomic transfer), but its configuration and ref layout are destructive
    to keep: the config is rewritten, the snapshot namespace is seeded
    from the heads that just arrived (trivially identical to upstream —
    nothing local can exist yet), and foreign namespaces a mirror drags
    along (``refs/pull/*``, ``refs/notes/*``, …) are dropped while they
    are provably nothing but upstream residue.
    """
    _ensure_gate_config(gate_dir)
    commands = [
        f"create {_SNAPSHOT_PREFIX}{branch} {sha}"
        for branch, sha in _read_refs(gate_dir, _HEADS_PREFIX).items()
    ]
    keep = (_HEADS_PREFIX, "refs/tags/", _TEROK_NAMESPACE + "/")
    commands += [
        f"delete refs/{ref}"
        for ref in _read_refs(gate_dir, "refs/")
        if not f"refs/{ref}".startswith(keep)
    ]
    if commands:
        subprocess.run(  # nosec B603 B607 — argv built from fixed verbs + repo-relative paths — binary PATH lookup is the cross-distro contract
            ["git", "-C", str(gate_dir), "update-ref", "--stdin"],
            input="\n".join(commands) + "\n",
            capture_output=True,
            text=True,
            timeout=60,
            check=True,
        )


def _empty_sync_result(path: str, *, created: bool) -> GateSyncResult:
    """The all-quiet sync result a remoteless gate reports."""
    return {
        "path": path,
        "upstream_url": None,
        "created": created,
        "migrated": False,
        "success": True,
        "errors": [],
        "notes": [],
        "applied": [],
        "pending": [],
        "gate_only_branches": [],
        "cache_refreshed": False,
        "cache_error": None,
    }


_TAG_CLOBBER_MARKER = "would clobber existing tag"
_REJECTED_TAG_RE = re.compile(r"!\s+\[rejected\]\s+(\S+)\s+->\s+\S+\s+\(would clobber")


def _only_tag_clobbers(stderr: str) -> bool:
    """True when a failed fetch complained about nothing but moved tags.

    git refuses to move an existing tag without ``--force`` and exits
    non-zero — on *every* fetch, forever, until the tag is resolved by
    hand.  That refusal is correct (a gate tag is not sync's to rewrite)
    but must not read as a sync failure, or one moved upstream tag wedges
    the gate permanently.
    """
    if _TAG_CLOBBER_MARKER not in stderr:
        return False
    rejects = [line for line in stderr.splitlines() if "[rejected]" in line]
    return bool(rejects) and all(_TAG_CLOBBER_MARKER in line for line in rejects)


def _rejected_tags(stderr: str) -> set[str]:
    """Extract the tag names a fetch refused to clobber."""
    return set(_REJECTED_TAG_RE.findall(stderr))


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


def _count_commits_range(
    gate_dir: Path, from_ref: str, to_ref: str, env: dict | None
) -> int | None:
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
