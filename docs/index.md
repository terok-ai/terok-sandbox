# terok-sandbox

The hardened-Podman runtime that powers terok.

terok-sandbox launches per-task containers with a credential vault,
a gated git server, and an installed egress firewall already in
place — so the calling tool can hand the container an agent and a
prompt, and the security boundary is set up before the agent ever
runs.

![terok ecosystem — terok-sandbox sits between the per-task launcher and the firewall it installs](architecture.svg)

## What it provides

- **Hardened container lifecycle** — rootless Podman containers
  launched through a single `Sandbox` facade.  No daemon, no setuid,
  no escalation surface from the host.
- **Credential vault** — long-lived secrets stay on the host,
  [SQLCipher-encrypted at rest](credentials-encryption.md).  The
  container receives short-lived phantom tokens that are exchanged
  for the real value at the moment of use, scoped per route, audited
  per request.
- **Per-task git gate** — a token-authenticated HTTP mirror of the
  upstream repository.  Tasks clone and push only through the gate;
  the operator chooses whether the gate forwards to upstream
  automatically or only on human review.
- **Shield install + drive** — a thin adapter that installs the
  terok-shield OCI hooks at setup time and drives the firewall at
  runtime (allow / deny / up / down).
- **Clearance composed in-supervisor** — every container gets a
  per-container supervisor that hosts the hub, verdict server, and
  desktop notifier in one short-lived process, spawned by an OCI
  hook and reaped by `podman wait`.  No lingering daemons between
  tasks; `pgrep terok` is empty when no containers are running.
- **Setup as one call** — `sandbox_setup()` brings the whole stack up
  idempotently; `sandbox_uninstall()` undoes it.

## Per-container supervisor

The supervisor model replaces the long-running `terok-vault`,
clearance-hub, verdict, and notifier daemons with a single
in-process composition per container.  Lifecycle, end-to-end:

1. `terok-sandbox prepare` writes a sidecar config to
   `$XDG_STATE_HOME/terok/sidecar/<container-name>.json` and emits
   the podman flags for the container (vault socket bind-mount,
   shield annotations, gate token, …).
2. The operator (or the calling orchestrator) runs `podman run`.
3. The OCI prestart hook installed by `terok-sandbox setup` reads
   the sidecar, then `Popen`s a stdlib-only wrapper that supervises
   `terok-sandbox supervisor <id>` — the actual long-running asyncio
   loop.  The wrapper restarts the supervisor up to five times on
   non-zero exit, with exponential backoff capped at 60 s.
4. The supervisor brings up a `VerdictServer`, `ClearanceHub`,
   `VaultProxy` (in `socket` or `tcp` mode per the sidecar), and a
   desktop notifier subscriber; awaits `podman wait <id>`; tears
   them down in reverse order on container exit.
5. The OCI poststop hook SIGTERMs the wrapper PID (recorded under
   `$XDG_STATE_HOME/terok/pids/supervisor-<id>.pid`) and unlinks the
   PID file.

Operators don't usually invoke `terok-sandbox supervisor` directly —
it's hidden from the main help and spawned by the hook chain — but
running it under a debugger after `terok-sandbox prepare` is a
supported way to step through a faulty composition without involving
podman.

## Where it sits in the stack

terok-sandbox is the boundary layer.  Above it, single-task callers
([terok-executor](https://github.com/terok-ai/terok-executor)) and
multi-task orchestrators
([terok](https://github.com/terok-ai/terok)) treat the sandbox as a
black-box "give me a hardened container."  Below it, it composes
[terok-shield](https://github.com/terok-ai/terok-shield) for egress
filtering and
[terok-clearance](https://github.com/terok-ai/terok-clearance) for
the operator-in-the-loop verdict path.

The split exists so that callers do not need to understand
nftables, OCI hook wiring, vault sockets, or systemd unit lifecycles
to get a safe container.

## Installation

```bash
pip install terok-sandbox
```

For most users this dependency is pulled in transitively by
`terok-executor` or `terok`.  Install it directly only when building
a custom orchestrator on top of the sandbox API.

## Quick start

```python
from pathlib import Path

from terok_sandbox import RunSpec, Sandbox, SandboxConfig

sandbox = Sandbox(SandboxConfig())
sandbox.run(
    RunSpec(
        container_name="task-001",
        image="terok-l1-cli:ubuntu-24.04",
        env={},
        volumes=(),
        command=(),
        task_dir=Path("/var/lib/myapp/task-001"),
    )
)
```

See the developer guide for the full lifecycle and integration
patterns.
