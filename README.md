# terok-sandbox

[![License: Apache-2.0](https://img.shields.io/badge/License-Apache_2.0-blue.svg)](https://opensource.org/licenses/Apache-2.0)
[![REUSE status](https://api.reuse.software/badge/github.com/terok-ai/terok-sandbox)](https://api.reuse.software/info/github.com/terok-ai/terok-sandbox)

The hardened-Podman runtime that powers terok.

terok-sandbox launches per-task containers with a credential vault,
a gated git server, and an installed egress firewall already in
place — so the calling tool can hand the container an agent and a
prompt, and the security boundary is set up before the agent ever
runs.

<p align="center">
  <img src="docs/architecture.svg" alt="terok ecosystem — terok-sandbox sits between the per-task launcher and the firewall it installs">
</p>

## What it provides

- **Hardened container lifecycle** — rootless Podman containers
  launched through a single `Sandbox` facade.  No daemon, no setuid,
  no escalation surface from the host.
- **Credential vault** — long-lived secrets stay on the host.  The
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
- **Clearance install** — wires the desktop notifier daemon
  (terok-clearance) onto every blocked outbound connection, so the
  operator can authorise destinations live without restarting the
  container.
- **Setup as one call** — `sandbox_setup()` brings the whole stack up
  idempotently (services, hooks, systemd units, port reservations);
  `sandbox_uninstall()` undoes it.

## Where it sits in the stack

terok-sandbox is the boundary layer.  Above it, single-task callers
([terok-executor](https://github.com/terok-ai/terok-executor)) and
multi-task orchestrators
([terok](https://github.com/terok-ai/terok)) treat the sandbox as a
black-box "give me a hardened container."  Below it, it composes
[terok-shield](https://github.com/terok-ai/terok-shield) for egress
filtering and [terok-clearance](https://github.com/terok-ai/terok-clearance)
for the operator-in-the-loop verdict path.

The split exists so that callers do not need to understand
nftables, OCI hook wiring, vault sockets, or systemd unit lifecycles
to get a safe container.

## Public API

```python
from terok_sandbox import (
    # Lifecycle
    Sandbox, SandboxConfig, RunSpec, VolumeSpec, Sharing,
    # Runtime backends
    PodmanRuntime, NullRuntime, ContainerRuntime,
    # Vault
    VaultManager, CredentialDB, SSHManager,
    start_vault, stop_vault, ensure_vault_reachable,
    # Gate
    GateServerManager, TokenStore, GitGate,
    start_daemon, stop_daemon, create_token,
    # Shield adapter
    ShieldState, make_shield,
    # Setup / teardown
    sandbox_setup, sandbox_uninstall, needs_setup,
)
```

The full export list lives in
[`src/terok_sandbox/__init__.py`](src/terok_sandbox/__init__.py).

## CLI

| Command | Purpose |
|---------|---------|
| `terok-sandbox setup` | Install hooks, vault, gate, notifier; idempotent |
| `terok-sandbox uninstall` | Reverse of setup |
| `terok-sandbox doctor` | Run health checks against installed services |
| `terok-sandbox vault …` | Vault management subcommands |
| `terok-sandbox gate …` | Gate management subcommands |
| `terok-sandbox shield …` | Shield install / status / direct control |
| `terok-sandbox ssh …` | Per-container SSH key provisioning |
| `terok-gate` | Long-running gate daemon (systemd unit entry point) |
| `terok-vault` | Long-running vault token broker (systemd unit entry point) |

## Requirements

- Linux with **Podman** (rootless, ≥ 5.6 recommended)
- **systemd** user session (for gate / vault / clearance services)
- **nftables** (`nft` binary) — provided by terok-shield's runtime
- **D-Bus** session bus — for the clearance notifier path; the system
  degrades gracefully when D-Bus is absent
- Python 3.12+

## Installation

```bash
pip install terok-sandbox
```

For most users this dependency is pulled in transitively by
`terok-executor` or `terok`.  Install it directly only when building
a custom orchestrator on top of the sandbox API.

## License

Apache-2.0 — see [LICENSES/Apache-2.0.txt](LICENSES/Apache-2.0.txt).
