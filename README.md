<p align="center">
  <picture>
    <source media="(prefers-color-scheme: dark)" srcset="https://terok-ai.github.io/terok/terok-logo-w.svg">
    <img src="https://terok-ai.github.io/terok/terok-logo-b.svg" alt="terok-sandbox" width="120">
  </picture>
</p>

# terok-sandbox

[![PyPI](https://img.shields.io/pypi/v/terok-sandbox)](https://pypi.org/project/terok-sandbox/)
[![License: Apache-2.0](https://img.shields.io/badge/License-Apache_2.0-blue.svg)](https://opensource.org/licenses/Apache-2.0)
[![REUSE status](https://api.reuse.software/badge/github.com/terok-ai/terok-sandbox)](https://api.reuse.software/info/github.com/terok-ai/terok-sandbox)
[![Quality Gate Status](https://sonarcloud.io/api/project_badges/measure?project=terok-ai_terok-sandbox&metric=alert_status)](https://sonarcloud.io/summary/new_code?id=terok-ai_terok-sandbox)

The hardened-Podman runtime — terok-sandbox launches per-task containers with a credential vault,
a gated git server, and an egress firewall.

<p align="center">
  <img src="https://terok-ai.github.io/terok/img/architecture.svg" alt="terok ecosystem — terok-sandbox sits between the per-task launcher and the firewall it installs">
</p>

## What it provides

- **Hardened container lifecycle** — rootless Podman containers.
- **Credential vault** — long-lived secrets stay in an encrypted database on the host.
  The container receives short-lived phantom tokens and never sees the real credentials.
- **Per-task git gate** — a token-authenticated HTTP mirror of an arbitrary
  upstream *git* repository. Tasks clone and push through the gate, and
  the gate forwards to upstream automatically (online mode) or after
  operator review (gatekeeping mode).
- **Shield firewall** — installs the [terok-shield](https://github.com/terok-ai/terok-shield) OCI hooks at setup time and drives the firewall at runtime.
- **Clearance in-supervisor** — each container's supervisor hosts the
  [terok-clearance](https://github.com/terok-ai/terok-clearance) hub, verdict server, and desktop notifier, so the operator can authorise blocked outbound connections live.
- **Setup as one call** — idempotent `terok-sandbox setup` installs the shield +
  supervisor OCI hooks and provisions the encrypted credentials DB;
  `terok-sandbox uninstall` reverses it.

## Where it sits in the stack

terok-sandbox is the boundary layer.  Above it, single-task callers
([terok-executor](https://github.com/terok-ai/terok-executor)) and
multi-task orchestrators
([terok](https://github.com/terok-ai/terok)) treat the sandbox as a
black-box "give me a hardened container."  Below it, it composes
[terok-shield](https://github.com/terok-ai/terok-shield) for egress
filtering and [terok-clearance](https://github.com/terok-ai/terok-clearance)
for the operator-in-the-loop verdict path.

## Public API

```python
from terok_sandbox import (
    # Lifecycle
    Sandbox, SandboxConfig, RunSpec, VolumeSpec, Sharing,
    # Runtime backends
    PodmanRuntime, KrunRuntime, NullRuntime, ContainerRuntime,
    # Vault + credentials
    CredentialDB, SSHManager, NoPassphraseError, WrongPassphraseError,
    # Gate
    GateServer, GitGate, mint_gate_token,
    # Shield adapter
    ShieldManager, ShieldHooks, check_environment,
    # Per-container wiring / setup state
    write_sidecar, remove_container_state, sandbox_uninstall, needs_setup,
)
```

The full export list lives in
[`src/terok_sandbox/__init__.py`](src/terok_sandbox/__init__.py).

## CLI

| Command | Purpose |
|---------|---------|
| `terok-sandbox setup` | Install shield + supervisor OCI hooks, provision the credentials DB; idempotent |
| `terok-sandbox uninstall` | Reverse of setup |
| `terok-sandbox prepare` / `run` / `cleanup` | Wire a user-owned container into the sandbox services |
| `terok-sandbox doctor` | Run host-side sandbox health checks |
| `terok-sandbox vault …` | Vault status / unlock / lock / passphrase-tier management |
| `terok-sandbox gate …` | Git gate inspection (`gate path <project>`) |
| `terok-sandbox shield …` | Shield hooks install / status / direct control |
| `terok-sandbox ssh …` | Per-scope SSH key management in the credentials DB |
| `terok-sandbox credentials encrypt-db` | Encrypt (migrate) a plaintext credentials DB |

## Requirements

- Linux with **Podman** (rootless, ≥ 5.6 recommended)
- **systemd ≥ 257** — optional; backs the `systemd-creds` vault passphrase tier (gate / vault / clearance run inside the per-container supervisor, no systemd units)
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
