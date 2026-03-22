# terok-sandbox

Hardened Podman container runner with gate server and shield integration.

## What it does

terok-sandbox provides a generic, security-hardened Podman container runtime
with integrated network firewalling (via terok-shield) and a gated git HTTP
server for controlled repository access.

### Key properties

- **Hardened runtime** — runs containers with a locked-down Podman configuration
- **Shield integration** — automatic egress firewalling via terok-shield
- **Gate server** — HTTP git server with per-task token authentication
- **SSH management** — per-container SSH key provisioning
- **Label-based grouping** — containers are identified by labels, not project config
- **XDG-compliant paths** — respects XDG base directory specification

## Installation

```bash
pip install git+https://github.com/terok-ai/terok-sandbox.git
```

## Quick start

```python
from terok_sandbox import __version__

print(f"terok-sandbox {__version__}")
```
