# terok-sandbox

Hardened Podman container runner with gate server and shield integration.

## Overview

terok-sandbox provides a generic, security-hardened Podman container runtime with:

- **Shield integration** — automatic egress firewalling via [terok-shield](https://github.com/terok-ai/terok-shield)
- **Gate server** — HTTP git server with per-task token authentication
- **SSH management** — per-container SSH key provisioning
- **Label-based grouping** — containers identified by labels, not project config

## Installation

```bash
pip install terok-sandbox
```

## License

Apache-2.0
