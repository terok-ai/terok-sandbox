# `prepare` / `run` / `cleanup` — sandbox a user-owned container

`terok-sandbox` provides three top-level CLI verbs for wiring a container
you launch yourself (via `podman run`, Podman Desktop, Compose, k8s, …)
into the sandbox's services — egress firewall (`shield`), credential
broker, git gate, and vault-served SSH agent.  Container lifecycle stays
with you; sandbox only manages the services and per-container ancillary
state (tokens, shield rules).

```
terok-sandbox prepare  <container>  [flags] [--json]
terok-sandbox run      <container>  [flags] -- <image> [cmd...]
terok-sandbox cleanup  <container>
```

| Verb      | What it does                                                  |
|-----------|---------------------------------------------------------------|
| `prepare` | Mints tokens, persists per-container state, **prints** podman flags |
| `run`     | Same composition + collision check + `os.execv` into `podman run` |
| `cleanup` | Revokes minted tokens, calls `shield.down`, removes state     |

## Flags (shared by `prepare` and `run`)

```
--shield   / --no-shield     egress firewall via shield's OCI hook   default on
--gate     / --no-gate       git gate (mints a gate token)           default on
--broker   / --no-broker     vault token broker bridging             default on
--scope SCOPE                credential scope; enables SSH wiring    no default
--profiles ...               pass-through override for shield        no default
--json                       (prepare only) emit JSON array          off
```

`--scope` is the toggle for the SSH agent: passing it enables the
vault SSH signer, omitting it disables it (the flag inherently needs a
value).  `--gate` and `--broker` also require `--scope` because they
issue per-scope tokens; when `--scope` is omitted they skip silently
with a note on stderr.

## Container-side contract

Sandbox produces a short stream of podman flags and environment
variables; the *container side* needs two things to consume them:

1. **`socat` installed in the image** — the broker, gate, and SSH-agent
   bridges all use socat to relay between in-container endpoints and the
   host-side vault/gate/signer.
2. **The bridge script sourced from your entrypoint** — `ensure-bridges.sh`
   reads the sandbox-injected env vars and launches the right bridges
   idempotently.

Two equally supported delivery paths for the bridge script:

### Build-time (`COPY` into image)

```dockerfile
FROM ubuntu:24.04
RUN apt-get update && apt-get install -y socat git openssh-client \
    && rm -rf /var/lib/apt/lists/*

# Drop the bridges next to your other in-container helpers.
COPY --from=ghcr.io/terok-ai/sandbox-bridges /usr/local/share/terok-sandbox/bridges/ \
     /usr/local/share/terok-sandbox/bridges/

# Source it from your entrypoint (or /etc/bash.bashrc for interactive shells).
RUN echo 'source /usr/local/share/terok-sandbox/bridges/ensure-bridges.sh' \
    >> /etc/bash.bashrc
```

(In practice, copy from your local `pip show -f terok-sandbox` location
or vendor the two scripts; an OCI image with just the bridges is one
sensible layering.)

### Runtime (bind-mounted by sandbox)

```dockerfile
FROM ubuntu:24.04
RUN apt-get update && apt-get install -y socat git openssh-client \
    && rm -rf /var/lib/apt/lists/*
```

Add the source line to your entrypoint, e.g.:

```sh
#!/usr/bin/env bash
source /usr/local/share/terok-sandbox/bridges/ensure-bridges.sh
exec "$@"
```

Sandbox bind-mounts the bridge resources read-only at
`/usr/local/share/terok-sandbox/bridges/` when any of `--gate` /
`--broker` / `--scope` is active.  Without `socat` in the image the
container is still sandboxed (shield/userns apply) but the bridges
can't connect — clear failure mode, not a security hole.

## Trust model

Every subsystem you opt into authenticates the container via a
**container-held secret validated host-side**:

| Flag      | Token form                                | Wire (inside the container)                              |
|-----------|-------------------------------------------|----------------------------------------------------------|
| `--broker`| phantom (`credentials.db`)                | HTTP `Authorization: Bearer …`                           |
| `--gate`  | gate token (`tokens.json`)                | HTTP basic-auth username                                 |
| `--scope` | phantom (provider=`ssh` row in same DB)   | `socat → ssh-agent-bridge.sh` → vault container-facing endpoint |

Subject for token rows is the **container name**.  Cleanup keys on
`(scope, container)`; the same name is used in shield's per-container
rule namespace.

> The per-scope host-local SSH agent socket
> (`<runtime_dir>/ssh-agent-local-<scope>.sock`) is the right path for
> *host* callers (e.g. terok's own `gate-sync`).  It is **not** what
> `--scope` mounts into the container — that would mix the host trust
> model into a container path.

## Recipes

### Full sandbox

```bash
terok-sandbox run mybox --scope myproj \
    -- my-image:latest bash -lc 'env | grep TEROK; ssh-add -l'
# … work in container …
terok-sandbox cleanup mybox
```

### Shield-only

```bash
terok-sandbox run mybox --no-broker --no-gate \
    -- ubuntu:24.04 bash
terok-sandbox cleanup mybox
```

### Emit flags only (Podman Desktop / Compose users)

```bash
$ terok-sandbox prepare mybox --scope myproj
--annotation=… --userns=keep-id:uid=1000,gid=1000 \
  -v /run/user/1000/terok/sandbox/vault.sock:/run/terok/vault.sock:z \
  -v /run/user/1000/terok/sandbox/ssh-agent.sock:/run/terok/ssh-agent.sock:z \
  -v /run/user/1000/terok/sandbox/gate-server.sock:/run/terok/gate-server.sock:z \
  -v <pkg>/resources/bridges:/usr/local/share/terok-sandbox/bridges:z,ro \
  -e TEROK_VAULT_LOOPBACK_PORT=9419 \
  -e TEROK_GATE_SOCKET=/run/terok/gate-server.sock \
  -e TEROK_GATE_TOKEN=terok-g-… \
  -e TEROK_SSH_SIGNER_TOKEN=terok-p-… \
  -e TEROK_SSH_SIGNER_SOCKET=/run/terok/ssh-agent.sock \
  --name mybox
```

Splice into your own `podman run`:

```bash
podman run -d --rm $(terok-sandbox prepare mybox --scope myproj) \
    my-image:latest sleep infinity
```

Or use `--json` for tooling that wants structured output.

## Collisions sandbox rejects

`run` exec's into podman after the `--` separator.  These flags and
volume targets belong to sandbox; using them after `--` raises an
explicit error:

- Flags: `--name`, `--network` (`--net`), `--hooks-dir`, `--annotation`,
  `--cap-add`, `--cap-drop`, `--userns`.
- Volume targets: the bridge mount path
  (`/usr/local/share/terok-sandbox/bridges/`) and the three socket paths
  under `/run/terok/`.

## Lifecycle split

| Concern                     | Owned by  |
|-----------------------------|-----------|
| `podman create` / `start` / `stop` / `rm` | **You**           |
| Shield rules (per-container) | sandbox (`prepare` / `cleanup`) |
| Minted tokens               | sandbox (`prepare` / `cleanup`) |
| Per-container state dir     | sandbox (`prepare` / `cleanup`) |
| Vault / gate / shield services themselves | sandbox (`setup` / `uninstall`) |

If your container crashes or you forget to call `cleanup`, the next
`cleanup <container>` is idempotent — running it on an unknown
container is a no-op.
