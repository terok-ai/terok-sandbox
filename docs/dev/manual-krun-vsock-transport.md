<!--
SPDX-FileCopyrightText: 2026 Jiri Vyskocil
SPDX-License-Identifier: Apache-2.0
-->

# Manual verification: `VsockSSHTransport`

Podman-dependent tests don't run in CI per the project rules.  This
recipe lets a maintainer exercise [`VsockSSHTransport`][terok_sandbox.VsockSSHTransport]
end-to-end against a real krun microVM on a dev machine.

## Prerequisites

- `podman` 5.x with `--runtime krun` support (`crun-krun` or `libkrun` from
  your distro).
- `socat` on PATH.
- KVM access: `/dev/kvm` readable + writable by the invoking user (member
  of the `kvm` group, or rootful podman).
- An L0G guest image built by terok-executor with a `KRUN_HOST_PUBKEY`
  matching the keypair you'll use below — see
  `terok-executor/src/terok_executor/container/build.py:build_l0g_image`.

## Steps

1. **Generate a host-side keypair** (or export the `%host` keypair from
   the vault — see terok-side `krun` skill, not yet wired):

   ```bash
   ssh-keygen -t ed25519 -f /tmp/host.key -N '' -C 'krun-host'
   ```

2. **Build the L0G guest image** with the matching pubkey:

   ```bash
   python -c '
   from pathlib import Path
   from terok_executor.container.build import build_l0g_image
   pk = Path("/tmp/host.key.pub").read_text().strip()
   print(build_l0g_image("fedora:44", host_pubkey=pk))
   '
   ```

3. **Run the guest** under krun, allocating a free vsock CID
   (pick something > 2; CIDs 0/1/2 are reserved):

   ```bash
   podman run -d --rm \
     --runtime krun \
     --annotation terok.krun.cid=42 \
     --name krun-smoke \
     terok-l0g:fedora-44
   ```

4. **Exec a command via the transport** from a Python REPL:

   ```python
   from pathlib import Path
   from terok_sandbox import KrunRuntime, VsockSSHTransport, podman_annotation_resolver
   from terok_sandbox.runtime.podman import PodmanRuntime

   transport = VsockSSHTransport(
       identity_file=Path("/tmp/host.key"),
       endpoint_resolver=podman_annotation_resolver(),
   )
   rt = KrunRuntime(transport=transport, podman=PodmanRuntime())
   result = rt.exec(rt.container("krun-smoke"), ["uname", "-a"])
   print(result.exit_code, result.stdout, result.stderr)
   ```

5. **Verify no TCP listen on the host side**:

   ```bash
   ss -tlnp | grep -E ':22\b' || echo "no TCP sshd — good"
   ```

6. **Cleanup**:

   ```bash
   podman rm -f krun-smoke
   rm -f /tmp/host.key /tmp/host.key.pub
   ```

## What success looks like

- Step 4 prints `0`, kernel info on stdout, empty stderr.
- Step 5 confirms no host-side `:22` listen socket associated with the
  guest — sshd inside the guest is bound to AF_VSOCK only.
- A subsequent `exec` call returns the same way without re-prompting for
  the host key (the transport disables `StrictHostKeyChecking` because
  vsock is structurally private to this host).

## Common failures

- `no .* annotation` on exec — you forgot `--annotation terok.krun.cid=…`
  at `podman run` time, or you used a different key.
- `Permission denied (publickey)` — `KRUN_HOST_PUBKEY` baked at build
  time doesn't match `/tmp/host.key`.  Rebuild the L0G image with the
  current pubkey.
- `socat: VSOCK_CONNECT: Connection refused` — the guest's
  sshd-on-vsock socket unit didn't enable.  Inspect with
  `podman exec krun-smoke systemctl status sshd.socket` (the regular
  `podman exec` works for inspection because crun-side podman commands
  bypass the vsock issue; only the in-guest reach does not).
