<!--
SPDX-FileCopyrightText: 2026 Jiri Vyskocil
SPDX-License-Identifier: Apache-2.0
-->

# Manual verification: `VsockSSHTransport`

End-to-end exercise of [`VsockSSHTransport`][terok_sandbox.VsockSSHTransport]
against a real krun microVM.

## Prerequisites

- `podman` with `--runtime krun` support (`crun-krun` or `libkrun`).
- `socat` on PATH (used by the transport's `ProxyCommand`).
- `/dev/kvm` readable + writable by the invoking user.
- An L0G guest image built by terok-executor with a `KRUN_HOST_PUBKEY`
  matching the keypair used below — see
  `terok-executor/src/terok_executor/container/build.py:build_l0g_image`.

## Steps

1. Generate a host-side keypair:

   ```bash
   ssh-keygen -t ed25519 -f /tmp/host.key -N '' -C 'krun-host'
   ```

2. Build the L0G guest image with that pubkey baked in:

   ```bash
   python -c '
   from pathlib import Path
   from terok_executor.container.build import build_l0g_image
   pk = Path("/tmp/host.key.pub").read_text().strip()
   print(build_l0g_image("fedora:44", host_pubkey=pk))
   '
   ```

3. Run the guest, allocating a vsock CID > 2 (0–2 are spec-reserved)
   under the annotation key
   [`DEFAULT_CID_ANNOTATION`][terok_sandbox.runtime.krun_transport.DEFAULT_CID_ANNOTATION]:

   ```bash
   podman run -d --rm \
     --runtime krun \
     --annotation terok.krun.cid=42 \
     --name krun-smoke \
     terok-l0g:fedora-44
   ```

4. Exec a command via the transport:

   ```python
   from pathlib import Path
   from terok_sandbox import (
       KrunRuntime,
       PodmanRuntime,
       VsockSSHTransport,
       podman_annotation_resolver,
   )

   transport = VsockSSHTransport(
       identity_file=Path("/tmp/host.key"),
       endpoint_resolver=podman_annotation_resolver(),
   )
   rt = KrunRuntime(transport=transport, podman=PodmanRuntime())
   result = rt.exec(rt.container("krun-smoke"), ["uname", "-a"])
   print(result.exit_code, result.stdout, result.stderr)
   ```

   Expected: `exit_code == 0`, kernel info on `stdout`, empty `stderr`.

5. Cleanup:

   ```bash
   podman rm -f krun-smoke
   rm -f /tmp/host.key /tmp/host.key.pub
   ```

## Common failures

- `no terok.krun.cid annotation` raised by the resolver — the `podman
  run` call was missing `--annotation terok.krun.cid=<N>` (or used a
  different key than the resolver was configured with).
- `Permission denied (publickey)` — the `KRUN_HOST_PUBKEY` baked into
  the L0G image at build time doesn't match `/tmp/host.key`. Rebuild
  the image with the current pubkey.
- `socat: VSOCK_CONNECT: Connection refused` — the guest's vsock
  socket unit is not listening. The L0G image enables `ssh.socket`
  (deb) or `sshd.socket` (rpm) with a `ListenStream=vsock::22` drop-in
  at build time; verify by inspecting the image's
  `/etc/systemd/system/<unit>.d/vsock.conf`.
